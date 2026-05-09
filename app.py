import os
import time
import sqlite3
import threading
from datetime import datetime
from flask import Flask, render_template, request, redirect, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

app = Flask(__name__)
app.secret_key = "behavioralfingerprint"

# Initialize Flask-Limiter for automated rate-limiting
# limiter = Limiter(
#     get_remote_address,
#     app=app,
#     default_limits=["200 per day", "50 per hour"],
#     storage_uri="memory://"
# )

# In-memory tracking for feature extraction
session_data = {}
last_request_time = {}

def get_session_data(ip):
    if ip not in session_data:
        session_data[ip] = {
            'total_requests': 0,
            'login_attempts': 0,
            'failed_logins': 0,
            'request_timestamps': []
        }
    return session_data[ip]

# Ensure directories
os.makedirs('database', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

MODEL_PATH = 'models/isolation_forest.pkl'

# ---------------- DATABASE SETUP ---------------- #
def init_db():
    conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')

    # Drop existing to easily upgrade schema for ML
    cursor.execute('DROP TABLE IF EXISTS fingerprints')
    
    cursor.execute('''
        CREATE TABLE fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            fingerprint_hash TEXT,
            request_frequency REAL,
            retry_interval REAL,
            failure_rate REAL,
            behavior TEXT,
            timestamp TEXT,
            risk_score INTEGER,
            is_anomaly INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blocked_ips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT UNIQUE
        )
    ''')

    # Seed baseline "normal" data for the ML model so it can train instantly
    cursor.execute("SELECT COUNT(*) FROM fingerprints")
    if cursor.fetchone()[0] == 0:
        for _ in range(20):
            cursor.execute('''
                INSERT INTO fingerprints (
                    ip_address, fingerprint_hash, request_frequency, retry_interval, failure_rate, behavior, timestamp, risk_score, is_anomaly
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                "127.0.0.1", "baseline_normal", 
                float(np.random.uniform(0.5, 10.0)), # Normal freq: 0.5 to 10 req/min
                float(np.random.uniform(5.0, 30.0)), # Normal interval: 5 to 30s
                float(np.random.uniform(0.0, 0.5)), # Normal fail rate: 0 to 0.5
                "normal_baseline", str(datetime.now()), 0, 0
            ))

    conn.commit()
    conn.close()

def log_activity(message):
    with open("logs/activity.log", "a", encoding="utf-8") as file:
        file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# ---------------- MACHINE LEARNING ---------------- #
def extract_features(ip):
    data = get_session_data(ip)
    timestamps = data['request_timestamps']
    
    # 1. Request Frequency (requests per minute based on window)
    if len(timestamps) > 1:
        time_diff = timestamps[-1] - timestamps[0]
        # if time_diff is very small, cap it to prevent division by zero
        time_diff = max(time_diff, 1.0)
        freq = (len(timestamps) / time_diff) * 60.0
    else:
        freq = 1.0

    # 2. Avg Retry Interval
    if len(timestamps) > 1:
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        retry_interval = sum(intervals) / len(intervals)
    else:
        retry_interval = 10.0 # Default safe interval
        
    # 3. Failure Rate
    if data['login_attempts'] > 0:
        failure_rate = data['failed_logins'] / data['login_attempts']
    else:
        failure_rate = 0.0
        
    return freq, retry_interval, failure_rate

def train_ml_model():
    """ Periodically train the Isolation Forest on the fingerprints table """
    conn = sqlite3.connect('database/users.db')
    df = pd.read_sql_query("SELECT request_frequency, retry_interval, failure_rate FROM fingerprints", conn)
    conn.close()
    
    if len(df) > 10: # Need some baseline data to train
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(df.values)
        joblib.dump(model, MODEL_PATH)
        print(f"[{datetime.now()}] ML Model trained with {len(df)} samples.")
        
def predict_anomaly(freq, retry_interval, failure_rate):
    """ Returns True if anomaly detected, False otherwise """
    if not os.path.exists(MODEL_PATH):
        return False # No model trained yet
        
    try:
        model = joblib.load(MODEL_PATH)
        features = np.array([[freq, retry_interval, failure_rate]])
        prediction = model.predict(features)
        return prediction[0] == -1
    except Exception as e:
        print("ML Prediction Error:", e)
        return False

# Background thread to train model every 60 seconds
def ml_scheduler():
    while True:
        try:
            train_ml_model()
        except Exception as e:
            print("ML Scheduler Error:", e)
        time.sleep(60)

# ---------------- MIDDLEWARE ---------------- #
@app.before_request
def track_behavior():
    if request.path.startswith('/static') or request.path == '/reset':
        return
        
    ip = request.remote_addr
    
    # Check if blocked
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM blocked_ips WHERE ip_address=?", (ip,))
    blocked = cursor.fetchone()
    conn.close()
    if blocked:
        return "Your IP has been permanently blocked by the IDS.", 403

    # Update session data
    data = get_session_data(ip)
    data['total_requests'] += 1
    data['request_timestamps'].append(time.time())
    
    # Keep only last 50 timestamps to avoid memory leak
    if len(data['request_timestamps']) > 50:
        data['request_timestamps'].pop(0)

# ---------------- ROUTES ---------------- #
@app.route('/')
def home():
    session.pop('user', None) # Force clear session so you can always see the login page
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
        
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users(username, password) VALUES(?, ?)', (username, password))
        conn.commit()
        log_activity(f"NEW USER REGISTERED - {username}")
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

    return redirect('/')

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'GET':
        return render_template("login.html")


    username = request.form['username']
    password = request.form['password']
    ip_address = request.remote_addr


    # session tracking
    data = get_session_data(ip_address)

    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()


    # ===============================
    # CHECK BLOCKED IP
    # ===============================

    cursor.execute(
        "SELECT * FROM blocked_ips WHERE ip_address=?",
        (ip_address,)
    )

    blocked = cursor.fetchone()

    if blocked:
        conn.close()
        return "Suspicious Behavior Detected - Access Blocked"


    # ===============================
    # CHECK USER
    # ===============================

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()


    # ===============================
    # FAILED LOGIN
    # ===============================

    if not user:

        data['failed_logins'] += 1

        freq, retry_interval, failure_rate = extract_features(
            ip_address
        )

        is_anomaly = predict_anomaly(
            freq,
            retry_interval,
            failure_rate
        )


        behavior_style = (
            f"{int(freq/10)}_"
            f"{int(retry_interval)}_"
            f"{round(failure_rate,2)}"
        )


        risk_score = data['failed_logins'] * 20


        # Override ML for a single or slow failure to just be "Failed Login"
        if is_anomaly and data['failed_logins'] > 1 and freq > 10.0:
            risk_score += 50
            behavior_msg = "ML Anomaly Detected"
            log_activity(f"🚨 ML ANOMALY DETECTED - IP: {ip_address}")
        else:
            is_anomaly = False # Force false for normal failed logins
            behavior_msg = "Failed Login"
            log_activity(f"FAILED LOGIN - IP: {ip_address}")

        # store fingerprint
        cursor.execute(

            '''
            INSERT INTO fingerprints(

                ip_address,
                fingerprint_hash,
                request_frequency,
                retry_interval,
                failure_rate,
                behavior,
                risk_score

            )

            VALUES(?,?,?,?,?,?,?)
            ''',

            (

                ip_address,
                behavior_style,
                freq,
                retry_interval,
                failure_rate,
                behavior_msg,
                risk_score

            )
        )

        conn.commit()


        # block after 5 attempts
        if data['failed_logins'] >= 5:

            cursor.execute(

                '''
                INSERT OR IGNORE
                INTO blocked_ips(ip_address)
                VALUES(?)
                ''',

                (ip_address,)
            )

            conn.commit()
            conn.close()

            return "Suspicious Behavior Detected - Access Blocked"


        conn.close()

        return "Invalid username or password"


    # ===============================
    # SUCCESS LOGIN
    # ===============================

    data['failed_logins'] = 0

    session['user'] = username

    conn.close()

    return redirect('/dashboard')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        log_activity(f"UNAUTHORIZED ACCESS ATTEMPT - IP: {request.remote_addr}")
        return "Unauthorized Access Detected", 401

    user_ip = request.remote_addr
    current_time = time.time()
    data = get_session_data(user_ip)
    
    # 3. Automated Retry Detection (Dashboard specific)
    if user_ip in last_request_time:
        interval = current_time - last_request_time[user_ip]
        if interval < 1:
            log_activity(f"AUTOMATED RETRY DETECTED - IP: {user_ip}")
            return "Automated Retry Detected - Please wait before refreshing again."
    last_request_time[user_ip] = current_time

    # 4. Bot Activity Detection
    if data['total_requests'] > 30:
        log_activity(f"BOT ACTIVITY DETECTED - IP: {user_ip}")
        return "Bot Activity Detected - Maximum requests exceeded."

    conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM fingerprints")
    total_attacks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM fingerprints WHERE is_anomaly=1 OR risk_score >= 80")
    high_risk = cursor.fetchone()[0]

    cursor.execute("SELECT ip_address, fingerprint_hash, request_frequency, failure_rate, risk_score, behavior FROM fingerprints ORDER BY id DESC LIMIT 10")
    recent_fingerprints = cursor.fetchall()

    conn.close()

    ml_status = "Active" if os.path.exists(MODEL_PATH) else "Training Base Data..."

    try:
        with open("logs/activity.log", "r", encoding="utf-8") as file:
            logs = file.readlines()
            logs = logs[-15:] 
            logs.reverse() 
    except FileNotFoundError:
        logs = []

    return render_template(
        'dashboard.html', 
        username=session['user'],
        total_users=total_users,
        total_attacks=total_attacks,
        high_risk=high_risk,
        logs=logs,
        ml_status=ml_status,
        recent_fingerprints=recent_fingerprints
    )



@app.route('/reset')
def reset_testing():
    """Testing utility: instantly unblocks the user's IP and resets their request counters."""
    ip = request.remote_addr
    
    # Clear in-memory bot counters
    if ip in session_data:
        session_data[ip] = {
            'total_requests': 0,
            'login_attempts': 0,
            'failed_logins': 0,
            'request_timestamps': []
        }
    if ip in last_request_time:
        del last_request_time[ip]
        
    # Unblock IP from SQLite database
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blocked_ips WHERE ip_address=?", (ip,))
    conn.commit()
    conn.close()
    
    return redirect('/')

if __name__ == "__main__":
    init_db()
    
    # Start ML training thread after DB init
    threading.Thread(target=ml_scheduler, daemon=True).start()
    
    app.run(host='0.0.0.0', debug=True, port=5000, use_reloader=False) # use_reloader=False prevents double training thread