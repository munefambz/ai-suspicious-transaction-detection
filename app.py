import os
import json
import csv
import urllib.request
import datetime
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, has_request_context
from flask_bcrypt import Bcrypt
from config import Config
from models import db, User, Transaction, Alert, AuditLog, SanctionedIndividual
import anthropic
import ai_assistant
import explain

app = Flask(__name__)
app.config.from_object(Config)
bcrypt = Bcrypt(app)

# Initialize database
db.init_app(app)

# Loaded models dictionary to cache them
_loaded_models = {}

def get_model(model_name):
    """Dynamically loads and caches the selected machine learning pipeline."""
    if model_name not in _loaded_models:
        model_path = app.config['MODELS'].get(model_name)
        if model_path and os.path.exists(model_path):
            try:
                _loaded_models[model_name] = joblib.load(model_path)
            except Exception as e:
                app.logger.error(f"Error loading model {model_name}: {e}")
                return None
        else:
            app.logger.error(f"Model file not found for {model_name} at {model_path}")
            return None
    return _loaded_models.get(model_name)

# Active model setting (can be changed dynamically)
ACTIVE_MODEL = Config.DEFAULT_MODEL
ai_assistant.active_model_getter = lambda: ACTIVE_MODEL
ai_assistant.model_getter = lambda: get_model(ACTIVE_MODEL)

# Helper function to check if user is logged in
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function to create audit logs
def log_audit_event(username, event):
    try:
        ip = request.remote_addr if has_request_context() else '127.0.0.1'
        log = AuditLog(
            username=username,
            event=event,
            ip_address=ip
        )
        db.session.add(log)
        db.session.commit()
        app.logger.info(f"[AUDIT LOG] User: {username} | Event: {event} | IP: {ip}")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to log audit event: {e}")

@app.context_processor
def inject_global_settings():
    return {
        'active_model': ACTIVE_MODEL,
        'all_models': list(app.config['MODELS'].keys()),
        'current_user': session.get('username'),
        'current_role': session.get('role')
    }

# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            if user.status != 'Active':
                flash('Your account has been suspended. Contact an administrator.', 'danger')
                return redirect(url_for('login'))
                
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            
            log_audit_event(user.username, "Successful login")
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            # Log failed login if username exists
            log_audit_event(username or "unknown", f"Failed login attempt")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    username = session.get('username')
    if username:
        log_audit_event(username, "Logged out")
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    # Gather statistics
    total_tx = Transaction.query.count()
    unassigned_alerts = Alert.query.filter_by(status='Pending').count()
    total_alerts = Alert.query.count()
    
    # Calculate unusual transaction counts
    # A transaction is unusual if it has a high or medium risk level
    unusual_tx = Transaction.query.filter(Transaction.risk_level != 'Low').count()
    
    # Verification chart counts
    legitimate_count = Transaction.query.filter_by(status='Legitimate').count()
    fraudulent_count = Transaction.query.filter_by(status='Suspicious').count()
    under_inv_count = Transaction.query.filter_by(status='Under Investigation').count()
    
    # Alert statistics
    resolved_alerts = Alert.query.filter(Alert.status == 'Closed').count()
    pending_alerts = Alert.query.filter(Alert.status == 'Pending').count()
    review_alerts = Alert.query.filter(Alert.status == 'Under Review').count()
    
    # Recent activity for charts (grouped by risk levels for recent 15 transactions)
    recent_transactions = Transaction.query.order_by(Transaction.timestamp.desc()).limit(15).all()
    # Format for JSON passing
    chart_labels = [tx.timestamp.strftime('%H:%M') for tx in reversed(recent_transactions)]
    chart_amounts = [tx.amount for tx in reversed(recent_transactions)]
    chart_risks = [tx.risk_score for tx in reversed(recent_transactions)]
    
    # Recent alerts table
    recent_alerts = Alert.query.order_by(Alert.created_at.desc()).limit(5).all()
    
    # Ongoing investigations
    ongoing_investigations = Alert.query.filter(Alert.status == 'Under Review').order_by(Alert.created_at.desc()).limit(5).all()
    
    return render_template(
        'dashboard.html',
        total_transactions=total_tx,
        unassigned_alerts=unassigned_alerts,
        unusual_transactions=unusual_tx,
        legitimate_count=legitimate_count,
        fraudulent_count=fraudulent_count,
        under_inv_count=under_inv_count,
        resolved_alerts=resolved_alerts,
        pending_alerts=pending_alerts,
        review_alerts=review_alerts,
        recent_alerts=recent_alerts,
        ongoing_investigations=ongoing_investigations,
        chart_labels=chart_labels,
        chart_amounts=chart_amounts,
        chart_risks=chart_risks
    )

@app.route('/transactions', methods=['GET', 'POST'])
@login_required
def transactions():
    # If POST, a manual transaction is being ingested
    if request.method == 'POST':
        sender = request.form.get('sender_wallet')
        sender_name = request.form.get('sender_name', '')
        recipient = request.form.get('recipient_wallet')
        recipient_name = request.form.get('recipient_name', '')
        amount = float(request.form.get('amount'))
        currency = request.form.get('currency')
        payment_type = request.form.get('payment_type')
        is_unfamiliar = request.form.get('is_unfamiliar') == '1'
        is_new_device = request.form.get('is_new_device') == '1'
        account_age = int(request.form.get('account_age_days', 365))
        
        # Evaluate transaction using helper
        tx = process_aml_evaluation(
            sender=sender,
            recipient=recipient,
            amount=amount,
            currency=currency,
            payment_type=payment_type,
            is_unfamiliar=is_unfamiliar,
            is_new_device=is_new_device,
            account_age_days=account_age,
            sender_name=sender_name,
            recipient_name=recipient_name
        )
        
        if tx:
            log_audit_event(session.get('username', 'SYSTEM'), f"Manually ingested transaction {tx.transaction_id} (Amount: {tx.amount} {tx.currency})")
            flash(f"Transaction {tx.transaction_id} ingested. Risk: {tx.risk_level} ({tx.risk_score*100:.1f}%)", "success" if tx.risk_level == 'Low' else "warning")
        else:
            flash("Error processing transaction. Check logs.", "danger")
            
        return redirect(url_for('transactions'))
        
    # GET: render and filter transactions
    query = Transaction.query
    
    # Extract filter parameters
    currency_filter = request.args.get('currency', '')
    risk_filter = request.args.get('risk_level', '')
    type_filter = request.args.get('payment_type', '')
    
    if currency_filter:
        query = query.filter_by(currency=currency_filter)
    if risk_filter:
        query = query.filter_by(risk_level=risk_filter)
    if type_filter:
        query = query.filter_by(payment_type=type_filter)
        
    # Pagination
    page = request.args.get('page', 1, type=int)
    tx_list = query.order_by(Transaction.timestamp.desc()).paginate(page=page, per_page=15)
    
    return render_template(
        'transactions.html',
        transactions=tx_list,
        selected_currency=currency_filter,
        selected_risk=risk_filter,
        selected_type=type_filter
    )

@app.route('/alerts')
@login_required
def alerts():
    status_filter = request.args.get('status', 'Pending')
    alerts_list = Alert.query.filter_by(status=status_filter).order_by(Alert.created_at.desc()).all()
    
    selected_id = request.args.get('id', type=int)
    selected_alert = None
    if selected_id:
        selected_alert = Alert.query.get(selected_id)
    elif alerts_list:
        selected_alert = alerts_list[0]
        
    explanation = None
    if selected_alert and 'SANCTIONED_INDIVIDUAL' not in (selected_alert.reason_codes or ''):
        model = get_model(ACTIVE_MODEL)
        if model:
            explanation = explain.explain(model, explain.transaction_features(selected_alert.transaction))

    return render_template(
        'alerts.html',
        alerts=alerts_list,
        active_status=status_filter,
        selected_alert=selected_alert,
        explanation=explanation
    )

@app.route('/alerts/<int:alert_id>', methods=['POST'])
@login_required
def update_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    action = request.form.get('action')
    notes = request.form.get('notes', '')
    
    if action == 'investigate':
        alert.status = 'Under Review'
        alert.assigned_to = session['username']
        alert.transaction.status = 'Under Investigation'
        log_audit_event(session['username'], f"Assigned Alert {alert.id} ({alert.transaction_id}) to self")
        flash(f"Alert {alert.id} is now under review.", "info")
        
    elif action == 'close_legitimate':
        alert.status = 'Closed'
        alert.resolution_notes = notes
        alert.transaction.status = 'Legitimate'
        log_audit_event(session['username'], f"Closed Alert {alert.id} ({alert.transaction_id}) as Legitimate")
        flash(f"Alert {alert.id} closed as Legitimate.", "success")
        
    elif action == 'close_fraudulent':
        alert.status = 'Closed'
        alert.resolution_notes = notes
        alert.transaction.status = 'Suspicious'
        log_audit_event(session['username'], f"Closed Alert {alert.id} ({alert.transaction_id}) as Confirmed Fraud")
        flash(f"Alert {alert.id} closed as Confirmed Fraud.", "danger")
        
    db.session.commit()
    return redirect(url_for('alerts', status=alert.status))

@app.route('/audit-logs')
@login_required
def audit_logs():
    page = request.args.get('page', 1, type=int)
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).paginate(page=page, per_page=20)
    return render_template('audit_logs.html', logs=logs)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    global ACTIVE_MODEL
    
    # Load metrics for performance displays
    metrics_path = os.path.join(app.config['MODEL_DIR'], 'metrics.json')
    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            
    if request.method == 'POST':
        # Change model
        new_model = request.form.get('active_model')
        if new_model in app.config['MODELS']:
            ACTIVE_MODEL = new_model
            log_audit_event(session['username'], f"Changed active classifier to: {new_model}")
            flash(f"Active model successfully updated to {new_model}.", "success")
            
        # Change thresholds
        try:
            app.config['MEDIUM_RISK_THRESHOLD'] = float(request.form.get('medium_risk_threshold', 0.3))
            app.config['HIGH_RISK_THRESHOLD'] = float(request.form.get('high_risk_threshold', 0.7))
            log_audit_event(session['username'], f"Updated thresholds: Medium={app.config['MEDIUM_RISK_THRESHOLD']}, High={app.config['HIGH_RISK_THRESHOLD']}")
            flash("Thresholds updated successfully.", "success")
        except ValueError:
            flash("Invalid threshold values.", "danger")
            
        return redirect(url_for('settings'))
        
    return render_template('settings.html', metrics=metrics_data)

@app.route('/sanctions')
@login_required
def sanctions():
    query_str = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)
    
    if query_str:
        individuals = SanctionedIndividual.query.filter(
            SanctionedIndividual.name.ilike(f"%{query_str}%")
        ).paginate(page=page, per_page=20)
    else:
        individuals = SanctionedIndividual.query.order_by(SanctionedIndividual.name.asc()).paginate(page=page, per_page=20)
        
    return render_template('sanctions.html', individuals=individuals, search_query=query_str)

@app.route('/sanctions/sync', methods=['POST'])
@login_required
def sync_sanctions():
    try:
        url = 'https://www.treasury.gov/ofac/downloads/sdn.csv'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=12) as response:
            lines = [line.decode('utf-8') for line in response.readlines()]
            reader = csv.reader(lines)
            
            new_entries = 0
            for row in reader:
                if len(row) > 2 and row[2].strip().lower() == 'individual':
                    name = row[1].strip()
                    program = f"US OFAC ({row[3].strip()})"
                    remarks = row[11].strip() if len(row) > 11 else "SDN Individual profile"
                    
                    exists = SanctionedIndividual.query.filter_by(name=name).first()
                    if not exists:
                        ind = SanctionedIndividual(name=name, source=program, reason=remarks)
                        db.session.add(ind)
                        new_entries += 1
                        if new_entries >= 100:
                            break
                            
            db.session.commit()
            log_audit_event(session['username'], f"Synced OFAC SDN registry. Added {new_entries} entries.")
            flash(f"Sanctions registry synced successfully! Added {new_entries} new sanctioned individuals.", "success")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error syncing sanctions: {e}")
        flash(f"Failed to sync sanctions registry from Treasury feed: {e}", "danger")
        
    return redirect(url_for('sanctions'))

@app.route('/api-docs')
@login_required
def api_docs():
    return render_template('api_docs.html')


# --- PLUGGABLE MOBILE MONEY API ENDPOINT ---

@app.route('/api/v1/evaluate-transaction', methods=['POST'])
def api_evaluate_transaction():
    """
    Endpoint for mobile money gateways (EcoCash, OneMoney, etc.) to check transactions in real-time.
    Expects JSON:
    {
      "sender_wallet": "+263771111111",
      "recipient_wallet": "+263772222222",
      "amount": 150.00,
      "currency": "USD",
      "payment_type": "P2P",
      "is_unfamiliar": 1,
      "is_new_device": 0,
      "account_age_days": 180
    }
    """
    data = request.get_json() or {}
    
    # Input validation
    required_fields = ['sender_wallet', 'recipient_wallet', 'amount', 'currency', 'payment_type']
    missing_fields = [f for f in required_fields if f not in data]
    
    if missing_fields:
        return jsonify({
            'success': False,
            'error': f"Missing required parameters: {', '.join(missing_fields)}"
        }), 400
        
    # Extract params
    sender = data.get('sender_wallet')
    recipient = data.get('recipient_wallet')
    amount = float(data.get('amount'))
    currency = data.get('currency')
    payment_type = data.get('payment_type')
    is_unfamiliar = bool(data.get('is_unfamiliar', 0))
    is_new_device = bool(data.get('is_new_device', 0))
    account_age = int(data.get('account_age_days', 365))
    sender_name = data.get('sender_name')
    recipient_name = data.get('recipient_name')
    
    # Process through pipeline
    tx = process_aml_evaluation(
        sender=sender,
        recipient=recipient,
        amount=amount,
        currency=currency,
        payment_type=payment_type,
        is_unfamiliar=is_unfamiliar,
        is_new_device=is_new_device,
        account_age_days=account_age,
        api_call=True,
        sender_name=sender_name,
        recipient_name=recipient_name
    )
    
    if not tx:
        return jsonify({
            'success': False,
            'error': "Engine classification failed."
        }), 500
        
    # Formulate recommendation
    # APPROVED for low risk
    # FLAGGED for medium risk (transaction finishes, but triggers dashboard alert)
    # BLOCKED for high risk (transaction is stopped)
    decision = 'APPROVED'
    if tx.risk_level == 'High':
        decision = 'BLOCKED'
    elif tx.risk_level == 'Medium':
        decision = 'FLAGGED'
        
    # Get alert reasons
    reasons = []
    if tx.alert:
        reasons = tx.alert.reason_codes.split(',')
    elif tx.risk_level == 'Medium':
        # Temporary reasons for flagged items that might not have Alert rows
        if is_new_device: reasons.append('NEW_DEVICE')
        if amount > 1000: reasons.append('HIGH_AMOUNT')
        
    response = {
        'success': True,
        'transaction_id': tx.transaction_id,
        'decision': decision,
        'risk_score': round(tx.risk_score, 4),
        'risk_level': tx.risk_level.upper(),
        'reasons': reasons,
        'evaluation_timestamp': tx.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'active_model_used': ACTIVE_MODEL
    }
    
    # Audit log entry for API usage
    log_audit_event("API_GATEWAY", f"Evaluated API transaction {tx.transaction_id} from {sender}. Decision: {decision}")
    
    return jsonify(response)


# --- CORE AML EVALUATION LOGIC ---

def process_aml_evaluation(sender, recipient, amount, currency, payment_type, is_unfamiliar, is_new_device, account_age_days, api_call=False, sender_name=None, recipient_name=None):
    """
    Core business logic function that:
    1. Screens sender and recipient names against a sanctions registry.
    2. Computes transaction velocity and daily volume in real-time.
    3. Runs feature vector through preprocessor + ML classifier.
    4. Triggers database records (Transaction + Alert).
    """
    try:
        now = datetime.datetime.utcnow()
        hour_of_day = now.hour
        
        # 1. Sanctions screening check
        sanctioned_match = None
        sanctions_reason = ""
        from models import SanctionedIndividual
        
        if sender_name:
            match = SanctionedIndividual.query.filter(
                SanctionedIndividual.name.ilike(f"%{sender_name}%")
            ).first()
            if match:
                sanctioned_match = match
                sanctions_reason = f"Sender name matched sanctions list: {match.name} ({match.source}) - {match.reason}"
                
        if not sanctioned_match and recipient_name:
            match = SanctionedIndividual.query.filter(
                SanctionedIndividual.name.ilike(f"%{recipient_name}%")
            ).first()
            if match:
                sanctioned_match = match
                sanctions_reason = f"Recipient name matched sanctions list: {match.name} ({match.source}) - {match.reason}"

        # 2. Calculate Real-time variables
        # Look back 1 hour for sender transactions
        one_hour_ago = now - datetime.timedelta(hours=1)
        prev_tx_1h = Transaction.query.filter(
            Transaction.sender_wallet == sender,
            Transaction.timestamp >= one_hour_ago
        ).count()
        velocity_1h = prev_tx_1h + 1 # Include the current transaction
        
        # Look back 24 hours for daily cumulative volume
        twenty_four_hours_ago = now - datetime.timedelta(hours=24)
        prev_tx_24h = Transaction.query.filter(
            Transaction.sender_wallet == sender,
            Transaction.timestamp >= twenty_four_hours_ago
        ).all()
        daily_volume = sum(tx.amount for tx in prev_tx_24h) + amount
        
        # 3. Decision mapping (Sanctions override ML model)
        model = None
        if sanctioned_match:
            risk_score = 1.0
            risk_level = 'High'
            status = 'Suspicious'
        else:
            # Prepare feature dictionary matching format in train_models.py
            feature_dict = {
                'currency': [currency],
                'payment_type': [payment_type],
                'amount': [amount],
                'hour_of_day': [hour_of_day],
                'velocity_1h': [velocity_1h],
                'daily_volume': [daily_volume],
                'account_age_days': [account_age_days],
                'is_unfamiliar': [1 if is_unfamiliar else 0],
                'is_new_device': [1 if is_new_device else 0]
            }
            
            X_eval = pd.DataFrame(feature_dict)
            model = get_model(ACTIVE_MODEL)
            
            if model:
                # Predict probability of suspicious classification
                risk_score = float(model.predict_proba(X_eval)[0, 1])
            else:
                # Safe Fallback if models are not generated yet (failsafe rule-based classification)
                app.logger.warning("ML models not loaded. Falling back to rule-based risk classification.")
                risk_score = 0.0
                if velocity_1h > 10 or amount > 5000 or (is_new_device and is_unfamiliar):
                    risk_score = 0.85
                elif amount > 1000 or velocity_1h > 4:
                    risk_score = 0.45
                    
            # Classify risk level
            if risk_score >= app.config['HIGH_RISK_THRESHOLD']:
                risk_level = 'High'
                status = 'Under Investigation'
            elif risk_score >= app.config['MEDIUM_RISK_THRESHOLD']:
                risk_level = 'Medium'
                status = 'Legitimate' # Medium is flagged but not necessarily auto-investigated
            else:
                risk_level = 'Low'
                status = 'Legitimate'
            
        # Unique ID generation
        tx_count = Transaction.query.count()
        transaction_id = f"TX-{100000 + tx_count + 1}"
        
        # Create Transaction record
        tx = Transaction(
            transaction_id=transaction_id,
            sender_wallet=sender,
            sender_name=sender_name or "Unknown Sender",
            recipient_wallet=recipient,
            recipient_name=recipient_name or "Unknown Recipient",
            amount=amount,
            currency=currency,
            payment_type=payment_type,
            timestamp=now,
            hour_of_day=hour_of_day,
            velocity_1h=velocity_1h,
            daily_volume=daily_volume,
            is_unfamiliar=is_unfamiliar,
            is_new_device=is_new_device,
            account_age_days=account_age_days,
            risk_score=risk_score,
            risk_level=risk_level,
            status=status
        )
        
        db.session.add(tx)
        
        # 4. Create Alert if risk exceeds threshold or sanctions matched
        if risk_level in ['Medium', 'High'] or sanctioned_match:
            # Determine reason codes based on feature triggers
            reasons = []
            if sanctioned_match:
                reasons.append("SANCTIONED_INDIVIDUAL")
            elif model:
                # Reasons come from the model itself: the features that pushed this score up most
                features = {k: v[0] for k, v in feature_dict.items()}
                reasons = explain.reason_codes(explain.explain(model, features))
            else:
                if velocity_1h > 5: reasons.append("HIGH_VELOCITY")
                large_amt_limit = 2000
                if currency == 'ZWG':
                    large_amt_limit = 50000
                elif currency == 'ZAR':
                    large_amt_limit = 36000
                if amount > large_amt_limit: reasons.append("LARGE_AMOUNT")
                if is_new_device: reasons.append("NEW_DEVICE")
                if is_unfamiliar: reasons.append("UNFAMILIAR_RECIPIENT")
                if account_age_days < 30: reasons.append("NEW_ACCOUNT")
                if hour_of_day in [0, 1, 2, 3, 4, 23]: reasons.append("SUSPICIOUS_HOUR")
                
                if not reasons:
                    reasons.append("ML_CLASSIFIED_ANOMALY")
                
            alert = Alert(
                transaction_id=transaction_id,
                risk_score=risk_score,
                risk_level=risk_level,
                reason_codes=",".join(reasons),
                status='Pending',
                resolution_notes=sanctions_reason if sanctioned_match else None
            )
            db.session.add(alert)
            
        db.session.commit()
        return tx
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in process_aml_evaluation: {e}")
        return None


# --- GENERATIVE AI (OPENROUTER & NVIDIA NIM) INTEGRATION ---

def call_llm_api(prompt, system_prompt="You are an expert AI Anti-Money Laundering (AML) Compliance Specialist."):
    """
    Unified LLM API Caller supporting:
    1. NVIDIA NIM Microservices (if NVIDIA_NIM_API_KEY is configured)
    2. OpenRouter Unified API Gateway (if OPENROUTER_API_KEY is configured)
    3. Intelligent Rule-Based AI Summary Heuristics Engine (Fallback if no key is set)
    """
    nvidia_key = app.config.get('NVIDIA_NIM_API_KEY')
    openrouter_key = app.config.get('OPENROUTER_API_KEY')

    # 0. Try Claude (Anthropic) when ANTHROPIC_API_KEY is configured
    if ai_assistant.is_configured():
        try:
            text = ai_assistant.complete(prompt, system_prompt)
            if text:
                return text, f"Claude ({ai_assistant.MODEL})"
        except anthropic.APIError as e:
            app.logger.warning(f"Claude API call failed: {e}. Trying other providers...")

    # 1. Try NVIDIA NIM
    if nvidia_key:
        try:
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {nvidia_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": app.config.get('NVIDIA_NIM_MODEL', 'meta/llama-3.1-70b-instruct'),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 250
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                text = data['choices'][0]['message']['content'].strip()
                return text, "NVIDIA NIM (Llama-3.1-70B)"
        except Exception as e:
            app.logger.warning(f"NVIDIA NIM API call failed: {e}. Trying OpenRouter...")

    # 2. Try OpenRouter
    if openrouter_key:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "HTTP-Referer": "http://localhost:5002",
                "X-Title": "AML System",
                "Content-Type": "application/json"
            }
            payload = {
                "model": app.config.get('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct:free'),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 250
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                text = data['choices'][0]['message']['content'].strip()
                return text, "OpenRouter (Llama 3.1)"
        except Exception as e:
            app.logger.warning(f"OpenRouter API call failed: {e}. Using fallback generator...")

    return None, None


def generate_ai_transaction_summary(tx):
    """
    Generates an automated AI Forensic Case Summary for a given Transaction/Alert.
    """
    reasons_str = tx.alert.reason_codes if tx.alert else "ML_ANOMALY"
    
    prompt = f"""
    Analyze the following financial transaction for AML risk:
    - Transaction ID: {tx.transaction_id}
    - Sender: {tx.sender_name} ({tx.sender_wallet})
    - Recipient: {tx.recipient_name} ({tx.recipient_wallet})
    - Amount: {tx.amount:,.2f} {tx.currency}
    - Payment Type: {tx.payment_type}
    - Risk Level: {tx.risk_level} (Model Probability: {tx.risk_score * 100:.1f}%)
    - 1-Hour Velocity: {tx.velocity_1h} transactions
    - 24-Hour Daily Volume: {tx.daily_volume:,.2f} {tx.currency}
    - Unfamiliar Recipient: {tx.is_unfamiliar}
    - New Hardware Device: {tx.is_new_device}
    - Triggered Indicators: {reasons_str}

    Provide a concise 2-sentence executive summary explaining the risk level and recommended next action.
    """
    
    llm_output, provider = call_llm_api(prompt)
    if llm_output:
        return {
            'summary': llm_output,
            'provider': provider
        }
        
    # Heuristic AI Engine Fallback
    bullets = []
    if tx.risk_level == 'High':
        bullets.append(f"Transaction {tx.transaction_id} flagged HIGH RISK ({tx.risk_score*100:.1f}%) due to significant behavioral deviations.")
    elif tx.risk_level == 'Medium':
        bullets.append(f"Transaction {tx.transaction_id} flagged MEDIUM RISK ({tx.risk_score*100:.1f}%) requiring routine compliance review.")
    else:
        bullets.append(f"Transaction {tx.transaction_id} verified LOW RISK ({tx.risk_score*100:.1f}%) consistent with baseline wallet behavior.")
        
    if tx.velocity_1h > 3:
        bullets.append(f"High transfer frequency observed ({tx.velocity_1h} transactions within 1 hour).")
    if tx.amount > 2000 or (tx.currency == 'ZWG' and tx.amount > 50000):
        bullets.append(f"Substantial outbound transfer of {tx.amount:,.2f} {tx.currency} exceeds standard single-transaction limits.")
    if tx.is_new_device:
        bullets.append("Execution from an unverified hardware device ID.")
    if tx.is_unfamiliar:
        bullets.append("Transfer directed to a newly linked, unfamiliar recipient wallet.")
        
    summary_text = " ".join(bullets) + " Recommended Action: " + (
        "Immediately block funds and file a Suspicious Activity Report (SAR) with FIU." if tx.risk_level == 'High'
        else "Assign case to compliance analyst for KYC re-verification." if tx.risk_level == 'Medium'
        else "Approve transaction for normal settlement."
    )
    
    return {
        'summary': summary_text,
        'provider': 'AI Rule Heuristic Engine'
    }


# --- FLASK AI API ENDPOINTS ---

@app.route('/api/ai/explain/<transaction_id>', methods=['GET', 'POST'])
@login_required
def api_explain_transaction(transaction_id):
    tx = Transaction.query.filter_by(transaction_id=transaction_id).first_or_404()
    result = generate_ai_transaction_summary(tx)
    return jsonify({
        'success': True,
        'transaction_id': tx.transaction_id,
        'risk_level': tx.risk_level,
        'risk_score': round(tx.risk_score, 4),
        'ai_summary': result['summary'],
        'provider': result['provider']
    })


@app.route('/api/ai/executive-summary', methods=['GET'])
@login_required
def api_executive_summary():
    total_tx = Transaction.query.count()
    high_risk_count = Transaction.query.filter_by(risk_level='High').count()
    pending_alerts = Alert.query.filter_by(status='Pending').count()
    
    prompt = f"""
    You are the Chief Compliance AI Officer.
    System Status:
    - Total Transactions Processed: {total_tx}
    - Total High Risk / Suspicious Incidents: {high_risk_count}
    - Unresolved Pending Alerts: {pending_alerts}

    Provide a 2-sentence executive summary of the overall platform security posture.
    """
    
    llm_output, provider = call_llm_api(prompt)
    if not llm_output:
        llm_output = f"Platform security monitoring active across {total_tx} transactions. Currently maintaining {high_risk_count} high-risk incidents and {pending_alerts} pending alerts requiring compliance queue review."
        provider = "AI Rule Heuristic Engine"
        
    return jsonify({
        'success': True,
        'summary': llm_output,
        'provider': provider
    })


@app.route('/assistant')
@login_required
def assistant():
    return render_template('assistant.html', ai_configured=ai_assistant.is_configured(),
                           ai_model=ai_assistant.MODEL)


@app.route('/api/ai/chat', methods=['POST'])
@login_required
def api_ai_chat():
    data = request.get_json() or {}
    user_query = (data.get('message') or data.get('query') or '').strip()
    if not user_query:
        return jsonify({'success': False, 'error': 'Please type a question.'}), 400
    if data.get('transaction_id'):
        user_query += f"\n\n(Context: I am looking at transaction {data['transaction_id']}.)"
    if not ai_assistant.is_configured():
        return jsonify({'success': False, 'configured': False,
                        'error': 'The AI assistant is not connected yet. Set the ANTHROPIC_API_KEY '
                                 'environment variable and restart the app.'}), 503

    log_audit_event(session.get('username', 'unknown'), f"AI Assistant query: {user_query[:120]}")
    result = ai_assistant.chat(user_query, data.get('conversation_id'))
    if result.get('error'):
        return jsonify({'success': False, 'error': result['error'],
                        'conversation_id': result['conversation_id']}), 502
    return jsonify({'success': True, 'response': result['response'],
                    'conversation_id': result['conversation_id'],
                    'tools_used': result['tools_used'],
                    'provider': f"Claude ({result['model']})"})


@app.route('/api/ai/chat/reset', methods=['POST'])
@login_required
def api_ai_chat_reset():
    ai_assistant.reset((request.get_json() or {}).get('conversation_id'))
    return jsonify({'success': True})


if __name__ == '__main__':
    # Ensure directory layout exists
    os.makedirs(os.path.join(app.root_path, 'data'), exist_ok=True)
    
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=True, host='0.0.0.0', port=port)
