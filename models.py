import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='Analyst') # 'Administrator' or 'Analyst'
    status = db.Column(db.String(20), nullable=False, default='Active') # 'Active' or 'Suspended'
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    transaction_id = db.Column(db.String(50), primary_key=True)
    sender_wallet = db.Column(db.String(50), nullable=False)
    sender_name = db.Column(db.String(100), nullable=True)
    recipient_wallet = db.Column(db.String(50), nullable=False)
    recipient_name = db.Column(db.String(100), nullable=True)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(10), nullable=False) # 'USD', 'ZiG'
    payment_type = db.Column(db.String(20), nullable=False) # 'P2P', 'Cash-Out', 'Merchant'
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    hour_of_day = db.Column(db.Integer, nullable=False)
    velocity_1h = db.Column(db.Integer, nullable=False)
    daily_volume = db.Column(db.Float, nullable=False)
    is_unfamiliar = db.Column(db.Boolean, default=False)
    is_new_device = db.Column(db.Boolean, default=False)
    account_age_days = db.Column(db.Integer, nullable=True) # Wallet age when the transaction was made
    risk_score = db.Column(db.Float, nullable=False, default=0.0) # Model probability output
    risk_level = db.Column(db.String(10), nullable=False, default='Low') # 'Low', 'Medium', 'High'
    status = db.Column(db.String(20), nullable=False, default='Legitimate') # 'Legitimate', 'Suspicious', 'Under Investigation'
    
    # Relationship to Alert
    alert = db.relationship('Alert', backref='transaction', uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Transaction {self.transaction_id} - {self.amount} {self.currency}>'

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(50), db.ForeignKey('transactions.transaction_id'), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(10), nullable=False) # 'Low', 'Medium', 'High'
    reason_codes = db.Column(db.String(200), nullable=False) # comma-separated like "HIGH_VELOCITY,LARGE_AMOUNT"
    status = db.Column(db.String(20), nullable=False, default='Pending') # 'Pending', 'Under Review', 'Closed'
    assigned_to = db.Column(db.String(80), nullable=True) # Username of the analyst
    resolution_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<Alert {self.id} for {self.transaction_id}>'

class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    event = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    ip_address = db.Column(db.String(50), nullable=True)
    
    def __repr__(self):
        return f'<AuditLog {self.username} - {self.event}>'

class SanctionedIndividual(db.Model):
    __tablename__ = 'sanctioned_individuals'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False, index=True)
    source = db.Column(db.String(50), nullable=False) # e.g. "US OFAC", "UN", "EU"
    reason = db.Column(db.String(250), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<SanctionedIndividual {self.name} ({self.source})>'
