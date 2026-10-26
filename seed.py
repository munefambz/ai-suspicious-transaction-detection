import os
import datetime
import random
import joblib
import pandas as pd
from app import app, bcrypt
import explain
from train_models import generate_synthetic_data
from models import db, User, Transaction, Alert, AuditLog, SanctionedIndividual

def seed_database():
    print("Initializing Database...")
    
    with app.app_context():
        # Drop and recreate all tables
        db.drop_all()
        db.create_all()
        
        print("Creating Users...")
        # Create administrator account
        admin_pass = bcrypt.generate_password_hash("admin123").decode('utf-8')
        admin = User(
            username="admin",
            password_hash=admin_pass,
            role="Administrator",
            status="Active"
        )
        
        # Create security analyst account
        analyst_pass = bcrypt.generate_password_hash("analyst123").decode('utf-8')
        analyst = User(
            username="analyst",
            password_hash=analyst_pass,
            role="Analyst",
            status="Active"
        )
        
        db.session.add(admin)
        db.session.add(analyst)
        db.session.commit()
        
        print("Creating Sanctioned Individuals...")
        sanctioned_names = [
            ("Vladimir Putin", "EU/US Sanctions", "President of Russia, sanctioned for invasion of Ukraine"),
            ("Viktor Bout", "US OFAC SDN", "Arms dealer, transnational criminal agent"),
            ("Slobodan Milosevic", "EU Sanctions", "Former Yugoslav President, war crimes (historical)"),
            ("Bashar al-Assad", "US OFAC SDN", "President of Syria, human rights violations"),
            ("Nicolas Maduro", "US OFAC SDN", "President of Venezuela, drug trafficking and corruption"),
            ("Kim Jong Un", "UN Sanctions", "Supreme Leader of North Korea, nuclear weapons development"),
            ("Viktor Vekselberg", "US OFAC SDN", "Russian oligarch, close ties to Kremlin"),
            ("Dawood Ibrahim", "UN Sanctions", "Underworld leader, terrorist activities"),
            ("Joaquin Guzman", "US OFAC SDN", "Former leader of Sinaloa Cartel, drug trafficking"),
            ("Viktor Yanukovych", "EU Sanctions", "Former President of Ukraine, corruption and treason"),
            ("Alexander Lukashenko", "EU/US Sanctions", "President of Belarus, human rights violations"),
            ("Ismael Zambada Garcia", "US OFAC SDN", "Sinaloa Cartel leader"),
            ("Nemecio Oseguera Cervantes", "US OFAC SDN", "CJNG cartel leader"),
            ("Ramzan Kadyrov", "US OFAC SDN", "Head of Chechen Republic, human rights violations"),
            ("Viktor Zubkov", "UK Sanctions", "Chairman of Gazprom Board"),
            ("Joseph Kony", "UN Sanctions", "Leader of Lord's Resistance Army"),
            ("Kweku Adoboli", "UK FCA", "Former trader convicted of fraud (regulatory bans)")
        ]
        
        for name, source, reason in sanctioned_names:
            ind = SanctionedIndividual(name=name, source=source, reason=reason)
            db.session.add(ind)
        db.session.commit()
        
        model_path = app.config['MODELS'][app.config['DEFAULT_MODEL']]
        if not os.path.exists(model_path):
            print("Trained model not found! Please run train_models.py first.")
            return
        model = joblib.load(model_path)

        # Fresh transactions the model has never seen (different random seed from training),
        # dated over the 30 days up to today, and scored by the real classifier.
        live = generate_synthetic_data(1000, seed=2026, id_start=100000,
                                       start_time=datetime.datetime(2026, 9, 21))
        live = live.sort_values('timestamp').reset_index(drop=True)
        X = live[explain.FEATURES]
        live['risk_score'] = model.predict_proba(X)[:, 1]

        medium, high = app.config['MEDIUM_RISK_THRESHOLD'], app.config['HIGH_RISK_THRESHOLD']
        live['risk_level'] = ['High' if p >= high else 'Medium' if p >= medium else 'Low' for p in live['risk_score']]
        flagged = live[live['risk_level'] != 'Low']
        # Analyst workflow: the newest alerts are still waiting, recent ones are being worked,
        # older ones are closed with the true outcome (so genuine false positives appear).
        n = len(flagged)
        pending_ids = set(flagged['transaction_id'].iloc[-3:])
        review_ids = set(flagged['transaction_id'].iloc[int(n * 0.6):-3])

        transactions_to_add, alerts_to_add = [], []
        print(f"Scoring and seeding {len(live)} transactions with {app.config['DEFAULT_MODEL']}...")
        for _, row in live.iterrows():
            ts = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            tx_id = row['transaction_id']
            is_flagged = row['risk_level'] != 'Low'
            if not is_flagged:
                status = 'Legitimate'
            elif tx_id in pending_ids or tx_id in review_ids:
                status = 'Under Investigation'
            else:
                status = 'Suspicious' if int(row['label']) == 1 else 'Legitimate'

            transactions_to_add.append(Transaction(
                transaction_id=tx_id,
                sender_wallet=row['sender_wallet'],
                recipient_wallet=row['recipient_wallet'],
                amount=float(row['amount']),
                currency=row['currency'],
                payment_type=row['payment_type'],
                timestamp=ts,
                hour_of_day=int(row['hour_of_day']),
                velocity_1h=int(row['velocity_1h']),
                daily_volume=float(row['daily_volume']),
                is_unfamiliar=bool(row['is_unfamiliar']),
                is_new_device=bool(row['is_new_device']),
                account_age_days=int(row['account_age_days']),
                risk_score=float(row['risk_score']),
                risk_level=row['risk_level'],
                status=status
            ))

            if is_flagged:
                features = {f: row[f] for f in explain.FEATURES}
                codes = explain.reason_codes(explain.explain(model, features))
                if tx_id in pending_ids:
                    alert_status, assigned, resolution = 'Pending', None, None
                elif tx_id in review_ids:
                    alert_status, assigned, resolution = 'Under Review', 'analyst', None
                elif int(row['label']) == 1:
                    alert_status, assigned = 'Closed', 'analyst'
                    resolution = "Confirmed fraud after customer contact. Wallet frozen and case reported."
                else:
                    alert_status, assigned = 'Closed', 'analyst'
                    resolution = "False positive: customer verified the transaction as genuine."
                alerts_to_add.append(Alert(
                    transaction_id=tx_id,
                    risk_score=float(row['risk_score']),
                    risk_level=row['risk_level'],
                    reason_codes=",".join(codes),
                    status=alert_status,
                    assigned_to=assigned,
                    resolution_notes=resolution,
                    created_at=ts + datetime.timedelta(minutes=2)
                ))

        db.session.bulk_save_objects(transactions_to_add)
        db.session.commit()
        db.session.bulk_save_objects(alerts_to_add)
        db.session.commit()

        caught = int(((live['risk_level'] != 'Low') & (live['label'] == 1)).sum())
        print(f"Model flagged {n} of {len(live)}; {caught} of {int(live['label'].sum())} labelled-suspicious caught.")

        # Log system setup events
        db.session.add(AuditLog(username="SYSTEM", event="Database schemas initialized"))
        db.session.add(AuditLog(username="SYSTEM", event="Seeded admin and analyst accounts"))
        db.session.add(AuditLog(username="SYSTEM", event=f"Imported {len(transactions_to_add)} transactions scored by {app.config['DEFAULT_MODEL']}"))
        db.session.commit()
        
        print(f"Seeding completed successfully! Imported {len(transactions_to_add)} transactions and {len(alerts_to_add)} alerts.")

if __name__ == "__main__":
    seed_database()
