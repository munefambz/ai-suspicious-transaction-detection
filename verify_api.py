import json
from app import app, db
from models import Transaction, Alert

def test_sentinel_api():
    print("Initializing Flask test client...")
    client = app.test_client()
    
    with app.app_context():
        # Read database counts before testing
        start_tx_count = Transaction.query.count()
        start_alert_count = Alert.query.count()
        print(f"Current DB state: Transactions={start_tx_count}, Alerts={start_alert_count}")
        
        # Test Case 1: Legitimate low-amount transaction
        print("\nTesting Case 1: Legitimate P2P Transfer...")
        payload_1 = {
            "sender_wallet": "+263772222222",
            "recipient_wallet": "+263773333333",
            "amount": 25.50,
            "currency": "USD",
            "payment_type": "P2P",
            "is_unfamiliar": 0,
            "is_new_device": 0,
            "account_age_days": 300
        }
        response_1 = client.post('/api/v1/evaluate-transaction', 
                                 data=json.dumps(payload_1), 
                                 content_type='application/json')
        
        assert response_1.status_code == 200, f"Expected 200, got {response_1.status_code}"
        res_data_1 = json.loads(response_1.data)
        print("API Response:", json.dumps(res_data_1, indent=2))
        assert res_data_1['decision'] == 'APPROVED', f"Expected APPROVED, got {res_data_1['decision']}"
        assert res_data_1['risk_level'] == 'LOW', f"Expected LOW, got {res_data_1['risk_level']}"
        
        # Test Case 2: High Amount Anomalous Transfer (blocked)
        print("\nTesting Case 2: Suspicious Large Off-hours Transfer...")
        payload_2 = {
            "sender_wallet": "+263775555555",
            "recipient_wallet": "+263718888888",
            "amount": 15000.00,
            "currency": "USD",
            "payment_type": "Merchant",
            "is_unfamiliar": 1,
            "is_new_device": 1,
            "account_age_days": 5
        }
        response_2 = client.post('/api/v1/evaluate-transaction', 
                                 data=json.dumps(payload_2), 
                                 content_type='application/json')
        
        assert response_2.status_code == 200, f"Expected 200, got {response_2.status_code}"
        res_data_2 = json.loads(response_2.data)
        print("API Response:", json.dumps(res_data_2, indent=2))
        assert res_data_2['decision'] == 'BLOCKED', f"Expected BLOCKED, got {res_data_2['decision']}"
        assert res_data_2['risk_level'] == 'HIGH', f"Expected HIGH, got {res_data_2['risk_level']}"
        
        # Verify changes in SQLite
        end_tx_count = Transaction.query.count()
        end_alert_count = Alert.query.count()
        unassigned_alerts = Alert.query.filter_by(status='Pending').count()
        print(f"\nFinal DB state: Transactions={end_tx_count}, Alerts={end_alert_count}, Unassigned Alerts={unassigned_alerts}")
        
        assert end_tx_count == start_tx_count + 2, "Expected 2 new transactions in DB"
        assert end_alert_count == start_alert_count + 1, "Expected 1 new alert in DB (Case 2)"
        
        # Verify exact target statistics requested by user
        assert end_tx_count == 1003, f"Expected exactly 1003 transactions, got {end_tx_count}"
        assert end_alert_count == 58, f"Expected exactly 58 alerts, got {end_alert_count}"
        assert unassigned_alerts == 2, f"Expected exactly 2 unassigned pending alerts, got {unassigned_alerts}"
        
        print("\nAll automated API integration assertions passed successfully!")

if __name__ == "__main__":
    test_sentinel_api()
