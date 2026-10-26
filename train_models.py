import os
import json
import random
import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def scale_amount(amount, currency):
    if currency == 'ZWG':
        return round(amount * 25.0, 2)
    elif currency == 'ZAR':
        return round(amount * 18.0, 2)
    return amount

def _hour(rng, kind):
    """Legitimate activity follows a daytime pattern; some fraud prefers the night."""
    if kind == 'night':
        return rng.choice([0, 1, 2, 3, 4, 23])
    return int(rng.choices(range(24), weights=[1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9,
                                                 9, 9, 9, 9, 9, 9, 8, 7, 6, 4, 3, 2])[0])


def generate_synthetic_data(num_records=5000, seed=42, id_start=100000,
                            start_time=datetime.datetime(2026, 8, 1)):
    """Simulate Zimbabwean mobile money transactions with realistic overlap.

    Real fraud does not separate cleanly from normal behaviour, so besides
    ordinary customers the data includes legitimate look-alikes (busy agents and
    merchants, genuine large payments, people on a new phone, brand-new accounts)
    and subtle fraud that resembles normal use. About 0.5% of labels are also
    flipped to mimic investigation errors in real case data.
    """
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    payment_types = ['P2P', 'Cash-Out', 'Merchant']
    currencies = ['USD', 'ZWG', 'ZAR']
    data = []

    for i in range(num_records):
        sender = f"{rng.choice(['+26377', '+26378', '+26371', '+26373'])}{rng.randint(1000000, 9999999)}"
        recipient = f"{rng.choice(['+26377', '+26378', '+26371', '+26373'])}{rng.randint(1000000, 9999999)}"
        payment_type = rng.choice(payment_types)
        currency = rng.choices(currencies, weights=[0.6, 0.25, 0.15])[0]
        account_age_days = rng.randint(31, 1500)
        is_unfamiliar = int(rng.random() < 0.15)
        is_new_device = int(rng.random() < 0.05)
        hour = _hour(rng, 'day')
        velocity_1h = rng.randint(1, 4)
        label = 0

        kind = rng.choices(
            ['normal', 'busy_agent', 'legit_large', 'new_phone', 'new_customer',
             'structuring', 'large_takeout', 'account_takeover', 'subtle'],
            weights=[0.80, 0.045, 0.035, 0.03, 0.03, 0.017, 0.014, 0.017, 0.012])[0]

        if kind == 'normal':
            amount = nrng.exponential(35.0) + 1.0
        elif kind == 'busy_agent':          # agents and merchants legitimately transact fast
            payment_type = rng.choice(['Merchant', 'Cash-Out'])
            amount = rng.uniform(5, 300)
            velocity_1h = rng.randint(5, 18)
            is_unfamiliar = int(rng.random() < 0.4)
        elif kind == 'legit_large':         # rent, school fees, stock purchases
            amount = rng.uniform(800, 8000)
            is_unfamiliar = int(rng.random() < 0.4)
            is_new_device = int(rng.random() < 0.1)
            if rng.random() < 0.15:
                hour = _hour(rng, 'night')
        elif kind == 'new_phone':           # customer upgraded their handset
            amount = nrng.exponential(60.0) + 1.0
            is_new_device = 1
            is_unfamiliar = int(rng.random() < 0.3)
            velocity_1h = rng.randint(1, 6)
        elif kind == 'new_customer':
            amount = nrng.exponential(40.0) + 1.0
            account_age_days = rng.randint(1, 30)
            is_unfamiliar = int(rng.random() < 0.5)
            is_new_device = int(rng.random() < 0.3)
        elif kind == 'structuring':         # many small transfers to stay under limits
            label = 1
            amount = rng.uniform(5, 200)
            velocity_1h = rng.randint(6, 30)
            is_unfamiliar = int(rng.random() < 0.8)
            is_new_device = int(rng.random() < 0.5)
            if rng.random() < 0.4:
                hour = _hour(rng, 'night')
        elif kind == 'large_takeout':       # one big transfer out, often from a new account at night
            label = 1
            amount = rng.uniform(1500, 20000)
            velocity_1h = rng.randint(1, 3)
            is_unfamiliar = int(rng.random() < 0.8)
            is_new_device = int(rng.random() < 0.6)
            if rng.random() < 0.6:
                account_age_days = rng.randint(1, 30)
            if rng.random() < 0.5:
                hour = _hour(rng, 'night')
        elif kind == 'account_takeover':    # SIM swap / stolen PIN on an established account
            label = 1
            amount = rng.uniform(100, 1500)
            velocity_1h = rng.randint(3, 12)
            is_unfamiliar = int(rng.random() < 0.75)
            is_new_device = int(rng.random() < 0.85)
            account_age_days = rng.randint(100, 1500)
        else:                               # subtle: mule wallet the victim already knows
            label = 1
            amount = rng.uniform(200, 2000)
            velocity_1h = rng.randint(2, 6)
            is_unfamiliar = int(rng.random() < 0.3)
            is_new_device = int(rng.random() < 0.3)

        amount = max(0.5, scale_amount(round(float(amount), 2), currency))
        daily_volume = round(max(amount, amount * velocity_1h * rng.uniform(0.6, 1.3)
                                 if velocity_1h > 4 else amount * rng.uniform(1.0, 2.5)), 2)
        if rng.random() < 0.005:            # imperfect investigation labels
            label = 1 - label

        timestamp = start_time + datetime.timedelta(days=rng.randint(0, 29))
        timestamp = timestamp.replace(hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59))
        data.append({
            'transaction_id': f"TX-{id_start + i}",
            'sender_wallet': sender,
            'recipient_wallet': recipient,
            'amount': amount,
            'currency': currency,
            'payment_type': payment_type,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'hour_of_day': hour,
            'velocity_1h': velocity_1h,
            'daily_volume': daily_volume,
            'is_unfamiliar': is_unfamiliar,
            'account_age_days': account_age_days,
            'is_new_device': is_new_device,
            'scenario': kind,
            'label': label
        })

    return pd.DataFrame(data)

def train_and_evaluate():
    print("Generating simulated mobile money dataset...")
    df = generate_synthetic_data(5000)
    
    # Save the raw training set to CSV for recordkeeping / visualization
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/simulated_transactions.csv', index=False)
    
    # Define features and target
    categorical_features = ['currency', 'payment_type']
    numerical_features = ['amount', 'hour_of_day', 'velocity_1h', 'daily_volume', 'account_age_days', 'is_unfamiliar', 'is_new_device']
    target = 'label'
    
    X = df[categorical_features + numerical_features]
    y = df[target]
    
    # Split into train/test (80/20) BEFORE fitting preprocessing pipeline (avoiding data leakage!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scenarios_test = df.loc[X_test.index, 'scenario']
    
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"Suspicious cases: Train={y_train.sum()} ({y_train.mean()*100:.2f}%), Test={y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Preprocessor definition
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
    
    # Dictionary of models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    }
    
    metrics_summary = {}
    details = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Build pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else [0.0]*len(y_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        print(f"Results for {model_name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        metrics_summary[model_name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': cm
        }
        
        # Detection rate per simulated scenario, to show which fraud types are caught or missed
        breakdown = {}
        for scen in sorted(scenarios_test.unique()):
            mask = (scenarios_test == scen).values
            yt, yp = y_test.values[mask], y_pred[mask]
            breakdown[scen] = {'transactions': int(mask.sum()), 'labelled_suspicious': int(yt.sum()),
                               'caught': int(((yp == 1) & (yt == 1)).sum()),
                               'false_alarms': int(((yp == 1) & (yt == 0)).sum())}
        details[model_name] = breakdown

        # Save trained pipeline
        model_filename = f"data/{model_name.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(pipeline, model_filename)
        print(f"Saved model pipeline to {model_filename}")
        
    # Save metrics to JSON
    with open('data/metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    with open('data/evaluation_details.json', 'w') as f:
        json.dump({'test_set_size': int(len(y_test)), 'test_set_suspicious': int(y_test.sum()),
                   'per_scenario': details}, f, indent=4)

    print("\nTraining complete! Metrics saved to data/metrics.json")

if __name__ == "__main__":
    train_and_evaluate()
