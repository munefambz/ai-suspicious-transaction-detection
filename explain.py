"""Per-transaction explanations taken from the trained model itself.

For tree models (Decision Tree, Random Forest) each prediction is decomposed by
following the decision path of every tree: whenever a split moves the predicted
fraud probability up or down, that change is credited to the feature the split
used (the "path contribution" method). For Logistic Regression the contribution
is the feature's weight times its scaled value. The result shows how much each
input pushed this particular transaction's risk score up or down.
"""
import numpy as np
import pandas as pd

FEATURES = ['currency', 'payment_type', 'amount', 'hour_of_day', 'velocity_1h', 'daily_volume',
            'account_age_days', 'is_unfamiliar', 'is_new_device']

# reason code stored on alerts, and the label shown to analysts
REASON = {
    'velocity_1h': ('HIGH_VELOCITY', 'Transfers in the last hour'),
    'amount': ('LARGE_AMOUNT', 'Transaction amount'),
    'daily_volume': ('HIGH_DAILY_VOLUME', '24-hour transfer volume'),
    'is_new_device': ('NEW_DEVICE', 'Device used'),
    'is_unfamiliar': ('UNFAMILIAR_RECIPIENT', 'Recipient familiarity'),
    'account_age_days': ('ACCOUNT_AGE', 'Account age'),
    'hour_of_day': ('SUSPICIOUS_HOUR', 'Time of day'),
    'currency': ('CURRENCY_PATTERN', 'Currency'),
    'payment_type': ('PAYMENT_TYPE_PATTERN', 'Payment type'),
}


def _original_feature(transformed_name):
    """'cat__currency_USD' -> 'currency', 'num__amount' -> 'amount'."""
    name = transformed_name.split('__', 1)[-1]
    for f in FEATURES:
        if name == f or name.startswith(f + '_') and f in ('currency', 'payment_type'):
            return f
    return name


def _describe(feature, value):
    if feature == 'velocity_1h':
        return f"{int(value)} transfer(s) in the last hour"
    if feature == 'amount':
        return f"{value:,.2f}"
    if feature == 'daily_volume':
        return f"{value:,.2f} sent in 24 hours"
    if feature == 'is_new_device':
        return 'new, unrecognised device' if value else 'known device'
    if feature == 'is_unfamiliar':
        return 'first transfer to this recipient' if value else 'known recipient'
    if feature == 'account_age_days':
        return 'unknown' if value is None or pd.isna(value) else f"{int(value)} days old"
    if feature == 'hour_of_day':
        return f"{int(value):02d}:00"
    return str(value)


def explain(pipeline, features):
    """Return the model's explanation for one transaction.

    features: dict with the 9 model inputs.
    Result: {'base': probability before any feature, 'score': final probability,
             'factors': [{'feature', 'code', 'label', 'value', 'contribution', 'share'}, ...]}
    sorted by contribution, largest risk-increasing factor first.
    """
    X = pd.DataFrame({f: [features[f]] for f in FEATURES})
    pre = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['classifier']
    Xt = pre.transform(X)
    if hasattr(Xt, 'toarray'):
        Xt = Xt.toarray()
    names = pre.get_feature_names_out()
    pos = list(clf.classes_).index(1)
    contrib = np.zeros(len(names))

    trees = clf.estimators_ if hasattr(clf, 'estimators_') else ([clf] if hasattr(clf, 'tree_') else None)
    if trees is not None:
        base = 0.0
        for tree in trees:
            t = tree.tree_
            vals = t.value[:, 0, :]
            p1 = vals[:, pos] / vals.sum(axis=1)
            path = tree.decision_path(Xt).indices
            base += p1[path[0]]
            for parent, child in zip(path[:-1], path[1:]):
                contrib[t.feature[parent]] += p1[child] - p1[parent]
        base /= len(trees)
        contrib /= len(trees)
        score = base + contrib.sum()
    else:  # linear model: contribution in log-odds, rescaled to the probability change
        logit = clf.coef_[0] * Xt[0]
        base_logit = clf.intercept_[0]
        score = 1 / (1 + np.exp(-(base_logit + logit.sum())))
        base = 1 / (1 + np.exp(-base_logit))
        total = logit.sum()
        contrib = logit * ((score - base) / total) if total else logit * 0

    by_feature = {}
    for name, c in zip(names, contrib):
        f = _original_feature(name)
        by_feature[f] = by_feature.get(f, 0.0) + float(c)

    positive = sum(c for c in by_feature.values() if c > 0) or 1.0
    factors = []
    for f, c in sorted(by_feature.items(), key=lambda kv: kv[1], reverse=True):
        code, label = REASON.get(f, (f.upper(), f))
        factors.append({'feature': f, 'code': code, 'label': label,
                        'value': _describe(f, features[f]),
                        'contribution': round(c, 4),
                        'share': round(c / positive, 3) if c > 0 else 0.0})
    return {'base': round(float(base), 4), 'score': round(float(score), 4), 'factors': factors}


def reason_codes(explanation, min_share=0.1, limit=4):
    """Codes for the features that pushed the score up the most."""
    codes = [f['code'] for f in explanation['factors'] if f['contribution'] > 0 and f['share'] >= min_share]
    return codes[:limit] or ['ML_CLASSIFIED_ANOMALY']


def transaction_features(tx, account_age_default=365):
    """Model inputs from a stored Transaction row."""
    age = getattr(tx, 'account_age_days', None)
    return {
        'currency': tx.currency, 'payment_type': tx.payment_type, 'amount': tx.amount,
        'hour_of_day': tx.hour_of_day, 'velocity_1h': tx.velocity_1h, 'daily_volume': tx.daily_volume,
        'account_age_days': age if age is not None else account_age_default,
        'is_unfamiliar': int(bool(tx.is_unfamiliar)), 'is_new_device': int(bool(tx.is_new_device)),
    }
