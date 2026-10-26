"""AI compliance assistant: Claude with read-only tools over the AML database.

Claude answers questions by calling the tools below (which query the live
SQLite/MySQL data through SQLAlchemy), so every figure it quotes comes from the
system rather than from the model's memory.
"""
import json
import os
import uuid
from collections import Counter

import anthropic
from anthropic import beta_tool
from sqlalchemy import case, func, or_

from models import db, Transaction, Alert, AuditLog

MODEL = os.environ.get('AI_MODEL', 'claude-opus-5-5')
METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'metrics.json')

SYSTEM_PROMPT = """You are the AI Compliance Assistant built into an AML (anti-money laundering) \
suspicious transaction detection system for mobile money platforms in Zimbabwe (EcoCash, OneMoney, \
InnBucks style wallets, USD and ZiG currencies).

The system scores each transaction with a machine learning classifier (Random Forest by default, \
with Logistic Regression and Decision Tree also trained), assigns a Low/Medium/High risk level, raises \
alerts for analysts, screens names against sanctions lists, and keeps an audit trail.

You help compliance analysts and administrators investigate activity, prioritise alerts, explain why \
transactions were flagged, and draw conclusions and recommendations.

How to work:
- Use the tools to get facts before answering anything about the data. Never invent transaction IDs, \
names, amounts or counts. If a tool returns nothing relevant, say so.
- Quote specific evidence: transaction IDs, amounts with currency, risk scores, reason codes.
- Reason with AML judgement: velocity, structuring (many small transfers), unfamiliar recipients, \
new devices, new accounts, unusual hours, cash-out patterns, sanctions hits.
- When asked for conclusions or a summary, give clear findings, what they mean, and concrete next \
steps. Be explicit about limitations: the dataset is simulated, and risk scores are indicators for \
human review, not proof of fraud.
- You are read-only: you cannot change alerts or transactions. Tell the user which page to use \
(Investigations, Transactions, Sanctions, Settings) if they want to act.
- Write clearly for a busy analyst. Use short paragraphs, and bullet points or a small table only \
when they make things easier to scan."""


def _tx_brief(t):
    return {
        'transaction_id': t.transaction_id,
        'timestamp': t.timestamp.isoformat(timespec='minutes') if t.timestamp else None,
        'sender': f"{t.sender_name or 'Unknown'} ({t.sender_wallet})",
        'recipient': f"{t.recipient_name or 'Unknown'} ({t.recipient_wallet})",
        'amount': round(t.amount, 2),
        'currency': t.currency,
        'payment_type': t.payment_type,
        'risk_level': t.risk_level,
        'risk_score': round(t.risk_score, 3),
        'status': t.status,
        'alert_reasons': t.alert.reason_codes if t.alert else None,
    }


def _clamp(limit, default=20, hi=50):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(limit, hi))


@beta_tool
def get_system_overview() -> str:
    """Get headline statistics for the whole system: transaction counts by risk level and status,
    alert counts by status, totals by currency, the date range covered, and the most common alert
    reason codes. Call this first for any broad question or summary."""
    total = Transaction.query.count()
    by_risk = dict(db.session.query(Transaction.risk_level, func.count()).group_by(Transaction.risk_level).all())
    by_status = dict(db.session.query(Transaction.status, func.count()).group_by(Transaction.status).all())
    alerts_by_status = dict(db.session.query(Alert.status, func.count()).group_by(Alert.status).all())
    by_currency = {c: {'count': n, 'total_amount': round(s or 0, 2)} for c, n, s in
                   db.session.query(Transaction.currency, func.count(), func.sum(Transaction.amount))
                   .group_by(Transaction.currency).all()}
    first, last = db.session.query(func.min(Transaction.timestamp), func.max(Transaction.timestamp)).one()
    reasons = Counter()
    for (codes,) in db.session.query(Alert.reason_codes).all():
        reasons.update(c.strip() for c in (codes or '').split(',') if c.strip())
    return json.dumps({
        'total_transactions': total,
        'by_risk_level': by_risk,
        'by_transaction_status': by_status,
        'alerts_by_status': alerts_by_status,
        'total_alerts': sum(alerts_by_status.values()),
        'by_currency': by_currency,
        'date_range': [first.isoformat(timespec='minutes') if first else None,
                       last.isoformat(timespec='minutes') if last else None],
        'top_alert_reason_codes': reasons.most_common(10),
    })


@beta_tool
def search_transactions(risk_level: str = "", status: str = "", wallet_or_name: str = "",
                        min_amount: float = 0, payment_type: str = "", currency: str = "",
                        limit: int = 20) -> str:
    """Search transactions, highest risk score first.

    Args:
        risk_level: Optional filter: Low, Medium or High.
        status: Optional filter: Legitimate, Suspicious or Under Investigation.
        wallet_or_name: Optional text matched against sender/recipient wallet or name.
        min_amount: Optional minimum amount.
        payment_type: Optional filter: P2P, Cash-Out or Merchant.
        currency: Optional filter: USD or ZiG.
        limit: Maximum rows to return (1-50, default 20).
    """
    q = Transaction.query
    if risk_level:
        q = q.filter(func.lower(Transaction.risk_level) == risk_level.lower())
    if status:
        q = q.filter(func.lower(Transaction.status) == status.lower())
    if payment_type:
        q = q.filter(func.lower(Transaction.payment_type) == payment_type.lower())
    if currency:
        q = q.filter(func.lower(Transaction.currency) == currency.lower())
    if min_amount:
        q = q.filter(Transaction.amount >= float(min_amount))
    if wallet_or_name:
        like = f"%{wallet_or_name}%"
        q = q.filter(or_(Transaction.sender_wallet.ilike(like), Transaction.recipient_wallet.ilike(like),
                         Transaction.sender_name.ilike(like), Transaction.recipient_name.ilike(like)))
    matched = q.count()
    rows = q.order_by(Transaction.risk_score.desc(), Transaction.timestamp.desc()).limit(_clamp(limit)).all()
    return json.dumps({'matched': matched, 'returned': len(rows), 'transactions': [_tx_brief(t) for t in rows]})


@beta_tool
def get_transaction_details(transaction_id: str) -> str:
    """Get every field of one transaction, including behavioural features, the model's own
    explanation of its risk score (how many probability points each feature added), and its
    alert (reasons, status, assignee, analyst notes). Use this to explain why something was flagged.

    Args:
        transaction_id: The exact transaction ID, e.g. TXN-000123.
    """
    t = Transaction.query.filter_by(transaction_id=transaction_id.strip()).first()
    if not t:
        return json.dumps({'error': f'No transaction with ID {transaction_id}'})
    d = _tx_brief(t)
    d.update({
        'hour_of_day': t.hour_of_day,
        'velocity_1h': t.velocity_1h,
        'daily_volume': round(t.daily_volume, 2),
        'unfamiliar_recipient': t.is_unfamiliar,
        'new_device': t.is_new_device,
        'account_age_days': t.account_age_days,
    })
    model = model_getter()
    if model is not None:
        import explain
        e = explain.explain(model, explain.transaction_features(t))
        d['model_explanation'] = {
            'baseline_probability': e['base'], 'final_probability': e['score'],
            'factors': [{'factor': f['label'], 'value': f['value'],
                         'probability_points': round(f['contribution'] * 100, 1)}
                        for f in e['factors'] if abs(f['contribution']) >= 0.005],
            'note': 'Contributions taken from the active model\'s decision paths for this transaction.'}
    if t.alert:
        a = t.alert
        d['alert'] = {'alert_id': a.id, 'status': a.status, 'assigned_to': a.assigned_to,
                      'reason_codes': a.reason_codes, 'resolution_notes': a.resolution_notes,
                      'created_at': a.created_at.isoformat(timespec='minutes') if a.created_at else None}
    return json.dumps(d)


@beta_tool
def list_alerts(status: str = "", risk_level: str = "", limit: int = 20) -> str:
    """List alerts raised by the detection engine, highest risk first.

    Args:
        status: Optional filter: Pending, Under Review or Closed.
        risk_level: Optional filter: Medium or High.
        limit: Maximum rows to return (1-50, default 20).
    """
    q = Alert.query
    if status:
        q = q.filter(func.lower(Alert.status) == status.lower())
    if risk_level:
        q = q.filter(func.lower(Alert.risk_level) == risk_level.lower())
    matched = q.count()
    rows = q.order_by(Alert.risk_score.desc(), Alert.created_at.desc()).limit(_clamp(limit)).all()
    out = []
    for a in rows:
        t = a.transaction
        out.append({'alert_id': a.id, 'transaction_id': a.transaction_id, 'status': a.status,
                    'risk_level': a.risk_level, 'risk_score': round(a.risk_score, 3),
                    'reason_codes': a.reason_codes, 'assigned_to': a.assigned_to,
                    'amount': f"{t.amount:,.2f} {t.currency}" if t else None,
                    'sender': t.sender_name if t else None, 'recipient': t.recipient_name if t else None})
    return json.dumps({'matched': matched, 'returned': len(out), 'alerts': out})


@beta_tool
def get_wallet_profile(wallet_or_name: str) -> str:
    """Profile one customer wallet: money sent and received, number of high-risk and alerted
    transactions, counterparties, and recent activity. Use it to judge whether an account looks
    compromised or like a mule.

    Args:
        wallet_or_name: Wallet ID or customer name (partial match allowed).
    """
    like = f"%{wallet_or_name}%"
    sent = Transaction.query.filter(or_(Transaction.sender_wallet.ilike(like), Transaction.sender_name.ilike(like))).all()
    recv = Transaction.query.filter(or_(Transaction.recipient_wallet.ilike(like), Transaction.recipient_name.ilike(like))).all()
    if not sent and not recv:
        return json.dumps({'error': f'No wallet or customer matching {wallet_or_name}'})
    every = {t.transaction_id: t for t in sent + recv}.values()
    recent = sorted(every, key=lambda t: t.timestamp, reverse=True)[:10]
    return json.dumps({
        'transactions_sent': len(sent),
        'transactions_received': len(recv),
        'amount_sent_by_currency': {c: round(sum(t.amount for t in sent if t.currency == c), 2) for c in {t.currency for t in sent}},
        'amount_received_by_currency': {c: round(sum(t.amount for t in recv if t.currency == c), 2) for c in {t.currency for t in recv}},
        'high_risk_transactions': sum(1 for t in every if t.risk_level == 'High'),
        'alerted_transactions': sum(1 for t in every if t.alert),
        'distinct_recipients': len({t.recipient_wallet for t in sent}),
        'distinct_senders': len({t.sender_wallet for t in recv}),
        'recent_transactions': [_tx_brief(t) for t in recent],
    })


@beta_tool
def get_risk_breakdown(group_by: str) -> str:
    """Break transactions down to find patterns: counts, high-risk counts and high-risk rate per group.

    Args:
        group_by: One of payment_type, currency, hour_of_day, reason_code.
    """
    if group_by == 'reason_code':
        c = Counter()
        for (codes,) in db.session.query(Alert.reason_codes).all():
            c.update(x.strip() for x in (codes or '').split(',') if x.strip())
        return json.dumps({'alerts_containing_reason': dict(c.most_common())})
    column = {'payment_type': Transaction.payment_type, 'currency': Transaction.currency,
              'hour_of_day': Transaction.hour_of_day}.get(group_by)
    if column is None:
        return json.dumps({'error': 'group_by must be payment_type, currency, hour_of_day or reason_code'})
    rows = db.session.query(column, func.count(),
                            func.sum(case((Transaction.risk_level == 'High', 1), else_=0)),
                            func.sum(Transaction.amount)).group_by(column).order_by(column).all()
    return json.dumps({'group_by': group_by, 'groups': [
        {'group': g, 'transactions': n, 'high_risk': int(h or 0),
         'high_risk_rate': round((h or 0) / n, 3) if n else 0, 'total_amount': round(s or 0, 2)}
        for g, n, h, s in rows]})


@beta_tool
def get_model_performance() -> str:
    """Get the evaluation metrics (accuracy, precision, recall, F1, confusion matrix) of the three
    trained classifiers, and which one is currently active."""
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except (OSError, ValueError) as e:
        return json.dumps({'error': f'Could not read metrics: {e}'})
    from flask import current_app
    return json.dumps({'active_model': active_model_getter(),
                       'thresholds': {'medium': current_app.config.get('MEDIUM_RISK_THRESHOLD'),
                                      'high': current_app.config.get('HIGH_RISK_THRESHOLD')},
                       'metrics': metrics,
                       'note': 'Metrics come from a held-out test split of the simulated dataset.'})


@beta_tool
def get_recent_audit_logs(limit: int = 20) -> str:
    """Get the most recent audit trail entries (logins, configuration changes, alert updates).

    Args:
        limit: Maximum rows to return (1-50, default 20).
    """
    rows = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(_clamp(limit)).all()
    return json.dumps([{'time': r.timestamp.isoformat(timespec='minutes'), 'user': r.username,
                        'event': r.event, 'ip': r.ip_address} for r in rows])


# Set by app.py so the assistant can report the classifier currently in use.
active_model_getter = lambda: None
model_getter = lambda: None

TOOLS = [get_system_overview, search_transactions, get_transaction_details, list_alerts,
         get_wallet_profile, get_risk_breakdown, get_model_performance, get_recent_audit_logs]

# Conversation history kept server-side so assistant turns (including their thinking and tool
# blocks) are sent back to the API exactly as returned. In-memory: fine for a single-process
# prototype; conversations reset when the server restarts.
_CONVERSATIONS = {}
_MAX_CONVERSATIONS = 200


def is_configured():
    return bool(os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN'))


def chat(user_message, conversation_id=None):
    """Run one assistant turn. Returns dict(response, conversation_id, tools_used, error)."""
    if not conversation_id or conversation_id not in _CONVERSATIONS:
        if len(_CONVERSATIONS) >= _MAX_CONVERSATIONS:
            _CONVERSATIONS.pop(next(iter(_CONVERSATIONS)))
        conversation_id = uuid.uuid4().hex
        _CONVERSATIONS[conversation_id] = []
    history = _CONVERSATIONS[conversation_id]
    working = history + [{'role': 'user', 'content': user_message}]

    client = anthropic.Anthropic()
    tools_used = []
    try:
        runner = client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=working,
            output_config={'effort': 'medium'},
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
            max_iterations=12,
        )
        last = None
        for message in runner:
            last = message
            working.append({'role': 'assistant', 'content': message.content})
            tools_used += [b.name for b in message.content if b.type == 'tool_use']
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                working.append(tool_response)
    except anthropic.AuthenticationError:
        return {'conversation_id': conversation_id, 'tools_used': [],
                'error': 'The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.'}
    except anthropic.RateLimitError:
        return {'conversation_id': conversation_id, 'tools_used': [],
                'error': 'The AI service is rate-limiting requests. Wait a moment and try again.'}
    except anthropic.APIConnectionError:
        return {'conversation_id': conversation_id, 'tools_used': [],
                'error': 'Could not reach the AI service. Check the internet connection.'}
    except anthropic.APIStatusError as e:
        return {'conversation_id': conversation_id, 'tools_used': [],
                'error': f'The AI service returned an error ({e.status_code}): {e.message}'}

    if last is None:
        return {'conversation_id': conversation_id, 'tools_used': tools_used, 'error': 'No response from the AI service.'}
    if last.stop_reason == 'refusal':
        text = "I can't help with that request."
    else:
        text = '\n\n'.join(b.text for b in last.content if b.type == 'text').strip()
        if last.stop_reason == 'max_tokens':
            text += '\n\n(Response cut short; ask me to continue.)'
    # Commit the completed turn only after it succeeds, so a failed turn leaves history intact.
    _CONVERSATIONS[conversation_id] = working
    return {'response': text, 'conversation_id': conversation_id,
            'tools_used': list(dict.fromkeys(tools_used)), 'model': last.model}


def reset(conversation_id):
    _CONVERSATIONS.pop(conversation_id, None)


def complete(prompt, system=SYSTEM_PROMPT):
    """Single short completion (no tools) for the alert and executive summaries."""
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
        output_config={'effort': 'low'},
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',
    )
    if response.stop_reason == 'refusal':
        return None
    return '\n\n'.join(b.text for b in response.content if b.type == 'text').strip() or None
