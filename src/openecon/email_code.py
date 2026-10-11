"""Passwordless email challenges for the team control plane.

Only a delivered, unexpired code plus the initiating client's proof can mint a
Firebase session. Codes and client verifiers never enter saved documents/logs.
The mail transport is explicit; an unconfigured service cannot claim delivery.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import ssl
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openecon.team_auth import TeamAuthError, _EMAIL
from openecon.team_store import TeamError

PREFIX = '/api/auth/email-code'
LIFETIME = 600
COOLDOWN = 60
MAX_ATTEMPTS = 5


class EmailCodeBegin(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    email: str = Field(min_length=3, max_length=254)
    challenge: str = Field(pattern=r'^[0-9a-f]{64}$')

    @field_validator('email')
    @classmethod
    def normalized_email(cls, value):
        value = value.strip().lower()
        if not value.isascii() or not _EMAIL.fullmatch(value):
            raise ValueError('Enter a valid email address.')
        return value


class EmailCodeVerify(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    code: str = Field(pattern=r'^[0-9]{6}$')
    verifier: str = Field(pattern=r'^[0-9a-f]{64}$')


def public_email_code(path, method):
    return method == 'POST' and (path == PREFIX or bool(
        re.fullmatch(re.escape(PREFIX) + r'/[0-9a-f]{32}/verify', path)))


def invalid_code():
    return TeamError('EMAIL_CODE_INVALID', 'The code is incorrect or has expired. Request a new code.', 401)


class EmailCodes:
    def __init__(self, db, *, secret, send, identity, issue, clock=None):
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError('Email codes require a private key of at least 32 bytes.')
        self.db, self.secret = db, secret
        self.send, self.identity, self.issue = send, identity, issue
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def digest(self, purpose, value):
        return hmac.new(self.secret, (purpose + ':' + value).encode(), hashlib.sha256).hexdigest()

    def begin(self, email, challenge):
        instant = self.clock()
        request_id = uuid4().hex
        code = f'{secrets.randbelow(1000000):06d}'
        mail_hash = self.digest('email', email)
        quota_path = 'oe_limits/email-code-' + mail_hash
        path = 'oe_email_codes/' + request_id
        expires = instant + timedelta(seconds=LIFETIME)

        def reserve(db):
            quota = db.get(quota_path) or {}
            global_path = f'oe_limits/email-code-global-{instant:%Y%m%d}'
            global_quota = db.get(global_path) or {'count': 0}
            if quota.get('retry_at', instant) > instant:
                raise TeamError('EMAIL_CODE_LIMIT', 'Wait one minute before requesting another code.', 429)
            if quota.get('window_end', instant) <= instant:
                quota.update(count=0, window_end=instant + timedelta(hours=1))
            if quota['count'] >= 5 or global_quota['count'] >= 512:
                raise TeamError('EMAIL_CODE_LIMIT', 'Too many code requests. Try again later.', 429)
            # A resend invalidates the previous challenge, including its code.
            previous = quota.get('request_id')
            if previous:
                db.delete('oe_email_codes/' + previous)
            quota.update(count=quota['count'] + 1, retry_at=instant + timedelta(seconds=COOLDOWN),
                         request_id=request_id, expires_at=instant + timedelta(days=1))
            global_quota.update(count=global_quota['count'] + 1,
                                expires_at=instant + timedelta(days=2))
            db.put(quota_path, quota)
            db.put(global_path, global_quota)
            db.put(path, {'email': email, 'challenge': challenge, 'expires_at': expires,
                          'digest': self.digest('code', request_id + ':' + code),
                          'attempts': 0, 'delivered': False})

        self.db.atomic(reserve)
        try:
            self.send(email, code)
        except Exception:
            self.db.atomic(lambda db: db.delete(path))
            raise TeamError('EMAIL_DELIVERY_UNAVAILABLE',
                            'The sign-in email could not be sent. Try again later.', 503) from None

        def delivered(db):
            item = db.get(path)
            if item:
                item['delivered'] = True
                db.put(path, item)
        self.db.atomic(delivered)
        return {'request_id': request_id, 'expires_at': expires.isoformat(), 'retry_after': COOLDOWN}

    def verify(self, request_id, code, verifier):
        instant = self.clock()
        proof = hashlib.sha256(verifier.encode()).hexdigest()
        path = 'oe_email_codes/' + request_id

        def consume(db):
            item = db.get(path)
            if (not item or item['expires_at'] <= instant or not item['delivered']
                    or item['attempts'] >= MAX_ATTEMPTS):
                return None
            # Wrong codes/proofs burn an attempt. Return instead of raising so
            # the transaction persists attempts even when authentication fails.
            valid = (hmac.compare_digest(item['challenge'], proof)
                     and hmac.compare_digest(item['digest'], self.digest('code', request_id + ':' + code)))
            if not valid:
                item['attempts'] += 1
                db.put(path, item)
                return None
            db.delete(path)
            return item['email']

        email = self.db.atomic(consume)
        if email is None:
            raise invalid_code()
        # Consume before any external call: retries/concurrent callers cannot
        # reuse the code, including after an identity or signer failure.
        uid = self.identity(email)
        return {'custom_token': self.issue(uid)}


class SmtpCodeDelivery:
    def __init__(self, config):
        required = {'host', 'port', 'username', 'password', 'sender'}
        if (not isinstance(config, dict) or set(config) != required
                or type(config['port']) is not int or config['port'] not in {465, 587}
                or any(not isinstance(config[k], str) or not config[k]
                       or any(ord(c) < 32 for c in config[k]) for k in required - {'port'})
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]+', config['host'])
                or not _EMAIL.fullmatch(config['sender'])):
            raise ValueError('Email delivery requires an explicit authenticated TLS SMTP configuration.')
        self.config = config

    def __call__(self, email, code):
        message = EmailMessage()
        message['From'] = self.config['sender']
        message['To'] = email
        message['Subject'] = 'Your OpenEconometrics sign-in code'
        message.set_content(f'Your OpenEconometrics sign-in code is: {code}\n\n'
                            'This code expires in 10 minutes and can be used once.\n'
                            'If you did not request it, ignore this email.\n')
        context = ssl.create_default_context()
        cls = smtplib.SMTP_SSL if self.config['port'] == 465 else smtplib.SMTP
        options = {'context': context} if self.config['port'] == 465 else {}
        with cls(self.config['host'], self.config['port'], timeout=15, **options) as smtp:
            if self.config['port'] == 587:
                smtp.starttls(context=context)
            smtp.login(self.config['username'], self.config['password'])
            if smtp.send_message(message):
                raise RuntimeError('Email delivery was rejected.')


def firebase_email_identity(project, email):
    """Resolve the existing UID; membership is never copied or granted here."""
    from firebase_admin import auth, get_app, initialize_app
    from openecon.team_auth import _APP_LOCK
    if os.environ.get('FIREBASE_AUTH_EMULATOR_HOST'):
        raise TeamAuthError('AUTH_UNAVAILABLE', 'Email sign-in is temporarily unavailable.', 503)
    with _APP_LOCK:
        try:
            app = get_app('openecon-team-' + project)
        except ValueError:
            app = initialize_app(options={'projectId': project, 'httpTimeout': 10},
                                 name='openecon-team-' + project)
        if app.project_id != project:
            raise TeamAuthError('AUTH_UNAVAILABLE', 'Email sign-in is temporarily unavailable.', 503)
    try:
        try:
            user = auth.get_user_by_email(email, app=app)
        except auth.UserNotFoundError:
            try:
                user = auth.create_user(email=email, email_verified=True, app=app)
            except auth.EmailAlreadyExistsError:
                user = auth.get_user_by_email(email, app=app)
        if user.disabled:
            raise invalid_code()
        if not user.email_verified:
            # Production rejects all password-provider tokens. Revoke any old
            # sessions before verifying a previously unverified email identity.
            auth.revoke_refresh_tokens(user.uid, app=app)
            user = auth.update_user(user.uid, email_verified=True, app=app)
        return user.uid
    except TeamError:
        raise
    except Exception:
        raise TeamAuthError('AUTH_UNAVAILABLE', 'Email sign-in is temporarily unavailable.', 503) from None


def configured_email_codes(store, storage, project):
    smtp, secret = os.environ.get('OPENECON_EMAIL_CODE_SMTP'), os.environ.get('OPENECON_EMAIL_CODE_KEY')
    if not smtp and not secret:
        return None
    if not smtp or not secret or not re.fullmatch(r'[0-9a-f]{64}', secret):
        raise ValueError('Email-code delivery and its private key must be configured together.')
    from openecon.desktop_cloud import firebase_custom_token
    return EmailCodes(store.db, secret=bytes.fromhex(secret), send=SmtpCodeDelivery(json.loads(smtp)),
                      identity=lambda email: firebase_email_identity(project, email),
                      issue=lambda uid: firebase_custom_token(storage, uid, email_code=True))


def attach_email_code_routes(app, service):
    def available():
        if service is None:
            raise TeamError('EMAIL_DELIVERY_UNAVAILABLE', 'Email sign-in is temporarily unavailable.', 503)
        return service

    @app.post(PREFIX, status_code=202)
    def begin(body: EmailCodeBegin):
        return available().begin(body.email, body.challenge)

    @app.post(PREFIX + '/{request_id}/verify')
    def verify(request_id: str, body: EmailCodeVerify):
        if not re.fullmatch(r'[0-9a-f]{32}', request_id):
            raise invalid_code()
        return available().verify(request_id, body.code, body.verifier)
