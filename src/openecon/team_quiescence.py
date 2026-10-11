"""Bound the backend writes of every admitted team request in time.

The sync service runs no background work. Each request records its admission
time before authentication. Firestore transaction attempts and Cloud Storage
writes may start only within ``SIDE_EFFECT_SECONDS`` of admission, which equals
the Cloud Run request timeout, so no request that could still answer its client
is affected. Firestore aborts any transaction ``TRANSACTION_SECONDS`` after it
begins and storage uploads have a shorter client timeout and no retries.

Authentication rejects disabled identities on every request. Once an identity is
disabled, every write admitted for it has therefore finished within
``QUIESCENCE_SECONDS``. Disposable QA cleanup relies on this published contract
and refuses services that do not advertise exactly these values.
"""
from __future__ import annotations

from contextvars import ContextVar
import time

SIDE_EFFECT_SECONDS = 360  # Cloud Run request timeout enforced by deploy_team_control.py
TRANSACTION_SECONDS = 270  # Firestore's maximum transaction lifetime
STORAGE_WRITE_SECONDS = 45  # TeamStorage.put upload timeout; uploads are never retried
CLOCK_MARGIN_SECONDS = 60
QUIESCENCE_SECONDS = (SIDE_EFFECT_SECONDS + max(TRANSACTION_SECONDS, STORAGE_WRITE_SECONDS)
                      + CLOCK_MARGIN_SECONDS)
CONTRACT = {'version': 1, 'side_effect_seconds': SIDE_EFFECT_SECONDS,
            'quiescence_seconds': QUIESCENCE_SECONDS}

_ADMITTED = ContextVar('openecon_team_request_admitted', default=None)


def admit():
    """Start one request's write window; return the token for :func:`release`."""
    return _ADMITTED.set(time.monotonic())


def release(token):
    _ADMITTED.reset(token)


def require_write_window():
    """Refuse a write attempt that starts after its request's window closed.

    Code outside a request, such as operator recovery tools, has no window.
    """
    admitted = _ADMITTED.get()
    if admitted is not None and time.monotonic() - admitted > SIDE_EFFECT_SECONDS:
        from openecon.team_store import TeamError
        raise TeamError('REQUEST_EXPIRED', 'The request took too long and was stopped before saving '
                        'further changes. Try again.', 503)
