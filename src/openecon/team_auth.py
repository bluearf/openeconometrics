"""Firebase authentication for the separate multi-project control API.

Authentication establishes a UID. Project membership and account enablement are
separate, live authorization checks; neither email nor token claims confer roles.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
import re
import threading
from typing import Any

_PROJECT_ID = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]\Z")
_UID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_EMAIL = re.compile(r"[^@\s<>\x00-\x1f\x7f]+@[^@\s<>\x00-\x1f\x7f]+\.[^@\s<>\x00-\x1f\x7f]+\Z")
_APP_LOCK = threading.Lock()


class TeamAuthError(ValueError):
    """Safe API error with no token, provider response, or credential details."""

    def __init__(self, code: str, message: str, status_code: int = 401):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class TeamIdentity:
    uid: str
    email: str
    name: str = ""
    email_verified: bool = False


class TeamAuth:
    """Verify Firebase ID tokens against one explicitly configured project.

    ``verifier`` is a trusted dependency-injection boundary for isolated tests.
    Production uses Firebase Admin, including revocation/disabled-user checks.
    The public account-status endpoint may use ``require_verified=False``;
    project resources must retain the default and check membership separately.
    """

    def __init__(self, project_id: str, *, verifier: Callable[[str], Mapping[str, Any]] | None = None):
        if not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id):
            raise ValueError("A canonical Firebase project ID is required.")
        if verifier is not None and not callable(verifier):
            raise TypeError("verifier must be callable.")
        self.project_id = project_id
        self._verifier = verifier or self._firebase_verify

    def _firebase_verify(self, token: str) -> Mapping[str, Any]:
        # The Admin SDK deliberately skips signature checks in emulator mode.
        # That development switch must never weaken the public control API.
        if os.environ.get("FIREBASE_AUTH_EMULATOR_HOST"):
            raise TeamAuthError("AUTH_UNAVAILABLE", "Sign-in verification is unavailable.", 503)
        try:
            import firebase_admin
            from firebase_admin import auth

            app_name = f"openecon-team-{self.project_id}"
            try:
                with _APP_LOCK:
                    try:
                        app = firebase_admin.get_app(app_name)
                    except ValueError:
                        app = firebase_admin.initialize_app(
                            options={"projectId": self.project_id, "httpTimeout": 10},
                            name=app_name,
                        )
                    if app.project_id != self.project_id:
                        raise ValueError("The Firebase app belongs to a different project.")
            except Exception:
                raise TeamAuthError(
                    "AUTH_UNAVAILABLE", "Sign-in verification is unavailable.", 503
                ) from None
            try:
                return auth.verify_id_token(token, app=app, check_revoked=True)
            except (ValueError, auth.InvalidIdTokenError, auth.UserDisabledError,
                    auth.TenantIdMismatchError):
                raise TeamAuthError(
                    "INVALID_TOKEN", "The sign-in token is invalid. Sign in again."
                ) from None
            except Exception:
                raise TeamAuthError(
                    "AUTH_UNAVAILABLE", "Sign-in verification is temporarily unavailable.", 503
                ) from None
        except (ImportError, OSError) as exc:
            raise TeamAuthError(
                "AUTH_UNAVAILABLE", "Sign-in verification is temporarily unavailable.", 503
            ) from exc

    def verify(self, id_token: str, *, require_verified: bool = True) -> TeamIdentity:
        if not isinstance(id_token, str) or not id_token:
            raise TeamAuthError("AUTH_REQUIRED", "Sign in to continue.")
        if len(id_token) > 16384 or any(character.isspace() for character in id_token):
            raise TeamAuthError("INVALID_TOKEN", "The sign-in token is invalid. Sign in again.")
        try:
            claims = self._verifier(id_token)
        except TeamAuthError:
            raise
        except Exception:
            # Revoked, expired, disabled, malformed and otherwise invalid tokens
            # share a stable response; provider details must not leak to clients.
            raise TeamAuthError("INVALID_TOKEN", "The sign-in token is invalid. Sign in again.") from None
        if not isinstance(claims, Mapping):
            raise TeamAuthError("INVALID_TOKEN", "The sign-in token is invalid. Sign in again.")
        uid, email = claims.get("uid"), claims.get("email")
        firebase = claims.get("firebase")
        valid = (
            claims.get("aud") == self.project_id
            and claims.get("iss") == f"https://securetoken.google.com/{self.project_id}"
            and isinstance(uid, str) and _UID.fullmatch(uid)
            and claims.get("sub") == uid
            and isinstance(email, str) and len(email) <= 254 and _EMAIL.fullmatch(email)
            and isinstance(firebase, Mapping)
            and isinstance(firebase.get("sign_in_provider"), str)
            and (firebase.get("sign_in_provider") in {"password", "google.com"}
                 or (firebase.get("sign_in_provider") == "custom" and claims.get("openecon_desktop") is True))
            and not firebase.get("tenant")
            and isinstance(claims.get("email_verified"), bool)
        )
        if not valid:
            raise TeamAuthError("INVALID_TOKEN", "The sign-in token is invalid. Sign in again.")
        verified = claims["email_verified"]
        if require_verified and not verified:
            raise TeamAuthError(
                "EMAIL_VERIFICATION_REQUIRED", "Verify your email address before opening projects.", 403
            )
        name = claims.get("name", "")
        return TeamIdentity(uid=uid, email=email.lower(),
                            name=name.strip()[:200] if isinstance(name, str) else "",
                            email_verified=verified)
