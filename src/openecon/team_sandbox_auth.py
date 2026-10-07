"""Verify the control service's full Google ID token inside the compute broker.

Cloud Run IAM remains the outer gate. This gate also protects direct requests
from a sandbox network namespace, which need not pass through Cloud Run ingress.
It uses public Google certificates only: no ADC, metadata, or compute IAM roles.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
import re
import time

import httpx

CERTIFICATE_URL = "https://www.googleapis.com/oauth2/v1/certs"
MAX_CERTIFICATE_BYTES = 65536
MAX_TOKEN_BYTES = 8192
_ISSUERS = ("https://accounts.google.com", "accounts.google.com")


class BrokerAuthError(ValueError):
    def __init__(self, status_code: int = 401):
        self.status_code = status_code
        super().__init__("Compute authentication unavailable." if status_code == 503
                         else "Compute authentication required.")


class BrokerAuth:
    def __init__(self, audience: str, caller_email: str, caller_sub: str):
        if (not isinstance(audience, str) or not re.fullmatch(
                r"https://[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)?\.run\.app", audience)
                or not isinstance(caller_email, str) or not re.fullmatch(
                    r"openecon-control@[a-z][a-z0-9-]{4,61}[a-z0-9]\.iam\.gserviceaccount\.com",
                    caller_email)
                or not isinstance(caller_sub, str) or not re.fullmatch(r"[0-9]{10,32}", caller_sub)):
            raise ValueError("Explicit compute audience and control service identity are required.")
        self.audience, self.caller_email, self.caller_sub = audience, caller_email, caller_sub
        self._keys: dict = {}
        self._expires = 0.0
        self._refresh_after = 0.0
        self._lock = asyncio.Lock()

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("OPENECON_BROKER_AUDIENCE"),
                   os.environ.get("OPENECON_BROKER_CALLER_EMAIL"),
                   os.environ.get("OPENECON_BROKER_CALLER_SUB"))

    async def verify(self, authorization: list[str]) -> None:
        # Never trust the signature-stripped X-Serverless-Authorization token,
        # forwarded identity headers, a query argument, or a caller-chosen issuer.
        if len(authorization) != 1 or not isinstance(authorization[0], str):
            raise BrokerAuthError()
        value = authorization[0]
        if not value.startswith("Bearer "):
            raise BrokerAuthError()
        token = value[7:]
        if (not 1 <= len(token) <= MAX_TOKEN_BYTES or not re.fullmatch(
                r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token)):
            raise BrokerAuthError()
        try:
            import jwt

            header = jwt.get_unverified_header(token)
            if (set(header) - {"alg", "kid", "typ"} or header.get("alg") != "RS256"
                    or header.get("typ", "JWT") != "JWT"
                    or not isinstance(header.get("kid"), str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", header["kid"])):
                raise BrokerAuthError()
            async with asyncio.timeout(4):
                key = await self._key(header["kid"])
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=self.audience,
                                issuer=_ISSUERS, leeway=30,
                                options={"strict_aud": True, "require": [
                                    "iss", "sub", "aud", "iat", "exp", "email", "email_verified"]})
            if (claims["sub"] != self.caller_sub or claims["email"] != self.caller_email
                    or claims["email_verified"] is not True
                    or type(claims["iat"]) is not int or type(claims["exp"]) is not int
                    or not 0 < claims["exp"] - claims["iat"] <= 3900):
                raise BrokerAuthError()
        except BrokerAuthError:
            raise
        except (TimeoutError, ImportError):
            raise BrokerAuthError(503) from None
        except Exception:
            # JWT, certificate and transport diagnostics must not echo tokens.
            raise BrokerAuthError() from None

    async def _key(self, kid: str):
        now = time.monotonic()
        if now < self._expires and kid in self._keys:
            return self._keys[kid]
        async with self._lock:
            now = time.monotonic()
            if now < self._expires and kid in self._keys:
                return self._keys[kid]
            if now < self._refresh_after:
                raise BrokerAuthError(401 if now < self._expires else 503)
            # Single-flight refresh, including unknown keys, at most once per
            # minute. Never use expired cached keys after a failed refresh.
            self._refresh_after = now + 60
            try:
                keys, ttl = await self._fetch_certificates()
            except Exception:
                raise BrokerAuthError(503) from None
            self._keys, self._expires = keys, time.monotonic() + ttl
            if kid not in keys:
                raise BrokerAuthError()
            return keys[kid]

    @staticmethod
    async def _fetch_certificates() -> tuple[dict, int]:
        import json
        from cryptography import x509
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate certificate key")
                result[key] = value
            return result

        async with asyncio.timeout(3):
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False,
                                         timeout=2) as client:
                async with client.stream("GET", CERTIFICATE_URL,
                                         headers={"Accept-Encoding": "identity"}) as response:
                    if (response.status_code != 200
                            or response.headers.get("content-encoding", "identity") != "identity"):
                        raise ValueError("Certificate response unavailable")
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(raw) + len(chunk) > MAX_CERTIFICATE_BYTES:
                            raise ValueError("Certificate response too large")
                        raw.extend(chunk)
                    cache = response.headers.get("cache-control", "").lower()
                    match = re.search(r"(?:^|,)\s*max-age=([0-9]+)(?:\s*,|\s*$)", cache)
                    max_age = int(match.group(1)) if match else 300
                    age = response.headers.get("age", "0")
                    if not age.isdigit():
                        raise ValueError("Invalid certificate age")
                    ttl = min(3600, max(0, max_age - int(age)))
                    if "no-cache" in cache or "no-store" in cache:
                        ttl = 0
        certs = json.loads(raw, object_pairs_hook=pairs)
        if not isinstance(certs, dict) or not 1 <= len(certs) <= 32:
            raise ValueError("Invalid certificates")
        keys = {}
        now = datetime.now(timezone.utc)
        for kid, pem in certs.items():
            if (not isinstance(kid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", kid)
                    or not isinstance(pem, str) or len(pem) > 8192):
                raise ValueError("Invalid certificate")
            cert = x509.load_pem_x509_certificate(pem.encode("ascii"))
            key = cert.public_key()
            if (not isinstance(key, RSAPublicKey) or not 2048 <= key.key_size <= 8192
                    or not cert.not_valid_before_utc <= now < cert.not_valid_after_utc):
                raise ValueError("Invalid signing key")
            ttl = min(ttl, max(0, int((cert.not_valid_after_utc - now).total_seconds())))
            keys[kid] = key
        return keys, ttl
