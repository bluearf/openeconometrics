"""Owner-only authentication for an explicitly configured Google IAP deployment.

IAP must also be enabled on the Cloud Run service. Its signed assertion is
checked independently here; unsigned identity headers never grant access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import math
import re
from typing import Any
from urllib.parse import urlsplit


IAP_ISSUER = "https://cloud.google.com/iap"
IAP_JWKS_URL = "https://www.gstatic.com/iap/verify/public_key-jwk"
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_AUDIENCE = re.compile(
    r"/projects/[1-9][0-9]{0,19}/locations/[a-z]+(?:-[a-z]+)+[0-9]+"
    r"/services/[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?\Z"
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\Z")


class CloudAuthError(ValueError):
    """An IAP assertion is missing, invalid, or belongs to another user."""


def _validate_origin(value: str) -> None:
    if not isinstance(value, str) or not value.isascii():
        raise ValueError("Cloud public_origin must be a canonical HTTPS origin.")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        valid = (
            parsed.scheme == "https"
            and value == f"https://{host}"
            and 0 < len(host) <= 253
            and len(host.split(".")) >= 2
            and all(_DNS_LABEL.fullmatch(label) for label in host.split("."))
        )
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            valid = False
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Cloud public_origin must be a canonical HTTPS DNS origin without "
            "a path, port, credentials, query, or fragment."
        )


@dataclass(frozen=True)
class CloudAccess:
    """Validated deployment identity and a bounded, cached IAP key resolver."""

    public_origin: str
    audience: str
    owner_email: str
    _jwks: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_origin(self.public_origin)
        if not isinstance(self.audience, str) or not _AUDIENCE.fullmatch(self.audience):
            raise ValueError("Cloud audience must identify one Cloud Run service.")
        if (
            not isinstance(self.owner_email, str)
            or len(self.owner_email) > 254
            or not _EMAIL.fullmatch(self.owner_email)
            or any(not label or label.startswith("-") or label.endswith("-")
                   for label in self.owner_email.rsplit("@", 1)[-1].split("."))
        ):
            raise ValueError("Cloud owner_email must contain exactly one email address.")
        # No key fetch occurs until verify(); local-only users do not need the
        # cloud extra because this import only runs for a configured instance.
        from jwt import PyJWKClient

        object.__setattr__(self, "_jwks", PyJWKClient(
            IAP_JWKS_URL, cache_jwk_set=True, lifespan=300,
            cache_keys=False, timeout=5,
        ))

    def verify(self, assertion: str) -> dict[str, Any]:
        """Return verified owner claims, failing closed without logging tokens."""
        if (
            not isinstance(assertion, str)
            or not assertion
            or len(assertion) > 16384
            or any(character.isspace() for character in assertion)
        ):
            raise CloudAuthError("A valid Google IAP assertion is required.")
        import jwt

        try:
            header = jwt.get_unverified_header(assertion)
            if (
                header.get("alg") != "ES256"
                or not isinstance(header.get("kid"), str)
                or not 1 <= len(header["kid"]) <= 256
            ):
                raise CloudAuthError("The Google IAP assertion is invalid.")
            key = self._jwks.get_signing_key_from_jwt(assertion)
            claims = jwt.decode(
                assertion, key.key, algorithms=["ES256"], issuer=IAP_ISSUER,
                audience=self.audience, leeway=30,
                options={
                    "require": ["exp", "iat", "sub", "email", "iss", "aud"],
                    "strict_aud": True,
                },
            )
            issued, expires = claims["iat"], claims["exp"]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (issued, expires)
            ) or not 0 < expires - issued <= 660:
                raise CloudAuthError("The Google IAP assertion is invalid.")
            subject = claims["sub"]
            if not isinstance(subject, str) or not subject.strip() or subject != subject.strip():
                raise CloudAuthError("The Google IAP assertion is invalid.")
            if claims["email"] != self.owner_email:
                raise CloudAuthError("This workspace is restricted to its owner.")
            return claims
        except CloudAuthError:
            raise
        except (jwt.PyJWTError, OSError, ValueError, TypeError, OverflowError) as exc:
            raise CloudAuthError("The Google IAP assertion could not be verified.") from exc
