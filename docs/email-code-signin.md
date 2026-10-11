# Google and email-code sign-in

The sign-in screen has two choices: **Continue with Google**, or enter an email
address and choose **Send sign-in code**. Google uses Firebase's verified Google
identity. Email sign-in requires the six-digit code delivered to that address on
each new sign-in. There is no registration password, password reset, or account
linking step. A first successful email-code verification creates the identity;
an existing verified Google/email identity keeps its Firebase UID and projects.
Existing Google, app and email-code sessions continue until sign-out or
revocation. Legacy password-provider sessions must sign in again with Google or
a new email code; the retained UID and project memberships stay unchanged.
Local projects remain available without an account or internet connection.

Email and code entry stay in the desktop application. The native broker sends
only the explicit email/challenge or code/verifier to the fixed team API; it
cannot accept a caller-selected UID, password, token or redirect in these bodies.
Google continues to use the system browser for Google authentication and the
existing app-bound approval exchange. The browser sign-in screen offers the
same two methods. Analyses still run locally.

## Delivery configuration and rollout

Firebase's built-in passwordless email method sends an email **link**, rather
than the six-digit code used here. This implementation retains Firebase for
identity/session verification and uses an explicitly configured mail transport
for code delivery. See [Firebase's email-link contract](https://firebase.google.com/docs/auth/web/email-link-auth)
and [custom authentication](https://firebase.google.com/docs/auth/admin/create-custom-tokens).

Configure both settings on the team control service, using secret-backed service
configuration. Never put these values in source, a browser bundle, receipts or
chat messages:

- `OPENECON_EMAIL_CODE_KEY`: a freshly generated 32-byte private key encoded as
  64 lowercase hexadecimal characters. It must be shared across service replicas.
- `OPENECON_EMAIL_CODE_SMTP`: a JSON object containing only `host`, `port`,
  `username`, `password`, and `sender`. Use an authorized transactional sender.
  Port 465 uses TLS from connection establishment; port 587 requires STARTTLS
  with normal certificate validation. Plaintext SMTP is unsupported.

The runtime service identity needs `firebaseauth.users.get`,
`firebaseauth.users.create` and `firebaseauth.users.update` for identity lookup,
first-account creation, verification and refresh-session revocation. Use a
custom role with these permissions rather than granting user deletion or
authentication-configuration access. Grant Secret Manager access only on the
two delivery secrets, and update both secret-backed environment settings
together. Do not set just the key or just the SMTP configuration.

The deployment helper can attach the fixed Secret Manager resources
`openecon-email-code-key` and `openecon-email-code-smtp` in the same revision:
pass both `--email-code-key-version <number>` and
`--email-code-smtp-version <number>` to `scripts/deploy_team_control.py`.
Only pinned positive version numbers are accepted; neither raw credentials nor
`latest` are supported. Review the default read-only preview plan, then use
`--apply` for preview acceptance before `--production --apply`. Without these
options, the existing service environment remains authoritative.

No code is mailed at application startup. An unconfigured sender advertises
`email_code_available=false`; requests return 503 and the UI does not claim a
code was sent. A transport failure keeps the user on the email-entry screen.
SMTP acceptance is not proof of inbox delivery. Before production rollout,
verify receipt and code entry with the real sender, including a native sign-in,
Google sign-in, same-UID project access, and a sign-out/new-code sign-in. Keep the
previous service revision active until these live checks can be completed.

Configure Firestore TTL on `oe_email_codes.expires_at`. Rate-limit documents in
`oe_limits` carry an `expires_at` for this flow too, but enabling TTL on that
shared collection requires a separate retention review. TTL is cleanup only: the
service checks expiry and attempts on every verification, without relying on
deletion timing. Existing non-email limit documents do not acquire a TTL.

## Verification boundaries

Codes expire after ten minutes and can be consumed once. Resending invalidates
the previous challenge, with a 60-second cooldown, five sends per email per hour,
and 512 total sends per service per UTC day. Five wrong-code/client-proof attempts
invalidate the challenge. Codes use a cryptographic random source; only a keyed
HMAC is stored. A separate random client proof remains in the initiating screen's
memory, and is discarded on cancellation/exit. Codes do not appear in URLs,
responses, application logs or persisted browser preferences.

Consumption is transactional and precedes identity resolution/token signing;
concurrent requests cannot reuse the code. Failure after consumption requires a
new code. New and existing emails receive the same challenge response, and
account lookup happens only after successful verification. Disabled identities
are never reenabled. Previously unverified identities have old refresh sessions
revoked before their email is marked verified. Production rejects password
provider ID tokens; Google, verified email-code custom tokens and existing
desktop custom sessions retain Firebase revocation/disabled-user checks.

Project access remains a separate live UID-based membership check. Creating or
verifying an identity never grants or copies memberships. No account is merged
by a browser-supplied email/UID claim. Legacy account-link endpoints return 410
in the production passwordless mode; historical fixtures remain for older API
contracts and do not expose an account-linking screen in the application.
