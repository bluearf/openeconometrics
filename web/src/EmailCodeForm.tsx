import { useEffect, useRef, useState, type FormEvent } from "react";
import type { EmailCodeSession } from "./email-code";

export default function EmailCodeForm({
  session,
  available,
  disabled,
  onBusy,
}: {
  session: EmailCodeSession;
  available: boolean;
  disabled: boolean;
  onBusy: (busy: boolean) => void;
}) {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [retryAt, setRetryAt] = useState(0);
  const [now, setNow] = useState(Date.now());
  const pending = useRef(false);
  const generation = useRef(0);
  useEffect(
    () => () => {
      ++generation.current;
      session.cancel();
    },
    [session],
  );
  useEffect(() => {
    if (disabled) {
      ++generation.current;
      session.cancel();
      setSent(false);
      setCode("");
    }
  }, [disabled, session]);
  useEffect(() => {
    if (retryAt <= Date.now()) return;
    const timer = window.setInterval(() => {
      setNow(Date.now());
      if (Date.now() >= retryAt) window.clearInterval(timer);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [retryAt]);
  const remaining = Math.max(0, Math.ceil((retryAt - now) / 1000));
  async function perform(verify: boolean) {
    if (pending.current || disabled || !available) return;
    const ticket = ++generation.current;
    pending.current = true;
    setBusy(true);
    onBusy(true);
    setError("");
    try {
      if (verify) await session.verify(code);
      else {
        const cooldown = await session.send(email);
        if (ticket !== generation.current) return;
        setSent(true);
        setCode("");
        setNow(Date.now());
        setRetryAt(Date.now() + cooldown * 1000);
      }
    } catch (failure) {
      if (ticket === generation.current)
        setError(
          failure instanceof Error
            ? failure.message
            : "Could not complete email sign-in. Try again.",
        );
    } finally {
      if (ticket === generation.current) {
        pending.current = false;
        setBusy(false);
        onBusy(false);
      }
    }
  }
  function submit(event: FormEvent) {
    event.preventDefault();
    void perform(sent);
  }
  return (
    <>
      {!available && (
        <p role="status">
          Email sign-in is temporarily unavailable. Continue with Google or try
          again later.
        </p>
      )}
      {error && (
        <div className="team-notice is-error" role="alert">
          {error}
        </div>
      )}
      <form onSubmit={submit}>
        <label>
          Email
          <input
            type="email"
            autoComplete="email"
            required
            maxLength={254}
            disabled={busy || disabled || sent}
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </label>
        {sent && (
          <>
            <p role="status">
              Enter the six-digit code sent to {email.trim()}. It expires in 10
              minutes.
            </p>
            <label>
              Verification code
              <input
                autoComplete="one-time-code"
                inputMode="numeric"
                pattern="[0-9]{6}"
                minLength={6}
                maxLength={6}
                required
                autoFocus
                disabled={busy || disabled}
                value={code}
                onChange={(event) =>
                  setCode(event.target.value.replace(/\D/g, "").slice(0, 6))
                }
              />
            </label>
          </>
        )}
        <button
          className="team-primary"
          disabled={!available || busy || disabled}
        >
          {busy
            ? sent
              ? "Verifying…"
              : "Sending code…"
            : sent
              ? "Verify and sign in"
              : "Send sign-in code"}
        </button>
      </form>
      {sent && (
        <div className="team-auth-links">
          <button
            disabled={busy || disabled || remaining > 0}
            onClick={() => void perform(false)}
          >
            {remaining > 0 ? `Resend in ${remaining}s` : "Resend code"}
          </button>
          <button
            disabled={busy || disabled}
            onClick={() => {
              ++generation.current;
              session.cancel();
              setSent(false);
              setCode("");
              setError("");
            }}
          >
            Use another email
          </button>
        </div>
      )}
    </>
  );
}
