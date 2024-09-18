# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private vulnerability reporting
(repository → *Security* → *Report a vulnerability*) or email the maintainers listed in the repository
profile. Include the affected version or commit, reproduction steps and the impact. We aim to acknowledge
reports within 5 working days.

Supported versions: the latest release on `main`.

## Threat model (summary)

DefectLens is designed for a plant/lab network: a single site, a few dozen users, cameras or upload
clients on the same LAN. It is **not** hardened for direct exposure to the public internet. If you expose
it anyway, put it behind a TLS-terminating reverse proxy with additional authentication (SSO or VPN).

| Asset | Protection |
|---|---|
| Accounts | Argon2id password hashes; minimum length plus a common-password blocklist; per-IP login rate limit; default/temporary passwords flagged until changed |
| Sessions | Signed JWT in an `httpOnly`, `SameSite=Strict` cookie (`Secure` when `DL_COOKIE_SECURE=true`); a `token_version` per user, so logout, password change, role change or deactivation revokes all existing tokens server-side |
| CSRF | Every cookie-authenticated state-changing request needs the `X-Requested-With: defectlens` header (cross-site forms cannot set it), plus `SameSite=Strict` |
| Authorization | Roles `admin` / `inspector` / `viewer` enforced on every API route server-side (the UI only hides controls) |
| Uploads | Size limits; decoding with Pillow/OpenCV and a decompression-bomb pixel limit; extension allow-list; content hashing; never executed or served with a sniffable type (`X-Content-Type-Options: nosniff`) |
| ZIP archives | Path traversal neutralised (names flattened), symlinks skipped, member count/size and total uncompressed limits, compression-ratio > 200 rejected as a likely bomb |
| Audit trail | Append-only (ORM-level update/delete blocked) and hash-chained; *Settings → Audit log → Verify chain* or `defectlens verify-audit` detects tampering, including raw SQL edits |
| AI provider keys | Stored server-side, encrypted with a key derived from `DL_SECRET_KEY` (Fernet); never returned to the browser (masked); can instead be read from environment variables |
| Image privacy | Images are **never** sent to AI providers; only structured facts are. Cloud providers stay blocked until an admin allows them under *Privacy* |
| Browser | Strict Content-Security-Policy (`script-src 'self'`, no inline scripts, `object-src 'none'`, `frame-ancestors 'self'` in production), `Referrer-Policy`, `Permissions-Policy` (camera only for the app itself) |
| Rate limiting | Per-IP sliding window on all API routes (`DL_RATE_LIMIT_PER_MIN`) and stricter on login |

## Production checklist

- `DL_ENV=production`. Startup then refuses a missing or short `DL_SECRET_KEY` and a weak admin password,
  and the demo inspector/viewer accounts are never created.
- Serve over HTTPS and set `DL_COOKIE_SECURE=true`.
- Use PostgreSQL (`DL_DATABASE_URL`), back up with `defectlens backup --out ...`, and test restores.
- Keep `DL_VAR_DIR` on an access-controlled volume (it holds images, models and the database).
- Rotate `DL_SECRET_KEY` only together with re-entering provider keys: stored keys are encrypted with it,
  and rotating it signs everyone out.
- Review retention under *Settings → Privacy & retention* to match your data-protection obligations.

## Known limitations

- Rate limits are in-process, so they are per worker. Run one API process, or enforce limits at the proxy.
- No built-in SSO/LDAP, MFA or account lockout beyond rate limiting.
- PostgreSQL is supported by the code and migrations but was **not** exercised in this project's test
  runs (SQLite was).
- Dependency and secret scanning run in CI (`pip-audit`, `npm audit`, `gitleaks`). Review their reports
  before each release.
