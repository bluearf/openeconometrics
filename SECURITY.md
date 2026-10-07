# Security reports

Do not put credentials, personal data, exploitable live service details or
private project files in a public issue. Use GitHub's **Report a vulnerability**
on the repository when private vulnerability reporting is enabled. If that
option is unavailable, contact a repository maintainer privately before sharing
sensitive details. No response-time or supported-version guarantee is implied.

The local console executes trusted Python with the user's account permissions.
Its separate worker process is not an operating-system security sandbox. Cloud
sharing and desktop authentication require their own authorization checks.

Before publication, scan the exact source snapshot and intended Git history,
review reported findings without exposing their values, and check bundled
assets and external research fixtures separately. A clean secret-scanner result
does not establish that a dataset is anonymous or licensed for redistribution.
