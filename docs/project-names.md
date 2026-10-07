# Project names

Project owners can use **Proje adını değiştir** beside a project's name, either
in the project list or in the open workspace header. Enter or the check mark
saves; Escape or **Vazgeç** cancels. Editors and viewers can open projects but
cannot rename them.

Names contain 1–100 Unicode characters after ordinary surrounding whitespace
is trimmed. Control characters and line separators are rejected. Names appear
immediately while saving. Only the rename control waits; the Python editor,
its pending source changes and its execution session remain available. A failed
save restores the typed name for retry. A successful response updates the
account's desktop profile cache; an unacknowledged name is never cached as a
completed cloud change.

Renaming updates the shared project name and pending invitation labels. Project
IDs, links, membership, folders, file identities, source, data and environments
remain unchanged. The cloud transaction checks verified owner membership and
`name_version` together, records one audit event for an actual change, and rejects
stale names with `VERSION_CONFLICT`. Older projects begin at name version zero
without a migration. Same-name requests are idempotent only at the current
version. After a conflict, refreshed metadata supplies the version for retry.

The API is `PATCH /api/projects/{project_id}` with exactly:

```json
{"name": "Research project", "name_version": 0}
```

An acknowledged response contains the project summary and its name version.
The native bridge allows this exact project route and method. Account changes
and closed UI contexts suppress late callbacks; pending or acknowledged name
updates cannot grant roles or restore removed project membership.
