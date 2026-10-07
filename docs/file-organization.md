# Project file organization

The **Dosyalar** sidebar organizes Python scripts, registered data files and
folders in a project. Folders expand and collapse; they can contain other
folders, scripts and data files.

Header button descriptions close after four seconds, or immediately on click,
Escape, mouse exit or focus loss. They remain closed until a new hover or focus
session. Menus and naming forms remain available until completed or cancelled.

| Action | Control |
|---|---|
| Rename a script, data file or folder | Open its **…** menu, choose **Yeniden adlandır**, then press Enter or ✓. Escape cancels. |
| Create a folder | Use **Yeni klasör** in the sidebar header. |
| Create or import within a folder | Select the folder first, then use the header actions. **Ana dizin** selects the project root. |
| Move an item | Choose **Taşı** and select a folder or **Dosyalar** for the root. |
| Change sibling order | Choose **Yukarı taşı** or **Aşağı taşı**. |
| Drag an item | Drop onto a folder to move inside, onto a file to place before it, or onto empty sidebar space to move to the root. |

The main script can be renamed like other scripts. Names appear in the sidebar,
active editor tab and downloads. Python script names retain `.py`; data files
retain their existing extension. Sibling names must be distinct, including
case differences. Path separators, control characters and unsafe names are
rejected. Folder nesting is bounded to 16 levels.

These are logical project names and folders. Renaming or moving an item keeps
its immutable identity, stored source, data snapshot and Python reading path.
It does not move an operating-system directory or rewrite `oe.read(...)` paths
inside scripts. Existing analyses therefore retain their input references.

Valid names and folder changes appear immediately, and the form closes while
the local metadata write completes. Renaming does not flush or wait for the
active script's independent autosave; its identity and source remain unchanged.
Typing and execution stay available during metadata persistence. A failed local
save restores the entered name for correction or retry. Running, uploading and
pending file operations disable competing layout changes. Viewers can browse
the tree; owners and editors can change its organization.

Cloud projects share one versioned layout. Saves check the current version and
file inventory, so one team member cannot silently overwrite another member's
layout. Newly registered files append to the root until placed in a folder.
An inventory conflict refreshes the file catalog before a retry.

The desktop edition acknowledges the durable local metadata write without
waiting for source uploads, cloud requests or remote catalog refreshes. It shows
**Yerelde** until the layout has been acknowledged by the cloud. Cloud writes
serialize in the background, preserving any newer local renames that arrive
during an upload. The acknowledgement indicator updates through a scoped
workspace subscription. Synchronization also retries when connectivity returns
or the project is reopened. A competing cloud edit preserves the local layout
and shows a conflict. **Yerel düzeni indir, güncel düzeni aç** exports the local
layout as JSON before opening the cloud version. Authentication or permission
errors are never treated as an offline synchronization success.

Layout metadata uses `GET` and versioned `PUT /files/layout`. Each ordered entry
contains `kind`, `id`, `name` and a folder `parent` or `null`. Code and data
continue to use their existing identity-based APIs; organization does not
execute Python.
