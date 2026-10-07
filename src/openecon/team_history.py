"""Bounded metadata pages. Result blobs are read only when a member opens one."""
from __future__ import annotations

import base64
from datetime import date, datetime, timezone
import hashlib
import json
import re
import unicodedata

from openecon.team_store import TeamError, ident, now

MAX_HISTORY_COUNT = 50
MAX_HISTORY_BYTES = 256 * 1024
MAX_HISTORY_SCAN = 100
HISTORY_FIELDS = ('id', 'created_at', 'code', 'state', 'record_summary', 'email', 'generation')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def history_filters(query='', since='', until=''):
    if (not isinstance(query, str) or len(query) > 256
            or any(unicodedata.category(c) in {'Cc', 'Cs'} for c in query)):
        raise TeamError('INVALID_HISTORY_FILTER', 'Search with up to 256 characters without control characters.', 422)
    for value in (since, until):
        try:
            if value and (not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) or date.fromisoformat(value).isoformat() != value):
                raise ValueError
        except (ValueError, TypeError):
            raise TeamError('INVALID_HISTORY_FILTER', 'Use a valid YYYY-MM-DD date.', 422) from None
    if since and until and since > until:
        raise TeamError('INVALID_HISTORY_FILTER', 'The start date must be on or before the end date.', 422)
    return {'query': query.strip().casefold(), 'since': since, 'until': until}


def cursor_value(project_id, filters, snapshot, before):
    return base64.urlsafe_b64encode(encoded({'v': 1, 'project': project_id,
                                           'filter_hash': hashlib.sha256(encoded(filters)).hexdigest(),
                                           'snapshot': snapshot, 'before': list(before) if before else None})).decode().rstrip('=')


def read_cursor(cursor, project_id, filters):
    try:
        if not isinstance(cursor, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,2048}', cursor):
            raise ValueError
        value = json.loads(base64.b64decode(cursor + '=' * (-len(cursor) % 4), altchars=b'-_', validate=True))
        if (set(value) != {'v', 'project', 'filter_hash', 'snapshot', 'before'} or type(value['v']) is not int
                or value['v'] != 1 or value['project'] != project_id
                or value['filter_hash'] != hashlib.sha256(encoded(filters)).hexdigest()
                or (value['before'] is not None and
                    (not isinstance(value['before'], list) or len(value['before']) != 2))):
            raise ValueError
        for timestamp in ([value['snapshot'], value['before'][0]] if value['before'] else [value['snapshot']]):
            if (not isinstance(timestamp, str) or len(timestamp) > 40
                    or datetime.fromisoformat(timestamp).tzinfo != timezone.utc):
                raise ValueError
        if value['before']:
            ident(value['before'][1])
            if value['before'][0] > value['snapshot']:
                raise ValueError
        return value['snapshot'], tuple(value['before']) if value['before'] else None
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise TeamError('INVALID_HISTORY_CURSOR', 'This history page is invalid. Refresh the search.', 422) from None


def history_page(store, project_id, user, *, cursor=None, query='', since='', until='', limit=20, max_bytes=64 * 1024):
    # Authorization precedes cursor parsing and runs again after metadata reads.
    store.project(project_id, user)
    if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_COUNT:
        raise TeamError('INVALID_HISTORY_LIMIT', 'History pages support 1–50 runs.', 422)
    if type(max_bytes) is not int or not 4096 <= max_bytes <= MAX_HISTORY_BYTES:
        raise TeamError('INVALID_HISTORY_LIMIT', 'History pages support 4–256 KiB.', 422)
    filters = history_filters(query, since, until)
    snapshot, before = read_cursor(cursor, project_id, filters) if cursor is not None else (now(), None)
    rows = store.db.history_scan(f'oe_projects/{project_id}/runs', snapshot=snapshot,
                                 before=before, limit=MAX_HISTORY_SCAN + 1)
    page = {'runs': [], 'next_cursor': None, 'page_cursor': cursor_value(project_id, filters, snapshot, before),
            'snapshot_at': snapshot, 'scanned': 0}
    last = before
    for row in rows[:MAX_HISTORY_SCAN]:
        key = (row['created_at'], row['_document_id'])
        code, day = row.get('code', ''), row['created_at'][:10]
        matches = ((not filters['query'] or filters['query'] in code.casefold()
                    or filters['query'] in row['id'].casefold())
                   and (not since or day >= since) and (not until or day <= until))
        if matches:
            summary = row.get('record_summary') or {}
            entry = {'id': row['id'], 'created_at': row['created_at'], 'state': row['state'],
                     'status': summary.get('status'), 'code_preview': code[:240],
                     'code_preview_truncated': len(code) > 240, 'actor_email': row.get('email', '')[:256]}
            # Reserve the full cursor/envelope; never consume an entry that did
            # not fit, so the next page includes it with its original identity.
            candidate = {**page, 'runs': [*page['runs'], entry], 'scanned': page['scanned'] + 1,
                         'next_cursor': cursor_value(project_id, filters, snapshot, key)}
            if len(page['runs']) >= limit or len(encoded(candidate)) > max_bytes:
                break
            page['runs'].append(entry)
        page['scanned'] += 1
        last = key
    if last and (page['scanned'] < len(rows)):
        page['next_cursor'] = cursor_value(project_id, filters, snapshot, last)
    store.project(project_id, user)
    if len(encoded(page)) > max_bytes:
        raise TeamError('HISTORY_LIMIT', 'The history metadata exceeds the page budget.', 413)
    return page
