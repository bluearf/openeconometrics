"""Fail-closed production wiring for the team sync backend.

The service stores accounts, memberships, files and shared desktop results. It
has no compute backend: analyses run only in the local desktop app.
"""
from __future__ import annotations

import json
import os

REQUIRED_SETTINGS = ('OPENECON_PROJECT_ID', 'OPENECON_PUBLIC_ORIGIN', 'OPENECON_OWNER_EMAIL',
                     'OPENECON_BUCKET', 'OPENECON_SIGNER_EMAIL', 'OPENECON_FIREBASE_CONFIG')


def team_app():
    from openecon.team_auth import TeamAuth
    from openecon.team_server import create_team_app
    from openecon.team_storage import TeamStorage
    from openecon.team_store import FirestoreDocuments, TeamStore
    if any(not os.environ.get(key) for key in REQUIRED_SETTINGS):
        raise ValueError('Team serving requires explicit identity and storage configuration.')
    project = os.environ['OPENECON_PROJECT_ID']
    firebase = json.loads(os.environ['OPENECON_FIREBASE_CONFIG'])
    if (not isinstance(firebase, dict) or firebase.get('projectId') != project
            or not all(isinstance(firebase.get(k), str) and firebase[k]
                       for k in ('apiKey', 'authDomain', 'appId'))):
        raise ValueError('Firebase web configuration must belong to this project.')
    origin = os.environ['OPENECON_PUBLIC_ORIGIN']
    store = TeamStore(FirestoreDocuments(project), owner_email=os.environ['OPENECON_OWNER_EMAIL'],
                      public_origin=origin)
    storage = TeamStorage(project, os.environ['OPENECON_BUCKET'], os.environ['OPENECON_SIGNER_EMAIL'])
    return create_team_app(
        store=store,
        storage=storage,
        auth=TeamAuth(project),
        public_origin=origin, firebase_config=firebase,
        # Set at image build time from the reviewed source commit; read-only.
        source_commit_id=os.environ.get('OPENECON_SOURCE_COMMIT'),
    )
