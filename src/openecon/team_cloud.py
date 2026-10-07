"""Fail-closed production wiring for the isolated multi-project edition."""
from __future__ import annotations

import json
import os


def team_app():
    from openecon.team_auth import TeamAuth
    from openecon.team_server import create_team_app
    from openecon.team_storage import TeamStorage
    from openecon.team_store import FirestoreDocuments, TeamStore
    required = ('OPENECON_PROJECT_ID', 'OPENECON_PUBLIC_ORIGIN', 'OPENECON_OWNER_EMAIL',
                'OPENECON_BUCKET', 'OPENECON_SIGNER_EMAIL',
                'OPENECON_COMPUTE_EMAIL', 'OPENECON_FIREBASE_CONFIG')
    if any(not os.environ.get(key) for key in required):
        raise ValueError('Team serving requires explicit identity, storage and isolated compute configuration.')
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
    backend = os.environ.get('OPENECON_RUNNER', 'jobs')
    region = os.environ.get('OPENECON_REGION', 'us-central1')
    if backend not in ('sandbox', 'jobs'):
        raise ValueError('An explicitly supported isolated compute deployment is required.')
    settings = ('OPENECON_COMPUTE_SERVICE', 'OPENECON_COMPUTE_ORIGIN', 'OPENECON_COMPUTE_IMAGE')
    # Keep both handle readers across a rollback: choosing jobs for new work
    # must not send an already accepted sandbox execution to the Jobs API.
    configured_sandbox = backend == 'sandbox' or any(os.environ.get(key) for key in settings)
    if configured_sandbox:
        from openecon.team_dispatch import MigratingSandboxRunner
        from openecon.team_runner import GoogleJobRunner
        from openecon.team_sandbox_runner import GoogleSandboxRunner
        if any(not os.environ.get(key) for key in settings):
            raise ValueError('Sandbox compute requires an explicit private service, origin and image digest.')
        sandbox = GoogleSandboxRunner(
            project, region, os.environ[settings[0]], origin=os.environ[settings[1]],
            bucket=os.environ['OPENECON_BUCKET'], service_account=os.environ['OPENECON_COMPUTE_EMAIL'],
            image=os.environ[settings[2]], storage=storage, store=store,
            caller_subject=os.environ.get('OPENECON_BROKER_CALLER_SUB'),
        )
        legacy = (GoogleJobRunner(project, region, os.environ['OPENECON_COMPUTE_JOB'],
                                  service_account=os.environ['OPENECON_COMPUTE_EMAIL'])
                  if os.environ.get('OPENECON_COMPUTE_JOB') else None)
        runner = MigratingSandboxRunner(sandbox, legacy, start_backend=backend)
    elif backend == 'jobs' and os.environ.get('OPENECON_COMPUTE_JOB'):
        from openecon.team_runner import GoogleJobRunner
        runner = GoogleJobRunner(project, region, os.environ['OPENECON_COMPUTE_JOB'],
                                 service_account=os.environ['OPENECON_COMPUTE_EMAIL'])
    else:
        raise ValueError('An explicitly supported isolated compute deployment is required.')
    return create_team_app(
        store=store,
        storage=storage,
        auth=TeamAuth(project),
        runner=runner,
        public_origin=origin, firebase_config=firebase,
    )
