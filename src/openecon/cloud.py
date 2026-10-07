"""Single-owner Cloud Run entry point. Local serving retains its loopback guard."""
from __future__ import annotations

import os
from pathlib import Path

from openecon.cloud_access import CloudAccess


def cloud_app():
    if os.environ.get('OPENECON_MODE') == 'teams':
        from openecon.team_cloud import team_app
        return team_app()
    required = ("OPENECON_PUBLIC_ORIGIN", "OPENECON_IAP_AUDIENCE", "OPENECON_OWNER_EMAIL")
    if any(not os.environ.get(name) for name in required):
        raise ValueError("Cloud serving requires an HTTPS origin, IAP audience and workspace owner.")
    from openecon.server import create_app

    access = CloudAccess(
        public_origin=os.environ[required[0]], audience=os.environ[required[1]],
        owner_email=os.environ[required[2]],
    )
    workspace = Path(os.environ.get("OPENECON_WORKSPACE", "/tmp/openecon"))
    return create_app(workspace, cloud_access=access)


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    uvicorn.run(cloud_app(), host="0.0.0.0", port=port, workers=1,
                proxy_headers=True, forwarded_allow_ips="*", access_log=False)


if __name__ == "__main__":
    main()
