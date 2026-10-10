"""Cloud Run entry point for the team sync backend. It never runs analyses.

Local serving (`openecon serve`) keeps its loopback guard and runs analyses on
the user's machine. The retired single-owner cloud workbench is not available.
"""
from __future__ import annotations

import os


def cloud_app():
    if os.environ.get('OPENECON_MODE') != 'teams':
        raise ValueError('The cloud service only provides team sync (OPENECON_MODE=teams). '
                         'Analyses run in the OpenEconometrics desktop app.')
    from openecon.team_cloud import team_app
    return team_app()


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    uvicorn.run(cloud_app(), host="0.0.0.0", port=port, workers=1,
                proxy_headers=True, forwarded_allow_ips="*", access_log=False)


if __name__ == "__main__":
    main()
