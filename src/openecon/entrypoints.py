"""Small CLI bootstrap with actionable errors for library-only installations."""
import sys


def main():
    from openecon.optional_dependencies import require_extra
    try:
        require_extra("cli", "typer")
        from openecon.cli import app
        app()
    except ImportError as error:
        if "Install with pip install 'openecon[" not in str(error):
            raise
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
