"""PyInstaller launcher: spawn dispatch must run before the server entrypoint."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from openecon.desktop_entry import main
    raise SystemExit(main())
