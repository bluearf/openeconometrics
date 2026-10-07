

# Public snapshot policy: external research fixture bytes are distributed by
# their original providers. These fixture-backed modules remain in source for
# review, but do not enter the default public collection without those files.
# The explicit excluded module list is recorded in SOURCE-MANIFEST.json.
def pytest_ignore_collect(collection_path, config):
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / 'SOURCE-MANIFEST.json').read_text())
    try:
        relative = Path(collection_path).resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    return relative in record['external_fixture_test_modules']
