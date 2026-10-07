"""Route existing executions independently of the configured launch backend."""
from __future__ import annotations


class MigratingSandboxRunner:
    def __init__(self, sandbox, legacy=None, *, start_backend='sandbox'):
        if start_backend not in ('sandbox', 'jobs') or (start_backend == 'jobs' and legacy is None):
            raise ValueError('A configured isolated launch backend is required.')
        self.sandbox, self.legacy = sandbox, legacy
        self.start_backend = start_backend

    def start(self, input_url, timeout_seconds=120):
        runner = self.sandbox if self.start_backend == 'sandbox' else self.legacy
        return runner.start(input_url, timeout_seconds=timeout_seconds)

    def _runner(self, handle):
        if not isinstance(handle, str):
            raise ValueError('An isolated execution handle is required.')
        if handle.startswith('sandbox:'):
            return self.sandbox
        if self.legacy is None:
            raise ValueError('No legacy execution backend is configured.')
        # The legacy runner independently checks full project/region/job scope;
        # an unknown or foreign handle cannot select a different cloud resource.
        return self.legacy

    def status(self, operation):
        return self._runner(operation).status(operation)

    def cancel(self, execution):
        return self._runner(execution).cancel(execution)
