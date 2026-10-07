"""Trusted Cloud Run Jobs bridge; this module never executes submitted Python."""
from __future__ import annotations

import math
import re
from typing import Any

from openecon.team_job import storage_url


class JobRunnerError(RuntimeError):
    """A cloud execution could not be started, inspected, or cancelled."""


def _clients():
    from google.cloud import run_v2
    return run_v2.JobsClient(), run_v2.ExecutionsClient()


def _operation_dict(operation) -> dict:
    from google.protobuf.json_format import MessageToDict
    return MessageToDict(operation, preserving_proto_field_name=True)


class GoogleJobRunner:
    """Launch a fixed roleless job with per-run input capabilities only.

    This bridge never changes the job image, identity, retry policy, or resource
    limits. Those settings belong to its reviewed deployment configuration.
    """
    def __init__(self, project: str, region: str, job: str,
                 bucket: str | None = None, service_account: str | None = None):
        if not isinstance(project, str) or not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
            raise ValueError("A Google Cloud project ID is required.")
        if not isinstance(region, str) or not re.fullmatch(r"[a-z]+(?:-[a-z]+)+[0-9]+", region):
            raise ValueError("A Cloud Run region is required.")
        if not isinstance(job, str) or not re.fullmatch(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", job):
            raise ValueError("A fixed Cloud Run job name is required.")
        if service_account is not None and (not isinstance(service_account, str) or not re.fullmatch(
                r"[a-z][a-z0-9-]{4,28}[a-z0-9]@[a-z][a-z0-9-]{4,61}[a-z0-9]\.iam\.gserviceaccount\.com",
                service_account)):
            raise ValueError("An explicit worker service account email is required.")
        self.project, self.region, self.job = project, region, job
        self.bucket, self.service_account = bucket, service_account
        self.parent = f"projects/{project}/locations/{region}"
        self.name = f"{self.parent}/jobs/{job}"
        self._jobs, self._executions = _clients()

    def _operation_name(self, name: str) -> str:
        prefix = self.parent + "/operations/"
        if (not isinstance(name, str) or not name.startswith(prefix)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name[len(prefix):])):
            raise ValueError("The operation must belong to the configured project and region.")
        return name

    def _execution_name(self, name: str) -> str:
        prefix = self.name + "/executions/"
        if (not isinstance(name, str) or not name.startswith(prefix)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name[len(prefix):])):
            raise ValueError("The execution must belong to the configured job.")
        return name

    def _summary(self, raw: Any) -> dict:
        operation = _operation_dict(raw)
        self._operation_name(operation["name"])
        execution = operation.get("response") or operation.get("metadata") or {}
        execution_name = execution.get("name")
        if execution_name:
            self._execution_name(execution_name)
        done = bool(operation.get("done", False))
        error = operation.get("error")
        status = "running" if execution.get("start_time") else "queued"
        if done:
            if error:
                status = "cancelled" if error.get("code") == 1 else "failed"
            elif execution.get("cancelled_count", 0):
                status = "cancelled"
            elif execution.get("failed_count", 0):
                status = "failed"
            else:
                status = "succeeded" if execution.get("succeeded_count") == 1 else "failed"
                for condition in execution.get("conditions", []):
                    if condition.get("type_", condition.get("type")) == "Completed":
                        status = ("succeeded" if condition.get("state") == "CONDITION_SUCCEEDED"
                                  else "failed")
        return {"operation": operation["name"], "execution": execution_name,
                "status": status, "message": "Cloud execution failed." if status == "failed" else None}

    def _verify_worker(self) -> None:
        """Reject identity or isolation drift before granting a run capability.

        Deployment must separately ensure this identity has no effective IAM
        roles. Reading the task template cannot establish its permissions.
        """
        if self.service_account is None:
            return
        try:
            job = self._jobs.get_job(request={"name": self.name}, retry=None, timeout=10)
            execution = job.template
            task = execution.template
            safe = (job.name == self.name and task.service_account == self.service_account
                    and execution.task_count == 1 and execution.parallelism in (0, 1)
                    and task.max_retries == 0 and len(task.containers) == 1
                    and not task.volumes and not task.vpc_access.connector
                    and not task.vpc_access.network_interfaces)
            if safe:
                safe = not any(variable.value_source.secret_key_ref.secret
                               for variable in task.containers[0].env)
        except Exception as exc:
            raise JobRunnerError("Worker configuration could not be verified; no run was launched.") from exc
        if not safe:
            raise JobRunnerError("Worker configuration does not meet isolation requirements; no run was launched.")

    def start(self, input_url: str, timeout_seconds: float = 120) -> dict:
        storage_url(input_url)
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not .05 <= timeout_seconds <= 120):
            raise ValueError("Execution timeout must be between 0.05 and 120 seconds.")
        self._verify_worker()
        try:
            operation = self._jobs.run_job(request={
                "name": self.name,
                "overrides": {
                    "container_overrides": [{"env": [{"name": "OPENECON_RUN_INPUT_URL", "value": input_url}]}],
                    "task_count": 1,
                    # Includes Python import/startup, bounded input/output transfer,
                    # and worker cleanup; the user-code limit remains <=120s.
                    "timeout": {"seconds": math.ceil(timeout_seconds) + 120},
                },
            }, retry=None, timeout=20)
            return self._summary(operation.operation)
        except Exception as exc:
            # A network failure may have an unknown acceptance outcome. The API
            # must not blindly retry this launch and create duplicate executions.
            raise JobRunnerError("Cloud launch could not be confirmed; do not automatically retry.") from exc

    def status(self, operation_name: str) -> dict:
        self._operation_name(operation_name)
        try:
            raw = self._jobs.transport.operations_client.get_operation(
                operation_name, retry=None, timeout=10,
            )
            return self._summary(raw)
        except Exception as exc:
            raise JobRunnerError("Cloud execution status is temporarily unavailable.") from exc

    def cancel(self, execution_name: str) -> dict:
        self._execution_name(execution_name)
        try:
            operation = self._executions.cancel_execution(
                request={"name": execution_name}, retry=None, timeout=15,
            )
            raw = _operation_dict(operation.operation)
            operation_name = self._operation_name(raw["name"])
            return {"operation": operation_name, "execution": execution_name, "status": "cancelling"}
        except Exception as exc:
            raise JobRunnerError("The execution cancellation could not be confirmed.") from exc
