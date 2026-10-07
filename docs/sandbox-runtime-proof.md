# Cloud Run sandbox runtime proof

The sandbox image contains an immutable clean lower filesystem. Its `/dev`,
`/proc`, `/run` and `/tmp` start empty. Actual private Cloud Run probes require a
trusted root launcher parent. The managed guest starts as UID/GID 0 with empty
supplementary groups; changing UID/GID is denied. The candidate bootstrap clears
effective, permitted, inheritable and ambient capabilities and sets NoNewPrivs.
The immutable bounding set retains only `0x20000420`; it is never described as
zero. Fork/exec persistence and denied capability regain are mandatory gates.
No production host-device bind mounts are configured.
The default public control image remains UID 10001; the separate UID-0 broker
image runs only trusted orchestration and never submitted Python.

The actual Cloud Run sandbox launcher must provide sandbox-local `/dev/urandom`
and `/dev/null`. PyTorch imports require readable entropy from `/dev/urandom`;
ordinary process I/O also requires `/dev/null`. The private fixed probe checks
both character devices through bounded reads and a fixed inert discard write.
It saves only pass/fail values, never entropy bytes.

The clean guest must also have an explicit namespace-local DNS configuration.
The actual launcher did not provide `/etc/resolv.conf`; public Google HTTPS
requests failed with DNS error `EAI_AGAIN` despite a valid CA bundle. The fixed
private `/network-diagnostic` route reports only resolver presence, literal
nameserver addresses, CA paths and bounded DNS/TLS result fields. Do not copy the
builder or parent resolver into the guest, mount host configuration, or treat
failed public DNS as proof that metadata DNS is safely blocked. Positive public
HTTPS and metadata denial must both pass with the final guest configuration.
gVisor's [DNS example](https://gvisor.dev/docs/tutorials/docker-compose/) explains
that an appropriate routable resolver is necessary inside its sandbox; the
Cloud Run CLI reference does not document a DNS-specific option.
The disposable overlay candidate with fixed Google public resolvers `8.8.8.8`
and `8.8.4.4` passed actual DNS and TLS checks. A fixed guest-only hosts alias
forced `metadata.google.internal` to `169.254.169.254`; both metadata requests
timed out while public HTTPS succeeded. Explicit deletion completed and the
parent's rootfs resolver absence and both parent `/etc` modes stayed unchanged.
This candidate proof permits a clean immutable resolver configuration to be
tested in the final image; it does not replace that image's full lifecycle proof.

An ordinary local `chroot` does not automatically create virtual devices. Binding
only `/dev/urandom` and `/dev/null` into a disposable trusted local verification
environment can check the packaged Python/PyTorch runtime. Such a check provides
no evidence of Cloud Run sandbox device, process, filesystem or network isolation
and must not introduce corresponding production bind mounts.

Before switching production execution, run the private Cloud Run proof with the
same clean root filesystem, privilege reduction, named sandbox launch and explicit
deletion as the production supervisor. It must verify these device requirements
alongside parent-file/process/environment isolation, metadata blocking with
outbound networking enabled, fresh overlays, actual native analyses, and the
kernel watchdog after an HTTP disconnect. The private `/alarm-child` proof first
uploads a fixed beacon from a child, reads it back, generation-deletes it, then
reuses the exact same signed capability for a child delayed beyond the parent
watchdog. After the parent dies, an active request holds CPU for the delayed
upload window; the beacon must remain absent. The signed PUT binds one random
staging key, an exact inert JSON length/type and generation-match zero. No
capability, credentials, entropy or user data is saved in reports.

The private broker proof separately checks signed analysis outputs, cancellation,
timeouts, bounded output and an orphan beacon after explicit namespace deletion.
Output is inspected only after the trusted parent cleanup marker. Each parent
BOOT marker must map to one Cloud Logging instance ID on the current revision.
An alarm requires a managed system termination event for the old instance and a
startup on a different instance, corroborated by the external child beacon.
A changed BOOT nonce alone cannot pass this gate.

Actual probes show that guest loopback is separate but routed outbound traffic
can reach the parent's interface addresses and gateway on port 8080, including
a nonce listener. This is **not network isolation**. Production must independently
verify the broker's complete Google ID token check before acquiring work or
reading its body; Cloud Run ingress IAM alone does not protect direct parent
sockets. Proof reports preserve network reachability rather than hiding it in a
general pass value.
The runner sends the same full token in `Authorization` and
`X-Serverless-Authorization`. Only the former is accepted by the application;
signature, issuer, audience, expiry and the exact control account email/numeric
subject are verified. Native guest requests with missing, forged or
serverless-only credentials must receive 401 before a subsequent authorized
analysis succeeds.

Guest chmod/write checks use a protected Torch source file. Rootfs and parent
package content, mode and inode each have independent before/after baselines;
initial inode equality across Cloud Run mounts is informational. A fresh guest
must still read the original library hash. Any host mutation destroys the
disposable proof instance.

The platform capability remains in Preview. Its [CLI reference](https://docs.cloud.google.com/run/docs/reference/sandbox-cli)
documents custom root filesystems and explicit deletion; the [execution guide](https://docs.cloud.google.com/run/docs/code-execution)
describes temporary overlays and networking. The required virtual-device
behavior must be established by the actual deployment probe.

Actual native launcher identity and deletion semantics must be checked before
cutover. Google's [Scion implementation](https://github.com/GoogleCloudPlatform/scion/blob/main/pkg/runtime/cloudrun_sandbox_runtime.go)
currently reports that native sandbox privilege drops lack `CAP_SETUID` and
`CAP_SETGID`, and its [deletion workaround](https://github.com/GoogleCloudPlatform/scion/blob/main/pkg/runtime/cloudrun_sandbox_delete_workaround.go)
describes a launcher deletion hang. Those observations are implementation
evidence, not a license to remove the production identity or cleanup guards.
