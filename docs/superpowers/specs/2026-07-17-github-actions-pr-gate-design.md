# GitHub Actions PR Gate Design

## Goal

Add a mandatory GitHub pull-request gate for TileXR. A pull request targeting
`main` must pass fast host checks and all single-host tests supported by the
eight Ascend 910B3 devices on `blue` before it can be merged.

This design covers continuous integration and merge protection. It does not
publish packages, deploy TileXR, validate multi-host communication, or validate
the A5 / Ascend950 UDMA data plane.

## Current State

- `LingquLab/TileXR` is a public repository with `main` as its default branch.
- The repository has no project-owned GitHub Actions workflow yet.
- The active default-branch ruleset requires two approvals, a code-owner
  approval when an owner applies, and resolution of review threads. It blocks
  deletion and non-fast-forward updates and has no bypass actor.
- The ruleset does not currently require a status check.
- The repository does not currently contain a `CODEOWNERS` file.
- GitHub Actions has read-only default workflow permissions. It currently
  allows all Actions and requires approval only for first-time external
  contributors.
- `blue` is an aarch64 openEuler 22.03 host with eight healthy Ascend 910B3
  devices and driver 25.5.0.
- `blue` currently has a system CANN 8.2.RC1 installation. TileXR requires
  CANN 9.1.0.
- The root filesystem on `blue` has little free space, while `/home` has enough
  space for the runner, toolchain, workspaces, and logs.
- `blue` is shared with other users and has no active cluster scheduler.

## Design Decisions

1. Use a two-stage pipeline: fast checks on a GitHub-hosted runner, followed by
   CANN build and NPU testing on one dedicated self-hosted runner on `blue`.
2. Run the self-hosted stage automatically for same-repository pull requests.
   Require maintainer approval before any workflow from an external fork runs.
3. Test the GitHub-generated pull-request merge commit, not only the pull
   request head commit.
4. Use one self-hosted runner so NPU jobs from different pull requests queue
   instead of running concurrently.
5. Share the NPU host with existing users. Wait for all eight devices to be
   idle and never terminate another user's process.
6. Limit resource waiting to six hours and NPU execution to two hours.
7. Require all tests supported by the single-host 910B3 environment. Explicitly
   exclude multi-host and A5 / Ascend950 UDMA data-plane validation.
8. Add one stable `PR Gate` required check to the default-branch ruleset only
   after a successful trial run has produced that check.

## Workflow Architecture

### Pull-request workflow

Add `.github/workflows/pr-ci.yml` with a `pull_request` trigger for pull
requests targeting `main`. Handle these activity types:

- `opened`
- `reopened`
- `synchronize`
- `ready_for_review`

The workflow has three externally visible jobs:

1. `Host Checks`
2. `NPU Gate`
3. `PR Gate`

`Host Checks` runs first. `NPU Gate` runs only for a non-draft pull request
whose host checks passed. `PR Gate` reports success only when every required
job for the current merge commit succeeded.

Draft pull requests run host checks but do not consume the NPU runner. Moving a
draft to ready-for-review triggers the complete gate.

Use per-pull-request concurrency with cancellation enabled. A new commit to a
pull request cancels its obsolete run. Do not use one global Actions
`concurrency` group for all NPU jobs: GitHub keeps only one pending run in such
a group and replaces older pending runs, which would not provide the required
cross-PR queue.

### Trusted reusable NPU workflow

Add `.github/workflows/npu-ci.yml` as a reusable `workflow_call` workflow. The
pull-request workflow calls the copy fixed on `main`. The organization runner
group permits only this trusted workflow to target the NPU runner.

The reusable workflow derives the pull request number and expected merge SHA
from the GitHub event payload. It invokes a versioned control package installed
outside the runner workspace and checks out `refs/pull/${PR_NUMBER}/merge` into
a separate untrusted source directory, where `PR_NUMBER` comes from that event.

The source for the control package lives under a CODEOWNERS-protected CI
directory in the repository. Provisioning installs the reviewed version under
`/home/tilexr-ci/control/v1`, makes it administrator-owned and read-only, and
updates an administrator-owned `current` link. The initial reusable workflow
expects control API version `v1` and fails before executing pull-request code if
the installed version does not match. A future incompatible control change must
use a new versioned directory and update the workflow's expected version.

Before building, it verifies that the checked-out source commit equals the
event's merge SHA. The sealed control package owns resource acquisition,
timeouts, monitoring, cleanup, and result classification. Pull-request code is
still compiled and executed, because that is the purpose of the test, but it
cannot replace the control package on disk.

The read-only GitHub token is a one-shot controller input, not a child
environment variable. The reusable workflow removes token variables, creates
private FD 3, clears its shell copy, and uses a final `exec` into the sealed
controller. The controller disables Linux process dumpability before a bounded
read, closes the FD, and strips token and GitHub command-file variables from
every untrusted child environment.

Use only the standard `pull_request` event. Do not use `pull_request_target` to
check out or execute pull-request code.

### Repository components

The implementation adds these owned components:

- `.github/CODEOWNERS` for the CI security boundary;
- `.github/workflows/pr-ci.yml` for pull-request orchestration;
- `.github/workflows/npu-ci.yml` for the trusted reusable NPU job;
- `scripts/ci/host_checks.sh` for GitHub-hosted checks;
- `scripts/ci/control/` as the source of the sealed `v1` control package;
- `scripts/ci/provision/` for reviewed, one-time runner and toolchain setup.

The trusted control package contains the test manifest, resource waiter, usage
monitor, timeout wrapper, process cleanup, artifact collector, and failure
classifier. It also contains a runner job-completed hook that removes the
finished job's workspace after Actions post-job steps have run. Test
implementation files remain in their existing suite directories.

## GitHub-hosted Host Checks

Run host checks in an Ubuntu 20.04 container on a GitHub-hosted Linux runner.
This preserves coverage of TileXR's documented target OS while the hardware
stage covers the real openEuler deployment host.

The required host checks are:

- build and run all `tests/comm` binaries, including the source guard;
- build `tests/ep` in `source-only` mode and run all four tests;
- build and run both `tests/data_as_flag` tests;
- run the collectives Python source, patch, and profile-report tests;
- validate shell syntax for the CI entrypoints and affected repository scripts;
- validate workflow and CMake configuration syntax with the tools installed by
  the host-check image.

A host-check failure prevents scheduling the expensive `blue` job.

## Self-hosted Runner

### Account and service

Create a dedicated `tilexr-ci` system account with these properties:

- home directory `/home/tilexr-ci`;
- locked password and no interactive SSH login;
- membership in `HwHiAiUser` for Ascend device and driver access;
- no sudo permission;
- no membership in the `docker` group;
- a systemd-managed GitHub Actions runner service running as `tilexr-ci`.

Register one aarch64 runner with labels that include:

- `tilexr`
- `ascend910b`
- `npu8`

Place it in an organization runner group named `TileXR-NPU`. Restrict the group
to the TileXR repository and the trusted reusable NPU workflow. The group
configuration must:

- allow use by this public repository;
- use selected-repository visibility with only `LingquLab/TileXR` selected;
- enable selected-workflow restriction; and
- allow only
  `LingquLab/TileXR/.github/workflows/npu-ci.yml@refs/heads/main`.

Creating and restricting this group requires organization-owner authorization.

Keep the runner, workspaces, temporary files, and logs under
`/home/tilexr-ci`. The only new root-filesystem content is the small account and
systemd configuration required to run the service.

### CANN 9.1 toolchain

Install CANN 9.1 Toolkit first and the Ascend 910B Ops package second under a
versioned directory in `/home`, separate from `/usr/local/Ascend` and the
existing system CANN 8.2 installation.

The stable toolchain path is:

```text
/home/tilexr-ci/toolchains/cann/9.1.0
```

Use a trusted provisioning checkout and the repository's local CANN installer
with an explicit install path. After installation and validation, make the
toolchain and trusted CI control scripts administrator-owned and read-only to
the runner. The workflow supplies the stable path through an explicit
`TILEXR_CANN_HOME`; `scripts/common_env.sh` preserves that caller-provided path
while retaining its current repo-local default. Pull-request workspaces do not
install or update CANN.

One-time provisioning validates:

- driver version 25.5.0;
- all eight 910B3 devices are visible and healthy;
- CANN reports version 9.1.0;
- Toolkit and 910B Ops are present;
- `bisheng`, CMake, and the required headers and libraries resolve from the
  dedicated toolchain;
- a minimal ACL initialization succeeds;
- runtime `libascend_hal.so` resolves from the driver, not CANN `devlib`.

The workflow never receives sudo and cannot modify the sealed toolchain.

## Build Stage on Blue

The build stage runs before waiting for all NPUs to become idle. Compilation
does not reserve NPU resources and can fail quickly without occupying the
hardware queue longer than necessary.

For each merge commit:

1. Create a clean source and build tree.
2. Initialize the pinned `spdlog` submodule.
3. Source `scripts/common_env.sh` against the sealed CANN 9.1 toolchain.
4. Configure a clean top-level build with tests, checker, EP, and collectives
   enabled.
5. Build and install `tile-comm`, EP, collectives, checker, and their test
   targets.
6. Run the complete top-level CTest suite.
7. Build the dedicated comm, UDMA, SDMA, memory, and EP test trees. Force the
   hardware demo targets required for 910B instead of accepting their current
   optional skip behavior.
8. Check installed headers and libraries, dynamic dependencies, and RPATH or
   RUNPATH entries.

The control process monitors NPU usage during the build. A process owned by
`tilexr-ci` appearing on an NPU before resource acquisition is a policy
violation and aborts the job.

Missing `bisheng`, missing required binaries, a skipped required target, or a
runtime dependency on the stub HAL is a build failure. A required test may not
silently become a skip.

## NPU Queue and Resource Ownership

The single self-hosted runner provides the primary FIFO-like queue across pull
requests. A host-side `flock` in a directory owned by `tilexr-ci` provides a
second mutual-exclusion boundary in case another CI entrypoint is added later.

After the build passes, the trusted control script:

1. Acquires the CI lock.
2. Polls the process table from `npu-smi info` every 60 seconds and queries
   `npu-smi info -t health -i ${DEVICE_ID}` for each device.
3. Requires devices 0 through 7 to report `Health: OK` and have no process-table
   entry for two consecutive samples, giving a continuous 60-second idle
   window.
4. Waits at most six hours.
5. Logs the busy device, process ID, process owner, elapsed time, and remaining
   wait time.
6. Revalidates the pull-request head and merge SHA before starting hardware
   execution so an obsolete run cannot consume the devices.

The workflow never calls `npu-smi release` and never sends a signal to a
process owned by another account.

During hardware execution, a monitor checks NPU usage for foreign processes.
If another user starts an NPU process after the idle check, the monitor stops
the CI-owned test process group, reports a resource collision, and leaves the
foreign process untouched.

Because other users do not participate in the CI lock, a race between the idle
check and an external process cannot be eliminated. Monitoring and immediate
CI failure are the accepted behavior.

GitHub may fail a self-hosted job that is not assigned to a runner within its
24-hour routing timeout. A deep queue that reaches this limit is classified as
an infrastructure failure and requires a manual rerun.

## Hardware Test Matrix

After all eight devices are acquired, run the following tests against devices
0 through 7.

### Environment and communicator smoke

- capture driver, CANN, compiler, device topology, health, and usage versions;
- initialize ACL on each device;
- run applicable comm integration tests, including UDMA layout, registry,
  source-boundary, and graceful-unavailable behavior on 910B3.

The UDMA checks on this host prove compatibility and fallback behavior only.
They do not claim UDMA data-plane validation.

### Peer memory and EP

- run the peer-memory DataCopy demo with eight ranks and the 910B compiler
  target;
- run the EP dispatch/combine demo with eight local ranks.

### Standalone collectives

Run eight-rank correctness tests for:

- AllGather;
- AllToAll;
- AllReduce;
- ReduceScatter;
- Broadcast with root 0;
- Broadcast with root 7.

Also run checked, multi-size performance smoke tests for the supported
collectives. These tests validate output and execution over representative
small and medium message sizes. They record timing but do not enforce a
performance threshold in this phase.

### SDMA

Run the SDMA demo on each device from 0 through 7 with representative sizes,
including 64 bytes, 4096 bytes, and 1 MiB.

### Explicit exclusions

The gate records these as out of scope rather than silently skipping them:

- A5 / Ascend950 UDMA data-plane demos;
- multi-host collectives and EP;
- vLLM model inference;
- performance-regression thresholds.

## Timeouts, Cancellation, and Cleanup

- NPU resource wait: six hours.
- Hardware execution after acquisition: two hours total.
- Each multi-rank launch: ten minutes.
- Test failures are not retried automatically.
- Resource and infrastructure failures can be rerun manually after the cause
  is resolved.

The trusted driver handles normal exit, failure, cancellation, `SIGINT`, and
`SIGTERM`. It tracks the CI process group, terminates only processes owned by
`tilexr-ci`, verifies that no CI NPU process remains, and then releases the file
lock. An administrator-owned runner job-completed hook removes the untrusted
workspace after checkout and artifact post-job actions finish, including after
failure or cancellation.

Mutable compiler caches are not shared between untrusted pull requests in the
initial implementation. The CANN toolchain is shared only because it is sealed
read-only.

## Results and Artifacts

Every run writes a GitHub step summary containing:

- pull request number, head SHA, merge SHA, and base SHA;
- host-check, build, queue, and hardware-stage outcomes;
- wait duration and execution duration;
- test counts and failed test names;
- explicit out-of-scope coverage.

Upload artifacts on success, failure, and cancellation where cleanup permits:

- environment and version snapshot;
- CMake configuration and build logs;
- CTest and JUnit results;
- dynamic dependency and RPATH reports;
- pre-run and post-run NPU state;
- per-rank memory, EP, and collective logs;
- SDMA logs for devices 0 through 7.

Retain artifacts for 14 days. This phase does not publish an install package or
GitHub Release.

Host evidence is created below the repository's real `.ci-artifacts/host`
directory. NPU evidence is created below the runner's real temporary directory,
outside the untrusted merge checkout. The collector never follows source
links, accepts only bounded regular-file evidence, and writes a final manifest
and rejection report. Before untrusted phases run, the controller pins the
runner-visible step-summary alias and complete directory chain using file
descriptors, authenticates the original inode and prefix, and performs bounded
transactional appends. A replaced or corrupted path is rolled back,
neutralized, or quarantined without adopting a substituted target.

Classify failures as one of:

- `code-or-test-failure`;
- `resource-timeout`;
- `resource-collision`;
- `runner-or-toolchain-failure`;
- `cancelled-or-obsolete`.

Every class except a draft-only skip keeps `PR Gate` from succeeding.

## Security Controls

The repository is public, and GitHub recommends against attaching persistent
self-hosted runners to public repositories because pull requests can execute
dangerous code. This design accepts that residual risk and narrows the exposed
boundary as follows:

- dedicated account with no personal files, SSH login, sudo, or Docker access;
- no repository or environment secret in pull-request jobs;
- workflow permissions limited to `contents: read` and required Actions read
  access;
- `actions/checkout` uses `persist-credentials: false`;
- GitHub-owned Actions are pinned to full commit SHAs;
- repository Actions policy permits only repository-local and required
  GitHub-owned Actions;
- all external contributors require workflow approval;
- only the trusted reusable workflow can use the NPU runner group;
- trusted control code and the untrusted merge checkout are separate;
- the GitHub token uses the one-shot private-FD/final-`exec` handoff and is not
  inherited by pull-request-controlled processes;
- CANN and control scripts are read-only;
- host build, NPU artifact, runner workspace, and step-summary operations
  enforce their fixed real-directory boundaries and reject symbolic-link
  redirection;
- clean workspace and process cleanup after every job;
- no mutable cross-PR build cache;
- no use of `pull_request_target` for untrusted code.

Create an organization team named `ci-maintainers` with `Kur0x` and `chaowick`
as team maintainers. Grant `@LingquLab/ci-maintainers` write access only to the
TileXR repository, then assign that team in `.github/CODEOWNERS` to:

- `.github/CODEOWNERS`;
- `.github/workflows/`;
- `.github/actions/` if repository-local Actions are added;
- `scripts/ci/control/` and `scripts/ci/provision/`;
- `scripts/cann_download_install.sh` and `scripts/cann_local_install.sh` because
  provisioning executes them while installing the sealed toolchain;
- `.gitmodules`.

The existing ruleset's code-owner requirement then protects changes to the
workflow and runner boundary. Either team member can review a CI change authored
by the other; authors cannot approve their own pull requests. The current
two-approval requirement remains in force.

These controls reduce persistence and privilege but do not make arbitrary
pull-request execution on a shared host risk-free. That limitation is accepted
for this design.

In particular, build systems execute pull-request-controlled code, the runner
and tests use the same operating-system account, and that account can access
Ascend devices. A deliberately malicious approved pull request could interfere
with same-user processes or attempt to bypass the advisory queue even though it
cannot modify the sealed controller or CANN toolchain. The queue and cleanup
mechanisms enforce normal CI behavior; they are not a security sandbox.
Maintainer approval of an external-fork workflow is therefore an explicit
authorization to execute that reviewed commit on the shared host.

## GitHub Repository Settings

After the workflow trial succeeds:

1. Keep default workflow permissions read-only and keep workflows unable to
   approve pull requests.
2. Change fork workflow approval from first-time contributors to all external
   contributors.
3. Limit allowed Actions to the repository-local and GitHub-owned Actions used
   by this design.
4. Require full-SHA pinning for referenced Actions.
5. Add `PR Gate` as a required status check to the active default-branch
   ruleset.
6. Require the pull-request branch to be up to date with `main` before `PR Gate`
   can satisfy the rule.
7. Keep the existing two approvals, code-owner review, resolved-thread,
   no-bypass, deletion, and non-fast-forward rules.

The required check should accept only the GitHub Actions source. Its job name
must remain stable as `PR Gate` because required status checks match job names,
not workflow event types.

## Rollout

1. Create `ci-maintainers`, add `Kur0x` and `chaowick` as its only team
   maintainers, grant the team write access only to TileXR, and verify the
   membership and repository scope.
2. Add CODEOWNERS, workflows, and CI scripts through a normally reviewed pull
   request. Do not require `PR Gate` yet.
3. Restrict the Actions policy and runner group after the trusted reusable
   workflow exists on `main`.
4. Create and validate the `tilexr-ci` account and filesystem layout.
5. Install and seal CANN 9.1, its 910B Ops package, and the matching trusted
   control package.
6. Register the runner in the restricted runner group.
7. Run a trial pull request through the complete successful path.
8. Exercise a host-check failure and confirm that `blue` is not scheduled.
9. Exercise a hardware-test failure and confirm logs and cleanup.
10. Exercise cancellation by updating a pull request while it is waiting or
   running.
11. Queue two pull requests and confirm that only one NPU job runs at a time.
12. Exercise the resource-wait logic with a shortened validation timeout and
    confirm that foreign processes are never terminated.
13. Verify the external-fork approval policy through repository settings and a
    controlled fork workflow run.
14. Confirm that `PR Gate` appears on a real pull request, then add it to the
    ruleset as a strict required check.
15. Confirm that a failing or pending gate blocks merge and a successful,
    up-to-date gate permits merge.

## Rollback

If the gate prevents normal development because of runner or infrastructure
problems:

1. Remove only the `PR Gate` required status check from the ruleset.
2. Preserve the existing review and branch-update protections.
3. Stop or disable the self-hosted runner service.
4. Keep the account and sealed toolchain for diagnosis unless a separate
   cleanup is explicitly requested.

The workflows can remain in the repository while the runner is disabled; they
do not receive secrets or deployment authority.

## Acceptance Criteria

- A same-repository, non-draft pull request automatically starts host checks.
- An external-fork pull request cannot run until a maintainer approves it.
- A host-check failure does not schedule an NPU job.
- The NPU job builds the exact GitHub merge commit with CANN 9.1.
- Only one NPU job runs at a time.
- The job waits until all eight 910B3 devices are continuously idle for 60
  seconds, for at most six hours.
- The job never terminates or releases a foreign NPU process.
- All required host, build, eight-rank memory, EP, collective, and per-device
  SDMA tests pass on the positive trial.
- A5 / Ascend950 UDMA and multi-host exclusions are visible in the report and
  are not represented as successful runtime coverage.
- Cancellation and failure leave no NPU process owned by `tilexr-ci`.
- Logs and test reports are uploaded and retained for 14 days.
- `main` cannot be updated through a pull request while `PR Gate` is pending or
  failing.
- An up-to-date pull request with two approvals, the required code-owner
  approval, resolved discussions, and a successful `PR Gate` can be merged.
