# TileXR CI Operations

This runbook covers the pull-request gate for `LingquLab/TileXR`. Repository
changes alone do not activate the gate: the maintainer team, Actions policy,
runner group, `blue` installation, trial runs, and required check are separate
operator steps. Do not make `PR Gate` required until a real successful check
run has supplied its GitHub Actions integration ID.

## Workflow and trust model

`.github/workflows/pr-ci.yml` runs for `pull_request` events targeting `main`.
It runs `Host Checks` in an Ubuntu 20.04 container, calls the copy of
`.github/workflows/npu-ci.yml` on `main` for a ready, non-draft pull request,
and reports the stable aggregate check `PR Gate`. A draft runs only host checks.
A newer commit cancels the older run for the same pull request.

The reusable NPU workflow is the only workflow allowed to use the
`TileXR-NPU` organization runner group. It checks out
`refs/pull/NUMBER/merge` below the runner workspace and verifies its SHA before
the administrator-owned controller runs. The merge checkout is untrusted; the
controller and CANN toolchain are sealed, read-only installations outside it.
The check result belongs to that merge commit; a newer commit cancels the old
run and must produce its own successful `PR Gate` result.

Workflow permissions are `contents: read`. Checkout credentials are not
persisted, no repository or environment secret is exposed, and
`pull_request_target` is not used. The workflow does not pass `github.token` to
the controller. Untrusted child phases do not inherit Actions runtime or OIDC
tokens, or GitHub command-file paths. Before any pull-request code runs, the
Linux controller makes itself non-dumpable so same-account child processes
cannot read the controller's inherited environment through `/proc`.

This is defense in depth, not a sandbox. Pull-request code is built and run as
`tilexr-ci` with NPU access. Approval of an external-fork workflow explicitly
authorizes the reviewed merge commit to execute on the shared host.

## Blue layout

| Purpose | Account, group, or path |
| --- | --- |
| Service account | `tilexr-ci`, primary group `tilexr-ci`, device group `HwHiAiUser` |
| Home | `/home/tilexr-ci` |
| Sealed CANN 9.1 | `/home/tilexr-ci/toolchains/cann/9.1.0` |
| Sealed Bisheng compiler | `/home/tilexr-ci/toolchains/cann/9.1.0/cann/tools/bisheng_compiler/bin/bisheng` |
| Sealed controller | `/home/tilexr-ci/control/v1` |
| Active controller link | `/home/tilexr-ci/control/current` |
| Actions runner | `/home/tilexr-ci/actions-runner` |
| Runner workspace root | `/home/tilexr-ci/actions-runner/_work` |
| Fixed repository workspace | `/home/tilexr-ci/actions-runner/_work/TileXR/TileXR` |
| Provisioning staging | `/home/tilexr-ci/install-work` |

The account has a locked password, `nologin` shell, no SSH key, no sudo rule,
and no Docker membership. The job-completed hook is
`/home/tilexr-ci/control/current/job_completed.sh`; its path is stored in the
root-owned runner `.env`. The hook removes children of the fixed repository
workspace only after Actions post-job steps complete. It refuses symbolic-link
redirection and never removes the CANN tree.

Host checks use `.ci-build/tilexr-host-default` by default. A
`TILEXR_CI_BUILD_ROOT` override is accepted only as an appropriately named
direct child of the real repository build parent or the real `RUNNER_TEMP`.
The NPU workflow uses `${RUNNER_TEMP}/tilexr-ci-scratch` for temporary build and
test logs. Neither workflow uploads artifacts. The trusted controller reports
its result through its exit code and the Actions log; it never writes a GitHub
step summary.

## Time and resource policy

The single self-hosted runner provides the cross-PR queue. The build runs before
waiting for NPUs and has a two-hour timeout. Any NPU process owned by
`tilexr-ci` during build is a policy violation. Idle waiting has a six-hour
budget. Devices 0 through 7 must be healthy and process-free for two consecutive
samples 60 seconds apart.

Hardware execution has a two-hour total timeout; each multi-rank launch is
bounded to ten minutes. A foreign NPU process observed after acquisition stops
only the tracked CI process tree and reports a collision. CI never calls
`npu-smi release` and never signals another account's process. The complete NPU
job has an 11-hour Actions timeout to cover build, queueing, hardware, and
cleanup.

## Test matrix

`Host Checks` runs these eight recorded cases:

- shell syntax for tracked CI and affected test scripts;
- the complete standalone `tests/ci` CTest suite;
- comm logging, spdlog compile, and source-guard binaries;
- all four EP source-only tests;
- both data-as-flag tests;
- collectives vLLM patch tests;
- collectives vLLM integration-source tests;
- collectives profile-report tests.

On `blue`, the build manifest performs a clean CANN 9.1 top-level configure,
build, install, and CTest run; builds the comm, UDMA, SDMA, EP, and memory test
trees; requires the hardware demo binaries; and validates installed headers,
libraries, dependencies, and RPATH/RUNPATH. Stub `libascend_hal.so` resolution
through CANN `devlib` is a failure.

The build phase logs these named cases:

- top level: `top-level-ctest`;
- comm: `comm-log`, `comm-spdlog-compile`, and `comm-source-guards`;
- UDMA: `udma-transport-layout`, `udma-registry`, `udma-demo-sources`, and
  `udma-source-guard`;
- SDMA: `sdma-metadata`, `sdma-api-invalid`, `sdma-transport-disabled`,
  `sdma-comm-wiring`, `sdma-source-guard`, and `sdma-header-compile`;
- EP: `ep-layout`, `ep-api-sources`, `ep-kernel-sources`, and
  `ep-host-validation`;
- memory: `memory-demo-sources`.

The top-level configure/build, suite builds, top-level reinstall, required-file
checks, and dependency/RPATH checks are also mandatory. They fail the build
phase directly.

The hardware manifest runs:

- health checks for devices 0 through 7;
- single-rank and eight-rank UDMA compatibility/fallback tests on 910B3;
- SDMA-disabled communicator validation;
- the eight-rank peer-memory DataCopy demo;
- the eight-rank EP dispatch/combine demo;
- eight-rank AllGather, AllToAll, AllReduce, ReduceScatter, and Broadcast
  correctness, with Broadcast roots 0 and 7;
- checked multi-size performance smoke for AllGather, AllToAll, AllReduce,
  ReduceScatter, and Broadcast, from 4 bytes through 1 MiB;
- the SDMA demo on every device for 64 bytes, 4096 bytes, and 1 MiB.

Explicitly out of scope are A5 / Ascend950 UDMA data-plane validation,
multi-host collectives and EP, vLLM model inference, and performance-regression
thresholds. A 910B3 UDMA fallback pass must not be reported as UDMA data-plane
coverage.

## Results and failures

Host and NPU results are available in the Actions log. `Host Checks` also writes
a short, non-authoritative GitHub step summary. No CI artifacts are uploaded.

Failure classes and controller exit codes are:

- `code-or-test-failure` (`20`): a build, test, or required target failed;
- `resource-timeout` (`21`): the six-hour NPU idle budget expired;
- `resource-collision` (`22`): CI touched an NPU during build, a foreign process
  appeared during hardware, or another resource-ownership policy was violated;
- `runner-or-toolchain-failure` (`23`): controller, CANN, runner, host-health
  infrastructure, or final cleanup failed, including a
  leftover NPU process owned by `tilexr-ci`;
- `cancelled-or-obsolete` (`130`): Actions cancelled the run or the checked-out
  merge SHA did not match the event.

Resolve code failures in the pull request. For resource or infrastructure
failures, inspect the Actions log, confirm all NPUs are healthy and no CI
process remains, then rerun the failed job. Do not retry a resource collision
until the competing workload is understood.

## Local repository validation

Run from a clean TileXR checkout:

```bash
bash scripts/ci/host_checks.sh
ruby tests/ci/test_workflows.rb
bash -n scripts/ci/host_checks.sh scripts/ci/control/*.sh scripts/ci/provision/*.sh
git diff --check
git status --short
```

`gate.py` requires Linux `prctl`, `/proc`, pidfd, and subreaper
semantics for live orchestration. Its unit tests contain macOS-compatible
fakes, but stock macOS `/bin/bash` 3.2 lacks `wait -n`; the collectives launcher
source test therefore cannot complete there. Run the complete entrypoint in
the documented Ubuntu 20.04 container. Local macOS execution is not a
substitute for Ubuntu host checks or the real `blue` acceptance and hardware
runs.

## Provision and verify blue

Provision only from a fresh, commit-addressed checkout of reviewed `main`.
File transfer to `blue`, when needed, must use mutagen; the normal bootstrap
below initializes the public repository and fetches the resolved commit
directly on the server.

```bash
CI_SHA="$(gh api repos/LingquLab/TileXR/commits/main --jq .sha)"
BOOTSTRAP="/home/d00520898/tilexr-ci-bootstrap-${CI_SHA}"
ssh blue "test ! -e '${BOOTSTRAP}'"
ssh blue "git init '${BOOTSTRAP}'"
ssh blue "git -C '${BOOTSTRAP}' remote add origin https://github.com/LingquLab/TileXR.git"
ssh blue "git -C '${BOOTSTRAP}' fetch --depth 1 origin '${CI_SHA}'"
ssh blue "git -C '${BOOTSTRAP}' checkout --detach '${CI_SHA}'"
ssh blue "test \"\$(git -C '${BOOTSTRAP}' rev-parse HEAD)\" = '${CI_SHA}'"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/account.sh"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/cann.sh"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/control.sh"
gh api --method POST orgs/LingquLab/actions/runners/registration-token --jq .token | \
  ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/runner.sh"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/verify.sh"
```

The provisioning scripts are idempotent, accept `--dry-run`, and require root
for live changes. `runner.sh` reads the short-lived registration token from
standard input, passes it only through `ACTIONS_RUNNER_INPUT_TOKEN` while the
runner configuration process is isolated, and does not log it or place it in
process arguments.

## GitHub policy verification

The operator must be an active `LingquLab` organization administrator. Verify
the repository policy, team boundary, restricted runner group, runner, and
ruleset without changing them:

```bash
gh api user/memberships/orgs/LingquLab --jq '[.state,.role] | @tsv'
gh api repos/LingquLab/TileXR/actions/permissions
gh api repos/LingquLab/TileXR/actions/permissions/selected-actions
gh api repos/LingquLab/TileXR/actions/permissions/workflow
gh api repos/LingquLab/TileXR/actions/permissions/fork-pr-contributor-approval

diff -u \
  <(printf '%s\n' Kur0x chaowick | sort -f) \
  <(gh api --paginate orgs/LingquLab/teams/ci-maintainers/members \
      --jq '.[].login' | sort -f)
for login in Kur0x chaowick; do
  [[ "$(gh api \
    "orgs/LingquLab/teams/ci-maintainers/memberships/${login}" \
    --jq '[.state,.role] | @tsv')" == $'active\tmaintainer' ]]
done
diff -u \
  <(printf '%s\n' LingquLab/TileXR) \
  <(gh api --paginate orgs/LingquLab/teams/ci-maintainers/repos \
      --jq '.[].full_name' | sort)
[[ "$(gh api repos/LingquLab/TileXR/teams --jq \
  '.[] | select(.slug == "ci-maintainers") | .permission')" == push ]]

GROUP_ID="$(gh api orgs/LingquLab/actions/runner-groups --jq \
  '.runner_groups[] | select(.name == "TileXR-NPU") | .id')"
[[ "${GROUP_ID}" =~ ^[0-9]+$ ]]
gh api "orgs/LingquLab/actions/runner-groups/${GROUP_ID}" --jq \
  '{name,visibility,allows_public_repositories,restricted_to_workflows,selected_workflows}'
diff -u \
  <(printf '%s\n' LingquLab/TileXR) \
  <(gh api --paginate \
      "orgs/LingquLab/actions/runner-groups/${GROUP_ID}/repositories" \
      --jq '.repositories[].full_name' | sort)
diff -u \
  <(printf '%s\n' blue-tilexr-npu8) \
  <(gh api --paginate \
      "orgs/LingquLab/actions/runner-groups/${GROUP_ID}/runners" \
      --jq '.runners[].name' | sort)
gh api "orgs/LingquLab/actions/runner-groups/${GROUP_ID}/runners" --jq \
  '.runners[] | select(.name == "blue-tilexr-npu8") | {status,busy,labels:[.labels[].name]}'
gh api repos/LingquLab/TileXR/rulesets
```

Expected policy is selected Actions with full-SHA pinning; only repository
local Actions and `actions/checkout@*`; read-only
workflow tokens; no workflow PR approval; and approval required for all
external contributors. `TileXR-NPU` must allow this public repository but be
selected-repository only, contain only TileXR, and be restricted exactly to
`LingquLab/TileXR/.github/workflows/npu-ci.yml@refs/heads/main`.

## Enable the required check

Do this only after a successful trial pull request. First resolve `PR Gate`
from the trial merge commit and generate a candidate that preserves every
existing rule and every non-`PR Gate` required check. It removes any stale or
duplicate `PR Gate` entries, then adds exactly one entry bound to the observed
GitHub Actions App:

```bash
PR_NUMBER="$(gh pr list --repo LingquLab/TileXR --state all \
  --head ci/pr-gate-positive --json number --jq '.[0].number')"
MERGE_SHA="$(gh api "repos/LingquLab/TileXR/pulls/${PR_NUMBER}" --jq .merge_commit_sha)"
PR_GATE_APP_ID_JQ='
  [.check_runs[]
    | select(
        .name == "PR Gate"
        and .conclusion == "success"
        and .app.slug == "github-actions"
      )
    | .app.id] as $ids
  | if ($ids | length) == 1 and ($ids[0] | type) == "number" then
      $ids[0]
    else
      error("expected exactly one successful GitHub Actions PR Gate check run")
    end
'
ACTIONS_APP_ID="$(gh api "repos/LingquLab/TileXR/commits/${MERGE_SHA}/check-runs" --jq \
  "${PR_GATE_APP_ID_JQ}")"
[[ "${ACTIONS_APP_ID}" =~ ^[0-9]+$ ]]

RULESET_ID="$(gh api repos/LingquLab/TileXR/rulesets --jq \
  '.[] | select(.name == "master" and .target == "branch") | .id')"
[[ "${RULESET_ID}" =~ ^[0-9]+$ ]]
RULESET_BEFORE="$(mktemp)"
RULESET_PAYLOAD="$(mktemp)"
RULESET_CURRENT="$(mktemp)"
trap 'rm -f "${RULESET_BEFORE}" "${RULESET_PAYLOAD}" "${RULESET_CURRENT}"' EXIT
gh api "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" >"${RULESET_BEFORE}"
jq --argjson app_id "${ACTIONS_APP_ID}" '
  def pr_gate: {context: "PR Gate", integration_id: $app_id};
  ([.rules[] | select(.type == "required_status_checks")] | length) as $count
  | if $count > 1 then
      error("multiple required_status_checks rules require manual resolution")
    else
      {
        name: .name,
        target: .target,
        enforcement: .enforcement,
        bypass_actors: (.bypass_actors // []),
        conditions: .conditions,
        rules: (
          if $count == 0 then
            .rules + [{
              type: "required_status_checks",
              parameters: {
                do_not_enforce_on_create: false,
                required_status_checks: [pr_gate],
                strict_required_status_checks_policy: true
              }
            }]
          else
            [.rules[]
              | if .type == "required_status_checks" then
                  .parameters.required_status_checks = (
                    ((.parameters.required_status_checks // [])
                      | map(select(.context != "PR Gate"))) + [pr_gate]
                  )
                  | .parameters.do_not_enforce_on_create = false
                  | .parameters.strict_required_status_checks_policy = true
                else . end]
          end
        )
      }
    end
' "${RULESET_BEFORE}" >"${RULESET_PAYLOAD}"
diff -u \
  <(jq -S '{name,target,enforcement,bypass_actors,conditions,rules}' "${RULESET_BEFORE}") \
  <(jq -S . "${RULESET_PAYLOAD}")
```

Stop here for explicit maintainer review. The diff may change only the strict
and create-time status-check policy fields and the deduplicated `PR Gate`
entry; every other rule and required check must remain byte-for-byte equivalent
in normalized JSON. Do not apply a candidate that contains any other change.

Only after that review, run the mutation as a separate subsequent command. It
first proves that the live ruleset has not changed since the candidate was
created:

```bash
gh api "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" >"${RULESET_CURRENT}"
if cmp -s "${RULESET_BEFORE}" "${RULESET_CURRENT}"; then
  gh api --method PUT "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" \
    --input "${RULESET_PAYLOAD}" >/dev/null
else
  echo "ERROR: live ruleset changed after candidate review; regenerate it" >&2
  false
fi
```

Read the ruleset back and verify that it still requires two approvals,
code-owner review, resolved threads, deletion and non-fast-forward protections,
has empty bypass actors, and preserves every unrelated rule and required check.
Verify `PR Gate` occurs exactly once, is bound to the GitHub Actions App, and
strict up-to-date checking is true.

## Runner upgrade

The runner is configured with `--disableupdate`; upgrade it deliberately when
`blue-tilexr-npu8` is online, not busy, and has no active workflow. Use a fresh
reviewed `main` bootstrap, stop the service, and rerun `runner.sh`. It downloads
the latest stable ARM64 release from the official API and verifies the asset's
published SHA-256 digest before re-registering.

```bash
CI_SHA="$(gh api repos/LingquLab/TileXR/commits/main --jq .sha)"
BOOTSTRAP="/home/d00520898/tilexr-ci-bootstrap-${CI_SHA}"
ssh blue "test ! -e '${BOOTSTRAP}'"
ssh blue "git init '${BOOTSTRAP}'"
ssh blue "git -C '${BOOTSTRAP}' remote add origin https://github.com/LingquLab/TileXR.git"
ssh blue "git -C '${BOOTSTRAP}' fetch --depth 1 origin '${CI_SHA}'"
ssh blue "git -C '${BOOTSTRAP}' checkout --detach '${CI_SHA}'"
ssh blue "test \"\$(git -C '${BOOTSTRAP}' rev-parse HEAD)\" = '${CI_SHA}'"
ssh blue 'sudo /home/tilexr-ci/actions-runner/svc.sh stop'
gh api --method POST orgs/LingquLab/actions/runners/registration-token --jq .token | \
  ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/runner.sh"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/verify.sh"
gh api orgs/LingquLab/actions/runners --jq \
  '.runners[] | select(.name == "blue-tilexr-npu8") | {status,busy,os,labels:[.labels[].name]}'
```

Record the previous and new runner versions and the verification output in the
operations record. Do not upgrade during a queued or running NPU gate.

## Rollback

If CI infrastructure blocks normal development, remove only `PR Gate` from the
active ruleset. The following transform preserves unrelated status checks and
all other rules. Inspect the diff before applying it.

```bash
RULESET_ID="$(gh api repos/LingquLab/TileXR/rulesets --jq \
  '.[] | select(.name == "master" and .target == "branch") | .id')"
RULESET_BEFORE="$(mktemp)"
RULESET_PAYLOAD="$(mktemp)"
RULESET_CURRENT="$(mktemp)"
trap 'rm -f "${RULESET_BEFORE}" "${RULESET_PAYLOAD}" "${RULESET_CURRENT}"' EXIT
gh api "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" >"${RULESET_BEFORE}"
jq '{
  name: .name,
  target: .target,
  enforcement: .enforcement,
  bypass_actors: (.bypass_actors // []),
  conditions: .conditions,
  rules: [
    .rules[]
    | if .type == "required_status_checks" then
        .parameters.required_status_checks |= map(select(.context != "PR Gate"))
      else . end
    | select(.type != "required_status_checks" or
        (.parameters.required_status_checks | length) > 0)
  ]
}' "${RULESET_BEFORE}" >"${RULESET_PAYLOAD}"
diff -u \
  <(jq -S '{name,target,enforcement,bypass_actors,conditions,rules}' "${RULESET_BEFORE}") \
  <(jq -S . "${RULESET_PAYLOAD}")
```

Stop for explicit review. After confirming that the candidate removes only
`PR Gate`, apply it separately and only if the live ruleset is unchanged:

```bash
gh api "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" >"${RULESET_CURRENT}"
if cmp -s "${RULESET_BEFORE}" "${RULESET_CURRENT}"; then
  gh api --method PUT "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" \
    --input "${RULESET_PAYLOAD}" >/dev/null
else
  echo "ERROR: live ruleset changed after rollback review; regenerate it" >&2
  false
fi
```

Then stop and disable the runner while preserving the account, sealed
controller, and CANN toolchain:

```bash
ssh blue 'service="$(sudo cat /home/tilexr-ci/actions-runner/.service)"; \
  sudo systemctl disable --now "${service}"'
```

Do not delete `/home/tilexr-ci/toolchains/cann/9.1.0` as part of an incident
rollback. The workflows may remain in the repository; they hold no deployment
authority or persistent secret. Re-enable the service and rerun
`scripts/ci/provision/verify.sh` only after the cause is resolved.
