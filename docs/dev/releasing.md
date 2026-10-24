# Releasing Deepwave

Release builds qualify pull request changes and manual dispatches. A standalone
branch push does not run the required workflows. An unchanged squash merge or
release tag does not repeat the expensive builds. Publication is a separate
manual promotion of accepted artifacts; it never rebuilds them.

## Account setup (maintainer)

On PyPI, configure project `deepwave` with a Trusted Publisher for GitHub owner
`ar4`, repository `deepwave`, workflow `publish.yml`, environment `pypi`.
In GitHub Settings → Environments, create `pypi`, restrict deployments to
`master`, and require the maintainer's approval. Allow the sole maintainer to
approve their own initiated deployment: do not enable prevention of self review.
Configure TestPyPI separately if it is used; it is optional.

The publication workflow is restricted to `ar4/deepwave` and dispatch from
`master`. Only its final Linux upload job receives `id-token: write`. The
unprivileged validation job runs trusted master code, downloads explicit artifact
IDs, and verifies source identity, required checks, completeness and hashes. The
upload job checks out no source and executes no scripts from downloaded artifacts.
The official PyPI action is pinned to a full commit. Upload concurrency does not
cancel an upload already in progress, and `skip-existing` is not enabled.

The prepared 0.1.0 metadata omits the obsolete August placeholder date and
citation month. Its eventual tag/release records the publication date. If a date
is added to source metadata, do that before qualification, since it changes the
accepted tree and sdist. The changelog version section is undated.

## Push and review

From the clean release checkout, inspect the branch and its one-commit history:

```bash
git switch release/0.1.0
git status --short
git log --oneline origin/master..HEAD
git push origin release/0.1.0:refs/heads/release/0.1.0
gh pr create --base master --head release/0.1.0 --title 'Release Deepwave 0.1.0'
```

Never push private campaign refs, all branches, a mirror or all tags. Preserve
existing public attribution. Review the PR changes and check every required job
in Build and test, Tests, Docs, Lint and Sanitizers. The build's aggregate job
rejects failed, cancelled or skipped matrix work and missing artifacts. Review
conditional skips in test logs: CPU runners cannot execute CUDA cases, and
optional reference interpreter or compiler availability can cause explicit skips.
Mandatory workflow jobs themselves must all succeed. No commit-message test
bypass is supported.

Download and retain `release-manifest-a<ATTEMPT>`,
`release-sdist-a<ATTEMPT>`, and all three
`release-wheels-<PLATFORM>-a<ATTEMPT>` artifacts. Platforms are `ubuntu-latest`,
`windows-2022` and `macos-14`. Each wheel set contains exactly
one repaired `py3-none` wheel. The 95 MiB wheel limit is enforced. The sdist is
built once, then used by every release wheel builder. Matching public checkout
content is required for every shipped source member except generated `PKG-INFO`;
all tracked `src/` files and the build configuration must be present. Archive
hashes alone would also accept a self-consistent archive of stale sources.
Matching public checkout fixtures are supplied by cibuildwheel's
`test-sources`; the installed wheel
provides the imported Python package and native library. Qualification checks
every registered kernel symbol, including all dimensions, accuracies, dtypes
and variants. Linux and Windows release wheels must also contain the complete
CUDA symbol set. These checks need no GPU; they do not establish CUDA execution
correctness.

Record the successful build run ID and **current completed run attempt**, tested
Git tree, PR head/base and links to all required checks. Default PR checkout may
be a synthetic merge commit: its actual Git tree, rather than just the PR head
SHA, is the identity that must be promoted. Linux Actions performs CPU wheel
validation. Install the accepted Linux wheel for the planned Colab GPU tests,
using the matching source tests and preserving the wheel checksum and execution
reports. GPU execution of the repaired source is required separately from CPU
Actions results. Tests requiring two visible GPUs remain a separate qualification
item when the Colab runtime exposes one device.

## Merge, tag and promote

Use GitHub's **Squash and merge**, edit the message to a concise maintainer
release description, and retain linear public history. Local preparation has
already squashed the private campaign before the first push. A PR squash alone
would not remove previously uploaded private history.

After merge, verify the merged and tagged content equals the accepted manifest:

```bash
git fetch origin master
git switch master
git merge --ff-only origin/master
TESTED_TREE=$(python -c 'import json; print(json.load(open("release-manifest.json"))["tree"])')
test "$(git rev-parse HEAD^{tree})" = "$TESTED_TREE"
git tag -a v0.1.0 -m 'Deepwave 0.1.0'
test "$(git rev-parse v0.1.0^{tree})" = "$TESTED_TREE"
git push origin refs/tags/v0.1.0:refs/tags/v0.1.0
```

If master or PR base changes the content, update the candidate and obtain fresh
affected CI evidence. The validator requires all five workflows to attest to the
same tree and PR head/base identity; dispatch all five on master if using a
trusted manual qualification. Do not change tested content just to add result
links: save links in PR/release evidence and release assets. Commit SHAs may
change through squash or rebase while the tree remains identical.

In Actions → **Publish accepted distributions** → **Run workflow**, select
`master`, supply `build_run_id`, `build_attempt` and `release_tag` (`v0.1.0`).
Equivalent command:

```bash
gh workflow run publish.yml --ref master \
  -f build_run_id=ACCEPTED_RUN_ID -f build_attempt=ACCEPTED_ATTEMPT \
  -f release_tag=v0.1.0
```

The validator rejects untrusted fork PRs, arbitrary workflows, unsuccessful or
incomplete runs, stale attempts, expired/missing/ambiguous artifacts, mismatched
Git trees, failed/skipped gates, incomplete platforms, extra distributions,
wrong metadata versions and hashes. It resolves each artifact ID within the
selected run. Uploaded metadata is parsed as data; no downloaded code is run.
It runs `twine check` and saves `publication-validation.json` with run and
artifact identities, gate receipts, local GPU scope and checksums. Approve the
`pypi` environment deployment after reviewing this evidence.

After publication, compare PyPI's file SHA-256 digests with
`publication-evidence-a<ATTEMPT>/sha256sums.txt`, and verify a fresh installation:

```bash
python -m venv release-install-check
release-install-check/bin/python -m pip install deepwave==0.1.0
release-install-check/bin/python -c 'import deepwave; print(deepwave.__version__)'
release-install-check/bin/python scripts/platform_wheel_smoke.py
```

Use the equivalent `Scripts/python.exe` paths on Windows. Keep all accepted
distributions, the public qualification summary, manifest, CI links and
publication receipts as permanent release assets when creating the GitHub
Release. Actions retention is 90 days, subject to repository policy; artifacts
are not a permanent archive. External platform and workflow execution remain
pending until those future Actions runs succeed.

The Docs workflow validates and retains HTML; it does not deploy the public
website. Download `docs-html-a<ATTEMPT>` from the accepted Docs run recorded in
the publication evidence and publish it through the documentation host's
deployment process. Coordinate that update with the package release: the README
and package metadata link to `https://ausargeo.com/deepwave`, including the new
upgrade guide and API pages. Verify those entry points and the displayed version
before announcing the release. Retain the HTML with the release assets so the
documentation corresponding to the accepted code is available later.

## Manual upload fallback

If Trusted Publisher account setup is incomplete, validate and download the
**same accepted artifacts** from a fresh trusted master checkout (with the tag
and full master history available). This does not rebuild anything:

```bash
python -m pip install twine
export GH_TOKEN="$(gh auth token)"
python -m scripts.prepare_publication \
  --run-id ACCEPTED_RUN_ID --attempt ACCEPTED_ATTEMPT \
  --tag v0.1.0 --directory accepted-manual
python -m twine check accepted-manual/publish/*
python -m twine upload accepted-manual/publish/*
```

Inspect `accepted-manual/evidence/publication-validation.json` and its checksum
list before uploading. The helper applies the same checks and downloads only the
selected accepted sdist and three wheel sets. Never rebuild locally for this
fallback. Optional TestPyPI publication has its own credentials/configuration
and uses the accepted distributions; it is not another build or GPU campaign.

## Wheel test fixture inventory

The public test tree includes `tests` (ABI reference, ASCII physics fixtures and
pytest hooks), `scripts` (dialect/manifest helpers and platform native smoke),
`benchmarks` (legacy parity harness), `docs` (contracts inspected by tests),
`src/deepwave` (static Python inspection and complete native harness inputs),
`pyproject.toml` (strict marker configuration), the root README, changelog,
citation/contributor metadata, and lint/native format configuration. Source
fixtures sit under `src/`; this directory is never added to `PYTHONPATH`, and no
editable install is used. Tests importing `deepwave` resolve the installed wheel.
The dialect test creates its own temporary Git repository. No wheel tests are
excluded for missing checkout data. Research under `exploration` is excluded by
`testpaths = ["tests"]` and by the explicit sdist allowlist.

The Linux selection uses two bounded workers and excludes existing gradcheck,
backend-parity and compile selections. Windows and the Apple Silicon macOS job retain full
scalar tests, fast acoustic tests, every family's native smoke and Unicode
storage-path checks. Collection on Linux does not qualify Windows/macOS.

References: [cibuildwheel test sources](https://cibuildwheel.pypa.io/en/v3.3.0/options/#test-sources),
[PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/using-a-publisher/),
[GitHub merge methods](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/about-merge-methods-on-github).
