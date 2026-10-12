"""Download and validate an accepted Actions artifact set in an unprivileged job."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from scripts.release_artifacts import (
    GATES,
    PLATFORMS,
    REPOSITORY,
    git,
    require,
    trusted_identity,
    validate_gates,
    validate_manifest,
    verify_files,
    version,
)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the GitHub token to artifact storage."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        """Redirect securely to signed artifact storage."""
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        require(urllib.parse.urlparse(newurl).scheme == "https", "non-HTTPS redirect")
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    """Read repository-scoped API data; no remote mutations."""

    def __init__(self, token: str) -> None:
        """Configure an authenticated, read-only client."""
        self.token = token
        self.opener = urllib.request.build_opener(SafeRedirect())

    def request(self, path: str) -> Any:
        """Open a repository API response."""
        require(path.startswith("/") and "://" not in path, "invalid API path")
        request = urllib.request.Request(
            "https://api.github.com" + path,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        return self.opener.open(request, timeout=120)

    def get(self, path: str) -> Any:
        """Decode one JSON API response."""
        with self.request(path) as response:
            return json.load(response)

    def pages(self, path: str, key: str) -> list[dict[str, Any]]:
        """Read all pages, rejecting a truncated response."""
        result = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            data = self.get(f"{path}{separator}per_page=100&page={page}")[key]
            result.extend(data)
            if len(data) < 100:
                return result
        raise ValueError("API pagination limit; narrow the selected evidence")

    def download(self, artifact: dict[str, Any], output: Path) -> None:
        """Download an exact artifact ID and unpack only flat data files."""
        require(not artifact["expired"], "expired artifact")
        output.mkdir(parents=True, exist_ok=False)
        temporary = output.parent / f"{artifact['id']}.zip"
        with (
            self.request(
                f"/repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"
            ) as source,
            temporary.open("wb") as target,
        ):
            total = 0
            while block := source.read(1024 * 1024):
                total += len(block)
                require(total <= 512 * 1024 * 1024, "artifact size limit")
                target.write(block)
        with zipfile.ZipFile(temporary) as archive:
            seen = set()
            require(
                sum(m.file_size for m in archive.infolist()) <= 512 * 1024 * 1024,
                "expanded artifact size limit",
            )
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                # Every expected artifact is flat. Reject extra files, symlinks,
                # traversal and any executable from an uploaded artifact.
                require(
                    "/" not in name and "\\" not in name and name not in seen,
                    "unsafe/duplicate artifact path",
                )
                require(
                    (member.external_attr >> 16) & 0o170000 != 0o120000,
                    "artifact symlink",
                )
                require(
                    name in {"receipt.json", "release-manifest.json"}
                    or name.endswith((".whl", ".tar.gz")),
                    "unexpected artifact file",
                )
                seen.add(name)
                (output / name).write_bytes(archive.read(member))
        temporary.unlink()


def check_run(api: GitHub, run: dict[str, Any], workflow: str, attempt: int) -> None:
    """Reject a wrong workflow, source, attempt or unsuccessful job."""
    require(run["repository"]["full_name"] == REPOSITORY, "wrong run repository")
    require(run["head_repository"]["full_name"] == REPOSITORY, "fork run")
    require(run["path"] == f".github/workflows/{workflow}.yml", "wrong build workflow")
    require(
        run["run_attempt"] == attempt,
        "mismatched attempt; select the current completed attempt",
    )
    require(
        run["status"] == "completed" and run["conclusion"] == "success",
        "incomplete/failed run",
    )
    if run["event"] == "workflow_dispatch":
        require(run["head_branch"] == "master", "untrusted manual source")
    else:
        require(
            run["event"] == "pull_request" and len(run["pull_requests"]) == 1,
            "untrusted run event",
        )
        pr = api.get(f"/repos/{REPOSITORY}/pulls/{run['pull_requests'][0]['number']}")
        require(
            pr["head"]["repo"]["full_name"] == REPOSITORY
            and pr["base"]["repo"]["full_name"] == REPOSITORY
            and pr["base"]["ref"] == "master",
            "untrusted PR",
        )
    jobs = api.pages(
        f"/repos/{REPOSITORY}/actions/runs/{run['id']}/attempts/{attempt}/jobs", "jobs"
    )
    require(
        bool(jobs) and all(j["conclusion"] == "success" for j in jobs),
        "failed/cancelled/skipped required job",
    )


def artifacts_for(
    api: GitHub, run_id: int, names: set[str]
) -> dict[str, dict[str, Any]]:
    """Resolve exact artifact IDs, never a latest artifact by name."""
    artifacts = api.pages(
        f"/repos/{REPOSITORY}/actions/runs/{run_id}/artifacts", "artifacts"
    )
    selected = {}
    for name in names:
        matches = [a for a in artifacts if a["name"] == name]
        require(
            len(matches) == 1 and not matches[0]["expired"],
            f"missing/ambiguous/expired artifact {name}",
        )
        artifact = matches[0]
        require(artifact["workflow_run"]["id"] == run_id, "wrong artifact run")
        selected[name] = artifact
    return selected


def check_receipt_run(
    api: GitHub, receipt: dict[str, Any], run: dict[str, Any]
) -> None:
    """Bind uploaded identity to the actual API run and Git tree."""
    trusted_identity(receipt)
    require(
        receipt["run_id"] == run["id"]
        and receipt["attempt"] == run["run_attempt"]
        and receipt["event"] == run["event"],
        "receipt run mismatch",
    )
    commit = api.get(f"/repos/{REPOSITORY}/git/commits/{receipt['commit']}")
    require(commit["tree"]["sha"] == receipt["tree"], "recorded Git tree mismatch")
    if receipt["event"] == "pull_request":
        pr = run["pull_requests"][0]
        require(receipt["pr"]["number"] == pr["number"], "PR association mismatch")
        # PR associations may expose live head/base fields after merge. Bind the
        # event receipt to the immutable run SHA and the actual merge parents,
        # rather than requiring a mutable PR base to remain at its old SHA.
        if receipt["commit"] != receipt["pr"]["head"]:
            parents = {parent["sha"] for parent in commit["parents"]}
            require(
                {receipt["pr"]["head"], receipt["pr"]["base"]} <= parents,
                "recorded PR head/base do not match checkout merge parents",
            )
        require(
            run["head_sha"] in {receipt["pr"]["head"], receipt["commit"]},
            "wrong PR run SHA",
        )
    else:
        require(receipt["commit"] == run["head_sha"], "manual run commit mismatch")


def prepare(
    api: GitHub, run_id: int, attempt: int, tag: str, directory: Path
) -> dict[str, Any]:
    """Validate tag, all required workflows and every accepted distribution."""
    require(
        os.environ.get("GITHUB_REPOSITORY", REPOSITORY) == REPOSITORY,
        "repository restriction",
    )
    require(
        os.environ.get("GITHUB_REF", "refs/heads/master") == "refs/heads/master",
        "publication must run from master",
    )
    require(re.fullmatch(r"v\d+\.\d+\.\d+", tag) is not None, "invalid release tag")
    release = tag[1:]
    require(release == version(), "tag/metadata version mismatch")
    tag_commit = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    subprocess.run(  # noqa: S603 - validated Git identity
        ["git", "merge-base", "--is-ancestor", tag_commit, "origin/master"],  # noqa: S607
        check=True,
    )
    tree = git("rev-parse", f"{tag_commit}^{{tree}}")
    run = api.get(f"/repos/{REPOSITORY}/actions/runs/{run_id}")
    check_run(api, run, "build", attempt)
    names = {
        f"release-sdist-a{attempt}",
        f"release-manifest-a{attempt}",
        *[f"release-wheels-{p}-a{attempt}" for p in PLATFORMS],
    }
    artifacts = artifacts_for(api, run_id, names)
    directory.mkdir(parents=True, exist_ok=False)
    manifest_name = f"release-manifest-a{attempt}"
    api.download(artifacts[manifest_name], directory / manifest_name)
    manifest = json.loads(
        (directory / manifest_name / "release-manifest.json").read_text()
    )
    validate_manifest(
        manifest, tree=tree, run_id=run_id, attempt=attempt, release=release
    )
    check_receipt_run(api, manifest, run)
    gates = {}
    gate_identities = {}
    for name in GATES:
        runs = api.pages(
            f"/repos/{REPOSITORY}/actions/workflows/{name}.yml/runs?head_sha={run['head_sha']}&status=success&event={run['event']}",
            "workflow_runs",
        )
        errors = []
        for candidate in runs:
            try:
                check_run(api, candidate, name, candidate["run_attempt"])
                artifact_name = f"qualification-{name}-a{candidate['run_attempt']}"
                artifact = artifacts_for(api, candidate["id"], {artifact_name})[
                    artifact_name
                ]
                location = (
                    directory
                    / f"gate-{name}-{candidate['id']}-a{candidate['run_attempt']}"
                )
                api.download(artifact, location)
                receipt = json.loads((location / "receipt.json").read_text())
                check_receipt_run(api, receipt, candidate)
                # Validate all gates together below; match candidate content here.
                require(
                    receipt["tree"] == tree
                    and receipt["pr"] == manifest["pr"]
                    and receipt["event"] == manifest["event"],
                    "gate candidate mismatch",
                )
                gates[name] = receipt
                gate_identities[name] = {
                    "run_id": candidate["id"],
                    "attempt": candidate["run_attempt"],
                    "artifact_id": artifact["id"],
                    "url": candidate["html_url"],
                }
                break
            except (ValueError, urllib.error.HTTPError) as error:
                errors.append(str(error))
        require(name in gates, f"missing accepted {name} evidence: {errors}")
    validate_gates(manifest, gates)
    for name in sorted(names - {manifest_name}):
        api.download(artifacts[name], directory / name)
        receipt = json.loads((directory / name / "receipt.json").read_text())
        expected = (
            manifest
            if name.startswith("release-sdist")
            else next(
                manifest["platforms"][p]
                for p in PLATFORMS
                if name == f"release-wheels-{p}-a{attempt}"
            )
        )
        for key in (
            "tree",
            "commit",
            "run_id",
            "attempt",
            "version",
            "source_members",
            "checkout_members",
            "wheels",
            "sdist",
        ):
            if key in expected:
                require(receipt.get(key) == expected[key], "artifact receipt mismatch")
    verify_files(manifest, directory, directory / "publish")
    distributions = sorted(str(p) for p in (directory / "publish").iterdir())
    subprocess.run([sys.executable, "-m", "twine", "check", *distributions], check=True)  # noqa: S603 - validated metadata filenames
    summary = {
        "tag": tag,
        "tag_commit": tag_commit,
        "tree": tree,
        "build_run_id": run_id,
        "attempt": attempt,
        "build_url": run["html_url"],
        "artifact_ids": {n: a["id"] for n, a in artifacts.items()},
        "required_gates": gate_identities,
        "manifest": manifest,
        "gate_receipts": gates,
    }
    evidence = directory / "evidence"
    evidence.mkdir()
    (evidence / "publication-validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (evidence / "sha256sums.txt").write_text(
        "".join(
            f"{r['sha256']}  {r['filename']}\n"
            for r in [
                manifest["sdist"],
                *[w for p in PLATFORMS for w in manifest["platforms"][p]["wheels"]],
            ]
        )
    )
    print(
        json.dumps(
            {
                "validated_tree": tree,
                "build_run_id": run_id,
                "attempt": attempt,
                "tag": tag,
            }
        )
    )
    return summary


def main() -> None:
    """Validate a maintainer-selected completed Actions run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--directory", type=Path, default=Path("accepted"))
    args = parser.parse_args()
    require(args.run_id > 0 and args.attempt > 0, "positive run ID/attempt required")
    prepare(
        GitHub(os.environ["GH_TOKEN"]),
        args.run_id,
        args.attempt,
        args.tag,
        args.directory,
    )


if __name__ == "__main__":
    main()
