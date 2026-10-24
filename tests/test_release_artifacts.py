"""Artifact promotion rejects incomplete, untrusted and mismatched evidence."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from scripts import prepare_publication as publication
from scripts import release_artifacts as release


@pytest.mark.parametrize("actions", ["true", "false"])
def test_qualification_cli_reports_failed_prerequisites(
    tmp_path: Path, actions: str
) -> None:
    """A real CLI failure names every failed gate without hiding its traceback."""
    jobs = {
        "lint": {"result": "failure"},
        "build%windows\nsecond line": {"result": "cancelled"},
        "docs": {"result": "success"},
    }
    result = subprocess.run(  # noqa: S603 - fixed local CLI and generated metadata
        [
            sys.executable,
            release.__file__,
            "aggregate",
            "--directory",
            str(tmp_path),
            "--needs",
            json.dumps(jobs),
        ],
        env={**os.environ, "GITHUB_ACTIONS": actions},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1
    assert "ValueError: required jobs did not succeed:" in result.stderr
    assert "lint: failure" in result.stderr
    assert "build%windows\nsecond line: cancelled" in result.stderr
    assert "docs: success" not in result.stderr
    if actions == "true":
        assert result.stdout == (
            "::error title=Release qualification failed::"
            "required jobs did not succeed: lint: failure; "
            "build%25windows%0Asecond line: cancelled\n"
        )
    else:
        assert not result.stdout


def candidate(tmp_path: Path) -> dict[str, Any]:
    """Five real metadata archives with a matching internal PR receipt."""
    identity = {
        "repository": release.REPOSITORY,
        "kind": "release",
        "commit": "commit",
        "tree": "tree",
        "run_id": 123,
        "attempt": 2,
        "version": "0.1.0",
        "event": "pull_request",
        "ref": "refs/pull/7/merge",
        "head_sha": "merge",
        "pr": {
            "number": 7,
            "head": "head",
            "base": "base",
            "head_repository": release.REPOSITORY,
            "base_repository": release.REPOSITORY,
        },
        "checkout_members": {},
        "jobs": {
            "build_sdist": {"result": "success"},
            "build_wheels": {"result": "success"},
        },
    }
    data = b"Metadata-Version: 2.1\nName: deepwave\nVersion: 0.1.0\n"
    source_files = {
        "CMakeLists.txt": b"project(deepwave LANGUAGES C)\n",
        "pyproject.toml": b'[project]\nname = "deepwave"\n',
        "src/deepwave/__init__.py": b'__version__ = "0.1.0"\n',
        "src/deepwave/scalar.c": b"/* candidate native source */\n",
    }
    identity["checkout_members"] = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in source_files.items()
    }
    sdist = tmp_path / "deepwave-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, payload in {**source_files, "PKG-INFO": data}.items():
            member = tarfile.TarInfo(f"deepwave-0.1.0/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    identity["sdist"] = {
        "filename": sdist.name,
        "sha256": release.sha256(sdist),
        "bytes": sdist.stat().st_size,
    }
    identity["source_members"] = {
        **identity["checkout_members"],
        "PKG-INFO": hashlib.sha256(data).hexdigest(),
    }
    platforms = {}
    for platform, tag in zip(
        release.PLATFORMS,
        (
            "manylinux_2_28_x86_64",
            "win_amd64",
            "macosx_11_0_arm64",
        ),
    ):
        wheel = tmp_path / f"deepwave-0.1.0-py3-none-{tag}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("deepwave-0.1.0.dist-info/METADATA", data)
        receipt = copy.deepcopy(identity)
        receipt["kind"] = platform
        receipt["wheels"] = [
            {
                "filename": wheel.name,
                "sha256": release.sha256(wheel),
                "bytes": wheel.stat().st_size,
            }
        ]
        platforms[platform] = receipt
    identity["platforms"] = platforms
    return identity


def validate(manifest: dict[str, Any]) -> None:
    """The accepted run/tree/version are supplied independently."""
    release.validate_manifest(
        manifest, tree="tree", run_id=123, attempt=2, release="0.1.0"
    )


def test_complete_artifact_set(tmp_path: Path) -> None:
    """A complete set passes and preserves every distribution byte."""
    manifest = candidate(tmp_path)
    validate(manifest)
    output = tmp_path.parent / (tmp_path.name + "-publish")
    release.verify_files(manifest, tmp_path, output)
    assert len(list(output.iterdir())) == 4
    gates = {
        name: {**manifest, "kind": name, "jobs": {"gate": {"result": "success"}}}
        for name in release.GATES
    }
    release.validate_gates(manifest, gates)


@pytest.mark.parametrize("fault", ["changed", "missing", "untracked"])
def test_sdist_must_contain_reviewed_sources(tmp_path: Path, fault: str) -> None:
    """Self-consistent archive hashes cannot attest to different source code."""
    manifest = candidate(tmp_path)
    name = "src/deepwave/scalar.c"
    if fault == "changed":
        manifest["source_members"][name] = hashlib.sha256(b"stale code").hexdigest()
    elif fault == "missing":
        del manifest["source_members"][name]
    else:
        manifest["source_members"]["src/deepwave/unreviewed.py"] = "extra"
    with pytest.raises(ValueError, match=r"sdist.*checkout"):
        validate(manifest)


def test_sdist_need_not_ship_test_and_documentation_fixtures(tmp_path: Path) -> None:
    """Source identity does not undo the deliberate small-sdist policy."""
    manifest = candidate(tmp_path)
    manifest["checkout_members"].update(
        {"tests/test_wave.py": "test", "docs/a.rst": "doc"}
    )
    validate(manifest)


@pytest.mark.parametrize(
    "fault",
    [
        "tree",
        "run",
        "attempt",
        "version",
        "platform",
        "failed",
        "skipped",
        "repository",
        "fork",
        "manual_branch",
        "wheel_tag",
    ],
)
def test_manifest_negative_controls(tmp_path: Path, fault: str) -> None:
    """No wrong identity, omitted job or untrusted source can be promoted."""
    manifest = candidate(tmp_path)
    if fault in {"tree", "attempt", "version", "repository"}:
        manifest[fault] = "wrong"
    elif fault == "run":
        manifest["run_id"] = 456
    elif fault == "platform":
        del manifest["platforms"][release.PLATFORMS[0]]
    elif fault in {"failed", "skipped"}:
        manifest["jobs"]["build_wheels"]["result"] = fault
    elif fault == "fork":
        manifest["pr"]["head_repository"] = "fork/deepwave"
    elif fault == "manual_branch":
        manifest.update(event="workflow_dispatch", ref="refs/heads/feature")
    else:
        manifest["platforms"]["macos-14"]["wheels"][0]["filename"] = (
            "deepwave-0.1.0-py3-none-macosx_11_0_x86_64.whl"
        )
    with pytest.raises(ValueError, match=r".+"):
        validate(manifest)


@pytest.mark.parametrize("fault", ["hash", "missing", "extra", "metadata"])
def test_distribution_negative_controls(tmp_path: Path, fault: str) -> None:
    """Hashes, completeness and metadata are verified before copying."""
    manifest = candidate(tmp_path)
    path = tmp_path / manifest["sdist"]["filename"]
    if fault == "hash":
        path.write_bytes(path.read_bytes() + b"changed")
    elif fault == "missing":
        path.unlink()
    elif fault == "extra":
        (tmp_path / "unexpected.whl").write_bytes(b"extra")
    else:
        wheel = (
            tmp_path
            / manifest["platforms"][release.PLATFORMS[0]]["wheels"][0]["filename"]
        )
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "deepwave-0.1.0.dist-info/METADATA", "Name: deepwave\nVersion: 9.9.9\n"
            )
        manifest["platforms"][release.PLATFORMS[0]]["wheels"][0].update(
            sha256=release.sha256(wheel), bytes=wheel.stat().st_size
        )
    with pytest.raises(ValueError, match=r".+"):
        release.verify_files(manifest, tmp_path, tmp_path / "publish")


@pytest.mark.parametrize("fault", ["missing", "tree", "skipped", "failed", "pr_base"])
def test_required_gate_negative_controls(tmp_path: Path, fault: str) -> None:
    """Other workflows must pass for exactly the accepted PR content."""
    manifest = candidate(tmp_path)
    gates = {
        name: copy.deepcopy(
            {**manifest, "kind": name, "jobs": {"gate": {"result": "success"}}}
        )
        for name in release.GATES
    }
    if fault == "missing":
        del gates["docs"]
    elif fault == "tree":
        gates["lint"]["tree"] = "other"
    elif fault == "pr_base":
        gates["tests"]["pr"]["base"] = "other"
    else:
        gates["sanitizers"]["jobs"]["gate"]["result"] = fault
    with pytest.raises(ValueError, match=r".+"):
        release.validate_gates(manifest, gates)


class MockAPI:
    """Only fixture API data, never a network request."""

    def __init__(self) -> None:
        """Create successful API fixtures."""
        self.jobs = [{"conclusion": "success"}]
        self.artifacts = [
            {
                "id": 42,
                "name": "release-manifest-a2",
                "expired": False,
                "workflow_run": {"id": 123},
            }
        ]

    def get(self, path: str) -> dict[str, Any]:
        """Return an internal master PR."""
        assert path.endswith("/pulls/7")
        return {
            "head": {"repo": {"full_name": release.REPOSITORY}},
            "base": {"ref": "master", "repo": {"full_name": release.REPOSITORY}},
        }

    def pages(self, path: str, key: str) -> list[dict[str, Any]]:
        """Return mock jobs or artifacts."""
        assert path.startswith(f"/repos/{release.REPOSITORY}/actions/")
        return self.jobs if key == "jobs" else self.artifacts


@pytest.mark.parametrize(
    "fault",
    [None, "workflow", "attempt", "fork", "failed", "skipped", "event", "branch"],
)
def test_api_run_controls(fault: str | None) -> None:
    """API run, workflow and all job states are checked independently."""
    api = MockAPI()
    run = {
        "id": 123,
        "repository": {"full_name": release.REPOSITORY},
        "head_repository": {"full_name": release.REPOSITORY},
        "path": ".github/workflows/build.yml",
        "run_attempt": 2,
        "status": "completed",
        "conclusion": "success",
        "event": "pull_request",
        "pull_requests": [{"number": 7}],
    }
    if fault == "workflow":
        run["path"] = ".github/workflows/benchmarks.yml"
    elif fault == "attempt":
        run["run_attempt"] = 1
    elif fault == "fork":
        run["head_repository"]["full_name"] = "fork/deepwave"
    elif fault in {"failed", "skipped"}:
        api.jobs[0]["conclusion"] = fault
    elif fault == "event":
        run["event"] = "push"
    elif fault == "branch":
        run.update(event="workflow_dispatch", head_branch="feature")
    if fault is None:
        publication.check_run(api, run, "build", 2)  # type: ignore[arg-type]
    else:
        with pytest.raises(ValueError, match=r".+"):
            publication.check_run(api, run, "build", 2)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fault", [None, "expired", "missing", "duplicate", "wrong_run"]
)
def test_api_artifact_id_resolution(fault: str | None) -> None:
    """Select exact IDs from the chosen run/attempt, rejecting ambiguity."""
    api = MockAPI()
    if fault == "expired":
        api.artifacts[0]["expired"] = True
    elif fault == "missing":
        api.artifacts = []
    elif fault == "duplicate":
        api.artifacts *= 2
    elif fault == "wrong_run":
        api.artifacts[0]["workflow_run"]["id"] = 456
    if fault is None:
        result = publication.artifacts_for(api, 123, {"release-manifest-a2"})  # type: ignore[arg-type]
        assert result["release-manifest-a2"]["id"] == 42
    else:
        with pytest.raises(ValueError, match=r".+"):
            publication.artifacts_for(api, 123, {"release-manifest-a2"})  # type: ignore[arg-type]


def test_aggregate_requires_actual_distribution_files(tmp_path: Path) -> None:
    """Matrix receipts cannot substitute for a missing wheel file."""
    manifest = candidate(tmp_path)
    sdist_receipt = {k: v for k, v in manifest.items() if k != "platforms"}
    sdist_receipt["kind"] = "sdist"
    sdist_dir = tmp_path / "sdist"
    sdist_dir.mkdir()
    (sdist_dir / "receipt.json").write_text(json.dumps(sdist_receipt))
    for name, receipt in manifest["platforms"].items():
        platform = tmp_path / name
        platform.mkdir()
        (platform / "receipt.json").write_text(json.dumps(receipt))
    actual = release.aggregate(tmp_path, manifest["jobs"])
    assert actual["source_members"] == manifest["source_members"]
    (
        tmp_path / manifest["platforms"][release.PLATFORMS[0]]["wheels"][0]["filename"]
    ).unlink()
    with pytest.raises(ValueError, match="missing/extra distributions"):
        release.aggregate(tmp_path, manifest["jobs"])


def test_redirect_strips_token() -> None:
    """A signed external artifact URL never receives the GitHub token."""
    request = publication.urllib.request.Request(
        "https://api.github.com/repos/ar4/deepwave/actions/artifacts/42/zip",
        headers={"Authorization": "Bearer private-token"},
    )
    redirected = publication.SafeRedirect().redirect_request(
        request, None, 302, "Found", {}, "https://example.invalid/artifact.zip"
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None
    with pytest.raises(ValueError, match="non-HTTPS redirect"):
        publication.SafeRedirect().redirect_request(
            request, None, 302, "Found", {}, "http://example.invalid/artifact.zip"
        )


class ArtifactAPI(MockAPI):
    """Supply an entire accepted build and its four independent gate workflows."""

    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        """Bind synthetic archive bytes to API artifacts and exact receipts."""
        super().__init__()
        self.directory = directory
        self.manifest = manifest
        self.names = [
            "release-sdist-a2",
            "release-manifest-a2",
            *[f"release-wheels-{p}-a2" for p in release.PLATFORMS],
        ]

    def run(self, workflow: str) -> dict[str, Any]:
        """Return independently completed workflow identities."""
        return {
            "id": 123 if workflow == "build" else 200 + release.GATES.index(workflow),
            "repository": {"full_name": release.REPOSITORY},
            "head_repository": {"full_name": release.REPOSITORY},
            "path": f".github/workflows/{workflow}.yml",
            "run_attempt": 2,
            "status": "completed",
            "conclusion": "success",
            "event": "pull_request",
            "head_sha": "head",
            "pull_requests": [
                {"number": 7, "head": {"sha": "head"}, "base": {"sha": "base"}}
            ],
            "html_url": f"https://example.invalid/{workflow}",
        }

    def get(self, path: str) -> dict[str, Any]:
        """Return run, PR or Git-tree data without external services."""
        if "/git/commits/" in path:
            return {
                "tree": {"sha": "tree"},
                "parents": [{"sha": "head"}, {"sha": "base"}],
            }
        if "/actions/runs/" in path:
            run_id = int(path.rsplit("/", 1)[1])
            return self.run("build" if run_id == 123 else release.GATES[run_id - 200])
        return super().get(path)

    def pages(self, path: str, key: str) -> list[dict[str, Any]]:
        """Resolve run-scoped artifact IDs and successful gate candidates."""
        if key == "workflow_runs":
            workflow = path.split("/workflows/")[1].split(".yml", maxsplit=1)[0]
            return [self.run(workflow)]
        if key == "jobs":
            return self.jobs
        run_id = int(path.split("/runs/")[1].split("/", maxsplit=1)[0])
        names = (
            self.names
            if run_id == 123
            else [f"qualification-{release.GATES[run_id - 200]}-a2"]
        )
        return [
            {
                "id": run_id * 10 + i,
                "name": name,
                "expired": False,
                "workflow_run": {"id": run_id},
            }
            for i, name in enumerate(names)
        ]

    def download(self, artifact: dict[str, Any], output: Path) -> None:
        """Write only the fixture archive and JSON bytes for the selected ID."""
        output.mkdir()
        name = artifact["name"]
        if name.startswith("release-manifest"):
            (output / "release-manifest.json").write_text(json.dumps(self.manifest))
            return
        if name.startswith("qualification"):
            kind = name.removeprefix("qualification-").removesuffix("-a2")
            receipt = {
                **self.manifest,
                "run_id": self.run(kind)["id"],
                "kind": kind,
                "jobs": {"gate": {"result": "success"}},
            }
            distributions = []
        elif name.startswith("release-sdist"):
            receipt = {**self.manifest, "kind": "sdist"}
            distributions = [self.manifest["sdist"]]
        else:
            kind = name.removeprefix("release-wheels-").removesuffix("-a2")
            receipt = self.manifest["platforms"][kind]
            distributions = receipt["wheels"]
        (output / "receipt.json").write_text(json.dumps(receipt))
        for distribution in distributions:
            (output / distribution["filename"]).write_bytes(
                (self.directory / distribution["filename"]).read_bytes()
            )


def test_full_publication_validation_uses_existing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complete promotion path uses four existing distributions and no upload."""
    source = tmp_path / "source"
    source.mkdir()
    manifest = candidate(source)
    api = ArtifactAPI(source, manifest)
    monkeypatch.setattr(publication, "version", lambda: "0.1.0")
    monkeypatch.setattr(
        publication, "git", lambda *args: "tag-commit" if "--verify" in args else "tree"
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", release.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    commands = []
    monkeypatch.setattr(
        publication.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    output = tmp_path / "accepted"
    result = publication.prepare(api, 123, 2, "v0.1.0", output)  # type: ignore[arg-type]
    assert result["artifact_ids"]["release-manifest-a2"] == 1231
    assert len(list((output / "publish").iterdir())) == 4
    assert set(result["required_gates"]) == set(release.GATES)
    assert commands[0][1:3] == ["merge-base", "--is-ancestor"]
    assert commands[1][1:4] == ["-m", "twine", "check"]
    assert all("upload" not in command for command in commands)


@pytest.mark.parametrize("fault", [None, "base", "head"])
def test_receipt_uses_recorded_merge_parents_after_squash(
    tmp_path: Path, fault: str | None
) -> None:
    """A live PR base update must not invalidate an unchanged accepted tree."""
    manifest = candidate(tmp_path)
    api = ArtifactAPI(tmp_path, manifest)
    run = api.run("build")
    run["pull_requests"][0]["base"]["sha"] = "new-master-after-squash"
    if fault is not None:
        manifest["pr"][fault] = "wrong"
        with pytest.raises(ValueError, match="merge parents"):
            publication.check_receipt_run(api, manifest, run)  # type: ignore[arg-type]
    else:
        publication.check_receipt_run(api, manifest, run)  # type: ignore[arg-type]
