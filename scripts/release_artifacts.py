"""Record and verify tested release distributions without executing their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from platform import python_version
from typing import Any

REPOSITORY = "ar4/deepwave"
PLATFORMS = ("ubuntu-latest", "windows-2022", "macos-14")
GATES = ("tests", "docs", "lint", "sanitizers")
BUDGET = 99 * 1024 * 1024
LOCAL_GPU = {
    "qualified_toolchain": {
        "compiler": "GCC 13.3.1 Red Hat 13.3.1-2",
        "cuda_compiler": "13.3.73",
        "glibc": "2.28",
        "execution_python": "3.13.5",
        "execution_torch": "2.13.0+cu130",
        "gpu_architectures": ["sm86", "sm89"],
    },
    "qualified_runtime_sha256": (
        "1fd0137383370fe7b5a150d0563e07d3ce20e58cfa920cf0eb818a1bb45963cc"
    ),
    "qualified_revision": "35dcf3d5fb2a503f5cddccb8a39c53f5d9ee30c1",
    "qualified_manifest_sha256": (
        "ba768051649e206a6107eab65900c03514b3f61543d2a4ca47df64d48514766a"
    ),
    "scope": (
        "Historical local sm86/sm89 execution and sanitizer evidence for the "
        "qualified_revision and runtime hash above. Changed runtime/native "
        "inputs require fresh GPU qualification. Actions distributions have "
        "CPU execution only; no claim of GPU execution of these wheels."
    ),
}


def sha256(path: Path) -> str:
    """Hash a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    """Read a Git identity."""
    return subprocess.check_output(["git", *args], text=True).strip()  # noqa: S603,S607 - fixed Git command


def version() -> str:
    """Read the version as data, never by importing the source package."""
    match = re.search(
        r'__version__ = [\'"]([^\'"]+)', Path("src/deepwave/_version.py").read_text()
    )
    if match is None:
        raise ValueError("missing version")
    return match[1]


def require(condition: bool, message: str) -> None:
    """Reject an incomplete or mismatched release."""
    if not condition:
        raise ValueError(message)


def identity() -> dict[str, Any]:
    """Record the actual checkout, independently of the PR head."""
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pr = event.get("pull_request", {})
    return {
        "repository": os.environ["GITHUB_REPOSITORY"],
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "event": os.environ["GITHUB_EVENT_NAME"],
        "ref": os.environ["GITHUB_REF"],
        "head_sha": os.environ["GITHUB_SHA"],
        "pr": {
            "number": pr.get("number"),
            "head": pr.get("head", {}).get("sha"),
            "base": pr.get("base", {}).get("sha"),
            "head_repository": pr.get("head", {}).get("repo", {}).get("full_name"),
            "base_repository": pr.get("base", {}).get("repo", {}).get("full_name"),
        },
        "version": version(),
    }


def successful_jobs(jobs: dict[str, Any]) -> None:
    """Require every declared gate, including matrix jobs."""
    require(bool(jobs), "empty required jobs")
    failed = []
    for name, job in jobs.items():
        result = job.get("result") if isinstance(job, dict) else job
        if result != "success":
            failed.append(f"{name}: {result}")
    require(not failed, "required jobs did not succeed: " + "; ".join(failed))


def source_members(path: Path) -> dict[str, str]:
    """Hash every regular sdist member; reject links and duplicate paths."""
    members = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            require(not member.issym() and not member.islnk(), "sdist link")
            parts = Path(member.name).parts
            require(
                not Path(member.name).is_absolute() and ".." not in parts,
                "unsafe sdist path",
            )
            if member.isfile():
                name = "/".join(parts[1:])
                require(name not in members, "duplicate sdist member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("missing sdist member stream")
                members[name] = hashlib.sha256(stream.read()).hexdigest()
    return members


def metadata(path: Path) -> tuple[str, str]:
    """Read distribution metadata without unpacking or running package code."""
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as wheel:
            names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
            require(len(names) == 1, "wheel metadata count")
            data = wheel.read(names[0])
    else:
        with tarfile.open(path) as archive:
            members = [
                m
                for m in archive.getmembers()
                if len(Path(m.name).parts) == 2 and m.name.endswith("/PKG-INFO")
            ]
            require(len(members) == 1, "sdist metadata count")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise ValueError("missing metadata stream")
            data = stream.read()
    parsed = BytesParser().parsebytes(data)
    return str(parsed["Name"]), str(parsed["Version"])


def distribution(path: Path) -> dict[str, Any]:
    """Describe a single distribution."""
    name, release = metadata(path)
    require(
        name == "deepwave" and release == version(),
        "distribution version/name mismatch",
    )
    if path.suffix == ".whl":
        require(path.stat().st_size <= BUDGET, "99 MiB wheel limit")
    return {"filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def validate_source_members(receipt: dict[str, Any]) -> None:
    """Bind packaged sources to checkout bytes, allowing generated PKG-INFO.

    Separate hashes of an archive and a checkout do not establish that they
    contain the same code. Require all tracked build/package inputs and check
    every shipped file against the checkout; docs/tests may remain excluded.
    """
    sources = receipt["source_members"]
    checkout = receipt["checkout_members"]
    required = {"CMakeLists.txt", "pyproject.toml", "src/deepwave/__init__.py"}
    required.update(name for name in checkout if name.startswith("src/"))
    require(required <= sources.keys(), "sdist missing checkout build/package inputs")
    for name, digest in sources.items():
        if name == "PKG-INFO":
            continue
        require(name in checkout, f"sdist member absent from checkout: {name}")
        require(digest == checkout[name], f"sdist differs from checkout: {name}")


def record(kind: str, directory: Path, needs: dict[str, Any]) -> None:
    """Create a receipt only after all required work succeeds."""
    directory.mkdir(parents=True, exist_ok=True)
    receipt = identity()
    receipt["kind"] = kind
    receipt["schema"] = 1
    receipt["jobs"] = needs
    if kind == "sdist":
        archives = list(directory.glob("*.tar.gz"))
        require(len(archives) == 1, "expected sole sdist")
        receipt["sdist"] = distribution(archives[0])
        receipt["source_members"] = source_members(archives[0])
        receipt["checkout_members"] = {
            p: sha256(Path(p))
            for p in git("ls-files").splitlines()
            if Path(p).is_file()
        }
        validate_source_members(receipt)
    elif kind in PLATFORMS:
        wheels = sorted(directory.glob("*.whl"))
        require(len(wheels) == 1, "expected one py3-none wheel per platform")
        receipt["wheels"] = [distribution(p) for p in wheels]
        receipt["build_inputs"] = {
            key: value for key, value in os.environ.items() if key.startswith("CIBW_")
        }
        receipt["build_inputs"]["python"] = f"Python {python_version()}"
        receipt["build_inputs"]["workflow_sha256"] = sha256(
            Path(".github/workflows/build.yml")
        )
        receipt["build_inputs"]["pyproject_sha256"] = sha256(Path("pyproject.toml"))
        receipt["build_inputs"]["compiler_versions"] = (
            "Actual compiler versions in cibuildwheel build logs; Linux GCC toolset "
            "13 / CUDA 12.4 and Windows CUDA 12.4.1 in the hashed workflow"
        )
        receipt["build_inputs"]["runner_image"] = os.environ.get(
            "ImageVersion",  # noqa: SIM112 - GitHub runner variable
            "not supplied",
        )
        receipt["tests"] = (
            "cibuildwheel configured CPU tests and platform native smoke completed; "
            "conditional CUDA/optional-reference skips; no GPU runner"
        )
    else:
        successful_jobs(needs)
    (directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def aggregate(directory: Path, needs: dict[str, Any]) -> dict[str, Any]:
    """Combine the source and each expected successful platform artifact set."""
    successful_jobs(needs)
    receipts = [json.loads(p.read_text()) for p in directory.rglob("receipt.json")]
    require(len(receipts) == 1 + len(PLATFORMS), "missing/extra artifact set")
    by_kind = {r["kind"]: r for r in receipts}
    require(set(by_kind) == {"sdist", *PLATFORMS}, "missing/duplicate platform")
    reference = by_kind["sdist"]
    validate_source_members(reference)
    for r in receipts:
        for key in ("tree", "commit", "run_id", "attempt", "version", "pr", "event"):
            require(r[key] == reference[key], f"artifact {key} mismatch")
    manifest = dict(reference)
    manifest["kind"] = "release"
    manifest["jobs"] = needs
    manifest["platforms"] = {p: by_kind[p] for p in PLATFORMS}
    inputs = {
        name: digest
        for name, digest in reference["checkout_members"].items()
        if name.startswith("src/") or name == "CMakeLists.txt"
    }
    runtime_digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["local_gpu"] = {
        **LOCAL_GPU,
        "actual_runtime_sha256": runtime_digest,
        "source_match": runtime_digest == LOCAL_GPU["qualified_runtime_sha256"],
        "applicability": (
            "retained under the maintainer's no-repeat-GPU decision"
            if runtime_digest == LOCAL_GPU["qualified_runtime_sha256"]
            else "NOT APPLICABLE: runtime/native inputs differ from local evidence"
        ),
    }
    with tempfile.TemporaryDirectory() as temporary:
        verify_files(manifest, directory, Path(temporary) / "verified")
    (directory / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def validate_manifest(
    manifest: dict[str, Any], *, tree: str, run_id: int, attempt: int, release: str
) -> None:
    """Check accepted artifact provenance and completeness."""
    require(manifest["repository"] == REPOSITORY, "untrusted repository")
    for key, expected in {
        "tree": tree,
        "run_id": run_id,
        "attempt": attempt,
        "version": release,
    }.items():
        require(manifest[key] == expected, f"wrong {key}")
    trusted_identity(manifest)
    validate_source_members(manifest)
    successful_jobs(manifest["jobs"])
    require(set(manifest["platforms"]) == set(PLATFORMS), "missing/extra platform")
    for platform, r in manifest["platforms"].items():
        for key in (
            "tree",
            "commit",
            "run_id",
            "attempt",
            "version",
            "pr",
            "event",
            "repository",
        ):
            require(r[key] == manifest[key], f"platform {platform} wrong {key}")
        require(r["kind"] == platform and len(r["wheels"]) == 1, "wheel set mismatch")
        filename = r["wheels"][0]["filename"]
        tags = {
            "ubuntu-latest": r"manylinux[0-9_]+_x86_64(?:\.manylinux[0-9_]+_x86_64)*",
            "windows-2022": r"win_amd64",
            "macos-14": r"macosx_\d+_\d+_arm64(?:\.macosx_\d+_\d+_arm64)*",
        }
        pattern = re.escape(f"deepwave-{release}-py3-none-") + tags[platform] + r"\.whl"
        require(
            re.fullmatch(pattern, filename) is not None, "wrong wheel platform/name"
        )
    require(
        manifest["sdist"]["filename"] == f"deepwave-{release}.tar.gz",
        "sdist name mismatch",
    )


def trusted_identity(receipt: dict[str, Any]) -> None:
    """Only internal master PRs and manual master runs may supply evidence."""
    require(receipt["repository"] == REPOSITORY, "untrusted repository")
    if receipt["event"] == "workflow_dispatch":
        require(receipt["ref"] == "refs/heads/master", "untrusted manual branch")
    else:
        require(receipt["event"] == "pull_request", "untrusted event")
        require(
            receipt["pr"]["head_repository"] == REPOSITORY
            and receipt["pr"]["base_repository"] == REPOSITORY,
            "untrusted PR source",
        )
        require(
            bool(receipt["pr"]["number"])
            and bool(receipt["pr"]["head"])
            and bool(receipt["pr"]["base"]),
            "missing PR identity",
        )


def validate_gates(manifest: dict[str, Any], gates: dict[str, dict[str, Any]]) -> None:
    """Require same tested content and PR identity for every mandatory workflow."""
    require(set(gates) == set(GATES), "missing mandatory gate")
    for name, receipt in gates.items():
        trusted_identity(receipt)
        require(receipt["kind"] == name, "wrong gate kind")
        for key in ("tree", "version", "event", "pr"):
            require(receipt[key] == manifest[key], f"gate {name} wrong {key}")
        successful_jobs(receipt["jobs"])


def verify_files(manifest: dict[str, Any], directory: Path, output: Path) -> None:
    """Verify exact distributions, then copy only those bytes for publication."""
    expected = [
        manifest["sdist"],
        *[w for p in PLATFORMS for w in manifest["platforms"][p]["wheels"]],
    ]
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and (p.name.endswith(".whl") or p.name.endswith(".tar.gz"))
    ]
    require(
        len(files) == len(expected)
        and {p.name for p in files} == {r["filename"] for r in expected},
        "missing/extra distributions",
    )
    require(not output.exists() or not any(output.iterdir()), "output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for r in expected:
        matches = [p for p in files if p.name == r["filename"]]
        require(len(matches) == 1, "ambiguous distribution")
        path = matches[0]
        require(
            sha256(path) == r["sha256"] and path.stat().st_size == r["bytes"],
            "distribution hash/size mismatch",
        )
        require(
            metadata(path) == ("deepwave", manifest["version"]),
            "wrong metadata version",
        )
        if path.suffix == ".whl":
            require(path.stat().st_size <= BUDGET, "99 MiB limit")
        else:
            require(
                source_members(path) == manifest["source_members"],
                "sdist member mismatch",
            )
        shutil.copyfile(path, output / path.name)


def main() -> None:
    """Record build evidence or aggregate completed matrix jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "aggregate", "verify"))
    parser.add_argument("--kind")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--needs", default="{}")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        record(args.kind, args.directory, json.loads(args.needs))
    elif args.command == "aggregate":
        aggregate(args.directory, json.loads(args.needs))
    else:
        require(
            args.manifest is not None and args.output is not None,
            "manifest/output required",
        )
        manifest = json.loads(args.manifest.read_text())
        verify_files(manifest, args.directory, args.output)


def cli() -> None:
    """Expose validation failures as Actions annotations, retaining tracebacks."""
    try:
        main()
    except (ValueError, OSError) as error:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            message = str(error).replace("%", "%25").replace("\r", "%0D")
            message = message.replace("\n", "%0A")
            print(f"::error title=Release qualification failed::{message}", flush=True)
        raise


if __name__ == "__main__":
    cli()
