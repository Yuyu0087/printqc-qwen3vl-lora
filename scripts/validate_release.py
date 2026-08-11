from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from printqc.artifacts import ensure_adapter
from printqc.config import load_release_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PRIVATE_PATTERNS = (
    ("private-posix-path", re.compile(r"/home/[A-Za-z0-9_.-]+/[^\s'\"<>]+")),
    ("private-windows-path", re.compile(r"[A-Za-z]:\\[^\s'\"<>]+")),
    ("env-file-reference", re.compile(r"(^|[/\\])\.env($|[.\s])")),
    ("credential-token", re.compile(r"(ghp|github_pat|hf|sk)-[A-Za-z0-9_=-]{12,}")),
)

FORBIDDEN_TRACKED_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".bin",
    ".7z",
    ".zip",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
}

SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "dist", "release_work", "validation_reports"}


def _git_files(repo_root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [line for line in result.stdout.splitlines() if line]


def _walk_files(repo_root: Path) -> list[str]:
    files: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(repo_root).parts):
            continue
        files.append(path.relative_to(repo_root).as_posix())
    return files


def _text_or_none(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _redacted_finding(rule: str, file_name: str, source: str = "worktree") -> dict[str, str]:
    return {"source": source, "file": file_name, "rule": rule, "match": "<redacted>"}


def scan_public_repo(repo_root: str | Path, *, git_history: bool = False) -> list[dict[str, str]]:
    root = Path(repo_root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"repo root does not exist: {root}")

    findings: list[dict[str, str]] = []
    files = _git_files(root) or _walk_files(root)
    for file_name in files:
        suffix = Path(file_name).suffix.lower()
        if suffix in FORBIDDEN_TRACKED_SUFFIXES:
            findings.append(_redacted_finding(f"forbidden-tracked-suffix:{suffix}", file_name))
            continue
        text = _text_or_none(root / file_name)
        if text is None:
            continue
        for rule, pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                findings.append(_redacted_finding(rule, file_name))

    if git_history:
        findings.extend(_scan_git_history(root))
    return findings


def _scan_git_history(repo_root: Path) -> list[dict[str, str]]:
    try:
        commits = subprocess.run(
            ["git", "-C", str(repo_root), "rev-list", "--all"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    findings: list[dict[str, str]] = []
    for commit in commits:
        try:
            tree = subprocess.run(
                ["git", "-C", str(repo_root), "ls-tree", "-r", "--name-only", commit],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.splitlines()
        except subprocess.CalledProcessError:
            continue
        for file_name in tree:
            suffix = Path(file_name).suffix.lower()
            if suffix in FORBIDDEN_TRACKED_SUFFIXES:
                findings.append(
                    _redacted_finding(
                        f"forbidden-history-suffix:{suffix}",
                        f"{commit[:12]}:{file_name}",
                        source="git-history",
                    )
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PrintQC release manifest and adapter asset.")
    parser.add_argument("--asset")
    parser.add_argument("--manifest")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--git-history", action="store_true")
    parser.add_argument("--redact-matches", action="store_true")
    args = parser.parse_args()

    if bool(args.asset) != bool(args.manifest):
        raise SystemExit("--asset and --manifest must be provided together")
    if args.asset and args.manifest:
        asset = Path(args.asset)
        manifest = load_release_manifest(args.manifest)
        if asset.stat().st_size != manifest.adapter_asset_size:
            raise SystemExit("asset size does not match release manifest")
        if sha256_file(asset) != manifest.adapter_asset_sha256:
            raise SystemExit("asset SHA256 does not match release manifest")
        cache_root = Path(args.cache_root) if args.cache_root else asset.parent / "_validate_cache"
        ensure_adapter(manifest, cache_root=cache_root, fetcher=lambda _url: asset.read_bytes())
        print("release asset validation passed")

    if args.repo_root:
        findings = scan_public_repo(args.repo_root, git_history=args.git_history)
        if findings:
            print(json.dumps(findings, indent=2, sort_keys=True))
            raise SystemExit("public repository scan failed")
        print("public repository scan passed")
    if not args.asset and not args.repo_root:
        raise SystemExit("provide --asset/--manifest, --repo-root, or both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
