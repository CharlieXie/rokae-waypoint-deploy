"""Fail-closed check that the running transformers package is our reviewed copy.

Why this module exists
----------------------
openpi replaces five files inside the installed ``transformers`` package
(``src/openpi/models_pytorch/transformers_replace``).  Those files change
attention semantics.  If the environment is rebuilt -- a plain ``uv sync``, a
fresh venv, or a ``pip install -U transformers`` -- upstream transformers comes
back while the git tree stays clean, and the model silently computes something
different.

Upstream ships one check (``transformers.models.siglip.check``), but it is only
invoked from ``Pi0Pytorch.__init__``.  The waypoint stack never constructs
``Pi0Pytorch``, so **the production inference path ran no check at all**:

    scripts/serve_policy.py           -> RokaeWaypointPolicy   (0 checks)
    src/openpi/waypoint/rokae_policy.py                        (0 checks)
    scripts/eval_rokae_offline.py                              (0 checks)

Worse, the upstream check is *itself* one of the five replaced files.  Reinstall
transformers and you get upstream's ``check.py`` back -- which happily returns
True about upstream's own code.  That is why this module does not stop at the
official check: it also compares all five runtime files byte-for-byte against
the reviewed repository copies.  A reinstall is the **default** step of any
recovery procedure, so this is the scenario that matters most.

The logic was extracted verbatim from ``scripts/train_waypoint_joint.py``
(training already failed closed); the trainer now imports it from here so the
two paths cannot drift apart.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

TRANSFORMERS_REPLACEMENT_FILES = (
    "models/gemma/configuration_gemma.py",
    "models/gemma/modeling_gemma.py",
    "models/paligemma/modeling_paligemma.py",
    "models/siglip/check.py",
    "models/siglip/modeling_siglip.py",
)

INSTALL_HINT = (
    "copy src/openpi/models_pytorch/transformers_replace/* into the active "
    "transformers package after syncing the environment"
)

# One guard run per (repo_root) per process. The five files are hashed once;
# calling the guard from three entry points must not cost three hashings.
_CACHE: dict[str, dict[str, str]] = {}


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a streaming SHA-256 for a file recorded in provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replacement_tree_fingerprint(root: Path) -> tuple[str, dict[str, str]]:
    """Fingerprint the exact transformers replacement files under ``root``."""
    combined = hashlib.sha256()
    combined.update(b"openpi-transformers-replacement-v1\0")
    per_file: dict[str, str] = {}
    for relative_path in TRANSFORMERS_REPLACEMENT_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"transformers replacement file is missing: {path}")
        file_sha = sha256_file(path)
        per_file[relative_path] = file_sha
        combined.update(relative_path.encode())
        combined.update(b"\0")
        combined.update(file_sha.encode())
        combined.update(b"\0")
    return combined.hexdigest(), per_file


def default_repo_root() -> Path:
    """The repository this module was imported from.

    ``<repo>/src/openpi/waypoint/transformers_guard.py`` -> ``<repo>``.  Holds
    for the training repo and for the shipped deploy bundle, which keeps the
    same ``src/openpi/...`` layout and ships ``transformers_replace`` too.
    """
    return Path(__file__).resolve().parents[3]


def transformers_replacement_provenance(repo_root: Path) -> dict[str, str]:
    """Fail closed unless the live transformers package matches our replacement.

    Returns the two provenance fields to record alongside a run; raises
    ``RuntimeError`` with an actionable hint otherwise.
    """
    try:
        import transformers
        from transformers.models.siglip import check

        official_check_ok = check.check_whether_transformers_replace_is_installed_correctly()
    except Exception as exc:
        raise RuntimeError(
            f"transformers replacement validation could not run; {INSTALL_HINT}: {exc}"
        ) from exc
    if official_check_ok is not True:
        raise RuntimeError(f"transformers replacement official check failed; {INSTALL_HINT}")

    transformers_file = getattr(transformers, "__file__", None)
    if not transformers_file:
        raise RuntimeError("cannot locate the active transformers package for replacement validation")
    runtime_root = Path(transformers_file).resolve().parent
    source_root = Path(repo_root) / "src/openpi/models_pytorch/transformers_replace"

    try:
        source_sha, source_files = replacement_tree_fingerprint(source_root)
        runtime_sha, runtime_files = replacement_tree_fingerprint(runtime_root)
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"transformers replacement files are incomplete; {INSTALL_HINT}: {exc}"
        ) from exc

    mismatched = [
        relative_path
        for relative_path in TRANSFORMERS_REPLACEMENT_FILES
        if source_files[relative_path] != runtime_files[relative_path]
    ]
    if mismatched or runtime_sha != source_sha:
        raise RuntimeError(
            "active transformers replacement differs from the reviewed repository copy: "
            f"{mismatched}; {INSTALL_HINT}"
        )

    return {
        "transformers_version": str(transformers.__version__),
        "transformers_replacement_sha256": runtime_sha,
    }


def assert_transformers_replacement(repo_root: Path | None = None, *, caller: str = "") -> dict[str, str]:
    """Guard entry point for inference. Runs once per process, then returns the cache.

    ``caller`` only decorates the error message so an operator can tell which
    entry point refused to start.
    """
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    key = str(root)
    if key in _CACHE:
        return _CACHE[key]
    try:
        provenance = transformers_replacement_provenance(root)
    except RuntimeError as exc:
        where = f" ({caller})" if caller else ""
        raise RuntimeError(f"refusing to run{where}: {exc}") from exc
    _CACHE[key] = provenance
    return provenance
