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

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path

# The set as of 2026-08-28.  This is a *floor*, not the definition: the guard
# discovers the real list from the repository copy (see
# ``discover_replacement_files``) so that a sixth replacement file added later is
# covered automatically.  A file listed here that discovery cannot find means the
# repository copy is damaged, and the guard fails closed.
TRANSFORMERS_REPLACEMENT_FILES = (
    "models/gemma/configuration_gemma.py",
    "models/gemma/modeling_gemma.py",
    "models/paligemma/modeling_paligemma.py",
    "models/siglip/check.py",
    "models/siglip/modeling_siglip.py",
)

# Build artefacts that live inside a python package but are not part of it.
_IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd", ".so"})

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


def discover_replacement_files(source_root: Path) -> tuple[str, ...]:
    """Every file openpi overlays onto transformers, read off the repository copy.

    Enumerating a hardcoded list is the bug this function exists to remove: a
    sixth file added to ``transformers_replace/`` would have been overlaid onto
    the runtime package and never hashed, so the fingerprint would not change and
    the guard would pass on an environment it had not actually checked.  The
    directory itself is therefore the source of truth.

    Returns sorted POSIX-style relative paths, so the fingerprint does not depend
    on filesystem iteration order.  ``TRANSFORMERS_REPLACEMENT_FILES`` is enforced
    as a floor: a known file that is not on disk means the repository copy is
    damaged, which must not be silently fingerprinted as "a smaller overlay".
    """
    if not source_root.is_dir():
        raise FileNotFoundError(f"transformers replacement directory is missing: {source_root}")
    discovered = sorted(
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix not in _IGNORED_SUFFIXES
        and not _IGNORED_DIRECTORY_NAMES.intersection(path.relative_to(source_root).parts)
    )
    missing = [known for known in TRANSFORMERS_REPLACEMENT_FILES if known not in discovered]
    if missing:
        raise FileNotFoundError(
            f"transformers replacement files are missing from {source_root}: {missing}"
        )
    return tuple(discovered)


def replacement_tree_fingerprint(root: Path, relative_paths: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Fingerprint ``relative_paths`` as they exist under ``root``.

    The caller passes the file list (from ``discover_replacement_files``) so that
    the same set is hashed on both sides of the comparison -- the repository copy
    and the live transformers package.  A file present in the repository but
    absent from the runtime package raises, which is the "you added a replacement
    file and forgot to install it" case.
    """
    combined = hashlib.sha256()
    combined.update(b"openpi-transformers-replacement-v1\0")
    per_file: dict[str, str] = {}
    for relative_path in relative_paths:
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
        replacement_files = discover_replacement_files(source_root)
        source_sha, source_files = replacement_tree_fingerprint(source_root, replacement_files)
        runtime_sha, runtime_files = replacement_tree_fingerprint(runtime_root, replacement_files)
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"transformers replacement files are incomplete; {INSTALL_HINT}: {exc}"
        ) from exc

    mismatched = [
        relative_path
        for relative_path in replacement_files
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
