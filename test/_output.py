"""Shared output folder for the test scripts.

Everything the tests write (figures, caches, …) lands in
``<repo root>/test_output/``, so the whole folder can be gitignored and
cleaned in one go instead of naming each artifact individually.
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "test_output"


def out_path(name):
    """Absolute path for an output file, creating folders on demand."""
    p = OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def resolve_save(save):
    """Route a ``--save`` argument into OUT_DIR unless it is absolute.

    ``None`` passes through (interactive show); an absolute path is
    honoured verbatim; a relative one lands in the shared folder.
    """
    if save is None:
        return None
    p = Path(save)
    return str(p) if p.is_absolute() else out_path(save)
