#!/usr/bin/env python3
"""
Migrate Foragax checkpoint folders from results/ to checkpoint/

Source pattern:
  results/foragax/ForagaxTwoBiomeSmall-v2-<something>/<run>/checkpoint
Target pattern:
  checkpoint/foragax/ForagaxTwoBiomeSmall-v2-<something>/<run>

This script finds all matching source checkpoint dirs and moves their contents into
the corresponding target run directory, creating directories as needed.

Features:
 - dry-run (default) prints actions without performing them
 - --move actually moves files (uses os.rename when possible, falls back to copy+rm)
 - --copy copies files instead of moving
 - --overwrite allows overwriting existing target files
 - --verbose prints detailed info

Example:
  ./scripts/migrate_foragax_checkpoints.py --dry-run

"""
import argparse
import fnmatch
import os
import shutil
from pathlib import Path

ROOT = Path(os.getcwd())
SRC_GLOB = ROOT / "results" / "foragax"
DST_ROOT = ROOT / "checkpoint" / "foragax"


def find_sources(pattern_dirname_prefix="ForagaxTwoBiomeSmall-v2-*"):
    """Yield tuples (src_checkpoint_dir, target_run_dir)

    src_checkpoint_dir: Path to the 'checkpoint' folder inside a results run
    target_run_dir: Path to the destination run folder under checkpoint/foragax
    """
    if not SRC_GLOB.exists():
        return

    for name in os.listdir(SRC_GLOB):
        if not fnmatch.fnmatch(name, pattern_dirname_prefix):
            continue
        version_dir = SRC_GLOB / name
        if not version_dir.is_dir():
            continue
        # each version_dir contains multiple run folders (something2)
        for run_name in os.listdir(version_dir):
            run_dir = version_dir / run_name
            if not run_dir.is_dir():
                continue
            src_checkpoint = run_dir / "checkpoint"
            if src_checkpoint.exists() and src_checkpoint.is_dir():
                # target should be checkpoint/foragax/<version>/<run_name>
                target_run_dir = DST_ROOT / name / run_name
                yield src_checkpoint, target_run_dir


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path, overwrite=False, move=False, verbose=False):
    """Copy or move contents of src into dst. Does not copy src itself, only its contents."""
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            sfile = Path(root) / f
            tfile = target_root / f
            if tfile.exists():
                if not overwrite:
                    if verbose:
                        print(f"Skipping existing file {tfile}")
                    continue
                else:
                    if verbose:
                        print(f"Overwriting file {tfile}")
                    if tfile.is_file():
                        tfile.unlink()
            if move:
                try:
                    sfile.rename(tfile)
                except Exception:
                    # fallback to copy then remove
                    shutil.copy2(sfile, tfile)
                    sfile.unlink()
                if verbose:
                    print(f"Moved {sfile} -> {tfile}")
            else:
                shutil.copy2(sfile, tfile)
                if verbose:
                    print(f"Copied {sfile} -> {tfile}")


def migrate(dry_run=True, move=False, copy=False, overwrite=False, verbose=False):
    n = 0
    for src_checkpoint, target_run_dir in find_sources():
        n += 1
        if dry_run:
            print(f"[DRY] Would migrate {src_checkpoint} -> {target_run_dir}")
            if verbose:
                for p in src_checkpoint.rglob("*"):
                    print(f"    {p.relative_to(src_checkpoint)}")
            continue

        print(f"Migrating {src_checkpoint} -> {target_run_dir}")
        # ensure destination parent exists
        target_run_dir.mkdir(parents=True, exist_ok=True)
        # move/copy contents of src_checkpoint into target_run_dir
        copy_tree(src_checkpoint, target_run_dir, overwrite=overwrite, move=move, verbose=verbose)
        # if move, attempt to remove the now-empty src_checkpoint tree
        if move:
            try:
                shutil.rmtree(src_checkpoint)
            except Exception as e:
                print(f"Warning: failed to remove source checkpoint dir {src_checkpoint}: {e}")
    if n == 0:
        print("No matching checkpoint directories found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Foragax checkpoint folders from results/ to checkpoint/")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--move", action="store_true", help="Move files (default: copy). Will remove source files when possible.")
    group.add_argument("--copy", action="store_true", help="Copy files (default if neither --move nor --copy provided is copy/dry-run).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files at destination if present")
    parser.add_argument("--pattern", type=str, default="ForagaxTwoBiomeSmall-v2-*", help="Pattern for version directories inside results/foragax")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Do not perform changes, just show what would be done (default)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Perform actions")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # override pattern
    PATTERN = args.pattern

    # update SRC_GLOB/PATTERN handling by monkeypatching find_sources closure variable via redefinition
    def find_sources_local(pattern_dirname_prefix=PATTERN):
        if not SRC_GLOB.exists():
            return
        for name in os.listdir(SRC_GLOB):
            if not fnmatch.fnmatch(name, pattern_dirname_prefix):
                continue
            version_dir = SRC_GLOB / name
            if not version_dir.is_dir():
                continue
            for run_name in os.listdir(version_dir):
                run_dir = version_dir / run_name
                if not run_dir.is_dir():
                    continue
                src_checkpoint = run_dir / "checkpoint"
                if src_checkpoint.exists() and src_checkpoint.is_dir():
                    target_run_dir = DST_ROOT / name / run_name
                    yield src_checkpoint, target_run_dir

    # Replace the find_sources function used by migrate
    globals()["find_sources"] = find_sources_local

    migrate(dry_run=args.dry_run, move=args.move, copy=args.copy, overwrite=args.overwrite, verbose=args.verbose)
