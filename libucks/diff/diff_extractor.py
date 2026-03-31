"""DiffExtractor — git diff HEAD → List[DiffEvent] using gitpython + unidiff."""
from __future__ import annotations

from pathlib import Path
from typing import List

import git
import structlog
from unidiff import PatchSet

from libucks.models.events import DiffEvent, DiffHunk

log = structlog.get_logger(__name__)


class DiffExtractor:
    def __init__(self, repo_path: Path) -> None:
        self._repo = git.Repo(str(repo_path), search_parent_directories=True)

    def extract(self, filepath: Path) -> List[DiffEvent]:
        """Run ``git diff HEAD -- <filepath> --find-renames`` and parse into DiffEvents."""
        rel = str(filepath)
        try:
            diff_text = self._repo.git.diff(
                "HEAD", "--find-renames", "--", rel
            )
        except git.GitCommandError as exc:
            log.warning("diff_extractor.git_error", file=rel, error=str(exc))
            return []

        if not diff_text:
            log.debug("diff_extractor.no_diff", file=rel)
            return []

        # Detect binary files — git outputs "Binary files … differ"
        if "Binary files" in diff_text:
            log.warning("diff_extractor.binary_skipped", file=rel)
            return []

        try:
            patch = PatchSet(diff_text)
        except Exception as exc:
            log.warning("diff_extractor.parse_error", file=rel, error=str(exc))
            return []

        events: List[DiffEvent] = []
        for patched_file in patch:
            is_rename = patched_file.is_rename
            old_path = patched_file.source_file.lstrip("a/") if is_rename else None
            new_path = patched_file.target_file.lstrip("b/") if is_rename else None

            hunks: List[DiffHunk] = []
            for hunk in patched_file:
                added = [line.value.rstrip("\n") for line in hunk if line.is_added]
                removed = [line.value.rstrip("\n") for line in hunk if line.is_removed]

                source_start = hunk.source_start
                source_length = hunk.source_length or 0
                target_start = hunk.target_start
                target_length = hunk.target_length or 0

                hunks.append(
                    DiffHunk(
                        file=rel,
                        old_start=source_start,
                        old_end=source_start + max(source_length - 1, 0),
                        new_start=target_start,
                        new_end=target_start + max(target_length - 1, 0),
                        added_lines=added,
                        removed_lines=removed,
                    )
                )

            event = DiffEvent(
                file=rel,
                hunks=hunks,
                is_rename=is_rename,
                old_path=old_path,
                new_path=new_path,
            )
            events.append(event)
            log.info(
                "diff_extractor.event",
                file=rel,
                hunks=len(hunks),
                is_rename=is_rename,
            )

        return events
