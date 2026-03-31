"""BucketStore — CRUD for bucket .md files with YAML front-matter.

File format (strict):
    ---
    <YAML block>
    ---

    <Markdown prose>

The two --- delimiters are the only structural contract. write_prose() must
not touch anything before the second ---\\n, and write_front_matter() must
not touch anything after the second ---\\n.
"""

from pathlib import Path
from typing import List

import yaml

from libucks.models.bucket import BucketFrontMatter
from libucks.models.chunk import ChunkMetadata

_DELIM = "---\n"


def _split(raw: str) -> tuple[str, str]:
    """Split raw file text into (yaml_block, prose).

    Expects the file to start with ---\\n, followed by YAML, followed by
    another ---\\n, then the prose body.  Raises ValueError if the
    structure is not found.
    """
    if not raw.startswith(_DELIM):
        raise ValueError("Bucket file does not start with YAML front-matter delimiter '---'")
    # Skip past the opening ---\n and find the closing ---\n
    rest = raw[len(_DELIM):]
    close_idx = rest.find(_DELIM)
    if close_idx == -1:
        raise ValueError("Bucket file is missing closing '---' front-matter delimiter")
    yaml_block = rest[:close_idx]
    prose = rest[close_idx + len(_DELIM):]
    return yaml_block, prose


def _join(yaml_block: str, prose: str) -> str:
    return f"{_DELIM}{yaml_block}{_DELIM}{prose}"


class BucketStore:
    def __init__(self, bucket_dir: Path) -> None:
        self._dir = bucket_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, bucket_id: str) -> Path:
        return self._dir / f"{bucket_id}.md"

    def _require_exists(self, bucket_id: str) -> Path:
        path = self._path(bucket_id)
        if not path.exists():
            raise FileNotFoundError(f"Bucket not found: {bucket_id!r} ({path})")
        return path

    @staticmethod
    def _bfm_to_yaml_block(bfm: BucketFrontMatter) -> str:
        """Serialise a BucketFrontMatter to the raw YAML block string (no delimiters)."""
        data: dict = {
            "bucket_id": bfm.bucket_id,
            "domain_label": bfm.domain_label,
            "centroid_embedding": bfm.centroid_embedding,
            "token_count": bfm.token_count,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source_file": c.source_file,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "git_sha": c.git_sha,
                    "token_count": c.token_count,
                }
                for c in bfm.chunks
            ],
        }
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    @staticmethod
    def _yaml_block_to_bfm(yaml_block: str) -> BucketFrontMatter:
        parsed = yaml.safe_load(yaml_block)
        chunks = [ChunkMetadata(**c) for c in (parsed.get("chunks") or [])]
        return BucketFrontMatter(
            bucket_id=parsed["bucket_id"],
            domain_label=parsed["domain_label"],
            centroid_embedding=parsed["centroid_embedding"],
            token_count=parsed["token_count"],
            chunks=chunks,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        bucket_id: str,
        domain_label: str,
        centroid: str,
        chunks: List[ChunkMetadata],
        prose: str,
    ) -> Path:
        bfm = BucketFrontMatter(
            bucket_id=bucket_id,
            domain_label=domain_label,
            centroid_embedding=centroid,
            token_count=sum(c.token_count for c in chunks),
            chunks=chunks,
        )
        yaml_block = self._bfm_to_yaml_block(bfm)
        path = self._path(bucket_id)
        path.write_text(_join(yaml_block, prose), encoding="utf-8")
        return path

    def read(self, bucket_id: str) -> tuple[BucketFrontMatter, str]:
        path = self._require_exists(bucket_id)
        raw = path.read_text(encoding="utf-8")
        yaml_block, prose = _split(raw)
        return self._yaml_block_to_bfm(yaml_block), prose

    def write_prose(self, bucket_id: str, prose: str) -> None:
        path = self._require_exists(bucket_id)
        raw = path.read_text(encoding="utf-8")
        yaml_block, _ = _split(raw)
        path.write_text(_join(yaml_block, prose), encoding="utf-8")

    def write_front_matter(self, bucket_id: str, front_matter: BucketFrontMatter) -> None:
        path = self._require_exists(bucket_id)
        raw = path.read_text(encoding="utf-8")
        _, prose = _split(raw)
        yaml_block = self._bfm_to_yaml_block(front_matter)
        path.write_text(_join(yaml_block, prose), encoding="utf-8")

    def delete(self, bucket_id: str) -> None:
        path = self._require_exists(bucket_id)
        path.unlink()

    def list_all(self) -> List[str]:
        return [p.stem for p in sorted(self._dir.glob("*.md"))]
