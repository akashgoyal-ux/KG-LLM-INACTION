"""
RunLogger
=========
Writes a Run record to Neo4j (if available) and to a JSONL file on disk.
Every import/analysis job should start and finish a run.

Usage
-----
from ChaptersFinancial._platform.obs.run_logger import RunLogger

with RunLogger(chapter="ch04_fin", module="community_louvain") as run:
    run.set_metric("nodes_processed", 5000)
    do_work()
# On exit: status set to COMPLETED (or FAILED on exception), endedAt written.
"""

from __future__ import annotations

import datetime
import json
import traceback
import uuid
from pathlib import Path


class RunLogger:
    def __init__(
        self,
        chapter: str,
        module: str,
        run_id: str | None = None,
        log_dir: str | Path | None = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())
        self.chapter = chapter
        self.module = module
        self.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.ended_at: str | None = None
        self.status = "RUNNING"
        self._metrics: dict = {}

        repo_root = Path(__file__).parent.parent.parent.parent
        default_log_dir = repo_root / "data_fin" / "runs"
        self._log_dir = Path(log_dir) if log_dir else default_log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------
    def __enter__(self) -> "RunLogger":
        self._write_log()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if exc_type is None:
            self.status = "COMPLETED"
        else:
            self.status = "FAILED"
            self._metrics["error"] = traceback.format_exc()
        self._write_log()
        self._write_neo4j()
        return False  # Do not suppress exceptions

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def set_metric(self, key: str, value) -> None:
        self._metrics[key] = value

    def increment(self, key: str, by: int = 1) -> None:
        self._metrics[key] = self._metrics.get(key, 0) + by

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "runId": self.run_id,
            "chapter": self.chapter,
            "module": self.module,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "status": self.status,
            "metrics": self._metrics,
        }

    def _write_log(self) -> None:
        log_file = self._log_dir / "runs.jsonl"
        with log_file.open("a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")

    def _write_neo4j(self) -> None:
        try:
            from ChaptersFinancial._platform.providers.graph import GraphProvider  # noqa: PLC0415
            gp = GraphProvider()
            if not gp.ping():
                return
            cypher = """
            MERGE (r:Run {runId: $runId})
            SET r.chapter   = $chapter,
                r.module    = $module,
                r.startedAt = $startedAt,
                r.endedAt   = $endedAt,
                r.status    = $status,
                r.metrics   = $metrics
            """
            gp.run(cypher, {
                **self.to_dict(),
                "metrics": json.dumps(self._metrics),
            })
            gp.close()
        except Exception:
            pass  # observability must never break the main job
