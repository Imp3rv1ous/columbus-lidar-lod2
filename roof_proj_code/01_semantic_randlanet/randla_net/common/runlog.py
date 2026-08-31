"""Append-only Markdown run report.

A whole remote run (prep -> labels -> tests -> train -> infer) appends sections
to ONE markdown file, so after the run the user syncs back a single reviewable
artifact (per-epoch per-class IoU, inference class distributions, warnings).

Path comes from the RANDLA_REPORT env var; if unset, every call is a no-op, so the
same scripts run unchanged locally without a report.
"""
from __future__ import annotations

import os
from pathlib import Path


class RunLogger:
    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "RunLogger":
        return cls(os.environ.get("RANDLA_REPORT"))

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _w(self, text: str) -> None:
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)

    # --- markdown primitives -------------------------------------------------
    def header(self, title: str, level: int = 2) -> None:
        self._w(f"\n{'#' * level} {title}\n\n")

    def line(self, text: str = "") -> None:
        self._w(text + "\n")

    def kv(self, d: dict) -> None:
        for k, v in d.items():
            self._w(f"- **{k}**: {v}\n")
        self._w("\n")

    def table_header(self, headers: list) -> None:
        self._w("| " + " | ".join(str(h) for h in headers) + " |\n")
        self._w("| " + " | ".join("---" for _ in headers) + " |\n")

    def row(self, values: list) -> None:
        self._w("| " + " | ".join(str(v) for v in values) + " |\n")

    def table(self, headers: list, rows: list) -> None:
        self.table_header(headers)
        for r in rows:
            self.row(r)
        self._w("\n")

    def warn(self, msg: str) -> None:
        self._w(f"> ⚠️ {msg}\n\n")
