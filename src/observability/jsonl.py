from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.observability.models import (
    RunRecord,
)


class JsonlRunWriter:
    """Append FastContext run records to a JSONL file."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Return the JSONL output path."""

        return self._path

    def write(
        self,
        record: RunRecord,
    ) -> None:
        """Append one run record to the JSONL file."""

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = asdict(
            record
        )

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

        with self._path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as file:
            file.write(
                serialized
            )
            file.write(
                "\n"
            )


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""

    if not path.exists():
        return []

    records: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            normalized_line = (
                line.strip()
            )

            if not normalized_line:
                continue

            value = json.loads(
                normalized_line
            )

            if not isinstance(
                value,
                dict,
            ):
                raise ValueError(
                    "Expected a JSON object "
                    f"at line {line_number}."
                )

            records.append(
                value
            )

    return records