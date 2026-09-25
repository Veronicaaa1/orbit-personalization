"""CLI for strict legacy registration and compatibility reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.legacy import import_allowlisted, register_legacy_selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Validate and register the twelve allowlisted legacy cells")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approvals", type=Path, help="JSON mapping exact cell keys to missing field names and approval reasons; exceptions remain unverified")
    parser.add_argument("--register-selected-value", action="store_true")
    parser.add_argument("--evidence", help="actual report/config path or explicit approval record")
    parser.add_argument("--report", type=Path)
    arguments, cfg = parse_config(parser, argv)
    payload: dict[str, object] = {}
    if arguments.register_selected_value:
        payload["selection"] = register_legacy_selection(cfg, arguments.evidence or "", dry_run=arguments.dry_run)
    approvals = json.loads(arguments.approvals.read_text(encoding="utf-8")) if arguments.approvals else None
    payload["runs"] = import_allowlisted(cfg, dry_run=arguments.dry_run, approvals=approvals)
    encoded = json.dumps(payload, indent=2, default=str)
    print(encoded)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(encoded + "\n", encoding="utf-8")
    blocked = any(str(row.get("decision", "")).startswith("blocked") or str(row.get("decision", "")).endswith("_unverified") for row in payload["runs"])
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
