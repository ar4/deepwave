"""Enforce the existing 92% statement gate on a branch-enabled coverage report.

Coverage.py's combined percentage includes branches when they are measured.
Keep the statement requirement explicit while collecting a branch baseline;
do not round a below-threshold result up to a passing result.
"""

import argparse
import json
from pathlib import Path


def check_report(path: Path) -> None:
    """Print both measurements and reject empty or insufficient statement data."""
    totals = json.loads(path.read_text())["totals"]
    covered = totals["covered_lines"]
    statements = totals["num_statements"]
    if not isinstance(covered, int) or not isinstance(statements, int):
        raise TypeError("Coverage statement counts must be integers.")
    if not 0 <= covered <= statements or statements == 0:
        raise ValueError("Coverage report has invalid or empty statement counts.")
    print(
        f"Statement coverage: {covered}/{statements} "
        f"({100 * covered / statements:.3f}%)"
    )
    print(f"Branch coverage: {totals['covered_branches']}/{totals['num_branches']}")
    if 100 * covered < 92 * statements:
        raise ValueError("Statement coverage is below the required 92%.")


def main() -> None:
    """Check the JSON report produced by coverage.py or pytest-cov."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    check_report(parser.parse_args().report)


if __name__ == "__main__":
    main()
