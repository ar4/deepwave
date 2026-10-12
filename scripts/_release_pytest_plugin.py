"""Machine-readable pytest accounting for scripts/release_check.py."""

import json
import os
from pathlib import Path

import pytest


class ReleaseResult:
    """Record collection and one terminal outcome for each selected test."""

    def __init__(self, *, worker: bool = False) -> None:
        self.selected: list[str] = []
        self.deselected: set[str] = set()
        self.outcomes: dict[str, str] = {}
        self.collection_errors: list[str] = []
        self.properties: dict[str, dict[str, object]] = {}
        self.worker = worker

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected.update(item.nodeid for item in items)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        if session.items:
            self.selected = [item.nodeid for item in session.items]

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(
        self, node: object, ids: list[str]
    ) -> None:
        """The xdist controller has no ``session.items`` of its own."""
        if not self.selected:
            self.selected = ids
        elif self.selected != ids:
            self.collection_errors.append("xdist workers collected different test IDs")

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: object, error: object) -> None:
        """Merge worker deselections and collection errors into the controller."""
        payload = getattr(node, "workeroutput", {}).get("deepwave_release", {})
        self.deselected.update(payload.get("deselected_ids", []))
        self.collection_errors.extend(payload.get("collection_errors", []))
        if error:
            self.collection_errors.append(f"xdist worker exited: {error}")

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors.append(report.nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.user_properties:
            self.properties[report.nodeid] = dict(report.user_properties)
        if report.failed:
            self.outcomes[report.nodeid] = "failed"
            destination = os.environ.get("DEEPWAVE_RELEASE_RESULT")
            if destination and not self.worker:
                # Preserve diagnosis if an interrupted runner cannot finish
                # its final JSON report. Workers report to the controller;
                # only that controller writes the append-only failure log.
                try:
                    with (
                        Path(destination).with_suffix(".failures.log").open("a") as log
                    ):
                        log.write(f"{report.nodeid}\n{report.longrepr}\n\n")
                except OSError as exc:
                    self.collection_errors.append(
                        f"Could not preserve failure log: {exc}"
                    )
        elif report.skipped and self.outcomes.get(report.nodeid) != "failed":
            self.outcomes[report.nodeid] = "skipped"
        elif report.when == "call" and report.passed:
            self.outcomes[report.nodeid] = "passed"

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if hasattr(session.config, "workerinput"):
            session.config.workeroutput["deepwave_release"] = {
                "deselected_ids": sorted(self.deselected),
                "collection_errors": self.collection_errors,
            }
            return
        destination = os.environ.get("DEEPWAVE_RELEASE_RESULT")
        if not destination:
            return
        payload = {
            "selected_ids": self.selected,
            "deselected_ids": sorted(self.deselected),
            "outcomes": self.outcomes,
            "collection_errors": self.collection_errors,
            "properties": self.properties,
            "pytest_exit_code": int(exitstatus),
        }
        Path(destination).write_text(json.dumps(payload, sort_keys=True))


def pytest_configure(config: pytest.Config) -> None:
    """Register once, before collection and run reports arrive."""
    config.pluginmanager.register(
        ReleaseResult(worker=hasattr(config, "workerinput")),
        "deepwave-release-result",
    )
