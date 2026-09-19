from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^uses:\s+[^\s@]+@[0-9a-f]{40}(\s+#.*)?$")
CANONICAL_WORKFLOWS = {"ci.yml", "ci-standard.yml", "ci-standard-heavy.yml"}


def test_all_workflow_actions_are_pinned_to_commit_shas():
    workflow_files = sorted(WORKFLOWS.glob("*.yml"))
    assert workflow_files
    for path in workflow_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("uses:"):
                assert PINNED_ACTION.match(line.strip()), (
                    f"{path.name}:{lineno} must pin the action to a commit SHA: {line.strip()}"
                )


def test_only_the_canonical_workflows_remain():
    assert {path.name for path in WORKFLOWS.glob("*.yml")} == CANONICAL_WORKFLOWS


def test_stale_tts_compose_file_is_removed():
    assert not (REPO / "docker-compose.tts.yaml").exists()
