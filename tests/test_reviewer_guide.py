import re
from pathlib import Path


MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_reviewer_guide_local_links_resolve() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    guide_path = repository_root / "docs" / "reviewer_guide.md"
    guide = guide_path.read_text()

    local_targets = []
    for target in MARKDOWN_LINK.findall(guide):
        if "://" in target or target.startswith("#"):
            continue
        path_text = target.split("#", maxsplit=1)[0]
        local_targets.append((guide_path.parent / path_text).resolve())

    assert local_targets
    assert all(target.exists() for target in local_targets)
