"""Thin wrapper — the tag partition now lives in tag_build.py.

Kept for backward compatibility. `tag_build.py` builds tag_groups.json (and
test_games.json) as part of its normal run. This just rebuilds tag_groups.json
from the existing tag_vocab.json using tag_build's curated lists, so the curated
MECHANICS / STORY / SUBJECTIVE definitions have a single home (tag_build.py).
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tag_build import build_tag_groups  # noqa: E402


def build():
    tags_dir = SCRIPT_DIR / "tags"
    vocab = json.loads((tags_dir / "tag_vocab.json").read_text(encoding="utf-8"))["tags"]
    groups = build_tag_groups(vocab)
    (tags_dir / "tag_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"mechanics={len(groups['mechanics'])} story={len(groups['story'])} "
          f"subjective={len(groups['subjective'])} content={len(groups['content'])} drop={len(groups['drop'])}")
    print(f"wrote {tags_dir / 'tag_groups.json'}")


if __name__ == "__main__":
    build()
