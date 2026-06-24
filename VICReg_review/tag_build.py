"""Build the tag vocabulary and per-game label matrix for the VICReg tag probe.

The probe (train_tag_probe.py) is a validation-only head: it freezes the VICReg
encoder, turns each game's reviews into a (num_latents, output_dim) code, and
tries to predict the game's Steam tags. If a simple head can decode the tags
from the frozen code, the encoder has learned a robust game representation.

This script:
  1. Reads the h5 game_names (the games we actually trained on).
  2. Looks up each game's tags in games.json (h5 name "<appid>_<n>" -> appid).
  3. Counts tags, keeps those frequent enough, and assigns each a stable id.
  4. Writes a label matrix aligned to game_names.

Outputs (under VICReg_review/tags/ by default). For --source tags this single
script now builds the whole data-prep chain (previously tag_groups.py +
build_test_games.py):
  tag_vocab.json          ordered tag list + metadata
  tag_labels.npz          game_names, appids, labels, raw_counts, normalized_counts
  tag_groups.json         mechanics / story / subjective / content / drop partition
  test_games.json         games.json INTERSECT training set, emotional tags dropped
  non_emotional_tags.json the kept (non-subjective) tag vocab
Use --no-groups / --no-test-games to skip the last three.

Tags in games.json are a {tag_name: vote_count} dict. Two target modes:
  binary  label = 1.0 if the game has the tag, else 0.0  (default, cleanest probe)
  weight  label = vote_count / max_vote_in_that_game     (soft 0..1 regression)
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_H5 = SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_GAMES_JSON = (
    PROJECT_ROOT
    / "game_review_data"
    / "Steam Games Metadata and Player Reviews (2020–2024"
    / "games.json"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "tags"


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_write_bytes(write_fn, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        write_fn(tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_game_names(h5_path):
    with h5py.File(h5_path, "r") as h5:
        return [decode_name(name) for name in h5["game_names"][:]]


def game_tag_dict(record, source):
    """Return {tag: weight} for one game record under the chosen source field."""
    if source == "tags":
        tags = record.get("tags") or {}
        if isinstance(tags, dict):
            return {str(name): float(count) for name, count in tags.items()}
        return {str(name): 1.0 for name in tags}
    # genres / categories are plain lists with no weights.
    values = record.get(source) or []
    return {str(name): 1.0 for name in values}


# ---------------------------------------------------------------------------
# Curated tag partition by role w.r.t. the "keep mechanics+story, drop subjective"
# hypothesis. Edit these lists to change the partition; everything downstream
# (tag_groups.json, test_games.json, the selectivity probe) follows. CONTENT =
# mechanics + story. Anything not listed becomes "drop" (technical / meta / mode).
# ---------------------------------------------------------------------------
MECHANICS = [
    "Action", "Adventure", "RPG", "Open World", "Simulation", "Strategy", "Shooter",
    "FPS", "Action RPG", "Sandbox", "Survival", "Stealth", "Tactical", "Platformer",
    "Racing", "Driving", "Fighting", "Puzzle", "Card Game", "Deckbuilding",
    "Card Battler", "Tower Defense", "RTS", "Turn-Based", "Turn-Based Strategy",
    "Turn-Based Combat", "Turn-Based Tactics", "Real Time Tactics", "Hack and Slash",
    "Souls-like", "Rogue-lite", "Rogue-like", "Action Roguelike", "Metroidvania",
    "Dungeon Crawler", "MMORPG", "JRPG", "CRPG", "Tactical RPG", "Strategy RPG",
    "Party-Based RPG", "City Builder", "Colony Sim", "Base-Building", "Building",
    "Crafting", "Management", "Resource Management", "Farming Sim", "Life Sim",
    "Automobile Sim", "Space Sim", "Grand Strategy", "4X", "Battle Royale",
    "Bullet Hell", "Beat 'em up", "Visual Novel", "Interactive Fiction",
    "Walking Simulator", "Point & Click", "Looter Shooter", "Third-Person Shooter",
    "Vehicular Combat", "Rhythm", "Sports", "Hunting", "Fishing", "Mining",
    "Parkour", "Swordplay", "Gun Customization", "Character Customization",
    "Inventory Management", "Loot", "Open World Survival Craft",
    "Procedural Generation", "Wargame", "Tanks", "Action-Adventure", "Immersive Sim",
    "3D Platformer", "2D Platformer", "3D Fighter", "Spectacle fighter",
    "Character Action Game", "Automation", "Agriculture", "Exploration", "Combat",
]
STORY = [
    "Story Rich", "Narrative", "Choices Matter", "Multiple Endings",
    "Choose Your Own Adventure", "Lore-Rich", "Sci-fi", "Fantasy", "Dark Fantasy",
    "Horror", "Survival Horror", "Psychological Horror", "Medieval", "Space",
    "Post-apocalyptic", "Cyberpunk", "Historical", "Military", "War",
    "World War II", "Mythology", "Zombies", "Aliens", "Robots", "Demons",
    "Dragons", "Vampire", "Supernatural", "Magic", "Crime", "Detective",
    "Mystery", "Romance", "Dating Sim", "Dystopian", "Alternate History",
    "Gothic", "Political", "Assassin", "Martial Arts", "Futuristic",
    "Conversation", "Text-Based", "Narration",
]
SUBJECTIVE = [
    "Atmospheric", "Great Soundtrack", "Funny", "Relaxing", "Difficult",
    "Beautiful", "Emotional", "Cute", "Colorful", "Cinematic", "Dark",
    "Dark Humor", "Comedy", "Drama", "Psychological", "Surreal", "Psychedelic",
    "Replay Value", "Memes", "Thriller", "Classic", "Old School", "Retro",
    "Stylized", "Cartoony", "Hand-drawn", "Fast-Paced", "Realistic",
    "Soundtrack", "Music",
]


def build_tag_groups(vocab_tags):
    """Partition the kept vocab into mechanics / story / subjective / drop.

    content = mechanics + story. Returns a dict of name -> ordered tag list,
    restricted to tags actually in the vocab.
    """
    valid = set(vocab_tags)

    def clean(names):
        seen, out = set(), []
        for name in names:
            if name in valid and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    mechanics, story, subjective = clean(MECHANICS), clean(STORY), clean(SUBJECTIVE)
    assigned = set(mechanics) | set(story) | set(subjective)
    drop = [tag for tag in vocab_tags if tag not in assigned]
    return {
        "mechanics": mechanics,
        "story": story,
        "subjective": subjective,
        "content": mechanics + story,
        "drop": drop,
    }


def build_test_games(game_names, appids, games, vocab_tags, subjective_set):
    """In-domain game pool: games.json INTERSECT the training set, with emotional
    (subjective) tags dropped and tags restricted to the vocab.

    Returns (test_games dict {appid: record}, kept non-emotional tag list).
    """
    keep_tags = [tag for tag in vocab_tags if tag not in subjective_set]
    keep_set = set(keep_tags)
    seen = set()
    test_games = {}
    for appid in appids:
        if appid in seen or appid not in games:
            continue
        seen.add(appid)
        record = dict(games[appid])  # shallow copy; don't mutate loaded games.json
        raw_tags = record.get("tags") or {}
        if isinstance(raw_tags, dict):
            record["tags"] = {t: c for t, c in raw_tags.items() if t in keep_set}
        else:
            record["tags"] = {t: 1 for t in raw_tags if t in keep_set}
        test_games[appid] = record
    return test_games, keep_tags


def build(args):
    games_json = Path(args.games_json)
    h5_path = Path(args.h5)
    out_dir = Path(args.out_dir)

    game_names = read_game_names(h5_path)
    appids = [name.split("_")[0] for name in game_names]

    with games_json.open("r", encoding="utf-8") as fh:
        games = json.load(fh)

    missing_appids = [name for name, appid in zip(game_names, appids) if appid not in games]
    per_game_tags = []
    document_frequency = Counter()
    for appid in appids:
        record = games.get(appid, {})
        tags = game_tag_dict(record, args.source)
        per_game_tags.append(tags)
        for tag in tags:
            document_frequency[tag] += 1

    if args.top_k > 0:
        kept = [tag for tag, _ in document_frequency.most_common(args.top_k)]
    else:
        kept = [tag for tag, freq in document_frequency.items() if freq >= args.min_count]
    # Stable order: most common first, then alphabetical for ties.
    kept.sort(key=lambda tag: (-document_frequency[tag], tag))
    tag_to_id = {tag: index for index, tag in enumerate(kept)}
    num_tags = len(kept)

    labels = np.zeros((len(game_names), num_tags), dtype=np.float32)
    raw_counts = np.zeros((len(game_names), num_tags), dtype=np.float32)
    normalized_counts = np.zeros((len(game_names), num_tags), dtype=np.float32)
    for row, tags in enumerate(per_game_tags):
        if not tags:
            continue
        present = {tag: weight for tag, weight in tags.items() if tag in tag_to_id}
        if not present:
            continue
        max_weight = max(present.values()) or 1.0
        for tag, weight in present.items():
            col = tag_to_id[tag]
            raw_counts[row, col] = weight
            normalized_counts[row, col] = weight / max_weight
        if args.target_mode == "weight":
            for tag, weight in present.items():
                labels[row, tag_to_id[tag]] = weight / max_weight
        else:
            for tag in present:
                labels[row, tag_to_id[tag]] = 1.0

    positives_per_game = (labels > 0).sum(axis=1)
    games_without_labels = [
        name for name, count in zip(game_names, positives_per_game) if count == 0
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "tags": kept,
        "num_tags": num_tags,
        "target_mode": args.target_mode,
        "source": args.source,
        "min_count": args.min_count,
        "top_k": args.top_k,
        "num_games": len(game_names),
        "games_without_labels": games_without_labels,
        "missing_appids": missing_appids,
        "document_frequency": {tag: document_frequency[tag] for tag in kept},
        "h5": str(h5_path.resolve()),
        "games_json": str(games_json.resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    atomic_write_bytes(
        lambda tmp: tmp.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"),
        out_dir / "tag_vocab.json",
    )
    def write_npz(tmp):
        # Pass a file handle so np.savez does not append ".npz" to the temp name.
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                game_names=np.asarray(game_names),
                appids=np.asarray(appids),
                labels=labels,
                raw_counts=raw_counts,
                normalized_counts=normalized_counts,
                tags=np.asarray(kept),
            )

    atomic_write_bytes(write_npz, out_dir / "tag_labels.npz")

    print(f"games={len(game_names)} missing_appids={len(missing_appids)}")
    print(
        f"unique_tags_seen={len(document_frequency)} kept={num_tags} "
        f"(min_count={args.min_count} top_k={args.top_k}) target_mode={args.target_mode} source={args.source}"
    )
    print(
        f"positives/game: min={int(positives_per_game.min())} "
        f"max={int(positives_per_game.max())} mean={positives_per_game.mean():.1f} "
        f"games_without_labels={len(games_without_labels)}"
    )
    print(f"wrote {out_dir / 'tag_vocab.json'}")
    print(f"wrote {out_dir / 'tag_labels.npz'}")

    # The role partition and the in-domain test pool only make sense for the Steam
    # `tags` source (mechanics/story/subjective are tag names). Skip for genres/categories.
    if args.source != "tags":
        if not args.no_groups or not args.no_test_games:
            print(f"skip groups/test_games: only built for --source tags (got {args.source})")
        return

    groups = build_tag_groups(kept)
    if not args.no_groups:
        atomic_write_bytes(
            lambda tmp: tmp.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"),
            out_dir / "tag_groups.json",
        )
        print(f"tag_groups: mechanics={len(groups['mechanics'])} story={len(groups['story'])} "
              f"subjective={len(groups['subjective'])} content={len(groups['content'])} drop={len(groups['drop'])}")
        print(f"wrote {out_dir / 'tag_groups.json'}")

    if not args.no_test_games:
        subjective_set = set(groups["subjective"])
        test_games, keep_tags = build_test_games(game_names, appids, games, kept, subjective_set)
        atomic_write_bytes(
            lambda tmp: tmp.write_text(json.dumps(test_games, ensure_ascii=False, indent=2), encoding="utf-8"),
            out_dir / "test_games.json",
        )
        atomic_write_bytes(
            lambda tmp: tmp.write_text(
                json.dumps({"tags": keep_tags, "dropped_emotional": sorted(subjective_set)},
                           ensure_ascii=False, indent=2),
                encoding="utf-8"),
            out_dir / "non_emotional_tags.json",
        )
        print(f"test_games: in-domain games={len(test_games)} (intersection with games.json) "
              f"non_emotional_tags={len(keep_tags)} dropped_emotional={len(subjective_set)}")
        print(f"wrote {out_dir / 'test_games.json'}")
        print(f"wrote {out_dir / 'non_emotional_tags.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-json", default=str(DEFAULT_GAMES_JSON))
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--source", choices=["tags", "genres", "categories"], default="tags")
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Keep tags that appear in at least this many games (ignored if --top-k > 0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If > 0, keep only the K most frequent tags instead of using --min-count.",
    )
    parser.add_argument("--target-mode", choices=["binary", "weight"], default="binary")
    parser.add_argument("--no-groups", action="store_true",
                        help="Skip writing tag_groups.json (mechanics/story/subjective/drop).")
    parser.add_argument("--no-test-games", action="store_true",
                        help="Skip writing test_games.json (in-domain pool) and non_emotional_tags.json.")
    return parser.parse_args()


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
