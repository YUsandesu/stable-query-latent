"""Coarse Steam tag families for description-facing tag prediction.

The fine Steam keyword vocabulary is too granular for short mechanism/story
descriptions. These families merge near-duplicates such as Card Battler and
Deckbuilding into Card Game, and keep only labels that are plausibly inferable
from gameplay/story text.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


COARSE_TAG_ALIASES = OrderedDict(
    [
        ("Card Game", ["Card Game", "Card Battler", "Deckbuilding"]),
        ("Roguelike", ["Rogue-like", "Rogue-lite", "Action Roguelike", "Procedural Generation", "Dungeon Crawler"]),
        ("Turn-Based", ["Turn-Based", "Turn-Based Strategy", "Turn-Based Combat", "Turn-Based Tactics"]),
        ("Strategy/Tactics", ["Strategy", "Tactical", "RTS", "Grand Strategy", "4X", "Real Time Tactics", "Tower Defense", "Wargame", "Tactical RPG", "Strategy RPG"]),
        ("RPG", ["RPG", "Action RPG", "JRPG", "CRPG", "Party-Based RPG", "Tactical RPG", "Strategy RPG", "MMORPG"]),
        ("Shooter/FPS", ["Shooter", "FPS", "Third-Person Shooter", "Looter Shooter"]),
        ("Action/Adventure", ["Action", "Adventure", "Action-Adventure", "Combat", "Hack and Slash", "Souls-like"]),
        ("Open World/Survival", ["Open World", "Exploration", "Survival", "Sandbox", "Open World Survival Craft", "Crafting", "Building", "Base-Building", "Hunting", "Fishing", "Mining"]),
        ("Co-op/Multiplayer", ["Co-op", "Online Co-Op", "Multiplayer", "Massively Multiplayer", "PvP", "PvE", "Team-Based", "Class-Based", "Local Co-Op", "Co-op Campaign", "Local Multiplayer", "Split Screen", "Competitive"]),
        ("Narrative/Choices", ["Visual Novel", "Interactive Fiction", "Story Rich", "Narrative", "Choices Matter", "Multiple Endings", "Choose Your Own Adventure", "Text-Based", "Conversation", "Narration", "Walking Simulator", "Point & Click", "Lore-Rich"]),
        ("Stealth/Immersive", ["Stealth", "Immersive Sim"]),
        ("Sci-fi/Cyberpunk", ["Sci-fi", "Cyberpunk", "Futuristic", "Space", "Robots", "Aliens", "Dystopian", "Post-apocalyptic"]),
        ("Fantasy", ["Fantasy", "Dark Fantasy", "Magic", "Dragons", "Mythology", "Medieval", "Gothic"]),
        ("Horror", ["Horror", "Survival Horror", "Psychological Horror", "Zombies", "Demons", "Vampire", "Supernatural"]),
        ("Historical/War", ["Historical", "Military", "War", "World War II", "Alternate History"]),
        ("Crime/Mystery", ["Crime", "Detective", "Mystery", "Assassin", "Political"]),
        ("Puzzle", ["Puzzle"]),
        ("Platformer", ["Platformer", "2D Platformer", "3D Platformer", "Metroidvania", "Parkour"]),
        ("Fighting/Melee", ["Fighting", "Beat 'em up", "3D Fighter", "Spectacle fighter", "Character Action Game", "Martial Arts", "Swordplay"]),
        ("Racing/Driving", ["Racing", "Driving", "Vehicular Combat", "Tanks"]),
        ("Simulation/Management", ["Simulation", "Management", "Resource Management", "City Builder", "Colony Sim", "Farming Sim", "Life Sim", "Automobile Sim", "Space Sim", "Automation", "Agriculture"]),
        ("Sports/Rhythm", ["Sports", "Rhythm"]),
        ("Bullet Hell", ["Bullet Hell"]),
    ]
)


COARSE_KEYWORDS = {
    "Card Game": ["card", "cards", "deck", "deckbuild", "卡牌", "牌组", "构筑"],
    "Roguelike": ["rogue", "roguelite", "roguelike", "肉鸽", "程序生成", "随机事件", "每次旅程", "run"],
    "Turn-Based": ["turn-based", "turn based", "回合制", "按速度行动"],
    "Strategy/Tactics": ["strategy", "tactical", "tactics", "策略", "战术", "队伍", "路线"],
    "RPG": ["rpg", "role-playing", "角色", "职业", "属性", "等级", "perk", "技能树", "义体", "装备"],
    "Shooter/FPS": ["shooter", "fps", "gun", "guns", "枪", "枪战", "射击", "手枪", "步枪"],
    "Action/Adventure": ["action", "adventure", "combat", "动作", "冒险", "战斗", "近战"],
    "Open World/Survival": ["open world", "survival", "sandbox", "exploration", "开放世界", "探索", "生存", "制作", "建造", "自由探索"],
    "Co-op/Multiplayer": ["co-op", "coop", "multiplayer", "pvp", "pve", "合作", "多人", "联机", "2-4"],
    "Narrative/Choices": ["story", "narrative", "choice", "ending", "lore", "故事", "剧情", "选择", "结局", "主线", "支线", "世界观", "对话"],
    "Stealth/Immersive": ["stealth", "immersive", "潜行", "黑客", "破解", "摄像头", "炮塔", "非致命"],
    "Sci-fi/Cyberpunk": ["sci-fi", "cyberpunk", "future", "space", "robot", "ai", "赛博朋克", "未来", "科幻", "义体", "夜之城", "荒坂", "黑墙"],
    "Fantasy": ["fantasy", "magic", "dragon", "medieval", "奇幻", "魔法", "王国", "公主", "国王", "巫师"],
    "Horror": ["horror", "zombie", "demon", "vampire", "恐怖", "僵尸", "恶魔", "吸血鬼"],
    "Historical/War": ["war", "military", "historical", "战争", "军事", "历史"],
    "Crime/Mystery": ["crime", "detective", "mystery", "assassin", "犯罪", "侦探", "神秘", "刺客"],
    "Puzzle": ["puzzle", "解谜", "谜题"],
    "Platformer": ["platform", "metroidvania", "parkour", "平台", "跑酷"],
    "Fighting/Melee": ["fighting", "martial", "sword", "melee", "格斗", "武士刀", "拳"],
    "Racing/Driving": ["racing", "driving", "vehicle", "赛车", "驾驶", "载具"],
    "Simulation/Management": ["simulation", "management", "resource", "模拟", "管理", "资源"],
    "Sports/Rhythm": ["sports", "rhythm", "体育", "节奏"],
    "Bullet Hell": ["bullet hell", "弹幕"],
}


def coarse_names() -> list[str]:
    return list(COARSE_TAG_ALIASES.keys())


def coarsen_tag_dict(tags: dict[str, float], keep_names: set[str] | None = None) -> dict[str, float]:
    """Map fine Steam tags to coarse families, keeping the strongest vote."""
    source = {str(name): float(value) for name, value in (tags or {}).items()}
    out: dict[str, float] = {}
    for coarse, aliases in COARSE_TAG_ALIASES.items():
        if keep_names is not None and coarse not in keep_names:
            continue
        weights = []
        if coarse in source:
            weights.append(source[coarse])
        weights.extend(source[name] for name in aliases if name in source)
        if weights:
            out[coarse] = max(weights)
    return out


def coarse_vector(fine_tags: set[str] | list[str], names: list[str] | None = None) -> np.ndarray:
    names = names or coarse_names()
    fine = set(str(tag) for tag in fine_tags)
    return np.array(
        [name in fine or any(alias in fine for alias in COARSE_TAG_ALIASES.get(name, [name])) for name in names],
        dtype=np.int8,
    )


def keyword_scores(text: str, tags: list[str]) -> np.ndarray:
    """Simple multilingual lexical prior for real descriptions."""
    lower = str(text or "").lower()
    scores = np.zeros(len(tags), dtype=np.float32)
    for index, tag in enumerate(tags):
        keywords = list(COARSE_TAG_ALIASES.get(tag, [])) + list(COARSE_KEYWORDS.get(tag, []))
        score = 0.0
        seen: set[str] = set()
        for keyword in keywords:
            key = keyword.lower()
            if key in seen:
                continue
            seen.add(key)
            if key and key in lower:
                score += 1.0 if len(key) > 3 else 0.5
        scores[index] = score
    max_score = float(scores.max()) if scores.size else 0.0
    return scores / max_score if max_score > 0 else scores
