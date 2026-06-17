import argparse
import csv
import json
import random
import re
from pathlib import Path


SCORE_COLUMNS = [
    "psychological_meaning",
    "psychological_mastery",
    "psychological_curiosity",
    "psychological_autonomy",
    "psychological_immersion",
    "functional_progress_feedback",
    "functional_ease_of_control",
    "functional_audiovisual_appeal",
    "functional_goals_and_rules",
    "functional_challenge",
]

GAME_KEY_COLUMN = "game_id"
AVERAGE_COUNT_COLUMN = "score_sample_count"
EXCLUDED_OUTPUT_COLUMNS = {"game_title"}


OPENINGS = [
    "This is a {genre}.",
    "Overall, this {genre} offers a clear player experience.",
    "As a {genre}, it leaves a fairly distinct impression.",
    "This {genre} has noticeable strengths and weaknesses across several play dimensions.",
]


CONNECTORS = [
    "At the same time, ",
    "In addition, ",
    "During play, ",
    "From the player's perspective, ",
    "More specifically, ",
    "In practice, ",
    "Another point is that ",
    "",
]


SYNONYM_GROUPS = [
    ["clear", "specific", "precise", "detailed", "well-defined"],
    ["difficult", "challenging", "demanding", "hard to complete", "tough"],
    ["smooth", "fluid", "responsive", "natural", "easy to handle"],
    ["accurate", "precise", "reliable", "consistent"],
    ["simple", "straightforward", "easy to understand", "lightweight"],
    ["appealing", "attractive", "engaging", "pleasant"],
    ["meaningful", "valuable", "worthwhile", "rewarding"],
    ["progress feedback", "progress signals", "advancement feedback", "status feedback"],
    ["rules", "mechanics", "systems", "constraints"],
    ["visuals", "art direction", "visual presentation", "graphics"],
    ["audio", "sound design", "music and effects", "sound feedback"],
]


TEMPLATES = {
    "psychological_meaning": {
        "very_low": [
            "The game communicates almost no sense of meaning, so player actions feel unimportant.",
            "The experience feels rather empty beyond completing the immediate tasks.",
            "Players are unlikely to feel emotional or personal value in what they are doing.",
        ],
        "low": [
            "The game has some meaningful elements, but they remain weak.",
            "Players may occasionally feel that their actions matter, but the feeling is inconsistent.",
            "The theme is present, yet the overall sense of meaning is rather limited.",
        ],
        "medium": [
            "The game provides a basic sense of purpose and meaning.",
            "Players can understand what they are doing, though the emotional weight is moderate.",
            "The meaning of the experience is balanced rather than especially strong or absent.",
        ],
        "high": [
            "The game makes player choices and actions feel meaningful.",
            "The goals and play structure create a strong sense of value.",
            "Players can feel a clear connection between effort, progress, and emotional payoff.",
        ],
        "very_high": [
            "The game creates a very strong sense that player actions matter.",
            "It gives players a deep feeling of value, purpose, and emotional investment.",
            "The theme and goals work together well, making the experience feel highly meaningful.",
        ],
    },
    "psychological_mastery": {
        "very_low": [
            "Players rarely feel that they are improving or becoming more skilled.",
            "The game gives little feedback that supports a sense of growth.",
            "It is hard for players to build a feeling of mastery.",
        ],
        "low": [
            "The game offers a small amount of mastery, but skill growth is not very visible.",
            "Players can learn a few techniques, yet the improvement feedback is limited.",
            "The sense of becoming more competent is present but weak.",
        ],
        "medium": [
            "The game supports a basic learning and mastery process.",
            "Players can gradually understand the rules and feel a moderate amount of improvement.",
            "Skill growth feels reasonably balanced.",
        ],
        "high": [
            "The game clearly lets players feel that they are becoming better over time.",
            "As play continues, players gain a strong sense of control and competence.",
            "The design supports effective learning and rewarding skill development.",
        ],
        "very_high": [
            "The game delivers a very strong sense of mastery and skill growth.",
            "Players can clearly feel themselves learning, practicing, and controlling the system.",
            "The experience strongly rewards effort with a feeling of increasing competence.",
        ],
    },
    "psychological_curiosity": {
        "very_low": [
            "The game does little to create curiosity and quickly feels repetitive.",
            "Players have little reason to explore new mechanics or content.",
            "There is not much mystery, surprise, or freshness in the experience.",
        ],
        "low": [
            "The game has a little room for exploration, but curiosity is weak.",
            "Players may occasionally wonder what comes next, though the pull is limited.",
            "New content appears, but it does not create much desire to explore.",
        ],
        "medium": [
            "The game maintains a basic level of curiosity.",
            "Some levels, mechanics, or situations can interest the player.",
            "The sense of discovery is moderate.",
        ],
        "high": [
            "The game keeps players curious about what they will find next.",
            "Players are likely to look forward to new mechanics, levels, or strategies.",
            "Variation in the content creates a strong sense of freshness.",
        ],
        "very_high": [
            "The game strongly encourages exploration and discovery.",
            "It is full of new situations and unknown outcomes that keep players interested.",
            "Players are easily drawn in by new mechanics, surprises, and changes.",
        ],
    },
    "psychological_autonomy": {
        "very_low": [
            "The game gives players almost no choice, so autonomy feels very low.",
            "Players mostly have to follow one fixed path or method.",
            "The experience feels restricted, with little freedom in how to play.",
        ],
        "low": [
            "The game offers a few choices, but the space for autonomy is limited.",
            "Players can make some decisions, though the overall route feels fixed.",
            "Freedom is low, and different choices do not change the experience very much.",
        ],
        "medium": [
            "The game provides a basic amount of choice and decision-making.",
            "Players can play in their own way within a limited range.",
            "The sense of autonomy is moderate.",
        ],
        "high": [
            "The game gives players many choices and a strong sense of agency.",
            "Players can solve problems or pursue goals in different ways.",
            "The decision space is broad enough to make play feel flexible.",
        ],
        "very_high": [
            "The game gives players a very strong sense of freedom and autonomy.",
            "Players can act and decide according to their own preferences.",
            "The experience makes players feel that they control the pace and direction of play.",
        ],
    },
    "psychological_immersion": {
        "very_low": [
            "The game rarely feels immersive, and attention is easy to break.",
            "Players are unlikely to feel absorbed in the world or the play flow.",
            "The overall sense of immersion is very weak.",
        ],
        "low": [
            "The game has some involvement, but immersion remains limited.",
            "Players may focus briefly, but it is hard to stay absorbed for long.",
            "The feeling of being pulled into the experience is weak.",
        ],
        "medium": [
            "The game provides a basic immersive experience.",
            "Players can stay reasonably focused while playing.",
            "The sense of involvement is moderate.",
        ],
        "high": [
            "The game makes it fairly easy for players to become immersed.",
            "Players can naturally settle into the rhythm and situation of play.",
            "The experience feels coherent and supports strong concentration.",
        ],
        "very_high": [
            "The game is highly immersive and can make players lose track of time.",
            "It draws players deeply into the mechanics, atmosphere, and pacing.",
            "Players can feel strongly absorbed by the overall experience.",
        ],
    },
    "functional_progress_feedback": {
        "very_low": [
            "The game provides almost no clear progress feedback.",
            "Players may not know how far they have advanced or what they have achieved.",
            "Rewards, milestones, and status signals are very weak.",
        ],
        "low": [
            "The game gives some progress feedback, but it is not very clear.",
            "Players can see some results, though the sense of advancement is limited.",
            "Progress signals are sparse and may leave players unsure of their direction.",
        ],
        "medium": [
            "The game provides basic information about progress and results.",
            "Players can generally understand how far they have advanced.",
            "Rewards and milestone feedback are moderate.",
        ],
        "high": [
            "The game gives clear progress feedback and makes advancement easy to recognize.",
            "Rewards and milestones are communicated effectively.",
            "Players can understand their current goals and completion status.",
        ],
        "very_high": [
            "The game provides very clear progress and reward feedback.",
            "Each step forward receives a noticeable response.",
            "Milestones, achievements, and result signals are detailed and satisfying.",
        ],
    },
    "functional_ease_of_control": {
        "very_low": [
            "The controls feel very awkward and strongly harm the experience.",
            "Players struggle to perform the actions they intend.",
            "Input feedback is confusing, and the game feels hard to control.",
        ],
        "low": [
            "The controls feel somewhat awkward and require extra adjustment.",
            "The control scheme is usable, but it is not very smooth.",
            "Players may sometimes feel blocked by control issues.",
        ],
        "medium": [
            "The controls are acceptable and support basic play.",
            "The control scheme is not overly complex and becomes manageable with practice.",
            "The overall control feel is moderate.",
        ],
        "high": [
            "The game controls smoothly and provides clear input feedback.",
            "Players can perform intended actions with relative ease.",
            "The control design reduces unnecessary friction.",
        ],
        "very_high": [
            "The controls are highly intuitive, smooth, and responsive.",
            "Players can start playing naturally with little extra adjustment.",
            "Input feedback is precise, making the game feel very comfortable to control.",
        ],
    },
    "functional_audiovisual_appeal": {
        "very_low": [
            "The visual and audio presentation has very little appeal.",
            "The audiovisual style feels rough and does not strengthen the experience.",
            "The art and sound design are unlikely to leave much impression.",
        ],
        "low": [
            "The audiovisual presentation is somewhat weak and only mildly appealing.",
            "The visuals or audio show some design, but they are not very memorable.",
            "The art and sound support basic play, though their impact is limited.",
        ],
        "medium": [
            "The visuals and audio are serviceable and fairly standard.",
            "The audiovisual design meets basic expectations.",
            "The art direction and sound design are moderate.",
        ],
        "high": [
            "The game has appealing visuals and sound.",
            "The art style and audio noticeably improve the experience.",
            "The audiovisual design feels polished and strengthens the atmosphere.",
        ],
        "very_high": [
            "The game has excellent audiovisual appeal.",
            "The art, animation, and sound design work together very well.",
            "The polished visual and audio presentation strongly improves immersion and enjoyment.",
        ],
    },
    "functional_goals_and_rules": {
        "very_low": [
            "The goals and rules are unclear, so players may not know what to do.",
            "The rules are confusing and difficult to understand.",
            "The game lacks clear direction and makes the objective hard to identify.",
        ],
        "low": [
            "The goals and rules are somewhat vague and require extra guessing.",
            "The basic rules can be understood, but explanation and guidance are limited.",
            "Players may sometimes be unsure about the next step.",
        ],
        "medium": [
            "The goals and rules are mostly understandable.",
            "Players can grasp the main mechanics and success conditions.",
            "The rule explanation is moderate.",
        ],
        "high": [
            "The goals are clear, and the rules are easy to understand.",
            "Players can quickly see what they should do and how to proceed.",
            "The rules and objectives are well-defined and reduce confusion.",
        ],
        "very_high": [
            "The goals are very clear, and the rules are explained in detail.",
            "Players can quickly understand the mechanics, limits, and win conditions.",
            "Goals, rules, and guidance work together to create a very strong sense of direction.",
        ],
    },
    "functional_challenge": {
        "very_low": [
            "The game offers almost no challenge and feels very easy.",
            "The levels are simple, and players rarely face meaningful obstacles.",
            "There is little pressure or difficulty in the experience.",
        ],
        "low": [
            "The game has a small amount of challenge, but most content is easy.",
            "Players occasionally need to think or practice, though the difficulty stays low.",
            "The overall challenge is limited, with little pressure from failure.",
        ],
        "medium": [
            "The game is moderately challenging while remaining balanced.",
            "Some levels require thought, but they do not feel too difficult.",
            "The challenge level is fair, with some resistance but not too much frustration.",
        ],
        "high": [
            "The game includes many difficult levels.",
            "The game is challenging and asks players to retry, learn, and improve.",
            "Players need focus and skill to get through the harder parts.",
        ],
        "very_high": [
            "The game is very difficult and demands strong skill and patience.",
            "Its levels and tasks are highly challenging, with frequent failure and retrying.",
            "It constantly tests the player's reactions, understanding, and strategy.",
        ],
    },
}


def score_to_five_point_integer(score):
    if score <= -1.8:
        return 1
    if score <= -0.6:
        return 2
    if score < 0.6:
        return 3
    if score < 1.8:
        return 4
    return 5


def five_point_integer_to_bucket(score):
    bucket_by_score = {
        1: "very_low",
        2: "low",
        3: "medium",
        4: "high",
        5: "very_high",
    }
    try:
        return bucket_by_score[int(score)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Expected a 1-5 integer score, got {score!r}") from exc


def parse_score(value, column):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} contains a non-numeric score: {value!r}") from exc


def genre_label(row):
    raw = (row.get("genre_name") or "game").strip()
    return raw or "game"


def diversify_sentence(sentence, rng, replacement_rate=0.55):
    for group in SYNONYM_GROUPS:
        patterns = {
            word: re.compile(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])")
            for word in group
        }
        matches = [word for word, pattern in patterns.items() if pattern.search(sentence)]
        if not matches:
            continue

        source = max(matches, key=len)
        if rng.random() > replacement_rate:
            continue

        choices = [word for word in group if word != source]
        if choices:
            sentence = patterns[source].sub(rng.choice(choices), sentence, count=1)

    return sentence


def lower_first_alpha(text):
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.lower() + text[index + 1 :]
    return text


def generate_text(
    row,
    rng,
    include_opening=True,
    shuffle_sentences=True,
    lexical_variation=True,
):
    sentences = []

    if include_opening:
        opening = rng.choice(OPENINGS).format(genre=genre_label(row))
        if lexical_variation:
            opening = diversify_sentence(opening, rng)
        sentences.append(opening)

    feature_sentences = []
    for column in SCORE_COLUMNS:
        score = parse_score(row.get(column), column)
        bucket = five_point_integer_to_bucket(score)
        sentence = rng.choice(TEMPLATES[column][bucket])
        if lexical_variation:
            sentence = diversify_sentence(sentence, rng)
        connector = rng.choice(CONNECTORS)
        if connector:
            sentence = lower_first_alpha(sentence)
        feature_sentences.append(f"{connector}{sentence}" if connector else sentence)

    if shuffle_sentences:
        rng.shuffle(feature_sentences)

    sentences.extend(feature_sentences)
    return " ".join(sentences)


def read_rows(input_path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = [GAME_KEY_COLUMN] + SCORE_COLUMNS
        missing = [column for column in required_columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        return list(reader), list(reader.fieldnames or [])


def average_rows_by_game(rows):
    grouped_rows = {}
    for row in rows:
        game_key = (row.get(GAME_KEY_COLUMN) or "").strip()
        if not game_key:
            raise ValueError(f"Column {GAME_KEY_COLUMN!r} contains an empty game id.")
        grouped_rows.setdefault(game_key, []).append(row)

    averaged_rows = []
    for game_rows in grouped_rows.values():
        output_row = dict(game_rows[0])
        for column in SCORE_COLUMNS:
            total = sum(parse_score(row.get(column), column) for row in game_rows)
            mapped_score = score_to_five_point_integer(total / len(game_rows))
            output_row[column] = str(mapped_score)
        output_row[AVERAGE_COUNT_COLUMN] = str(len(game_rows))
        if "sample_index" in output_row:
            output_row["sample_index"] = "average"
        averaged_rows.append(output_row)

    return averaged_rows


def write_csv(output_path, rows, fieldnames):
    output_fields = ["generated_text"] + fieldnames
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(output_path, rows):
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(
    input_path,
    variants_per_row,
    seed,
    include_opening,
    shuffle_sentences,
    lexical_variation,
):
    rng = random.Random(seed)
    source_rows, fieldnames = read_rows(input_path)
    averaged_rows = average_rows_by_game(source_rows)
    generated_rows = []

    for row in averaged_rows:
        for variant_index in range(variants_per_row):
            output_row = {
                key: value
                for key, value in row.items()
                if key not in EXCLUDED_OUTPUT_COLUMNS
            }
            output_row["generated_text"] = generate_text(
                row,
                rng,
                include_opening=include_opening,
                shuffle_sentences=shuffle_sentences,
                lexical_variation=lexical_variation,
            )
            output_row["text_variant_index"] = str(variant_index + 1)
            generated_rows.append(output_row)

    output_fieldnames = [
        fieldname
        for fieldname in fieldnames
        if fieldname not in EXCLUDED_OUTPUT_COLUMNS
    ]
    if AVERAGE_COUNT_COLUMN not in output_fieldnames:
        output_fieldnames.append(AVERAGE_COUNT_COLUMN)
    output_fieldnames.append("text_variant_index")
    return generated_rows, output_fieldnames


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo English game-description text from benchmark scores."
    )
    parser.add_argument("--input", default="bench_data/benchmark.csv", help="Input benchmark CSV path.")
    parser.add_argument(
        "--output",
        default="bench_data/pseudo_text_data.csv",
        help="Output path. Use .csv or .jsonl.",
    )
    parser.add_argument(
        "--variants-per-row",
        type=int,
        default=1,
        help="How many text variants to generate for each averaged game row.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-opening",
        action="store_true",
        help="Do not prepend a genre-based opening sentence.",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep feature sentences in column order instead of shuffling them.",
    )
    parser.add_argument(
        "--no-lexical-variation",
        action="store_true",
        help="Disable random synonym replacement inside selected templates.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.variants_per_row < 1:
        raise ValueError("--variants-per-row must be at least 1")

    generated_rows, fieldnames = build_dataset(
        input_path=input_path,
        variants_per_row=args.variants_per_row,
        seed=args.seed,
        include_opening=not args.no_opening,
        shuffle_sentences=not args.keep_order,
        lexical_variation=not args.no_lexical_variation,
    )

    if output_path.suffix.lower() == ".jsonl":
        write_jsonl(output_path, generated_rows)
    elif output_path.suffix.lower() == ".csv":
        write_csv(output_path, generated_rows, fieldnames)
    else:
        raise ValueError("Output file must end with .csv or .jsonl")

    print(f"Generated {len(generated_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
