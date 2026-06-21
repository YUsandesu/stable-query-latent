# Latent Query Benchmark

This project uses a **latent query** architecture to predict 1–5 ratings across ten scoring dimensions from sentence-level text embeddings.
The input is a sequence of sentence vectors split from pseudo text; the output is a 5-class classification for each of the 10 dimensions.

## Task Definition

- **Input:** Each sample is a sequence of sentence embeddings (`input_dim=1024`) derived from pseudo text.
- **Output:** 10 scoring dimensions, each treated as a 5-class classification (scores 1–5).
- **Output size:** `10 × 5 = 50`

Scoring columns:

`psychological_meaning`, `psychological_mastery`, `psychological_curiosity`, `psychological_autonomy`, `psychological_immersion`,
`functional_progress_feedback`, `functional_ease_of_control`, `functional_audiovisual_appeal`, `functional_goals_and_rules`, `functional_challenge`

## Model Details

The core model is defined in [`latent_query_model.py`](./latent_query_model.py).

### 1. Input Projection

Each 1024-dimensional sentence vector is linearly projected into a hidden space:

- `Linear(input_dim → hidden_dim)`
- Default `hidden_dim=64`

### 2. Three-Stage Latent Query Cross-Attention

The model uses 3 Cross-Attention Blocks with query counts of:

- `32`
- `16`
- `8`

Each block contains:

- Learnable queries: `nn.Parameter(num_queries, dim)`
- `LayerNorm` on both queries and context
- `MultiheadAttention`
- Residual connection
- FFN: `LayerNorm → Linear → GELU → Dropout → Linear → Dropout`

Default hyperparameters:

- `num_heads=8`
- `dropout=0.0`
- `mlp_ratio=4.0`

### 3. Flatten Head

The 8 latent vectors from the final block are flattened:

- `LayerNorm`
- `Linear(8 × hidden_dim → flat_dim)`
- `GELU`
- `Dropout`
- `LayerNorm`
- `Linear(flat_dim → output_dim)`

Default `flat_dim=128`.

### 4. Training Objective

Each scoring dimension is treated as a 5-class classification:

- Loss: `CrossEntropyLoss`
- Labels are in range `1..5`
- Prediction: `argmax + 1`

## Data Pipeline

1. `generate_pseudo_text.py` — generates pseudo text from `benchmark.csv`
2. `embed_pseudo_text_sentences.py` — splits into sentences and produces sentence embeddings
3. `build_h5.py` — packages embeddings, masks, and labels into HDF5
4. `test_latent_query_model.py` — loads HDF5, trains, and evaluates the model

## Pseudo Text Generation

Pseudo text is synthesized by `generate_pseudo_text.py` from `benchmark.csv`. Rather than translating raw scores literally, it maps each sample's 10 rating dimensions into readable English description sentences.

### 1. Input and Grouping

- Input must contain `game_id` and the 10 scoring columns.
- If a `game_id` spans multiple rows, they are aggregated first: each scoring column is averaged and rounded back to an integer in 1–5.
- A `score_sample_count` column is added to indicate how many original rows were averaged.
- `game_title` is not written to output.

### 2. Score-to-Bucket Mapping

Each score is mapped to one of five buckets:

| Threshold | Bucket | Label |
|---|---|---|
| `<= -1.8` | `1` | `very_low` |
| `<= -0.6` | `2` | `low` |
| `< 0.6` | `3` | `medium` |
| `< 1.8` | `4` | `high` |
| `>= 1.8` | `5` | `very_high` |

A sentence is then sampled from the template pool for that dimension and bucket.

### 3. Sentence Structure

Each pseudo text consists of:

1. An opening sentence
2. One sentence per scoring dimension (10 total)

The opening sentence is chosen randomly from:

- `This is a {genre}.`
- `Overall, this {genre} offers a clear player experience.`
- `As a {genre}, it leaves a fairly distinct impression.`
- `This {genre} has noticeable strengths and weaknesses across several play dimensions.`

Each dimension sentence may be prefixed by a random connector such as:
`At the same time, ` / `In addition, ` / `During play, ` / `From the player's perspective, ` / `More specifically, ` / `In practice, ` / `Another point is that ` / (empty)

When a connector is added, the first letter of the following sentence is lowercased.

### 4. Lexical Variation

To reduce template artifacts, key words are lightly substituted with synonyms, for example:

- `clear` / `specific` / `precise` / `detailed`
- `difficult` / `challenging` / `demanding`
- `smooth` / `fluid` / `responsive`
- `meaningful` / `valuable` / `rewarding`
- `rules` / `mechanics` / `systems`
- `visuals` / `art direction` / `graphics`

Substitution happens randomly with a default probability of ~55%.

### 5. Order and Variants

- Dimension sentence order is shuffled by default; use `--keep-order` to preserve column order.
- The opening sentence is included by default; use `--no-opening` to omit it.
- Lexical variation is enabled by default; use `--no-lexical-variation` to disable.
- `--variants-per-row` generates multiple distinct pseudo texts per aggregated game record.

### 6. Output Fields

The output CSV/JSONL includes:

- `generated_text`
- Original identifier fields: `game_id`, `genre_id`, `genre_name`, etc.
- 10 scoring columns
- `score_sample_count`
- `text_variant_index`

### 7. Example Commands

Generate one pseudo text per game:

```bash
python generate_pseudo_text.py --input benchmark.csv --output pseudo_text_data_one_per_game.csv
```

Generate multiple variants per game:

```bash
python generate_pseudo_text.py --input benchmark.csv --output pseudo_text_data_multi.csv --variants-per-row 4
```

The full pipeline is: generate pseudo text → split sentences and embed → pack into HDF5.

Default HDF5 input file: `benchmark_sentence_latent_query_multi.h5`

## Training Configuration

| Parameter | Value |
|---|---|
| `epochs` | 400 |
| `batch_size` | 128 |
| `learning_rate` | 3e-4 |
| `min_learning_rate` | 1e-5 |
| `test_ratio` | 0.2 |
| `seed` | 42 |
| `split_by` | `score_combo` |

Optimizer and scheduler:

- `AdamW(weight_decay=1e-4)`
- `CosineAnnealingLR`

## Results

Best checkpoint saved at: `latent_query_benchmark_multi_classifier.pt`

Model configuration:

| Parameter | Value |
|---|---|
| `input_dim` | 1024 |
| `score_dim` | 10 |
| `output_dim` | 50 |
| `hidden_dim` | 64 |
| `flat_dim` | 128 |
| `query_sizes` | (32, 16, 8) |
| `num_heads` | 8 |
| `dropout` | 0.0 |
| `split` | `score_combo`, 304 groups, 61 test groups |

Best test results at `epoch=280`:

| Metric | Value |
|---|---|
| `train_ce` | 0.001558 |
| `test_ce` | 0.198992 |
| `test_mae` | 0.088788 |
| `test_accuracy` | 0.942424 |

Per-dimension results:

| score_column | test_ce | test_mae | test_accuracy |
|---|---:|---:|---:|
| psychological_meaning | 0.159680 | 0.057576 | 0.966667 |
| psychological_mastery | 0.050443 | 0.042424 | 0.981818 |
| psychological_curiosity | 0.377497 | 0.166667 | 0.906061 |
| psychological_autonomy | 0.241912 | 0.093939 | 0.939394 |
| psychological_immersion | 0.345395 | 0.181818 | 0.872727 |
| functional_progress_feedback | 0.215814 | 0.090909 | 0.933333 |
| functional_ease_of_control | 0.081608 | 0.054545 | 0.975758 |
| functional_audiovisual_appeal | 0.171714 | 0.075758 | 0.954545 |
| functional_goals_and_rules | 0.125997 | 0.069697 | 0.939394 |
| functional_challenge | 0.219857 | 0.054545 | 0.954545 |

## Weight Attribution Test

A grad-times-input backward attribution was performed on row 0 of `pseudo_text_data_multi.csv`, reusing the sentence embeddings from `benchmark_sentence_latent_query_multi.h5`. The target class for each dimension is the model's predicted score.

Sample info:

- `row_index=0`, `game_id=72`, `game_name=21`, `genre_name=Puzzle game`, `text_variant_index=1`
- `checkpoint=latent_query_benchmark_multi_classifier.pt`

Input text:

> This is a Puzzle game. Another point is that the decision space is broad enough to make play feel flexible. During play, goals, systems, and guidance work together to create a very strong sense of direction. During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. At the same time, the game is highly immersive and can make players lose track of time. From the player's perspective, the game provides very specific progress and reward feedback. In addition, players are easily drawn in by new constraints, surprises, and changes. From the player's perspective, the game makes player choices and actions feel meaningful. Input feedback is accurate, making the game feel very comfortable to control. In practice, the game is moderately challenging while remaining balanced. During play, the game clearly lets players feel that they are becoming better over time.

Predictions on 4 sampled dimensions all match the CSV labels:

| score_column | true_score | predicted_score | predicted_probability |
|---|---:|---:|---:|
| psychological_meaning | 4 | 4 | 0.998968 |
| psychological_autonomy | 4 | 4 | 0.998343 |
| functional_goals_and_rules | 5 | 5 | 0.999368 |
| functional_challenge | 3 | 3 | 0.997105 |

Top-10 sentence attributions per dimension. `importance` = sentence-level sum of `abs(grad × input)`; `signed` = sentence-level sum of `grad × input` (positive = pushes target logit up, negative = pushes it down).

### `psychological_meaning`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 8 | 28.469210 | -0.018509 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 2 | 2 | 23.160913 | -0.006788 | Another point is that the decision space is broad enough to make play feel flexible. |
| 3 | 4 | 20.712833 | 0.006157 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |
| 4 | 5 | 20.285248 | -0.009097 | At the same time, the game is highly immersive and can make players lose track of time. |
| 5 | 11 | 18.705542 | 0.001047 | During play, the game clearly lets players feel that they are becoming better over time. |
| 6 | 7 | 14.920107 | -0.007795 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 7 | 6 | 12.899150 | -0.004846 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 8 | 9 | 11.894173 | 0.003073 | Input feedback is accurate, making the game feel very comfortable to control. |
| 9 | 10 | 11.708222 | 0.000188 | In practice, the game is moderately challenging while remaining balanced. |
| 10 | 1 | 10.869076 | 0.002041 | This is a Puzzle game. |

### `psychological_autonomy`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 2 | 26.957672 | 0.012270 | Another point is that the decision space is broad enough to make play feel flexible. |
| 2 | 8 | 19.949272 | 0.013524 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 3 | 9 | 18.452248 | 0.015421 | Input feedback is accurate, making the game feel very comfortable to control. |
| 4 | 5 | 17.724504 | -0.001987 | At the same time, the game is highly immersive and can make players lose track of time. |
| 5 | 11 | 16.456701 | 0.001752 | During play, the game clearly lets players feel that they are becoming better over time. |
| 6 | 4 | 16.389551 | 0.011282 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |
| 7 | 7 | 15.521434 | 0.002830 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 8 | 3 | 14.815602 | 0.008010 | During play, goals, systems, and guidance work together to create a very strong sense of direction. |
| 9 | 6 | 13.883622 | -0.000842 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 10 | 1 | 8.039020 | 0.002425 | This is a Puzzle game. |

### `functional_goals_and_rules`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 3 | 49.745472 | 0.007319 | During play, goals, systems, and guidance work together to create a very strong sense of direction. |
| 2 | 6 | 21.964949 | 0.000058 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 3 | 11 | 20.030167 | 0.013754 | During play, the game clearly lets players feel that they are becoming better over time. |
| 4 | 2 | 14.509171 | 0.002870 | Another point is that the decision space is broad enough to make play feel flexible. |
| 5 | 9 | 12.777904 | 0.004639 | Input feedback is accurate, making the game feel very comfortable to control. |
| 6 | 4 | 11.495865 | 0.003872 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |
| 7 | 7 | 11.023729 | 0.003466 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 8 | 8 | 10.983618 | 0.009394 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 9 | 10 | 7.959258 | 0.004081 | In practice, the game is moderately challenging while remaining balanced. |
| 10 | 5 | 7.674747 | -0.001358 | At the same time, the game is highly immersive and can make players lose track of time. |

### `functional_challenge`

| rank | sentence # | importance | signed | sentence |
|---:|---:|---:|---:|---|
| 1 | 9 | 34.623436 | 0.010462 | Input feedback is accurate, making the game feel very comfortable to control. |
| 2 | 11 | 21.271782 | -0.006053 | During play, the game clearly lets players feel that they are becoming better over time. |
| 3 | 2 | 20.202581 | -0.002018 | Another point is that the decision space is broad enough to make play feel flexible. |
| 4 | 8 | 19.821337 | -0.007812 | From the player's perspective, the game makes player choices and actions feel meaningful. |
| 5 | 10 | 19.815519 | -0.003833 | In practice, the game is moderately challenging while remaining balanced. |
| 6 | 7 | 19.667583 | -0.012425 | In addition, players are easily drawn in by new constraints, surprises, and changes. |
| 7 | 3 | 17.551899 | -0.003004 | During play, goals, systems, and guidance work together to create a very strong sense of direction. |
| 8 | 1 | 16.997639 | 0.004039 | This is a Puzzle game. |
| 9 | 6 | 16.036831 | -0.003248 | From the player's perspective, the game provides very specific progress and reward feedback. |
| 10 | 4 | 14.870131 | -0.009054 | During play, the polished visual and sound design presentation strongly improves immersion and enjoyment. |

To reproduce:

```bash
python visualize_backprop_attribution.py --text "<sample text>" --score-column functional_goals_and_rules --device cpu
```

## Usage

Train and evaluate:

```bash
python test_latent_query_model.py
```

To override hyperparameters:

```bash
python test_latent_query_model.py --epochs 500 --batch-size 64 --device cuda
```

## Key Files

- [`latent_query_model.py`](./latent_query_model.py)
- [`test_latent_query_model.py`](./test_latent_query_model.py)
- [`build_h5.py`](./build_h5.py)
- [`embed_pseudo_text_sentences.py`](./embed_pseudo_text_sentences.py)
- [`generate_pseudo_text.py`](./generate_pseudo_text.py)
- [`latent_query_benchmark_multi_classifier.pt`](./latent_query_benchmark_multi_classifier.pt)
- [`latent_query_training_history_multi_classifier.txt`](./latent_query_training_history_multi_classifier.txt)
- [`latent_query_per_dim_multi_classifier.txt`](./latent_query_per_dim_multi_classifier.txt)
