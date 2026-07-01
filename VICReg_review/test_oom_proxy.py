import math

from VICReg_review import oom_proxy


def test_standard_transient_matches_full_chunk_attention_score():
    transient = oom_proxy.estimate_standard_transient_bytes(
        worst_game_sentences=534_641,
        view=0.8,
        num_latents=1024,
    )

    assert math.isclose(transient / oom_proxy.GIB, 13.05, rel_tol=0.01)


def test_view80_1024_latents_routes_to_split_on_48gb_card():
    gib = oom_proxy.GIB
    calib = {
        "_meta": {"input_dim": 1024},
        "1024|standard": {"C": 8.52 * 1024, "R": 0.31 * gib},
        "1024|split_recompute": {"C": 1.0, "R": 0.18 * gib},
    }

    plan = oom_proxy.plan_combo_chunked(
        calib,
        worst_game_sentences=534_641,
        free_vram_bytes=44.31 * gib,
        num_latents=1024,
        view=0.8,
        batch_size=128,
        safety=0.85,
        try_paired=False,
        total_sentences=2_422_551,
        cache_bytes=0,
        ram_budget=0,
    )

    assert plan["backward_mode"] == "split_recompute"
    assert plan["standard_peak_gib"] == 31.8
    assert plan["standard_transient_gib"] == 13.05
    assert plan["standard_required_gib"] > plan["budget_gib"]
    assert plan["stem_chunk_size"] >= int(534_641 * 0.8)
    assert plan["stem_chunk_size"] < 1_000_000


def test_same_combo_can_use_standard_when_required_memory_fits():
    gib = oom_proxy.GIB
    calib = {
        "_meta": {"input_dim": 1024},
        "1024|standard": {"C": 8.52 * 1024, "R": 0.31 * gib},
        "1024|split_recompute": {"C": 1.0, "R": 0.18 * gib},
    }

    plan = oom_proxy.plan_combo_chunked(
        calib,
        worst_game_sentences=534_641,
        free_vram_bytes=80.0 * gib,
        num_latents=1024,
        view=0.8,
        batch_size=128,
        safety=0.85,
        try_paired=False,
        total_sentences=2_422_551,
        cache_bytes=0,
        ram_budget=0,
    )

    assert plan["backward_mode"] == "standard"
    assert plan["standard_required_gib"] < plan["budget_gib"]
