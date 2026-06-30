"""Clean redesign of the VICReg review sweep.

Replaces the monkey-patched run_data_view_sweep + sweep_cloud orchestration with
three explicit layers:

* config.py    -- one sweep.yaml = the whole experiment (declarative).
* ledger.py    -- append-only JSONL state machine; the single source of truth
                  for resume. 'running' records the worker PID + heartbeat.
* planner.py   -- wraps oom_proxy: per-combo memory plan (backward_mode, paired,
                  stem_chunk_size). Chunk, never cap.

The worker (loads the embedding cache once, trains combos) and supervisor
(spawns/watches/restarts the worker, owns the ledger) live alongside.
"""
