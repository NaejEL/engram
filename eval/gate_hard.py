# SPDX-License-Identifier: AGPL-3.0-or-later
"""X8.1 — Test causal de la loi « gates binaires » (protocole pré-enregistré,
journal 2026-08-21).

Intervention à un seul bouton : le facteur entropie garde son signal et son seuil
(mid 2.0), seule la pente du sigmoïde change (tau 0.5 lisse → 0.02 dur). Si
l'anomalie E3 disparaît, la non-monotonie en g est démontrée par intervention.

Configs : none / entropy tau 0.5 / entropy tau 0.02 / keysim (réf) /
two_factor tau_H 0.02. Mesures : E3 (10 secrets), fraction des lectures dans la
zone toxique g ∈ [0.05, 0.95], et E1 exact pour les configs candidates.

Usage : python eval/gate_hard.py [--model ... --layer ...]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from collateral import NEUTRAL_TEXT  # noqa: E402
from fact_injection import FACT_TEMPLATE, SECRETS, run_one  # noqa: E402
from engram import EngramConfig, EngramEngine  # noqa: E402

CONFIGS = [
    ("none", {"read_gate": "none"}),
    ("entropy tau=0.5", {"read_gate": "entropy", "gate_entropy_tau": 0.5}),
    ("entropy tau=0.02", {"read_gate": "entropy", "gate_entropy_tau": 0.02}),
    ("keysim (réf)", {"read_gate": "keysim"}),
    ("two_factor dur", {"read_gate": "two_factor", "gate_entropy_tau": 0.02}),
]
E1_CONFIGS = {"keysim (réf)", "two_factor dur"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    print(f"{'config':<18} {'E3':>9} {'zone toxique':>13} {'E1 exact':>18}")
    for label, overrides in CONFIGS:
        cfg = EngramConfig(track_keys=True, **overrides)
        if args.model:
            cfg.model_name = args.model
        if args.layer is not None:
            cfg.layer_index = args.layer
        engine = EngramEngine(cfg)

        def neutral_nll(log_gates: bool = False) -> float:
            engine.clear_context()
            engine.memory.gate_log = [] if log_gates else None
            recs = engine.stream(NEUTRAL_TEXT, read=True, write=False)
            return statistics.mean(r.nll for r in recs if r.nll is not None)

        engine.reset_memory()
        base = neutral_nll()
        deltas, toxic = [], []
        for secret in SECRETS:
            engine.reset_memory()
            engine.clear_context()
            engine.stream(FACT_TEMPLATE.format(secret=secret), force_write=True)
            deltas.append(neutral_nll(log_gates=True) - base)
            gates = engine.memory.gate_log or []
            if gates:
                toxic.append(sum(0.05 < g < 0.95 for g in gates) / len(gates))
        engine.memory.gate_log = None
        e3 = statistics.mean(deltas)
        tox = statistics.mean(toxic) if toxic else float("nan")

        e1_str = ""
        if label in E1_CONFIGS:
            e1 = [run_one(engine, s)["deltas"]["exact"] for s in SECRETS]
            e1_str = f"{statistics.mean(e1):+7.3f} ± {statistics.stdev(e1):.3f}"
        print(f"{label:<18} {e3:+9.4f} {tox:>12.1%} {e1_str:>18}")


if __name__ == "__main__":
    main()
