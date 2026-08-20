"""X8 — Banc d'essai du gate de lecture (protocole pré-enregistré, EXTENSIONS.md §X8).

Quatre modes comparés à config égale par ailleurs :
  none       = cap global seul (comportement X1b, la référence)
  entropy    = gate incertitude seule (baseline — angle mort attendu : E1c)
  keysim     = gate pertinence seule (baseline — risque X9 : sature sur mots communs)
  two_factor = soft-OR des deux (le candidat)

Trois évals par mode :
  E1  : Δlogp exact moyen, 10 secrets (protocole standard).
  E3  : dommage collatéral sur texte neutre (cible : ≤ +0.05, et l'écraser).
  E1c : correction de fait sous confiance erronée — le cortex prédit « Paris » avec
        une entropie très basse après « The capital of France is » ; M détient la
        mise à jour « Marseille ». Le gate entropie-seule doit échouer ici (il coupe
        la lecture au moment exact où elle sert), le two_factor doit passer.

Usage : python eval/read_gate.py [--model ... --layer ... --cap ...]
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

MODES = ["none", "entropy", "keysim", "two_factor"]

E1C_FACT = ("Breaking news: the capital of France has moved. "
            "The capital of France is now Marseille.")
E1C_QUESTION = "The capital of France is"


def eval_e1(engine: EngramEngine) -> tuple[float, float]:
    deltas = [run_one(engine, s)["deltas"]["exact"] for s in SECRETS]
    return statistics.mean(deltas), statistics.stdev(deltas)


def eval_e3(engine: EngramEngine) -> float:
    def neutral_nll() -> float:
        engine.clear_context()
        recs = engine.stream(NEUTRAL_TEXT, read=True, write=False)
        return statistics.mean(r.nll for r in recs if r.nll is not None)

    engine.reset_memory()
    base = neutral_nll()
    deltas = []
    for secret in SECRETS:
        engine.reset_memory()
        engine.clear_context()
        engine.stream(FACT_TEMPLATE.format(secret=secret), force_write=True)
        deltas.append(neutral_nll() - base)
    return statistics.mean(deltas)


def eval_e1c(engine: EngramEngine) -> tuple[float, float]:
    """Retourne (Δlogp de « Marseille », Δlogp de « Paris ») avec M chargée du
    fait de correction, vs M=0 — mêmes prompts exacts."""
    def both(label_reset: bool) -> tuple[float, float]:
        if label_reset:
            engine.reset_memory()
        out = []
        for word in (" Marseille", " Paris"):
            engine.clear_context()
            out.append(engine.logprob_continuation(E1C_QUESTION, word)[0])
        return out[0], out[1]

    engine.reset_memory()
    engine.clear_context()
    engine.stream(E1C_FACT, force_write=True)
    m_mars, m_paris = both(label_reset=False)
    b_mars, b_paris = both(label_reset=True)
    return m_mars - b_mars, m_paris - b_paris


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--cap", type=float, default=None)
    args = parser.parse_args()

    rows = []
    for mode in MODES:
        cfg = EngramConfig(read_gate=mode, track_keys=True)
        if args.model:
            cfg.model_name = args.model
        if args.layer is not None:
            cfg.layer_index = args.layer
        if args.cap is not None:
            cfg.max_read_norm = args.cap
        print(f"--- mode={mode} ({cfg.summary()})")
        engine = EngramEngine(cfg)

        e1_mean, e1_sd = eval_e1(engine)
        e3 = eval_e3(engine)
        e1c_y, e1c_x = eval_e1c(engine)
        rows.append((mode, e1_mean, e1_sd, e3, e1c_y, e1c_x))
        print(f"    E1 {e1_mean:+.3f} ± {e1_sd:.3f} | E3 {e3:+.4f} | "
              f"E1c ΔMarseille {e1c_y:+.3f} / ΔParis {e1c_x:+.3f}")

    print("\n  mode        E1 Δexact          E3 (seuil 0.05)   E1c ΔMarseille  E1c ΔParis")
    for mode, e1m, e1s, e3, ey, ex in rows:
        ok = "✓" if e3 <= 0.05 else "✗"
        print(f"  {mode:<11} {e1m:+.3f} ± {e1s:.3f}    {e3:+.4f} {ok}         "
              f"{ey:+.3f}          {ex:+.3f}")


if __name__ == "__main__":
    main()
