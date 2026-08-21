# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagnostic « traction vers le marginal » (candidat pour la dérive non spécifique
d'E4s, et 4ᵉ observation potentielle du mécanisme X7).

Méthode : la projection W_U de X7, mais sur TOUT le vocabulaire plutôt que sur la
cible. À chaque position d'un texte neutre (M chargée d'un fait, lecture active,
écriture coupée), on projette la lecture r sur l'unembedding : s = W_U·r, un score
par token. Puis deux corrélations, moyennées sur les positions :
  - corr(s, log-fréquence unigramme) sur les tokens observés dans un corpus local
    → une corrélation positive démontre la traction vers les tokens fréquents ;
  - corr(s, logits du modèle à cette position) → r amplifie-t-il la distribution
    courante (sharpening) ou la contredit-il (flattening) ?

Si corr(s, fréquence) > 0 systématiquement : la dérive E4s est expliquée par X7
(les tokens conformes d'E4s étant souvent les options rares) — le « bug » devient
une théorie à quatre observations.

Usage : python eval/marginal_pull.py [--model ... --layer ...]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from collateral import NEUTRAL_TEXT  # noqa: E402
from fact_injection import FACT_TEMPLATE  # noqa: E402
from engram import EngramConfig, EngramEngine  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def unigram_logfreq(engine: EngramEngine) -> tuple[torch.Tensor, torch.Tensor]:
    """(ids observés, log-fréquences) à partir des corpus locaux disponibles."""
    text = NEUTRAL_TEXT
    for name in ("rfc9293.txt", "pnp_narrative.txt"):
        p = DATA_DIR / name
        if p.exists():
            text += p.read_text(encoding="utf-8", errors="replace")[:200_000]
    counts = Counter(engine.tokenizer.encode(text))
    ids = torch.tensor(sorted(counts))
    freqs = torch.tensor([float(counts[i]) for i in ids.tolist()])
    return ids, torch.log(freqs)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--positions", type=int, default=120)
    parser.add_argument("--gate", default=None, choices=["none", "entropy", "keysim", "two_factor"])
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.gate is not None:
        cfg.read_gate = args.gate
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer

    print(f"[marginal pull] {cfg.summary()}")
    engine = EngramEngine(cfg)
    w_u = engine.cortex.model.lm_head.weight.detach().float()
    ids, logfreq = unigram_logfreq(engine)
    ids = ids.to(engine.cortex.device)
    logfreq = logfreq.to(engine.cortex.device)
    print(f"  vocabulaire observé : {len(ids)} types")

    # M chargée d'un fait (protocole E1), puis streaming du texte neutre.
    engine.reset_memory()
    engine.clear_context()
    engine.stream(FACT_TEMPLATE.format(secret="swordfish"), force_write=True)
    engine.clear_context()

    corr_freq, corr_logits, read_norms = [], [], []
    for tid in engine.tokenizer.encode(NEUTRAL_TEXT)[: args.positions]:
        engine._consume(tid, read=True, write=False, force_write=False)
        r = engine.memory.read(engine.cortex.last_h_pre).float()
        read_norms.append(r.norm().item())
        if r.norm() < 1e-6:
            continue  # gate fermé à cette position : pas de projection à analyser
        s = w_u @ r
        corr_freq.append(pearson(s[ids], logfreq))
        corr_logits.append(pearson(s, engine._last_logits))

    n_active = len(corr_freq)
    print(f"  positions lues (gate ouvert) : {n_active}/{args.positions} "
          f"(‖r‖ moyen {statistics.mean(read_norms):.3f})")
    if n_active >= 3:
        print(f"  corr(W_U·r, log-fréquence) : {statistics.mean(corr_freq):+.3f} "
              f"± {statistics.stdev(corr_freq):.3f}  (>0 = traction vers le fréquent)")
        print(f"  corr(W_U·r, logits modèle) : {statistics.mean(corr_logits):+.3f} "
              f"± {statistics.stdev(corr_logits):.3f}  (>0 = amplifie, <0 = contredit)")
    else:
        print("  gate presque toujours fermé sur texte neutre — refaire avec --gate none"
              " (cohérent avec E3 ≈ 0 en mode keysim).")


if __name__ == "__main__":
    main()
