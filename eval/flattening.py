"""X7 — Test de l'hypothèse d'aplatissement (protocole pré-enregistré, journal 2026-08-21).

Trois mesures sur le même run :
  1. Courbe Δlogp vs prior — cibles échantillonnées par RANG dans la distribution du
     modèle après la question (top-heavy : rangs 1–500 sur-représentés, le flanc du
     nuage vide jusqu'ici), chacune injectée puis rappelée via le protocole E1.
  2. Décomposition logit-lens de la lecture : r = λ·M·φ(h) projeté sur la direction
     d'unembedding W_U[cible]. cos(r, u_cible) = fraction de rappel dirigé ; le reste
     est du bruit qui disperse la masse. (Approximation logit lens : r vit dans
     l'espace résiduel de la couche L, u dans l'espace d'unembedding — le chemin
     direct ignore les couches L..N ; on compare donc à une base de cosinus vs
     tokens aléatoires, pas à zéro absolu.)
  3. Entropie de sortie avec/sans M au prompt de rappel — le symptôme brut.
Plus le volet multi-tokens : Δlogp du token de tête vs tokens suivants sur les 10
secrets historiques (l'aplatissement frappe-t-il seulement au point d'entrée ?).

Usage :
  python eval/flattening.py
  python eval/flattening.py --model HuggingFaceTB/SmolLM2-360M --layer 16 --cap 0.1
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from fact_injection import FACT_TEMPLATE, SECRETS  # noqa: E402
from engram import EngramConfig, EngramEngine  # noqa: E402

QUESTION = "The password is"
# Top-heavy : 14 rangs dans 1–500, 6 au-delà.
RANKS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500,
         1000, 3000, 8000, 15000, 25000, 35000]


def entropy(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-12)).sum().item()


def pick_targets(engine: EngramEngine, ranks: list[int]) -> list[tuple[int, int, str]]:
    """Cibles = tokens aux rangs demandés dans la distribution du modèle vierge après
    QUESTION (on avance au rang suivant jusqu'à un token « mot » : espace + alpha)."""
    engine.reset_memory()
    engine.clear_context()
    engine.stream(QUESTION, read=False, write=False)
    order = torch.argsort(engine._last_logits, descending=True)
    picked, used = [], set()
    for r in ranks:
        i = r - 1
        while True:
            tid = int(order[i].item())
            s = engine.tokenizer.decode([tid])
            if tid not in used and s.startswith(" ") and s[1:].isalpha() and len(s) > 3:
                picked.append((i + 1, tid, s))
                used.add(tid)
                break
            i += 1
    return picked


def unembed_row(engine: EngramEngine, tid: int) -> torch.Tensor:
    return engine.cortex.model.lm_head.weight[tid].detach().float()


def measure_target(engine: EngramEngine, tid: int, tok: str) -> dict:
    """Protocole E1 sur un token cible + décomposition de la lecture au rappel."""
    fact = f"The password is{tok}. Remember it well."
    engine.reset_memory()
    engine.clear_context()
    engine.stream(fact, force_write=True)

    engine.clear_context()
    engine.stream(QUESTION, read=True, write=False)
    lp_mem = -engine._nll_of(tid)
    ent_mem = entropy(engine._last_logits)
    h = engine.cortex.last_h_pre
    r_vec = engine.memory.read(h).float()
    u = unembed_row(engine, tid).to(r_vec.device)
    cos_target = F.cosine_similarity(r_vec, u, dim=0).item()

    engine.reset_memory()
    engine.clear_context()
    engine.stream(QUESTION, read=True, write=False)
    lp_base = -engine._nll_of(tid)
    ent_base = entropy(engine._last_logits)

    return {
        "tok": tok.strip(),
        "lp_base": lp_base,
        "delta": lp_mem - lp_base,
        "cos_target": cos_target,
        "read_norm": r_vec.norm().item(),
        "d_entropy": ent_mem - ent_base,
    }


def random_cos_baseline(engine: EngramEngine, r_vec_dim_token: int, n: int = 200) -> float:
    """|cos| moyen entre une lecture type et des directions d'unembedding aléatoires."""
    g = torch.Generator().manual_seed(0)
    vocab = engine.cortex.model.lm_head.weight.shape[0]
    ids = torch.randint(0, vocab, (n,), generator=g)
    # lecture type : celle du dernier measure_target (approx suffisante pour la base)
    h = engine.cortex.last_h_pre
    r_vec = engine.memory.read(h).float()
    if r_vec.norm() == 0:  # M vide à cet instant : recharger un fait quelconque
        engine.stream(FACT_TEMPLATE.format(secret="swordfish"), force_write=True)
        engine.clear_context()
        engine.stream(QUESTION, read=True, write=False)
        r_vec = engine.memory.read(engine.cortex.last_h_pre).float()
    rows = engine.cortex.model.lm_head.weight[ids.to(engine.cortex.device)].detach().float()
    cs = F.cosine_similarity(rows, r_vec.unsqueeze(0), dim=1).abs()
    return cs.mean().item()


def per_token_multi(engine: EngramEngine, secret: str) -> tuple[float, float]:
    """Δlogp du token de tête vs somme des tokens suivants (secret multi-tokens)."""
    ids_prompt = engine.tokenizer.encode(QUESTION)
    ids_full = engine.tokenizer.encode(QUESTION + f" {secret}")
    ids_cont = ids_full[len(ids_prompt):]

    def logps() -> list[float]:
        engine.clear_context()
        engine.stream(QUESTION, read=True, write=False)
        out = []
        for tid in ids_cont:
            out.append(-engine._nll_of(tid))
            engine._consume(tid, read=True, write=False, force_write=False)
        return out

    engine.reset_memory()
    engine.clear_context()
    engine.stream(FACT_TEMPLATE.format(secret=secret), force_write=True)
    mem = logps()
    engine.reset_memory()
    base = logps()
    d = [m - b for m, b in zip(mem, base)]
    return d[0], sum(d[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--cap", type=float, default=None)
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer
    if args.cap is not None:
        cfg.max_read_norm = args.cap

    print(f"[X7 flattening] {cfg.summary()}")
    engine = EngramEngine(cfg)

    targets = pick_targets(engine, RANKS)
    print(f"  {len(targets)} cibles, rangs {targets[0][0]} → {targets[-1][0]}\n")
    rows = []
    for rank, tid, tok in targets:
        m = measure_target(engine, tid, tok)
        m["rank"] = rank
        rows.append(m)
        print(
            f"  rang {rank:>6} {m['tok']:<14} prior={m['lp_base']:7.3f}  "
            f"Δlogp={m['delta']:+7.3f}  cos(r,u)={m['cos_target']:+.3f}  ΔH={m['d_entropy']:+.4f}"
        )

    base_cos = random_cos_baseline(engine, targets[0][1])
    deltas = [m["delta"] for m in rows]
    priors = [m["lp_base"] for m in rows]
    coses = [m["cos_target"] for m in rows]
    dents = [m["d_entropy"] for m in rows]
    zero_cross = [m["rank"] for m in rows if m["delta"] < 0]

    print(f"\n  corr(Δlogp, prior)          : {statistics.correlation(deltas, priors):+.2f}")
    print(f"  corr(Δlogp, cos(r,u))       : {statistics.correlation(deltas, coses):+.2f}")
    print(f"  corr(prior, cos(r,u))       : {statistics.correlation(priors, coses):+.2f}")
    print(f"  |cos| base aléatoire        : {base_cos:.3f}  (cos cible moyen : {statistics.mean(coses):+.3f})")
    print(f"  ΔH entropie moyen           : {statistics.mean(dents):+.4f} nats (P2 : > 0 = aplatissement)")
    print(f"  rangs à Δlogp NÉGATIF       : {zero_cross or 'aucun'}")

    print("\n  multi-tokens (P3 — tête vs suite) :")
    firsts, rests = [], []
    for s in SECRETS:
        d1, dr = per_token_multi(engine, s)
        firsts.append(d1)
        rests.append(dr)
        print(f"    {s:<12} Δtête={d1:+7.3f}  Δsuite={dr:+7.3f}")
    print(f"    moyennes : Δtête {statistics.mean(firsts):+.3f} / Δsuite {statistics.mean(rests):+.3f}")


if __name__ == "__main__":
    main()
