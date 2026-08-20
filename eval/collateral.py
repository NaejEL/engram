"""E3 — Dommage collatéral : une M chargée rend-elle le modèle plus bête ailleurs ?

Le scénario d'échec le plus probable du PoC n'est pas « pas de signal », c'est
« signal payé en compétence générale » — le dilemme stabilité/plasticité. Ni E1 ni E2
ne le voient (spec : ARCHITECTURE.md §5, E3).

Protocole : pour chaque secret, charger M via le protocole E1 (injection forcée du
fait), vider le cache KV, puis mesurer la NLL moyenne sur un texte NEUTRE (registre
générique, aucun rapport avec le fait), lecture active, écriture coupée. La condition
contrôle (M = 0, même texte, même contexte vide) est déterministe et indépendante du
secret : calculée une seule fois. Métrique : Δ NLL/token (positif = dommage).
Seuil d'alarme : +0.05 nats/token (spec).

Usage :
  python eval/collateral.py                          # défauts (X1 : dg=8192/64, λ=2, η=0.2)
  python eval/collateral.py --dg-dim 0 --lam 1.0 --eta 0.1   # point X0 pour comparaison
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 vs Δ/±

from fact_injection import FACT_TEMPLATE, SECRETS  # noqa: E402
from engram import EngramConfig, EngramEngine  # noqa: E402

# Texte neutre : registre générique, ~450 tokens GPT-2, aucun vocabulaire commun avec
# les secrets de E1. Embarqué pour que l'éval soit auto-contenue et reproductible.
NEUTRAL_TEXT = """
The morning began like most others in the small town. People walked to work along
the main street, stopping now and then to greet a neighbor or look at the shop
windows. The bakery on the corner opened early, and the smell of fresh bread
drifted out onto the sidewalk. A few children waited at the bus stop, talking
about their plans for the weekend.

Later in the day, clouds gathered over the hills to the west. The forecast had
mentioned a chance of rain, and many residents carried umbrellas just in case.
At the market, vendors arranged fruit and vegetables in neat rows, calling out
prices to passing customers. An older man bought apples and asked about the
weather, and the vendor shrugged and said it would probably clear up by evening.

In the afternoon, the library was quiet except for the sound of pages turning.
Students sat at long wooden tables, preparing for their examinations. The
librarian moved slowly between the shelves, returning books to their places. Near
the entrance, a notice board displayed announcements about community events, a
concert in the park, and a meeting of the town council scheduled for the
following week.

By evening, the streets grew calm again. Lights came on in the houses, and the
smell of cooking drifted from open windows. A dog barked somewhere in the
distance. The rain never arrived, just as the vendor had predicted, and the sky
turned a deep shade of orange as the sun went down behind the hills. Tomorrow,
the town would wake and begin again, the same familiar rhythm it had followed
for as long as anyone could remember.
""".strip()


def mean_nll_on_neutral(engine: EngramEngine) -> float:
    """NLL moyenne par token du texte neutre dans le contexte courant (vide),
    lecture active, écriture coupée (le rappel ne doit jamais contaminer M)."""
    engine.clear_context()
    records = engine.stream(NEUTRAL_TEXT, read=True, write=False)
    return statistics.mean(r.nll for r in records if r.nll is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--dg-dim", type=int, default=None)
    parser.add_argument("--dg-topk", type=int, default=None)
    parser.add_argument("--cap", type=float, default=None, help="max_read_norm (fraction de ‖h‖)")
    parser.add_argument("--secrets", type=int, default=len(SECRETS))
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer
    if args.lam is not None:
        cfg.lam = args.lam
    if args.eta is not None:
        cfg.eta = args.eta
    if args.dg_dim is not None:
        cfg.dg_dim = args.dg_dim
    if args.dg_topk is not None:
        cfg.dg_topk = args.dg_topk
    if args.cap is not None:
        cfg.max_read_norm = args.cap

    print(f"[E3 collateral] {cfg.summary()}")
    engine = EngramEngine(cfg)

    engine.reset_memory()
    base = mean_nll_on_neutral(engine)
    n_tokens = len(engine.tokenizer.encode(NEUTRAL_TEXT))
    print(f"  texte neutre : {n_tokens} tokens | NLL de référence (M=0) : {base:.4f} nats/token")

    deltas = []
    for secret in SECRETS[: args.secrets]:
        engine.reset_memory()
        engine.clear_context()
        engine.stream(FACT_TEMPLATE.format(secret=secret), force_write=True)
        nll_mem = mean_nll_on_neutral(engine)
        delta = nll_mem - base
        deltas.append(delta)
        flag = "  !! > seuil" if delta > 0.05 else ""
        print(f"  {secret:<12} NLL={nll_mem:.4f}  ΔNLL/token={delta:+.4f}{flag}")

    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    print(f"\n  dommage collatéral moyen : {mean:+.4f} ± {sd:.4f} nats/token (max {max(deltas):+.4f})")
    print(f"  seuil d'alarme (spec §5) : +0.05 — {'DÉPASSÉ' if mean > 0.05 else 'sous le seuil'}")


if __name__ == "__main__":
    main()
