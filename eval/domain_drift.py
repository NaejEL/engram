"""E2 — Dérive de domaine : M actif fait-il baisser la NLL au fil d'un long document ?

Protocole (docs/ARCHITECTURE.md §5, E2) : streamer un long texte technique en chunks
(le cache KV est vidé entre chunks — indispensable : GPT-2 est limité à 1024 positions,
et surtout ça isole la contribution de M de celle du contexte), dans deux conditions :
M actif (gating normal) vs M inerte. Métrique : NLL par moitié de document, et
l'interaction (amélioration 2ᵉ moitié due à M).

Usage :
  python eval/domain_drift.py --file chemin/vers/doc.txt
  python eval/domain_drift.py            # télécharge et cache RFC 9293 (TCP, long et technique)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 vs Δ/±

from engram import EngramConfig, EngramEngine  # noqa: E402

DEFAULT_URL = "https://www.rfc-editor.org/rfc/rfc9293.txt"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_text(path: str | None, max_tokens_hint: int = 8000) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    DATA_DIR.mkdir(exist_ok=True)
    cached = DATA_DIR / "rfc9293.txt"
    if not cached.exists():
        print(f"  téléchargement de {DEFAULT_URL} → {cached}")
        urllib.request.urlretrieve(DEFAULT_URL, cached)
    # ~4 chars/token : on tronque large, le tokenizer fera le compte exact.
    return cached.read_text(encoding="utf-8", errors="replace")[: max_tokens_hint * 4]


def stream_in_chunks(engine: EngramEngine, ids: list[int], chunk: int, write: bool) -> list[float]:
    """Streame des ids par chunks avec clear_context entre chaque ; NLL par token.
    Le premier token de chaque chunk n'a pas de NLL (pas de logits antérieurs)."""
    nlls: list[float] = []
    for start in range(0, len(ids), chunk):
        engine.clear_context()
        for tid in ids[start : start + chunk]:
            rec = engine._consume(tid, read=True, write=write, force_write=False)
            if rec.nll is not None:
                nlls.append(rec.nll)
    return nlls


def halves(nlls: list[float]) -> tuple[float, float]:
    mid = len(nlls) // 2
    return statistics.mean(nlls[:mid]), statistics.mean(nlls[mid:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None, help="texte long ; défaut : RFC 9293 (cache data/)")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--chunk", type=int, default=512, help="taille de chunk (cache KV vidé entre)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--thr", type=float, default=None, help="seuil de surprise (0 = toujours écrire)")
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--cap", type=float, default=None, help="max_read_norm (fraction de ‖h‖)")
    parser.add_argument("--dg-dim", type=int, default=None, help="X1 : dimension gyrus denté (0 = off)")
    parser.add_argument("--dg-topk", type=int, default=None)
    parser.add_argument("--hebbian", action="store_true", help="ablation D5 : Hebb pur")
    args = parser.parse_args()

    cfg = EngramConfig()
    if args.model:
        cfg.model_name = args.model
    if args.layer is not None:
        cfg.layer_index = args.layer
    if args.thr is not None:
        cfg.surprise_threshold = args.thr
    if args.lam is not None:
        cfg.lam = args.lam
    if args.eta is not None:
        cfg.eta = args.eta
    if args.cap is not None:
        cfg.max_read_norm = args.cap
    if args.dg_dim is not None:
        cfg.dg_dim = args.dg_dim
    if args.dg_topk is not None:
        cfg.dg_topk = args.dg_topk
    if args.hebbian:
        cfg.hebbian_only = True

    print(f"[E2 domain drift] {cfg.summary()}")
    engine = EngramEngine(cfg)
    ids = engine.tokenizer.encode(load_text(args.file))[: args.max_tokens]
    print(f"  document : {len(ids)} tokens, chunks de {args.chunk}")

    # Condition contrôle d'abord (M reste vierge), puis condition mémoire.
    engine.reset_memory()
    base = stream_in_chunks(engine, ids, args.chunk, write=False)
    engine.reset_memory()
    mem = stream_in_chunks(engine, ids, args.chunk, write=True)
    writes = engine.memory.write_count

    b1, b2 = halves(base)
    m1, m2 = halves(mem)
    interaction = (m2 - m1) - (b2 - b1)  # < 0 ⇒ M accélère l'adaptation au domaine

    print(f"  sans M : NLL 1ʳᵉ moitié {b1:.4f} → 2ᵉ moitié {b2:.4f} (Δ {b2 - b1:+.4f})")
    print(f"  avec M : NLL 1ʳᵉ moitié {m1:.4f} → 2ᵉ moitié {m2:.4f} (Δ {m2 - m1:+.4f})")
    print(f"  interaction (négatif = M aide) : {interaction:+.4f} nats/token, writes={writes}")
    if writes == 0:
        print("  !! 0 write : run INVALIDE (seuil de surprise trop haut pour ce texte).")


if __name__ == "__main__":
    main()
