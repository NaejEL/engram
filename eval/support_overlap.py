# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recouvrement des supports de `topk(G·h)` — protocole
`experiments/EXP-2026-08-23-recouvrement-supports.md` (PRE-ENREGISTRE).

Ce fichier porte l'instrument de mesure du cycle. Il est **hors `engram/`**, il
n'instancie **jamais** `M`, il n'injecte **aucune** lecture, il ne modifie
**aucune** NLL et il ne touche **aucun** hyperparamètre d'`EngramConfig` :
les quantités sont géométriques, sur des états déjà capturés (§4.9-1).

**Ordre d'exécution gravé** (§10, et amendement de gate §14) :

  1. `V-cache` — les états `h` de v4 sont sur disque, **hash calculé et publié** ;
  2. `V-G`     — reproduire `A3` **sur le chemin d'origine, bit-à-bit**, puis
                 mesurer en fp64 ; **deux nombres et l'écart publiés** (D26) ;
  3. écriture du banc D14-S, puis mesure — **conditionnées au PASS de `V-G`**.

`V-G` est **bloquante** : son échec est un **arrêt de provenance (D14-R)**,
jamais un repli (§4.7, §6.A). Ce module implémente donc la barrière
structurellement : `main()` n'ouvre pas l'étage de mesure tant que les deux
portes de provenance ne rendent pas PASS.

Aucun backprop, aucun optimiseur, `G` gelée (D9). Précision : les états sont
relus tels quels depuis les `.npz` (dtype publié, D21) ; les quantités de
provenance sont produites **dans le dtype du chemin d'origine** pour la
reproduction bit-à-bit, **et** en fp64 pour la double mesure (D26).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import pool_v4 as p4  # noqa: E402
from engram.config import EngramConfig  # noqa: E402
from engram.hippocampus import FastWeightMemory  # noqa: E402

PASS, FAIL = "PASS", "FAIL"

RES_V4 = ROOT / "experiments" / "results" / "v4-materiel"
RAW_V4 = RES_V4 / "raw"
SORTIE = ROOT / "experiments" / "results" / "recouvrement-supports"

# §7 — variables fixées, recopiées, jamais ajustées après mesure.
MODELES = ("gpt2", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-1.5B")
COUCHE_REF = {"gpt2": 6, "HuggingFaceTB/SmolLM2-360M": 16, "Qwen/Qwen2.5-1.5B": 14}
DIM = {"gpt2": 768, "HuggingFaceTB/SmolLM2-360M": 960, "Qwen/Qwen2.5-1.5B": 1536}
STRATES = ("S3", "S2", "S1", "S0")
CENTRAGES = ("aucun", "glob", "type", "auto", "plac")

# Fichiers de mesure v4 d'où `A3` est **relu** (D14-R : jamais de mémoire, §6.I).
MESURE_V4 = {
    "gpt2": "mesure-gpt2-seed0.json",
    "HuggingFaceTB/SmolLM2-360M": "mesure-smollm2-360m-seed0.json",
    "Qwen/Qwen2.5-1.5B": "mesure-qwen2.5-1.5b-seed0.json",
}


def slug(nom_modele: str) -> str:
    return nom_modele.replace("/", "_")


# =========================================================================
#  V-cache — les états sont sur disque, hash calculé et PUBLIÉ
#
#  Limite nommée par le protocole (§3, dernière ligne du tableau de
#  provenance) : ce hash **n'a jamais été pré-enregistré au cycle v4**. Le
#  graver ici est une **limite**, pas une vérification : il atteste que les
#  fichiers relus par ce cycle sont ceux qui ont été relus ensuite, non qu'ils
#  sont ceux qui ont été écrits alors.
# =========================================================================

def sha256_fichier(p: Path, bloc: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            morceau = f.read(bloc)
            if not morceau:
                break
            h.update(morceau)
    return h.hexdigest()


def v_cache(modeles=MODELES) -> tuple:
    """`V-cache` : présence, lisibilité, **hash publié**, dtype publié (D21).

    Échec ⇒ **re-forward autorisé** par la gate PI (§14-4) : 53,82 s,
    VRAM ≤ 4,688 Gio, seul poste GPU du cycle. Ce module ne le déclenche pas
    lui-même : il rend FAIL et nomme le poste, le déclenchement est une
    décision d'exécution.
    """
    det = {"repertoire": str(RAW_V4), "fichiers": {},
           "limite_nommee": "le hash des `.npz` de v4 n'a jamais été "
                            "pré-enregistré : il est gravé ici pour la suite, "
                            "ce qui est une LIMITE, pas une vérification (§3)",
           "poste_gpu_si_echec": {"duree_s": 53.82, "vram_gio_max": 4.688,
                                  "statut": "non déclenché tant que V-cache PASS"}}
    manquants = []
    for m in modeles:
        p = RAW_V4 / f"etats-{slug(m)}.npz"
        if not p.exists():
            manquants.append(str(p))
            det["fichiers"][m] = {"present": False, "chemin": str(p)}
            continue
        z = np.load(p)
        cles = sorted(z.keys())
        exemple = f"{sorted(k for k in cles if k.endswith('|t'))[0]}"
        det["fichiers"][m] = {
            "present": True, "chemin": str(p), "taille_octets": p.stat().st_size,
            "sha256": sha256_fichier(p), "n_cles": len(cles),
            "dtype_etats": str(z[exemple].dtype),
            "forme_etats": list(z[exemple].shape),
            "cle_exemple": exemple,
        }
    ok = not manquants
    det["fichiers_manquants"] = manquants
    return (PASS if ok else FAIL), det


# =========================================================================
#  Matériau v4 — relu depuis `pool_v4.json`, jamais reconstruit
# =========================================================================

def materiau_v4() -> dict:
    """Le matériau qualifié v4, **relu** depuis le brut du cycle précédent.

    Le reconstruire demanderait les tokenizers HF ; le relire garantit que
    l'ordre des unités est **exactement** celui sous lequel les états ont été
    capturés (l'indexation des `.npz` en dépend).
    """
    j = json.loads((RES_V4 / "pool_v4.json").read_text(encoding="utf-8"))
    return j["materiau"]


def indices_decisionnels(mat: dict) -> list:
    return [i for i, u in enumerate(mat["unites"]) if u["pontee"]]


# =========================================================================
#  A3 — les DEUX chemins de calcul, nommés et exécutés séparément
#
#  Le protocole (0-118, §4.7) désigne `hippocampus.phi` comme « le chemin
#  d'origine » d'`A3`. Cette porte **exécute les deux** et publie l'écart : la
#  désignation est une affirmation de provenance, et une affirmation de
#  provenance se vérifie par exécution, jamais par lecture (D14-R).
# =========================================================================

def _cos_v4(A, B=None):
    """Le cosinus tel qu'écrit dans `eval/materiel_v4.py::_cos` — recopié, non
    importé, pour que la reproduction ne dépende pas d'une évolution du module
    d'origine (le protocole exige une reproduction **bit-à-bit**)."""
    A = np.asarray(A, dtype=np.float32)
    B = A if B is None else np.asarray(B, dtype=np.float32)
    na = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    nb = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return (na @ nb.T).astype(np.float64)


def _cos_fp64(A):
    A = np.asarray(A, dtype=np.float64)
    na = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
    return na @ na.T


def _moyennes_par_strate(COS, dec) -> dict:
    n = len(dec)
    par = {}
    for a in range(n):
        for b in range(a + 1, n):
            par.setdefault(p4.strate(dec[a], dec[b]), []).append(COS[a, b])
    return {k: {"cos_phi_moyen": float(np.mean(par[k])), "n_paires": len(par[k])}
            for k in STRATES}


def a3_chemin_materiel_v4(z, mat, couche: int, cfg: EngramConfig,
                          fp64: bool = False) -> dict:
    """`A3` **tel qu'il a été produit** : `eval/materiel_v4.py::descriptif_A3`.

    `G` y est tirée par **numpy** (`default_rng(cfg.seed).standard_normal`) et
    divisée par `√d` ; la coupure est un **seuil** `|z| ≥ q` (et non un `topk`
    d'exactement `k` indices). Ces deux points sont des faits de code, relevés
    par exécution ; ils sont publiés parce que la porte les décide.
    """
    dec = mat["unites_decisionnelles"]
    idx = indices_decisionnels(mat)
    cellules = list(mat["cellules"].keys())
    d = z[f"{cellules[0]}|t"].shape[2]
    rng = np.random.default_rng(cfg.seed)
    G = rng.standard_normal((cfg.dg_dim, d)).astype(np.float32) / np.sqrt(d)
    n = len(dec)
    COS = np.zeros((n, n), dtype=np.float64)
    tailles = []
    for cle in cellules:
        H = z[f"{cle}|t"][couche][idx]
        H = H.astype(np.float64) if fp64 else H.astype(np.float32)
        Z = H @ (G.astype(np.float64) if fp64 else G).T
        seuil = np.partition(np.abs(Z), -cfg.dg_topk, axis=1)[:, -cfg.dg_topk]
        PHI = np.where(np.abs(Z) >= seuil[:, None], Z, 0.0)
        tailles.append(int(np.max((PHI != 0).sum(axis=1))))
        COS += _cos_fp64(PHI) if fp64 else _cos_v4(PHI)
    COS /= len(cellules)
    return {"par_strate": _moyennes_par_strate(COS, dec),
            "dtype_G": str(G.dtype), "fp64": fp64,
            "taille_max_du_support": max(tailles),
            "coupure": "seuil |z| >= q_(k) (peut dépasser k sur ex æquo)",
            "generateur_de_G": "numpy.random.default_rng(seed).standard_normal",
            "echelle_de_G": "1/sqrt(d)"}


def a3_chemin_hippocampus(z, mat, couche: int, cfg: EngramConfig,
                          fp64: bool = False) -> dict:
    """`A3` recalculé sur `engram.hippocampus.FastWeightMemory.phi` — le chemin
    que le protocole **désigne** comme celui d'origine (§4.7).

    En fp32 la fonction du projet est appelée telle quelle (`phi` force
    `h.float()`). En fp64 la **même `G`** est relue depuis l'objet et remontée
    en double précision : c'est la précision de l'arithmétique qui change, pas
    le tirage (D9 — `G` reste gelée, seed 0).
    """
    dec = mat["unites_decisionnelles"]
    idx = indices_decisionnels(mat)
    cellules = list(mat["cellules"].keys())
    d = z[f"{cellules[0]}|t"].shape[2]
    mem = FastWeightMemory(d, cfg, device="cpu")   # M jamais lue ni écrite
    n = len(dec)
    COS = np.zeros((n, n), dtype=np.float64)
    for cle in cellules:
        H = torch.from_numpy(np.ascontiguousarray(z[f"{cle}|t"][couche][idx]))
        if fp64:
            G = mem.G.double()
            Zc = H.double() @ G.T
            ii = Zc.abs().topk(cfg.dg_topk, dim=1).indices
            P = torch.zeros_like(Zc)
            P.scatter_(1, ii, Zc.gather(1, ii))
            P = P / P.norm(dim=1, keepdim=True)
            COS += (P @ P.T).numpy()
        else:
            P = torch.stack([mem.phi(H[r].float()) for r in range(H.shape[0])])
            COS += (P @ P.T).double().numpy()
    COS /= len(cellules)
    return {"par_strate": _moyennes_par_strate(COS, dec),
            "dtype_G": str(mem.G.dtype), "fp64": fp64,
            "taille_max_du_support": int(cfg.dg_topk),
            "coupure": "z.abs().topk(k) — exactement k indices",
            "generateur_de_G": "torch.randn(dg_dim, d, generator=manual_seed(seed))",
            "echelle_de_G": "aucune (pas de 1/sqrt(d))"}


def a3_publie(nom_modele: str) -> dict:
    """`A3` **relu depuis `experiments/results/`** (D14-R, §6.I) : jamais cité de
    mémoire. Publie les deux colonnes séparément — `cos_dg` (le cosinus de
    `φ(h)`) et `compression` (`cos_brut − cos_dg`) — parce que ce sont deux
    quantités distinctes et que la borne `L(c)` du §4.2 s'applique à la
    première."""
    j = json.loads((RES_V4 / MESURE_V4[nom_modele]).read_text(encoding="utf-8"))
    a3 = j["metrics"]["descriptif_A3"]
    return {
        "cos_phi_par_strate": {k: a3[k]["cos_dg"] for k in STRATES},
        "cos_brut_par_strate": {k: a3[k]["cos_brut"] for k in STRATES},
        "compression_par_strate": {k: a3[k]["compression"] for k in STRATES},
        "n_paires_par_strate": {k: a3[k]["n"] for k in STRATES},
        "amplitude_de_compression": j["metrics"]["descriptif_A3_amplitude"],
        "source": str(RES_V4 / MESURE_V4[nom_modele]),
    }


def _ecart(obtenu: dict, publie: dict) -> dict:
    return {k: obtenu[k]["cos_phi_moyen"] - publie[k] for k in STRATES}


def v_g(modeles=MODELES, cfg: EngramConfig | None = None) -> tuple:
    """`V-G` — porte de PROVENANCE, bloquante (§4.7, §6.A).

    Exigence du protocole, mot pour mot : *« Reproduire `A3` en fp32, sur le
    chemin d'origine (`hippocampus.phi`), bit-à-bit ; puis mesurer en fp64 ;
    publier les deux nombres et l'écart (D26). »*

    La porte rend PASS si et seulement si le chemin **désigné** reproduit `A3`
    bit-à-bit. Le chemin **effectivement exécuté au cycle v4** est calculé lui
    aussi, et son écart publié : sans ce second nombre, un échec ne serait pas
    diagnostique — c'est précisément le mode 0-118 que la porte doit éviter.
    """
    cfg = cfg or EngramConfig()
    det = {"exigence": "reproduire A3 en fp32, sur le chemin d'origine "
                       "(hippocampus.phi), bit-à-bit ; puis mesurer en fp64 ; "
                       "publier les deux nombres et l'écart (D26)",
           "cfg": cfg.summary(), "modeles": {}}
    mat = materiau_v4()
    ok_global = True
    for m in modeles:
        t0 = time.time()
        z = np.load(RAW_V4 / f"etats-{slug(m)}.npz")
        couche = COUCHE_REF[m]
        pub = a3_publie(m)
        d_att = DIM[m]
        d_lu = int(z[f"{list(mat['cellules'])[0]}|t"].shape[2])
        phi32 = a3_chemin_hippocampus(z, mat, couche, cfg, fp64=False)
        phi64 = a3_chemin_hippocampus(z, mat, couche, cfg, fp64=True)
        v4_32 = a3_chemin_materiel_v4(z, mat, couche, cfg, fp64=False)
        v4_64 = a3_chemin_materiel_v4(z, mat, couche, cfg, fp64=True)
        e_phi = _ecart(phi32["par_strate"], pub["cos_phi_par_strate"])
        e_v4 = _ecart(v4_32["par_strate"], pub["cos_phi_par_strate"])
        bit_phi = all(v == 0.0 for v in e_phi.values())
        bit_v4 = all(v == 0.0 for v in e_v4.values())
        ok_global = ok_global and bit_phi
        det["modeles"][m] = {
            "couche": couche, "d_attendu": d_att, "d_lu": d_lu,
            "A3_publie": pub,
            "chemin_designe_par_le_protocole": {
                "nom": "engram.hippocampus.FastWeightMemory.phi",
                "fp32": phi32, "fp64": phi64,
                "ecart_fp32_vs_publie": e_phi,
                "ecart_fp64_moins_fp32": {
                    k: phi64["par_strate"][k]["cos_phi_moyen"]
                       - phi32["par_strate"][k]["cos_phi_moyen"] for k in STRATES},
                "reproduction_bit_a_bit": bit_phi,
            },
            "chemin_effectivement_execute_au_cycle_v4": {
                "nom": "eval/materiel_v4.py::descriptif_A3",
                "fp_natif": v4_32, "fp64": v4_64,
                "ecart_fp_natif_vs_publie": e_v4,
                "ecart_fp64_moins_fp_natif": {
                    k: v4_64["par_strate"][k]["cos_phi_moyen"]
                       - v4_32["par_strate"][k]["cos_phi_moyen"] for k in STRATES},
                "reproduction_bit_a_bit": bit_v4,
            },
            "duree_s": round(time.time() - t0, 2),
        }
    det["verdict_porte"] = ("le chemin DÉSIGNÉ par le protocole reproduit A3 "
                            "bit-à-bit" if ok_global else
                            "le chemin DÉSIGNÉ par le protocole ne reproduit PAS "
                            "A3 ; arrêt de provenance (D14-R, §6.A)")
    return (PASS if ok_global else FAIL), det


# =========================================================================
#  Étage de MESURE — fermé tant que les portes de provenance ne passent pas
# =========================================================================

class ArretDeProvenance(RuntimeError):
    """§6.A — `V-G` échoue ⇒ arrêt de provenance (D14-R), **jamais un repli**.

    Levée au lieu d'ouvrir l'étage de mesure : la barrière est structurelle,
    pas une consigne de rédaction.
    """


def mesure(**_):
    raise ArretDeProvenance(
        "étage de mesure fermé : le protocole (§10, §14) grave l'ordre "
        "V-cache → V-G → banc D14-S (E = 0) → mesure, et §6.A fait de l'échec "
        "de V-G un arrêt de provenance sans analyse")


# =========================================================================
#  CLI
# =========================================================================

def construire_parseur() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recouvrement des supports de topk(G·h) — "
                    "EXP-2026-08-23-recouvrement-supports")
    p.add_argument("--model", default=None, choices=list(MODELES),
                   help="modèle ; par défaut les trois du §7")
    p.add_argument("--layer", type=int, default=None,
                   help="couche de capture ; par défaut COUCHE_REF du §7")
    p.add_argument("--center", default="aucun", choices=list(CENTRAGES),
                   help="condition de centrage (§8, variable manipulée)")
    p.add_argument("--strate", default=None, choices=list(STRATES),
                   help="strate ; par défaut les quatre")
    p.add_argument("--R", type=int, default=None,
                   help="nombre de directions de placebo ; DÉRIVÉ (§4.5), "
                        "plafond 300 — une valeur donnée à la main est publiée "
                        "comme telle")
    p.add_argument("--seed", type=int, default=0, help="seed de G (D9) et du tirage")
    p.add_argument("--out", default=str(SORTIE), help="répertoire des bruts")
    return p


def main(argv=None) -> int:
    a = construire_parseur().parse_args(argv)
    cfg = EngramConfig(seed=a.seed)
    print(cfg.summary())
    modeles = (a.model,) if a.model else MODELES
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    vc, dc = v_cache(modeles)
    print(f"\nV-cache : {vc}")
    for m, f in dc["fichiers"].items():
        if f["present"]:
            print(f"  {m}\n    {f['chemin']}\n    sha256 = {f['sha256']}\n"
                  f"    {f['taille_octets']} octets, dtype={f['dtype_etats']}, "
                  f"forme={f['forme_etats']}")
        else:
            print(f"  {m} : ABSENT — {f['chemin']}")
    print(f"  limite nommée : {dc['limite_nommee']}")
    (out / "V-cache.json").write_text(
        json.dumps({"porte": "V-cache", "verdict": vc, "detail": dc},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if vc != PASS:
        print("\nV-cache FAIL ⇒ re-forward autorisé par la gate PI (§14-4) : "
              "53,82 s, VRAM ≤ 4,688 Gio. Ce script ne le déclenche pas.")
        return 2

    vg, dg = v_g(modeles, cfg)
    print(f"\nV-G : {vg}")
    for m, r in dg["modeles"].items():
        pub = r["A3_publie"]["cos_phi_par_strate"]
        des = r["chemin_designe_par_le_protocole"]
        exe = r["chemin_effectivement_execute_au_cycle_v4"]
        print(f"\n  {m} (couche {r['couche']}, d={r['d_lu']})")
        print("    A3 relu depuis experiments/results/ (cos de φ(h)) : "
              + ", ".join(f"{k}={pub[k]:.12f}" for k in STRATES))
        print("    compression relue (cos_brut − cos_dg)             : "
              + ", ".join(f"{k}={r['A3_publie']['compression_par_strate'][k]:.12f}"
                          for k in STRATES))
        print(f"    chemin DÉSIGNÉ ({des['nom']}) :")
        print("      fp32 : " + ", ".join(
            f"{k}={des['fp32']['par_strate'][k]['cos_phi_moyen']:.12f}" for k in STRATES))
        print("      fp64 : " + ", ".join(
            f"{k}={des['fp64']['par_strate'][k]['cos_phi_moyen']:.12f}" for k in STRATES))
        print("      écart fp32 − publié : " + ", ".join(
            f"{k}={des['ecart_fp32_vs_publie'][k]:+.3e}" for k in STRATES))
        print("      écart fp64 − fp32   : " + ", ".join(
            f"{k}={des['ecart_fp64_moins_fp32'][k]:+.3e}" for k in STRATES))
        print(f"      reproduction bit-à-bit : {des['reproduction_bit_a_bit']}")
        print(f"    chemin EXÉCUTÉ au cycle v4 ({exe['nom']}) :")
        print("      natif : " + ", ".join(
            f"{k}={exe['fp_natif']['par_strate'][k]['cos_phi_moyen']:.12f}"
            for k in STRATES))
        print("      fp64  : " + ", ".join(
            f"{k}={exe['fp64']['par_strate'][k]['cos_phi_moyen']:.12f}"
            for k in STRATES))
        print("      écart natif − publié : " + ", ".join(
            f"{k}={exe['ecart_fp_natif_vs_publie'][k]:+.3e}" for k in STRATES))
        print("      écart fp64 − natif   : " + ", ".join(
            f"{k}={exe['ecart_fp64_moins_fp_natif'][k]:+.3e}" for k in STRATES))
        print(f"      reproduction bit-à-bit : {exe['reproduction_bit_a_bit']}")
        print(f"      G : {exe['fp_natif']['generateur_de_G']}, échelle "
              f"{exe['fp_natif']['echelle_de_G']}, dtype "
              f"{exe['fp_natif']['dtype_G']}, coupure « "
              f"{exe['fp_natif']['coupure']} », support max "
              f"{exe['fp_natif']['taille_max_du_support']}")
        print(f"      G du chemin désigné : {des['fp32']['generateur_de_G']}, "
              f"échelle {des['fp32']['echelle_de_G']}, dtype "
              f"{des['fp32']['dtype_G']}, coupure « {des['fp32']['coupure']} »")
    (out / "V-G.json").write_text(
        json.dumps({"porte": "V-G", "verdict": vg, "detail": dg},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ntemps CPU total : {time.time() - t0:.1f} s")
    if vg != PASS:
        print("\n" + dg["verdict_porte"])
        print("§6.A : aucun verdict n'est écrit, aucune analyse n'est conduite, "
              "l'étage de mesure reste fermé.")
        return 3
    print("\nportes de provenance PASS — la suite gravée est le banc D14-S "
          "(E = 0) AVANT toute mesure (§10, §14).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
