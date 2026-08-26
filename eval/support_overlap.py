# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recouvrement des supports de `topk(G·h)` — protocole
`experiments/EXP-2026-08-23-recouvrement-supports.md` (PRE-ENREGISTRE,
correction de provenance du 2026-08-26 au §15).

Deux moitiés, strictement séparées, comme `eval/materiel_v4.py` :

  * **le noyau décisionnel** — partitions, nulles, portes, bornes : CPU pur,
    **aucune donnée, aucun modèle**. C'est lui que le banc D14-S exerce clause
    par clause, **avant toute mesure** (§6.E) ;
  * **la mesure** — un seul passage sur trois modèles, à partir des états `.npz`
    déjà en cache, exécutée **uniquement** si le banc rend `E = 0` et si les
    portes de provenance rendent PASS.

Interdits structurels, vérifiables dans ce fichier : `M` n'est **jamais**
instanciée, aucune lecture n'est injectée, aucune NLL n'est modifiée, aucun
backprop, `G` **gelée** (D9), `engram/` **non modifié**, `eval/pool.py` et
`eval/pool_v4.py` **gelés**. Les quantités sont **géométriques**, sur le cortex
gelé (§4.9-1).

Précision : `G·h`, `topk`, cosinus et intersections en **fp64** (§7) ; le dtype
des `.npz` est **publié** (D21). `O` est publié en **fractions exactes `p/64`**
(0-132) — jamais en flottant tronqué.

Ordre d'exécution **GRAVÉ** (§10, §14, amendement §15) :
  `V-cache` → `V-G` (v2) → **banc D14-S (`E = 0`)** → mesure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
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
SANS_OBJET = "SANS OBJET"
EN_ATTENTE = "EN-ATTENTE"

RES_V4 = ROOT / "experiments" / "results" / "v4-materiel"
RAW_V4 = RES_V4 / "raw"
SORTIE = ROOT / "experiments" / "results" / "recouvrement-supports"

# =========================================================================
#  §7 — variables fixées. Recopiées, jamais ajustées après mesure.
# =========================================================================

MODELES = ("gpt2", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-1.5B")
COUCHE_REF = {"gpt2": 6, "HuggingFaceTB/SmolLM2-360M": 16, "Qwen/Qwen2.5-1.5B": 14}
DIM = {"gpt2": 768, "HuggingFaceTB/SmolLM2-360M": 960, "Qwen/Qwen2.5-1.5B": 1536}
STRATES = ("S3", "S2", "S1", "S0")
CENTRAGES = ("aucun", "glob", "type", "auto", "plac")

K_TOPK = 64                 # `dg_topk` — relu d'`EngramConfig`, jamais en dur
D_DG = 8192                 # `dg_dim`  — idem
# `n` a DEUX unités, et le protocole donne les deux (§4.1, §7) : en INDICES,
# `n = k²/D = 0.5 indice` ; en FRACTION de `k` — l'unité de publication de `O`
# (0-132) — `n = 0.5/64 = 1/128` et `2n = 1/64 = 0.015625`. Ce sont les
# secondes qui entrent dans les classes, `O` étant une fraction `p/64`. Les
# mélanger rend le corridor 64 fois trop large, donc `BAS` et `c-mec` vraies
# presque partout : c'est le défaut que le banc a trouvé sur quatre clauses.
N_HASARD_INDICES = Fraction(K_TOPK * K_TOPK, D_DG)   # n = k²/D = 1/2 indice
N_HASARD = Fraction(K_TOPK, D_DG)                    # n en fraction : 1/128
CORRIDOR = 2 * N_HASARD                              # 2n = 1/64 = 0.015625
K_EFF = p4.K_EFF            # 10 tiges (0-50)
P_MIN_PAR_CELLULE = 8       # `V-P8` (0-119)
N_CELL_BASCULE = 30         # §14-2 — split-half principal si `min n_cell < 30`
B_BOOT = 10_000             # bootstrap de TIGES, B = 10⁴
B_PERM = 10_000             # nulle de permutation de `ε*`
ALPHA = 0.05
Q_ENVELOPPE = 0.95          # `ε*` = q₀.₉₅ — LIGNE CANONIQUE UNIQUE (§4.5, §7)
R_PILOTE = 20               # `R₀` — pilote de dérivation de `R`
R_PLAFOND = 300             # plafond publié ; dépassement remonté au PI
CORE_SEUIL = 9              # `Core` : retenu sur ≥ 9 des 10 unités de `S`
CORE_TAILLE_S = 10          # `|S| = 10`, une unité par tige (`V-core-S`)

# `cum(j)` — masses cumulées de la version THÉORIQUE de la borne (§4.2).
#
# **RÈGLE GRAVÉE (Q-M5), appliquée mécaniquement ci-dessous : *aucune valeur de
# `cum` n'est jamais interpolée*.** `cum` est **concave** (les `t_j²`
# décroissent), donc l'interpolation linéaire **sous-estime toujours** — et
# elle a déjà **déplacé une cellule** (`SmolLM2 S3`, frontière 5/6 à 0.26 %).
#
# Seules les **sept ancres exactes** de `lab-math` (Q-M5, vérifiées contre les
# ancres de `lab-verifier`, écart < 1e−4) sont gravées ici. Les valeurs
# `j = 8..20` sont **dues par `lab-math` (Q-M6, en cours)** et se **chargent**
# depuis un fichier : la table est une constante CHARGÉE, pas des nombres en
# dur que le code compléterait tout seul.
CUM_ANCRES = {1: 0.02590, 2: 0.04952, 3: 0.07180, 4: 0.09314,
              5: 0.11376, 6: 0.13379, 7: 0.15332}
CUM_TABLE_FICHIER = SORTIE / "cum_table_QM6.json"

# **Q-M6 — NON RENDUE.** Tant que la table `j = 8..20` et les douze `p_sym`
# n'arrivent pas, le second membre de `V-borne` (« la réalisée ENCADRE la
# théorique ») n'a **pas de seuil**. En poser un serait **0-52**, troisième
# occurrence du même mode dans ce cycle.
ENCADREMENT_TABLE_SEUIL = None

# Fichiers de mesure v4 d'où `A3` est **relu** (D14-R, §6.I) — jamais de mémoire.
MESURE_V4 = {
    "gpt2": "mesure-gpt2-seed0.json",
    "HuggingFaceTB/SmolLM2-360M": "mesure-smollm2-360m-seed0.json",
    "Qwen/Qwen2.5-1.5B": "mesure-qwen2.5-1.5b-seed0.json",
}

# Vocabulaire INTERDIT (§2, entrées (i)-(xx)). `v_schema` est une porte de
# SCHÉMA : elle échoue si l'un de ces termes apparaît dans la sortie.
VOCABULAIRE_INTERDIT = (
    # (xiv) — toute mention de l'étage d'ÉCRITURE
    "chemin d'écriture", "chemin d'ecriture", "étage d'écriture", "etage d'ecriture",
    # (xv)
    "sémantique", "semantique",
    # (xviii) — toute désignation biologique du centrage
    "inhibition tonique", "normalisation divisive",
    "retrait de mode commun par les interneurones",
    "le centrage est ce que fait le gyrus denté",
    # (xix)
    "code de fond du gyrus denté", "code de fond du dg",
    # (xx)
    "le centrage répare", "le centrage repare", "remède validé", "remede valide",
    # (ix)-(xiii) reconduits de v4, et le bin dur sous toute forme
    "séparation de patterns", "separation de patterns", "bin dur",
    "tendance", "suggère", "suggere", "va dans le sens de",
)

# Phrase gravée (xvi) — celle de `lab-neuro`, verbatim, à recopier.
PHRASE_XVI = (
    "Sur les états de <modèle>, capturés à <locus>, dans la cellule <type>, les "
    "supports de topk(G·h) partagent en moyenne X indices sur 64 (médiane X̃, "
    "IQR [·,·]), contre 0.5 indice sous tirage indépendant (k²/D), et "
    "l'intersection porte f = · de l'énergie avec un accord de signe de "
    "σ± = ·. Cette phrase porte sur l'étage de LECTURE de φ, sur ce locus et ce "
    "matériau seulement ; elle ne porte ni sur le chemin d'écriture de M "
    "(jamais instanciée ici, +57 % d'E2 acquis et intact), ni sur un quelconque "
    "effet aval (E1, E2, E3 non mesurés)."
)

# La phrase gravée contient elle-même un terme proscrit (« chemin d'écriture ») :
# elle NOMME la limite, elle ne l'emploie jamais comme évidence. Elle est donc
# retirée du texte AVANT le balayage — sinon la porte échouerait sur la
# formulation que le protocole rend obligatoire (mécanisme de `_PHRASES_GRAVEES`
# de `materiel_v4.py`).
PHRASES_GRAVEES = (PHRASE_XVI,)

# Opérationnalisations DÉCLARÉES : le protocole fixe la clause, pas la
# statistique d'exécution. Publiées, **hors `E`**, jamais ajustées après lecture.
OPERATIONNALISATIONS = {
    "cellule d'estimation de `μ_type`": (
        "§8 grave « h − μ_type (LOO par paire) » sans dire ce qu'est un type. "
        "Déclaré : la cellule d'estimation est la CELLULE DE CAPTURE du "
        "matériau v4 (T1-A, T1-B, T2-A, T2-B, T3-A, T3-B) ; ses membres sont "
        "les 60 unités décisionnelles de cette cellule, d'où n_cell = 60. "
        "Mélanger les variantes mélangerait deux positions de capture."),
    "paire → tige, pour le bootstrap et la permutation": (
        "§5-6 grave « bootstrap de TIGES » et §4.5 « permutation intra-tige », "
        "or une paire S1/S0 a DEUX tiges. Déclaré : (a) poids d'une paire dans "
        "un rééchantillon de multiplicités m — m[t_a]·m[t_b] si t_a ≠ t_b, "
        "m[t_a] sinon ; (b) pour la permutation des étiquettes {plac, type}, "
        "une paire est affectée à la tige min(t_a, t_b) et le tirage de "
        "l'étiquette est CONSTANT dans une tige."),
    "unité de paire": (
        "une paire est un couple d'unités décisionnelles DANS UNE MÊME CELLULE "
        "de capture : les supports sont des ENSEMBLES et ne se moyennent pas "
        "entre cellules comme le cosinus d'A3 le faisait."),
    "ensemble `S` de `Core`": (
        "`V-core-S` exige |S| = 10, une unité par tige, sans dire laquelle. "
        "Déclaré : la PREMIÈRE unité décisionnelle de chaque tige dans l'ordre "
        "du matériau — règle déterministe, sans aléa, publiée avec "
        "l'appartenance."),
    "règle de tirage des `R` directions": (
        "`V-seed` exige seed, générateur et règle gelés. Déclaré : "
        "torch.randn(R, d, generator=manual_seed(seed_dir)) puis normalisation "
        "L2 par ligne ; indices 0..R−1 ; les directions NE SONT PAS "
        "transportables entre modèles (d = 768/960/1536, 0-133) — "
        "l'appariement inter-modèles se fait par la NORME, jamais par le "
        "vecteur ; cardinal publié PAR MODÈLE."),
    "marge de `V-ulp`": (
        "§4.7 exige la marge à la coupure PUBLIÉE pour les cinq conditions ; "
        "il ne fixe aucun seuil. Déclaré : la porte est une porte de "
        "COUVERTURE — elle échoue si l'une des cinq conditions n'a pas sa "
        "marge. Les marges elles-mêmes sont publiées en ULP, en descriptif. "
        "Poser un seuil non pré-enregistré serait 0-52."),
}


def slug(nom_modele: str) -> str:
    return nom_modele.replace("/", "_")


def frac(p: int, q: int) -> Fraction:
    return Fraction(int(p), int(q))


def texte_fraction(f: Fraction) -> str:
    """`O` se publie en fraction EXACTE, jamais en flottant tronqué (0-132)."""
    return f"{f.numerator}/{f.denominator}"


# =========================================================================
#  ============  NOYAU DÉCISIONNEL — aucune donnée, aucun modèle  =========
#
#  Tout ce qui suit est exercé par le banc D14-S, clause par clause, avec un
#  cas passant ET un cas échouant, AVANT toute mesure (§6.E).
# =========================================================================

# ------------------------------------------------------------- supports

def support_topk(z, k: int = K_TOPK):
    """Support de `topk(G·h)` : les `k` indices de plus grande **magnitude**.

    Recopie la sémantique d'`engram/hippocampus.py::phi` — `z.abs().topk(k)`,
    **exactement `k` indices**, jamais un seuil `|z| ≥ q` (le seuil peut en
    rendre davantage sur ex æquo : c'est l'une des quatre différences relevées
    au 0-135, la seule qui soit inerte, `0.0` exact sur 4/4).

    fp64 = chemin nominal (§7). Départage des ex æquo : par **indice
    croissant**, déterministe — `np.argpartition` puis tri stable sur
    `(-|z|, indice)`.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("support_topk attend un vecteur")
    if not np.isfinite(z).all():
        raise ValueError("NaN ou inf dans z — clause d'abandon §6.C")
    k = int(k)
    if not 1 <= k <= z.size:
        raise ValueError(f"k = {k} hors de [1, {z.size}]")
    a = np.abs(z)
    cand = np.argpartition(a, -k)[-k:]
    ordre = np.lexsort((cand, -a[cand]))
    return np.sort(cand[ordre][:k])


def phi_de(z, idx):
    """`φ = topk(G·h)` normalisée L2 sur son support, en fp64."""
    z = np.asarray(z, dtype=np.float64)
    v = np.zeros_like(z)
    v[idx] = z[idx]
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def intersection(a, b) -> int:
    """`|A ∩ B|` — comptage ENTIER, jamais un flottant (0-132)."""
    n = int(np.intersect1d(np.asarray(a), np.asarray(b), assume_unique=True).size)
    if not 0 <= n <= K_TOPK:
        raise ValueError(f"|A∩B| = {n} hors de [0, {K_TOPK}] — §6.C")
    return n


def o_fraction(somme_intersections: int, n_paires: int) -> Fraction:
    """`O` agrégé en **fraction exacte** `p/64` (0-132).

    Support vide (`n_paires = 0`) ⇒ la quantité n'a **pas de domaine de
    définition** : on rend `None`, qui se publie `SANS OBJET`, **jamais `0`**.
    """
    if n_paires <= 0:
        return None
    return Fraction(int(somme_intersections), K_TOPK * int(n_paires))


# ------------------------------------------------------- borne `L(c)`

def charger_cum(chemin: Path | None = None) -> dict:
    """Charge la table `cum(j)`. **Constante CHARGÉE, jamais complétée par le
    code** : les sept ancres exactes sont gravées, le reste vient du fichier
    de `lab-math` (Q-M6) s'il existe. Aucune interpolation, aucune
    extrapolation — la table s'attend, elle ne se devine pas.
    """
    table = dict(CUM_ANCRES)
    source = ["ancres exactes gravées (Q-M5, j = 1..7)"]
    p = chemin or CUM_TABLE_FICHIER
    if Path(p).exists():
        brut = json.loads(Path(p).read_text(encoding="utf-8"))
        table.update({int(k): float(v) for k, v in brut.get("cum", {}).items()})
        source.append(f"table chargée : {p}")
    js = sorted(table)
    contigue = js == list(range(1, len(js) + 1))
    concave = all(table[js[i + 1]] - table[js[i]] <= table[js[i]] - table[js[i - 1]]
                  + 1e-12 for i in range(1, len(js) - 1)) if len(js) > 2 else True
    return {"cum": table, "j_max": max(js), "sources": source,
            "contigue_depuis_1": contigue, "concave": concave,
            "Q-M6": ("rendue" if max(js) >= 20 else "NON RENDUE — j = 8..20 "
                                                    "attendus de lab-math"),
            "regle": "aucune valeur de cum n'est jamais interpolée (Q-M5)"}


def p_sym_theorique(cos: float, table: dict | None = None) -> dict:
    """`p_sym = min{p : cum(p) ≥ cos}` — borne **SERRÉE**, version THÉORIQUE
    (§16, adoptée sous D30 alinéa 2 le 2026-08-26).

    La dérivation gelée majorait `m_ψ ≤ 1` pour obtenir `min(m_φ, m_ψ) ≥ cos²`
    — **cette marche jette un facteur `cos`**. Or **les deux** masses sont
    bornées par `T(p)`, d'où l'identité serrée, de même statut et sans
    hypothèse supplémentaire : `cos ≤ √(T_φ(p)·T_ψ(p)) ≤ T(p)`, donc le seuil
    est **`≥ cos`** et non `≥ cos²`.

    Si `cos` dépasse la dernière valeur connue, la fonction rend
    **`EN-ATTENTE`** — elle **n'extrapole pas** (règle Q-M5).
    """
    t = (table or charger_cum())["cum"]
    c = float(cos)
    for p in sorted(t):
        if t[p] >= c:
            return {"p_sym": p, "statut": PASS, "cos": c, "cum_p": t[p],
                    "j_max_connu": max(t)}
    return {"p_sym": None, "statut": EN_ATTENTE, "cos": c, "j_max_connu": max(t),
            "cum_j_max": t[max(t)],
            "raison": f"cos = {c:.5f} dépasse cum({max(t)}) = {t[max(t)]:.5f} : "
                      f"la valeur exige la table j = 8..20 de Q-M6. Aucune "
                      f"interpolation, aucune extrapolation (règle Q-M5)."}


def _masse_cumulee_realisee(v):
    """`T(p)` réalisée : masses `φ_i²` observées, triées décroissantes, cumulées."""
    s = np.sort(np.asarray(v, dtype=np.float64) ** 2)[::-1]
    return np.cumsum(s)


def p_sym_realise(phi_a, phi_b, cos_ab: float) -> int:
    """Version **RÉALISÉE par paire** de la borne serrée — la MESURE et la
    porte : `p_sym = min{p : √(T_φ(p)·T_ψ(p)) ≥ cos}`, `T` calculée sur les
    `φ_i²` **observés**.

    Exact, par paire, **sans hypothèse distributionnelle** : `m_φ ≤ T_φ(q)`,
    `m_ψ ≤ T_ψ(q)` et `cos ≤ √(m_φ·m_ψ)` ⇒ `q ≥ p_sym`. Toute violation est un
    **BUG DE MESURE**, jamais un résultat.
    """
    ta, tb = _masse_cumulee_realisee(phi_a), _masse_cumulee_realisee(phi_b)
    n = min(ta.size, tb.size)
    g = np.sqrt(ta[:n] * tb[:n])
    j = int(np.searchsorted(g, float(cos_ab), side="left"))
    return min(j + 1, n)


def p_pair_realise(phi_a, phi_b, cos_ab: float) -> dict:
    """Borne **LÂCHE**, conservée en **DESCRIPTIF** (§16-3, double mesure D26).

    `p = min{p : T(p) ≥ cos²}` sur chaque membre, `p_pair = max(p_a, p_b)`.
    C'est ce que le gel portait ; elle est publiée avec l'écart à `p_sym` pour
    laisser trace de ce que la borne serrée a gagné (facteur ~7 à `cos ≈ 0.15`).
    """
    c2 = float(cos_ab) ** 2

    def _p(v):
        cum = _masse_cumulee_realisee(v)
        j = int(np.searchsorted(cum, c2, side="left"))
        return min(j + 1, cum.size)

    pa, pb = _p(phi_a), _p(phi_b)
    psym = p_sym_realise(phi_a, phi_b, cos_ab)
    return {"p_a": pa, "p_b": pb, "p_pair": max(pa, pb), "p_sym": psym,
            "ecart_psym_moins_ppair": psym - max(pa, pb),
            "cos": float(cos_ab), "cos2": c2,
            "statut_p_pair": "DESCRIPTIF (borne lâche, §16-3)",
            "statut_p_sym": "DÉCISIONNEL (borne serrée, §16-1)"}


# ------------------------------------------------------------ PARTITIONS
#
#  Ordres d'évaluation GRAVÉS (D18) — la PREMIÈRE classe satisfaite emporte,
#  la complémentation est évaluée EN DERNIER.
#    `N` : HAUT → − → BAS → ind_L
#    `C` : c-anti (court-circuitante) → c-mec → c-cent → ind_Δ

CLASSES_N = ("HAUT", "−", "BAS", "ind_L")
CLASSES_C = ("c-anti", "c-mec", "c-cent", "ind_Δ")


def classe_n(ic_lambda, ic_o, eps_lambda: float,
             n: Fraction = N_HASARD) -> str:
    """Partition `N` — NIVEAU, **par modèle ET par strate, jamais poolée**.

    `ic_lambda` = IC de `Λ = O_type − 2n` ; `ic_o` = IC de `O` **relatif à
    `n`** (c'est-à-dire de `O − n`). Corridor **ABSOLU** `2n`, jamais réglé sur
    l'enveloppe nulle de son propre estimateur (0-81 / D28).
    """
    lo_l, hi_l = float(ic_lambda[0]), float(ic_lambda[1])
    lo_o, hi_o = float(ic_o[0]), float(ic_o[1])
    deux_n = float(2 * n)
    if lo_l > float(eps_lambda):
        return "HAUT"
    if hi_o < 0.0:
        return "−"
    # `IC(O) ⊂ [0, 2n]` : `ic_o` est exprimé relativement à `n`, on le ramène.
    lo_abs, hi_abs = lo_o + float(n), hi_o + float(n)
    if 0.0 <= lo_abs and hi_abs <= deux_n:
        return "BAS"
    return "ind_L"


def classe_c(ic_delta, eps_etoile: float, n: Fraction = N_HASARD) -> str:
    """Partition `C` — CENTRAGE, `Δ* = O_plac − O_type`, **par modèle**.

    Ordre gravé, **négligeabilité physique d'abord** (M6, sens conservateur
    CONTRE l'hypothèse portée) : `c-anti` est évaluée en premier et
    **court-circuitante**.
    """
    lo, hi = float(ic_delta[0]), float(ic_delta[1])
    e = float(eps_etoile)
    deux_n = float(2 * n)
    if hi < -e:
        return "c-anti"
    if -deux_n <= lo and hi <= deux_n:
        return "c-mec"
    if lo > e:
        return "c-cent"
    return "ind_Δ"


def cellules_decisionnelles() -> dict:
    """**Espace de verdict recompté par ÉNUMÉRATION**, jamais par affirmation
    (famille 0-76(i) / 0-86 / 0-94 / 0-101 / 0-133).

    `c-anti` est court-circuitante : elle **ne croise pas** `N`. L'espace
    décisionnel est donc `4 (N) × 3 (C décisionnelles)`.
    """
    c_decisionnelles = tuple(c for c in CLASSES_C if c != "c-anti")
    cellules = [(a, b) for a in CLASSES_N for b in c_decisionnelles]
    return {"cellules": cellules, "cardinal": len(cellules),
            "classes_N": list(CLASSES_N),
            "classes_C_decisionnelles": list(c_decisionnelles),
            "c_anti": "court-circuitante — évaluée en premier, ne croise pas N",
            "cardinal_recompte_par": "énumération à l'exécution"}


def objets_du_banc() -> dict:
    """Les **10 objets simulés** du §4.6 : 4 classes de `N`, 4 états de `C`,
    2 issues de `Core`. Recomptés par énumération."""
    objets = ([("N", c) for c in CLASSES_N] + [("C", c) for c in CLASSES_C]
              + [("Core", "vide"), ("Core", "non vide")])
    return {"objets": objets, "cardinal": len(objets),
            "cardinal_recompte_par": "énumération à l'exécution"}


def ecritures_obligatoires(eps_etoile: float, n: Fraction = N_HASARD,
                           region_masquee=None) -> dict:
    """Les **deux écritures obligatoires** du §4.6 (M6, défaut 0-131).

    1. Si `ε* ≥ 2n`, le recouvrement de `c-mec` et `c-cent` est **VIDE** et
       l'ordre est **SANS OBJET** — à déclarer en toutes lettres.
    2. Sinon, publier la **région masquée**
       `P(IC ⊂ corridor ∧ IC_inf > ε*)` sous l'alternative : détection
       **étouffée assumée, jamais silencieuse**.
    """
    deux_n = float(2 * n)
    if float(eps_etoile) >= deux_n:
        return {"cas": 1, "epsilon_etoile": float(eps_etoile), "deux_n": deux_n,
                "declaration": "ε* ≥ 2n : le recouvrement des conditions "
                               "`c-mec` et `c-cent` est VIDE et l'ordre est "
                               "SANS OBJET",
                "region_masquee": SANS_OBJET}
    return {"cas": 2, "epsilon_etoile": float(eps_etoile), "deux_n": deux_n,
            "declaration": "ε* < 2n : la région masquée est publiée — "
                           "détection étouffée assumée, jamais silencieuse",
            "region_masquee": (SANS_OBJET if region_masquee is None
                               else float(region_masquee))}


# ---------------------------------------------------------------- `R` dérivé

def r_derive(sigma_dir: float, sigma_delta: float,
             plafond: int = R_PLAFOND) -> dict:
    """`R = ⌈10·σ̂_dir²/σ̂_Δ²⌉` — **dérivé, jamais choisi** (§4.5, anti-0-52).

    Si la formule dépasse le plafond, le dépassement est **publié et remonté au
    PI** : il n'est **pas tronqué en silence**.
    """
    sd, sD = float(sigma_dir), float(sigma_delta)
    if not (sD > 0 and np.isfinite(sd) and np.isfinite(sD)):
        return {"R": None, "statut": SANS_OBJET,
                "raison": "σ̂_Δ nulle ou non finie : la formule n'a pas de "
                          "domaine de définition"}
    brut = int(np.ceil(10.0 * sd * sd / (sD * sD)))
    depasse = brut > plafond
    return {"R_formule": brut, "R": min(brut, plafond), "plafond": plafond,
            "sigma_dir": sd, "sigma_delta": sD,
            "depassement": depasse,
            "statut": ("DÉPASSEMENT PUBLIÉ — remonté au PI, non absorbé"
                       if depasse else "sous plafond"),
            "formule": "R = ceil(10·σ̂_dir²/σ̂_Δ²), pilote R₀ = 20"}


# ------------------------------------------------------------------ PORTES

def v_p8(n_paires_par_cellule: dict, minimum: int = P_MIN_PAR_CELLULE) -> tuple:
    """`V-P8` — `Λ` et le corridor **uniquement** sur des moyennes de ≥ 8 paires.

    Motif (0-119) : `|A∩B| ~ Hypergéom(8192, 64, 64)` a `q95 = 2` indices **par
    paire**, soit le DOUBLE du corridor ; une paire isolée le dépasse 7.6 % du
    temps sous la nulle. Toute cellule à `< 8` paires est **exclue**, cardinal
    publié. `> 2` strates touchées sur un modèle ⇒ **modèle exclu** (§6.F).
    """
    exclues = {k: int(v) for k, v in n_paires_par_cellule.items()
               if int(v) < minimum}
    par_modele = {}
    for cle in exclues:
        m = cle[0] if isinstance(cle, tuple) else str(cle).split("|")[0]
        par_modele[m] = par_modele.get(m, 0) + 1
    modeles_exclus = sorted(m for m, c in par_modele.items() if c > 2)
    ok = not exclues
    return (PASS if ok else FAIL), {
        "minimum": minimum, "cellules_exclues": {str(k): v for k, v in exclues.items()},
        "cardinal_exclu": len(exclues),
        "n_cellules": len(n_paires_par_cellule),
        "modeles_exclus": modeles_exclus,
        "regle_modele": "> 2 strates touchées sur un modèle ⇒ modèle exclu (§6.F)"}


def v_t1(registre_t1=None, cardinal_intra_tige=None) -> tuple:
    """`V-t1` — **`t−1` refusé pour ce cycle** (0-130, ratifié par le PI).

    Deux raisons indépendantes, chacune suffisante : `A4` a établi que le
    porteur du domaine s'évanouit à `t−1` ; et à `t−1` les états d'une même
    tige sont **bit-identiques**, de sorte que `S3` et `S2` — qui sont
    **définies** par le partage de tige — rendraient `64/64` **par
    arithmétique** (0-64).

    La porte est donc : **aucune quantité de ce cycle n'est calculée à `t−1`**,
    et le **cardinal des paires intra-tige** (celles qui dégénéreraient) est
    **publié par strate et par modèle** (0-134). Une quantité `t−1` dont
    l'ensemble de comparaison contient deux unités de même tige fait échouer la
    porte — c'est le contre-exemple échouant obligatoire.

    *Tension relevée et publiée, non tranchée ici :* le libellé « exclusion de
    toute paire intra-tige » du §4.7, pris à `t`, viderait `S3` et `S2`, que le
    §7 et la table `L(c)` du §4.2 mesurent. La lecture retenue est celle qui
    laisse les deux sections exécutables ; elle est **déclarée**.
    """
    reg = list(registre_t1 or [])
    fautes = [q for q in reg if q.get("intra_tige")]
    ok = not fautes and not reg
    return (PASS if ok else FAIL), {
        "locus_du_cycle": "t", "t_moins_1": "REFUSÉ (0-130, ratifié PI)",
        "quantites_a_t_moins_1": reg, "n_quantites_a_t_moins_1": len(reg),
        "fautes_intra_tige": fautes,
        "cardinal_intra_tige_publie": (SANS_OBJET if cardinal_intra_tige is None
                                       else cardinal_intra_tige),
        "lecture_declaree": "la porte interdit `t−1`, et publie le cardinal "
                            "intra-tige ; elle ne retire pas S3/S2 de la "
                            "mesure à `t`, que le §7 et le §4.2 exigent"}


def v_core_s(S, tige_de) -> tuple:
    """`V-core-S` — `|S| = 10`, **une unité par tige**, appartenance publiée.

    Motif (0-124) : deux états intra-tige sont **corrélés** et brisent la nulle
    binomiale ; avec deux unités d'une même tige dans `S`, `Core > 0` devient
    **attendu**, et le chiffre le plus spectaculaire du cycle (`p < 10⁻¹³`)
    serait un artefact d'échantillonnage.
    """
    S = list(S)
    tiges = [tige_de[u] for u in S]
    doublons = sorted({t for t in tiges if tiges.count(t) > 1})
    ok = len(S) == CORE_TAILLE_S and not doublons and len(set(tiges)) == CORE_TAILLE_S
    return (PASS if ok else FAIL), {
        "taille_S": len(S), "taille_attendue": CORE_TAILLE_S,
        "appartenance": {str(u): tige_de[u] for u in S},
        "tiges_distinctes": len(set(tiges)), "tiges_en_double": doublons}


def v_ulp(marges_par_condition: dict) -> tuple:
    """`V-ulp` — marge à la coupure `k = 64` publiée pour les **CINQ**
    conditions, **y compris centrées et placebo** (0-125).

    Motif : les états **centrés** et **placebo** sont ceux dont les coordonnées
    sont rapprochées de zéro par soustraction ⇒ **les plus exposés** au
    basculement de rang à la coupure. La porte manquait exactement là où le
    risque est maximal. Frère de 0-51.

    Porte de **COUVERTURE** (opérationnalisation déclarée) : elle échoue si
    l'une des cinq conditions n'a pas sa marge. Les marges sont publiées en
    ULP, en descriptif ; **aucun seuil n'est posé** — en poser un que le
    protocole n'a pas pré-enregistré serait 0-52. fp64 = chemin nominal.
    """
    manquantes = [c for c in CENTRAGES
                  if c not in marges_par_condition
                  or marges_par_condition[c] is None]
    non_finies = [c for c, v in marges_par_condition.items()
                  if v is not None and not np.isfinite(float(
                      v["marge_ulp_min"] if isinstance(v, dict) else v))]
    ok = not manquantes and not non_finies
    return (PASS if ok else FAIL), {
        "conditions_attendues": list(CENTRAGES),
        "conditions_publiees": sorted(marges_par_condition),
        "conditions_manquantes": manquantes,
        "marges_non_finies": non_finies,
        "marges": {c: marges_par_condition.get(c) for c in CENTRAGES},
        "chemin_nominal": "fp64",
        "seuil": SANS_OBJET}


def v_norm(configs_par_modele: dict) -> tuple:
    """`V-norm` — LayerNorm vs RMSNorm **cité par ligne de config** des trois
    modèles, **jamais par croyance** (0-123).

    LayerNorm **centre** ; RMSNorm **ne centre pas** (Zhang & Sennrich 2019) ⇒
    `G` est appliquée à un vecteur que le cortex lui-même n'utilise jamais sous
    cette forme, **et pas de la même manière selon le modèle**. C'est la seule
    cause candidate de `C-mod` vérifiable dans le code des modèles.
    """
    manques = [m for m in MODELES if m not in configs_par_modele]
    sans_citation = [m for m, c in configs_par_modele.items()
                     if not c.get("ligne_de_config") or not c.get("normalisation")]
    inconnues = [m for m, c in configs_par_modele.items()
                 if c.get("normalisation") not in ("LayerNorm", "RMSNorm")]
    ok = not manques and not sans_citation and not inconnues
    return (PASS if ok else FAIL), {
        "modeles_manquants": manques, "sans_citation": sans_citation,
        "normalisations_inconnues": inconnues,
        "configs": configs_par_modele,
        "regle": "citée par ligne de config, jamais par croyance"}


def v_seed(spec: dict) -> tuple:
    """`V-seed` — seed, générateur et **règle de tirage** des `R` directions
    **gelés et publiés** ; cardinal **par modèle** (0-132, 0-133).

    Les directions **ne sont pas transportables** entre modèles : `d` vaut
    768 / 960 / 1536, et un vecteur de ℝ⁷⁶⁸ n'est pas un vecteur de ℝ¹⁵³⁶.
    Même seed, même générateur, même règle, mêmes indices ; l'appariement
    inter-modèles se fait par la **NORME**, jamais par le **VECTEUR**.
    """
    champs = ("seed", "generateur", "regle_de_tirage", "indices",
              "cardinal_par_modele", "appariement_inter_modeles")
    # `seed = 0` est la valeur du protocole (D9) : tester la VÉRITÉ du champ le
    # rejetterait comme absent. On teste la PRÉSENCE et la non-vacuité.
    manques = [c for c in champs
               if c not in spec or spec[c] is None or spec[c] in ("", {}, [])]
    card = spec.get("cardinal_par_modele") or {}
    modeles_manquants = [m for m in MODELES if m not in card]
    par_vecteur = str(spec.get("appariement_inter_modeles", "")).lower()
    faute_transport = "vecteur" in par_vecteur and "norme" not in par_vecteur
    ok = not manques and not modeles_manquants and not faute_transport
    return (PASS if ok else FAIL), {
        "champs_manquants": manques, "modeles_sans_cardinal": modeles_manquants,
        "appariement_par_le_vecteur": faute_transport,
        "spec": spec,
        "regle": "appariement inter-modèles par la NORME, jamais par le VECTEUR"}


def v_diag(schema: dict) -> tuple:
    """`V-diag` — **condition bloquante de `lab-neuro`** (0-120).

    `f(s)`, `σ±(s)` et le couple **`(O, cos)` conjoint** publiés **par strate,
    par condition, par modèle**. **Sans les trois, le rapport ne peut pas être
    écrit** (§6.D).

    Motif : sans `f`, l'état *« les supports coïncident »* et l'état *« quelques
    coordonnées géantes portent tout le cosinus »* rendent le **même `O`**,
    interprété de la même façon — et le second est précisément la branche de
    tort de Neuro (N7). Un dispositif dont aucune donnée ne peut réfuter
    l'expert qui le signe n'est pas un dispositif.

    Ces trois quantités sont **DIAGNOSTIQUES, jamais correctives** : elles ne
    modifient aucune borne, aucun seuil, aucune classe.
    """
    requis = ("f", "sigma_pm", "O_cos_conjoint")
    manques = []
    for m in schema.get("modeles", MODELES):
        for x in schema.get("conditions", CENTRAGES):
            for s in STRATES:
                cel = schema.get("cellules", {}).get(f"{m}|{x}|{s}")
                if cel is None:
                    manques.append(f"{m}|{x}|{s} : cellule absente")
                    continue
                for r in requis:
                    if r not in cel or cel[r] is None:
                        manques.append(f"{m}|{x}|{s} : {r} absent")
    ok = not manques
    return (PASS if ok else FAIL), {
        "quantites_requises": list(requis), "manques": manques[:50],
        "n_manques": len(manques),
        "statut": "DIAGNOSTIQUE, jamais correctif (Arbitrage X-1)"}


def v_perimetre(rapport: dict) -> tuple:
    """`V-perimetre` — les **trois** éléments du §4.9, obligatoires."""
    manques = [k for k in ("mecanisme", "limite_nommee", "successeur_designe")
               if not rapport.get(k)]
    return (PASS if not manques else FAIL), {"elements_manquants": manques}


def _contient_terme_interdit(texte: str) -> list:
    bas = texte.lower()
    for ph in PHRASES_GRAVEES:
        bas = bas.replace(ph.lower(), " ")
    return [t for t in VOCABULAIRE_INTERDIT if t.lower() in bas]


def v_schema(schema: dict) -> tuple:
    """Porte de SCHÉMA : aucun terme du vocabulaire interdit (i)-(xx), `O` en
    **fractions exactes**, phrase gravée (xvi) présente et verbatim."""
    texte = json.dumps(schema, ensure_ascii=False, default=str)
    termes = _contient_terme_interdit(texte)
    manques = []
    if schema.get("phrase_gravee_xvi") != PHRASE_XVI:
        manques.append("phrase gravée (xvi) absente ou altérée")
    if schema.get("unite_de_O") != "fraction exacte p/64":
        manques.append("unité de `O` : fraction exacte p/64 (0-132)")
    ok = not termes and not manques
    return (PASS if ok else FAIL), {"termes_interdits_detectes": termes,
                                    "champs_manquants": manques}


def v_borne(paires, seuil_encadrement=ENCADREMENT_TABLE_SEUIL) -> tuple:
    """`V-borne` — version **RÉALISÉE par paire** (0-128, candidate D30).

    Deux membres :

    * **(a) identité arithmétique** — `|A∩B| ≥ p_pair` sur **100 %** des
      paires. Toute violation est un **BUG DE MESURE**, jamais un résultat
      (§6.B). Ce membre est exécutable **maintenant** ;
    * **(b) encadrement** — la version réalisée doit **encadrer** la table
      théorique du §4.2. Ce membre a besoin d'un seuil ; **Q-M5 n'est pas
      rendue** et le protocole n'en pré-enregistre aucun. Il rend donc
      `EN-ATTENTE` — poser une valeur ici serait **0-52**.

    Le verdict global est `PASS` seulement si (a) passe **et** (b) est résolu.
    """
    paires = list(paires)
    viol, viol_lache = [], []
    for i, p in enumerate(paires):
        inter = int(p["inter"])
        ps = int(p["p_sym"])
        if inter < ps:
            viol.append({"i": i, "inter": inter, "p_sym": ps,
                         "cos": p.get("cos")})
        if "p_pair" in p and inter < int(p["p_pair"]):
            viol_lache.append({"i": i, "inter": inter,
                               "p_pair": int(p["p_pair"])})
    membre_a = PASS if not viol else FAIL
    if seuil_encadrement is None:
        membre_b = EN_ATTENTE
        det_b = {"statut": EN_ATTENTE,
                 "raison": "Q-M6 non rendue : `lab-math` doit fournir la table "
                           "`cum(j)` exacte pour j = 1..20 (fp64, double "
                           "normalisation 569.6 / 563.9) et les douze `p_sym`. "
                           "Aucun seuil n'est posé ici, et aucune valeur de "
                           "`cum` n'est extrapolée (règles Q-M5 et anti-0-52)."}
    else:
        membre_b = PASS
        det_b = {"statut": PASS, "seuil": float(seuil_encadrement)}
    verdict = membre_a if membre_a == FAIL else (
        PASS if membre_b == PASS else EN_ATTENTE)
    ecarts = [int(p["p_sym"]) - int(p["p_pair"]) for p in paires
              if "p_pair" in p]
    return verdict, {
        "borne_decisionnelle": "p_sym (serrée, §16, D30 alinéa 2)",
        "membre_a_identite": membre_a, "n_paires": len(paires),
        "violations_p_sym": viol[:20], "n_violations": len(viol),
        "taux_conformite": (None if not paires else
                            1.0 - len(viol) / len(paires)),
        "descriptif_p_pair": {
            "statut": "DESCRIPTIF (borne lâche, ce que le gel portait, §16-3)",
            "n_violations": len(viol_lache),
            "ecart_p_sym_moins_p_pair": (
                {"min": min(ecarts), "max": max(ecarts),
                 "moyen": float(np.mean(ecarts))} if ecarts else SANS_OBJET)},
        "membre_b_encadrement": det_b,
        "regle": "violation = BUG DE MESURE, jamais un résultat (§6.B)"}


def v_iid(G) -> tuple:
    """`V-iid` — **prouvée par provenance** (M3) : `G` est bâtie par
    `torch.randn(8192, d, generator=seed(cfg.seed))`, ses lignes sont donc iid
    gaussiennes **par définition**. Le MC conditionnel est **SUPPRIMÉ** du
    budget (0-126-i) : il dépenserait pour re-prouver une propriété de
    provenance.

    Résiduel : **sanité de moments**. **Pas de KS** (§4.7).
    """
    G = np.asarray(G, dtype=np.float64)
    mu, sd = float(G.mean()), float(G.std(ddof=1))
    ok = (abs(mu) < 0.02 and abs(sd - 1.0) < 0.02 and np.isfinite(G).all())
    return (PASS if ok else FAIL), {
        "moyenne": mu, "ecart_type": sd, "forme": list(G.shape),
        "preuve": "par provenance (torch.randn, lignes iid) — M3",
        "residuel": "sanité de moments ; PAS de KS",
        "tolerances": {"|moyenne|": 0.02, "|σ−1|": 0.02}}


# =========================================================================
#  V-cache et V-G (v2) — portes de PROVENANCE
# =========================================================================

def sha256_fichier(p: Path, bloc: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            m = f.read(bloc)
            if not m:
                break
            h.update(m)
    return h.hexdigest()


def sha256_tenseur(t) -> str:
    a = np.ascontiguousarray(np.asarray(t))
    return hashlib.sha256(a.tobytes()).hexdigest()


def v_cache(modeles=MODELES) -> tuple:
    """`V-cache` : présence, lisibilité, **hash publié**, dtype publié (D21).

    Limite nommée (§3) : ce hash **n'a jamais été pré-enregistré au cycle v4**.
    Le graver est une **limite**, pas une vérification. Échec ⇒ **re-forward
    autorisé** par la gate PI (§14-4) : 53,82 s, VRAM ≤ 4,688 Gio **en
    RÉSERVÉ** (0-142) — seul poste GPU du cycle.
    """
    det = {"repertoire": str(RAW_V4), "fichiers": {},
           "limite_nommee": "le hash des `.npz` de v4 n'a jamais été "
                            "pré-enregistré : il est gravé ici pour la suite, "
                            "ce qui est une LIMITE, pas une vérification (§3)",
           "poste_gpu_si_echec": {"duree_s": 53.82,
                                  "vram_gio_max_reservee": 4.688,
                                  "vram_gio_max_allouee": 4.188,
                                  "unite": "RÉSERVÉ (§7, 0-142)",
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
        ex = sorted(k for k in cles if k.endswith("|t"))[0]
        det["fichiers"][m] = {
            "present": True, "chemin": str(p), "taille_octets": p.stat().st_size,
            "sha256": sha256_fichier(p), "n_cles": len(cles),
            "dtype_etats": str(z[ex].dtype), "forme_etats": list(z[ex].shape),
            "cle_exemple": ex}
    det["fichiers_manquants"] = manquants
    return (PASS if not manquants else FAIL), det


def materiau_v4() -> dict:
    """Le matériau qualifié v4, **relu** depuis le brut du cycle précédent —
    jamais reconstruit : l'indexation des `.npz` dépend de l'ordre exact des
    unités sous lequel les états ont été capturés."""
    return json.loads((RES_V4 / "pool_v4.json").read_text(encoding="utf-8"))["materiau"]


def a3_publie(nom_modele: str) -> dict:
    """`A3` **relu depuis `experiments/results/`** (D14-R, §6.I) — jamais de
    mémoire. Les deux colonnes sont publiées séparément : `cos_dg` (le cosinus
    de la projection d'`A3`) et `compression` (`cos_brut − cos_dg`). Les
    confondre était le défaut **0-136**."""
    j = json.loads((RES_V4 / MESURE_V4[nom_modele]).read_text(encoding="utf-8"))
    a3 = j["metrics"]["descriptif_A3"]
    return {"cos_projection_A3": {k: a3[k]["cos_dg"] for k in STRATES},
            "cos_brut": {k: a3[k]["cos_brut"] for k in STRATES},
            "compression": {k: a3[k]["compression"] for k in STRATES},
            "n_paires": {k: a3[k]["n"] for k in STRATES},
            "amplitude_de_compression": j["metrics"]["descriptif_A3_amplitude"],
            "source": str(RES_V4 / MESURE_V4[nom_modele]),
            "avertissement": "A3 porte sur une AUTRE projection : sa `G` est "
                             "tirée par numpy.default_rng(0), pas par "
                             "torch.randn(generator=manual_seed(0)) — 0-135"}


def _cos_par_strate_fp(Z_par_cellule, dec, k: int, fp64: bool) -> dict:
    """`cos` de `φ(h)` par strate, moyenné sur les cellules — en fp32 ou fp64."""
    n = len(dec)
    COS = np.zeros((n, n), dtype=np.float64)
    for Z in Z_par_cellule:
        Zc = np.asarray(Z, dtype=np.float64 if fp64 else np.float32)
        P = np.zeros_like(Zc)
        for r in range(Zc.shape[0]):
            idx = support_topk(Zc[r].astype(np.float64), k)
            P[r, idx] = Zc[r, idx]
        nor = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / np.maximum(nor, np.finfo(P.dtype).tiny)
        COS += (P @ P.T).astype(np.float64)
    COS /= len(Z_par_cellule)
    par = {}
    for a in range(n):
        for b in range(a + 1, n):
            par.setdefault(p4.strate(dec[a], dec[b]), []).append(COS[a, b])
    return {s: float(np.mean(par[s])) for s in STRATES}


def v_g(modeles=MODELES, cfg: EngramConfig | None = None,
        tol_precision: float = 1e-6) -> tuple:
    """`V-G` **v2** — re-spécifiée le 2026-08-26 (§15, défauts 0-135 / 0-138).

    *La v1 exigeait la reproduction bit-à-bit d'`A3` par `hippocampus.phi`.
    Elle a échoué, et elle avait raison : `A3` n'a jamais été calculé avec la
    `G` du projet. La v1 est **inexécutable par construction**.*

    **v2**, trois membres, tous exécutables :

    (a) `G` **du projet** instanciée via `hippocampus.phi`
        (`torch.randn(8192, d, generator=seed(cfg.seed))`), **hash publié par
        modèle** ;
    (b) `cos` de `φ(h)` **par strate** mesuré et publié avec cette `G`, en
        **fp32 et fp64**, **écart publié** (D26) ;
    (c) **publication obligatoire de la divergence avec `A3`** — les douze
        écarts, le `corrcoef` entre les deux matrices, et la mention que
        **`A3` porte sur une autre projection**.

    Échec si : hash absent, `G` non instanciable, ou divergence fp32/fp64
    `> 1e−6` ⇒ **arrêt de provenance (D14-R)**.
    """
    cfg = cfg or EngramConfig()
    mat = materiau_v4()
    dec = mat["unites_decisionnelles"]
    idx = [i for i, u in enumerate(mat["unites"]) if u["pontee"]]
    cellules = list(mat["cellules"].keys())
    det = {"version": "v2 (§15, 2026-08-26)", "cfg": cfg.summary(),
           "tolerance_fp32_fp64": tol_precision, "modeles": {}}
    ok_global = True
    for m in modeles:
        t0 = time.time()
        z = np.load(RAW_V4 / f"etats-{slug(m)}.npz")
        couche = COUCHE_REF[m]
        d = int(z[f"{cellules[0]}|t"].shape[2])
        mem = FastWeightMemory(d, cfg, device="cpu")     # M jamais lue ni écrite
        G32 = mem.G.numpy()
        G64 = G32.astype(np.float64)
        H = [z[f"{c}|t"][couche][idx] for c in cellules]
        Z64 = [np.asarray(h, dtype=np.float64) @ G64.T for h in H]
        cos64 = _cos_par_strate_fp(Z64, dec, cfg.dg_topk, fp64=True)
        cos32 = _cos_par_strate_fp(
            [np.asarray(h, dtype=np.float32) @ G32.T for h in H],
            dec, cfg.dg_topk, fp64=False)
        ecart_prec = {s: cos64[s] - cos32[s] for s in STRATES}
        max_prec = max(abs(v) for v in ecart_prec.values())
        # (c) divergence avec `A3` — obligatoire
        pub = a3_publie(m)
        rng = np.random.default_rng(cfg.seed)
        G_a3 = rng.standard_normal((cfg.dg_dim, d)).astype(np.float32) / np.sqrt(d)
        corr = float(np.corrcoef(G64[:64].ravel(),
                                 np.asarray(G_a3[:64], np.float64).ravel())[0, 1])
        ecart_a3 = {s: cos32[s] - pub["cos_projection_A3"][s] for s in STRATES}
        ok = max_prec <= tol_precision and np.isfinite(G64).all()
        ok_global = ok_global and ok
        det["modeles"][m] = {
            "couche": couche, "d": d,
            "G_du_projet": {
                "generateur": "torch.randn(dg_dim, d, generator=manual_seed(seed))",
                "seed": cfg.seed, "forme": list(G32.shape),
                "dtype": str(mem.G.dtype), "echelle": "aucune (pas de 1/√d)",
                "sha256": sha256_tenseur(G32),
                "portee": "pour cette G, seed 0 (D9)"},
            "cos_phi_par_strate_fp32": cos32,
            "cos_phi_par_strate_fp64": cos64,
            "ecart_fp64_moins_fp32": ecart_prec,
            "ecart_max_de_precision": max_prec,
            "V-iid": dict(zip(("verdict", "detail"), v_iid(G64[:512]))),
            "divergence_avec_A3": {
                "A3_publie": pub, "ecart_cos_fp32_moins_A3": ecart_a3,
                "corrcoef_G_projet_vs_G_A3": corr,
                "mention_obligatoire": "A3 porte sur une AUTRE projection "
                                       "(G tirée par numpy, 0-135) : aucune "
                                       "reproduction d'A3 n'est exigée ni "
                                       "possible"},
            "verdict": PASS if ok else FAIL,
            "duree_s": round(time.time() - t0, 2)}
        del mem, G32, G64, G_a3
    det["verdict_porte"] = (
        "G du projet instanciée et hashée ; cos publiés en fp32 et fp64 ; "
        "divergence avec A3 publiée" if ok_global else
        "hash absent, G non instanciable, ou divergence fp32/fp64 > tolérance "
        "⇒ arrêt de provenance (D14-R)")
    return (PASS if ok_global else FAIL), det


# =========================================================================
#  ==========================  ÉTAGE DE MESURE  ==========================
#
#  Ouvert seulement après PASS des portes de provenance ET `E = 0` au banc
#  (§6.E). `M` n'est jamais instanciée ; aucune injection ; `engram/` intact.
# =========================================================================

class ArretDeProvenance(RuntimeError):
    """§6.A — arrêt de provenance (D14-R), **jamais un repli**."""


class BancNonFranchi(RuntimeError):
    """§6.E — **aucune mesure avant PASS intégral du banc**."""


def config_de_normalisation(nom_modele: str) -> dict:
    """`V-norm` : lecture de la config du modèle, **par ligne de config**.

    Aucun poids n'est chargé — seul `config.json`, déjà en cache, est lu.
    """
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(nom_modele, local_files_only=True).to_dict()
    if "layer_norm_epsilon" in c or "layer_norm_eps" in c:
        cle = "layer_norm_epsilon" if "layer_norm_epsilon" in c else "layer_norm_eps"
        norme, centre = "LayerNorm", True
    elif "rms_norm_eps" in c:
        cle, norme, centre = "rms_norm_eps", "RMSNorm", False
    else:
        cle, norme, centre = None, None, None
    return {"model_type": c.get("model_type"), "normalisation": norme,
            "centre_t_il": centre, "champ": cle,
            "ligne_de_config": (f'"{cle}": {c[cle]}' if cle else None),
            "reference": "Zhang & Sennrich 2019 — RMSNorm ne centre pas"}


def etats_decisionnels(nom_modele: str, couche: int) -> tuple:
    """Les états `h` des 60 unités décisionnelles, à `t`, par cellule.

    `t−1` **n'est jamais lu** (`V-t1`, 0-130).
    """
    mat = materiau_v4()
    idx = [i for i, u in enumerate(mat["unites"]) if u["pontee"]]
    z = np.load(RAW_V4 / f"etats-{slug(nom_modele)}.npz")
    cellules = list(mat["cellules"].keys())
    H = {c: np.asarray(z[f"{c}|t"][couche][idx], dtype=np.float64)
         for c in cellules}
    return mat, H, cellules


def marge_a_la_coupure(z, k: int = K_TOPK) -> float:
    """Marge à la coupure `k`, en **ULP** de la valeur de coupure (`V-ulp`)."""
    a = np.sort(np.abs(np.asarray(z, dtype=np.float64)))[::-1]
    coupe, suivant = a[k - 1], a[k]
    u = np.spacing(coupe) if coupe > 0 else np.spacing(1.0)
    return float((coupe - suivant) / u)


def _tiges_et_S(mat):
    dec = mat["unites_decisionnelles"]
    tige_de = {i: dec[i]["tige"] for i in range(len(dec))}
    S, vus = [], set()
    for i in range(len(dec)):          # PREMIÈRE unité de chaque tige
        if tige_de[i] not in vus:
            vus.add(tige_de[i])
            S.append(i)
    return dec, tige_de, S


def _directions(d: int, R: int, seed: int):
    """Règle de tirage GELÉE et PUBLIÉE (`V-seed`) — non transportable (0-133)."""
    g = torch.Generator().manual_seed(int(seed))
    r = torch.randn(int(R), int(d), generator=g).double().numpy()
    return r / np.linalg.norm(r, axis=1, keepdims=True)


# ------------------------------------------------- noyaux de la mesure
#
#  Tous purs : ils prennent des tableaux et rendent des nombres. Le banc les
#  exerce sur des cas synthétiques dont la réponse est connue par
#  construction ; aucun n'a besoin d'un `.npz` ni d'un modèle.

def z_centre(z_a, z_b, condition: str, ctx: dict):
    """`G·h` après centrage, par l'identité `G·(h − v) = z − G·v` (0-126-ii).

    Les cinq conditions passent par le **même chemin de calcul** : seule
    l'entrée change (§5-4). Aucun recalcul complet de `G·h` par direction —
    sans cette identité le placebo sort du budget.

    * `aucun` : `z`
    * `glob`  : `z − G·μ_global^{LOO}` ; `G·μ_{-i} = (Σz − z_i)/(N−1)`
    * `type`  : `z − G·μ_type^{LOO-paire}` ;
                `G·μ_{-{a,b}} = (Σ_cellule z − z_a − z_b)/(n_cell−2)`
    * `auto`  : `z − mean_coord(h)·(G·1)` — **seul centrage calculable EN
                LIGNE**, sur le seul état courant : aucun corpus, aucun LOO,
                aucune fuite
    * `plac`  : `z − c·(G·r)`, `‖c·r‖ = ‖μ_type^{LOO-paire}‖` — contrôle du
                décalage MÉCANIQUE de magnitudes (0-106)
    """
    if condition == "aucun":
        return z_a, z_b
    if condition == "glob":
        n = ctx["n_global"]
        s = ctx["somme_globale"]
        return z_a - (s - z_a) / (n - 1), z_b - (s - z_b) / (n - 1)
    if condition == "type":
        m = ctx["mu_type_z"]
        return z_a - m, z_b - m
    if condition == "auto":
        g1 = ctx["g1"]
        return z_a - ctx["moy_a"] * g1, z_b - ctx["moy_b"] * g1
    if condition == "plac":
        gr = ctx["Gr_colonne"]
        c = ctx["norme_mu_type"]
        return z_a - c * gr, z_b - c * gr
    raise ValueError(f"condition inconnue : {condition}")


def quantites_de_paire(z_a, z_b, k: int = K_TOPK) -> dict:
    """Toutes les quantités d'une paire, en fp64, à partir des deux `G·h`.

    Rend `|A∩B|` (entier), `cos(φ_a, φ_b)`, `f` (énergie de l'intersection,
    moyennée sur les deux membres), `σ±` (accord de signe — **`SANS OBJET`**
    quand l'intersection est vide, jamais `0`), et `p_pair` **réalisé**.
    """
    A = support_topk(z_a, k)
    B = support_topk(z_b, k)
    inter = intersection(A, B)
    pa, pb = phi_de(z_a, A), phi_de(z_b, B)
    cos = float(np.dot(pa, pb))
    if inter:
        comm = np.intersect1d(A, B, assume_unique=True)
        f = 0.5 * (float(np.sum(pa[comm] ** 2)) + float(np.sum(pb[comm] ** 2)))
        acc = int(np.sum(np.sign(pa[comm]) == np.sign(pb[comm])))
        sigma = frac(acc, inter)
    else:
        f, sigma = 0.0, None          # `f` est 0 par sommation vide ; `σ±` n'a
                                      # pas de domaine de définition → SANS OBJET
    borne = p_pair_realise(pa[A], pb[B], cos)
    return {"inter": inter, "cos": cos, "f": f,
            "sigma_pm": sigma,
            "p_sym": borne["p_sym"],            # DÉCISIONNELLE (§16)
            "p_pair": borne["p_pair"],          # DESCRIPTIVE (borne lâche)
            "ecart_psym_moins_ppair": borne["ecart_psym_moins_ppair"],
            "p_a": borne["p_a"], "p_b": borne["p_b"],
            "marge_ulp_a": marge_a_la_coupure(z_a, k),
            "marge_ulp_b": marge_a_la_coupure(z_b, k)}


def agreger(paires) -> dict:
    """Agrégation d'une cellule (modèle × condition × strate).

    `O` en **fraction exacte `p/64`** (0-132) ; **médiane et IQR publiées**,
    parce qu'une moyenne mélange un mode « noyau » et un mode nul (contrainte
    de la phrase gravée (xvi)). Cellule vide ⇒ **`SANS OBJET`, jamais `0`**.
    """
    paires = list(paires)
    P = len(paires)
    if P == 0:
        return {"P": 0, "O": SANS_OBJET, "O_fraction": None, "mediane": SANS_OBJET,
                "IQR": SANS_OBJET, "f": SANS_OBJET, "sigma_pm": SANS_OBJET,
                "cos_moyen": SANS_OBJET, "corr_O_cos": SANS_OBJET}
    inters = np.array([p["inter"] for p in paires], dtype=np.int64)
    cos = np.array([p["cos"] for p in paires], dtype=np.float64)
    O = o_fraction(int(inters.sum()), P)
    q1, med, q3 = (np.quantile(inters, q) for q in (0.25, 0.5, 0.75))
    avec = [p for p in paires if p["sigma_pm"] is not None]
    sig = (float(np.mean([float(p["sigma_pm"]) for p in avec]))
           if avec else SANS_OBJET)
    corr = (float(np.corrcoef(inters, cos)[0, 1])
            if inters.std() > 0 and cos.std() > 0 else SANS_OBJET)
    return {"P": P,
            "O": texte_fraction(O), "O_fraction": O, "O_flottant": float(O),
            "mediane": frac(int(med), K_TOPK), "mediane_texte":
                texte_fraction(frac(int(med), K_TOPK)),
            "IQR": [texte_fraction(frac(int(q1), K_TOPK)),
                    texte_fraction(frac(int(q3), K_TOPK))],
            "f": float(np.mean([p["f"] for p in paires])),
            "sigma_pm": sig, "n_paires_avec_intersection": len(avec),
            "cos_moyen": float(cos.mean()),
            "O_cos_conjoint": {"O_moyen": float(O), "cos_moyen": float(cos.mean()),
                               "corr_O_cos": corr},
            "corr_O_cos": corr}


def poids_bootstrap(tiges_de_paire, multiplicites: dict) -> np.ndarray:
    """Poids d'une paire dans un rééchantillon de tiges — **opérationnalisation
    déclarée** : `m[t_a]·m[t_b]` si `t_a ≠ t_b`, `m[t_a]` sinon.

    L'unité d'échange est la **TIGE** (0-50), `K_eff = 10` ; une paire `S1`/`S0`
    a **deux** tiges, et le protocole ne dit pas laquelle — c'est la clause que
    cette règle opérationnalise, publiée hors `E`.
    """
    w = np.empty(len(tiges_de_paire), dtype=np.float64)
    for i, (ta, tb) in enumerate(tiges_de_paire):
        ma, mb = multiplicites.get(ta, 0), multiplicites.get(tb, 0)
        w[i] = ma if ta == tb else ma * mb
    return w


def bootstrap_tiges(valeurs, tiges_de_paire, tiges, b: int = B_BOOT,
                    seed: int = 0) -> dict:
    """IC 95 % **bootstrap de TIGES**, `B = 10⁴`, percentile.

    Réserve reconduite : à 10 clusters un bootstrap percentile **sous-couvre
    légèrement** ; la protection est que la classe `ind_L` / `ind_Δ` soit
    prononcée plutôt qu'un seuil ajusté.
    """
    v = np.asarray(valeurs, dtype=np.float64)
    if v.size == 0:
        return {"estime": None, "IC": None, "statut": SANS_OBJET,
                "raison": "aucune paire : la statistique n'a pas de domaine de "
                          "définition"}
    rng = np.random.default_rng(seed)
    tiges = list(tiges)
    K = len(tiges)
    ech = np.empty(b, dtype=np.float64)
    for r in range(b):
        tirage = rng.integers(0, K, K)
        mult = {}
        for j in tirage:
            mult[tiges[j]] = mult.get(tiges[j], 0) + 1
        w = poids_bootstrap(tiges_de_paire, mult)
        sw = w.sum()
        ech[r] = float(np.dot(w, v) / sw) if sw > 0 else np.nan
    ech = ech[np.isfinite(ech)]
    lo, hi = np.quantile(ech, [ALPHA / 2, 1 - ALPHA / 2])
    return {"estime": float(v.mean()), "IC": [float(lo), float(hi)],
            "K_eff": K, "B": b, "seed": seed, "n_repliques_finies": int(ech.size),
            "demi_largeur": float((hi - lo) / 2),
            "reserve": "à 10 clusters le percentile sous-couvre légèrement ; "
                       "la protection est de prononcer ind_L / ind_Δ, jamais "
                       "d'ajuster un seuil"}


def eps_etoile(delta_par_paire, tiges_de_paire, tiges, b: int = B_PERM,
               seed: int = 0) -> dict:
    """**LIGNE CANONIQUE UNIQUE** de `ε*` — recopiée à l'identique du §4.5 et du
    §7, une seule fois dans le code (anti-0-82) :

    > `ε*` = q₀.₉₅ de `|Δ*|` recalculée sur `B = 10⁴` rééchantillons bootstrap
    > de tiges (`K_eff = 10`) sous permutation intra-tige des étiquettes
    > {plac, type}, le placebo étant la moyenne des `R` directions gelées ; une
    > valeur par modèle et par strate, calculée et gelée **AVANT** lecture des
    > `Δ*` observés.

    La permutation des étiquettes est **constante dans une tige**
    (opérationnalisation déclarée) : une paire est affectée à `min(t_a, t_b)`.
    """
    d0 = np.asarray(delta_par_paire, dtype=np.float64)
    if d0.size == 0:
        return {"epsilon_etoile": None, "statut": SANS_OBJET,
                "raison": "aucune paire"}
    tiges = list(tiges)
    K = len(tiges)
    rang = {t: i for i, t in enumerate(tiges)}
    affect = np.array([rang[min(ta, tb)] for ta, tb in tiges_de_paire])
    rng = np.random.default_rng(seed)
    stat = np.empty(b, dtype=np.float64)
    for r in range(b):
        tirage = rng.integers(0, K, K)
        mult = {}
        for j in tirage:
            mult[tiges[j]] = mult.get(tiges[j], 0) + 1
        w = poids_bootstrap(tiges_de_paire, mult)
        signes = rng.integers(0, 2, K) * 2 - 1          # ±1 par TIGE
        sw = w.sum()
        stat[r] = (float(np.dot(w, d0 * signes[affect]) / sw)
                   if sw > 0 else np.nan)
    stat = stat[np.isfinite(stat)]
    return {"epsilon_etoile": float(np.quantile(np.abs(stat), Q_ENVELOPPE)),
            "sigma_delta": float(d0.std(ddof=1)) if d0.size > 1 else 0.0,
            "B": b, "seed": seed, "K_eff": K, "n_paires": int(d0.size),
            "formule": "q₀.₉₅ de |Δ*| sur B = 10⁴ rééchantillons bootstrap de "
                       "tiges sous permutation intra-tige des étiquettes "
                       "{plac, type}",
            "gele_avant_lecture": True}


def core(supports_par_unite, seuil: int = CORE_SEUIL) -> dict:
    """`Core(S)` — indices retenus sur **≥ 9 des 10 unités de `S`**.

    Nulle **exacte** : `Bin(10, 64/8192)` par indice ⇒ `P(≥9) ≈ 1.1e−18`,
    espérance `≈ 9e−15` indice ⇒ **un seul indice est une signature**,
    `p < 10⁻¹³`, **aucun IC** (décision P5 du PI). Portée : **« pour cette
    `G` », seed 0** — `Core` est **G-spécifique** (D9).
    """
    sup = list(supports_par_unite)
    cnt = np.zeros(D_DG, dtype=np.int64)
    for s in sup:
        cnt[np.asarray(s, dtype=np.int64)] += 1
    idx = np.flatnonzero(cnt >= seuil)
    return {"Core": sorted(int(i) for i in idx), "taille": int(idx.size),
            "n_unites": len(sup), "seuil": seuil,
            "nulle": "Bin(10, 64/8192) par indice ; P(≥9) ≈ 1.1e-18 ; "
                     "un seul indice = signature, p < 1e-13, aucun IC",
            "portee": "pour cette G, seed 0 (D9) — Core est G-spécifique"}


def core_g(core_indices, support_de_G_mu_global) -> dict:
    """`Core-G = |Core(S) ∩ top64(G·μ_global)| / |Core(S)|`.

    `Core` vide ⇒ la fraction n'a **pas de domaine de définition** :
    **`SANS OBJET`, jamais `0`**.
    """
    c = np.asarray(sorted(core_indices), dtype=np.int64)
    if c.size == 0:
        return {"Core_G": SANS_OBJET, "taille_Core": 0,
                "raison": "Core vide : la fraction n'a pas de domaine de "
                          "définition"}
    inter = int(np.intersect1d(c, np.asarray(support_de_G_mu_global),
                               assume_unique=True).size)
    return {"Core_G": frac(inter, int(c.size)),
            "Core_G_texte": texte_fraction(frac(inter, int(c.size))),
            "Core_G_flottant": inter / c.size,
            "intersection": inter, "taille_Core": int(c.size)}


def bascule_split_half(n_cell_min: int) -> dict:
    """§14-2 — bascule **RATIFIÉE, AUTOMATIQUE et PRÉ-ENREGISTRÉE** (0-117).

    Si `min n_cell < 30`, le **split-half devient la mesure principale** et le
    LOO passe en contrôle. Motif : le biais résiduel du LOO,
    `+1/(n_cell−2) ≈ 0.05` de cosinus à `n_cell ≈ 20`, est **de la taille de la
    quantité décidée** — il fabriquerait `c-cent`. **La bascule étant
    pré-enregistrée, ce n'est pas un choix après lecture.**
    """
    n = int(n_cell_min)
    bascule = n < N_CELL_BASCULE
    return {"n_cell_min": n, "seuil": N_CELL_BASCULE, "bascule": bascule,
            "mesure_principale": "split-half" if bascule else "LOO",
            "controle": "LOO" if bascule else "split-half",
            "majorant_du_biais_residuel": (None if n <= 2 else 1.0 / (n - 2)),
            "statut": "pré-enregistrée (§14-2) — jamais un choix après lecture"}


def mesure(nom_modele: str, couche: int, cfg: EngramConfig, R: int,
           seed_dir: int, banc_e, provenance_ok: bool,
           conditions=CENTRAGES, b_boot: int = B_BOOT) -> dict:
    """Un passage complet sur un modèle. **Barrières structurelles d'abord** :
    l'étage n'est pas « non appelé », il est **fermé** tant que les portes de
    provenance et le banc ne l'ouvrent pas (§6.A, §6.E)."""
    if not provenance_ok:
        raise ArretDeProvenance(
            "§6.A : V-cache/V-G n'ont pas rendu PASS — l'étage de mesure reste "
            "fermé, aucune analyse n'est conduite")
    if banc_e != 0:
        raise BancNonFranchi(
            f"§6.E : E = {banc_e} au banc D14-S — aucune mesure avant PASS "
            f"intégral du banc")
    mat, H, cellules = etats_decisionnels(nom_modele, couche)
    dec, tige_de, S = _tiges_et_S(mat)
    n_dec, d = H[cellules[0]].shape
    mem = FastWeightMemory(d, cfg, device="cpu")     # M jamais lue ni écrite
    G = mem.G.numpy().astype(np.float64)
    del mem
    Z = {c: H[c] @ G.T for c in cellules}            # cache `Z` fp64
    ctx_global = {
        "n_global": n_dec * len(cellules),
        "somme_globale": sum(Z[c].sum(axis=0) for c in cellules),
        "g1": G @ np.ones(d, dtype=np.float64),
        "Gr": G @ _directions(d, R, seed_dir).T,     # (D, R), une fois
    }
    tiges = sorted({tige_de[i] for i in range(n_dec)})
    resultats = {x: {s: [] for s in STRATES} for x in conditions}
    for c in cellules:
        Zc, Hc = Z[c], H[c]
        sZ, sH = Zc.sum(axis=0), Hc.sum(axis=0)
        moy = Hc.mean(axis=1)
        for a in range(n_dec):
            for b in range(a + 1, n_dec):
                s = p4.strate(dec[a], dec[b])
                mu_z = (sZ - Zc[a] - Zc[b]) / (n_dec - 2)
                mu_h = (sH - Hc[a] - Hc[b]) / (n_dec - 2)
                ctx = dict(ctx_global, mu_type_z=mu_z, moy_a=moy[a],
                           moy_b=moy[b], norme_mu_type=float(np.linalg.norm(mu_h)))
                for x in conditions:
                    if x == "plac":
                        acc = []
                        for r in range(R):
                            ctx["Gr_colonne"] = ctx_global["Gr"][:, r]
                            za, zb = z_centre(Zc[a], Zc[b], x, ctx)
                            acc.append(quantites_de_paire(za, zb, cfg.dg_topk))
                        q = dict(acc[0])
                        q["inter"] = float(np.mean([u["inter"] for u in acc]))
                        q["cos"] = float(np.mean([u["cos"] for u in acc]))
                        q["f"] = float(np.mean([u["f"] for u in acc]))
                    else:
                        za, zb = z_centre(Zc[a], Zc[b], x, ctx)
                        q = quantites_de_paire(za, zb, cfg.dg_topk)
                    q["tiges"] = (tige_de[a], tige_de[b])
                    q["cellule"] = c
                    resultats[x][s].append(q)
    return {"modele": nom_modele, "couche": couche, "d": d, "n_dec": n_dec,
            "cellules": cellules, "R": R, "tiges": tiges,
            "n_cell": n_dec, "bascule": bascule_split_half(n_dec),
            "S": S, "appartenance_S": {str(i): tige_de[i] for i in S},
            "resultats": resultats}


# =========================================================================
#  CLI
# =========================================================================

def construire_parseur() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recouvrement des supports de topk(G·h) — "
                    "EXP-2026-08-23-recouvrement-supports")
    p.add_argument("--model", default=None, choices=list(MODELES))
    p.add_argument("--layer", type=int, default=None,
                   help="couche de capture ; à DÉFAUT seulement, COUCHE_REF du "
                        "§7 s'applique (défaut 0-141 : l'option primait "
                        "auparavant sur rien du tout). Exige --model.")
    p.add_argument("--center", default=None, choices=list(CENTRAGES))
    p.add_argument("--strate", default=None, choices=list(STRATES))
    p.add_argument("--R", type=int, default=None,
                   help="directions de placebo ; DÉRIVÉ (§4.5), plafond 300 — "
                        "une valeur donnée à la main est publiée comme telle")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(SORTIE))
    return p


def couche_effective(nom_modele: str, layer_cli) -> dict:
    """Défaut **0-141** fermé : `--layer` prime quand il est donné ; sinon
    `COUCHE_REF` du §7. La **source** est publiée — un paramètre en dur déguisé
    en option est un paramètre en dur."""
    if layer_cli is None:
        return {"couche": COUCHE_REF[nom_modele], "source": "COUCHE_REF (§7)"}
    return {"couche": int(layer_cli), "source": "CLI --layer",
            "couche_de_reference_du_protocole": COUCHE_REF[nom_modele],
            "avertissement": "hors du §7 : la portée du résultat n'est plus "
                             "celle du protocole"}


def main(argv=None) -> int:
    a = construire_parseur().parse_args(argv)
    if a.layer is not None and a.model is None:
        print("--layer exige --model : une couche unique n'a pas le même sens "
              "sur trois profondeurs différentes (12 / 32 / 28).")
        return 2
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
            print(f"  {m}\n    sha256 = {f['sha256']}\n"
                  f"    {f['taille_octets']} octets, dtype={f['dtype_etats']}, "
                  f"forme={f['forme_etats']}")
        else:
            print(f"  {m} : ABSENT — {f['chemin']}")
    (out / "V-cache.json").write_text(
        json.dumps({"porte": "V-cache", "verdict": vc, "detail": dc},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if vc != PASS:
        print("\nV-cache FAIL ⇒ re-forward autorisé (§14-4) : 53,82 s, "
              "VRAM ≤ 4,688 Gio en RÉSERVÉ. Ce script ne le déclenche pas.")
        return 2

    vg, dg = v_g(modeles, cfg)
    print(f"\nV-G (v2) : {vg}")
    for m, r in dg["modeles"].items():
        print(f"\n  {m} (couche {r['couche']}, d={r['d']})")
        print(f"    G du projet : sha256 = {r['G_du_projet']['sha256']}")
        print(f"                  {r['G_du_projet']['generateur']}, "
              f"seed {r['G_du_projet']['seed']}, forme "
              f"{r['G_du_projet']['forme']}, {r['G_du_projet']['echelle']}")
        print("    cos φ(h) fp32 : " + ", ".join(
            f"{s}={r['cos_phi_par_strate_fp32'][s]:.12f}" for s in STRATES))
        print("    cos φ(h) fp64 : " + ", ".join(
            f"{s}={r['cos_phi_par_strate_fp64'][s]:.12f}" for s in STRATES))
        print("    écart fp64−fp32 : " + ", ".join(
            f"{s}={r['ecart_fp64_moins_fp32'][s]:+.3e}" for s in STRATES))
        print(f"    écart max de précision = {r['ecart_max_de_precision']:.3e} "
              f"(tolérance {dg['tolerance_fp32_fp64']:.0e})")
        dv = r["divergence_avec_A3"]
        print("    divergence avec A3 (obligatoire) : écart cos fp32 − A3 = "
              + ", ".join(f"{s}={dv['ecart_cos_fp32_moins_A3'][s]:+.3e}"
                          for s in STRATES))
        print(f"      corrcoef(G projet, G d'A3) = "
              f"{dv['corrcoef_G_projet_vs_G_A3']:+.4f} — {dv['mention_obligatoire']}")
        print(f"    V-iid : {r['V-iid']['verdict']} "
              f"(moyenne {r['V-iid']['detail']['moyenne']:+.5f}, σ "
              f"{r['V-iid']['detail']['ecart_type']:.5f})")
    (out / "V-G.json").write_text(
        json.dumps({"porte": "V-G", "verdict": vg, "detail": dg},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if vg != PASS:
        print("\n" + dg["verdict_porte"])
        return 3

    vn = {m: config_de_normalisation(m) for m in modeles}
    v, dn = v_norm(vn)
    print(f"\nV-norm : {v}")
    for m, c in vn.items():
        print(f"  {m:<32} {c['normalisation']:<10} centre={c['centre_t_il']} "
              f"| {c['ligne_de_config']}")
    (out / "V-norm.json").write_text(
        json.dumps({"porte": "V-norm", "verdict": v, "detail": dn},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ntemps CPU total : {time.time() - t0:.1f} s")
    print("\nportes de provenance PASS. Suite gravée : banc D14-S "
          "(`.venv\\Scripts\\python eval/gate_bench.py --suite dgov`), "
          "`E = 0` exigé AVANT toute mesure (§6.E).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
