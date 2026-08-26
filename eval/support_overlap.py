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

# Les DEUX règles d'affectation du signe de permutation de `ε*`.
#
# Le §4.5 grave « permutation intra-tige des étiquettes {plac, type} » et
# `K_eff = 10` tiges ; il ne dit PAS ce que devient une paire dont les deux
# membres appartiennent à deux tiges différentes (`S1`, `S0`). Deux
# opérationnalisations sont également compatibles avec la clause gelée :
#
#   * `min`     — la paire est affectée à la tige `min(t_a, t_b)` et le signe
#                 est `s_{min}` (règle IMPLÉMENTÉE au premier tour) ;
#   * `produit` — le signe est `s_{t_a}·s_{t_b}`, un retournement indépendant
#                 par tige appliqué multiplicativement (règle ALTERNATIVE).
#
# **Aucune des deux n'est promue ici** : l'arbitrage appartient au PI. Les deux
# sont calculées sur les 12 cellules × 3 modèles et publiées côte à côte, avec
# les classes `C` que chacune produit. C'est une publication de SENSIBILITÉ,
# pas un changement de règle décisionnelle : elle ne dépend d'aucun résultat et
# ne peut que durcir la lecture (D30 alinéa 2).
EPS_REGLES = ("min", "produit")
EPS_REGLE_IMPLEMENTEE = "min"

# Split-half (§5-7, §4.8 — double mesure D26). Le protocole exige « LOO ET
# split-half publiés côte à côte » sans définir la partition. Déclaré :
# partition par PARITÉ de l'index de l'unité dans sa cellule de capture
# (60 unités, 6 par tige et contiguës ⇒ 3 par tige dans chaque moitié) ; pour
# une paire dont les DEUX membres tombent dans la même moitié `h`, `μ_type` est
# estimée sur la moitié `1−h`, qui ne contient **ni `a` ni `b`** — fuite nulle,
# par construction. Les paires **à cheval** n'ont pas de moitié disjointe : elles
# sont **exclues de la mesure split-half**, cardinal publié.
SPLIT_HALF_CONDITIONS = ("type_sh", "plac_sh", "type_loo_sub", "plac_loo_sub")

# Tolérance de la comparaison `g(p) ≥ |cos|` de `V-borne`, en **ULP relatifs**.
#
# Motif, mesuré : l'identité `|A∩B| ≥ p_sym` est EXACTE en arithmétique réelle,
# et sur les paires à `|A∩B| = 1` dont l'indice commun est la coordonnée de plus
# grande magnitude des DEUX clés, elle est atteinte **avec égalité** —
# `|cos| = g(1)` exactement. En fp64 les deux membres sont formés par des
# chemins de sommation différents (`cos` par produit sur l'intersection, `g` par
# cumul de masses triées) et l'égalité tombe du mauvais côté au dernier bit :
# 2 paires sur 669 060 au run du 2026-08-26, à **0.0** et **−1.0 ULP**.
# La tolérance ne desserre pas la borne — elle rend la comparaison fidèle à
# l'arithmétique que la borne suppose. Le nombre de paires DÉCIDÉES PAR ELLE est
# publié : sans ce compteur, une tolérance serait un masque.
TOL_ULP_BORNE = 8

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
CUM_TOLERANCE = 1.5e-4          # §16.1 — tolérance absolue gravée sur `cum`
CUM_Z_THEORIQUE = 569.6         # 64 × E[Z² | |Z| > 2.66] = 64 × 8.90
CUM_J_MAX = 20

# Les douze `p_sym` PRÉ-ENREGISTRÉS (§16.1, `lab-math`, Q-M6). Deux frontières
# sont **franches** — `gpt2 S1` à 0.04 %, `SmolLM2 S1` à 0.08 % — donc **sous le
# plancher numérique** : elles ne sont décidables par AUCUNE table théorique et
# s'écrivent en intervalle ; la version RÉALISÉE tranche. Propriété gravée :
# chaque frontière basculerait vers le `p` INFÉRIEUR, donc dans le sens
# CONSERVATEUR — aucune ne peut rendre la prédiction plus facile qu'annoncé.
P_SYM_ATTENDUS = {
    ("Qwen/Qwen2.5-1.5B", "S0"): (7, 7),
    ("Qwen/Qwen2.5-1.5B", "S2"): (8, 8),
    ("Qwen/Qwen2.5-1.5B", "S1"): (9, 9),
    ("gpt2", "S0"): (9, 9),
    ("gpt2", "S2"): (10, 10),
    ("Qwen/Qwen2.5-1.5B", "S3"): (11, 11),
    ("gpt2", "S1"): (12, 13),                       # frontière franche, 0.04 %
    ("HuggingFaceTB/SmolLM2-360M", "S2"): (13, 13),
    ("HuggingFaceTB/SmolLM2-360M", "S0"): (13, 13),
    ("gpt2", "S3"): (14, 14),
    ("HuggingFaceTB/SmolLM2-360M", "S1"): (16, 17),  # frontière franche, 0.08 %
    ("HuggingFaceTB/SmolLM2-360M", "S3"): (18, 18),
}

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
    "affectation du signe de permutation pour une paire INTER-tige": (
        "§4.5 grave « permutation intra-tige des étiquettes {plac, type} » "
        "mais une paire S1/S0 a DEUX tiges. DEUX opérationnalisations sont "
        "compatibles avec la clause : (a) `min` — signe s_{min(t_a,t_b)}, "
        "IMPLÉMENTÉE au premier tour ; (b) `produit` — signe s_{t_a}·s_{t_b}, "
        "ALTERNATIVE. Les deux sont calculées et publiées côte à côte avec "
        "leurs classes C ; AUCUNE n'est promue — l'arbitrage est PI. "
        "Propriété factuelle à lire avec elles : sous la règle `produit`, une "
        "paire INTRA-tige a s_t·s_t = +1 quel que soit le tirage — son signe "
        "ne peut pas se retourner ; la fraction de paires à signe non "
        "retournable est publiée par cellule et par règle."),
    "partition du split-half et sort des paires à cheval": (
        "§5-7 et §4.8 exigent « LOO ET split-half publiés côte à côte » sans "
        "définir la partition. Déclaré : moitiés par PARITÉ de l'index de "
        "l'unité dans sa cellule de capture ; pour une paire dont les deux "
        "membres sont dans la même moitié h, μ_type est estimée sur la moitié "
        "1−h (qui ne contient ni a ni b : fuite nulle par construction) ; les "
        "paires À CHEVAL n'ont aucune moitié disjointe et sont EXCLUES de la "
        "mesure split-half, cardinal publié. Le contrôle apparié "
        "`type_loo_sub` / `plac_loo_sub` refait le LOO sur EXACTEMENT le même "
        "sous-ensemble, pour que l'écart LOO↔split-half ne soit pas confondu "
        "avec l'écart entre deux jeux de paires."),
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

def _csv(v) -> str:
    """Champ CSV : `SANS OBJET` et l'absence restent LISIBLES et distincts
    (D23 — une quantité sans domaine ne se publie jamais `0`)."""
    if v is None:
        return "ABSENT"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v).replace(",", ";")


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

def _ndtri(p: float) -> float:
    """Inverse de la normale centrée réduite, **fp64** — algorithme AS 241 de
    Wichura (1988), précision relative ~1e−16 sur (0, 1).

    **Substitution DÉCLARÉE.** La clause du §16.1 grave *« le banc grave la
    version fp64 (`scipy.special.ndtri`) »*. `scipy` **n'est pas une dépendance
    du projet** (`requirements.txt` : torch, transformers, pytest) et l'ajouter
    pour une table de contrôle serait un changement de dépendance non audité.
    L'inverse est donc implémenté ici, et **vérifié indépendamment** par
    aller-retour à travers `math.erfc` — qui n'emprunte aucune ligne du code
    ci-dessous : `Φ(ndtri(p)) − p` est publié (`_verif_ndtri`).
    """
    import math
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r
                    + 67265.770927008700853) * r + 45921.953931549871457) * r
                  + 13731.693765509461125) * r + 1971.5909503065514427) * r
                + 133.14166789178437745) * r + 3.387132872796366608)
        den = (((((((5226.495278852545925 * r + 28729.085735721942674) * r
                    + 39307.89580009271061) * r + 21213.794301586595867) * r
                  + 5394.1960214247511077) * r + 687.1870074920579083) * r
                + 42.313330701600911252) * r + 1.0)
        return q * num / den
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r
                    + 0.24178072517745061177) * r + 1.27045825245236838258) * r
                  + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                + 4.6303378461565452959) * r + 1.42343711074968357734)
        den = (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r
                    + 0.0151986665636164571966) * r + 0.14810397642748007459) * r
                  + 0.68976733498510000455) * r + 1.6763848301838038494) * r
                + 2.05319162663775882187) * r + 1.0)
    else:
        r -= 5.0
        num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r
                    + 0.0012426609473880784386) * r + 0.026532189526576123093) * r
                  + 0.29656057182850489123) * r + 1.7848265399172913358) * r
                + 5.4637849111641143699) * r + 6.6579046435011037772)
        den = (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r
                    + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r
                  + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
                + 0.59983220655588793769) * r + 1.0)
    v = num / den
    return -v if q < 0 else v


def _verif_ndtri(js) -> dict:
    """Vérification INDÉPENDANTE de `_ndtri` : `Φ(z_j)` recalculée par
    `math.erfc`, qui ne partage aucune ligne avec l'inverse."""
    import math
    pires = 0.0
    for j in js:
        p = 1.0 - j / (2.0 * D_DG)
        z = _ndtri(p)
        phi = 0.5 * math.erfc(-z / math.sqrt(2.0))
        pires = max(pires, abs(phi - p))
    return {"residu_max_|Phi(ndtri(p))-p|": pires,
            "methode": "aller-retour par math.erfc, indépendant de _ndtri"}


def cum_recalculee(j_max: int = CUM_J_MAX, z_total: float = CUM_Z_THEORIQUE) -> dict:
    """Recalcul **fp64** de la table théorique — **c'est la GRAVURE** (§16.1).

    `t_j = Φ⁻¹(1 − j/(2D))` (deux queues), `cum(p) = Σ_{j≤p} t_j² / Z`,
    `Z = 64 × E[Z² | |Z| > 2.66] = 569.6`. La table livrée par `lab-math` est
    le **CONTRÔLE** : elle est comparée à `±1.5e−4` absolu, et tout écart
    au-dessus est un **défaut à publier**, jamais un ajustement.
    """
    js = list(range(1, int(j_max) + 1))
    t2 = np.array([_ndtri(1.0 - j / (2.0 * D_DG)) ** 2 for j in js],
                  dtype=np.float64)
    cum = np.cumsum(t2) / float(z_total)
    return {"cum": {j: float(cum[i]) for i, j in enumerate(js)},
             "t_j_carre": {j: float(t2[i]) for i, j in enumerate(js)},
             "Z": float(z_total), "j_max": int(j_max),
             "verification_de_l_inverse": _verif_ndtri(js),
             "gravure": "fp64, Φ⁻¹ = AS 241 (Wichura 1988), substitution "
                        "déclarée à scipy.special.ndtri (scipy hors "
                        "requirements.txt)"}


def charger_cum(chemin: Path | None = None) -> dict:
    """La table `cum(j)` **recalculée en fp64** (la gravure), confrontée à la
    table de contrôle de `lab-math` à **`±1.5e−4`** absolu (§16.1).

    La table livrée n'est **jamais** la source des valeurs : elle est le
    contrôle. Aucune valeur n'est interpolée ni extrapolée (règle Q-M5).
    """
    grav = cum_recalculee()
    table = dict(grav["cum"])
    det = {"cum": table, "j_max": max(table),
           "source_des_valeurs": "RECALCUL fp64 au banc (gravure)",
           "gravure": grav["gravure"],
           "verification_de_l_inverse": grav["verification_de_l_inverse"]}
    p = Path(chemin or CUM_TABLE_FICHIER)
    if p.exists():
        brut = json.loads(p.read_text(encoding="utf-8"))
        ctrl = {i + 1: float(v)
                for i, v in enumerate(brut.get("cum_theorique_Z569_6", []))}
        ecarts = {j: table[j] - ctrl[j] for j in sorted(ctrl) if j in table}
        pire = max((abs(v) for v in ecarts.values()), default=0.0)
        anc = {int(k): float(v) for k, v in brut.get("_ancres_verifieur", {}).items()}
        ecarts_anc = {j: table[j] - anc[j] for j in sorted(anc) if j in table}
        det.update({
            "controle": {"fichier": str(p), "source": brut.get("_source"),
                         "statut_du_fichier": brut.get("_statut"),
                         "tolerance": CUM_TOLERANCE,
                         "ecart_max_gravure_moins_controle": pire,
                         "conforme": pire <= CUM_TOLERANCE,
                         "ecarts": ecarts,
                         "ecart_max_vs_ancres_verifieur": max(
                             (abs(v) for v in ecarts_anc.values()), default=0.0),
                         "ecarts_vs_ancres_verifieur": ecarts_anc},
            "hybride_563_9": {
                "statut": "HYBRIDE DÉCLARÉ — profil d'ordre THÉORIQUE × masse "
                          "totale RÉALISÉE ; la vraie table réalisée est PAR "
                          "CLÉ et se calcule au banc ; cette colonne n'en est "
                          "pas un substitut",
                "non_utilisee_pour_decider": True},
            "Q-M6": "RENDUE"})
    else:
        det.update({"controle": {"fichier": str(p), "present": False},
                    "Q-M6": "NON RENDUE — table de contrôle absente"})
    js = sorted(table)
    det["contigue_depuis_1"] = js == list(range(1, len(js) + 1))
    det["concave"] = all(
        table[js[i + 1]] - table[js[i]]
        <= table[js[i]] - table[js[i - 1]] + 1e-12
        for i in range(1, len(js) - 1)) if len(js) > 2 else True
    det["croissante"] = all(table[js[i + 1]] > table[js[i]]
                            for i in range(len(js) - 1))
    det["regle"] = "aucune valeur de cum n'est jamais interpolée (Q-M5)"
    return det


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
    if c <= 0.0:            # `cum(0) = 0` : l'intersection vide est légitime
        return {"p_sym": 0, "statut": PASS, "cos": c, "cum_p": 0.0,
                "j_max_connu": max(t)}
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
    `m_ψ ≤ T_ψ(q)` et `|cos| ≤ √(m_φ·m_ψ)` ⇒ `q ≥ p_sym`. Toute violation est
    un **BUG DE MESURE**, jamais un résultat.

    Précision d'implémentation exigée par `lab-math` : **`|cos_pair|`** et la
    **MOYENNE GÉOMÉTRIQUE** des masses top-`p` des **deux clés de la paire** —
    **jamais** la table moyenne.
    """
    ta, tb = _masse_cumulee_realisee(phi_a), _masse_cumulee_realisee(phi_b)
    n = min(ta.size, tb.size)
    # `g(0) = 0` doit figurer dans la recherche : l'intersection VIDE est un
    # état légitime (`|cos| = 0` ⇒ `p_sym = 0`). Un plancher à 1 rendrait la
    # borne plus forte que vraie et fabriquerait des violations à `q = 0`.
    g = np.concatenate([[0.0], np.sqrt(ta[:n] * tb[:n])])
    c = abs(float(cos_ab))
    seuil = c - TOL_ULP_BORNE * np.spacing(max(c, 1e-300))
    return min(int(np.searchsorted(g, seuil, side="left")), n)


def p_pair_realise(phi_a, phi_b, cos_ab: float) -> dict:
    """Borne **LÂCHE**, conservée en **DESCRIPTIF** (§16-3, double mesure D26).

    `p = min{p : T(p) ≥ cos²}` sur chaque membre, `p_pair = max(p_a, p_b)`.
    C'est ce que le gel portait ; elle est publiée avec l'écart à `p_sym` pour
    laisser trace de ce que la borne serrée a gagné (facteur ~7 à `cos ≈ 0.15`).
    """
    c2 = float(cos_ab) ** 2

    s2 = c2 - TOL_ULP_BORNE * np.spacing(max(c2, 1e-300))

    def _p(v):
        cum = _masse_cumulee_realisee(v)
        n = cum.size
        cum = np.concatenate([[0.0], cum])      # `T(0) = 0`, même motif
        return min(int(np.searchsorted(cum, s2, side="left")), n)

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


CONDITIONS_DE_COUVERTURE = ("aucun", "type", "plac")


def couverture_publiee(publie: dict, modeles=MODELES, strates=STRATES,
                       regles=EPS_REGLES,
                       mesures=("LOO", "split-half"),
                       conditions=CONDITIONS_DE_COUVERTURE) -> tuple:
    """**Énumération** de la couverture exigée au tour de correction : sur
    chaque cellule, les **deux règles `ε*`** et les **deux mesures**
    (LOO, split-half) doivent être présentes. **Cardinal compté, jamais
    supposé** (famille 0-76(i) / 0-86 / 0-94 / 0-101 / 0-133 / 0-144).

    `publie` est `{"modèle|strate": {"eps": {règle: valeur}, "mesures":
    {nom: valeur}}}`.

    **Ambiguïté déclarée, non tranchée par le Builder** : la demande écrit
    « 12 cellules × 3 modèles » puis « 36 cellules ». Les cellules
    décisionnelles de `Δ*`/`ε*` sont les couples (modèle × strate) et il y en a
    **12 au total**, pas 12 par modèle. Les **36** ne se retrouvent qu'en
    comptant (modèle × strate × condition concernée), avec les trois conditions
    `aucun` / `type` / `plac`. Les **deux cardinaux sont comptés et publiés,
    avec leur unité nommée** ; la porte exige les deux. Arbitrage dû.
    """
    cles = [f"{m}|{s}" for m in modeles for s in strates]
    manques = []
    for cle in cles:
        e = (publie.get(cle) or {})
        for r in regles:
            if r not in (e.get("eps") or {}) or (e.get("eps") or {})[r] is None:
                manques.append(f"{cle} : règle ε* `{r}` absente")
        for mes in mesures:
            if mes not in (e.get("mesures") or {}) or \
                    (e.get("mesures") or {})[mes] is None:
                manques.append(f"{cle} : mesure `{mes}` absente")
    n_dec = len(cles)
    n_cond = n_dec * len(conditions)
    ok = (not manques and n_dec == len(modeles) * len(strates)
          and n_cond == len(modeles) * len(strates) * len(conditions))
    return (PASS if ok else FAIL), {
        "cardinal_cellules_decisionnelles_modele_x_strate": n_dec,
        "unite_1": "cellule décisionnelle = (modèle × strate) — 3 × 4 = 12",
        "cardinal_modele_x_strate_x_condition": n_cond,
        "unite_2": "(modèle × strate × condition concernée) — 3 × 4 × 3 = 36, "
                   "conditions `aucun`, `type`, `plac`",
        "regles_eps_exigees": list(regles), "mesures_exigees": list(mesures),
        "manques": manques[:50], "n_manques": len(manques),
        "regle": "cardinal recompté PAR ÉNUMÉRATION, jamais par affirmation",
        "ambiguite_declaree": "« 12 cellules × 3 modèles » et « 36 cellules » "
                              "ne désignent pas le même objet ; les DEUX "
                              "cardinaux sont comptés et publiés. Arbitrage dû."}


def bloc_anomalie_non_expliquee(cellules: dict, seuil_z: float = 5.0) -> dict:
    """`σ±` très au-dessous de sa nulle 0.5 sur certaines cellules :
    **consigné comme NON EXPLIQUÉ**.

    Quantité **DIAGNOSTIQUE (`V-diag`), non décisionnelle** : elle ne modifie
    aucune borne, aucun seuil, aucune classe. **Aucun mécanisme n'est proposé —
    nommer l'inconnu est le livrable.**
    """
    lignes = []
    for cle, c in sorted(cellules.items()):
        if not isinstance(c, dict):
            continue
        den = c.get("accord_denominateur")
        num = c.get("accord_numerateur")
        if not den:
            continue
        p = num / den
        sd = 0.5 / np.sqrt(den)
        z = (p - 0.5) / sd
        if z < -seuil_z:
            lignes.append({"cellule": cle, "sigma_pm": c.get("sigma_pm"),
                           "sigma_pm_flottant": float(p),
                           "accord": f"{num}/{den}",
                           "ecarts_types_sous_0.5": round(float(z), 2),
                           "sigma_pm_par_taille_d_intersection":
                               c.get("sigma_pm_par_intersection")})
    return {"statut": "NON EXPLIQUÉ",
            "nature": "DIAGNOSTIQUE (V-diag) — non décisionnelle : ne modifie "
                      "aucune borne, aucun seuil, aucune classe",
            "nulle": "σ± = 0.5 (accord de signe au hasard sur l'intersection)",
            "seuil_de_signalement": f"{seuil_z} écarts-types sous 0.5",
            "n_cellules_signalees": len(lignes), "cellules": lignes,
            "mecanisme": "AUCUN N'EST PROPOSÉ — nommer l'inconnu est le "
                         "livrable. Les descripteurs publiés (σ± par taille "
                         "d'intersection) sont là pour qu'un tour ultérieur "
                         "puisse instruire, pas pour conclure ici."}


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


def v_t1(registre_t1=None, cardinal_intra_tige=None, core_S=None,
         tige_de=None) -> tuple:
    """`V-t1` — **portée corrigée le 2026-08-26, défaut `0-143` (critique)**.

    La clause d'origine (« exclusion de toute paire intra-tige ») était
    **auto-contradictoire avec le §7 et le §4.2** : elle **vide `S3` et `S2`**,
    qui sont **définies** par le partage de tige. Son motif (0-134) était
    **doublement faux** — l'argument des états bit-identiques vaut à **`t−1`**,
    pas à `t` (à `t`, deux unités d'une même tige **diffèrent par le suffixe,
    qui EST le token de capture**) ; et « brise l'indépendance du bootstrap de
    tiges » est **à l'envers** : le bootstrap **par tige** est précisément ce
    qui **absorbe** la corrélation intra-tige — c'est tout l'objet de 0-50.

    **Trois membres :**

    (i)   aucune quantité à **`t−1`** dont l'ensemble de comparaison contient
          deux unités de même tige (`t−1` **reste refusé**, 0-130 : `A4` a
          établi que le porteur du domaine s'y évanouit, et `S3`/`S2` y
          rendraient `64/64` par arithmétique, 0-64) ;
    (ii)  `Core(S)` : **une seule unité par tige** (`V-core-S`) ;
    (iii) **à `t`, les paires intra-tige ne sont PAS exclues** ; le cardinal
          intra-tige est **publié** par strate et par modèle.
    """
    reg = list(registre_t1 or [])
    fautes = [q for q in reg if q.get("intra_tige")]
    m1 = not fautes and not reg
    if core_S is None or tige_de is None:
        m2, det2 = True, SANS_OBJET
    else:
        tg = [tige_de[u] for u in core_S]
        m2 = len(set(tg)) == len(tg)
        det2 = {"n_unites": len(tg), "n_tiges": len(set(tg))}
    m3 = cardinal_intra_tige is not None
    return (PASS if (m1 and m2 and m3) else FAIL), {
        "locus_du_cycle": "t", "t_moins_1": "REFUSÉ (0-130, ratifié PI)",
        "membre_i_aucune_quantite_a_t_moins_1": m1,
        "quantites_a_t_moins_1": reg, "fautes_intra_tige": fautes,
        "membre_ii_Core_une_unite_par_tige": m2, "detail_Core": det2,
        "membre_iii_cardinal_intra_tige_publie": m3,
        "cardinal_intra_tige": (SANS_OBJET if cardinal_intra_tige is None
                                else cardinal_intra_tige),
        "portee_corrigee": "0-143 — à `t`, les paires intra-tige ne sont PAS "
                           "exclues ; S3 et S2 sont mesurées"}


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


# =========================================================================
#  L1 / L2 / L3 — TOUR ADDITIF du 2026-08-26
#
#  **Rien de neuf n'est MESURÉ ici.** Les trois livrables sont soit des
#  colonnes DÉJÀ CALCULÉES et non tabulées (`V-diag`, l. 383 du protocole),
#  soit des **ré-expressions arithmétiques** de nombres déjà publiés (`p_sym`
#  au niveau cellule, décomposition `Δ* = Δ_rem + Δ_add`).
#
#  **Statut : DESCRIPTIF.** Aucune classe, aucun seuil, aucune borne, aucune
#  clause gelée n'est touchée. **`Δ*` reste la primaire gelée** ; `Δ_rem` et
#  `Δ_add` ne décident de rien.
# =========================================================================

# Les colonnes que `V-diag` (l. 383) exige **par strate, par condition, par
# modèle** : `f`, `σ±`, le couple `(O, cos)` conjoint — plus `p_sym` au niveau
# cellule et l'excès `O − p_sym`, ajoutés par ce tour.
COLONNES_V_DIAG = ("O_en_indices", "cos_moyen", "f", "sigma_pm",
                   "p_sym_cellule", "exces_O_moins_p_sym")


def p_sym_cellule(cellule: dict, table: dict | None = None) -> dict:
    """`p_sym` **au niveau CELLULE** et l'excès `O − p_sym`, **en indices**.

    **Ambiguïté DÉCLARÉE, non tranchée par le Builder** — la demande écrit
    « la même définition qu'au §16 (`min{p : cum(p) ≥ cos}` appliqué au `cos`
    de la cellule) », mais les **valeurs de contrôle** qu'elle fournit
    (excès ≈ 2.4 ± 0.4 sur 12/12 sous `aucun`, `gpt2` 2.65/2.55/2.93/2.43)
    sont celles de la **MÉDIANE PAR PAIRE des `p_sym` RÉALISÉS**, pas celles
    de la table théorique appliquée au `cos` moyen. Ce sont **deux objets
    différents** : le premier agrège des bornes calculées sur les `φ_i²`
    observés de chaque paire (§16-1, la version **décisionnelle** que porte
    `V-borne`) ; le second applique la table `cum` **théorique** (§16.1) à une
    moyenne de cosinus, ce qui n'est **pas** la borne d'une paire.

    **Les DEUX sont calculés et publiés côte à côte, avec leur définition
    nommée ; AUCUNE n'est promue. Arbitrage dû.** La lecture retenue pour
    laisser le livrable exécutable et comparable au contrôle fourni est la
    **réalisée médiane** ; la théorique est publiée à côté.

    `SANS OBJET` (jamais `0`) quand la cellule n'a pas de domaine (D23).
    """
    if not isinstance(cellule, dict) or not cellule.get("P"):
        return {"statut": SANS_OBJET,
                "raison": "cellule sans paire — support vide (D23)"}
    o_ind = float(cellule["O_en_indices"])
    pm = cellule.get("p_sym_median")
    th = p_sym_theorique(float(cellule["cos_moyen"]), table)
    pth = th.get("p_sym")
    return {
        "O_en_indices": o_ind,
        "cos_de_la_cellule": float(cellule["cos_moyen"]),
        # --- lecture 1 : RÉALISÉE, médiane des `p_sym` par paire (§16-1)
        "p_sym_realise_median": (None if pm is None else int(pm)),
        "exces_O_moins_p_sym_realise_median":
            (None if pm is None else o_ind - int(pm)),
        # --- lecture 2 : THÉORIQUE, `min{p : cum(p) ≥ cos_cellule}` (§16.1)
        "p_sym_theorique_du_cos_de_cellule": pth,
        "statut_p_sym_theorique": th.get("statut"),
        "cum_p_theorique": th.get("cum_p"),
        "exces_O_moins_p_sym_theorique":
            (None if pth is None else o_ind - int(pth)),
        "raison_theorique_indisponible": th.get("raison"),
        "unite": "indices sur 64",
        "statut": "DESCRIPTIF — ne modifie aucune classe, aucun seuil, "
                  "aucune borne (L2, tour additif)",
        "definitions": {
            "p_sym_realise_median": "médiane, sur les paires de la cellule, "
                                    "de p_sym = min{p : √(T_φ(p)·T_ψ(p)) ≥ "
                                    "|cos_paire|} calculé sur les φ_i² "
                                    "OBSERVÉS des deux clés (§16-1)",
            "p_sym_theorique_du_cos_de_cellule":
                "min{p : cum(p) ≥ cos_moyen de la cellule}, table cum "
                "théorique du §16.1 — aucune interpolation (Q-M5)"},
        "ambiguite_declaree": "les deux lectures ne désignent pas le même "
                              "objet ; les DEUX sont publiées, AUCUNE n'est "
                              "promue. Arbitrage dû."}


def v_diag_colonnes(cellules: dict, modeles=MODELES, conditions=CENTRAGES,
                    strates=STRATES, colonnes=COLONNES_V_DIAG) -> tuple:
    """**Couverture ÉNUMÉRÉE** des cellules × colonnes de `V-diag` (l. 383).

    `V-diag` exige `f`, `σ±` et le couple `(O, cos)` conjoint **par strate,
    par condition, par modèle** ; cette porte **compte** le cardinal —
    `3 modèles × 5 conditions × 4 strates = 60 cellules`, chacune × les
    colonnes exigées — **jamais elle ne le suppose** (famille 0-76(i) / 0-86 /
    0-94 / 0-101 / 0-133 / 0-144).

    **Lecture déclarée** : `SANS OBJET` est une **valeur publiée**, pas une
    absence (D23 — « une quantité sans domaine de définition se publie
    `SANS OBJET`, jamais `0` »). Elle est comptée à part, jamais comme un
    manque. Seuls la clé absente et `None` sont des manques.
    """
    manques, vues, sans_objet = [], [], []
    for m in modeles:
        for x in conditions:
            for s in strates:
                cle = f"{m}|{x}|{s}"
                cel = cellules.get(cle)
                if cel is None:
                    manques.append(f"{cle} : cellule absente")
                    continue
                vues.append(cle)
                for col in colonnes:
                    if col in ("p_sym_cellule", "exces_O_moins_p_sym"):
                        b = cel.get("p_sym_cellule")
                        champ = ("p_sym_realise_median"
                                 if col == "p_sym_cellule"
                                 else "exces_O_moins_p_sym_realise_median")
                        if not isinstance(b, dict):
                            v = b
                        elif b.get("statut") == SANS_OBJET:
                            v = SANS_OBJET      # support vide publié (D23)
                        else:
                            v = b.get(champ)
                    else:
                        v = cel.get(col, None)
                    if v == SANS_OBJET:
                        sans_objet.append(f"{cle} : `{col}`")
                    elif v is None:
                        manques.append(f"{cle} : colonne `{col}` absente")
    n_cel = len(set(vues))
    n_attendu = len(modeles) * len(conditions) * len(strates)
    n_paires_cel_col = n_cel * len(colonnes)
    ok = (not manques and n_cel == n_attendu)
    return (PASS if ok else FAIL), {
        "cardinal_cellules_enumere": n_cel,
        "cardinal_cellules_attendu": n_attendu,
        "decomposition_du_cardinal": (f"{len(modeles)} modèles × "
                                      f"{len(conditions)} conditions × "
                                      f"{len(strates)} strates"),
        "colonnes_exigees": list(colonnes),
        "cardinal_cellule_x_colonne": n_paires_cel_col,
        "cardinal_cellule_x_colonne_attendu": n_attendu * len(colonnes),
        "n_sans_objet_publie": len(sans_objet),
        "sans_objet_publie": sans_objet[:50],
        "manques": manques[:50], "n_manques": len(manques),
        "regle": "cardinal recompté PAR ÉNUMÉRATION, jamais par affirmation ; "
                 "SANS OBJET est une valeur publiée, pas un manque (D23)",
        "statut": "DIAGNOSTIQUE, jamais correctif (Arbitrage X-1)"}


def v_psym_conditions(cellules: dict, modeles=MODELES, conditions=CENTRAGES,
                      strates=STRATES) -> tuple:
    """`p_sym` **défini sur les CINQ conditions du §8** — cardinal **compté**.

    La porte échoue si une seule des `len(conditions)` conditions n'a pas son
    `p_sym` de cellule sur les `len(modeles) × len(strates)` cellules. Le
    cardinal par condition est **publié**, jamais supposé.
    """
    par_condition, manques = {}, []
    for x in conditions:
        n = 0
        for m in modeles:
            for s in strates:
                b = (cellules.get(f"{m}|{x}|{s}") or {}).get("p_sym_cellule")
                if not isinstance(b, dict) or \
                        b.get("p_sym_realise_median") is None:
                    manques.append(f"{m}|{x}|{s} : p_sym de cellule absent")
                else:
                    n += 1
        par_condition[x] = n
    attendu = len(modeles) * len(strates)
    n_conditions_couvertes = sum(1 for n in par_condition.values()
                                 if n == attendu)
    ok = (not manques and n_conditions_couvertes == len(conditions)
          and len(par_condition) == len(conditions))
    return (PASS if ok else FAIL), {
        "conditions_exigees": list(conditions),
        "cardinal_conditions_exige": len(conditions),
        "cardinal_conditions_couvertes_enumere": n_conditions_couvertes,
        "cardinal_par_condition": par_condition,
        "cardinal_par_condition_attendu": attendu,
        "unite": "cellules (modèle × strate) portant un p_sym de cellule",
        "manques": manques[:50], "n_manques": len(manques),
        "regle": "cardinal COMPTÉ, jamais supposé (0-144)"}


def identite_decomposition(d_rem: float, d_add: float, d_star: float,
                           echelle: float | None = None,
                           tol_ulp: int = TOL_ULP_BORNE) -> tuple:
    """`Δ_rem + Δ_add = Δ*` — **identité arithmétique**, vérifiée à la
    tolérance ULP **déjà en vigueur** (`TOL_ULP_BORNE`, l. 137).

    `Δ_rem = O_aucun − O_type`, `Δ_add = O_plac − O_aucun` : leur somme est
    `O_plac − O_type = Δ*` **exactement en arithmétique réelle**. En fp64 les
    deux membres sont formés par des chemins de sommation différents ;
    l'écart admis est celui, déjà gravé, de `V-borne`. **Tout écart au-delà
    est un BUG, jamais un résultat.**
    """
    e = abs((float(d_rem) + float(d_add)) - float(d_star))
    ech = float(np.spacing(max(abs(float(echelle if echelle is not None
                                       else d_star)), 1e-300)))
    n_ulp = e / ech if ech > 0 else float("inf")
    ok = n_ulp <= tol_ulp
    return (PASS if ok else FAIL), {
        "Delta_rem": float(d_rem), "Delta_add": float(d_add),
        "Delta_etoile": float(d_star),
        "somme_Delta_rem_plus_Delta_add": float(d_rem) + float(d_add),
        "ecart_absolu": e, "echelle_ULP": ech, "ecart_en_ULP": n_ulp,
        "tolerance_ULP": tol_ulp,
        "regle": "identité EXACTE en arithmétique réelle ; tout écart "
                 "au-delà de la tolérance ULP déjà en vigueur est un BUG"}


def decomposition_delta(vals_aucun, vals_type, vals_plac, tiges_de_paire,
                        tiges, b: int = B_BOOT, seed: int = 0,
                        tol_ulp: int = TOL_ULP_BORNE) -> dict:
    """**DESCRIPTIF (L3)** — `Δ* = Δ_rem + Δ_add`, recalculé **depuis les
    accumulateurs par paire**, avec IC bootstrap **de tiges**.

    - `Δ_rem = O_aucun − O_type` — le cran **RETIRÉ** par le centrage par type.
    - `Δ_add = O_plac − O_aucun` — le cran **AJOUTÉ** par la construction du
      placebo, le §8 **gelé** soustrayant la **même** `c·r` aux deux membres.

    **`Δ*` reste la primaire gelée.** `Δ_rem` et `Δ_add` sont **DESCRIPTIFS** :
    ils ne modifient **aucune classe, aucun seuil, aucune borne**.

    L'identité est vérifiée **par paire** (écart max en ULP) **et** sur les
    estimés agrégés.
    """
    a = np.asarray(vals_aucun, dtype=np.float64)
    t = np.asarray(vals_type, dtype=np.float64)
    p = np.asarray(vals_plac, dtype=np.float64)
    if not (a.shape == t.shape == p.shape):
        raise ValueError("décomposition : les trois conditions doivent porter "
                         "sur EXACTEMENT le même jeu de paires, dans le même "
                         "ordre")
    rem, add, star = a - t, p - a, p - t
    ic_rem = bootstrap_tiges(rem, tiges_de_paire, tiges, b=b, seed=seed)
    ic_add = bootstrap_tiges(add, tiges_de_paire, tiges, b=b, seed=seed)
    ic_star = bootstrap_tiges(star, tiges_de_paire, tiges, b=b, seed=seed)
    ecart_paire = np.abs((rem + add) - star)
    ech = np.spacing(np.maximum(np.maximum(np.abs(a), np.abs(t)),
                                np.maximum(np.abs(p), 1e-300)))
    ulp_paire = float(np.max(ecart_paire / ech)) if a.size else 0.0
    v_id, d_id = identite_decomposition(ic_rem["estime"], ic_add["estime"],
                                        ic_star["estime"],
                                        echelle=max(abs(float(ic_rem["estime"])),
                                                    abs(float(ic_add["estime"])),
                                                    abs(float(ic_star["estime"])),
                                                    1e-300),
                                        tol_ulp=tol_ulp)
    return {
        "P": int(a.size),
        "Delta_rem_estime": ic_rem["estime"], "IC_Delta_rem": ic_rem["IC"],
        "Delta_add_estime": ic_add["estime"], "IC_Delta_add": ic_add["IC"],
        "Delta_etoile_estime": ic_star["estime"],
        "IC_Delta_etoile": ic_star["IC"],
        "Delta_rem_en_indices": float(ic_rem["estime"]) * 64,
        "Delta_add_en_indices": float(ic_add["estime"]) * 64,
        "Delta_etoile_en_indices": float(ic_star["estime"]) * 64,
        "K_eff": ic_rem["K_eff"],
        "identite_par_paire": {
            "verdict": PASS if ulp_paire <= tol_ulp else FAIL,
            "ecart_max_en_ULP": ulp_paire, "tolerance_ULP": tol_ulp,
            "n_paires": int(a.size)},
        "identite_sur_les_estimes": {"verdict": v_id, "detail": d_id},
        "definitions": {
            "Delta_rem": "O_aucun − O_type — le cran RETIRÉ",
            "Delta_add": "O_plac − O_aucun — le cran AJOUTÉ par la "
                         "construction du placebo (§8 gelé : la MÊME c·r est "
                         "soustraite aux deux membres)",
            "Delta_etoile": "O_plac − O_type — PRIMAIRE GELÉE"},
        "statut": "DESCRIPTIF — Δ_rem et Δ_add ne modifient aucune classe, "
                  "aucun seuil, aucune borne ; Δ* reste la primaire gelée",
        "unite": "fraction p/64 ; la colonne `_en_indices` multiplie par 64"}


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


def _encadrement_theorique(encadrement) -> tuple:
    """Membre (b) de `V-borne` : la version **RÉALISÉE encadre la THÉORIQUE**
    (D26). `encadrement` est `{(modèle, strate): p_sym_réalisé_médian}` ; la
    prédiction pré-enregistrée est `P_SYM_ATTENDUS` (§16.1), les deux
    frontières franches étant des **intervalles**.

    `None` ⇒ **`EN-ATTENTE`** (la mesure n'a pas encore produit la réalisée) —
    jamais un `PASS` par défaut.
    """
    if encadrement is None:
        return EN_ATTENTE, {"statut": EN_ATTENTE,
                            "raison": "version réalisée non encore mesurée"}
    lignes, hors, hors_b = {}, [], []
    for cle, (lo, hi) in P_SYM_ATTENDUS.items():
        obs = encadrement.get(cle)
        if obs is None:
            hors.append({"cellule": list(cle), "observe": None,
                         "motif": "cellule absente de la mesure"})
            continue
        if isinstance(obs, dict):
            med, q1, q3 = obs["median"], obs["q1"], obs["q3"]
            mn, mx = obs["min"], obs["max"]
        else:
            med = q1 = q3 = mn = mx = int(obs)
        # Lecture A — « la réalisée (médiane) tombe dans la théorique »
        a_ok = lo <= med <= hi
        # Lecture B — LITTÉRALE : « la RÉALISÉE ENCADRE la THÉORIQUE », donc
        # l'intervalle théorique rencontre l'IQR réalisé
        b_ok = not (hi < q1 or lo > q3)
        # Lecture C — la plus FORTE de B : contenance stricte de l'intervalle
        # théorique dans l'IQR réalisé (`q1 ≤ p_théo ≤ q3` aux deux bornes).
        c_ok = (q1 <= lo) and (hi <= q3)
        lignes[f"{cle[0]}|{cle[1]}"] = {
            "p_sym_theorique": [lo, hi],
            "p_sym_realise": {"min": int(mn), "q1": int(q1), "median": int(med),
                              "q3": int(q3), "max": int(mx)},
            "lecture_A_mediane_dans_theorique": bool(a_ok),
            "lecture_B_IQR_realise_rencontre_theorique": bool(b_ok),
            "lecture_C_contenance_stricte_q1_lo_hi_q3": bool(c_ok),
            "ecart_mediane_au_plus_proche":
                0 if a_ok else int(min(abs(med - lo), abs(med - hi)))}
        if not a_ok:
            hors.append({"cellule": list(cle), "theorique": [lo, hi],
                         "realise_median": int(med)})
        if not b_ok:
            hors_b.append({"cellule": list(cle), "theorique": [lo, hi],
                           "IQR_realise": [int(q1), int(q3)]})
    n_tot = len(lignes)
    n_a = sum(1 for v in lignes.values()
              if v["lecture_A_mediane_dans_theorique"])
    n_b = sum(1 for v in lignes.values()
              if v["lecture_B_IQR_realise_rencontre_theorique"])
    n_c = sum(1 for v in lignes.values()
              if v["lecture_C_contenance_stricte_q1_lo_hi_q3"])
    return (PASS if not hors_b else FAIL), {
        "statut": PASS if not hors_b else FAIL, "par_cellule": lignes,
        "trois_lectures": {
            "A_mediane_realisee_dans_l_intervalle_theorique": f"{n_a}/{n_tot}",
            "B_IQR_realise_rencontre_l_intervalle_theorique": f"{n_b}/{n_tot}",
            "C_contenance_stricte_q1_<=_p_theo_<=_q3": f"{n_c}/{n_tot}",
            "note": "l'ambiguïté d'« encadrer » a été TRANCHÉE : « X encadre "
                    "Y » signifie Y contenu dans X, ce qui désigne la lecture "
                    "B (rôles syntaxiques respectés). La lecture C est la plus "
                    "FORTE de B. Les trois sont publiées côte à côte ; la "
                    "divergence entre elles est SANS EFFET sur le verdict, qui "
                    "reste porté par B."},
        "lecture_retenue": "B — littérale : « la version RÉALISÉE doit "
                           "ENCADRER la table théorique » (§16-1), donc "
                           "l'intervalle théorique est contenu dans l'IQR "
                           "réalisé (au sens : le rencontre)",
        "ambiguite_declaree": "trois lectures licites, publiées côte à côte "
                              "(A, B, C). La grammaire de la clause désigne B ; "
                              "C en est la version la plus forte. Divergence "
                              "SANS EFFET sur le verdict.",
        "cellules_hors_encadrement_lecture_B": hors_b, "n_hors_B": len(hors_b),
        "cellules_hors_encadrement_lecture_A": hors, "n_hors_A": len(hors),
        "frontieres_franches": {
            "gpt2|S1": "12-13 (0.04 %)", "HuggingFaceTB/SmolLM2-360M|S1":
                "16-17 (0.08 %)",
            "propriete_gravee": "chaque frontière basculerait vers le p "
                                "INFÉRIEUR, donc dans le sens CONSERVATEUR ; "
                                "aucune ne peut rendre la prédiction plus "
                                "facile qu'annoncé"}}


def v_borne(paires, encadrement=None) -> tuple:
    """`V-borne` — version **RÉALISÉE par paire** (0-128, candidate D30).

    Deux membres :

    * **(a) identité arithmétique** — `|A∩B| ≥ p_pair` sur **100 %** des
      paires. Toute violation est un **BUG DE MESURE**, jamais un résultat
      (§6.B). Ce membre est exécutable **maintenant** ;
    * **(b) encadrement** — la version **réalisée encadre la THÉORIQUE**
      (§16.1, douze `p_sym` pré-enregistrés). `EN-ATTENTE` tant que la mesure
      n'a pas produit la réalisée ; jamais un `PASS` par défaut.

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
    membre_b, det_b = _encadrement_theorique(encadrement)
    # Un FAIL de l'un OU l'autre membre est un FAIL : `EN-ATTENTE` ne dit que
    # « pas encore décidable », il ne doit jamais absorber un échec constaté.
    if membre_a == FAIL or membre_b == FAIL:
        verdict = FAIL
    elif membre_b == EN_ATTENTE:
        verdict = EN_ATTENTE
    else:
        verdict = PASS
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


def _indices_de_tige(tiges_de_paire, tiges):
    rang = {t: i for i, t in enumerate(tiges)}
    ta = np.array([rang[t[0]] for t in tiges_de_paire], dtype=np.int64)
    tb = np.array([rang[t[1]] for t in tiges_de_paire], dtype=np.int64)
    return ta, tb


def _poids_vect(ta, tb, mult):
    """Version vectorisée de `poids_bootstrap` — **même règle déclarée**.
    Le banc vérifie l'égalité des deux sur des entrées aléatoires : une
    version rapide qui diverge de la version lue est un défaut."""
    ma, mb = mult[ta], mult[tb]
    return np.where(ta == tb, ma, ma * mb).astype(np.float64)


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
    ta, tb = _indices_de_tige(tiges_de_paire, tiges)
    ech = np.empty(b, dtype=np.float64)
    for r in range(b):
        mult = np.bincount(rng.integers(0, K, K), minlength=K)
        w = _poids_vect(ta, tb, mult)
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


def _signes_de_paire(regle: str, signes, affect, ta, tb):
    """Signe de permutation appliqué à chaque paire, selon la règle.

    * `min`     — `s_{min(t_a, t_b)}` : le signe est constant DANS une tige,
                  la paire est affectée à `min(t_a, t_b)` ;
    * `produit` — `s_{t_a}·s_{t_b}` : un retournement indépendant par tige,
                  appliqué multiplicativement.

    Les deux respectent la clause gelée « permutation intra-tige des étiquettes
    {plac, type} » ; le protocole ne dit pas laquelle. **Aucune n'est promue.**
    """
    if regle == "min":
        return signes[affect]
    if regle == "produit":
        return signes[ta] * signes[tb]
    raise ValueError(f"règle de signe inconnue : {regle}")


def fraction_signe_non_retournable(regle: str, tiges_de_paire) -> float:
    """Fraction des paires dont le signe **ne peut pas** se retourner sous la
    règle donnée — quantité factuelle, publiée avec `ε*`.

    Sous `produit`, une paire intra-tige donne `s_t·s_t = +1` **quel que soit
    le tirage** ; sous `min`, aucune paire n'est dans ce cas.
    """
    tp = list(tiges_de_paire)
    if not tp:
        return 0.0
    if regle == "produit":
        return float(sum(1 for x, y in tp if x == y) / len(tp))
    return 0.0


def plancher_epsilon(tiges_de_paire, tiges, b: int = B_PERM, seed: int = 0,
                     regle: str = EPS_REGLE_IMPLEMENTEE) -> dict:
    """**Plancher structurel** du test de centrage : la valeur MINIMALE que
    `q₀.₉₅` peut atteindre avec `K_eff = 10` unités de signe **quand l'effet est
    constant** — c'est-à-dire la **puissance maximale disponible du test**,
    indépendamment de la donnée.

    Calculé par simulation avec `Δ*_i ≡ 1` : la statistique de permutation vaut
    alors la moyenne pondérée des signes, et `q₀.₉₅` de sa valeur absolue est le
    plancher **relatif à `|Δ*|`**. Le plancher **absolu** d'une cellule vaut
    `plancher_relatif × |Δ*_observé|`.

    Quand `plancher_relatif ≥ 1`, `IC_inf(Δ*) > ε*` est **inatteignable** sur
    cette cellule quel que soit l'effet : la classe informative de la partition
    `C` y est vide **par construction du test**, et cela se dit (mode de vacuité
    0-47 / 0-66). Aucune interprétation n'est attachée ici.
    """
    tp = list(tiges_de_paire)
    if not tp:
        return {"plancher_relatif": None, "statut": SANS_OBJET,
                "raison": "aucune paire"}
    r = eps_etoile(np.ones(len(tp), dtype=np.float64), tp, tiges, b=b,
                   seed=seed, regle=regle)
    return {"plancher_relatif": r["epsilon_etoile"], "regle": regle, "B": b,
            "seed": seed, "K_eff": r["K_eff"], "n_paires": len(tp),
            "fraction_signe_non_retournable":
                r["fraction_de_paires_a_signe_non_retournable"],
            "definition": "q₀.₉₅ de |Δ*| simulé sous effet CONSTANT = 1 ; "
                          "c'est la puissance maximale du test, indépendante "
                          "de la donnée",
            "unite": "relatif à |Δ*| — le plancher absolu vaut "
                     "plancher_relatif × |Δ*_observé|"}


def moities_de_cellule(n: int) -> np.ndarray:
    """Partition split-half **déterministe** d'une cellule de capture : parité
    de l'index de l'unité (opérationnalisation déclarée)."""
    return np.arange(int(n)) % 2


def indices_estimation_split_half(moitie, a: int, b: int):
    """Indices sur lesquels `μ_type` est estimée en split-half pour la paire
    `(a, b)` — **la moitié qui ne contient ni `a` ni `b`**.

    Rend `None` quand la paire est **à cheval** : il n'existe alors aucune
    moitié disjointe de la paire, et la paire est **exclue** de la mesure
    split-half (cardinal publié). Fuite nulle par construction : `a` et `b`
    n'appartiennent jamais à l'ensemble rendu.
    """
    m = np.asarray(moitie)
    if int(m[a]) != int(m[b]):
        return None
    return np.flatnonzero(m == (1 - int(m[a])))


def eps_etoile(delta_par_paire, tiges_de_paire, tiges, b: int = B_PERM,
               seed: int = 0, regle: str = EPS_REGLE_IMPLEMENTEE) -> dict:
    """**LIGNE CANONIQUE UNIQUE** de `ε*` — recopiée à l'identique du §4.5 et du
    §7, une seule fois dans le code (anti-0-82) :

    > `ε*` = q₀.₉₅ de `|Δ*|` recalculée sur `B = 10⁴` rééchantillons bootstrap
    > de tiges (`K_eff = 10`) sous permutation intra-tige des étiquettes
    > {plac, type}, le placebo étant la moyenne des `R` directions gelées ; une
    > valeur par modèle et par strate, calculée et gelée **AVANT** lecture des
    > `Δ*` observés.

    **Opérationnalisation du signe pour une paire INTER-tige : NON
    PRÉ-ENREGISTRÉE.** Le §4.5 ne dit pas ce que devient une paire à deux tiges.
    Deux règles également compatibles sont implémentées et **publiées côte à
    côte** — `min` (implémentée au premier tour) et `produit` (alternative) ;
    **aucune n'est promue**, l'arbitrage est PI (`EPS_REGLES`).
    """
    if regle not in EPS_REGLES:
        raise ValueError(f"règle de signe inconnue : {regle}")
    d0 = np.asarray(delta_par_paire, dtype=np.float64)
    if d0.size == 0:
        return {"epsilon_etoile": None, "statut": SANS_OBJET,
                "raison": "aucune paire", "regle": regle}
    tiges = list(tiges)
    K = len(tiges)
    rang = {t: i for i, t in enumerate(tiges)}
    affect = np.array([rang[min(x, y)] for x, y in tiges_de_paire])
    ta, tb = _indices_de_tige(tiges_de_paire, tiges)
    rng = np.random.default_rng(seed)
    stat = np.empty(b, dtype=np.float64)
    for r in range(b):
        mult = np.bincount(rng.integers(0, K, K), minlength=K)
        w = _poids_vect(ta, tb, mult)
        signes = rng.integers(0, 2, K) * 2 - 1          # ±1 par TIGE
        sw = w.sum()
        stat[r] = (float(np.dot(w, d0 * _signes_de_paire(regle, signes, affect,
                                                         ta, tb)) / sw)
                   if sw > 0 else np.nan)
    stat = stat[np.isfinite(stat)]
    return {"epsilon_etoile": float(np.quantile(np.abs(stat), Q_ENVELOPPE)),
            "sigma_delta": float(d0.std(ddof=1)) if d0.size > 1 else 0.0,
            "B": b, "seed": seed, "K_eff": K, "n_paires": int(d0.size),
            "regle": regle,
            "statut_de_la_regle": ("IMPLÉMENTÉE (opérationnalisation déclarée, "
                                   "non pré-enregistrée)"
                                   if regle == EPS_REGLE_IMPLEMENTEE
                                   else "ALTERNATIVE (publiée, NON promue)"),
            "fraction_de_paires_a_signe_non_retournable":
                fraction_signe_non_retournable(regle, tiges_de_paire),
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
    n = len(sup)
    # **Distribution ADJACENTE des comptes**, obligatoire : un `|Core| = 0`
    # publié en zéro plat masque qu'un indice à 8/10 annule déjà la nulle
    # `Bin(10, 1/128)`. Le seuil ≥ 9 ne bouge pas ; seule la publication
    # s'enrichit. Comptes bruts, aucune décision n'en dépend.
    dist = {f">={j}": int((cnt >= j).sum()) for j in range(1, n + 1)}
    return {"Core": sorted(int(i) for i in idx), "taille": int(idx.size),
            "n_unites": n, "seuil": seuil,
            "distribution_adjacente": dist,
            "compte_max_sur_S": int(cnt.max()) if cnt.size else 0,
            "statut_de_la_distribution": "DESCRIPTIVE — le seuil ≥ 9 reste "
                                         "pré-enregistré et inchangé ; la "
                                         "distribution est publiée pour qu'un "
                                         "|Core| = 0 ne se lise pas en zéro plat",
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


def _topk_batch(Z, k: int):
    """`topk(|z|)` par ligne, **exactement `k` indices**, fp64 — version
    batchée du noyau `support_topk`. Le banc vérifie l'**équivalence** des deux
    sur des entrées aléatoires : une version rapide qui diverge de la version
    lue est un défaut, pas une optimisation."""
    a = np.abs(Z)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    return np.sort(idx, axis=1)


def quantites_batch(Za, Zb, k: int = K_TOPK, avec_marge: bool = False) -> dict:
    """Toutes les quantités d'un bloc de paires, en fp64.

    Rend des **tableaux** : `inter` (entiers), `cos`, `f`, accord de signe
    (numérateur entier), `p_sym` (décisionnel) et `p_pair` (descriptif).
    """
    Za = np.asarray(Za, dtype=np.float64)
    Zb = np.asarray(Zb, dtype=np.float64)
    B, D = Za.shape
    A, Bi = _topk_batch(Za, k), _topk_batch(Zb, k)
    lig = np.arange(B)[:, None]
    mA = np.zeros((B, D), dtype=bool)
    mB = np.zeros((B, D), dtype=bool)
    mA[lig, A] = True
    mB[lig, Bi] = True
    mI = mA & mB
    inter = mI.sum(axis=1).astype(np.int64)
    na2 = np.where(mA, Za * Za, 0.0).sum(axis=1)
    nb2 = np.where(mB, Zb * Zb, 0.0).sum(axis=1)
    na, nb = np.sqrt(na2), np.sqrt(nb2)
    num = np.where(mI, Za * Zb, 0.0).sum(axis=1)
    cos = num / np.maximum(na * nb, np.finfo(np.float64).tiny)
    fa = np.where(mI, Za * Za, 0.0).sum(axis=1) / np.maximum(na2, 1e-300)
    fb = np.where(mI, Zb * Zb, 0.0).sum(axis=1) / np.maximum(nb2, 1e-300)
    f = 0.5 * (fa + fb)
    acc = (mI & (np.sign(Za) == np.sign(Zb))).sum(axis=1).astype(np.int64)
    # masses top-p réalisées, PAR CLÉ, puis moyenne géométrique (lab-math)
    va = np.take_along_axis(Za, A, axis=1) ** 2 / np.maximum(na2, 1e-300)[:, None]
    vb = np.take_along_axis(Zb, Bi, axis=1) ** 2 / np.maximum(nb2, 1e-300)[:, None]
    Ta = np.cumsum(np.sort(va, axis=1)[:, ::-1], axis=1)
    Tb = np.cumsum(np.sort(vb, axis=1)[:, ::-1], axis=1)
    # `g(0) = T(0) = 0` inclus : l'intersection VIDE est un état légitime, et
    # un plancher à `p = 1` rendrait la borne plus forte que vraie (violations
    # fabriquées à `q = 0`).
    z0 = np.zeros((Za.shape[0], 1), dtype=np.float64)
    g = np.concatenate([z0, np.sqrt(Ta * Tb)], axis=1)
    ac = np.abs(cos)
    seuil = (ac - TOL_ULP_BORNE * np.spacing(np.maximum(ac, 1e-300)))[:, None]
    p_sym = np.minimum((g < seuil).sum(axis=1), k)
    # paires que la seule tolérance sépare : publiées, jamais tues
    strict = np.minimum((g < ac[:, None]).sum(axis=1), k)
    c2 = cos * cos
    s2 = (c2 - TOL_ULP_BORNE * np.spacing(np.maximum(c2, 1e-300)))[:, None]
    p_a = np.minimum((np.concatenate([z0, Ta], axis=1) < s2).sum(axis=1), k)
    p_b = np.minimum((np.concatenate([z0, Tb], axis=1) < s2).sum(axis=1), k)
    out = {"inter": inter, "cos": cos, "f": f, "accord": acc,
           "p_sym": p_sym.astype(np.int64),
           "p_sym_strict": strict.astype(np.int64),
           "decidee_par_la_tolerance": (strict != p_sym),
           "p_pair": np.maximum(p_a, p_b).astype(np.int64)}
    if avec_marge:
        out["marge_ulp_a"] = _marge_batch(Za, k)
        out["marge_ulp_b"] = _marge_batch(Zb, k)
    return out


def _marge_batch(Z, k: int = K_TOPK):
    """Marge à la coupure `k`, en **ULP** de la valeur de coupure (`V-ulp`)."""
    a = np.abs(np.asarray(Z, dtype=np.float64))
    D = a.shape[1]
    part = np.partition(a, [D - k - 1, D - k], axis=1)
    coupe, suivant = part[:, D - k], part[:, D - k - 1]
    u = np.spacing(np.maximum(coupe, np.finfo(np.float64).tiny))
    return (coupe - suivant) / u


def _init_accumulateurs(conditions) -> dict:
    """Accumulateurs par condition et par strate. `somme_inter` reste ENTIER
    (exactitude de `O` en `p/64`, 0-132) ; les ventilations de `σ±` par taille
    d'intersection sont des vecteurs de longueur 7 (bin 6 = « ≥ 6 »)."""
    return {x: {s: {"inter": [], "cos": [], "f": [], "p_sym": [], "p_pair": [],
                    "tiges": [], "accord_num": 0, "accord_den": 0,
                    "accord_num_par_inter": np.zeros(7, dtype=np.float64),
                    "accord_den_par_inter": np.zeros(7, dtype=np.float64),
                    "somme_inter": 0, "n_unites": 0, "n_avec_inter": 0,
                    "viol_p_sym": 0, "viol_p_sym_strict": 0,
                    "n_decidees_par_la_tolerance": 0, "n_verifiees": 0}
                for s in STRATES} for x in conditions}


def mesure(nom_modele: str, couche: int, cfg: EngramConfig, R: int,
           seed_dir: int, banc_e, provenance_ok: bool,
           conditions=CENTRAGES, bloc: int = 128, journal=None) -> dict:
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
    t0 = time.time()
    mat, H, cellules = etats_decisionnels(nom_modele, couche)
    dec, tige_de, S = _tiges_et_S(mat)
    n_dec, d = H[cellules[0]].shape
    k = cfg.dg_topk
    mem = FastWeightMemory(d, cfg, device="cpu")     # M jamais lue ni écrite
    G = mem.G.numpy().astype(np.float64)
    del mem
    Z = {c: H[c] @ G.T for c in cellules}            # cache `Z` fp64
    g1 = G @ np.ones(d, dtype=np.float64)            # pour `auto`
    dirs = _directions(d, R, seed_dir)
    Gr = (G @ dirs.T)                                # (D, R) — une seule fois
    n_global = n_dec * len(cellules)
    somme_globale = sum(Z[c].sum(axis=0) for c in cellules)
    mu_global = sum(H[c].sum(axis=0) for c in cellules) / n_global
    tiges = sorted({tige_de[i] for i in range(n_dec)})

    ia, ib = np.triu_indices(n_dec, 1)
    strate_de = np.array([p4.strate(dec[a], dec[b]) for a, b in zip(ia, ib)])
    tiges_paire = [(tige_de[a], tige_de[b]) for a, b in zip(ia, ib)]
    intra = int(sum(1 for a, b in zip(ia, ib) if tige_de[a] == tige_de[b]))

    par = _init_accumulateurs(conditions)
    # ---- split-half (§5-7, double mesure D26) : accumulateurs séparés.
    par_sh = _init_accumulateurs(SPLIT_HALF_CONDITIONS)
    moitie = moities_de_cellule(n_dec)
    n_a_cheval = 0
    marges = {}
    for c in cellules:
        Zc, Hc = Z[c], H[c]
        sZ, sH = Zc.sum(axis=0), Hc.sum(axis=0)
        moy = Hc.mean(axis=1)
        premier_bloc = (c == cellules[0])
        # moyennes par MOITIÉ : `μ` de la moitié `1−h` ne contient ni `a` ni `b`
        mu_z_moitie = np.stack([Zc[moitie == h].mean(axis=0) for h in (0, 1)])
        mu_h_moitie = np.stack([Hc[moitie == h].mean(axis=0) for h in (0, 1)])
        cnorm_moitie = np.linalg.norm(mu_h_moitie, axis=1)
        for deb in range(0, len(ia), bloc):
            sl = slice(deb, min(deb + bloc, len(ia)))
            a, b = ia[sl], ib[sl]
            tp = tiges_paire[deb:deb + len(a)]
            mu_z = (sZ[None, :] - Zc[a] - Zc[b]) / (n_dec - 2)
            mu_h = (sH[None, :] - Hc[a] - Hc[b]) / (n_dec - 2)
            cnorm = np.linalg.norm(mu_h, axis=1)
            echantillon = premier_bloc and deb == 0
            # ---- split-half : paires NON à cheval seulement, cardinal publié
            ns = moitie[a] == moitie[b]
            if ns.any():
                a2, b2 = a[ns], b[ns]
                autre = 1 - moitie[a2]
                mu_sh = mu_z_moitie[autre]
                cn_sh = cnorm_moitie[autre]
                st2, cn2 = strate_de[sl][ns], cnorm[ns]
                tp2 = [t for t, k2 in zip(tp, ns) if k2]
                n_a_cheval += int((~ns).sum())
                _ranger(par_sh["type_sh"],
                        quantites_batch(Zc[a2] - mu_sh, Zc[b2] - mu_sh, k),
                        st2, tp2, premier=True, n_dir=1)
                _ranger(par_sh["type_loo_sub"],
                        quantites_batch(Zc[a2] - mu_z[ns], Zc[b2] - mu_z[ns], k),
                        st2, tp2, premier=True, n_dir=1)
                for r in range(R):
                    gr = Gr[:, r][None, :]
                    _ranger(par_sh["plac_sh"],
                            quantites_batch(Zc[a2] - cn_sh[:, None] * gr,
                                            Zc[b2] - cn_sh[:, None] * gr, k),
                            st2, tp2, premier=(r == 0), n_dir=R)
                    _ranger(par_sh["plac_loo_sub"],
                            quantites_batch(Zc[a2] - cn2[:, None] * gr,
                                            Zc[b2] - cn2[:, None] * gr, k),
                            st2, tp2, premier=(r == 0), n_dir=R)
            for x in conditions:
                if x == "plac":
                    for r in range(R):
                        gr = Gr[:, r][None, :]
                        q = quantites_batch(Zc[a] - cnorm[:, None] * gr,
                                            Zc[b] - cnorm[:, None] * gr, k,
                                            avec_marge=(echantillon and r == 0))
                        _ranger(par[x], q, strate_de[sl], tp,
                                premier=(r == 0), n_dir=R)
                        if echantillon and r == 0:
                            marges["plac"] = _resume_marge(q)
                    continue
                if x == "aucun":
                    Za, Zb = Zc[a], Zc[b]
                elif x == "glob":
                    Za = Zc[a] - (somme_globale[None, :] - Zc[a]) / (n_global - 1)
                    Zb = Zc[b] - (somme_globale[None, :] - Zc[b]) / (n_global - 1)
                elif x == "type":
                    Za, Zb = Zc[a] - mu_z, Zc[b] - mu_z
                elif x == "auto":
                    Za = Zc[a] - moy[a][:, None] * g1[None, :]
                    Zb = Zc[b] - moy[b][:, None] * g1[None, :]
                else:
                    raise ValueError(f"condition inconnue : {x}")
                q = quantites_batch(Za, Zb, k, avec_marge=echantillon)
                _ranger(par[x], q, strate_de[sl], tp, premier=True, n_dir=1)
                if echantillon:
                    marges[x] = _resume_marge(q)
    # `Core` : supports des 10 unités de `S`, une par tige, cellule de référence
    cref = cellules[0]
    sup_S = [support_topk(Z[cref][i], k) for i in S]
    noyau = core(sup_S)
    sup_mu = support_topk(G @ mu_global, k)
    noyau_g = core_g(noyau["Core"], sup_mu)
    return {"modele": nom_modele, "couche": couche, "d": d, "n_dec": n_dec,
            "cellules": cellules, "R": R, "seed_directions": seed_dir,
            "tiges": tiges, "n_paires_par_cellule_de_capture": len(ia),
            # **Unité NOMMÉE** (défaut mineur du tour de correction) : `intra`
            # est un cardinal PAR CELLULE DE CAPTURE, alors que `P` est agrégé
            # sur les 6 cellules. Les DEUX sont publiés, avec leur unité.
            "cardinal_intra_tige": intra,
            "cardinal_intra_tige_unite":
                "paires intra-tige PAR CELLULE DE CAPTURE (il y a "
                f"{len(cellules)} cellules) — ce n'est PAS l'unité de `P`",
            "cardinal_intra_tige_par_strate": {
                s: int(sum(1 for j in range(len(ia))
                           if strate_de[j] == s
                           and tiges_paire[j][0] == tiges_paire[j][1]))
                for s in STRATES},
            "cardinal_intra_tige_total_unite_de_P": intra * len(cellules),
            "cardinal_intra_tige_total_par_strate_unite_de_P": {
                s: int(sum(1 for j in range(len(ia))
                           if strate_de[j] == s
                           and tiges_paire[j][0] == tiges_paire[j][1]))
                * len(cellules)
                for s in STRATES},
            "n_cell": n_dec, "bascule": bascule_split_half(n_dec),
            "S": S, "appartenance_S": {str(i): tige_de[i] for i in S},
            "Core": noyau, "Core_G": noyau_g, "cellule_de_reference_Core": cref,
            "marges_ulp": marges, "par": par, "par_split_half": par_sh,
            "split_half": {
                "partition": "parité de l'index de l'unité dans sa cellule de "
                             "capture (opérationnalisation DÉCLARÉE)",
                "taille_des_moities": [int((moitie == 0).sum()),
                                       int((moitie == 1).sum())],
                "regle_d_estimation": "pour une paire dont les deux membres "
                                      "sont dans la moitié h, μ_type est "
                                      "estimée sur la moitié 1−h — elle ne "
                                      "contient ni a ni b : fuite NULLE par "
                                      "construction",
                "paires_a_cheval_exclues": int(n_a_cheval),
                "unite_du_cardinal": "paires, agrégées sur les "
                                     f"{len(cellules)} cellules de capture",
                "controle_apparie": "type_loo_sub / plac_loo_sub — le LOO "
                                    "refait sur EXACTEMENT le même "
                                    "sous-ensemble de paires",
                "statut": "CONTRÔLE (double mesure D26) ; le LOO reste "
                          "PRINCIPAL, la bascule §14-2 ne s'est pas déclenchée "
                          "(n_cell = 60 ≥ 30)"},
            "duree_s": round(time.time() - t0, 2)}


def _ranger(dest, q, strates, tiges_paire, premier: bool, n_dir: int):
    """Range un bloc de résultats par strate.

    `somme_inter` reste un **entier** — l'exactitude de `O` en fraction `p/64`
    en dépend (0-132). Les listes par paire servent à l'inférence ; sous
    placebo elles portent la **moyenne sur les `R` directions**, accumulée
    direction par direction (`premier` marque `r = 0`).

    `V-borne` est vérifiée sur **100 % des paires calculées**, toutes
    directions comprises — pas seulement sur celles gardées pour l'inférence.
    """
    for s in STRATES:
        m = strates == s
        if not m.any():
            continue
        d = dest[s]
        n = int(m.sum())
        inter_m = q["inter"][m]
        d["somme_inter"] += int(inter_m.sum())
        d["n_unites"] += n
        d["accord_num"] += int(q["accord"][m].sum())
        d["accord_den"] += int(inter_m.sum())
        # `σ±` ventilée par TAILLE d'intersection — descripteur de l'anomalie
        # `σ± ≪ 0.5`, consignée NON EXPLIQUÉE. Bin 6 = « ≥ 6 ».
        bins = np.minimum(inter_m, 6)
        d["accord_num_par_inter"] += np.bincount(
            bins, weights=q["accord"][m].astype(np.float64), minlength=7)
        d["accord_den_par_inter"] += np.bincount(
            bins, weights=inter_m.astype(np.float64), minlength=7)
        d["viol_p_sym"] += int((inter_m < q["p_sym"][m]).sum())
        d["viol_p_sym_strict"] += int((inter_m < q["p_sym_strict"][m]).sum())
        d["n_decidees_par_la_tolerance"] += int(q["decidee_par_la_tolerance"][m].sum())
        d["n_verifiees"] += n
        d["n_dir"] = n_dir
        if premier:
            # `inter` reste un SOMMATEUR ENTIER (divisé par `n_dir` à la
            # synthèse) : la médiane et l'IQR restent des rationnels exacts.
            d["inter"].extend(inter_m.tolist())
            d["cos"].extend((q["cos"][m] / n_dir).tolist())
            d["f"].extend((q["f"][m] / n_dir).tolist())
            d["p_sym"].extend(q["p_sym"][m].tolist())
            d["p_pair"].extend(q["p_pair"][m].tolist())
            d["tiges"].extend([t for t, keep in zip(tiges_paire, m) if keep])
            d["n_avec_inter"] += int((inter_m > 0).sum())
        else:
            base = len(d["inter"]) - n
            ii = inter_m.tolist()
            cc, ff = q["cos"][m] / n_dir, q["f"][m] / n_dir
            for j in range(n):
                d["inter"][base + j] += int(ii[j])
                d["cos"][base + j] += float(cc[j])
                d["f"][base + j] += float(ff[j])


def pilote_r(nom_modele: str, couche: int, cfg: EngramConfig, seed_dir: int,
             r0: int = R_PILOTE, n_paires: int = 512, seed_ech: int = 0) -> dict:
    """Pilote de dérivation de `R` (§4.5) — `R₀ = 20` directions sur **un**
    modèle. `R` est **DÉRIVÉ, jamais choisi** : le poser à la main serait 0-52.

    `σ̂_dir` = écart-type **entre directions** de `O_plac` d'une même paire,
    moyenné sur les paires ; `σ̂_Δ` = écart-type **entre paires** de
    `Δ* = O_plac − O_type`. Les deux dans l'unité de publication (fraction de
    `k`), sans quoi le rapport n'aurait pas de sens.
    """
    mat, H, cellules = etats_decisionnels(nom_modele, couche)
    dec, tige_de, _ = _tiges_et_S(mat)
    n_dec, d = H[cellules[0]].shape
    k = cfg.dg_topk
    mem = FastWeightMemory(d, cfg, device="cpu")
    G = mem.G.numpy().astype(np.float64)
    del mem
    c = cellules[0]
    Zc, Hc = H[c] @ G.T, H[c]
    ia, ib = np.triu_indices(n_dec, 1)
    rng = np.random.default_rng(seed_ech)
    sel = rng.choice(len(ia), size=min(n_paires, len(ia)), replace=False)
    a, b = ia[sel], ib[sel]
    sZ, sH = Zc.sum(axis=0), Hc.sum(axis=0)
    mu_z = (sZ[None, :] - Zc[a] - Zc[b]) / (n_dec - 2)
    mu_h = (sH[None, :] - Hc[a] - Hc[b]) / (n_dec - 2)
    cnorm = np.linalg.norm(mu_h, axis=1)
    q_type = quantites_batch(Zc[a] - mu_z, Zc[b] - mu_z, k)
    Gr = G @ _directions(d, r0, seed_dir).T
    inters = np.empty((r0, len(a)), dtype=np.float64)
    for r in range(r0):
        gr = Gr[:, r][None, :]
        qq = quantites_batch(Zc[a] - cnorm[:, None] * gr,
                             Zc[b] - cnorm[:, None] * gr, k)
        inters[r] = qq["inter"]
    o_dir = inters / k
    sigma_dir = float(np.mean(o_dir.std(axis=0, ddof=1)))
    delta = o_dir.mean(axis=0) - q_type["inter"] / k
    sigma_delta = float(delta.std(ddof=1))
    d_r = r_derive(sigma_dir, sigma_delta)
    d_r.update({"modele_du_pilote": nom_modele, "R0": r0,
                "n_paires_du_pilote": int(len(a)),
                "cellule_du_pilote": c, "seed_echantillon": seed_ech,
                "unite": "fraction de k (celle de la publication de O)",
                "O_plac_moyen": float(o_dir.mean()),
                "O_type_moyen": float((q_type["inter"] / k).mean()),
                "delta_moyen": float(delta.mean())})
    return d_r


def _resume_marge(q) -> dict:
    """Marge à la coupure publiée en ULP — descriptive, aucun seuil (0-52)."""
    v = np.concatenate([q["marge_ulp_a"], q["marge_ulp_b"]])
    return {"marge_ulp_min": float(v.min()), "marge_ulp_mediane": float(np.median(v)),
            "marge_ulp_max": float(v.max()), "n_etats": int(v.size),
            "chemin_nominal": "fp64", "seuil": SANS_OBJET}


SD_NULLE_PAR_PAIRE = 0.01097     # §4.3 — sd(O)/paire sous Hypergéom(8192,64,64)
Z_95 = 1.959963984540054


def epsilon_lambda(P: int) -> float:
    """`ε_Λ` — **marge de significativité 1×** de la partition `N` (§4.6).

    Dérivée de l'enveloppe nulle **pré-enregistrée au §4.3** :
    `0.0078 ± 1.96 × 0.01097/√P`, où `0.01097 = 0.702/64` est l'écart-type par
    paire de `|A∩B|/k` sous `Hypergéom(8192, 64, 64)`. **Le corridor `2n` reste
    ABSOLU** (0-81 / D28) : `ε_Λ` est la marge de significativité, pas le
    couloir d'équivalence — les confondre serait régler le couloir sur
    l'enveloppe nulle de son propre estimateur.
    """
    P = int(P)
    if P <= 0:
        return float("inf")
    return Z_95 * SD_NULLE_PAR_PAIRE / np.sqrt(P)


def synthese(res: dict, cfg: EngramConfig, b_boot: int = B_BOOT,
             b_perm: int = B_PERM, seed: int = 0) -> dict:
    """Agrégation, inférence et classement d'un modèle. Aucune interprétation.

    `O` en **fractions exactes `p/64`** ; médiane et IQR **exactes** (sous
    placebo la valeur par paire est une moyenne sur `R` directions, donc un
    rationnel de dénominateur `R`) ; `σ±` **`SANS OBJET`** quand aucune paire
    n'a d'intersection, **jamais `0`**.
    """
    k = cfg.dg_topk
    m = res["modele"]
    tiges = res["tiges"]
    out = {"cellules": {}, "n_paires_par_cellule": {}}
    brut = {}
    toutes = dict(res["par"])
    toutes.update(res.get("par_split_half", {}))
    for x, parx in toutes.items():
        for s in STRATES:
            d = parx[s]
            nd = d.get("n_dir", 1)
            P = len(d["inter"])
            cle = f"{m}|{x}|{s}"
            out["n_paires_par_cellule"][cle] = P
            if P == 0:
                out["cellules"][cle] = {"P": 0, "O": SANS_OBJET,
                                        "f": SANS_OBJET, "sigma_pm": SANS_OBJET,
                                        "O_cos_conjoint": SANS_OBJET}
                continue
            O = Fraction(int(d["somme_inter"]), k * int(d["n_unites"]))
            vals = np.array(d["inter"], dtype=np.float64) / (nd * k)  # fraction
            cos = np.array(d["cos"], dtype=np.float64)
            q1, med, q3 = (np.quantile(np.array(d["inter"]) , q) for q in (.25, .5, .75))
            sig = (Fraction(int(d["accord_num"]), int(d["accord_den"]))
                   if d["accord_den"] > 0 else None)
            corr = (float(np.corrcoef(vals, cos)[0, 1])
                    if vals.std() > 0 and cos.std() > 0 else SANS_OBJET)
            out["cellules"][cle] = {
                "P": P, "n_dir": nd,
                "O": texte_fraction(O), "O_flottant": float(O),
                "O_en_indices": float(O) * k,
                "mediane_indices": texte_fraction(Fraction(int(med), nd)),
                "IQR_indices": [texte_fraction(Fraction(int(q1), nd)),
                                texte_fraction(Fraction(int(q3), nd))],
                "f": float(np.mean(d["f"])),
                "sigma_pm": (texte_fraction(sig) if sig is not None
                             else SANS_OBJET),
                "sigma_pm_flottant": (float(sig) if sig is not None
                                      else SANS_OBJET),
                "n_paires_avec_intersection": d["n_avec_inter"],
                "accord_numerateur": int(d["accord_num"]),
                "accord_denominateur": int(d["accord_den"]),
                "sigma_pm_par_intersection": {
                    (f"|A∩B|={j}" if j < 6 else "|A∩B|>=6"):
                        (round(float(d["accord_num_par_inter"][j]
                                     / d["accord_den_par_inter"][j]), 6)
                         if d["accord_den_par_inter"][j] > 0 else SANS_OBJET)
                    for j in range(1, 7)},
                "cos_moyen": float(cos.mean()),
                "O_cos_conjoint": {"O_moyen": float(O),
                                   "cos_moyen": float(cos.mean()),
                                   "corr_O_cos": corr},
                "p_sym_median": int(np.median(d["p_sym"])),
                "p_sym_distribution": {
                    "min": int(np.min(d["p_sym"])),
                    "q1": int(np.quantile(d["p_sym"], .25)),
                    "median": int(np.median(d["p_sym"])),
                    "q3": int(np.quantile(d["p_sym"], .75)),
                    "max": int(np.max(d["p_sym"]))},
                "p_pair_median": int(np.median(d["p_pair"])),
                "violations_p_sym": int(d["viol_p_sym"]),
                "violations_p_sym_sans_tolerance": int(d["viol_p_sym_strict"]),
                "n_decidees_par_la_tolerance":
                    int(d["n_decidees_par_la_tolerance"]),
                "paires_verifiees": int(d["n_verifiees"])}
            brut[(x, s)] = (vals, d["tiges"])
    # ---- L2 (tour additif) : `p_sym` au niveau CELLULE et l'excès
    #      `O − p_sym`, en indices. **Ré-expression** de nombres déjà
    #      calculés ; DESCRIPTIF, ne modifie aucune classe. Table `cum`
    #      chargée UNE fois (elle est une constante chargée, pas en dur).
    _cum = charger_cum()
    for _cle, _cel in out["cellules"].items():
        if isinstance(_cel, dict):
            _cel["p_sym_cellule"] = p_sym_cellule(_cel, _cum)
    # ---- inférence : IC(O − n), Λ, Δ*, ε*, classes
    n = float(N_HASARD)
    infer, classes = {}, {}
    for x in toutes:
        for s in STRATES:
            if (x, s) not in brut:
                continue
            vals, tp = brut[(x, s)]
            ic_o = bootstrap_tiges(vals - n, tp, tiges, b=b_boot, seed=seed)
            infer[f"{m}|{x}|{s}"] = {"IC_O_moins_n": ic_o["IC"],
                                     "estime_O_moins_n": ic_o["estime"],
                                     "K_eff": ic_o["K_eff"]}

    def _classes(cond_type: str, cond_plac: str, regles) -> dict:
        """Le bloc décisionnel d'une mesure. `regles` = règles `ε*` calculées.
        La classe `C` publiée sous `classe_C` est celle de la règle
        **IMPLÉMENTÉE** ; les autres sont publiées à côté, **non promues**."""
        d = {}
        for s in STRATES:
            if (cond_type, s) not in brut or (cond_plac, s) not in brut:
                continue
            vt, tp = brut[(cond_type, s)]
            lam = bootstrap_tiges(vt - float(CORRIDOR), tp, tiges,
                                  b=b_boot, seed=seed)
            eps_lam = epsilon_lambda(len(vt))
            vp, _ = brut[(cond_plac, s)]
            delta = vp - vt
            ic_d = bootstrap_tiges(delta, tp, tiges, b=b_boot, seed=seed)
            cn = classe_n(lam["IC"], infer[f"{m}|{cond_type}|{s}"]["IC_O_moins_n"],
                          eps_lam, N_HASARD)
            par_regle = {}
            for reg in regles:
                e = eps_etoile(delta, tp, tiges, b=b_perm, seed=seed, regle=reg)
                pl = plancher_epsilon(tp, tiges, b=b_perm, seed=seed, regle=reg)
                ev, dv = e["epsilon_etoile"], abs(float(ic_d["estime"]))
                plr = pl["plancher_relatif"]
                par_regle[reg] = {
                    "epsilon_etoile": ev,
                    "statut_de_la_regle": e["statut_de_la_regle"],
                    "classe_C": classe_c(ic_d["IC"], ev, N_HASARD),
                    "rapport_eps_sur_Delta": (ev / dv) if dv > 0 else SANS_OBJET,
                    "fraction_de_paires_a_signe_non_retournable":
                        e["fraction_de_paires_a_signe_non_retournable"],
                    "plancher_structurel_relatif": plr,
                    "plancher_structurel_absolu": (None if plr is None
                                                   else plr * dv),
                    # `c-cent` exige `IC_inf(Δ*) > ε*`. Le plancher structurel
                    # est la valeur MINIMALE que `ε*` peut prendre à effet
                    # constant : si `IC_inf` ne le franchit pas, la classe
                    # informative est **inatteignable sur cette cellule**,
                    # quelle que soit la donnée. Vacuité 0-47 / 0-66, publiée.
                    "classe_informative_atteignable":
                        (None if plr is None
                         else bool(float(ic_d["IC"][0]) > plr * dv)),
                    "classe_informative_inatteignable":
                        (None if plr is None
                         else not bool(float(ic_d["IC"][0]) > plr * dv)),
                    "ecritures_obligatoires": ecritures_obligatoires(ev,
                                                                     N_HASARD)}
            impl = par_regle[EPS_REGLE_IMPLEMENTEE]
            d[s] = {
                "Lambda_estime": lam["estime"], "IC_Lambda": lam["IC"],
                "epsilon_Lambda": eps_lam, "classe_N": cn,
                "Delta_etoile_estime": ic_d["estime"],
                "IC_Delta_etoile": ic_d["IC"],
                "epsilon_etoile": impl["epsilon_etoile"],
                "sigma_Delta": (float(np.std(delta, ddof=1))
                                if delta.size > 1 else 0.0),
                "P": len(delta),
                "classe_C": impl["classe_C"],
                "epsilon_etoile_par_regle": par_regle,
                "classe_C_par_regle": {r: v["classe_C"]
                                       for r, v in par_regle.items()},
                "regle_publiee_en_classe_C": EPS_REGLE_IMPLEMENTEE,
                "arbitrage": "les deux règles sont publiées côte à côte ; "
                             "AUCUNE n'est promue — l'arbitrage est PI",
                "ecritures_obligatoires": impl["ecritures_obligatoires"],
                "resolution_demi_largeur": lam["demi_largeur"]}
        return d

    classes = _classes("type", "plac", EPS_REGLES)
    # ---- L3 (tour additif) : `Δ* = Δ_rem + Δ_add`, DESCRIPTIF, recalculé
    #      depuis les accumulateurs par paire. Les trois conditions `aucun`,
    #      `type`, `plac` portent EXACTEMENT le même jeu de paires, dans le
    #      même ordre (`_ranger` les range bloc par bloc sur `triu_indices`).
    decomp = {}
    for s in STRATES:
        if all((x, s) in brut for x in ("aucun", "type", "plac")):
            decomp[s] = decomposition_delta(
                brut[("aucun", s)][0], brut[("type", s)][0],
                brut[("plac", s)][0], brut[("type", s)][1], tiges,
                b=b_boot, seed=seed)
            # contrôle de cohérence : `Δ*` de la décomposition et `Δ*` de la
            # primaire gelée sont le MÊME bootstrap sur le MÊME vecteur.
            cl = classes.get(s)
            if cl is not None:
                decomp[s]["ecart_avec_la_primaire_gelee"] = abs(
                    float(decomp[s]["Delta_etoile_estime"])
                    - float(cl["Delta_etoile_estime"]))
    classes_sh = _classes("type_sh", "plac_sh", (EPS_REGLE_IMPLEMENTEE,))
    classes_sub = _classes("type_loo_sub", "plac_loo_sub",
                           (EPS_REGLE_IMPLEMENTEE,))
    cov = {f"{m}|{s}": {
        "eps": {r: (classes.get(s, {}).get("epsilon_etoile_par_regle", {})
                    .get(r, {}).get("epsilon_etoile")) for r in EPS_REGLES},
        "mesures": {"LOO": classes.get(s, {}).get("Delta_etoile_estime"),
                    "split-half": classes_sh.get(s, {}).get(
                        "Delta_etoile_estime")}} for s in STRATES}
    out.update({"modele": m, "couche": res["couche"], "inference": infer,
                "classes_par_strate": classes,
                "classes_par_strate_split_half": classes_sh,
                "classes_par_strate_LOO_sous_ensemble": classes_sub,
                "decomposition_delta_descriptive": decomp,
                "double_mesure_D26": {
                    "principale": "LOO (bascule §14-2 non déclenchée : "
                                  f"n_cell = {res['n_cell']} ≥ "
                                  f"{N_CELL_BASCULE})",
                    "controle": "split-half",
                    "apparie": "LOO_sous_ensemble — le LOO refait sur "
                               "EXACTEMENT les paires du split-half, pour que "
                               "l'écart LOO↔split-half ne se confonde pas avec "
                               "l'écart entre deux jeux de paires",
                    "regle": "toute divergence de classe entre les deux SE "
                             "PUBLIE, elle ne se moyenne pas"},
                "couverture_publiee": cov,
                "anomalie_sigma_pm": bloc_anomalie_non_expliquee(
                    out["cellules"]),
                "split_half": res.get("split_half"),
                "Core": res["Core"], "Core_G": res["Core_G"],
                "cellule_de_reference_Core": res["cellule_de_reference_Core"],
                "appartenance_S": res["appartenance_S"],
                "marges_ulp": res["marges_ulp"],
                "cardinal_intra_tige": res["cardinal_intra_tige"],
                "cardinal_intra_tige_unite": res["cardinal_intra_tige_unite"],
                "cardinal_intra_tige_par_strate":
                    res["cardinal_intra_tige_par_strate"],
                "cardinal_intra_tige_total_unite_de_P":
                    res["cardinal_intra_tige_total_unite_de_P"],
                "cardinal_intra_tige_total_par_strate_unite_de_P":
                    res["cardinal_intra_tige_total_par_strate_unite_de_P"],
                "fraction_intra_tige_par_strate_unite_de_P": {
                    s: (res["cardinal_intra_tige_total_par_strate_unite_de_P"][s]
                        / out["n_paires_par_cellule"][f"{m}|aucun|{s}"]
                        if out["n_paires_par_cellule"].get(f"{m}|aucun|{s}")
                        else SANS_OBJET) for s in STRATES},
                "bascule_split_half": res["bascule"], "R": res["R"],
                "duree_mesure_s": res["duree_s"]})
    return out


def phrase_gravee(modele: str, cellule: dict, strate: str) -> str:
    """La phrase (xvi), instanciée — **compte d'indices**, jamais fraction ;
    médiane et IQR ; `f` et `σ±` **dans la phrase même** ; **modèle et cellule
    nommés, jamais poolés** (0-63, 0-67)."""
    return (f"Sur les états de {modele}, capturés à t, dans la cellule "
            f"{strate}, les supports de topk(G·h) partagent en moyenne "
            f"{cellule['O_en_indices']:.3f} indices sur 64 (médiane "
            f"{cellule['mediane_indices']}, IQR "
            f"[{cellule['IQR_indices'][0]}, {cellule['IQR_indices'][1]}]), "
            f"contre 0.5 indice sous tirage indépendant (k²/D), et "
            f"l'intersection porte f = {cellule['f']:.4f} de l'énergie avec un "
            f"accord de signe de σ± = {cellule['sigma_pm']}.")


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
    p.add_argument("--banc-e", dest="banc_e", type=int, default=None,
                   help="valeur de E rendue par le banc D14-S. La mesure "
                        "n'est ouverte qu'à --banc-e 0 (§6.E) ; sans l'option, "
                        "seules les portes de provenance tournent.")
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

    if a.banc_e is None:
        print(f"\ntemps CPU total : {time.time() - t0:.1f} s")
        print("\nportes de provenance PASS. Suite gravée : banc D14-S "
              "(`.venv\\Scripts\\python eval/gate_bench.py --suite dgov`), "
              "`E = 0` exigé AVANT toute mesure (§6.E). Relancer avec "
              "`--banc-e 0` une fois le banc vert.")
        return 0

    # ------------------------------------------------------------- MESURE
    pil = (r_derive(0.0, 1.0) if a.R else
           pilote_r(MODELES[0], COUCHE_REF[MODELES[0]], cfg, a.seed))
    if a.R:
        pil = {"R": int(a.R), "statut": "VALEUR DONNÉE À LA MAIN, publiée "
                                        "comme telle (le protocole la veut "
                                        "DÉRIVÉE — §4.5)"}
    R = int(pil["R"])
    print(f"\nR dérivé : {R}  ({pil.get('statut')})")
    for cle in ("R_formule", "plafond", "sigma_dir", "sigma_delta",
                "depassement", "modele_du_pilote", "R0", "n_paires_du_pilote"):
        if cle in pil:
            print(f"    {cle} = {pil[cle]}")
    (out / "R-derive.json").write_text(
        json.dumps(pil, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    par_modele, n_par_cellule, toutes_marges = {}, {}, {}
    encadrement, couverture = {}, {}
    for m in modeles:
        ce = couche_effective(m, a.layer)
        print(f"\n--- mesure : {m} (couche {ce['couche']}, {ce['source']}) ---")
        res = mesure(m, ce["couche"], cfg, R, a.seed, a.banc_e, True)
        syn = synthese(res, cfg, seed=a.seed)
        syn["couche_source"] = ce
        par_modele[m] = syn
        n_par_cellule.update(syn["n_paires_par_cellule"])
        toutes_marges[m] = syn["marges_ulp"]
        couverture.update(syn["couverture_publiee"])
        for s in STRATES:
            encadrement[(m, s)] = \
                syn["cellules"][f"{m}|aucun|{s}"]["p_sym_distribution"]
        print(f"    durée {res['duree_s']} s ; "
              f"{res['n_paires_par_cellule_de_capture']} paires × "
              f"{len(res['cellules'])} cellules ; "
              f"cardinal intra-tige {res['cardinal_intra_tige']} "
              f"PAR CELLULE DE CAPTURE = "
              f"{res['cardinal_intra_tige_total_unite_de_P']} dans l'unité de P")
        print(f"    split-half : {res['split_half']['paires_a_cheval_exclues']} "
              f"paires à cheval exclues ; moitiés "
              f"{res['split_half']['taille_des_moities']}")
        for s in STRATES:
            c = syn["cellules"][f"{m}|aucun|{s}"]
            print(f"      {s} aucun : O = {c['O']} = {c['O_en_indices']:.3f} "
                  f"indices ; méd {c['mediane_indices']} IQR "
                  f"{c['IQR_indices']} ; f = {c['f']:.4f} ; σ± = "
                  f"{c['sigma_pm']} ; cos = {c['cos_moyen']:.4f} ; "
                  f"p_sym méd = {c['p_sym_median']}")
        print("    ε* — les DEUX règles, côte à côte (aucune promue) :")
        for s in STRATES:
            cl = syn["classes_par_strate"].get(s)
            if not cl:
                continue
            pr = cl["epsilon_etoile_par_regle"]
            print(f"      {s} : Δ* = {cl['Delta_etoile_estime']:.5f} "
                  f"IC {['%.5f' % v for v in cl['IC_Delta_etoile']]} | "
                  + " | ".join(
                      f"{r}: ε*={pr[r]['epsilon_etoile']:.5f} "
                      f"(ε*/Δ*={pr[r]['rapport_eps_sur_Delta']:.3f}, "
                      f"plancher={pr[r]['plancher_structurel_relatif']:.3f}, "
                      f"signe fixe {pr[r]['fraction_de_paires_a_signe_non_retournable']:.2f}) "
                      f"→ {pr[r]['classe_C']}" for r in EPS_REGLES))
        print("    LOO vs split-half (double mesure D26) :")
        for s in STRATES:
            cl_loo = syn["classes_par_strate"].get(s)
            bb = syn["classes_par_strate_split_half"].get(s)
            cc2 = syn["classes_par_strate_LOO_sous_ensemble"].get(s)
            if not (cl_loo and bb and cc2):
                continue
            ot = syn["cellules"][f"{m}|type|{s}"]["O_en_indices"]
            osh = syn["cellules"][f"{m}|type_sh|{s}"]["O_en_indices"]
            osub = syn["cellules"][f"{m}|type_loo_sub|{s}"]["O_en_indices"]
            print(f"      {s} : O_type LOO {ot:.4f} / LOO-sub {osub:.4f} / "
                  f"split-half {osh:.4f} | Λ {cl_loo['Lambda_estime']:+.5f} / "
                  f"{cc2['Lambda_estime']:+.5f} / {bb['Lambda_estime']:+.5f} | "
                  f"Δ* {cl_loo['Delta_etoile_estime']:.5f} / "
                  f"{cc2['Delta_etoile_estime']:.5f} / "
                  f"{bb['Delta_etoile_estime']:.5f} | classes "
                  f"{cl_loo['classe_N']}/{cl_loo['classe_C']} → "
                  f"{cc2['classe_N']}/{cc2['classe_C']} → "
                  f"{bb['classe_N']}/{bb['classe_C']}")
        an = syn["anomalie_sigma_pm"]
        print(f"    anomalie σ± : {an['statut']} — "
              f"{an['n_cellules_signalees']} cellule(s) signalée(s), "
              f"{an['nature']}")
        (out / f"mesure-{slug(m)}.json").write_text(
            json.dumps(syn, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")

    # ------------------------------------------------------------- PORTES
    portes = {}
    # `V-P8` est PRÉ-ENREGISTRÉE sur les cinq conditions du §8 : elle n'est pas
    # étendue aux cellules split-half, dont le cardinal est publié à part.
    n_pre = {c: v for c, v in n_par_cellule.items()
             if c.split("|")[1] in CENTRAGES}
    portes["V-P8"] = v_p8(n_pre)
    portes["V-P8"][1]["cellules_split_half_hors_porte"] = {
        c: v for c, v in n_par_cellule.items()
        if c.split("|")[1] in SPLIT_HALF_CONDITIONS}
    portes["V-couverture"] = couverture_publiee(couverture)
    viol = sum(c["violations_p_sym"] for syn in par_modele.values()
               for c in syn["cellules"].values() if isinstance(c, dict)
               and "violations_p_sym" in c)
    nver = sum(c["paires_verifiees"] for syn in par_modele.values()
               for c in syn["cellules"].values() if isinstance(c, dict)
               and "paires_verifiees" in c)
    vb, db = _encadrement_theorique(encadrement)
    strict = sum(c["violations_p_sym_sans_tolerance"]
                 for syn in par_modele.values() for c in syn["cellules"].values()
                 if isinstance(c, dict) and "violations_p_sym" in c)
    ntol = sum(c["n_decidees_par_la_tolerance"]
               for syn in par_modele.values() for c in syn["cellules"].values()
               if isinstance(c, dict) and "violations_p_sym" in c)
    portes["V-borne"] = (FAIL if viol else vb, {
        "membre_a_identite": PASS if not viol else FAIL,
        "n_violations_p_sym": viol, "n_paires_verifiees": nver,
        "taux_conformite": (1.0 - viol / nver) if nver else SANS_OBJET,
        "tolerance_ULP": TOL_ULP_BORNE,
        "n_violations_sans_tolerance": strict,
        "n_paires_decidees_par_la_tolerance": ntol,
        "note_tolerance": "l'identité est EXACTE en arithmétique réelle et "
                          "atteinte AVEC ÉGALITÉ sur les paires à |A∩B| = 1 "
                          "dont l'indice commun est la plus grande magnitude "
                          "des deux clés ; la tolérance rend la comparaison "
                          "fp64 fidèle à cette arithmétique. Le compteur "
                          "ci-dessus dit combien de paires elle a séparées — "
                          "sans lui, une tolérance serait un masque.",
        "membre_b_encadrement": db,
        "regle": "violation = BUG DE MESURE, jamais un résultat (§6.B)"})
    portes["V-ulp"] = v_ulp({x: toutes_marges[modeles[0]].get(x)
                             for x in CENTRAGES})
    portes["V-t1"] = v_t1([], {m: par_modele[m]["cardinal_intra_tige_par_strate"]
                               for m in modeles},
                          core_S=[int(i) for i in par_modele[modeles[0]]
                                  ["appartenance_S"]],
                          tige_de={int(i): t for i, t in
                                   par_modele[modeles[0]]["appartenance_S"].items()})
    portes["V-core-S"] = v_core_s(
        [int(i) for i in par_modele[modeles[0]]["appartenance_S"]],
        {int(i): t for i, t in
         par_modele[modeles[0]]["appartenance_S"].items()})
    portes["V-seed"] = v_seed({
        "seed": a.seed,
        "generateur": "torch.randn(R, d, generator=manual_seed(seed))",
        "regle_de_tirage": "normalisation L2 par ligne ; indices 0..R−1",
        "indices": f"0..{R - 1}",
        "cardinal_par_modele": {m: R for m in MODELES},
        "appariement_inter_modeles": "par la NORME, jamais par le vecteur"})
    diag = {"modeles": list(modeles), "conditions": list(CENTRAGES),
            "cellules": {k: v for syn in par_modele.values()
                         for k, v in syn["cellules"].items()}}
    portes["V-diag"] = v_diag(diag)
    # ---- tour additif : couverture ÉNUMÉRÉE des 60 cellules × colonnes,
    #      `p_sym` défini sur les CINQ conditions (cardinal COMPTÉ), et
    #      l'identité `Δ_rem + Δ_add = Δ*`. Portes NEUVES, additives : elles
    #      ne touchent aucune clause gelée.
    portes["V-diag-colonnes"] = v_diag_colonnes(diag["cellules"],
                                                modeles=modeles)
    portes["V-psym-conditions"] = v_psym_conditions(diag["cellules"],
                                                    modeles=modeles)
    id_det, id_ko = {}, 0
    for m2, syn in par_modele.items():
        for s, dd in syn["decomposition_delta_descriptive"].items():
            id_det[f"{m2}|{s}"] = {
                "par_paire": dd["identite_par_paire"],
                "sur_les_estimes": dd["identite_sur_les_estimes"]["detail"],
                "verdict_estimes": dd["identite_sur_les_estimes"]["verdict"],
                "ecart_avec_la_primaire_gelee":
                    dd.get("ecart_avec_la_primaire_gelee")}
            id_ko += int(dd["identite_par_paire"]["verdict"] != PASS)
            id_ko += int(dd["identite_sur_les_estimes"]["verdict"] != PASS)
    portes["V-identite-decomposition"] = (
        (PASS if id_ko == 0 else FAIL),
        {"n_cellules_verifiees": len(id_det), "n_echecs": id_ko,
         "tolerance_ULP": TOL_ULP_BORNE, "detail": id_det,
         "statut": "Δ_rem et Δ_add sont DESCRIPTIFS ; Δ* reste la primaire "
                   "gelée. Cette porte ne vérifie qu'une IDENTITÉ.",
         "regle": "tout écart au-delà de la tolérance ULP déjà en vigueur "
                  "est un BUG, jamais un résultat"})
    portes["V-norm"] = (v, dn)
    # **Compteur de vacuité** : combien de cellules ont, par règle, une classe
    # informative (`c-cent`) INATTEIGNABLE compte tenu du plancher structurel.
    # Publié, jamais tu — mode 0-47 / 0-66.
    vac, det_vac = {}, {}
    for r in EPS_REGLES:
        n_in = 0
        for m2, syn in par_modele.items():
            for s in STRATES:
                cl = syn["classes_par_strate"].get(s)
                if not cl:
                    continue
                pr = cl["epsilon_etoile_par_regle"][r]
                det_vac[f"{m2}|{s}|{r}"] = {
                    "plancher_structurel_relatif":
                        pr["plancher_structurel_relatif"],
                    "epsilon_etoile": pr["epsilon_etoile"],
                    "rapport_eps_sur_Delta": pr["rapport_eps_sur_Delta"],
                    "IC_inf_Delta": cl["IC_Delta_etoile"][0],
                    "classe_informative_inatteignable":
                        pr["classe_informative_inatteignable"],
                    "classe_C": pr["classe_C"]}
                n_in += 1 if pr["classe_informative_inatteignable"] else 0
        vac[r] = f"{n_in}/12"
    portes["V-vacuite"] = (
        PASS, {"cellules_a_classe_informative_inatteignable_par_regle": vac,
               "n_cellules": 12,
               "detail": det_vac,
               "definition": "c-cent exige IC_inf(Δ*) > ε* ; le plancher "
                             "structurel est la valeur MINIMALE que ε* peut "
                             "prendre à effet constant. Quand IC_inf ne le "
                             "franchit pas, la classe informative est vide sur "
                             "cette cellule quelle que soit la donnée",
               "statut": "COMPTEUR PUBLIÉ — descriptif, ne modifie aucune "
                         "classe ; la vacuité se publie, elle ne se découvre "
                         "pas après coup (0-47 / 0-66)"})
    print("\nPORTES")
    for nom, (verdict, _) in portes.items():
        print(f"  {nom:<10} {verdict}")

    (out / "portes.json").write_text(
        json.dumps({k: {"verdict": x, "detail": y}
                    for k, (x, y) in portes.items()},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lignes = ["modele,condition,strate,P,O_fraction,O_indices,mediane,IQR_bas,"
              "IQR_haut,f,sigma_pm,cos_moyen,corr_O_cos,p_sym_median,"
              "p_pair_median,"
              # colonnes AJOUTÉES par le tour additif — les existantes sont
              # inchangées, dans le même ordre
              "p_sym_cellule_realise_median,exces_O_moins_p_sym_realise,"
              "p_sym_cellule_theorique,exces_O_moins_p_sym_theorique"]
    for m, syn in par_modele.items():
        for x in tuple(CENTRAGES) + tuple(SPLIT_HALF_CONDITIONS):
            for s in STRATES:
                c = syn["cellules"][f"{m}|{x}|{s}"]
                b = c.get("p_sym_cellule") or {}
                b = b if isinstance(b, dict) else {}
                lignes.append(
                    f"{m},{x},{s},{c['P']},{c['O']},{c['O_en_indices']:.6f},"
                    f"{c['mediane_indices']},{c['IQR_indices'][0]},"
                    f"{c['IQR_indices'][1]},{c['f']:.6f},{c['sigma_pm']},"
                    f"{c['cos_moyen']:.6f},{c['O_cos_conjoint']['corr_O_cos']},"
                    f"{c['p_sym_median']},{c['p_pair_median']},"
                    f"{_csv(b.get('p_sym_realise_median'))},"
                    f"{_csv(b.get('exces_O_moins_p_sym_realise_median'))},"
                    f"{_csv(b.get('p_sym_theorique_du_cos_de_cellule'))},"
                    f"{_csv(b.get('exces_O_moins_p_sym_theorique'))}")
    (out / "summary.csv").write_text("\n".join(lignes) + "\n", encoding="utf-8")

    # ---- L1 : la table `V-diag` COMPLÈTE — 3 modèles × 5 conditions du §8
    #      × 4 strates, colonnes `O` (indices), `cos`, `f`, `σ±`, `p_sym`,
    #      excès. **Tabulation** de quantités déjà calculées.
    ld = ["modele,condition,strate,P,O_indices,cos,f,sigma_pm,"
          "sigma_pm_flottant,mediane_indices,IQR_bas,IQR_haut,corr_O_cos,"
          "p_sym_realise_median,exces_O_moins_p_sym_realise,"
          "p_sym_theorique_du_cos,exces_O_moins_p_sym_theorique,"
          "statut_p_sym_theorique"]
    for m, syn in par_modele.items():
        for x in CENTRAGES:
            for s in STRATES:
                c = syn["cellules"][f"{m}|{x}|{s}"]
                b = c.get("p_sym_cellule") or {}
                b = b if isinstance(b, dict) else {}
                ld.append(
                    f"{m},{x},{s},{c['P']},{c['O_en_indices']:.6f},"
                    f"{c['cos_moyen']:.6f},{c['f']:.6f},{c['sigma_pm']},"
                    f"{_csv(c['sigma_pm_flottant'])},{c['mediane_indices']},"
                    f"{c['IQR_indices'][0]},{c['IQR_indices'][1]},"
                    f"{_csv(c['O_cos_conjoint']['corr_O_cos'])},"
                    f"{_csv(b.get('p_sym_realise_median'))},"
                    f"{_csv(b.get('exces_O_moins_p_sym_realise_median'))},"
                    f"{_csv(b.get('p_sym_theorique_du_cos_de_cellule'))},"
                    f"{_csv(b.get('exces_O_moins_p_sym_theorique'))},"
                    f"{_csv(b.get('statut_p_sym_theorique'))}")
    (out / "diagnostic-V-diag.csv").write_text("\n".join(ld) + "\n",
                                               encoding="utf-8")

    # ---- L3 : `Δ* = Δ_rem + Δ_add`, DESCRIPTIF, avec IC bootstrap de tiges
    lz = ["modele,strate,P,Delta_rem,IC_rem_inf,IC_rem_sup,Delta_add,"
          "IC_add_inf,IC_add_sup,Delta_etoile,IC_Delta_inf,IC_Delta_sup,"
          "Delta_rem_indices,Delta_add_indices,Delta_etoile_indices,"
          "identite_par_paire,ecart_max_ULP,identite_sur_les_estimes,"
          "ecart_estimes_ULP,ecart_avec_la_primaire_gelee,statut"]
    for m, syn in par_modele.items():
        for s in STRATES:
            dd = syn["decomposition_delta_descriptive"].get(s)
            if not dd:
                continue
            ip, ie = dd["identite_par_paire"], dd["identite_sur_les_estimes"]
            lz.append(
                f"{m},{s},{dd['P']},{dd['Delta_rem_estime']:.8f},"
                f"{dd['IC_Delta_rem'][0]:.8f},{dd['IC_Delta_rem'][1]:.8f},"
                f"{dd['Delta_add_estime']:.8f},"
                f"{dd['IC_Delta_add'][0]:.8f},{dd['IC_Delta_add'][1]:.8f},"
                f"{dd['Delta_etoile_estime']:.8f},"
                f"{dd['IC_Delta_etoile'][0]:.8f},"
                f"{dd['IC_Delta_etoile'][1]:.8f},"
                f"{dd['Delta_rem_en_indices']:.6f},"
                f"{dd['Delta_add_en_indices']:.6f},"
                f"{dd['Delta_etoile_en_indices']:.6f},"
                f"{ip['verdict']},{ip['ecart_max_en_ULP']:.3f},"
                f"{ie['verdict']},{ie['detail']['ecart_en_ULP']:.3f},"
                f"{_csv(dd.get('ecart_avec_la_primaire_gelee'))},DESCRIPTIF")
    (out / "decomposition-delta.csv").write_text("\n".join(lz) + "\n",
                                                 encoding="utf-8")

    # ---- sensibilité `ε*` : les DEUX règles côte à côte, aucune promue
    ls = ["modele,strate,P,Delta_etoile,IC_inf,IC_sup,regle,statut_regle,"
          "epsilon_etoile,rapport_eps_sur_Delta,plancher_structurel_relatif,"
          "plancher_structurel_absolu,fraction_signe_non_retournable,classe_C"]
    for m, syn in par_modele.items():
        for s in STRATES:
            cl = syn["classes_par_strate"].get(s)
            if not cl:
                continue
            for r in EPS_REGLES:
                v = cl["epsilon_etoile_par_regle"][r]
                ls.append(
                    f"{m},{s},{cl['P']},{cl['Delta_etoile_estime']:.8f},"
                    f"{cl['IC_Delta_etoile'][0]:.8f},{cl['IC_Delta_etoile'][1]:.8f},"
                    f"{r},{v['statut_de_la_regle']},{v['epsilon_etoile']:.8f},"
                    f"{v['rapport_eps_sur_Delta']:.6f},"
                    f"{v['plancher_structurel_relatif']:.6f},"
                    f"{v['plancher_structurel_absolu']:.8f},"
                    f"{v['fraction_de_paires_a_signe_non_retournable']:.6f},"
                    f"{v['classe_C']}")
    (out / "epsilon-sensibilite.csv").write_text("\n".join(ls) + "\n",
                                                 encoding="utf-8")

    # ---- double mesure D26 : LOO / LOO sous-ensemble / split-half
    lm = ["modele,strate,mesure,P,O_type_indices,Lambda,IC_Lambda_inf,"
          "IC_Lambda_sup,classe_N,Delta_etoile,IC_Delta_inf,IC_Delta_sup,"
          "epsilon_etoile,classe_C"]
    for m, syn in par_modele.items():
        for nom, cle_cls, cond in (("LOO", "classes_par_strate", "type"),
                                   ("LOO_sous_ensemble",
                                    "classes_par_strate_LOO_sous_ensemble",
                                    "type_loo_sub"),
                                   ("split-half",
                                    "classes_par_strate_split_half",
                                    "type_sh")):
            for s in STRATES:
                cl = syn[cle_cls].get(s)
                if not cl:
                    continue
                o = syn["cellules"][f"{m}|{cond}|{s}"]["O_en_indices"]
                lm.append(
                    f"{m},{s},{nom},{cl['P']},{o:.6f},"
                    f"{cl['Lambda_estime']:.8f},{cl['IC_Lambda'][0]:.8f},"
                    f"{cl['IC_Lambda'][1]:.8f},{cl['classe_N']},"
                    f"{cl['Delta_etoile_estime']:.8f},"
                    f"{cl['IC_Delta_etoile'][0]:.8f},"
                    f"{cl['IC_Delta_etoile'][1]:.8f},"
                    f"{cl['epsilon_etoile']:.8f},{cl['classe_C']}")
    (out / "double-mesure-LOO-splithalf.csv").write_text("\n".join(lm) + "\n",
                                                         encoding="utf-8")
    print(f"\ntemps CPU total : {time.time() - t0:.1f} s")
    print(f"bruts : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
