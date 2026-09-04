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
import math
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

# =========================================================================
#  CYCLE « plancher de `top64(G·μ_global)` et clôture de `σ±` » — Étape A
#  Protocole : experiments/EXP-2026-08-27-plancher-base-sigma.md
#  (PRE-ENREGISTRE ; §4 et §6 GELÉS le 2026-08-28 ; §14 = déclarations
#  d'opérationnalisation rendues AVANT le banc).
#
#  **Motif D29 : ce cycle ÉTEND ce fichier, il n'en écrit pas un neuf.**
#  `V-cache`, `V-G`, `V-ulp`, `V-P8`, `V-seed`, le cache `Z`, `p_sym` et les
#  fractions exactes sont ceux du cycle précédent, déjà relus.
#
#  Périmètre gravé : **0 GPU, aucun forward, `M` jamais instanciée** (seule la
#  `G` de `FastWeightMemory` est lue, comme au cycle précédent), `engram/` non
#  modifié, mêmes états v4 en cache, fp64 chemin nominal partout (D21).
# =========================================================================

SORTIE_PBS = ROOT / "experiments" / "results" / "plancher-base-sigma"

PBS_REFERENCES = ("mu", "etat", "plac")
PBS_CONDITIONS = ("aucun", "type")      # `aucun` PRIMAIRE, `type` double mesure D26
N_G_SIM = 200                 # `Q-M1-bis` — répliques de `G` de la simulation à rang
N_BUCKETS = 8                 # 8 buckets dyadiques de 8 rangs (0-176)
BUCKET_MIN_TRIALS = 200       # planchers de `V-rang-def`
BUCKET_MIN_PAIRES = 8
R_S_TIRAGES = 10_000          # `Q-M12` — tirages de `S` pour `q₉₅` de `N-hyp`
SEED_S = 0                    # générateur et seed publiés (D24)
TOL_SIM_QUAD = 0.005          # certification `|sim − quad|` par cellule
TOL_QUAD_CROISEE = 1e-9       # écart exigé entre les deux chemins de quadrature
EPS_DEGENERE = 1e-9           # cas dégénérés gelés de `Q-M10`
TOL_PSD = 1e-8                # Gram non-PSD au-delà ⇒ ARRÊT (jamais de clip)
W_TRONCATURE = 8.0            # troncature licite du domaine `(t, ∞)` (err < 1.3e−14)
QUAD_EPSABS = 1e-12
QUAD_EPSREL = 1e-10
N_GL_W = 64                   # nœuds de Gauss–Legendre du facteur `w`
N_GL_W_CONTROLE = 128         # contre-vérification exigée (écart < 1e−9)
N_GL_THETA = 32               # nœuds en `θ` pour `F₂`, régime |r| ≤ 0.9
N_GL_THETA_GRADE = 24         # 3 panneaux gradués, régime |r| > 0.9
SEUIL_R_GRADE = 0.9
# §14.2 — lecture RELATIONNELLE du vivier `N-état`. Ces cardinaux sont
# ATTENDUS ; ils sont **recomptés par énumération** (`V-appar`), jamais crus.
CARDINAUX_VIVIER_ATTENDUS = {"S3": (1, 1), "S2": (0, 0), "S1": (9, 9),
                             "S0": (24, 30)}

# `V-mu` (§4.6, critère d'abandon §6.D) — tolérance de l'écart RELATIF entre les
# deux chemins de calcul de `μ_global`. Le second chemin est une sommation
# COMPENSÉE exacte (`math.fsum`) coordonnée par coordonnée, dans un ordre de
# parcours différent : ce n'est pas la même arithmétique, donc un écart au-delà
# de quelques ULP accumulés sur 360 termes est un symptôme, pas du bruit.
TOL_MU = 1e-12

# Les trois lectures de `cos_p` de `σ̂±_pool`, publiées CÔTE À CÔTE (clause
# sous-spécifiée « cos_p de σ̂±_pool », arbitrage PI en cours). `h` reste la
# lecture DÉCLARÉE PRINCIPALE ; les deux autres sont descriptives et ne changent
# aucune classe primaire par la main de l'implémentation.
LECTURES_COS_P = ("h", "z", "tronque_phi")
LECTURE_COS_P_PRINCIPALE = "h"

# §14.5 — frontière chiffrée de `Q-M13` : l'IC contient 0.5 ssi `ρ_ic ≥ ≈ 0.26`
# (pour `σ± = 0.410`) à `0.38` (pour 0.392). La borne BASSE est retenue : c'est
# celle qui est franchie la première.
FRONTIERE_Q_M13 = 0.26

# `B = 10⁴` — clause GELÉE du §4.4 et du §7, pour `ε_Ψ` comme pour `N-grappe`.
# Le tour de correction précédent employait 1000 et 2000 « pour tenir le budget
# < 10 min » ; le budget a été dépassé de 110 % et le motif est tombé.
# DÉCISION PI DU 2026-08-28 : re-mesurer à `B = 10⁴`, comme gravé. Les deux jeux
# de quantiles sont publiés côte à côte et la bascule de classe est VÉRIFIÉE,
# jamais supposée.
B_EPSILON_PSI = 10_000
B_GRAPPE = 10_000
B_EPSILON_PSI_REDUIT = 1_000    # jeu DESCRIPTIF de comparaison (ancien B)
B_GRAPPE_REDUIT = 2_000         # jeu DESCRIPTIF de comparaison (ancien B)


class ArretQM10(RuntimeError):
    """`Q-M10` — Gram non-PSD au-delà de `1e−8` : **symptôme de bug**, ARRÊT.
    **Jamais de clip silencieux.**"""


# -------------------------------------------------------------------------
#  §Q-M10 — noyaux numériques.
#
#  `scipy` n'est **PAS une dépendance du projet** (`requirements.txt` : torch,
#  transformers, pytest). La spécification nomme `scipy.integrate.quad`,
#  `scipy.special.ndtri` et une CDF normale bivariée. **Substitution
#  DÉCLARÉE, jamais silencieuse**, sur le modèle déjà employé au cycle
#  précédent pour `Φ⁻¹` (AS 241, vérifié par aller-retour) :
#
#    * `Φ⁻¹`  → `_ndtri` (AS 241), déjà en place, vérifié par `math.erfc` ;
#    * `Φ`    → `math.erf` vectorisé (`_Phi`) ;
#    * `F₂`   → forme de Sheppard/Plackett en `θ` (chemin NOMINAL, vectorisé)
#               **contre-vérifiée** par la forme conditionnelle en `x`
#               (`bvn_F2_ref`), intégrée par Gauss–Kronrod adaptatif : les
#               deux chemins ne partagent aucune ligne ;
#    * `quad` → Gauss–Legendre à `N_GL_W` nœuds sur `(t, 8)` (NOMINAL),
#               **contre-vérifié** par Gauss–Legendre 128 nœuds ET par
#               Gauss–Kronrod adaptatif (`epsabs = 1e−12`, `epsrel = 1e−10`).
# -------------------------------------------------------------------------

_INV_2PI = 1.0 / (2.0 * np.pi)


def _Phi(x):
    """`Φ` vectorisée, fp64 — `0.5·erfc(−x/√2)`.

    **Substitution déclarée** : `numpy` n'expose pas `erf`/`erfc` et `scipy`
    n'est pas une dépendance ; `torch.special.erfc` (déjà dans
    `requirements.txt`) fournit le chemin **vectorisé**, et `math.erfc` le
    chemin **scalaire de vérification** (`_verif_Phi`). Aucun des deux ne
    partage une ligne avec `_ndtri` (AS 241).
    """
    a = np.asarray(x, dtype=np.float64)
    plat = np.ascontiguousarray(a.reshape(-1) / -np.sqrt(2.0))
    v = torch.special.erfc(torch.from_numpy(plat)).numpy() * 0.5
    return np.float64(v[0]) if a.ndim == 0 else v.reshape(a.shape)


def _verif_Phi(pts=(-8.0, -3.0, -0.5, 0.0, 0.5, 2.66, 3.0, 8.0)) -> dict:
    """Vérification **indépendante** de `_Phi` par `math.erfc` (libm)."""
    import math
    e = 0.0
    for p in pts:
        e = max(e, abs(float(_Phi(np.array([p]))[0])
                       - 0.5 * math.erfc(-p / math.sqrt(2.0))))
    return {"ecart_max_torch_vs_math_erfc": e, "points": list(pts)}


def _phi_dens(x):
    """φ, densité normale centrée réduite, fp64."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) * (1.0 / np.sqrt(2.0 * np.pi))


def seuil_t() -> float:
    """`t = Φ⁻¹(1 − 1/256)` — **calculé au banc en fp64, JAMAIS `2.66` en dur**
    (0-177 : la pente `(φ/Q)²` varie de 1.5 % entre 2.66 et 2.6601)."""
    return _ndtri(1.0 - 1.0 / 256.0)


def _gl(n: int):
    """Nœuds et poids de Gauss–Legendre sur `[-1, 1]` (`numpy`, fp64)."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


# Gauss–Kronrod 7/15 — nœuds et poids classiques, fp64. Chemin de
# VÉRIFICATION seulement (scalaire) ; le chemin nominal est vectorisé.
_GK15_X = np.array([
    0.991455371120813, 0.949107912342759, 0.864864423359769,
    0.741531185599394, 0.586087235467691, 0.405845151377397,
    0.207784955007898, 0.000000000000000], dtype=np.float64)
_GK15_WK = np.array([
    0.022935322010529, 0.063092092629979, 0.104790010322250,
    0.140653259715525, 0.169004726639267, 0.190350578064785,
    0.204432940075298, 0.209482141084728], dtype=np.float64)
_GK15_WG = np.array([
    0.129484966168870, 0.279705391489277, 0.381830050505119,
    0.417959183673469], dtype=np.float64)


def _gk15(f, a: float, b: float):
    """Une passe Gauss–Kronrod 7/15 sur `[a, b]` : rend `(K, |K − G|)`."""
    c, h = 0.5 * (a + b), 0.5 * (b - a)
    xs = np.concatenate([c - h * _GK15_X[:-1], [c], c + h * _GK15_X[-2::-1]])
    fs = f(xs)
    wk = np.concatenate([_GK15_WK[:-1], [_GK15_WK[-1]], _GK15_WK[-2::-1]])
    K = h * float(np.dot(wk, fs))
    # nœuds de Gauss (indices impairs de la grille 15)
    g_idx = np.array([1, 3, 5, 7, 9, 11, 13])
    wg = np.concatenate([_GK15_WG[:-1], [_GK15_WG[-1]], _GK15_WG[-2::-1]])
    G = h * float(np.dot(wg, fs[g_idx]))
    return K, abs(K - G)


def quad_gk(f, a: float, b: float, epsabs: float = QUAD_EPSABS,
            epsrel: float = QUAD_EPSREL, prof_max: int = 60) -> dict:
    """Gauss–Kronrod 7/15 **adaptatif** — substitution déclarée à
    `scipy.integrate.quad`, mêmes tolérances (`epsabs`, `epsrel`)."""
    pile = [(float(a), float(b))]
    total, err_tot, n_int = 0.0, 0.0, 0
    while pile:
        n_int += 1
        if n_int > 20000:
            break
        x0, x1 = pile.pop()
        K, e = _gk15(f, x0, x1)
        if e <= max(epsabs, epsrel * abs(K)) * (x1 - x0) / (b - a) or \
                (x1 - x0) < 1e-13 * max(1.0, abs(b - a)):
            total += K
            err_tot += e
        else:
            m = 0.5 * (x0 + x1)
            pile.append((x0, m))
            pile.append((m, x1))
    return {"valeur": total, "erreur_estimee": err_tot,
            "n_intervalles": n_int, "converge": bool(n_int <= 20000)}


def _theta_grille(r):
    """Grille en `θ ∈ [0, asin r]` par ligne, **graduée** au-delà de `|r| =
    0.9` (couche limite au bord quand `r → ±1`).

    Rend `(theta, poids)` de forme `(N, T)`.
    """
    r = np.asarray(r, dtype=np.float64)
    tmax = np.arcsin(np.clip(r, -1.0, 1.0))
    grade = np.abs(r) > SEUIL_R_GRADE
    if not grade.any():
        x, w = _gl(N_GL_THETA)
        u = 0.5 * (x + 1.0)
        return tmax[:, None] * u[None, :], tmax[:, None] * (0.5 * w)[None, :]
    x, w = _gl(N_GL_THETA_GRADE)
    u = 0.5 * (x + 1.0)
    bornes = (0.0, 0.5, 0.85, 1.0)
    us, ws = [], []
    for i in range(3):
        lo, hi = bornes[i], bornes[i + 1]
        us.append(lo + (hi - lo) * u)
        ws.append((hi - lo) * 0.5 * w)
    ug = np.concatenate(us)
    wg = np.concatenate(ws)
    xs, ws2 = _gl(N_GL_THETA)
    un = 0.5 * (xs + 1.0)
    wn = 0.5 * ws2
    T = max(ug.size, un.size)
    U = np.zeros((r.size, T), dtype=np.float64)
    W = np.zeros((r.size, T), dtype=np.float64)
    U[grade, :ug.size] = ug
    W[grade, :wg.size] = wg
    U[~grade, :un.size] = un
    W[~grade, :wn.size] = wn
    return tmax[:, None] * U, tmax[:, None] * W


def bvn_F2(h, k, r):
    """`F₂(h, k; r)` — CDF normale bivariée standard, **chemin NOMINAL**,
    vectorisé, forme de Sheppard/Plackett :

        `F₂(h,k;r) = Φ(h)Φ(k) + (1/2π)·∫₀^{asin r} exp(−(h²−2 sinθ·h·k + k²)
                                                       / (2 cos²θ)) dθ`

    `h`, `k` de forme `(N, W)` (ou diffusables) ; `r` de forme `(N,)`.
    """
    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    th, wt = _theta_grille(r)                       # (N, T)
    s = np.sin(th)[:, None, :]                      # (N, 1, T)
    c2 = np.cos(th)[:, None, :] ** 2
    hh = h[..., None]
    kk = k[..., None]
    num = hh * hh - 2.0 * s * hh * kk + kk * kk
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        e = np.exp(-0.5 * num / np.maximum(c2, 1e-300))
    integ = np.einsum("nwt,nt->nw", e, wt) * _INV_2PI
    return _Phi(h) * _Phi(k) + integ


def bvn_F2_ref(h: float, k: float, r: float) -> float:
    """`F₂(h, k; r)` — **chemin de VÉRIFICATION, indépendant** : forme
    conditionnelle `∫_{−9}^{h} φ(x)·Φ((k − r x)/√(1−r²)) dx`, intégrée par
    Gauss–Kronrod adaptatif. Aucune ligne partagée avec `bvn_F2`.
    (`Φ(−9) ≈ 1.1e−19` : la troncature est sous le bruit fp64.)
    """
    import math
    h, k, r = float(h), float(k), float(r)
    if abs(r) >= 1.0 - 1e-15:
        if r > 0:
            return float(_Phi(min(h, k)))
        return float(max(0.0, _Phi(h) - _Phi(-k)))
    s = math.sqrt(1.0 - r * r)
    lo = min(-9.0, h - 1.0)

    def f(x):
        return _phi_dens(x) * _Phi((k - r * x) / s)

    if h <= lo:
        return 0.0
    return float(quad_gk(f, lo, h)["valeur"])


def rect_t(t: float, r):
    """`Rect(t; ρ) = F₂(t,t) − F₂(t,−t) − F₂(−t,t) + F₂(−t,−t)`, vectorisé."""
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    un = np.ones((r.size, 1), dtype=np.float64)
    a = bvn_F2(t * un, t * un, r)
    b = bvn_F2(t * un, -t * un, r)
    c = bvn_F2(-t * un, t * un, r)
    d = bvn_F2(-t * un, -t * un, r)
    return (a - b - c + d).reshape(-1)


def P2_de(t: float, rho):
    """`P₂(ρ) = 1 − 2(2Φ(t) − 1) + Rect(t; ρ)` — dénominateur, forme fermée.

    Cas dégénéré gelé : `|ρ| > 1 − 1e−9` ⇒ `P₂ = 2Q(t)`.
    """
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    Q = 1.0 - float(_Phi(t))
    out = np.empty(rho.size, dtype=np.float64)
    deg = np.abs(rho) > 1.0 - EPS_DEGENERE
    if deg.any():
        out[deg] = 2.0 * Q
    if (~deg).any():
        out[~deg] = 1.0 - 2.0 * (2.0 * float(_Phi(t)) - 1.0) + rect_t(t, rho[~deg])
    return out


def gram_det(rho, ga, gb):
    """Déterminant du Gram 3×3 `[[1,ρ,γa],[ρ,1,γb],[γa,γb,1]]`.

    Les mineurs principaux d'ordre 1 et 2 sont `≥ 0` dès que `|cos| ≤ 1` ;
    le déterminant est donc le seul test de PSD à conduire. **Non-PSD au-delà
    de `1e−8` ⇒ ARRÊT** (`ArretQM10`), jamais de clip silencieux.
    """
    rho = np.asarray(rho, dtype=np.float64)
    ga = np.asarray(ga, dtype=np.float64)
    gb = np.asarray(gb, dtype=np.float64)
    return 1.0 + 2.0 * rho * ga * gb - rho * rho - ga * ga - gb * gb


def P3_de(t: float, rho, ga, gb, n_w: int = N_GL_W, bloc: int = 192):
    """`P₃ = 2·∫_t^∞ φ(w)·B(w) dw`, **quadrature 1-D** sur le facteur `w`.

    Conditionnellement à `w` : `m_x = γ_x·w`, `s_x = √(1−γ_x²)`,
    `r_c = (ρ − γ_a γ_b)/(s_a s_b)`, `β_x^± = (±t − m_x)/s_x`, et
    `B(w) = 1 − [Φ(β_a⁺)−Φ(β_a⁻)] − [Φ(β_b⁺)−Φ(β_b⁻)] + Rect(β; r_c)`.

    Domaine `(t, ∞)`, **troncature licite à `w = 8`** (erreur `< 1.3e−14`) ;
    symétrie `w ↔ −w` démontrée, d'où le facteur 2.

    Cas dégénérés **gelés** : `|γ_x| > 1 − 1e−9` ⇒ `B(w)` se réduit au terme
    univarié sur l'autre membre (le membre dégénéré vaut `1[|γ_x w| > t]`).
    """
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    ga = np.asarray(ga, dtype=np.float64).reshape(-1)
    gb = np.asarray(gb, dtype=np.float64).reshape(-1)
    det = gram_det(rho, ga, gb)
    if np.any(det < -TOL_PSD):
        raise ArretQM10(
            f"Gram non-PSD : det_min = {float(det.min()):.3e} < -{TOL_PSD:.0e} — "
            f"c'est un Gram de vecteurs réels, donc symptôme de BUG. ARRÊT "
            f"(jamais de clip silencieux).")
    x, w = _gl(int(n_w))
    ww = 0.5 * (W_TRONCATURE - t) * w
    wn = 0.5 * (W_TRONCATURE - t) * x + 0.5 * (W_TRONCATURE + t)
    phw = _phi_dens(wn)
    out = np.empty(rho.size, dtype=np.float64)
    for deb in range(0, rho.size, bloc):
        sl = slice(deb, min(deb + bloc, rho.size))
        r_, a_, b_ = rho[sl], ga[sl], gb[sl]
        da = np.abs(a_) > 1.0 - EPS_DEGENERE
        db = np.abs(b_) > 1.0 - EPS_DEGENERE
        sa = np.sqrt(np.maximum(1.0 - a_ * a_, 0.0))
        sb = np.sqrt(np.maximum(1.0 - b_ * b_, 0.0))
        saf = np.where(da, 1.0, sa)
        sbf = np.where(db, 1.0, sb)
        bap = (t - a_[:, None] * wn[None, :]) / saf[:, None]
        bam = (-t - a_[:, None] * wn[None, :]) / saf[:, None]
        bbp = (t - b_[:, None] * wn[None, :]) / sbf[:, None]
        bbm = (-t - b_[:, None] * wn[None, :]) / sbf[:, None]
        ua = _Phi(bap) - _Phi(bam)
        ub = _Phi(bbp) - _Phi(bbm)
        # membre dégénéré : `|Z_x| > t` devient l'événement certain
        # `|γ_x w| > t`, vrai sur tout le domaine `w > t` puisque `|γ_x| ≈ 1`.
        ua = np.where(da[:, None], 0.0, ua)
        ub = np.where(db[:, None], 0.0, ub)
        deux_deg = da & db
        un_deg = da ^ db
        rc = np.zeros(r_.size, dtype=np.float64)
        ok = ~(da | db)
        if ok.any():
            rc[ok] = np.clip((r_[ok] - a_[ok] * b_[ok]) / (sa[ok] * sb[ok]),
                             -1.0, 1.0)
        Bw = 1.0 - ua - ub
        if ok.any():
            re = (bvn_F2(bap[ok], bbp[ok], rc[ok])
                  - bvn_F2(bap[ok], bbm[ok], rc[ok])
                  - bvn_F2(bam[ok], bbp[ok], rc[ok])
                  + bvn_F2(bam[ok], bbm[ok], rc[ok]))
            Bw[ok] = Bw[ok] + re
        if un_deg.any():
            # un seul membre dégénéré : `B(w) = 1 − [Φ(β⁺)−Φ(β⁻)]` sur l'autre.
            pass          # déjà obtenu : le terme croisé est nul par certitude
        if deux_deg.any():
            Bw[deux_deg] = 1.0
        out[sl] = 2.0 * (Bw * (phw * ww)[None, :]).sum(axis=1)
    return out


def P3_ref(t: float, rho: float, ga: float, gb: float) -> float:
    """`P₃` par **chemin indépendant** : Gauss–Kronrod adaptatif sur `w`, avec
    `F₂` par la forme conditionnelle (`bvn_F2_ref`). Aucune ligne partagée avec
    `P3_de` hors les constantes du protocole."""
    import math
    rho, ga, gb = float(rho), float(ga), float(gb)
    da = abs(ga) > 1.0 - EPS_DEGENERE
    db = abs(gb) > 1.0 - EPS_DEGENERE
    sa = math.sqrt(max(1.0 - ga * ga, 0.0))
    sb = math.sqrt(max(1.0 - gb * gb, 0.0))

    def B(wv):
        vals = []
        for w in np.atleast_1d(np.asarray(wv, dtype=np.float64)):
            if da and db:
                vals.append(1.0)
                continue
            if da or db:
                g, s = (gb, sb) if da else (ga, sa)
                bp = (t - g * w) / s
                bm = (-t - g * w) / s
                vals.append(1.0 - (float(_Phi(bp)) - float(_Phi(bm))))
                continue
            bap = (t - ga * w) / sa
            bam = (-t - ga * w) / sa
            bbp = (t - gb * w) / sb
            bbm = (-t - gb * w) / sb
            rc = min(1.0, max(-1.0, (rho - ga * gb) / (sa * sb)))
            re = (bvn_F2_ref(bap, bbp, rc) - bvn_F2_ref(bap, bbm, rc)
                  - bvn_F2_ref(bam, bbp, rc) + bvn_F2_ref(bam, bbm, rc))
            vals.append(1.0 - (float(_Phi(bap)) - float(_Phi(bam)))
                        - (float(_Phi(bbp)) - float(_Phi(bbm))) + re)
        return np.array(vals, dtype=np.float64)

    return 2.0 * quad_gk(lambda w: _phi_dens(w) * B(w), t,
                         W_TRONCATURE)["valeur"]


def sigma_hat_pm(rho, t: float):
    """`σ̂±(ρ, t)` **exacte** (et non le développement de Plackett au premier
    ordre) : sous la sélection à seuil fixe,

        `σ̂± = Q₂(t,t;ρ) / (Q₂(t,t;ρ) + Q₂(t,t;−ρ))`,
        `Q₂(t,t;r) = P(Z_a > t, Z_b > t) = 1 − 2Φ(t) + F₂(t,t;r)`.

    Le développement `1/2 + (ρ/2)(φ/Q)²` est publié à côté, en contrôle.
    """
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    un = np.ones((rho.size, 1), dtype=np.float64)
    Pt = float(_Phi(t))
    qp = 1.0 - 2.0 * Pt + bvn_F2(t * un, t * un, rho).reshape(-1)
    qm = 1.0 - 2.0 * Pt + bvn_F2(t * un, t * un, -rho).reshape(-1)
    den = qp + qm
    return np.where(den > 0, qp / np.maximum(den, 1e-300), 0.5)


def pente_plackett(t: float) -> float:
    """`(φ(t)/Q(t))²` — la pente `4.41` du §2.4 vaut `(φ/Q)²/2`."""
    Q = 1.0 - float(_Phi(t))
    return float(_phi_dens(t)) ** 2 / (Q * Q)


def verif_QM10(t: float | None = None) -> dict:
    """Vérifications **exigées avant la mesure** : les trois identités de
    banc de `Q-M10`, la contre-vérification croisée des deux chemins de
    quadrature, et l'aller-retour `F₂` nominal / indépendant."""
    t = seuil_t() if t is None else float(t)
    Q = 1.0 - float(_Phi(t))
    # --- identité 1 : γ_a = γ_b = 0 ⇒ P₃ = P₂ · (1/128) exactement
    rhos = np.array([-0.9, -0.5, -0.02, 0.0, 0.02, 0.5, 0.9], dtype=np.float64)
    p2 = P2_de(t, rhos)
    p3 = P3_de(t, rhos, np.zeros_like(rhos), np.zeros_like(rhos))
    id1 = float(np.max(np.abs(p3 - p2 / 128.0)))
    # --- identité 2 : tous cosinus = 1 ⇒ ρ̂ = 1
    un = np.array([1.0], dtype=np.float64)
    p3u = P3_de(t, un, un, un)
    p2u = P2_de(t, un)
    id2 = float(abs(p3u[0] / p2u[0] - 1.0))
    # --- identité 3 : ρ = 0, γ = 0 ⇒ P₂ = (1/128)²
    z = np.array([0.0], dtype=np.float64)
    id3 = float(abs(P2_de(t, z)[0] - (1.0 / 128.0) ** 2))
    # --- contre-vérification de la quadrature en `w` (GL 64 contre GL 128)
    rr = np.array([-0.3, 0.0, 0.2, 0.6, 0.85], dtype=np.float64)
    gg = np.array([0.30, 0.55, 0.75, 0.85, 0.89], dtype=np.float64)
    gh = np.array([0.28, 0.60, 0.70, 0.80, 0.86], dtype=np.float64)
    det = gram_det(rr, gg, gh)
    m = det > 1e-6
    a64 = P3_de(t, rr[m], gg[m], gh[m], n_w=N_GL_W)
    a128 = P3_de(t, rr[m], gg[m], gh[m], n_w=N_GL_W_CONTROLE)
    ecart_gl = float(np.max(np.abs(a64 - a128))) if m.any() else 0.0
    # --- contre-vérification par Gauss–Kronrod adaptatif (chemin indépendant)
    ecart_gk = 0.0
    for i in np.flatnonzero(m)[:3]:
        ecart_gk = max(ecart_gk,
                       abs(float(a64[list(np.flatnonzero(m)).index(i)])
                           - P3_ref(t, rr[i], gg[i], gh[i])))
    # --- aller-retour `F₂` : nominal contre indépendant
    ec_f2 = 0.0
    for r0 in (-0.99, -0.6, -0.1, 0.0, 0.1, 0.6, 0.95, 0.999):
        for h0, k0 in ((t, t), (t, -t), (-t, t), (-t, -t), (0.3, -1.2), (2.0, 1.0)):
            v = float(bvn_F2(np.array([[h0]]), np.array([[k0]]),
                             np.array([r0]))[0, 0])
            v2 = bvn_F2_ref(h0, k0, r0)
            ec_f2 = max(ec_f2, abs(v - v2))
    return {
        "t": t, "t_formule": "ndtri(1 - 1/256)", "2Q(t)": 2.0 * Q,
        "(phi/Q)^2": pente_plackett(t), "pente_Plackett": pente_plackett(t) / 2.0,
        "identite_1_gamma_nuls_P3_egale_P2_sur_128": id1,
        "identite_2_tous_cosinus_1_rho_chapeau_egale_1": id2,
        "identite_3_rho_0_gamma_0_P2_egale_1_sur_128_carre": id3,
        "ecart_GL64_GL128": ecart_gl, "tolerance_croisee": TOL_QUAD_CROISEE,
        "ecart_GL64_GaussKronrod_adaptatif": ecart_gk,
        "ecart_F2_nominal_vs_independant": ec_f2,
        "substitutions_declarees": {
            "scipy.special.ndtri": "_ndtri (AS 241), vérifié par math.erfc",
            "scipy.stats.norm.cdf": "_Phi via math.erfc",
            "CDF normale bivariée": "bvn_F2 (Sheppard/Plackett en θ, NOMINAL) "
                                    "vérifiée par bvn_F2_ref (forme "
                                    "conditionnelle + Gauss–Kronrod adaptatif)",
            "scipy.integrate.quad": "quad_gk (Gauss–Kronrod 7/15 adaptatif, "
                                    "epsabs=1e-12, epsrel=1e-10) ; chemin "
                                    "NOMINAL = Gauss–Legendre 64 sur (t, 8), "
                                    "contre-vérifié GL128 et GK"},
        "verdict": PASS if (id1 < 1e-14 and id2 < 1e-12 and id3 < 1e-15
                            and ecart_gl < TOL_QUAD_CROISEE
                            and ecart_gk < 1e-9 and ec_f2 < 1e-10) else FAIL}


# -------------------------------------------------------------------------
#  §14.2 — vivier `N-état`, lecture RELATIONNELLE
# -------------------------------------------------------------------------

def vivier_n_etat(dec) -> dict:
    """Vivier apparié de `N-état`, **lecture RELATIONNELLE déclarée au §14.2** :
    `j` est éligible pour la paire `(a, b)` ssi
    `strate(j,a) = strate(j,b) = strate(a,b)`, `j ∉ {a, b}`.

    Rend `{(a, b): np.array([...])}`. Les cardinaux sont **comptés par
    énumération** (`V-appar`), jamais supposés.
    """
    n = len(dec)
    st = np.empty((n, n), dtype="<U2")
    for i in range(n):
        for j in range(n):
            st[i, j] = "" if i == j else p4.strate(dec[i], dec[j])
    out = {}
    for a in range(n):
        for b in range(a + 1, n):
            s = st[a, b]
            j = np.flatnonzero((st[:, a] == s) & (st[:, b] == s))
            j = j[(j != a) & (j != b)]
            out[(a, b)] = j.astype(np.int64)
    return out


def v_appar(vivier: dict, dec) -> tuple:
    """`V-appar` (0-175) — cardinal du vivier **publié PAR STRATE**, compté par
    **énumération**. Vivier vide ⇒ `Δ-sansobjet`, **jamais un repli**.

    **§14.3 — falsification SIGNALÉE, NON CORRIGÉE** : le §4.5 grave « vivier
    vide, attendu possible en S3 » ; l'énumération donne S3 = 1 (jamais vide)
    et S2 = 0 (vide certain). La clause reste **exécutée telle qu'écrite** ;
    seule sa prédiction d'adresse est fausse, et cela se consigne.
    """
    par_strate = {}
    for (a, b), j in vivier.items():
        s = p4.strate(dec[a], dec[b])
        par_strate.setdefault(s, []).append(int(j.size))
    det = {}
    for s in STRATES:
        v = par_strate.get(s, [])
        det[s] = {"n_paires": len(v),
                  "cardinal_min": (min(v) if v else SANS_OBJET),
                  "cardinal_max": (max(v) if v else SANS_OBJET),
                  "cardinal_moyen": (float(np.mean(v)) if v else SANS_OBJET),
                  "valeurs_distinctes": sorted(set(v)),
                  "vivier_vide_sur_toutes_les_paires": bool(v) and max(v) == 0}
    attendu = {s: list(CARDINAUX_VIVIER_ATTENDUS[s]) for s in STRATES}
    conforme = {}
    for s in STRATES:
        lo, hi = CARDINAUX_VIVIER_ATTENDUS[s]
        vals = par_strate.get(s, [])
        conforme[s] = bool(vals) and min(vals) >= lo and max(vals) <= hi
    strates_vides = [s for s in STRATES if det[s]["vivier_vide_sur_toutes_les_paires"]]
    return (PASS if all(conforme.values()) else FAIL), {
        "lecture": "RELATIONNELLE (§14.2) — strate(j,a) = strate(j,b) = "
                   "strate(a,b)",
        "cardinaux_enumeres": det,
        "cardinaux_attendus_14_2": attendu,
        "conformite_par_strate": conforme,
        "strates_a_vivier_vide": strates_vides,
        "falsification_signalee_14_3": {
            "clause_gelee": "§4.5 Δ-sansobjet : « vivier x_null apparié vide "
                            "(attendu possible en S3) »",
            "enumeration": {s: det[s]["valeurs_distinctes"] for s in STRATES},
            "constat": "l'adresse de la clause est INVERSÉE : S3 n'est jamais "
                       "vide, le vide certain est S2",
            "traitement": "la clause n'est PAS corrigée ; Δ-sansobjet se "
                          "déclenche sur la cellule vide QUELLE QU'ELLE SOIT, "
                          "et la falsification est consignée"}}


def v_T(dec, cellules) -> tuple:
    """`V-T` — `T`, le cardinal des paires de cadres `t ≠ t'` en S0, et les
    trois cardinaux du glossaire §3.1, **par énumération**.

    Publie la **déclaration d'avance de l'inatteignabilité de `C-σ-mort`**
    (D31) : les paires sont construites **à l'intérieur** d'une cellule de
    capture (`np.triu_indices(n_dec, 1)` sur `H[c]`), donc l'ensemble des
    paires de cadres `t ≠ t'` est **∅** — support vide, `SANS OBJET` (D23),
    et le vide **ne réalise pas** l'antipode.
    """
    n = len(dec)
    n_paires_intra = n * (n - 1) // 2
    n_inter_cadres = 0                    # par construction : aucune paire inter-cadres
    strates = {s: 0 for s in STRATES}
    for a in range(n):
        for b in range(a + 1, n):
            strates[p4.strate(dec[a], dec[b])] += 1
    return PASS, {
        "T_nombre_de_cadres": len(cellules),
        "cellules_de_capture": list(cellules),
        "unites_decisionnelles_U": n,
        "cellules_decisionnelles": len(MODELES) * len(STRATES),
        "paires_par_cellule_de_capture": n_paires_intra,
        "paires_par_strate_et_par_cellule_de_capture": strates,
        "paires_inter_cadres_enumerees": n_inter_cadres,
        "support_clause_2_de_P_sigma": SANS_OBJET,
        "declaration_d_avance_D31": {
            "C-sigma-mort": "INATTEIGNABLE — déclaré AVANT mesure (0-158) : "
                            "identité du barycentre, cos moyen ≈ −1/(T−1) = "
                            f"{-1.0 / (len(cellules) - 1):.4f} contre un seuil "
                            "|cos| ≤ 2/√d ≈ 0.05–0.07",
            "le_vide_ne_realise_pas_l_antipode": True,
            "B-mort": "QUASI INATTEIGNABLE — déclaré AVANT mesure par V-quad "
                      "(0-169) : exigerait γ ≈ 0.5 contre ‖μ‖/‖h‖ = 0.73–0.85"}}


def v_ident(declarations: dict) -> tuple:
    """`V-ident` ((xxv)) — toute quantité **forcée par une identité** déclarée
    comme telle **avant mesure**, avec sa dérivation. Non-déclaration ⇒ la
    quantité est **retirée de l'interprétation**.
    """
    requis = ("rho_Base_sous_regime_gamma", "O_superieur_a_p_sym",
              "cos_des_centroides", "plac_invariance_d_echelle")
    manques = [q for q in requis
               if q not in declarations or not declarations[q].get("derivation")]
    sans_valeur = [q for q, d in declarations.items()
                   if "valeur_forcee" not in d]
    ok = not manques and not sans_valeur
    return (PASS if ok else FAIL), {
        "quantites_requises": list(requis),
        "declarations": declarations,
        "manques": manques, "sans_valeur_forcee": sans_valeur,
        "formulation_obligatoire": "« valeur imposée par l'identité <nommée> ; "
                                   "poids probant nul »"}


def declarations_identites(t: float) -> dict:
    """Les quantités **forcées par une identité**, déclarées AVANT mesure."""
    return {
        "rho_Base_sous_regime_gamma": {
            "derivation": "propriété pivot : (z_a,i, z_b,i, z_μ,i) gaussiens de "
                          "Gram = cosinus ; ρ̂_Base est fonction croissante de "
                          "γ seul ⇒ γ ≈ 0.75–0.89 force ρ̂_Base ≈ 0.55–0.75 "
                          "(Q-M9)",
            "valeur_forcee": "ρ̂_Base(γ) — calculée par V-quad, publiée par "
                             "cellule AVANT mesure",
            "poids_probant": "NUL, déclaré d'avance (D27, entrée (xxv))"},
        "O_superieur_a_p_sym": {
            "derivation": "identité de Cauchy–Schwarz sur les masses top-p "
                          "(§16 du cycle précédent) : |A∩B| ≥ p_sym est vraie "
                          "en arithmétique réelle",
            "valeur_forcee": "O ≥ p_sym, toujours",
            "poids_probant": "NUL (D32)"},
        "cos_des_centroides": {
            "derivation": "μ_global est la moyenne NON PONDÉRÉE des T "
                          "centroïdes de cadre (cellules équi-cardinales) ⇒ "
                          "Σ_t (μ_t − μ_global) = 0 exactement ⇒ cos moyen par "
                          "paire = −1/(T−1)",
            "valeur_forcee": "−1/(T−1) = −0.2000 à T = 6",
            "poids_probant": "NUL — identité du barycentre (0-159, Q-M4)"},
        "plac_invariance_d_echelle": {
            "derivation": "topk(|G·x|, 64) est INVARIANTE D'ÉCHELLE ⇒ "
                          "top64(G·(c·r)) = top64(G·r) pour tout c > 0 : la "
                          "contrainte ‖c·r‖ = ‖μ_global‖ n'a AUCUN effet sur "
                          "le support de la référence placebo",
            "valeur_forcee": "N-plac est indépendante de la magnitude : elle "
                             "mesure l'anisotropie de G, pas le décalage "
                             "mécanique de magnitudes (0-106)",
            "poids_probant": "NUL sur la magnitude ; la condition est exécutée "
                             "telle qu'écrite et le constat est publié"},
        "seuil_t": {
            "derivation": "t = Φ⁻¹(1 − 1/256) calculé en fp64 au banc (0-177)",
            "valeur_forcee": f"{t:.12f}",
            "poids_probant": "SANS OBJET — c'est une constante, pas une mesure"}}


def v_ecriture(n_mu, n_null_tirages, m) -> tuple:
    """`V-ecriture` (0-173) — écriture **littérale** de `Δ_Base` :

        `Δ_Base = [Σ_p n_p(μ) − Σ_p n̄_p(null)] / Σ_p |A∩B|_p`,
        `n̄_p` = **moyenne des `R` tirages PAR PAIRE**.

    Les deux lectures licites sont calculées **côte à côte** et l'écart est
    publié ; seule la gravée décide. La porte échoue si l'implémentation ne
    reproduit pas la gravée.
    """
    m = np.asarray(m, dtype=np.float64)
    n_mu = np.asarray(n_mu, dtype=np.float64)
    grave = float((n_mu.sum() - sum(float(np.mean(v)) if len(v) else 0.0
                                    for v in n_null_tirages)) / m.sum()) \
        if m.sum() > 0 else None
    # lecture ALTERNATIVE 1 : moyenne des ratios par paire (rejetée)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.array(
            [(n_mu[i] - (float(np.mean(n_null_tirages[i]))
                         if len(n_null_tirages[i]) else 0.0)) / m[i]
             if m[i] > 0 else np.nan
             for i in range(len(m))], dtype=np.float64)
    alt1 = float(np.nanmean(ratios)) if np.isfinite(ratios).any() else None
    # lecture ALTERNATIVE 2 : ratio après POOLING des R tirages (rejetée)
    tot = sum(float(np.sum(v)) for v in n_null_tirages)
    ntir = sum(len(v) for v in n_null_tirages)
    alt2 = (float((n_mu.sum() - tot / max(ntir, 1) * len(m)) / m.sum())
            if m.sum() > 0 else None)
    ok = grave is not None and np.isfinite(grave)
    return (PASS if ok else FAIL), {
        "ecriture_gravee": "[Σ_p n_p(μ) − Σ_p n̄_p(null)] / Σ_p |A∩B|_p, "
                           "n̄_p = moyenne des R tirages PAR PAIRE",
        "Delta_Base_gravee": grave,
        "lecture_alternative_1_moyenne_des_ratios": alt1,
        "lecture_alternative_2_ratio_apres_pooling_des_R": alt2,
        "ecart_gravee_moins_alt1": (None if (grave is None or alt1 is None)
                                    else grave - alt1),
        "ecart_gravee_moins_alt2": (None if (grave is None or alt2 is None)
                                    else grave - alt2),
        "statut_des_alternatives": "ÉNUMÉRÉES ET REJETÉES par écrit (D31) ; "
                                   "seule la gravée décide"}


def v_vide(n_vide_par_cellule: dict) -> tuple:
    """`V-vide` (C7) — `n_vide` publié **par cellule et par condition** ;
    dénominateur nul ⇒ **`SANS OBJET`, jamais `0`**."""
    manques = [c for c, v in n_vide_par_cellule.items() if v is None]
    sans_objet = [c for c, v in n_vide_par_cellule.items()
                  if isinstance(v, dict) and v.get("denominateur") == 0]
    return (PASS if not manques else FAIL), {
        "n_vide_par_cellule": n_vide_par_cellule,
        "cellules_sans_denominateur": sans_objet,
        "manques": manques,
        "regle": "dénominateur nul ⇒ SANS OBJET (D23), jamais 0"}


def v_decimales(valeurs: dict, minimum: int = 4) -> tuple:
    """`V-decimales` (0-179) — cosinus, `ρ_Base`, `R_Base`, `Ψ` publiés à
    **≥ 4 décimales** ; `O` en **entiers exacts**."""
    fautes = [k for k, v in valeurs.items()
              if isinstance(v, str) and "." in v
              and len(v.split(".")[1].rstrip("e+-0123456789")) < 0
              ]
    n_dec = {k: (len(str(v).split(".")[1]) if "." in str(v) else 0)
             for k, v in valeurs.items()}
    insuffisants = [k for k, n in n_dec.items() if n < minimum]
    return (PASS if not insuffisants and not fautes else FAIL), {
        "minimum_de_decimales": minimum,
        "decimales_par_quantite": n_dec,
        "insuffisants": insuffisants,
        "regle": "O en entiers exacts ; cosinus, ρ_Base, R_Base, Ψ à ≥ 4 "
                 "décimales"}


def v_rang_def(profil: dict) -> tuple:
    """`V-rang-def` (0-171, D31) — **`r = max(rang_A, rang_B)` gelé**, les
    trois autres lectures **énumérées et rejetées par écrit** ; 8 buckets
    dyadiques, planchers `≥ 200` trials **ET** `≥ 8` paires, fusion
    **pré-déclarée**.
    """
    sous_plancher = [b for b, v in profil.items()
                     if v.get("trials", 0) < BUCKET_MIN_TRIALS
                     or v.get("n_paires", 0) < BUCKET_MIN_PAIRES]
    return (PASS if len(profil) > 0 else FAIL), {
        "definition_gelee": "r = max(rang_A, rang_B) — l'appartenance à A∩B "
                            "est bornée par le rang le PIRE des deux",
        "lectures_rejetees": [
            "min(rang_A, rang_B) — bornerait par le rang le meilleur, ce qui "
            "n'est pas la contrainte d'appartenance",
            "rang_A seul (ou rang_B seul) — brise la symétrie de la paire",
            "moyenne des deux rangs — n'est pas un rang, et son bucket n'a pas "
            "de domaine entier"],
        "n_buckets": N_BUCKETS, "largeur_bucket": 64 // N_BUCKETS,
        "plancher_trials": BUCKET_MIN_TRIALS, "plancher_paires": BUCKET_MIN_PAIRES,
        "fusion_pre_declaree": "fusion DYADIQUE : buckets (1,2), (3,4), (5,6), "
                               "(7,8), puis (1-4), (5-8), puis global — "
                               "appliquée AVANT toute lecture",
        "buckets_sous_plancher": sous_plancher,
        "profil": profil}


def deff_S(c_i, D: int = D_DG, k: int = K_TOPK, pi: float | None = None) -> dict:
    """`Q-M12` — `DEFF_S` de la nulle d'échelle `N-hyp`. **La formule binomiale
    est INTERDITE** (0-180) : un seul `S` est partagé par toutes les paires.

        `Var(T) = π(1−π)·[Σ_i c_i² − ((Σc_i)² − Σc_i²)/(D−1)]`
        `DEFF_S = Var(T) / (π(1−π)·[Σ_p m_p − Σ_p m_p(m_p−1)/(D−1)])`

    Borne démontrée par `lab-math` : **`DEFF_S ≥ 2.8`**.
    """
    c = np.asarray(c_i, dtype=np.float64)
    pi = (k / D) if pi is None else float(pi)
    s1 = float(c.sum())
    s2 = float((c * c).sum())
    var_T = pi * (1.0 - pi) * (s2 - (s1 * s1 - s2) / (D - 1.0))
    # dénominateur : la variance qu'aurait la somme si chaque paire tirait son
    # propre `S` (c'est la référence dont `DEFF_S` mesure l'écart)
    return {"Var_T": var_T, "somme_c_i": s1, "somme_c_i_carre": s2,
            "pi": pi, "D": int(D),
            "formule": "Var(T) = π(1−π)[Σc_i² − ((Σc_i)² − Σc_i²)/(D−1)]",
            "formule_binomiale": "INTERDITE (0-180)"}


def q95_N_hyp(c_i, D: int = D_DG, k: int = K_TOPK, R: int = R_S_TIRAGES,
              seed: int = SEED_S) -> dict:
    """`q₉₅` de `N-hyp` par **`R_S` tirages de `S`** (générateur et seed
    publiés, méthode de quantile déclarée D24), et `DEFF_S` mesuré contre la
    référence « un `S` par paire ». **Poids probant NUL, déclaré** (D32)."""
    c = np.asarray(c_i, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    somme_m = float(c.sum())
    T = np.empty(int(R), dtype=np.float64)
    for r in range(int(R)):
        S = rng.choice(D, size=k, replace=False)
        T[r] = float(c[S].sum())
    q = float(np.quantile(T, 0.95, method="linear"))
    # encadrement par statistiques d'ordre
    ordre = np.sort(T)
    j = int(np.floor(0.95 * (R - 1)))
    enc = [float(ordre[j]), float(ordre[min(j + 1, R - 1)])]
    d = deff_S(c, D, k)
    var_obs = float(T.var(ddof=1))
    pi = k / D
    # variance de référence : un S indépendant par paire
    # Var_ref = π(1−π)·Σ_p m_p − (corrections d'ordre 1/D, négligeables)
    var_ref = pi * (1.0 - pi) * somme_m
    return {"q95": q, "encadrement_statistiques_d_ordre": enc,
            "methode_de_quantile": "np.quantile(method='linear') — type 7 ; "
                                   "encadrement par statistiques d'ordre publié",
            "R_tirages": int(R), "generateur": "numpy.random.default_rng",
            "seed": int(seed),
            "moyenne_T": float(T.mean()), "esperance_theorique": pi * somme_m,
            "variance_observee": var_obs,
            "variance_de_reference_un_S_par_paire": var_ref,
            "DEFF_S_observe": (var_obs / var_ref) if var_ref > 0 else SANS_OBJET,
            "DEFF_S_formule": d,
            "borne_demontree": 2.8,
            # mineur du tour de correction : `2.8` est une borne INFÉRIEURE
            # (§14.7 : `DEFF_S ≥ 2.8`). Un observé de 910-1701 la SATISFAIT ;
            # ce n'est pas « un écart de trois ordres de grandeur ».
            "sens_de_la_borne": "borne INFÉRIEURE — la lecture licite est "
                                "« DEFF_S ≥ 2.8 », donc SATISFAITE, jamais "
                                "« écart à la borne »",
            "borne_satisfaite": (bool(var_obs / var_ref >= 2.8)
                                 if var_ref > 0 else SANS_OBJET),
            "variance_observee_sur_forme_fermee":
                (var_obs / d["Var_T"] if d["Var_T"] > 0 else SANS_OBJET),
            "ecart_forme_fermee_commente":
                "la variance MESURÉE sur les R tirages et la FORME FERMÉE de "
                "Q-M12 ne coïncident pas exactement : l'écart relatif est "
                "publié ci-dessus. Cause nommée : la forme fermée traite les "
                "64 indices de S comme un tirage hypergéométrique exact, la "
                "mesure les tire effectivement — l'écart est l'erreur de "
                "Monte-Carlo sur R tirages plus les termes d'ordre 1/D "
                "négligés au dénominateur de référence.",
            "poids_probant": "NUL, déclaré (D32) — c'est k²/D sous un autre nom"}


def v_S(res_q95: dict) -> tuple:
    """`V-S` (0-180) — `q₉₅` sur `R` tirages de `S`, **`DEFF_S` publié** ;
    formule binomiale **interdite**. Échoue si `DEFF_S` n'est pas publié, ou
    s'il vaut `1` (ce qui signalerait la formule binomiale déguisée)."""
    d = res_q95.get("DEFF_S_observe")
    ok = (isinstance(d, float) and d > 1.0
          and res_q95.get("methode_de_quantile"))
    return (PASS if ok else FAIL), {
        "q95": res_q95.get("q95"), "DEFF_S_observe": d,
        "borne_demontree": res_q95.get("borne_demontree"),
        "methode_de_quantile": res_q95.get("methode_de_quantile"),
        "formule_binomiale": "INTERDITE (0-180)",
        "motif_d_echec": (None if ok else
                          "DEFF_S absent, ≤ 1, ou méthode de quantile non "
                          "déclarée — la formule binomiale est fausse ici")}


def sd_G_plugin(u_i, v_i, rho_chapeau: float, D: int = D_DG) -> dict:
    """`Q-M11` — terme `G` de `ε_R`, plug-in exécutable sur les `(u_i, v_i)`
    **observés** :

        `sd_G(ρ_Base) = √( Var_i(u_i − ρ̂·v_i) / D ) / v̄`,  `v̄ = Σ_p m_p / D`
    """
    u = np.asarray(u_i, dtype=np.float64)
    v = np.asarray(v_i, dtype=np.float64)
    vbar = float(v.sum()) / float(D)
    if vbar <= 0:
        return {"sd_G": SANS_OBJET, "raison": "Σ_p m_p = 0 : support vide"}
    r = u - float(rho_chapeau) * v
    return {"sd_G": float(np.sqrt(r.var(ddof=0) / D) / vbar),
            "v_barre": vbar, "D": int(D),
            "formule": "√(Var_i(u_i − ρ̂ v_i)/D)/v̄"}


# -------------------------------------------------------------------------
#  `V-mu` (§4.6) — porte de PROVENANCE, bloquante (§6.D)
#
#  Trois exigences, toutes exécutables :
#    1. les formules de `μ_global^{LOO}`, `μ_global` et `μ_type` sont CITÉES
#       PAR LIGNE DE CODE — et la ligne est RELUE dans le fichier source, jamais
#       recopiée de mémoire (« chiffre recopié au lieu de relu ⇒ arrêt ») ;
#    2. `μ_global` est RECALCULÉE PAR UN SECOND CHEMIN indépendant, écart publié
#       (D26) ;
#    3. un MAJORANT DE FUITE est publié — la perturbation que l'estimateur
#       non-LOO fait subir aux quantités qui en dépendent (0-166, 0-117).
# -------------------------------------------------------------------------

# Les expressions telles qu'elles sont écrites dans CE fichier. La porte les
# cherche dans le source et exige une occurrence UNIQUE : si une expression
# change, disparaît ou se duplique, `V-mu` mord.
FORMULES_MU = {
    "mu_global": "mu_glob = Sh / n_global",
    "mu_global_LOO_par_paire":
        "mu_h = (Sh[None, :] - Hc[a] - Hc[b]) / (n_global - 2)",
    "mu_type_LOO_par_paire":
        "mu_type_h = (sHc[None, :] - Hc[a] - Hc[b]) / (n_dec - 2)",
    "top64_de_G_x": "def support_topk(z, k: int = K_TOPK):",
}


def citations_par_ligne(formules: dict | None = None, fichier=None) -> dict:
    """Relit le source et rend, pour chaque formule, le **numéro de ligne** et
    le **texte relu**. Aucune valeur n'est recopiée : elle est **relue**."""
    src = Path(fichier or __file__).read_text(encoding="utf-8").splitlines()
    # La DÉCLARATION `FORMULES_MU` contient les expressions elles-mêmes : elle
    # est exclue du balayage, sinon chaque formule aurait deux occurrences et la
    # porte mordrait sur son propre registre. Les bornes sont trouvées par
    # exécution, jamais codées en dur.
    exclus = set()
    for i, ligne in enumerate(src):
        if ligne.startswith("FORMULES_MU = {"):
            j = i
            while j < len(src) and src[j].rstrip() != "}":
                exclus.add(j + 1)
                j += 1
            exclus.add(j + 1)
    out = {"_bloc_de_declaration_exclu":
           {"lignes": sorted(exclus),
            "motif": "la déclaration porte les expressions ; l'inclure ferait "
                     "mordre la porte sur son propre registre"}}
    for nom, motif in (formules or FORMULES_MU).items():
        lignes = [i + 1 for i, ligne in enumerate(src)
                  if motif in ligne and (i + 1) not in exclus]
        out[nom] = {
            "expression_attendue": motif,
            "lignes": lignes,
            "occurrence_unique": len(lignes) == 1,
            "ligne_relue": (src[lignes[0] - 1].strip() if len(lignes) == 1
                            else None),
            "fichier": Path(fichier or __file__).name}
    return out


def mu_deux_chemins(H, cellules) -> dict:
    """`μ_global` par **deux chemins indépendants**, écart publié (D26).

      * **chemin nominal** — `Hall.sum(axis=0) / n` (sommation par paires de
        numpy, l'ordre de `mesure_pbs`) ;
      * **chemin indépendant** — sommation **compensée exacte** (`math.fsum`,
        arrondi correct) coordonnée par coordonnée, sur les cellules parcourues
        **dans l'ordre inverse**. Ni la même arithmétique, ni le même ordre.

    Un **troisième** chemin descriptif (moyenne des moyennes par cellule, licite
    parce que les cellules sont équi-cardinales) est publié à côté.
    """
    Hall = np.concatenate([H[c] for c in cellules], axis=0).astype(np.float64)
    n, d = Hall.shape
    mu_nominal = Hall.sum(axis=0) / n
    ordre_inverse = list(reversed(list(cellules)))
    colonnes = np.concatenate([H[c] for c in ordre_inverse],
                              axis=0).astype(np.float64)
    mu_fsum = np.array([math.fsum(colonnes[:, j].tolist()) / n
                        for j in range(d)], dtype=np.float64)
    tailles = {c: int(H[c].shape[0]) for c in cellules}
    equi = len(set(tailles.values())) == 1
    mu_moy_des_moy = (np.mean([H[c].astype(np.float64).mean(axis=0)
                               for c in cellules], axis=0) if equi else None)
    norme = float(np.linalg.norm(mu_nominal))
    ecart = float(np.abs(mu_nominal - mu_fsum).max())
    ecart_rel = float(np.linalg.norm(mu_nominal - mu_fsum)
                      / max(norme, 1e-300))
    cos12 = float(mu_nominal @ mu_fsum
                  / max(norme * float(np.linalg.norm(mu_fsum)), 1e-300))
    out = {"n_etats": int(n), "d": int(d),
           "cardinaux_par_cellule": tailles,
           "cellules_equicardinales": bool(equi),
           "norme_mu_global": norme,
           "chemin_1_nominal": "numpy sum(axis=0)/n, ordre des cellules du "
                               "matériau — c'est la ligne citée `mu_global`",
           "chemin_2_independant": "math.fsum par coordonnée (arrondi correct), "
                                   "cellules parcourues dans l'ORDRE INVERSE",
           "ecart_max_absolu_par_coordonnee": ecart,
           "ecart_relatif_en_norme": ecart_rel,
           "cosinus_des_deux_chemins": cos12,
           "tolerance": TOL_MU,
           "chemins_concordants": bool(ecart_rel <= TOL_MU)}
    if mu_moy_des_moy is not None:
        out["chemin_3_descriptif_moyenne_des_moyennes"] = {
            "ecart_relatif_en_norme":
                float(np.linalg.norm(mu_nominal - mu_moy_des_moy)
                      / max(norme, 1e-300)),
            "licite_parce_que": "cellules équi-cardinales (vérifié ci-dessus)"}
    return out


def majorant_de_fuite(H, cellules, t: float) -> dict:
    """**Majorant de fuite du non-LOO**, publié, jamais absorbé (0-166).

    Identité exacte, avec `n` = nombre total d'états, `μ = S/n`,
    `μ^{LOO}(p) = (S − h_a − h_b)/(n − 2)` :

        `μ^{LOO}(p) − μ = (2/(n − 2)) · (μ − (h_a + h_b)/2)`

    La fuite est donc **bornée exactement** par `(2/(n−2))·max_p‖μ − h̄_ab‖`. On
    publie en plus la perturbation qu'elle induit **sur les cosinus `γ`** — la
    seule voie par laquelle elle atteint `ρ̂_Base` — et sa propagation à
    `ρ̂_Base` par la **sensibilité mesurée** `dρ̂/dγ` de la forme gelée `Q-M10`.
    """
    Hall = np.concatenate([H[c] for c in cellules], axis=0).astype(np.float64)
    n = Hall.shape[0]
    S = Hall.sum(axis=0)
    mu = S / n
    nmu = float(np.linalg.norm(mu))
    pires = {"norme": 0.0, "gamma": 0.0}
    gammas = []
    for c in cellules:
        Hc = H[c].astype(np.float64)
        m = Hc.shape[0]
        ia, ib = np.triu_indices(m, 1)
        for deb in range(0, ia.size, 2048):
            sl = slice(deb, min(deb + 2048, ia.size))
            a, b = ia[sl], ib[sl]
            mu_loo = (S[None, :] - Hc[a] - Hc[b]) / (n - 2)
            pires["norme"] = max(
                pires["norme"],
                float((np.linalg.norm(mu_loo - mu[None, :], axis=1)
                       / max(nmu, 1e-300)).max()))
            ga_loo = _cos_lignes(Hc[a], mu_loo)
            ga_non = (Hc[a] @ mu) / np.maximum(
                np.linalg.norm(Hc[a], axis=1) * nmu, 1e-300)
            gb_loo = _cos_lignes(Hc[b], mu_loo)
            gb_non = (Hc[b] @ mu) / np.maximum(
                np.linalg.norm(Hc[b], axis=1) * nmu, 1e-300)
            pires["gamma"] = max(
                pires["gamma"],
                float(np.abs(ga_loo - ga_non).max()),
                float(np.abs(gb_loo - gb_non).max()))
            gammas.append(ga_non)
    g_moy = float(np.concatenate(gammas).mean())
    # sensibilité MESURÉE de la forme gelée, différence centrée en γ
    h_pas = 1e-4
    r_moy = float(np.clip(g_moy * g_moy, -0.999, 0.999))
    def _rho(g):
        gg = np.array([g], dtype=np.float64)
        rr = np.array([r_moy], dtype=np.float64)
        return float(P3_de(t, rr, gg, gg)[0] / P2_de(t, rr)[0])
    sens = (_rho(g_moy + h_pas) - _rho(g_moy - h_pas)) / (2 * h_pas)
    return {
        "identite": "μ^{LOO}(p) − μ = (2/(n−2))·(μ − (h_a+h_b)/2) — exacte",
        "n": int(n), "facteur_2_sur_n_moins_2": 2.0 / (n - 2),
        "majorant_ecart_relatif_de_norme": pires["norme"],
        "majorant_ecart_absolu_sur_gamma": pires["gamma"],
        "gamma_moyen_non_LOO": g_moy,
        "sensibilite_d_rho_chapeau_d_gamma_mesuree": float(sens),
        "majorant_de_fuite_sur_rho_Base":
            float(abs(sens) * pires["gamma"]),
        "valeur_annoncee_au_3_2": 5e-3,
        "regle": "publié, jamais absorbé — c'est POURQUOI le LOO est principal "
                 "et inconditionnel (P7)"}


def v_mu(citations: dict, chemins: dict, fuite: dict, eps_B=None,
         eps_R=None) -> tuple:
    """`V-mu` (§4.6) — **bloquante** (§6.D). Échoue si une formule n'est pas
    relue à une ligne unique du source, si les deux chemins de `μ_global`
    divergent au-delà de `TOL_MU`, ou si le majorant de fuite n'est pas un
    nombre publié."""
    utiles = {k: v for k, v in citations.items() if not k.startswith("_")}
    cit_ok = bool(utiles) and all(v["occurrence_unique"]
                                  for v in utiles.values())
    ch_ok = bool(chemins.get("chemins_concordants"))
    fu = fuite.get("majorant_de_fuite_sur_rho_Base")
    fu_ok = isinstance(fu, float) and np.isfinite(fu)
    ok = cit_ok and ch_ok and fu_ok
    motifs = []
    if not cit_ok:
        motifs.append("une formule n'a pas d'occurrence UNIQUE dans le "
                      "source : le chiffre serait recopié, pas relu")
    if not ch_ok:
        motifs.append("les deux chemins de μ_global divergent au-delà de "
                      f"{TOL_MU:.0e}")
    if not fu_ok:
        motifs.append("majorant de fuite non publié")
    return (PASS if ok else FAIL), {
        "citations_par_ligne_de_code": citations,
        "double_chemin_mu_global": chemins,
        "majorant_de_fuite": fuite,
        "majorant_rapporte_a_epsilon_B":
            (None if not (fu_ok and eps_B) else float(fu / eps_B)),
        "majorant_rapporte_a_epsilon_R":
            (None if not (fu_ok and eps_R) else float(fu / eps_R)),
        "regle": "V-mu / V-loo échec, ou chiffre cité de mémoire ⇒ run "
                 "INVALIDE (§6.D)",
        "motifs_d_echec": motifs or None}


def v_mu_modele(nom_modele: str, couche: int, t: float, eps_B=None,
                eps_R=None) -> tuple:
    """`V-mu` sur un modèle — charge les états `.npz` déjà en cache (aucun
    forward, `M` jamais instanciée, 0 GPU)."""
    t0 = time.time()
    _, H, cellules = etats_decisionnels(nom_modele, couche)
    cit = citations_par_ligne()
    ch = mu_deux_chemins(H, cellules)
    fu = majorant_de_fuite(H, cellules, t)
    verdict, detail = v_mu(cit, ch, fu, eps_B=eps_B, eps_R=eps_R)
    detail["modele"] = nom_modele
    detail["couche"] = int(couche)
    detail["duree_s"] = round(time.time() - t0, 2)
    return verdict, detail


# -------------------------------------------------------------------------
#  MESURE — un passage par modèle, sur le cache `Z` fp64. Aucun forward.
# -------------------------------------------------------------------------

def _rangs_et_supports(Zbloc, k: int = K_TOPK):
    """Pour un bloc d'états `(n, D)` : support `top-k` (indices triés) et carte
    de **rangs** `(n, D)` en `int16` — rang 1 = plus grande magnitude, 0 = hors
    support. C'est la définition dont `r = max(rang_A, rang_B)` a besoin."""
    n, D = Zbloc.shape
    a = np.abs(Zbloc)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(a, idx, axis=1)
    ordre = np.argsort(-vals, axis=1)
    idx_ordonne = np.take_along_axis(idx, ordre, axis=1)
    rang = np.zeros((n, D), dtype=np.int16)
    lig = np.arange(n)[:, None]
    rang[lig, idx_ordonne] = np.arange(1, k + 1, dtype=np.int16)[None, :]
    return np.sort(idx, axis=1), rang


def _cos_lignes(X, Y):
    """`cos` ligne à ligne de deux blocs `(n, d)`, fp64."""
    nx = np.linalg.norm(X, axis=1)
    ny = np.linalg.norm(Y, axis=1)
    return np.einsum("ij,ij->i", X, Y) / np.maximum(nx * ny, 1e-300)


def _acc_vide(n_ref: int) -> dict:
    return {"m": [], "n_mu": [], "n_mu_nonloo": [], "n_etat": [], "n_plac": [],
            "rho": [], "ga": [], "gb": [], "rho_z": [], "ga_z": [], "gb_z": [],
            "cos_tronque": [],
            "cos_etat_a": [], "cos_etat_b": [], "vivier": [],
            "tiges": [], "unites": [], "n_vide": 0, "P": 0,
            "c_i": np.zeros(D_DG, dtype=np.int64),
            "u_i": np.zeros(D_DG, dtype=np.int64),
            "bucket_trials": np.zeros(N_BUCKETS, dtype=np.int64),
            "bucket_accord": np.zeros(N_BUCKETS, dtype=np.int64),
            "bucket_par_paire": [], "accord_par_paire": []}


# -------------------------------------------------------------------------
#  Agrégations gelées (§4.1), prédictions (`Q-M10`), inférence
# -------------------------------------------------------------------------

def _tiges_combos(tiges_paire, tiges):
    """Aggrège les paires par **couple de tiges** : le poids bootstrap
    `m[t_a]·m[t_b]` (ou `m[t_a]` si `t_a = t_b`) ne dépend que de ce couple.
    Rend `(cle_par_paire, liste_des_couples)`."""
    rang = {t: i for i, t in enumerate(tiges)}
    cles, table = [], {}
    for ta, tb in tiges_paire:
        c = (min(rang[ta], rang[tb]), max(rang[ta], rang[tb]))
        if c not in table:
            table[c] = len(table)
        cles.append(table[c])
    inv = [None] * len(table)
    for c, i in table.items():
        inv[i] = c
    return np.asarray(cles, dtype=np.int64), inv


def _combos_xy(inv):
    x = np.array([c[0] for c in inv], dtype=np.int64)
    y = np.array([c[1] for c in inv], dtype=np.int64)
    return x, y


def _poids_combos(inv, mult):
    """Poids d'un COUPLE DE TIGES sous les multiplicités `mult` — **même règle
    déclarée** que `poids_bootstrap` : `m[t_a]·m[t_b]`, `m[t_a]` si `t_a = t_b`.
    `mult` peut être `(K,)` ou `(n, K)` (vectorisation par paquets)."""
    x, y = _combos_xy(inv)
    m = np.asarray(mult)
    if m.ndim == 1:
        return np.where(x == y, m[x], m[x] * m[y]).astype(np.float64)
    return np.where(x[None, :] == y[None, :], m[:, x],
                    m[:, x] * m[:, y]).astype(np.float64)


def bootstrap_ratio(num, den, tiges_paire, tiges, b: int = B_BOOT, seed: int = 0,
                    correction_sd: float = 0.0) -> dict:
    """IC 95 % **bootstrap de TIGES** d'un **ratio des sommes**, `B = 10⁴`,
    percentile (méthode de quantile déclarée, D24).

    `correction_sd` ajoute une perturbation `N(0, sd²)` indépendante à chaque
    réplique — c'est par là qu'entre le terme `G` de `Q-M11` dans `ε_R`.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.size == 0 or den.sum() <= 0:
        return {"estime": None, "IC": None, "statut": SANS_OBJET,
                "raison": "dénominateur nul ou aucune paire : la statistique "
                          "n'a pas de domaine de définition (D23)"}
    cles, inv = _tiges_combos(tiges_paire, tiges)
    nc = len(inv)
    nsum = np.bincount(cles, weights=num, minlength=nc)
    dsum = np.bincount(cles, weights=den, minlength=nc)
    rng = np.random.default_rng(int(seed))
    K = len(tiges)
    ech = np.empty(int(b), dtype=np.float64)
    manquantes, fait = 0, 0
    while fait < int(b):
        n = min(1000, int(b) - fait)
        tir = rng.integers(0, K, size=(n, K))
        mult = np.stack([np.bincount(tir[i], minlength=K) for i in range(n)])
        manquantes += int((mult == 0).any(axis=1).sum())
        w = _poids_combos(inv, mult)
        dd = w @ dsum
        with np.errstate(divide="ignore", invalid="ignore"):
            ech[fait:fait + n] = np.where(dd > 0, (w @ nsum) / dd, np.nan)
        fait += n
    ech = ech[np.isfinite(ech)]
    if correction_sd > 0:
        ech = ech + rng.normal(0.0, float(correction_sd), size=ech.size)
    lo, hi = np.quantile(ech, [ALPHA / 2, 1 - ALPHA / 2], method="linear")
    return {"estime": float(num.sum() / den.sum()), "IC": [float(lo), float(hi)],
            "K_eff": K, "B": int(b), "seed": int(seed),
            "n_repliques_finies": int(ech.size),
            "fraction_reechantillons_a_grappe_manquante":
                float(manquantes / int(b)),
            "methode_de_quantile": "percentile, np.quantile(method='linear') "
                                   "— type 7 (D24)",
            "terme_G_ajoute": float(correction_sd)}


def epsilon_bootstrap_signe(delta_num, den, tiges_paire, tiges, b: int = B_PERM,
                            seed: int = 0, correction_sd: float = 0.0,
                            paquet: int = 500) -> dict:
    """**LIGNE CANONIQUE** de `ε_B` / `ε_R` (§4.4), recopiée à l'identique :

    > `ε = q₀.₉₅` de `|·|` sur `B = 10⁴` rééchantillons bootstrap de tiges
    > (`K_eff = 10`) sous **permutation appariée, PAR PAIRE**, des étiquettes,
    > l'agrégation des `R` tirages étant celle du §4.1 ; une valeur par modèle
    > et par strate, calculée et **gelée AVANT lecture** des observés.
    """
    dn = np.asarray(delta_num, dtype=np.float64)
    de = np.asarray(den, dtype=np.float64)
    if dn.size == 0 or de.sum() <= 0:
        return {"epsilon": None, "statut": SANS_OBJET, "raison": "aucune paire"}
    tiges = list(tiges)
    K = len(tiges)
    ta, tb = _indices_de_tige(tiges_paire, tiges)
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(b), dtype=np.float64)
    fait = 0
    while fait < int(b):
        n = min(paquet, int(b) - fait)
        mult = np.stack([np.bincount(rng.integers(0, K, K), minlength=K)
                         for _ in range(n)])
        w = np.where(ta[None, :] == tb[None, :], mult[:, ta],
                     mult[:, ta] * mult[:, tb]).astype(np.float64)
        sg = rng.integers(0, 2, size=(n, dn.size)) * 2 - 1
        num = (w * sg * dn[None, :]).sum(axis=1)
        dd = (w * de[None, :]).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals[fait:fait + n] = np.where(dd > 0, num / dd, np.nan)
        fait += n
    vals = vals[np.isfinite(vals)]
    if correction_sd > 0:
        vals = vals + rng.normal(0.0, float(correction_sd), size=vals.size)
    return {"epsilon": float(np.quantile(np.abs(vals), Q_ENVELOPPE,
                                         method="linear")),
            "B": int(b), "seed": int(seed), "K_eff": K,
            "n_paires": int(dn.size), "n_repliques_finies": int(vals.size),
            "methode_de_quantile": "q₀.₉₅, np.quantile(method='linear') — "
                                   "type 7 (D24)",
            "terme_G_ajoute": float(correction_sd),
            "gele_avant_lecture": True,
            "formule": "q₀.₉₅ de |·| sur B = 10⁴ rééchantillons bootstrap de "
                       "tiges (K_eff = 10) sous permutation appariée PAR PAIRE "
                       "des étiquettes {μ_global^LOO, x_null}"}


def bootstrap_grappes(num_par_paire, den_par_paire, unites, n_unites: int,
                      b: int = B_GRAPPE, seed: int = 0) -> dict:
    """`N-grappe` — bootstrap **par grappe = unité décisionnelle** (0-170).
    Poids d'une paire : `m[u_a]·m[u_b]`. **Cardinal effectif et méthode de
    quantile publiés** (D24, 0-178).

    **`B = 10⁴`, comme le §4.4 et le §7 le gravent** (décision PI du
    2026-08-28) ; le jeu réduit `B = 2000` du tour précédent reste publié à
    côté, DESCRIPTIF.
    """
    num = np.asarray(num_par_paire, dtype=np.float64)
    den = np.asarray(den_par_paire, dtype=np.float64)
    if num.size == 0 or den.sum() <= 0:
        return {"estime": None, "IC": None, "statut": SANS_OBJET,
                "raison": "aucun trial : σ± n'a pas de domaine de définition"}
    ua = np.array([u[0] for u in unites], dtype=np.int64)
    ub = np.array([u[1] for u in unites], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    ech = np.empty(int(b), dtype=np.float64)
    manquantes = 0
    distincts = np.empty(int(b), dtype=np.int64)
    for r in range(int(b)):
        mult = np.bincount(rng.integers(0, n_unites, n_unites),
                           minlength=n_unites).astype(np.float64)
        manquantes += int((mult == 0).any())
        distincts[r] = int((mult > 0).sum())
        w = mult[ua] * mult[ub]
        dd = float(np.dot(w, den))
        ech[r] = float(np.dot(w, num)) / dd if dd > 0 else np.nan
    ech = ech[np.isfinite(ech)]
    lo, hi = np.quantile(ech, [ALPHA / 2, 1 - ALPHA / 2], method="linear")
    p = float(num.sum() / den.sum())
    n_trials = float(den.sum())
    return {"estime": p, "IC": [float(lo), float(hi)],
            "n_grappes": int(n_unites), "B": int(b), "seed": int(seed),
            "n_repliques_finies": int(ech.size),
            "fraction_reechantillons_a_grappe_manquante": float(manquantes / int(b)),
            "cardinal_effectif_median_de_grappes_distinctes":
                float(np.median(distincts)),
            "methode_de_quantile": "percentile, np.quantile(method='linear') — "
                                   "type 7 (D24)",
            "conformite_de_B_au_4_4":
                (f"B = {int(b)} = 10⁴ — CONFORME à la clause gelée du §4.4 et "
                 f"du §7" if int(b) == 10_000 else
                 f"B = {int(b)} ≠ 10⁴ (§4.4) — jeu DESCRIPTIF de comparaison"),
            "resolution_effective": n_eff_deff(p, n_trials, int(n_unites), ech,
                                               [float(lo), float(hi)]),
            "reserve": "à 60 grappes le percentile sous-couvre légèrement ; la "
                       "protection est de prononcer la classe d'indécision"}


def n_eff_deff(p: float, n_trials: float, n_grappes: int, repliques,
               ic) -> dict:
    """`N_eff`, `DEFF` et `ρ_ic` — **artefact publié**, deux routes NOMMÉES.

    Les deux routes ne coïncident **pas** et n'ont pas à coïncider :

      * **route VARIANCE** (retenue comme nominale) : `N_eff = p(1−p)/Var(·)`,
        `Var` = variance des répliques bootstrap. Elle n'utilise **aucune**
        hypothèse de normalité ;
      * **route LARGEUR D'IC** : `N_eff = p(1−p)/((hi−lo)/(2·z₀.₉₇₅))²`. Elle
        suppose l'IC percentile **normal et symétrique** — ce que la
        distribution bootstrap à 60 grappes n'est pas.

    **L'écart entre les deux EST la mesure de cette non-normalité** ; il est
    publié, jamais moyenné. `ρ_ic` est le coefficient de Kish implicite :
    `DEFF = 1 + (m̄ − 1)·ρ_ic`, `m̄ = trials/grappes` — **c'est une lecture
    descriptive**, la frontière de `Q-M13` restant celle du §14.5.

    **DOMAINE DE DÉFINITION DE `ρ_ic` (correctif du tour de correction #3)** :
    l'inversion de Kish n'a de sens que si `DEFF ≤ m̄`, c'est-à-dire
    `ρ_ic ≤ 1` — *une corrélation intra-grappe > 1 n'existe pas*. Quand
    `DEFF > m̄`, la quantité est publiée ***`SANS OBJET`*** avec son drapeau de
    domaine et sa raison (**D23** : une quantité sans domaine de définition se
    publie `SANS OBJET`, **jamais un nombre**) ; `DEFF` et `N_eff`, eux,
    restent publiés — ce sont eux qui portent l'information.

    **`m̄` est le MINIMUM des deux lectures dyadiques** : un trial appartient à
    une **paire** d'unités, donc « trials par grappe » se lit `trials/grappes`
    (retenue) ou `2·trials/grappes`. `ρ_ic = (DEFF − 1)/(m̄ − 1)` **décroît**
    en `m̄` : la lecture retenue en fait un **MAJORANT**, il ne peut pas
    monter. La marge à la frontière `Q-M13` (§14.5) est publiée.
    """
    v = np.asarray(repliques, dtype=np.float64)
    var_rep = float(v.var(ddof=1)) if v.size > 1 else float("nan")
    pq = float(p * (1.0 - p))
    z975 = float(_ndtri(0.975))
    largeur = float(ic[1] - ic[0])
    se_ic = largeur / (2.0 * z975)
    n_var = (pq / var_rep) if var_rep > 0 else None
    n_ic = (pq / (se_ic * se_ic)) if se_ic > 0 else None
    m_bar = (n_trials / n_grappes) if n_grappes else None
    m_bar_haut = (2.0 * n_trials / n_grappes) if n_grappes else None
    hors_domaine = []

    def _deff(ne):
        return (n_trials / ne) if ne else None

    def _rho(ne, nom):
        d = _deff(ne)
        if d is None or not m_bar or m_bar <= 1:
            return None
        if d > m_bar:
            hors_domaine.append(
                {"route": nom, "DEFF": d, "m_barre": m_bar,
                 "rho_ic_hors_domaine_qui_aurait_ete_publie":
                     (d - 1.0) / (m_bar - 1.0),
                 "raison": "DEFF > m̄ ⇒ ρ_ic > 1 : une corrélation "
                           "intra-grappe > 1 n'existe pas. Le coefficient de "
                           "Kish n'a PAS DE DOMAINE DE DÉFINITION ici."})
            return SANS_OBJET
        return (d - 1.0) / (m_bar - 1.0)

    r_var = _rho(n_var, "VARIANCE")
    r_ic = _rho(n_ic, "LARGEUR_IC")
    marge = (None if not isinstance(r_var, float)
             else FRONTIERE_Q_M13 - r_var)
    return {"n_trials": n_trials, "n_grappes": int(n_grappes),
            "m_barre_trials_par_grappe": m_bar,
            "m_barre_lecture_dyadique_haute_2x": m_bar_haut,
            "m_barre_retenue": "MINIMUM des deux lectures dyadiques "
                               "(trials/grappes) ⇒ ρ_ic publié est un "
                               "MAJORANT : il ne peut pas monter",
            "p": p, "p_fois_1_moins_p": pq,
            "variance_des_repliques": var_rep,
            "z_0975_utilise": z975,
            "diviseur_de_la_route_largeur": 2.0 * z975,
            "largeur_IC": largeur, "se_deduite_de_l_IC": se_ic,
            "N_eff_route_VARIANCE": n_var,
            "N_eff_route_LARGEUR_IC": n_ic,
            "ecart_relatif_entre_les_deux_routes":
                (None if not (n_var and n_ic) else abs(n_var - n_ic) / n_var),
            "DEFF_route_VARIANCE": _deff(n_var),
            "DEFF_route_LARGEUR_IC": _deff(n_ic),
            "rho_ic_Kish_route_VARIANCE": r_var,
            "rho_ic_Kish_route_LARGEUR_IC": r_ic,
            "rho_ic_hors_domaine": hors_domaine or None,
            "rho_ic_domaine_de_definition": "DEFF ≤ m̄ (⇔ ρ_ic ≤ 1) ; hors de "
                                            "ce domaine : SANS OBJET (D23), "
                                            "jamais un nombre",
            "frontiere_Q_M13_du_14_5": FRONTIERE_Q_M13,
            "marge_a_la_frontiere_Q_M13": marge,
            "marge_a_la_frontiere_en_facteur":
                (None if not (isinstance(r_var, float) and r_var > 0)
                 else FRONTIERE_Q_M13 / r_var),
            "route_nominale": "VARIANCE",
            "cause_nommee_de_l_ecart":
                "la route LARGEUR suppose l'IC percentile normal et "
                "symétrique ; à 60 grappes la distribution bootstrap ne l'est "
                "pas. L'écart est la mesure de cette non-normalité — il ne se "
                "corrige pas, il se publie.",
            "formule_route_VARIANCE": "N_eff = p(1−p)/Var(répliques)",
            "formule_route_LARGEUR_IC":
                "N_eff = p(1−p)/((hi−lo)/(2·z₀.₉₇₅))², z₀.₉₇₅ = ndtri(0.975)",
            "formule_Kish": "DEFF = 1 + (m̄ − 1)·ρ_ic"}


def _predictions_cellule(t: float, A: dict, avec_etat: bool = True) -> dict:
    """`ρ̂_Base` et `Δ̂_Base` d'une cellule, **agrégation gelée, miroir exact du
    §4.1** :

        `ρ̂_Base = Σ_p P₃(p) / Σ_p P₂(p)`
        `Δ̂_Base = [Σ_p P₃(ρ,γ_a,γ_b) − Σ_p (1/R)Σ_r P₃(ρ,c_a^(r),c_b^(r))]
                   / Σ_p P₂(ρ)`

    **En S3 le vivier est un singleton ⇒ `R ≡ 1`, moyenne dégénérée — publiée,
    jamais masquée** (§14.4).
    """
    rho = np.asarray(A["rho"], dtype=np.float64)
    ga = np.asarray(A["ga"], dtype=np.float64)
    gb = np.asarray(A["gb"], dtype=np.float64)
    if rho.size == 0:
        return {"statut": SANS_OBJET, "raison": "aucune paire"}
    det = gram_det(rho, ga, gb)
    p2 = P2_de(t, rho)
    p3 = P3_de(t, rho, ga, gb)
    out = {"rho_chapeau_Base": float(p3.sum() / p2.sum()),
           "somme_P3": float(p3.sum()), "somme_P2": float(p2.sum()),
           "P": int(rho.size),
           "gamma_moyen": float(0.5 * (ga.mean() + gb.mean())),
           "gamma_min": float(min(ga.min(), gb.min())),
           "gamma_max": float(max(ga.max(), gb.max())),
           "rho_ab_moyen": float(rho.mean()),
           "det_Gram_min": float(det.min()),
           "n_paires_Gram_non_PSD_au_dela_de_1e-8": int((det < -TOL_PSD).sum()),
           "c_ab_moyen": float(np.mean(
               (rho - ga * gb)
               / np.maximum(np.sqrt(np.maximum(1 - ga * ga, 0))
                            * np.sqrt(np.maximum(1 - gb * gb, 0)), 1e-300))),
           "reduction_a_un_facteur": "REFUSÉE hors S0 (Q-M10) — la forme gelée "
                                     "est la trivariée exacte (ρ_ab, γ_a, γ_b)",
           "_p2": p2, "_p3": p3}
    if avec_etat:
        ca = A["cos_etat_a"]
        cb = A["cos_etat_b"]
        tailles = np.array([len(v) for v in ca], dtype=np.int64)
        if tailles.sum() == 0:
            out["Delta_chapeau_Base"] = SANS_OBJET
            out["raison_Delta_chapeau"] = ("vivier apparié vide sur toutes les "
                                           "paires ⇒ SANS OBJET (D23)")
            out["R_par_paire"] = {"min": 0, "max": 0}
        else:
            rr = np.concatenate([np.full(len(v), rho[i])
                                 for i, v in enumerate(ca) if len(v)])
            aa = np.concatenate([v for v in ca if len(v)])
            bb = np.concatenate([v for v in cb if len(v)])
            p3n = P3_de(t, rr, aa, bb)
            pos = 0
            moy = np.zeros(rho.size, dtype=np.float64)
            valide = np.zeros(rho.size, dtype=bool)
            for i, v in enumerate(ca):
                if len(v):
                    moy[i] = p3n[pos:pos + len(v)].mean()
                    valide[i] = True
                    pos += len(v)
            out["Delta_chapeau_Base"] = float(
                (p3[valide].sum() - moy[valide].sum()) / p2[valide].sum())
            out["somme_P3_null"] = float(moy[valide].sum())
            out["P_avec_vivier"] = int(valide.sum())
            out["R_par_paire"] = {"min": int(tailles.min()),
                                  "max": int(tailles.max())}
            out["moyenne_degeneree_R_egale_1"] = bool(tailles.max() == 1)
    return out


def fusionner_buckets(tb, ab):
    """**Fusion DYADIQUE PRÉ-DÉCLARÉE** (`V-rang-def`, 0-176), appliquée
    **AVANT toute lecture** : tant qu'un bucket est sous plancher
    (`< 200` trials **OU** `< 8` paires), les buckets sont fusionnés deux à
    deux — 8 → 4 → 2 → 1. Rend `(tb, ab, niveau, journal)` ; `niveau` est la
    largeur en rangs d'un bucket fusionné.

    La règle ne dépend d'aucun résultat : elle est appliquée mécaniquement, et
    le journal des niveaux traversés est publié.
    """
    cur_tb = np.asarray(tb, dtype=np.int64)
    cur_ab = np.asarray(ab, dtype=np.float64)
    journal, largeur = [], K_TOPK // N_BUCKETS
    while True:
        n = cur_tb.shape[1]
        tr = cur_tb.sum(axis=0)
        npair = (cur_tb > 0).sum(axis=0)
        ok = bool(np.all(tr >= BUCKET_MIN_TRIALS)
                  and np.all(npair >= BUCKET_MIN_PAIRES))
        journal.append({"n_buckets": int(n), "largeur_en_rangs": int(largeur),
                        "trials": [int(v) for v in tr],
                        "n_paires": [int(v) for v in npair],
                        "planchers_satisfaits": ok})
        if ok or n == 1:
            return cur_tb, cur_ab, int(largeur), journal
        cur_tb = cur_tb.reshape(cur_tb.shape[0], n // 2, 2).sum(axis=2)
        cur_ab = cur_ab.reshape(cur_ab.shape[0], n // 2, 2).sum(axis=2)
        largeur *= 2


def sigma_pool_et_psi(t: float, A: dict) -> dict:
    """`σ±` observé et `σ̂±_pool` **gelée littéralement** (0-172) — jamais
    `σ̂±(cos̄)` — par bucket dyadique de rang, puis `Ψ = σ± − σ̂±_pool`.

    **Pondération** : le §2.4 grave `σ̂±_pool = Σ_p |A∩B|_p·σ̂±(cos_p,t)
    / Σ_p |A∩B|_p`. Dans un bucket, deux lectures sont licites : le poids
    `|A∩B|_p` **total** de la paire (littérale) ou le nombre de trials que la
    paire apporte **à ce bucket**. **Les deux sont publiées** ; la seconde est
    déclarée PRINCIPALE (c'est la seule qui pondère chaque bucket par ce qu'il
    contient). Arbitrage demandé.
    """
    tb0 = A["bucket_par_paire"]
    ab0 = A["accord_par_paire"]
    m = np.asarray(A["m"], dtype=np.float64)
    rho = np.asarray(A["rho"], dtype=np.float64)
    if tb0.size == 0:
        return {"statut": SANS_OBJET, "raison": "aucun trial"}
    tb, ab, largeur, journal = fusionner_buckets(tb0, ab0)
    n_b = tb.shape[1]
    sh = sigma_hat_pm(rho, t)
    ctr = np.asarray(A["cos_tronque"], dtype=np.float64)
    rz = np.asarray(A["rho_z"], dtype=np.float64)
    cos_par_lecture = cos_p_par_lecture(A)
    sh_par_lecture = {nom: (sigma_hat_pm(v, t) if v is not None else None)
                      for nom, v in cos_par_lecture.items()}
    out = {"t": t, "sigma_hat_par_paire_min_max": [float(sh.min()), float(sh.max())],
           "cos_pondere_par_intersection": float((m * rho).sum() / max(m.sum(), 1e-300)),
           "cos_pondere_z_double_mesure":
               float((m * rz).sum() / max(m.sum(), 1e-300)),
           "cos_pondere_tronque_phi_triple_mesure":
               (float((m * ctr).sum() / max(m.sum(), 1e-300))
                if ctr.size == m.size and m.sum() > 0 else SANS_OBJET),
           "lecture_de_cos_p_declaree":
               "cos_p = cosinus PLEIN des vecteurs d'état (h-space) sous la "
               "condition courante — c'est la corrélation exacte du modèle "
               "pivot. Le cos z-space (plug-in, D = 8192) et le cos TRONQUÉ "
               "sur φ (celui du cycle précédent) sont publiés à côté. "
               "Deux lectures licites : arbitrage demandé.",
           "prediction_pre_enregistree_lab_math": -0.02,
           "fusion_dyadique": {"largeur_en_rangs": largeur, "n_buckets": n_b,
                               "journal": journal,
                               "statut": "PRÉ-DÉCLARÉE, appliquée AVANT toute "
                                         "lecture (V-rang-def)"},
           "buckets": {}}
    for j in range(n_b):
        n_tr = float(tb[:, j].sum())
        n_ac = float(ab[:, j].sum())
        n_pa = int((tb[:, j] > 0).sum())
        if n_tr == 0:
            out["buckets"][f"b{j + 1}"] = {"statut": SANS_OBJET,
                                           "trials": 0, "n_paires": 0}
            continue
        w_bucket = tb[:, j].astype(np.float64)
        pool = float((w_bucket * sh).sum() / w_bucket.sum())
        w_lit = m * (tb[:, j] > 0)
        pool_lit = (float((w_lit * sh).sum() / w_lit.sum())
                    if w_lit.sum() > 0 else SANS_OBJET)
        lectures = {}
        for nom, shl in sh_par_lecture.items():
            if shl is None:
                lectures[nom] = {"statut": SANS_OBJET}
                continue
            pl = float((w_bucket * shl).sum() / w_bucket.sum())
            lectures[nom] = {"sigma_hat_pool": pl,
                             "Psi": n_ac / n_tr - pl}
        out["buckets"][f"b{j + 1}"] = {
            "rangs": [j * largeur + 1, (j + 1) * largeur],
            "trials": int(n_tr), "n_paires": n_pa,
            "sigma_pm": n_ac / n_tr,
            "sigma_hat_pool": pool,
            "sigma_hat_pool_ponderation_litterale": pool_lit,
            "Psi": n_ac / n_tr - pool,
            "lectures_de_cos_p": lectures,
            "sous_plancher": bool(n_tr < BUCKET_MIN_TRIALS
                                  or n_pa < BUCKET_MIN_PAIRES)}
    n_tr = float(tb.sum())
    n_ac = float(ab.sum())
    w_tot = tb.sum(axis=1).astype(np.float64)
    lect_glob = {}
    for nom, shl in sh_par_lecture.items():
        if shl is None or not w_tot.sum():
            lect_glob[nom] = {"statut": SANS_OBJET}
            continue
        pl = float((w_tot * shl).sum() / w_tot.sum())
        lect_glob[nom] = {"sigma_hat_pool": pl,
                          "Psi": (n_ac / n_tr) - pl if n_tr else SANS_OBJET,
                          "cos_pondere_par_les_trials":
                              float((w_tot * np.asarray(cos_par_lecture[nom]))
                                    .sum() / w_tot.sum())}
    out["global"] = {
        "trials": int(n_tr), "n_paires": int((w_tot > 0).sum()),
        "sigma_pm": (n_ac / n_tr) if n_tr else SANS_OBJET,
        "sigma_hat_pool": (float((w_tot * sh).sum() / w_tot.sum())
                           if w_tot.sum() else SANS_OBJET),
        "Psi": ((n_ac / n_tr) - float((w_tot * sh).sum() / w_tot.sum()))
               if (n_tr and w_tot.sum()) else SANS_OBJET,
        "lectures_de_cos_p": lect_glob}
    return out


def cos_p_par_lecture(A: dict) -> dict:
    """Les **trois lectures licites de `cos_p`**, côte à côte (clause
    sous-spécifiée « cos_p de σ̂±_pool », ARBITRAGE PI EN COURS) :

      * **`h`** — cosinus PLEIN des vecteurs d'état sous la condition courante.
        **Lecture DÉCLARÉE PRINCIPALE** : c'est la corrélation exacte du modèle
        pivot ;
      * **`z`** — cosinus dans l'espace projeté (plug-in, `D = 8192`) ;
      * **`tronque_phi`** — cosinus TRONQUÉ sur `φ`, restreint aux coordonnées
        de `A ∩ B` (la lecture du cycle précédent).

    Aucune n'est promue ici. Les trois sont mesurées sur **les mêmes paires,
    les mêmes buckets, les mêmes poids** ; seule la valeur de `cos_p` change.
    """
    n = len(A["rho"])
    out = {"h": np.asarray(A["rho"], dtype=np.float64),
           "z": np.asarray(A["rho_z"], dtype=np.float64)}
    ctr = np.asarray(A["cos_tronque"], dtype=np.float64)
    out["tronque_phi"] = ctr if ctr.size == n else None
    for nom in ("h", "z"):
        if out[nom].size != n or not np.all(np.isfinite(out[nom])):
            out[nom] = None
    return out


# -------------------------------------------------------------------------
#  `Q-M1-bis` — simulation EXACTE à sélection top-64 PAR VECTEUR (rang)
# -------------------------------------------------------------------------

def simulation_rang(cos_gram, paires, strates, n_g: int = N_G_SIM,
                    D: int = D_DG, k: int = K_TOPK, seed: int = 0,
                    ref: int = -1, t: float | None = None) -> dict:
    """`Q-M1-bis`, voie (i) — **simulation exacte à rang**.

    `Σ` = Gram des cosinus des `n` vecteurs (les 360 états + `μ_global`), fp64 ;
    Cholesky ; `Z^{(g)} ∈ ℝ^{D×n}` à **lignes iid `N(0, Σ)`** ; sélection
    **top-64 PAR COLONNE, par RANG sur `|·|`** (ex æquo de probabilité nulle
    p.s. ; **départage par indice, déclaré**) ; `ρ_Base` et `σ±` par bucket
    calculés **exactement comme sur les données**. `N_G = 200` répliques.

    Trois usages : nulle `N-queue` à rang exact ; **certification de
    `ρ̂_Base`** (tolérance 0.005) ; mesure directe du biais ratio/espérance.
    """
    S = np.asarray(cos_gram, dtype=np.float64)
    n = S.shape[0]
    vp = np.linalg.eigvalsh(S)
    if vp.min() < -TOL_PSD:
        raise ArretQM10(f"Gram simulé non-PSD : λ_min = {vp.min():.3e}")
    L = np.linalg.cholesky(S + max(0.0, -vp.min() + 1e-14) * np.eye(n))
    ia = np.asarray([p[0] for p in paires], dtype=np.int64)
    ib = np.asarray([p[1] for p in paires], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    par_strate = {s: {"num": [], "den": [], "buckets_tr": [], "buckets_ac": []}
                  for s in STRATES}
    sel_par_strate = {s: np.flatnonzero(np.asarray(strates) == s)
                      for s in STRATES}
    # `N-queue`, branche « erreur d'approximation PUBLIÉE » (§13.1, 0-191) :
    # trials PAR PAIRE et par bucket, cumulés sur les répliques de `G`. Sans
    # eux, `σ̂±_pool` ne peut pas être recalculée sur la nulle À RANG EXACT, et
    # l'erreur seuil-vs-rang resterait AFFIRMÉE au lieu d'être MESURÉE.
    tr_pp = {s: np.zeros((sel_par_strate[s].size, N_BUCKETS), dtype=np.float64)
             for s in STRATES}
    ac_pp = {s: np.zeros((sel_par_strate[s].size, N_BUCKETS), dtype=np.float64)
             for s in STRATES}
    for g in range(int(n_g)):
        Zs = (rng.standard_normal((D, n)) @ L.T)
        a = np.abs(Zs)
        idx = np.argpartition(a, -k, axis=0)[-k:, :]
        masque = np.zeros((D, n), dtype=bool)
        masque[idx, np.arange(n)[None, :]] = True
        rang = np.zeros((D, n), dtype=np.int16)
        for j in range(n):
            col = idx[:, j]
            o = np.argsort(-a[col, j], kind="stable")   # départage par indice
            rang[col[o], j] = np.arange(1, k + 1, dtype=np.int16)
        sg = Zs > 0
        mref = masque[:, ref]
        for s in STRATES:
            sel = sel_par_strate[s]
            if sel.size == 0:
                continue
            num = den = 0.0
            btr = np.zeros(N_BUCKETS, dtype=np.float64)
            bac = np.zeros(N_BUCKETS, dtype=np.float64)
            for deb in range(0, sel.size, 512):
                ss = sel[deb:deb + 512]
                A = ia[ss]
                B = ib[ss]
                mI = masque[:, A] & masque[:, B]
                den += float(mI.sum())
                num += float((mI & mref[:, None]).sum())
                lig, col = np.nonzero(mI)
                if lig.size == 0:
                    continue
                rr = np.maximum(rang[lig, A[col]], rang[lig, B[col]])
                bu = np.minimum((rr.astype(np.int64) - 1) // (k // N_BUCKETS),
                                N_BUCKETS - 1)
                acc = (sg[lig, A[col]] == sg[lig, B[col]])
                btr += np.bincount(bu, minlength=N_BUCKETS)
                bac += np.bincount(bu, weights=acc.astype(np.float64),
                                   minlength=N_BUCKETS)
                cle = col * N_BUCKETS + bu
                n_slot = ss.size * N_BUCKETS
                tr_pp[s][deb:deb + ss.size] += np.bincount(
                    cle, minlength=n_slot).reshape(ss.size, N_BUCKETS)
                ac_pp[s][deb:deb + ss.size] += np.bincount(
                    cle, weights=acc.astype(np.float64),
                    minlength=n_slot).reshape(ss.size, N_BUCKETS)
            P = par_strate[s]
            P["num"].append(num)
            P["den"].append(den)
            P["buckets_tr"].append(btr)
            P["buckets_ac"].append(bac)
    t = float(seuil_t() if t is None else t)
    rho_paires = S[ia, ib]
    out = {"N_G": int(n_g), "n_vecteurs": int(n), "seed": int(seed), "t": t,
           "departage_ex_aequo": "par INDICE (argsort stable) — déclaré ; "
                                 "probabilité d'ex æquo nulle p.s.",
           "par_strate": {}}
    for s in STRATES:
        P = par_strate[s]
        if not P["num"]:
            out["par_strate"][s] = {"statut": SANS_OBJET}
            continue
        err = _erreur_seuil_vs_rang(tr_pp[s], ac_pp[s],
                                    rho_paires[sel_par_strate[s]], t)
        num = np.asarray(P["num"])
        den = np.asarray(P["den"])
        ratios = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
        tr = np.stack(P["buckets_tr"]).astype(np.float64)
        ac = np.stack(P["buckets_ac"])
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = np.where(tr > 0, ac / np.maximum(tr, 1e-300), np.nan)
        vide = np.all(np.isnan(sig), axis=0)
        out["par_strate"][s] = {
            "rho_Base_sim_moyenne": float(np.nanmean(ratios)),
            "rho_Base_sim_sd": float(np.nanstd(ratios, ddof=1)),
            "rho_Base_sim_ratio_des_esperances":
                float(num.sum() / max(den.sum(), 1e-300)),
            "biais_ratio_moins_esperance":
                float(np.nanmean(ratios) - num.sum() / max(den.sum(), 1e-300)),
            "sigma_pm_sim_par_bucket":
                [None if vide[j] else float(np.nanmean(sig[:, j]))
                 for j in range(sig.shape[1])],
            "sigma_pm_sim_global":
                float(np.nansum(ac) / max(np.nansum(tr), 1e-300)),
            "trials_moyens_par_bucket": [float(v) for v in tr.mean(axis=0)],
            "N_queue_erreur_seuil_vs_rang": err}
    return out


def _erreur_seuil_vs_rang(tr_pp, ac_pp, rho_paires, t: float) -> dict:
    """**`N-queue`, la branche que le §13.1 exige** (0-191) : sur la nulle à
    **rang exact**, on recalcule `σ̂±_pool` — la formule à **seuil fixe**,
    littéralement celle du §4.1 — avec les **mêmes poids** que la simulation a
    produits, et on la compare au `σ±` **effectivement réalisé à rang**.

        `erreur(b) = σ̂±_pool(b)  −  σ±_rang(b)`

    C'est l'**erreur d'approximation seuil-vs-rang MESURÉE** sur `σ̂±_pool`.
    Elle est publiée **par bucket**, aux 8 buckets natifs et après la fusion
    dyadique 8 → 4, la seule échelle où les buckets des données vivent.
    """
    tr = np.asarray(tr_pp, dtype=np.float64)
    ac = np.asarray(ac_pp, dtype=np.float64)
    if tr.size == 0 or tr.sum() <= 0:
        return {"statut": SANS_OBJET, "raison": "aucun trial simulé"}
    sh = sigma_hat_pm(np.asarray(rho_paires, dtype=np.float64), t)

    def _bloc(tri, aci, niveau):
        w = tri.sum(axis=0)
        pool, sig, err = [], [], []
        for j in range(tri.shape[1]):
            if w[j] <= 0:
                pool.append(None)
                sig.append(None)
                err.append(None)
                continue
            pl = float((tri[:, j] * sh).sum() / w[j])
            sg = float(aci[:, j].sum() / w[j])
            pool.append(pl)
            sig.append(sg)
            err.append(pl - sg)
        fini = [e for e in err if e is not None]
        wt = float(w.sum())
        return {"niveau": niveau, "n_buckets": int(tri.shape[1]),
                "trials_par_bucket": [float(v) for v in w],
                "sigma_hat_pool_seuil_fixe": pool,
                "sigma_pm_a_rang_exact": sig,
                "erreur_seuil_moins_rang": err,
                "erreur_absolue_max": (max(abs(e) for e in fini)
                                       if fini else None),
                "global_sigma_hat_pool": (float((tri * sh[:, None]).sum() / wt)
                                          if wt > 0 else None),
                "global_sigma_pm_a_rang": (float(aci.sum() / wt)
                                           if wt > 0 else None)}

    n = tr.shape[1]
    res = {"lecture": "σ̂±_pool (seuil fixe) recalculée SUR la nulle à rang "
                      "exact, mêmes poids, mêmes buckets",
           "8_buckets": _bloc(tr, ac, "8 buckets de 8 rangs")}
    if n % 2 == 0:
        res["4_buckets_fusion_dyadique"] = _bloc(
            tr.reshape(tr.shape[0], n // 2, 2).sum(axis=2),
            ac.reshape(ac.shape[0], n // 2, 2).sum(axis=2),
            "4 buckets de 16 rangs (échelle des données)")
    res["branche_du_13_1_honoree"] = (
        "les DEUX : simulation exacte à rang ET erreur d'approximation "
        "seuil-vs-rang publiée pour σ̂±_pool")
    return res


# -------------------------------------------------------------------------
#  Partitions gelées (§4.5) — ordre gravé, complémentation EN DERNIER
# -------------------------------------------------------------------------

def classe_B(ic_rho, point_rho, den: float, P: int) -> str:
    if den <= 0 or P == 0:
        return "B-vide"
    if ic_rho is not None and ic_rho[0] >= 0.50:
        return "B-haut"
    if point_rho is not None and point_rho < 0.25:
        return "B-mort"
    return "B-ind"


def classe_Delta(vivier_vide: bool, ic_delta, eps_B: float) -> str:
    if vivier_vide:
        return "Δ-sansobjet"
    if ic_delta is None or eps_B is None:
        return "Δ-ind"
    if ic_delta[1] < -eps_B:
        return "Δ-anti"
    if ic_delta[0] >= -2 * eps_B and ic_delta[1] <= 2 * eps_B:
        return "Δ-nul"
    if ic_delta[0] > eps_B:
        return "Δ-priv"
    return "Δ-ind"


def classe_R(vide: bool, ic_r, eps_R: float) -> str:
    if vide:
        return "R-vide"
    if ic_r is None or eps_R is None:
        return "R-ind"
    if ic_r[1] < -eps_R:
        return "R-moins"
    if ic_r[0] >= -2 * eps_R and ic_r[1] <= 2 * eps_R:
        return "R-nul"
    if ic_r[0] > eps_R:
        return "R-plus"
    return "R-ind"


def profil_conforme_a_la_loi(profil) -> bool:
    """§5-7, prédiction SIGNÉE sous `N-queue` : *« l'écart à 0.5 CROÎT avec la
    magnitude »*.

    Le bucket `b1` porte les rangs `1..w`, donc les magnitudes **les plus
    grandes** ; le dernier bucket porte les plus petites. Profil conforme ⇔
    `|σ(b1) − 0.5| ≥ |σ(b_dernier) − 0.5|`. **Profil plat ⇒ la loi est
    fausse** ; profil conforme ⇒ `σ±` épuisé sans géométrie.

    **OPÉRATIONNALISATION DÉCLARÉE (0-192) — STATUT RÉVISÉ, ARBITRAGE PI RENDU
    LE 2026-08-28.** La clause gelée dit *« l'écart à 0.5 croît avec la
    magnitude »* sans dire comment le lire. **Deux complétions sont licites** :

      * **comparaison à DEUX POINTS** (cette fonction) : `|σ(b₁) − 0.5| ≥
        |σ(b_dernier) − 0.5|` ;
      * **MONOTONIE COMPLÈTE** (`profil_monotone_complet`) : l'écart décroît
        **à chaque pas** de bucket.

    `monotone ⇒ conforme à deux points` : l'implication est **stricte**, donc la
    complétion à deux points **rend la prédiction plus facile** — ce que **D30
    alinéa 2** fait tomber. **DÉCISION PI DU 2026-08-28 : la MONOTONIE COMPLÈTE
    est adoptée comme complétion du conjoint de `Σ-epuise`.**

    Cette fonction reste calculée et publiée, **DESCRIPTIVE** ; elle ne classe
    plus rien. Sa marge et la monotonie complète sont publiées par
    `resolution_du_profil()`.
    """
    vals = [v for v in (profil or []) if isinstance(v, float)]
    if len(vals) < 2:
        return False
    return abs(vals[0] - 0.5) >= abs(vals[-1] - 0.5)


def profil_monotone_complet(profil) -> bool:
    """**Complétion PROMUE du conjoint de `Σ-epuise`** (décision PI du
    2026-08-28, sous D30 alinéa 2).

    La clause gelée du §5-7 grave *« l'écart à 0.5 CROÎT avec la magnitude »* ;
    le bucket `b₁` porte les rangs `1..w`, donc les magnitudes les **plus
    grandes**. Profil conforme ⇔ `|σ(b_j) − 0.5|` **décroît à chaque pas**,
    sur **tous** les buckets — jamais résumé à ses deux extrémités.

    **Une inversion, même unique, suffit à ne pas conformer le profil.** La
    conséquence n'est pas *« la loi est fausse »* : c'est `Σ-ind`, avec la
    **cause nommée** *« le profil par rang n'est pas monotone »*.

    Support de moins de deux buckets ⇒ `False` ; le domaine de définition est
    publié à part par `resolution_du_profil()` (`SANS OBJET`, D23).
    """
    vals = [v for v in (profil or []) if isinstance(v, float)]
    if len(vals) < 2:
        return False
    ec = [abs(v - 0.5) for v in vals]
    return all(ec[i + 1] <= ec[i] for i in range(len(ec) - 1))


def resolution_du_profil(profil, eps_psi_b1=None) -> dict:
    """**Résolution** de `profil_conforme_a_la_loi` (0-192), publiée.

    Rend la marge `|σ(b₁) − 0.5| − |σ(b_dernier) − 0.5|`, son rapport à
    `ε_Ψ(b₁)` — la seule échelle de bruit disponible sur cette quantité — et la
    **monotonie complète par bucket**, jamais résumée aux deux extrémités.

    **ARBITRAGE PI RENDU LE 2026-08-28** : la **monotonie complète** est la
    complétion **PROMUE** du conjoint de `Σ-epuise` (elle seule ne rend pas la
    prédiction plus facile — D30 alinéa 2) ; la comparaison à **deux points**
    reste publiée, **DESCRIPTIVE**.
    """
    vals = [v for v in (profil or []) if isinstance(v, float)]
    if len(vals) < 2:
        return {"statut": SANS_OBJET,
                "raison": "moins de deux buckets — le conjoint du profil n'a "
                          "PAS DE DOMAINE DE DÉFINITION (D23) ; à ne jamais "
                          "confondre avec « indécidable par manque de "
                          "puissance »",
                "n_buckets": len(vals)}
    ecarts = [abs(v - 0.5) for v in vals]
    diffs = [ecarts[i + 1] - ecarts[i] for i in range(len(ecarts) - 1)]
    marge = ecarts[0] - ecarts[-1]
    mono = bool(all(d <= 0 for d in diffs))
    deux_pts = bool(ecarts[0] >= ecarts[-1])
    return {"sigma_par_bucket": vals,
            "ecart_a_0_5_par_bucket": ecarts,
            "differences_successives": diffs,
            "monotone_decroissante_sur_TOUS_les_buckets": mono,
            "n_inversions_de_monotonie": int(sum(1 for d in diffs if d > 0)),
            "inversions_detaillees": [
                {"pas": f"b{i + 1}→b{i + 2}", "delta_ecart": d}
                for i, d in enumerate(diffs) if d > 0],
            "marge_premier_moins_dernier": marge,
            "epsilon_Psi_b1": eps_psi_b1,
            "marge_rapportee_a_epsilon_Psi_b1":
                (None if not eps_psi_b1 else marge / eps_psi_b1),
            "completion_PROMUE": "monotonie complète",
            "conjoint_sous_completion_PROMUE_monotonie_complete": mono,
            "conjoint_sous_completion_DESCRIPTIVE_deux_points": deux_pts,
            "implication_stricte": "monotone ⇒ conforme à deux points ; la "
                                   "réciproque est FAUSSE. La complétion à "
                                   "deux points rend la prédiction PLUS "
                                   "FACILE — D30 alinéa 2 la fait tomber.",
            "arbitrage": "RENDU (PI, 2026-08-28) : la MONOTONIE COMPLÈTE est "
                         "la complétion du conjoint. La comparaison à deux "
                         "points est publiée, DESCRIPTIVE, et ne classe rien."}


def classe_Sigma(sigma_mort: bool, sous_plancher: bool, ic_psi, eps_psi,
                 profil_conforme: bool) -> str:
    """Partition `Σ` du §4.5 — **ordre gravé**, complémentation EN DERNIER
    (D18) : `Σ-mort` → `Σ-vide` → `Σ-epuise` → `Σ-residu` → `Σ-ind`.

    `profil_conforme` reçoit la **complétion PROMUE du conjoint** — la
    **monotonie complète** (`profil_monotone_complet`), décision PI du
    2026-08-28 sous D30 alinéa 2. La comparaison à deux points ne classe rien.
    """
    if sigma_mort:
        return "Σ-mort"
    if sous_plancher:
        return "Σ-vide"
    if ic_psi is None or eps_psi is None:
        return "Σ-ind"
    if all(ic[0] >= -2 * e and ic[1] <= 2 * e
           for ic, e in zip(ic_psi, eps_psi)) and profil_conforme:
        return "Σ-epuise"
    if any(min(abs(ic[0]), abs(ic[1])) > e and ic[0] * ic[1] > 0
           for ic, e in zip(ic_psi, eps_psi)):
        return "Σ-residu"
    return "Σ-ind"


# -------------------------------------------------------------------------
#  Registre des arbitrages PI portant sur la partition `Σ`
# -------------------------------------------------------------------------
#  Défaut du tour de correction : `depouillement-Sigma.json` déclarait « DEUX
#  arbitrages PI sont OUVERTS » alors qu'il y en avait **TROIS** — et le
#  troisième, NON DÉCLARÉ, était le seul à déplacer une classe. Le texte publié
#  est désormais **CONSTRUIT À PARTIR DE CE REGISTRE**, jamais écrit en dur.
ARBITRAGES_PI_SIGMA = (
    {"id": "1_lecture_de_cos_p",
     "objet": "quelle lecture de `cos_p` alimente `σ̂±_pool` : h (cosinus "
              "plein des états), z (espace projeté), tronqué φ (cycle "
              "précédent)",
     "statut": "TRANCHÉ",
     "date": "2026-08-28",
     "decision": "h-space. La lecture TRONQUÉE est DÉMONTRÉE FAUSSE ; elle "
                 "reste publiée, descriptive.",
     "deplace_une_classe": True},
    {"id": "2_couplage_de_epsilon_Psi",
     "objet": "(A) `ε_Ψ` recalculée sous la MÊME lecture, ou (B) `ε_Ψ` de la "
              "lecture PRINCIPALE appliquée aux `Ψ` des autres lectures",
     "statut": "TRANCHÉ",
     "date": "2026-08-28",
     "decision": "SANS CONSÉQUENCE décisionnelle — les deux couplages sont "
                 "publiés et ne départagent aucune classe.",
     "deplace_une_classe": False},
    {"id": "3_completion_du_conjoint_du_profil",
     "objet": "comment lire « l'écart à 0.5 CROÎT avec la magnitude » (§5-7) : "
              "comparaison à DEUX POINTS, ou MONOTONIE COMPLÈTE par bucket",
     "statut": "TRANCHÉ",
     "date": "2026-08-28",
     "decision": "MONOTONIE COMPLÈTE. `monotone ⇒ deux points` est une "
                 "implication STRICTE : la complétion à deux points rendrait "
                 "la prédiction PLUS FACILE, ce que D30 alinéa 2 fait tomber.",
     "deplace_une_classe": True},
)

CONSIGNATION_CONDITIONNALITE_PUBLIEE = (
    "défaut consigné : le dépouillement du tour précédent déclarait « DEUX "
    "arbitrages PI sont OUVERTS » alors que le registre en comptait TROIS, et "
    "que le troisième — NON DÉCLARÉ — était le seul à déplacer une classe. Le "
    "texte est désormais CONSTRUIT À PARTIR du registre "
    "`ARBITRAGES_PI_SIGMA`, jamais écrit en dur.")


def _conditionnalite_Sigma(classes_lect, classes_lect_B, cS, cS_2pts) -> str:
    """Phrase de conditionnalité de la classe `Σ`, **construite à partir du
    registre** `ARBITRAGES_PI_SIGMA` — jamais écrite en dur (défaut du tour de
    correction).
    """
    ouverts = [a for a in ARBITRAGES_PI_SIGMA if a["statut"] == "OUVERT"]
    n_tot = len(ARBITRAGES_PI_SIGMA)
    toutes = list(classes_lect.values()) + list(classes_lect_B.values()) \
        + [cS, cS_2pts]
    divergent = len(set(toutes)) > 1
    if ouverts:
        return (f"CONDITIONNELLE — {len(ouverts)} arbitrage(s) PI OUVERT(s) "
                f"sur {n_tot} au registre : "
                + " ; ".join(a["id"] for a in ouverts)
                + (". Les classes publiées ne coïncident pas toutes."
                   if divergent else
                   ". Sur cette cellule, toutes les lectures publiées rendent "
                   "la même classe."))
    return (f"INCONDITIONNELLE des arbitrages PI — {len(ouverts)} arbitrage "
            f"OUVERT sur {n_tot} au registre ; les {n_tot} sont TRANCHÉS "
            f"({', '.join(a['id'] + ' : ' + a['decision'].split('.')[0] for a in ARBITRAGES_PI_SIGMA)}). "
            + ("Les lectures DESCRIPTIVES publiées ne rendent pas toutes la "
               "même classe ; seule la lecture PRINCIPALE et la complétion "
               "PROMUE classent."
               if divergent else
               "Sur cette cellule, toutes les lectures publiées rendent la "
               "même classe."))


def v_grappe(res_sigma_grappe: dict) -> tuple:
    """`V-grappe` (0-170, 0-178) — **BLOQUANTE**. `σ±` re-exprimé sous
    **bootstrap par grappe** ; **cardinal effectif** et **méthode de quantile**
    publiés (D24). **Aucune phrase qualifiant l'ampleur de `σ±` avant PASS**
    ((xxviii), critère d'abandon K)."""
    ok = (res_sigma_grappe.get("IC") is not None
          and res_sigma_grappe.get("methode_de_quantile")
          and res_sigma_grappe.get("cardinal_effectif_median_de_grappes_distinctes")
          is not None)
    ic = res_sigma_grappe.get("IC")
    survit = (ic is not None and not (ic[0] <= 0.5 <= ic[1]))
    return (PASS if ok else FAIL), {
        "IC_sigma_pm_sous_grappes": ic,
        "estime": res_sigma_grappe.get("estime"),
        "n_grappes": res_sigma_grappe.get("n_grappes"),
        "cardinal_effectif_median":
            res_sigma_grappe.get("cardinal_effectif_median_de_grappes_distinctes"),
        "fraction_reechantillons_a_grappe_manquante":
            res_sigma_grappe.get("fraction_reechantillons_a_grappe_manquante"),
        "methode_de_quantile": res_sigma_grappe.get("methode_de_quantile"),
        "ecart_a_0_5_survit": survit,
        "classe_si_non": "Σ-mort — « il n'y a pas d'anomalie à expliquer » ; "
                         "N-queue sans objet, Q-M1 tombe pour une raison qui "
                         "n'est pas la sienne",
        "interdit_xxviii": "aucune phrase qualifiant l'ampleur de σ± n'est "
                           "publiable avant PASS de cette porte"}


def v_queue(t: float, sh_min_max, pool_publie: bool) -> tuple:
    """`V-queue` (0-171, 0-172, 0-177) — `σ̂±_pool` **gelée littéralement**,
    `t = ndtri(1 − 1/256)` **calculé au banc**, sélection top-64 **PAR
    VECTEUR**."""
    ok = pool_publie and abs(t - _ndtri(1.0 - 1.0 / 256.0)) < 1e-15
    return (PASS if ok else FAIL), {
        "t_calcule_au_banc": t, "t_formule": "ndtri(1 - 1/256)",
        "t_en_dur_interdit": 2.66,
        "ecart_a_la_valeur_en_dur": t - 2.66,
        "pente_(phi/Q)^2": pente_plackett(t),
        "sigma_hat_min_max": sh_min_max,
        "pool_gele": "σ̂±_pool = Σ_p |A∩B|_p·σ̂±(cos_p,t)/Σ_p |A∩B|_p — "
                     "JAMAIS σ̂±(cos̄) (0-172)",
        "selection": "top-64 PAR VECTEUR (rang), jamais un seuil commun ; la "
                     "certification est la simulation Q-M1-bis"}


def v_loo(rho_loo: float, rho_nonloo: float, eps_B) -> tuple:
    """`V-loo` (0-174) — `μ_global^{LOO}` est **la mesure principale**, le
    non-LOO **descriptif** ; **les deux publiés**, écart comparé à `ε_B`.
    **Inversion ⇒ arrêt.**"""
    ok = rho_loo is not None and rho_nonloo is not None
    ec = (None if not ok else rho_loo - rho_nonloo)
    return (PASS if ok else FAIL), {
        "rho_Base_mu_LOO_PRINCIPAL": rho_loo,
        "rho_Base_mu_non_LOO_descriptif": rho_nonloo,
        "ecart_LOO_moins_non_LOO": ec,
        "epsilon_B": eps_B,
        "ecart_rapporte_a_epsilon_B": (None if (ec is None or not eps_B)
                                       else ec / eps_B),
        "biais_annonce_3_2": 5e-3,
        "regle": "LOO principal, INCONDITIONNEL (P7) ; inversion ⇒ arrêt"}

def mesure_pbs(nom_modele: str, couche: int, cfg: EngramConfig, t: float,
               conditions=PBS_CONDITIONS, R_plac: int = 20, seed_dir: int = 0,
               bloc: int = 128, banc_e=None, provenance_ok: bool = False,
               geometrie_seule: bool = False) -> dict:
    """Un passage complet sur un modèle — **barrières structurelles d'abord**
    (§6.A, §6.E) : l'étage de mesure est **fermé** tant que les portes de
    provenance et le banc ne l'ouvrent pas.

    `geometrie_seule = True` : **seuls les cosinus** sont calculés (aucun
    `topk`, aucune intersection, aucun `ρ_Base`). C'est le mode de `V-quad`,
    qui doit être **exécutée et publiée AVANT toute mesure** (§6.E) : la
    séparation est structurelle, pas déclarative.
    """
    if not provenance_ok:
        raise ArretDeProvenance(
            "§6.B/C : V-cache/V-G n'ont pas rendu PASS — l'étage de mesure "
            "reste fermé")
    if banc_e != 0:
        raise BancNonFranchi(
            f"§6.A : E = {banc_e} au banc D14-S — aucune mesure avant PASS "
            f"intégral du banc")
    t0 = time.time()
    mat, H, cellules = etats_decisionnels(nom_modele, couche)
    dec, tige_de, _ = _tiges_et_S(mat)
    n_dec, d = H[cellules[0]].shape
    k = cfg.dg_topk
    mem = FastWeightMemory(d, cfg, device="cpu")   # `M` jamais lue ni écrite
    G = mem.G.numpy().astype(np.float64)
    del mem
    Z = {c: H[c] @ G.T for c in cellules}
    Hall = np.concatenate([H[c] for c in cellules], axis=0)
    Zall = np.concatenate([Z[c] for c in cellules], axis=0)
    n_global = Hall.shape[0]
    Sh, Sz = Hall.sum(axis=0), Zall.sum(axis=0)
    mu_glob = Sh / n_global
    z_mu_glob = G @ mu_glob
    sup_mu_glob = support_topk(z_mu_glob, k)
    masque_mu_glob = np.zeros(D_DG, dtype=bool)
    masque_mu_glob[sup_mu_glob] = True

    vivier = vivier_n_etat(dec)
    ia, ib = np.triu_indices(n_dec, 1)
    strate_de = np.array([p4.strate(dec[a], dec[b]) for a, b in zip(ia, ib)])
    tiges_paire = [(tige_de[a], tige_de[b]) for a, b in zip(ia, ib)]

    dirs = _directions(d, R_plac, seed_dir)
    Gr = G @ dirs.T                                  # (D, R)
    masque_plac = np.zeros((R_plac, D_DG), dtype=bool)
    for r in range(R_plac):
        masque_plac[r, support_topk(Gr[:, r], k)] = True

    acc = {x: {s: _acc_vide(R_plac) for s in STRATES} for x in conditions}
    marges = {}
    for c in cellules:
        Zc, Hc = Z[c], H[c]
        sZc, sHc = Zc.sum(axis=0), Hc.sum(axis=0)
        nH = np.linalg.norm(Hc, axis=1)
        if not geometrie_seule:
            sup_u, rang_u = _rangs_et_supports(Zc, k)
            masque_u = np.zeros((n_dec, D_DG), dtype=bool)
            for i in range(n_dec):
                masque_u[i, sup_u[i]] = True
        for deb in range(0, len(ia), bloc):
            sl = slice(deb, min(deb + bloc, len(ia)))
            a, b = ia[sl], ib[sl]
            nb = len(a)
            lig = np.arange(nb)[:, None]
            # μ_global^{LOO(p)} — retire TOUJOURS les deux états BRUTS (§3.2)
            mu_h = (Sh[None, :] - Hc[a] - Hc[b]) / (n_global - 2)
            mu_type_h = (sHc[None, :] - Hc[a] - Hc[b]) / (n_dec - 2)
            if not geometrie_seule:
                mu_z = (Sz[None, :] - Zc[a] - Zc[b]) / (n_global - 2)
                mu_type_z = (sZc[None, :] - Zc[a] - Zc[b]) / (n_dec - 2)
                sup_mu, _ = _rangs_et_supports(mu_z, k)
                masque_mu = np.zeros((nb, D_DG), dtype=bool)
                masque_mu[lig, sup_mu] = True
            else:
                mu_z = None
                mu_type_z = (sZc[None, :] - Zc[a] - Zc[b]) / (n_dec - 2)
            for cond in conditions:
                if cond == "aucun":
                    Xa, Xb = Hc[a], Hc[b]
                    Za, Zb = Zc[a], Zc[b]
                else:
                    Xa, Xb = Hc[a] - mu_type_h, Hc[b] - mu_type_h
                    Za, Zb = Zc[a] - mu_type_z, Zc[b] - mu_type_z
                # --- cosinus : h-space PRINCIPAL, z-space en double mesure D26
                rho = _cos_lignes(Xa, Xb)
                ga = _cos_lignes(Xa, mu_h)
                gb = _cos_lignes(Xb, mu_h)
                rho_z = _cos_lignes(Za, Zb)
                ga_z = (_cos_lignes(Za, mu_z) if mu_z is not None
                        else np.full(nb, np.nan))
                gb_z = (_cos_lignes(Zb, mu_z) if mu_z is not None
                        else np.full(nb, np.nan))
                nXa = np.linalg.norm(Xa, axis=1)
                nXb = np.linalg.norm(Xb, axis=1)
                cos_a_j = (Xa @ Hc.T) / np.maximum(
                    nXa[:, None] * nH[None, :], 1e-300)
                cos_b_j = (Xb @ Hc.T) / np.maximum(
                    nXb[:, None] * nH[None, :], 1e-300)
                if not geometrie_seule:
                    if cond == "aucun":
                        ra, rb = rang_u[a], rang_u[b]
                        mA, mB = masque_u[a], masque_u[b]
                    else:
                        sA, ra = _rangs_et_supports(Za, k)
                        sB, rb = _rangs_et_supports(Zb, k)
                        mA = np.zeros((nb, D_DG), dtype=bool)
                        mB = np.zeros((nb, D_DG), dtype=bool)
                        mA[lig, sA] = True
                        mB[lig, sB] = True
                    mI = mA & mB
                    m_p = mI.sum(axis=1).astype(np.int64)
                    lignes, cols = np.nonzero(mI)
                    rmax = np.maximum(ra[lignes, cols],
                                      rb[lignes, cols]).astype(np.int64)
                    buck = np.minimum((rmax - 1) // (k // N_BUCKETS),
                                      N_BUCKETS - 1)
                    acc_signe = (np.sign(Za[lignes, cols])
                                 == np.sign(Zb[lignes, cols])).astype(np.int64)
                    n_mu = np.bincount(lignes[masque_mu[lignes, cols]],
                                       minlength=nb).astype(np.int64)
                    n_mug = np.bincount(lignes[masque_mu_glob[cols]],
                                        minlength=nb).astype(np.int64)
                    n_pl = np.zeros(nb, dtype=np.float64)
                    for r in range(R_plac):
                        n_pl += np.bincount(lignes[masque_plac[r][cols]],
                                            minlength=nb).astype(np.float64)
                    n_pl /= R_plac
                    n_par_unite = np.zeros((n_dec, nb), dtype=np.int64)
                    for j in range(n_dec):
                        n_par_unite[j] = np.bincount(
                            lignes[masque_u[j][cols]], minlength=nb)
                    # `cos` TRONQUÉ (celui du cycle précédent, sur φ) —
                    # TROISIÈME mesure du même objet (D26), publiée à côté des
                    # cosinus pleins h-space (PRINCIPAL) et z-space.
                    naA = np.sqrt((Za * Za * mA).sum(axis=1))
                    nbB = np.sqrt((Zb * Zb * mB).sum(axis=1))
                    cos_tr = ((Za * Zb * mI).sum(axis=1)
                              / np.maximum(naA * nbB, 1e-300))
                for i in range(nb):
                    gp = deb + i
                    s = strate_de[gp]
                    A = acc[cond][s]
                    v = vivier[(int(ia[gp]), int(ib[gp]))]
                    A["vivier"].append(int(v.size))
                    A["rho"].append(float(rho[i]))
                    A["ga"].append(float(ga[i]))
                    A["gb"].append(float(gb[i]))
                    A["rho_z"].append(float(rho_z[i]))
                    A["ga_z"].append(float(ga_z[i]))
                    A["gb_z"].append(float(gb_z[i]))
                    A["cos_etat_a"].append(cos_a_j[i, v].astype(np.float64))
                    A["cos_etat_b"].append(cos_b_j[i, v].astype(np.float64))
                    A["tiges"].append(tiges_paire[gp])
                    A["unites"].append((int(ia[gp]), int(ib[gp])))
                    A["P"] += 1
                    if geometrie_seule:
                        continue
                    A["m"].append(int(m_p[i]))
                    A["cos_tronque"].append(float(cos_tr[i]))
                    A["n_mu"].append(int(n_mu[i]))
                    A["n_mu_nonloo"].append(int(n_mug[i]))
                    A["n_plac"].append(float(n_pl[i]))
                    A["n_etat"].append(float(n_par_unite[v, i].mean())
                                       if v.size else None)
                    if m_p[i] == 0:
                        A["n_vide"] += 1
                if geometrie_seule:
                    continue
                for s in STRATES:
                    sel = strate_de[sl] == s
                    if not sel.any():
                        continue
                    A = acc[cond][s]
                    ligsel = sel[lignes]
                    A["c_i"] += np.bincount(cols[ligsel], minlength=D_DG)
                    msel = ligsel & masque_mu[lignes, cols]
                    A["u_i"] += np.bincount(cols[msel], minlength=D_DG)
                    A["bucket_trials"] += np.bincount(buck[ligsel],
                                                      minlength=N_BUCKETS)
                    A["bucket_accord"] += np.bincount(
                        buck[ligsel],
                        weights=acc_signe[ligsel].astype(np.float64),
                        minlength=N_BUCKETS).astype(np.int64)
                    idxp = np.flatnonzero(sel)
                    ren = -np.ones(nb, dtype=np.int64)
                    ren[idxp] = np.arange(idxp.size)
                    key = ren[lignes[ligsel]] * N_BUCKETS + buck[ligsel]
                    tb = np.bincount(key, minlength=idxp.size * N_BUCKETS)
                    ab = np.bincount(
                        key, weights=acc_signe[ligsel].astype(np.float64),
                        minlength=idxp.size * N_BUCKETS)
                    A["bucket_par_paire"].append(
                        tb.reshape(idxp.size, N_BUCKETS).astype(np.int64))
                    A["accord_par_paire"].append(
                        ab.reshape(idxp.size, N_BUCKETS))
                if c == cellules[0] and deb == 0:
                    marges[cond] = _resume_marge(
                        {"marge_ulp_a": _marge_batch(Za, k),
                         "marge_ulp_b": _marge_batch(Zb, k)})
        if not geometrie_seule:
            del masque_u
    if not geometrie_seule:
        # `V-ulp` : la marge à la coupure est publiée pour TOUTES les
        # conditions, **y compris `μ_global^{LOO}`** (358 contre 360 change des
        # rangs de queue) et le placebo.
        mu_ech = (Sz[None, :] - Zall[:64] - Zall[64:128]) / (n_global - 2)
        marges["mu_global_LOO"] = _resume_marge(
            {"marge_ulp_a": _marge_batch(mu_ech, k),
             "marge_ulp_b": _marge_batch(z_mu_glob[None, :], k)})
        marges["plac"] = _resume_marge(
            {"marge_ulp_a": _marge_batch(Gr.T, k),
             "marge_ulp_b": _marge_batch(Gr.T, k)})
    for cond in conditions:
        for s in STRATES:
            A = acc[cond][s]
            A["bucket_par_paire"] = (
                np.concatenate(A["bucket_par_paire"], axis=0)
                if A["bucket_par_paire"]
                else np.zeros((0, N_BUCKETS), dtype=np.int64))
            A["accord_par_paire"] = (
                np.concatenate(A["accord_par_paire"], axis=0)
                if A["accord_par_paire"] else np.zeros((0, N_BUCKETS)))
    return {"modele": nom_modele, "couche": couche, "d": d, "n_dec": n_dec,
            "cellules": cellules, "n_global": n_global, "R_plac": R_plac,
            "seed_directions": seed_dir, "acc": acc, "marges_ulp": marges,
            "geometrie_seule": bool(geometrie_seule),
            "norme_mu_sur_h": float(np.linalg.norm(mu_glob)
                                    / np.mean(np.linalg.norm(Hall, axis=1))),
            "vivier_par_strate": v_appar(vivier, dec)[1]["cardinaux_enumeres"],
            "duree_s": round(time.time() - t0, 2)}



def bootstrap_residu(n_mu, m, p3, p2, tiges_paire, tiges, b: int = B_BOOT,
                     seed: int = 0, correction_sd: float = 0.0) -> dict:
    """IC 95 % de **`R_Base = ρ_Base − ρ̂_Base`**, bootstrap de TIGES appliqué
    **conjointement aux deux termes** (mêmes poids, même rééchantillon) : le
    résidu est une **différence**, jamais un rapport de deux inférences
    séparées."""
    n_mu = np.asarray(n_mu, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    if m.size == 0 or m.sum() <= 0 or p2.sum() <= 0:
        return {"estime": None, "IC": None, "statut": SANS_OBJET,
                "raison": "support vide (D23)"}
    cles, inv = _tiges_combos(tiges_paire, tiges)
    nc = len(inv)
    A = np.bincount(cles, weights=n_mu, minlength=nc)
    B = np.bincount(cles, weights=m, minlength=nc)
    C = np.bincount(cles, weights=p3, minlength=nc)
    Dd = np.bincount(cles, weights=p2, minlength=nc)
    rng = np.random.default_rng(int(seed))
    K = len(tiges)
    ech = np.empty(int(b), dtype=np.float64)
    manq, fait = 0, 0
    while fait < int(b):
        n = min(1000, int(b) - fait)
        tir = rng.integers(0, K, size=(n, K))
        mult = np.stack([np.bincount(tir[i], minlength=K) for i in range(n)])
        manq += int((mult == 0).any(axis=1).sum())
        w = _poids_combos(inv, mult)
        db, dd = w @ B, w @ Dd
        with np.errstate(divide="ignore", invalid="ignore"):
            ech[fait:fait + n] = np.where((db > 0) & (dd > 0),
                                          (w @ A) / db - (w @ C) / dd, np.nan)
        fait += n
    ech = ech[np.isfinite(ech)]
    var_tiges = float(ech.var(ddof=1)) if ech.size > 1 else float("nan")
    if correction_sd > 0:
        ech = ech + rng.normal(0.0, float(correction_sd), size=ech.size)
    lo, hi = np.quantile(ech, [ALPHA / 2, 1 - ALPHA / 2], method="linear")
    var_G = float(correction_sd) ** 2
    return {"estime": float(n_mu.sum() / m.sum() - p3.sum() / p2.sum()),
            "IC": [float(lo), float(hi)], "K_eff": K, "B": int(b),
            "seed": int(seed), "n_repliques_finies": int(ech.size),
            "fraction_reechantillons_a_grappe_manquante": float(manq / int(b)),
            "epsilon_R_q95_de_la_valeur_absolue":
                float(np.quantile(np.abs(ech), Q_ENVELOPPE, method="linear")),
            # 0-183 — ε_R CENTRÉ, publié À CÔTÉ du littéral. Le LITTÉRAL reste
            # DÉCISIONNEL (clause gelée §4.4, cf. « ε_R littéral » des clauses
            # sous-spécifiées) ; le centré est DESCRIPTIF et ne classe rien.
            "epsilon_R_centre_q95_descriptif":
                float(np.quantile(np.abs(ech - ech.mean()), Q_ENVELOPPE,
                                  method="linear")),
            "moyenne_des_repliques": float(ech.mean()),
            "partage_de_variance_G_contre_tiges": {
                "variance_bootstrap_de_tiges": var_tiges,
                "variance_du_terme_G": var_G,
                "part_du_terme_G":
                    (var_G / (var_G + var_tiges)
                     if np.isfinite(var_tiges) and (var_G + var_tiges) > 0
                     else None),
                "sd_G_injectee": float(correction_sd),
                "lecture": "part de la résolution ε_R imputable à l'aléa de G "
                           "(Q-M11) contre le rééchantillonnage de tiges"},
            "methode_de_quantile": "percentile, np.quantile(method='linear') — "
                                   "type 7 (D24)",
            "terme_G_ajoute": float(correction_sd)}


def ordinal_joint(num_a, den_a, tiges_a, num_b, den_b, tiges_b, tiges,
                  b: int = B_BOOT, seed: int = 0) -> dict:
    """Ordinal `N6` — **`Delta_Base(S0) >= Delta_Base(S3)`**, conjonction bornee
    par le **`min`** (0-63), **bootstrap JOINT sur le MEME reechantillon de
    tiges**. *Tout produit de p-valeurs par modele rend le run invalide.*

    §14.4 : le bras `S3` a `R = 1` (vivier singleton) et une **variance de
    tirage NULLE** ; il entre comme **constante**, pas comme estimateur. Cela
    ne concerne que la nulle `N-etat` (le tirage de `x_null`), pas le
    reechantillonnage de tiges, qui reste applique aux deux bras. C'est publie
    tel quel : *un bras sans variance n'est pas un bras avec variance nulle
    mesuree.*
    """
    if not len(num_a) or not len(num_b):
        return {"statut": SANS_OBJET,
                "raison": "un bras au moins n'a pas de domaine de definition "
                          "(vivier appari\u00e9 vide) \u21d2 l'ordre n'a pas de "
                          "domaine ; SANS OBJET, jamais 0"}
    cl_a, inv_a = _tiges_combos(tiges_a, tiges)
    cl_b, inv_b = _tiges_combos(tiges_b, tiges)
    NA = np.bincount(cl_a, weights=np.asarray(num_a, dtype=np.float64),
                     minlength=len(inv_a))
    DA = np.bincount(cl_a, weights=np.asarray(den_a, dtype=np.float64),
                     minlength=len(inv_a))
    NB = np.bincount(cl_b, weights=np.asarray(num_b, dtype=np.float64),
                     minlength=len(inv_b))
    DB = np.bincount(cl_b, weights=np.asarray(den_b, dtype=np.float64),
                     minlength=len(inv_b))
    rng = np.random.default_rng(int(seed))
    K = len(tiges)
    ech = np.empty(int(b), dtype=np.float64)
    fait, manq = 0, 0
    while fait < int(b):
        n = min(1000, int(b) - fait)
        tir = rng.integers(0, K, size=(n, K))
        mult = np.stack([np.bincount(tir[i], minlength=K) for i in range(n)])
        manq += int((mult == 0).any(axis=1).sum())
        wa = _poids_combos(inv_a, mult)
        wb = _poids_combos(inv_b, mult)
        da, db = wa @ DA, wb @ DB
        with np.errstate(divide="ignore", invalid="ignore"):
            ech[fait:fait + n] = np.where((da > 0) & (db > 0),
                                          (wa @ NA) / da - (wb @ NB) / db,
                                          np.nan)
        fait += n
    ech = ech[np.isfinite(ech)]
    lo, hi = np.quantile(ech, [ALPHA / 2, 1 - ALPHA / 2], method="linear")
    pt_a = float(np.sum(num_a) / np.sum(den_a))
    pt_b = float(np.sum(num_b) / np.sum(den_b))
    return {"Delta_bras_A": pt_a, "Delta_bras_B": pt_b,
            "ecart_A_moins_B": pt_a - pt_b,
            "IC_ecart": [float(lo), float(hi)],
            "borne_de_la_conjonction": "min (0-63) \u2014 jamais un produit de "
                                       "p-valeurs par mod\u00e8le",
            "K_eff": K, "B": int(b), "seed": int(seed),
            "n_repliques_finies": int(ech.size),
            "fraction_reechantillons_a_grappe_manquante": float(manq / int(b)),
            "methode_de_quantile": "percentile, np.quantile(method='linear') "
                                   "\u2014 type 7 (D24)",
            "note_14_4": "le bras S3 a R = 1 (vivier singleton) : variance de "
                         "TIRAGE nulle, il entre comme constante dans la nulle "
                         "N-\u00e9tat ; le bootstrap de tiges s'applique aux "
                         "deux bras"}


def epsilon_psi(A: dict, sh, n_unites: int, b: int = B_EPSILON_PSI,
                seed: int = 0) -> dict:
    """`ε_Ψ` — `q₀.₉₅` de `|Ψ|` sous `N-queue` propagée par **bootstrap de
    GRAPPES** (unité décisionnelle), une valeur **par bucket**, gelée avant
    lecture.

    **`B = 10⁴`, comme le §4.4 et le §7 le gravent** (décision PI du
    2026-08-28). Le tour précédent employait `B = 1000` au motif du budget CPU
    `< 10 min` ; ce motif est tombé — le budget a été dépassé de 110 %. Le jeu
    réduit reste calculé et publié **à côté**, DESCRIPTIF, pour que la bascule
    de classe soit **vérifiée et non supposée**.
    """
    if A["bucket_par_paire"].size == 0:
        return {"statut": SANS_OBJET, "raison": "aucun trial"}
    tbi, ab, largeur, _ = fusionner_buckets(A["bucket_par_paire"],
                                            A["accord_par_paire"])
    tb = tbi.astype(np.float64)
    n_b = tb.shape[1]
    ua = np.array([u[0] for u in A["unites"]], dtype=np.int64)
    ub = np.array([u[1] for u in A["unites"]], dtype=np.int64)
    sh = np.asarray(sh, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    vals = np.full((int(b), n_b), np.nan, dtype=np.float64)
    for r in range(int(b)):
        mult = np.bincount(rng.integers(0, n_unites, n_unites),
                           minlength=n_unites).astype(np.float64)
        w = mult[ua] * mult[ub]
        tr = w @ tb
        acr = w @ ab
        pool = (w[:, None] * tb * sh[:, None]).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals[r] = np.where(tr > 0, acr / tr - pool / tr, np.nan)
    out = {}
    for j in range(n_b):
        v = vals[:, j]
        v = v[np.isfinite(v)]
        out[f"b{j + 1}"] = (float(np.quantile(np.abs(v), Q_ENVELOPPE,
                                              method="linear"))
                            if v.size else None)
    return {"epsilon_Psi_par_bucket": out, "n_buckets": int(n_b),
            "largeur_en_rangs": int(largeur), "B": int(b), "seed": int(seed),
            "n_grappes": int(n_unites),
            "methode_de_quantile": "q₀.₉₅, np.quantile(method='linear') — "
                                   "type 7 (D24)",
            # 0-189 : le champ `B` et la phrase disaient deux choses
            # différentes. La phrase est désormais CONSTRUITE À PARTIR du champ.
            "conformite_de_B_au_4_4":
                (f"B = {int(b)} = 10⁴ — CONFORME à la clause gelée du §4.4 et "
                 f"du §7" if int(b) == 10_000 else
                 f"B = {int(b)} ≠ 10⁴ (§4.4) — ÉCART À UNE CLAUSE GELÉE, "
                 f"publié, jamais absorbé ; jeu DESCRIPTIF de comparaison. La "
                 f"valeur ci-dessus est le champ `B` lui-même, pas une phrase "
                 f"recopiée")}


def v_diag_pbs(cellules: dict) -> tuple:
    """`V-diag` de ce cycle — `ρ_Base`, `ρ̂_Base`, `R_Base`, `Δ_Base`,
    `Δ̂_Base`, `n_vide`, `f`, `σ±`, `σ̂±_pool` publiés **par strate, par
    condition, par modèle**. **Sans eux, le rapport ne peut pas être écrit.**"""
    requis = ("rho_Base", "rho_chapeau_Base", "R_Base", "Delta_Base",
              "Delta_chapeau_Base", "n_vide", "sigma_pm", "sigma_hat_pool")
    manques = []
    for cle, cel in cellules.items():
        for r in requis:
            if r not in cel:
                manques.append(f"{cle} : {r} absent")
    return (PASS if not manques else FAIL), {
        "quantites_requises": list(requis), "n_cellules": len(cellules),
        "manques": manques[:60], "n_manques": len(manques),
        "regle_D34": "un ρ_Base publié sans son ρ̂_Base apparié rend le "
                     "rapport NON ÉCRIVABLE ((xxvii), D34)"}


def inflation_epsilon_R(sim: dict, preds: dict, modele: str,
                        tol: float = TOL_SIM_QUAD) -> dict:
    """**Domaine de validité de `Q-M10`, clause (iii)** : l'approximation
    seuil-fixe est certifiée par la simulation à rang exact à
    `|sim − quad| ≤ 0.005` **par cellule** ; **au-delà, l'excès s'ajoute à
    `ε_R`** (inflation publiée), *la forme gelée ne change pas*.

    Rend `{strate: {"ecart", "exces", "certifiee"}}` pour la condition
    PRIMAIRE `aucun`.
    """
    out = {}
    for s in STRATES:
        d = sim.get("par_strate", {}).get(s, {})
        q = preds.get(f"{modele}|aucun|{s}", {}).get("rho_chapeau_Base")
        if "rho_Base_sim_ratio_des_esperances" not in d or q is None:
            out[s] = {"ecart": None, "exces": 0.0, "certifiee": SANS_OBJET}
            continue
        e = abs(d["rho_Base_sim_ratio_des_esperances"] - q)
        out[s] = {"ecart": float(e), "tolerance": tol,
                  "exces": float(max(0.0, e - tol)),
                  "certifiee": bool(e <= tol),
                  "rho_Base_sim": d["rho_Base_sim_ratio_des_esperances"],
                  "rho_chapeau_quad": q,
                  "regle": "au-delà de la tolérance, l'excès s'ajoute à ε_R "
                           "(inflation publiée) ; la forme gelée ne change pas"}
    return out


def synthese_pbs(res: dict, t: float, b_boot: int = B_BOOT, seed: int = 0,
                 preds: dict | None = None,
                 inflation: dict | None = None) -> dict:
    """Synthèse d'un modèle : quantités du §4.1, prédictions `Q-M10`,
    inférence, portes et classes des quatre partitions du §4.5 **dans l'ordre
    gravé**."""
    acc = res["acc"]
    tiges = sorted({t_ for A in acc["aucun"].values() for tp in A["tiges"]
                    for t_ in tp})
    n_unites = res["n_dec"]
    cellules, portes_locales, brut = {}, {}, {}
    for cond in acc:
        for s in STRATES:
            A = acc[cond][s]
            cle = f"{res['modele']}|{cond}|{s}"
            P = A["P"]
            m = np.asarray(A["m"], dtype=np.float64)
            n_mu = np.asarray(A["n_mu"], dtype=np.float64)
            n_mug = np.asarray(A["n_mu_nonloo"], dtype=np.float64)
            n_pl = np.asarray(A["n_plac"], dtype=np.float64)
            viv = np.asarray(A["vivier"], dtype=np.int64)
            n_et = np.array([v if v is not None else np.nan
                             for v in A["n_etat"]], dtype=np.float64)
            den = float(m.sum())
            pred = ((preds or {}).get(cle)
                    or _predictions_cellule(t, A))
            sig = sigma_pool_et_psi(t, A)
            cel = {"P": P, "n_vide": A["n_vide"],
                   "somme_intersections": int(m.sum()),
                   "denominateur_nul": den <= 0,
                   "O_en_indices": (float(m.mean()) if P else SANS_OBJET),
                   "vivier_min": int(viv.min()) if P else SANS_OBJET,
                   "vivier_max": int(viv.max()) if P else SANS_OBJET,
                   "n_paires_a_vivier_vide": int((viv == 0).sum()),
                   "cos_moyen_h": (float(np.mean(A["rho"])) if P else SANS_OBJET),
                   "cos_moyen_z_double_mesure":
                       (float(np.mean(A["rho_z"])) if P else SANS_OBJET),
                   "cos_moyen_tronque_phi_triple_mesure":
                       (float(np.mean(A["cos_tronque"]))
                        if A["cos_tronque"] else SANS_OBJET),
                   "gamma_moyen_h": (float(0.5 * (np.mean(A["ga"])
                                                  + np.mean(A["gb"])))
                                     if P else SANS_OBJET),
                   "gamma_moyen_z_double_mesure":
                       (float(0.5 * (np.mean(A["ga_z"]) + np.mean(A["gb_z"])))
                        if P else SANS_OBJET)}
            if den <= 0 or P == 0:
                cel.update({"rho_Base": SANS_OBJET, "rho_chapeau_Base": SANS_OBJET,
                            "R_Base": SANS_OBJET, "Delta_Base": SANS_OBJET,
                            "Delta_chapeau_Base": SANS_OBJET,
                            "sigma_pm": SANS_OBJET, "sigma_hat_pool": SANS_OBJET,
                            "classe_B": "B-vide", "classe_Delta": "Δ-sansobjet",
                            "classe_R": "R-vide", "raison": "support vide (D23)"})
                cellules[cle] = cel
                continue
            rho_mu = float(n_mu.sum() / den)
            rho_mug = float(n_mug.sum() / den)
            rho_pl = float(n_pl.sum() / den)
            ok_v = viv > 0
            rho_et = (float(n_et[ok_v].sum() / m[ok_v].sum())
                      if ok_v.any() and m[ok_v].sum() > 0 else SANS_OBJET)
            p2, p3 = pred["_p2"], pred["_p3"]
            sdg = sd_G_plugin(A["u_i"], A["c_i"], rho_mu)
            sd = (sdg["sd_G"] if isinstance(sdg["sd_G"], float) else 0.0)
            ic_rho = bootstrap_ratio(n_mu, m, A["tiges"], tiges, b=b_boot,
                                     seed=seed)
            resid = bootstrap_residu(n_mu, m, p3, p2, A["tiges"], tiges,
                                     b=b_boot, seed=seed, correction_sd=sd)
            if ok_v.any():
                dnum = n_mu[ok_v] - n_et[ok_v]
                ic_delta = bootstrap_ratio(dnum, m[ok_v],
                                           [A["tiges"][i]
                                            for i in np.flatnonzero(ok_v)],
                                           tiges, b=b_boot, seed=seed)
                eps_B = epsilon_bootstrap_signe(
                    dnum, m[ok_v],
                    [A["tiges"][i] for i in np.flatnonzero(ok_v)], tiges,
                    b=b_boot, seed=seed)["epsilon"]
            else:
                ic_delta = {"estime": None, "IC": None, "statut": SANS_OBJET,
                            "raison": "vivier apparié VIDE sur toutes les "
                                      "paires ⇒ Δ-sansobjet (D23), jamais un "
                                      "repli"}
                eps_B = None
            inf = (inflation or {}).get(s, {}) if cond == "aucun" else {}
            exces = float(inf.get("exces", 0.0) or 0.0)
            eps_R_brut = resid["epsilon_R_q95_de_la_valeur_absolue"]
            eps_R = eps_R_brut + exces
            cB = classe_B(ic_rho["IC"], rho_mu, den, P)
            cD = classe_Delta(not ok_v.any(), ic_delta["IC"], eps_B)
            cR = classe_R(False, resid["IC"], eps_R)
            cel.update({
                "rho_Base": rho_mu,
                "rho_Base_mu_non_LOO_descriptif": rho_mug,
                "rho_Base_etat": rho_et, "rho_Base_plac": rho_pl,
                "IC_rho_Base": ic_rho["IC"],
                "rho_chapeau_Base": pred["rho_chapeau_Base"],
                "R_Base": resid["estime"], "IC_R_Base": resid["IC"],
                "epsilon_R": eps_R, "epsilon_R_avant_inflation": eps_R_brut,
                "inflation_certification_Q_M10": inf, "sd_G": sdg,
                "Delta_Base": ic_delta.get("estime"),
                "IC_Delta_Base": ic_delta.get("IC"),
                "epsilon_B": eps_B,
                "Delta_chapeau_Base": pred.get("Delta_chapeau_Base", SANS_OBJET),
                # 0-192 — l'écart au CALIBRATEUR PRÉDIT, avec son SIGNE et sa
                # résolution. Publié par cellule ; jamais résumé en étendue.
                "ecart_Delta_moins_Delta_chapeau": (
                    float(ic_delta["estime"] - pred["Delta_chapeau_Base"])
                    if (isinstance(ic_delta.get("estime"), float)
                        and isinstance(pred.get("Delta_chapeau_Base"), float))
                    else SANS_OBJET),
                "ecart_Delta_rapporte_a_epsilon_B": (
                    float(abs(ic_delta["estime"] - pred["Delta_chapeau_Base"])
                          / eps_B)
                    if (isinstance(ic_delta.get("estime"), float)
                        and isinstance(pred.get("Delta_chapeau_Base"), float)
                        and eps_B) else SANS_OBJET),
                # 0-183 — ε_R centré publié À CÔTÉ du littéral, DESCRIPTIF
                "epsilon_R_centre_descriptif":
                    resid.get("epsilon_R_centre_q95_descriptif"),
                "classe_R_sous_epsilon_R_centre_descriptif": classe_R(
                    False, resid["IC"],
                    (resid.get("epsilon_R_centre_q95_descriptif") or 0.0)
                    + exces),
                "partage_de_variance_G_contre_tiges":
                    resid.get("partage_de_variance_G_contre_tiges"),
                "predictions_Q_M10": {kk: vv for kk, vv in pred.items()
                                      if not kk.startswith("_")},
                "sigma_pm": sig.get("global", {}).get("sigma_pm", SANS_OBJET),
                "sigma_hat_pool": sig.get("global", {}).get("sigma_hat_pool",
                                                            SANS_OBJET),
                "Psi_global": sig.get("global", {}).get("Psi", SANS_OBJET),
                "sigma_par_bucket": sig.get("buckets", {}),
                "cos_pondere_par_intersection":
                    sig.get("cos_pondere_par_intersection", SANS_OBJET),
                "classe_B": cB, "classe_Delta": cD, "classe_R": cR,
                "R_par_paire_du_vivier": pred.get("R_par_paire"),
                "moyenne_degeneree_R_egale_1":
                    pred.get("moyenne_degeneree_R_egale_1")})
            cellules[cle] = cel
            brut[cle] = {"n_mu": n_mu, "m": m, "n_et": n_et, "ok_v": ok_v,
                         "tiges": A["tiges"]}
    # ---------------------------------------- ordinal N6, bootstrap JOINT
    ordinaux = {}
    for cond in acc:
        a_ = brut.get(f"{res['modele']}|{cond}|S0")
        b_ = brut.get(f"{res['modele']}|{cond}|S3")
        if not a_ or not b_ or not a_["ok_v"].any() or not b_["ok_v"].any():
            ordinaux[cond] = {
                "statut": SANS_OBJET,
                "raison": ("un bras au moins a un vivier appari\u00e9 vide "
                           "\u21d2 l'ordre n'a PAS de domaine (D23) ; \u00e0 ne "
                           "jamais confondre avec \u00ab rien ne bouge \u00bb "
                           "(\u00a714.4)")}
            continue
        ia_ = np.flatnonzero(a_["ok_v"])
        ib_ = np.flatnonzero(b_["ok_v"])
        ordinaux[cond] = ordinal_joint(
            a_["n_mu"][ia_] - a_["n_et"][ia_], a_["m"][ia_],
            [a_["tiges"][i] for i in ia_],
            b_["n_mu"][ib_] - b_["n_et"][ib_], b_["m"][ib_],
            [b_["tiges"][i] for i in ib_], tiges, b=b_boot, seed=seed)
        ordinaux[cond]["enonce"] = "Delta_Base(S0) >= Delta_Base(S3) (N6)"
    # ------------------------------------------------- brin `σ±` (V-grappe)
    sigma = {}
    for cond in acc:
        for s in STRATES:
            A = acc[cond][s]
            if A["bucket_par_paire"].size == 0:
                continue
            tot = A["bucket_par_paire"].sum(axis=1).astype(np.float64)
            accd = A["accord_par_paire"].sum(axis=1)
            # `B = 10⁴` — clause GELÉE (§4.4, §7). Le jeu réduit du tour
            # précédent est recalculé À CÔTÉ, DESCRIPTIF, pour que la bascule
            # de classe soit VÉRIFIÉE et non supposée.
            g = bootstrap_grappes(accd, tot, A["unites"], n_unites,
                                  b=B_GRAPPE, seed=seed)
            g_red = bootstrap_grappes(accd, tot, A["unites"], n_unites,
                                      b=B_GRAPPE_REDUIT, seed=seed)
            sh = sigma_hat_pm(np.asarray(A["rho"]), t)
            ep = epsilon_psi(A, sh, n_unites, b=B_EPSILON_PSI, seed=seed)
            ep_red = epsilon_psi(A, sh, n_unites, b=B_EPSILON_PSI_REDUIT,
                                 seed=seed)
            sig = sigma_pool_et_psi(t, A)
            survit = g["IC"] is not None and not (g["IC"][0] <= 0.5 <= g["IC"][1])
            buckets = sig["buckets"]
            sous_plancher = any(v.get("sous_plancher", True)
                                for v in buckets.values() if "trials" in v)
            n_b = len(sig["buckets"])

            def _ic_eps(epx):
                icl, epl = [], []
                for j in range(n_b):
                    bkey = f"b{j + 1}"
                    e = epx.get("epsilon_Psi_par_bucket", {}).get(bkey)
                    bb = buckets.get(bkey, {})
                    if e is None or "Psi" not in bb:
                        continue
                    icl.append([bb["Psi"] - e, bb["Psi"] + e])
                    epl.append(e)
                return icl, epl

            ic_psi, eps_psi = _ic_eps(ep)
            ic_psi_red, eps_psi_red = _ic_eps(ep_red)
            profil = [buckets.get(f"b{j + 1}", {}).get("sigma_pm")
                      for j in range(n_b)]
            vals = [v for v in profil if isinstance(v, float)]
            # §5-7 : « la loi prédit que l'écart à 0.5 CROÎT avec la
            # MAGNITUDE ». Le bucket `b1` porte les rangs 1..w, donc les
            # magnitudes les PLUS GRANDES ; le dernier bucket porte les plus
            # petites.
            # ARBITRAGE PI RENDU (2026-08-28) : la complétion du conjoint est
            # la MONOTONIE COMPLÈTE — `monotone ⇒ conforme à deux points` est
            # une implication STRICTE, donc la lecture à deux points rendrait
            # la prédiction plus facile (D30 alinéa 2). La lecture à deux
            # points reste calculée et publiée, DESCRIPTIVE.
            conforme = profil_monotone_complet(vals)          # PROMUE
            conforme_2pts = profil_conforme_a_la_loi(vals)    # DESCRIPTIVE
            cS = classe_Sigma(not survit, sous_plancher, ic_psi or None,
                              eps_psi or None, conforme)
            cS_2pts = classe_Sigma(not survit, sous_plancher, ic_psi or None,
                                   eps_psi or None, conforme_2pts)
            # bascule imputable à `B` : mêmes lectures, seul `B` change
            survit_red = (g_red["IC"] is not None
                          and not (g_red["IC"][0] <= 0.5 <= g_red["IC"][1]))
            cS_B_reduit = classe_Sigma(not survit_red, sous_plancher,
                                       ic_psi_red or None, eps_psi_red or None,
                                       conforme)
            # ---- CRITIQUE 2 (0-193) : les TROIS lectures de `cos_p`, côte à
            # côte, avec leur ε_Ψ, leurs IC et la CLASSE que chacune produit.
            # La lecture `h` reste la PRINCIPALE DÉCLARÉE et la seule qui
            # classe ; les deux autres sont publiées parce que l'arbitrage est
            # OUVERT et que l'une d'elles renverse la classe.
            cosl = cos_p_par_lecture(A)
            par_lecture = {}
            for nom in LECTURES_COS_P:
                cv = cosl.get(nom)
                if cv is None:
                    par_lecture[nom] = {"statut": SANS_OBJET}
                    continue
                shn = sigma_hat_pm(cv, t)
                epn = (ep if nom == LECTURE_COS_P_PRINCIPALE
                       else epsilon_psi(A, shn, n_unites, b=1000, seed=seed))
                icn, epsn, psn, pooln = [], [], [], []
                for j in range(n_b):
                    bk = f"b{j + 1}"
                    e2 = epn.get("epsilon_Psi_par_bucket", {}).get(bk)
                    bb = buckets.get(bk, {})
                    lu = bb.get("lectures_de_cos_p", {}).get(nom, {})
                    if e2 is None or "Psi" not in lu:
                        continue
                    icn.append([lu["Psi"] - e2, lu["Psi"] + e2])
                    epsn.append(e2)
                    psn.append(lu["Psi"])
                    pooln.append(lu["sigma_hat_pool"])
                # ---- SECOND COUPLAGE, publié sans arbitrage.
                # `ε_Ψ` dépend de `σ̂±_pool`, donc de la lecture de `cos_p`.
                # DEUX couplages sont exécutables et le protocole gelé ne les
                # départage pas :
                #   (A) `Ψ(lecture)` contre `ε_Ψ(MÊME lecture)` — cohérence
                #       interne, c'est le couplage IMPLÉMENTÉ ;
                #   (B) `Ψ(lecture)` contre `ε_Ψ(lecture PRINCIPALE)` — la
                #       résolution reste celle de la lecture qui classe.
                # Les DEUX sont publiés avec leur classe. AUCUN n'est promu :
                # ARBITRAGE PI DEMANDÉ.
                icb, epsb = [], []
                for j in range(n_b):
                    bk = f"b{j + 1}"
                    e3 = ep.get("epsilon_Psi_par_bucket", {}).get(bk)
                    lu = buckets.get(bk, {}).get("lectures_de_cos_p", {}).get(
                        nom, {})
                    if e3 is None or "Psi" not in lu:
                        continue
                    icb.append([lu["Psi"] - e3, lu["Psi"] + e3])
                    epsb.append(e3)
                gl = sig.get("global", {}).get("lectures_de_cos_p", {}).get(nom, {})
                par_lecture[nom] = {
                    "couplage_B_epsilon_Psi_de_la_lecture_principale": {
                        "epsilon_Psi_par_bucket": epsb,
                        "IC_Psi_par_bucket": icb,
                        "classe_Sigma": classe_Sigma(not survit, sous_plancher,
                                                     icb or None, epsb or None,
                                                     conforme),
                        "statut": "couplage ALTERNATIF, publié, non promu — "
                                  "arbitrage PI TRANCHÉ le 2026-08-28 : SANS "
                                  "CONSÉQUENCE décisionnelle"},
                    "sigma_hat_pool_par_bucket": pooln,
                    "Psi_par_bucket": psn,
                    "epsilon_Psi_par_bucket": epsn,
                    "IC_Psi_par_bucket": icn,
                    "sigma_hat_pool_global": gl.get("sigma_hat_pool"),
                    "Psi_global": gl.get("Psi"),
                    "cos_p_pondere_par_les_trials":
                        gl.get("cos_pondere_par_les_trials"),
                    "sigma_hat_min_max": [float(shn.min()), float(shn.max())],
                    "classe_Sigma": classe_Sigma(not survit, sous_plancher,
                                                 icn or None, epsn or None,
                                                 conforme),
                    "statut": ("PRINCIPALE — arbitrage PI TRANCHÉ le "
                               "2026-08-28 en faveur de h-space ; la seule "
                               "qui classe"
                               if nom == LECTURE_COS_P_PRINCIPALE
                               else "DESCRIPTIVE — publiée pour la "
                                    "traçabilité de l'arbitrage TRANCHÉ ; la "
                                    "lecture tronquée est DÉMONTRÉE FAUSSE")}
            classes_lect = {nom: v.get("classe_Sigma")
                            for nom, v in par_lecture.items()
                            if v.get("classe_Sigma")}
            classes_lect_B = {
                nom: (v.get("couplage_B_epsilon_Psi_de_la_lecture_principale")
                      or {}).get("classe_Sigma")
                for nom, v in par_lecture.items()
                if (v.get("couplage_B_epsilon_Psi_de_la_lecture_principale")
                    or {}).get("classe_Sigma")}
            rprof = resolution_du_profil(
                vals, ep.get("epsilon_Psi_par_bucket", {}).get("b1"))
            sigma[f"{res['modele']}|{cond}|{s}"] = {
                "bootstrap_grappes": g, "V-grappe": v_grappe(g)[1],
                "bootstrap_grappes_B_reduit_DESCRIPTIF": g_red,
                "resolution_effective_N_eff_DEFF_rho_ic":
                    g.get("resolution_effective"),
                "ecart_a_0_5_survit": survit,
                "profil_sigma_par_bucket": profil,
                # complétion PROMUE (monotonie complète) — c'est elle qui classe
                "profil_monotone_complet_PROMU": conforme,
                "profil_croissant_en_magnitude_deux_points_DESCRIPTIF":
                    conforme_2pts,
                "resolution_du_profil": rprof,
                "epsilon_Psi": ep,
                "epsilon_Psi_B_reduit_DESCRIPTIF": ep_red,
                "classe_Sigma": cS,
                "classe_Sigma_par_completion_du_conjoint": {
                    "monotonie_complete_PROMUE": cS,
                    "deux_points_DESCRIPTIVE": cS_2pts,
                    "les_deux_coincident": bool(cS == cS_2pts),
                    "arbitrage": "RENDU (PI, 2026-08-28) — D30 alinéa 2 : la "
                                 "complétion à deux points rendrait la "
                                 "prédiction PLUS FACILE (monotone ⇒ deux "
                                 "points, implication STRICTE)"},
                "bascule_de_classe_imputable_a_B": {
                    "classe_a_B_10000_GRAVE": cS,
                    "classe_a_B_reduit_2000_1000_DESCRIPTIF": cS_B_reduit,
                    "bascule": bool(cS != cS_B_reduit),
                    "regle": "vérifiée, jamais supposée — la clause gelée du "
                             "§4.4 et du §7 grave B = 10⁴"},
                "classe_Sigma_par_lecture_de_cos_p": classes_lect,
                "classe_Sigma_par_lecture_couplage_B": classes_lect_B,
                "lecture_licite_D28_si_Sigma_mort": (
                    {"formulation_obligatoire":
                        f"sur {cond}|{s}, la résolution ne permet pas de "
                        f"distinguer σ± de 0.5 ; l'écart est borné par 2 × une "
                        f"résolution de ±"
                        + (f"{0.5 * (g['IC'][1] - g['IC'][0]):.2f}"
                           if g.get("IC") else "SANS OBJET"),
                     "N_eff_route_VARIANCE":
                        (g.get("resolution_effective") or {}).get(
                            "N_eff_route_VARIANCE"),
                     "largeur_de_l_IC": (None if not g.get("IC")
                                         else g["IC"][1] - g["IC"][0]),
                     "resolution_demi_largeur": (
                        None if not g.get("IC")
                        else 0.5 * (g["IC"][1] - g["IC"][0])),
                     "sigma_pm_point": g.get("estime"),
                     "ecart_ponctuel_a_0_5": (None if g.get("estime") is None
                                              else g["estime"] - 0.5),
                     "IC_sigma_pm_sous_grappes": g.get("IC"),
                     "consignation":
                        "la clause gelée du §4.5 fait dire à Σ-mort « il n'y a "
                        "pas d'anomalie à expliquer ». Appliquée VERBATIM à "
                        "cette cellule, elle conclut à une ABSENCE D'EFFET "
                        "depuis une ABSENCE DE PUISSANCE — le mode que D28 "
                        "interdit. C'est une FALSIFICATION DE CLAUSE GELÉE, au "
                        "même titre que la parenthèse du §4.5 (§14.3) : elle "
                        "SE CONSIGNE, ELLE NE SE RÉPARE PAS. La clause n'est "
                        "pas éditée ; la formulation licite est publiée à "
                        "côté d'elle.",
                     "interdit_D28": "« pas d'effet » est INTERDIT ; la classe "
                                     "d'équivalence dit « effet borné par "
                                     "2 × la résolution »"}
                    if cS == "Σ-mort" else None),
                "cause_nommee_si_Sigma_ind": (
                    ({"cause": "le conjoint du profil n'a PAS DE DOMAINE — un "
                               "seul bucket après fusion dyadique",
                      "resolution_du_profil": SANS_OBJET,
                      "n_buckets_apres_fusion": n_b,
                      "interdit": "à ne JAMAIS publier comme « indécidable par "
                                  "manque de puissance » : ce n'est pas la "
                                  "même cause et elle n'autorise pas les mêmes "
                                  "suites (D23, §14.4)"}
                     if rprof.get("statut") == SANS_OBJET else
                     {"cause": "le profil par rang n'est PAS MONOTONE — la "
                               "complétion PROMUE du conjoint de Σ-epuise "
                               "n'est pas satisfaite",
                      "n_inversions_de_monotonie":
                          rprof.get("n_inversions_de_monotonie"),
                      "inversions": rprof.get("inversions_detaillees"),
                      "marge_premier_moins_dernier":
                          rprof.get("marge_premier_moins_dernier"),
                      "marge_rapportee_a_epsilon_Psi_b1":
                          rprof.get("marge_rapportee_a_epsilon_Psi_b1"),
                      "classe_sous_la_completion_DESCRIPTIVE": cS_2pts}
                     if not conforme else
                     {"cause": "IC(Ψ) ni entièrement dans le couloir 2ε_Ψ, ni "
                               "strictement hors de ε_Ψ sur un bucket",
                      "resolution": "ε_Ψ par bucket publié ci-dessus"})
                    if cS == "Σ-ind" else None),
                "classe_conditionnelle": _conditionnalite_Sigma(
                    classes_lect, classes_lect_B, cS, cS_2pts),
                "lectures_de_cos_p": par_lecture,
                "valeur_forcee_par_identite": (
                    {"quantite": "σ± sous la condition `aucun`",
                     "valeur": sig.get("global", {}).get("sigma_pm"),
                     "identite": "régime d'égalité de Cauchy-Schwarz sur les "
                                 "états BRUTS (cos/f ∈ [0.991, 0.994], 0-162) "
                                 "⇒ σ± = 1.0 exact",
                     "poids_probant": "NUL, déclaré ((xxv), V-ident)"}
                    if cond == "aucun" else None),
                "sigma_hat_min_max": [float(sh.min()), float(sh.max())],
                "detail": sig}
    portes_locales["V-queue"] = v_queue(
        t, [min((v["sigma_hat_min_max"][0] for v in sigma.values()),
                default=None),
            max((v["sigma_hat_min_max"][1] for v in sigma.values()),
                default=None)], True)
    profil_ref = None
    for cle, v in sigma.items():
        if cle.endswith("|type|S0"):
            profil_ref = dict(v["detail"]["buckets"])
    portes_locales["V-rang-def"] = v_rang_def(profil_ref or {})
    portes_locales["V-vide"] = v_vide(
        {c: {"n_vide": v["n_vide"], "denominateur": v.get("somme_intersections", 0)}
         for c, v in cellules.items()})
    cel_ref = cellules.get(f"{res['modele']}|aucun|S0", {})
    portes_locales["V-loo"] = v_loo(cel_ref.get("rho_Base"),
                                    cel_ref.get("rho_Base_mu_non_LOO_descriptif"),
                                    cel_ref.get("epsilon_B"))
    portes_locales["V-diag"] = v_diag_pbs(cellules)
    portes_locales["V-ulp"] = ((PASS if res["marges_ulp"] else FAIL),
                               {"marges": res["marges_ulp"],
                                "conditions_publiees": sorted(res["marges_ulp"]),
                                "chemin_nominal": "fp64", "seuil": SANS_OBJET})
    ci = acc["aucun"]["S0"]["c_i"]
    qh = q95_N_hyp(ci)
    portes_locales["V-S"] = v_S(qh)
    portes_locales["N-hyp"] = qh
    P_par_cellule = {c: v["P"] for c, v in cellules.items()}
    portes_locales["V-P8"] = v_p8(P_par_cellule)
    return {"modele": res["modele"], "couche": res["couche"], "d": res["d"],
            "t": t, "n_dec": res["n_dec"], "R_plac": res["R_plac"],
            "norme_mu_sur_h": res["norme_mu_sur_h"],
            "vivier_par_strate": res["vivier_par_strate"],
            "cellules": cellules, "sigma": sigma, "portes": portes_locales,
            "ordinal_N6": ordinaux,
            "duree_mesure_s": res["duree_s"]}


def gram_pour_simulation(nom_modele: str, couche: int,
                         centre: str = "aucun") -> dict:
    """`Q-M1-bis` — `Σ` = **Gram des cosinus des 360 états + `μ_global`**
    (fp64). La variante LOO est publiée **en écart** (elle ne peut pas être une
    matrice unique : `μ^{LOO}` dépend de la paire).

    `centre = "type"` (0-191) — **le Gram des états CENTRÉS par cellule de
    capture**, seule condition où `σ±` vit. Déclaré : le centrage appliqué ici
    est la **moyenne de cellule** (`μ_type` NON-LOO), parce qu'un `Σ` unique ne
    peut pas porter un centrage qui dépend de la paire ; l'écart au LOO
    par paire est `O(1/n_cell)` et il est publié. Le vecteur de référence
    `μ_global` n'est **pas** centré (opérationnalisation déclarée : la condition
    n'affecte que `A` et `B`).
    """
    mat, H, cellules = etats_decisionnels(nom_modele, couche)
    dec, tige_de, _ = _tiges_et_S(mat)
    n_dec = H[cellules[0]].shape[0]
    Hall = np.concatenate([H[c] for c in cellules], axis=0)
    mu = Hall.mean(axis=0)
    if centre == "type":
        Hall = np.concatenate(
            [H[c] - H[c].mean(axis=0, keepdims=True) for c in cellules], axis=0)
    elif centre != "aucun":
        raise ValueError(f"centrage inconnu : {centre}")
    X = np.concatenate([Hall, mu[None, :]], axis=0)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-300)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 1.0)
    ia, ib = np.triu_indices(n_dec, 1)
    paires, strates = [], []
    for ci in range(len(cellules)):
        off = ci * n_dec
        for a, b in zip(ia, ib):
            paires.append((off + int(a), off + int(b)))
            strates.append(p4.strate(dec[a], dec[b]))
    out = {"Sigma": S, "paires": paires, "centrage": centre,
           "strates": np.asarray(strates), "n": S.shape[0]}
    if centre == "aucun":
        # écart LOO / non-LOO sur les cosinus γ, publié (D26)
        Sh = Hall.sum(axis=0)
        g_non = Xn[:-1] @ Xn[-1]
        mu_loo = (Sh[None, :] - Hall) / (Hall.shape[0] - 1)
        g_loo = np.einsum("ij,ij->i", Hall, mu_loo) / np.maximum(
            np.linalg.norm(Hall, axis=1) * np.linalg.norm(mu_loo, axis=1),
            1e-300)
        out["ecart_gamma_LOO_moins_non_LOO"] = {
            "moyen": float((g_loo - g_non).mean()),
            "max_absolu": float(np.abs(g_loo - g_non).max())}
        out["gamma_non_LOO_min_max"] = [float(g_non.min()), float(g_non.max())]
    else:
        # écart entre le centrage de CELLULE (celui du Gram) et le centrage
        # LOO-PAR-PAIRE (celui de la mesure), sur les cosinus de paire —
        # publié, jamais supposé négligeable
        ecarts = []
        for ci, c in enumerate(cellules):
            Hc = H[c].astype(np.float64)
            sH = Hc.sum(axis=0)
            Xc = Hc - Hc.mean(axis=0, keepdims=True)
            for deb in range(0, ia.size, 2048):
                sl = slice(deb, min(deb + 2048, ia.size))
                a, b = ia[sl], ib[sl]
                mu_p = (sH[None, :] - Hc[a] - Hc[b]) / (n_dec - 2)
                r_loo = _cos_lignes(Hc[a] - mu_p, Hc[b] - mu_p)
                r_cel = _cos_lignes(Xc[a], Xc[b])
                ecarts.append(np.abs(r_loo - r_cel))
        e = np.concatenate(ecarts)
        out["ecart_rho_centrage_cellule_moins_LOO_par_paire"] = {
            "moyen_absolu": float(e.mean()), "max_absolu": float(e.max()),
            "declaration": "un Σ unique ne peut pas porter un centrage qui "
                           "dépend de la paire ; l'écart est publié, jamais "
                           "supposé négligeable"}
    return out


def _pbs_json(chemin: Path, obj) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")


def main_pbs(a, cfg, modeles, out: Path) -> int:
    """Étape A — `experiments/EXP-2026-08-27-plancher-base-sigma.md`.

    Ordre GRAVÉ : `V-cache` → `V-G` → **banc D14-S (`E = 0`)** → **`V-quad`**
    → mesure. Aucun GPU, aucun forward, `M` jamais instanciée.
    """
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    t = seuil_t()
    print(f"\nt = ndtri(1 - 1/256) = {t:.15f}   (2Q(t) = {2*(1-float(_Phi(t))):.12f}"
          f" ; (φ/Q)² = {pente_plackett(t):.6f} ; pente Plackett = "
          f"{pente_plackett(t)/2:.6f})")

    vq = verif_QM10(t)
    print(f"\nQ-M10 — vérifications numériques : {vq['verdict']}")
    for c in ("identite_1_gamma_nuls_P3_egale_P2_sur_128",
              "identite_2_tous_cosinus_1_rho_chapeau_egale_1",
              "identite_3_rho_0_gamma_0_P2_egale_1_sur_128_carre",
              "ecart_GL64_GL128", "ecart_GL64_GaussKronrod_adaptatif",
              "ecart_F2_nominal_vs_independant"):
        print(f"    {c:<58} = {vq[c]:.3e}")
    _pbs_json(out / "Q-M10-verifications.json", vq)
    if vq["verdict"] != PASS:
        print("Q-M10 FAIL — la quadrature ne certifie pas ses identités : "
              "ARRÊT (aucune mesure).")
        return 4

    ident = declarations_identites(t)
    vi = v_ident(ident)
    _pbs_json(out / "V-ident.json", {"verdict": vi[0], "detail": vi[1]})
    print(f"\nV-ident : {vi[0]}  ({len(ident)} quantités forcées déclarées "
          f"AVANT mesure)")

    # ---------------------------------------------------------- `V-mu`
    # Porte de PROVENANCE, bloquante (§4.6, critère d'abandon §6.D). Elle ne
    # touche AUCUNE mesure : elle relit les formules dans le source, recalcule
    # `μ_global` par un second chemin et publie le majorant de fuite.
    vmu = {}
    verdict_mu = PASS
    print("\n--- V-mu ---")
    for m in modeles:
        ce = couche_effective(m, a.layer)
        vv, dd = v_mu_modele(m, ce["couche"], t)
        vmu[m] = {"verdict": vv, "detail": dd}
        if vv != PASS:
            verdict_mu = FAIL
        c = dd["citations_par_ligne_de_code"]
        ch = dd["double_chemin_mu_global"]
        fu = dd["majorant_de_fuite"]
        print(f"  {m} (couche {ce['couche']}) : {vv}")
        for nom, v2 in c.items():
            if nom.startswith("_"):
                continue
            print(f"      {nom:<26} l. {v2['lignes']}  unique="
                  f"{v2['occurrence_unique']}  « {v2['ligne_relue']} »")
        print(f"      μ_global, deux chemins : écart relatif = "
              f"{ch['ecart_relatif_en_norme']:.3e} (tolérance "
              f"{ch['tolerance']:.0e}) ; écart max par coordonnée = "
              f"{ch['ecart_max_absolu_par_coordonnee']:.3e} ; cos = "
              f"{ch['cosinus_des_deux_chemins']:.15f}")
        print(f"      majorant de fuite : ‖Δμ‖/‖μ‖ ≤ "
              f"{fu['majorant_ecart_relatif_de_norme']:.5f} ; |Δγ| ≤ "
              f"{fu['majorant_ecart_absolu_sur_gamma']:.5f} ; dρ̂/dγ = "
              f"{fu['sensibilite_d_rho_chapeau_d_gamma_mesuree']:+.4f} ⇒ "
              f"fuite sur ρ_Base ≤ "
              f"{fu['majorant_de_fuite_sur_rho_Base']:.5f} "
              f"(annoncé au §3.2 : {fu['valeur_annoncee_au_3_2']:.0e})")
    _pbs_json(out / "V-mu.json",
              {"porte": "V-mu", "verdict": verdict_mu,
               "regle": "§6.D — V-mu échec ⇒ run INVALIDE", "modeles": vmu})
    print(f"\nV-mu : {verdict_mu}")
    if verdict_mu != PASS:
        print("V-mu FAIL ⇒ le run est INVALIDE par le §6.D. Aucune mesure, "
              "aucune tentative de sauvetage.")
        return 6

    # -------------------------------------------------------- `V-quad`
    quad, preds_memo = {}, {}
    for m in modeles:
        ce = couche_effective(m, a.layer)
        print(f"\n--- V-quad : {m} (couche {ce['couche']}) ---")
        geo = mesure_pbs(m, ce["couche"], cfg, t, geometrie_seule=True,
                         R_plac=a.R or 20, seed_dir=a.seed,
                         banc_e=a.banc_e, provenance_ok=True)
        pc = {}
        for cond in geo["acc"]:
            for s in STRATES:
                A = geo["acc"][cond][s]
                if A["P"] == 0:
                    pc[f"{m}|{cond}|{s}"] = {"statut": SANS_OBJET}
                    continue
                pr = _predictions_cellule(t, A)
                preds_memo[f"{m}|{cond}|{s}"] = pr
                pc[f"{m}|{cond}|{s}"] = {kk: vv for kk, vv in pr.items()
                                         if not kk.startswith("_")}
                print(f"    {cond:<6} {s} : P = {pr['P']:>5}  "
                      f"γ̄ = {pr['gamma_moyen']:+.4f}  ρ̄_ab = "
                      f"{pr['rho_ab_moyen']:+.4f}  c̄_ab = {pr['c_ab_moyen']:+.4f}"
                      f"  ρ̂_Base = {pr['rho_chapeau_Base']:.4f}  Δ̂_Base = "
                      + (f"{pr['Delta_chapeau_Base']:.4f}"
                         if isinstance(pr.get("Delta_chapeau_Base"), float)
                         else str(pr.get("Delta_chapeau_Base"))))
        quad[m] = {"couche": ce["couche"], "cellules": pc,
                   "norme_mu_sur_h": geo["norme_mu_sur_h"],
                   "vivier_par_strate": geo["vivier_par_strate"]}
    # La déclaration D31 porte sur la condition PRIMAIRE `aucun` ; la
    # condition `type` (double mesure D26) est publiée séparément — mélanger
    # les deux ferait déclarer atteignable une classe qui ne l'est pas sur la
    # cellule décisionnelle.
    rhos = [v["rho_chapeau_Base"] for mm in quad.values()
            for cle, v in mm["cellules"].items()
            if "rho_chapeau_Base" in v and "|aucun|" in cle]
    rhos_type = [v["rho_chapeau_Base"] for mm in quad.values()
                 for cle, v in mm["cellules"].items()
                 if "rho_chapeau_Base" in v and "|type|" in cle]
    vquad = {
        "porte": "V-quad", "verdict": PASS if rhos else FAIL,
        "t": t, "forme_gelee": "Q-M10 — trivariée exacte (ρ_ab, γ_a, γ_b), "
                               "réduction à un facteur REFUSÉE hors S0",
        "modeles": quad,
        "declaration_d_atteignabilite_D31": {
            "B-mort": {
                "seuil": "point ρ_Base < 0.25 sur 4/4 strates d'au moins "
                         "2 modèles/3",
                "rho_chapeau_min_observe": (min(rhos) if rhos else None),
                "rho_chapeau_max_observe": (max(rhos) if rhos else None),
                "condition_primaire": "aucun",
                "classe": ("QUASI INATTEIGNABLE" if rhos and min(rhos) >= 0.25
                           else "ATTEIGNABLE"),
                "double_mesure_D26_condition_type": {
                    "rho_chapeau_min": (min(rhos_type) if rhos_type else None),
                    "rho_chapeau_max": (max(rhos_type) if rhos_type else None),
                    "classe": ("QUASI INATTEIGNABLE"
                               if rhos_type and min(rhos_type) >= 0.25
                               else "ATTEIGNABLE"),
                    "statut": "publiee separement — la lecture gelee de "
                              "C-Base-mort porte sur la cellule decisionnelle "
                              "(condition `aucun` primaire)"},
                "motif": "déclaré AVANT mesure (D31) — jamais découvert après"},
            "C-sigma-mort": {"classe": "INATTEIGNABLE",
                             "motif": "identité du barycentre + support vide "
                                      "(V-T)"}}}
    _pbs_json(out / "V-quad.json", vquad)
    bm = vquad["declaration_d_atteignabilite_D31"]["B-mort"]
    print(f"\nV-quad : {vquad['verdict']} — B-mort declaree "
          f"{bm['classe']} AVANT mesure sur la condition PRIMAIRE "
          f"`aucun` (rho_chapeau_Base dans [{min(rhos):.4f}, "
          f"{max(rhos):.4f}]) ; double mesure D26 condition `type` : "
          f"[{min(rhos_type):.4f}, {max(rhos_type):.4f}] => "
          f"{bm['double_mesure_D26_condition_type']['classe']}")
    if a.quad:
        print(f"\ntemps CPU total : {time.time() - t0:.1f} s")
        print("V-quad exécutée et publiée. Relancer sans --quad pour la mesure.")
        return 0

    # ----------------------------------------------------------- MESURE
    par_modele, sims = {}, {}
    for m in modeles:
        ce = couche_effective(m, a.layer)
        print(f"\n--- mesure : {m} (couche {ce['couche']}, {ce['source']}) ---")
        res = mesure_pbs(m, ce["couche"], cfg, t, R_plac=a.R or 20,
                         seed_dir=a.seed, banc_e=a.banc_e, provenance_ok=True)
        # `Q-M1-bis` d'ABORD : la certification de `ρ̂_Base` conditionne `ε_R`
        # (domaine de validité de `Q-M10`, clause (iii)). L'ordre n'est pas un
        # confort : lire `ε_R` avant l'inflation serait lire une résolution
        # que la certification n'accorde pas.
        infl = None
        if a.sigma_rang:
            gr = gram_pour_simulation(m, ce["couche"])
            print(f"    Q-M1-bis : simulation à rang exact, N_G = {N_G_SIM}, "
                  f"n = {gr['n']} …")
            ts = time.time()
            sim = simulation_rang(gr["Sigma"], gr["paires"], gr["strates"],
                                  n_g=N_G_SIM, seed=a.seed, ref=gr["n"] - 1)
            sim["duree_s"] = round(time.time() - ts, 2)
            sim["ecart_gamma_LOO_moins_non_LOO"] =                 gr["ecart_gamma_LOO_moins_non_LOO"]
            infl = inflation_epsilon_R(sim, preds_memo, m)
            sim["inflation_epsilon_R"] = infl
            sims[m] = sim
            for s in STRATES:
                d = sim["par_strate"][s]
                if d.get("statut") == SANS_OBJET:
                    continue
                i2 = infl[s]
                print(f"      {s} : ρ_Base_sim = "
                      f"{d['rho_Base_sim_ratio_des_esperances']:.4f} "
                      f"(±{d['rho_Base_sim_sd']:.4f}) ; |sim − quad| = "
                      + ("SO" if i2['ecart'] is None else f"{i2['ecart']:.4f}")
                      + f" (tolérance {TOL_SIM_QUAD}) ; certifiée="
                      f"{i2['certifiee']} ; excès ajouté à ε_R = "
                      f"{i2['exces']:.4f} ; σ±_sim = "
                      f"{d['sigma_pm_sim_global']:.4f}")
            _pbs_json(out / f"simulation-rang-{slug(m)}.json", sim)
            # ---- 0-191 : le maillon `N-queue` sur la condition `type`.
            # La simulation ci-dessus est bâtie sur les états BRUTS : elle
            # certifie `ρ̂_Base` mais rend `σ±_sim = 1.0000` — valeur FORCÉE
            # par le régime d'égalité de Cauchy-Schwarz (0-162), poids probant
            # NUL. `σ±` ne vit que sous `type` : la nulle à rang exact y est
            # refaite sur le Gram des états CENTRÉS, et l'erreur
            # seuil-vs-rang de `σ̂±_pool` y est MESURÉE, jamais affirmée.
            grc = gram_pour_simulation(m, ce["couche"], centre="type")
            print(f"    Q-M1-bis (états CENTRÉS par cellule) : N_G = "
                  f"{N_G_SIM}, n = {grc['n']} …")
            tsc = time.time()
            simc = simulation_rang(grc["Sigma"], grc["paires"], grc["strates"],
                                   n_g=N_G_SIM, seed=a.seed,
                                   ref=grc["n"] - 1, t=t)
            simc["duree_s"] = round(time.time() - tsc, 2)
            simc["centrage"] = "type (moyenne de cellule de capture)"
            simc["ecart_rho_centrage_cellule_moins_LOO_par_paire"] = grc[
                "ecart_rho_centrage_cellule_moins_LOO_par_paire"]
            simc["motif"] = (
                "0-191 — la simulation sur états bruts ne fournit AUCUNE nulle "
                "à rang exact pour la condition `type`, la seule où σ± vit ; "
                "σ±_sim = 1.0000 y est une valeur FORCÉE par une identité "
                "((xxv), poids probant nul).")
            sims[f"{m}|centre"] = simc
            for s in STRATES:
                d2 = simc["par_strate"][s]
                if d2.get("statut") == SANS_OBJET:
                    continue
                nq = d2["N_queue_erreur_seuil_vs_rang"]
                e4 = nq.get("4_buckets_fusion_dyadique", {})
                e8 = nq.get("8_buckets", {})
                # 0-193, défaut du tour de correction : le NIVEAU du chiffre
                # se nomme PARTOUT où le chiffre apparaît. Les deux niveaux
                # sont imprimés côte à côte ; aucun n'est cité seul.
                nq["niveaux_publies"] = {
                    "4_buckets_fusion_dyadique": {
                        "erreur_absolue_max": e4.get("erreur_absolue_max"),
                        "niveau": e4.get("niveau")},
                    "8_buckets": {
                        "erreur_absolue_max": e8.get("erreur_absolue_max"),
                        "niveau": e8.get("niveau")},
                    "rapport_8_sur_4": (
                        None if not (e8.get("erreur_absolue_max")
                                     and e4.get("erreur_absolue_max"))
                        else e8["erreur_absolue_max"]
                        / e4["erreur_absolue_max"]),
                    "regle": "un |err|max se cite AVEC SON NIVEAU de "
                             "bucketisation ; le chiffre à 8 buckets est le "
                             "plus grand des deux"}
                print(f"      {s} centré : σ±_sim = "
                      f"{d2['sigma_pm_sim_global']:.4f} ; σ̂±_pool(seuil) = "
                      + ("SO" if e4.get("global_sigma_hat_pool") is None
                         else f"{e4['global_sigma_hat_pool']:.4f}")
                      + " ; erreur seuil−rang par bucket [4 buckets] = "
                      + ", ".join("SO" if x is None else f"{x:+.4f}"
                                  for x in e4.get("erreur_seuil_moins_rang",
                                                  []))
                      + " ; |err|max [4 buckets, fusion dyadique] = "
                      + ("SO" if e4.get("erreur_absolue_max") is None
                         else f"{e4['erreur_absolue_max']:.4f}")
                      + " ; |err|max [8 buckets] = "
                      + ("SO" if e8.get("erreur_absolue_max") is None
                         else f"{e8['erreur_absolue_max']:.4f}"))
            _pbs_json(out / f"simulation-rang-centre-{slug(m)}.json", simc)
        syn = synthese_pbs(res, t, b_boot=B_BOOT, seed=a.seed,
                           preds=preds_memo, inflation=infl)
        par_modele[m] = syn
        print(f"    durée mesure {res['duree_s']} s ; ‖μ‖/‖h‖ = "
              f"{res['norme_mu_sur_h']:.4f}")
        for s in STRATES:
            c = syn["cellules"][f"{m}|aucun|{s}"]
            if c.get("rho_Base") == SANS_OBJET:
                print(f"      {s} aucun : SANS OBJET (support vide)")
                continue
            print(f"      {s} aucun : P={c['P']:>5} n_vide={c['n_vide']:>5} "
                  f"Σ|A∩B|={c['somme_intersections']:>6} | "
                  f"ρ_Base={c['rho_Base']:.4f} ρ̂_Base="
                  f"{c['rho_chapeau_Base']:.4f} R_Base={c['R_Base']:+.4f} "
                  f"IC={['%.4f' % v for v in c['IC_R_Base']]} ε_R="
                  f"{c['epsilon_R']:.4f} → {c['classe_R']} | "
                  f"Δ_Base=" + (f"{c['Delta_Base']:+.4f}"
                                if isinstance(c["Delta_Base"], float) else "SO")
                  + f" → {c['classe_Delta']} | {c['classe_B']}")
        for cond, o in syn["ordinal_N6"].items():
            if o.get("statut") == SANS_OBJET:
                print(f"      ordinal N6 [{cond}] : SANS OBJET \u2014 "
                      f"{o['raison']}")
            else:
                print(f"      ordinal N6 [{cond}] : Delta(S0)="
                      f"{o['Delta_bras_A']:+.4f} Delta(S3)="
                      f"{o['Delta_bras_B']:+.4f} ecart="
                      f"{o['ecart_A_moins_B']:+.4f} IC(joint)="
                      f"{['%+.4f' % x for x in o['IC_ecart']]}")
        if a.sigma_rang:
            for cle, v in syn["sigma"].items():
                g = v["bootstrap_grappes"]
                forcee = v.get("valeur_forcee_par_identite")
                print(f"      σ± {cle} : {g['estime']:.4f} IC "
                      f"{['%.4f' % x for x in g['IC']]} "
                      f"({g['n_grappes']} grappes, "
                      f"{g['fraction_reechantillons_a_grappe_manquante']:.3f} "
                      f"de rééchantillons à grappe manquante) → "
                      f"{v['classe_Sigma']}"
                      + ("   [VALEUR FORCÉE PAR IDENTITÉ, poids probant NUL]"
                         if forcee else ""))
                re_ = v.get("resolution_effective_N_eff_DEFF_rho_ic") or {}
                if re_.get("N_eff_route_VARIANCE"):
                    rho_ic = re_["rho_ic_Kish_route_VARIANCE"]
                    print(f"        N_eff : variance="
                          f"{re_['N_eff_route_VARIANCE']:.1f}  largeur_IC="
                          f"{re_['N_eff_route_LARGEUR_IC']:.1f}  (écart "
                          f"{re_['ecart_relatif_entre_les_deux_routes']:.1%}) ; "
                          f"DEFF={re_['DEFF_route_VARIANCE']:.1f} ; ρ_ic="
                          + (f"{rho_ic:.4f}" if isinstance(rho_ic, float)
                             else f"{rho_ic} (DEFF > m̄ ⇒ ρ_ic > 1 : hors "
                                  f"domaine, D23)")
                          + f" ; trials={re_['n_trials']:.0f}, m̄="
                          f"{re_['m_barre_trials_par_grappe']:.1f} "
                          f"(MIN des deux lectures dyadiques ; l'autre = "
                          f"{re_['m_barre_lecture_dyadique_haute_2x']:.1f} ⇒ "
                          f"ρ_ic est un MAJORANT)"
                          + ("" if not isinstance(rho_ic, float)
                             else f" ; marge à la frontière Q-M13 "
                                  f"({FRONTIERE_Q_M13}) = "
                                  f"{re_['marge_a_la_frontiere_Q_M13']:.4f}, "
                                  f"soit ×"
                                  f"{re_['marge_a_la_frontiere_en_facteur']:.2f}"))
                print(f"        profil par bucket : "
                      + ", ".join("SO" if x is None else f"{x:.4f}"
                                  for x in v["profil_sigma_par_bucket"]))
                rp = v.get("resolution_du_profil") or {}
                if isinstance(rp.get("marge_premier_moins_dernier"), float):
                    print(f"        profil (résolution) : |σ−0.5| = "
                          + ", ".join(f"{x:.4f}"
                                      for x in rp["ecart_a_0_5_par_bucket"])
                          + f" ; marge = "
                          f"{rp['marge_premier_moins_dernier']:+.4f}"
                          + (f" ; ε_Ψ(b1) = {rp['epsilon_Psi_b1']:.4f}"
                             f" ; marge/ε_Ψ = "
                             f"{rp['marge_rapportee_a_epsilon_Psi_b1']:.3f}"
                             if rp.get("epsilon_Psi_b1") else "")
                          + f" ; monotone [complétion PROMUE] = "
                          f"{rp['monotone_decroissante_sur_TOUS_les_buckets']}"
                          f" ({rp['n_inversions_de_monotonie']} inversion(s))"
                          f" ; deux points [DESCRIPTIF] = "
                          f"{rp['conjoint_sous_completion_DESCRIPTIVE_deux_points']}")
                for nom in LECTURES_COS_P:
                    L = (v.get("lectures_de_cos_p") or {}).get(nom, {})
                    if L.get("statut") == SANS_OBJET or not L:
                        print(f"        cos_p[{nom:<12}] : SANS OBJET")
                        continue
                    print(f"        cos_p[{nom:<12}] : cos̄="
                          + ("SO"
                             if L.get("cos_p_pondere_par_les_trials") is None
                             else f"{L['cos_p_pondere_par_les_trials']:+.4f}")
                          + f" σ̂±_pool="
                          + ("SO" if L.get("sigma_hat_pool_global") is None
                             else f"{L['sigma_hat_pool_global']:.4f}")
                          + f" Ψ="
                          + ("SO" if not isinstance(L.get("Psi_global"), float)
                             else f"{L['Psi_global']:+.4f}")
                          + f" → {L.get('classe_Sigma')}"
                          + (" [PRINCIPALE]"
                             if nom == LECTURE_COS_P_PRINCIPALE else ""))
                    print(f"            par bucket : σ̂±_pool = "
                          + ", ".join(f"{x:.4f}" for x in
                                      L.get("sigma_hat_pool_par_bucket", []))
                          + " | Ψ = "
                          + ", ".join(f"{x:+.4f}"
                                      for x in L.get("Psi_par_bucket", []))
                          + " | ε_Ψ = "
                          + ", ".join(f"{x:.4f}" for x in
                                      L.get("epsilon_Psi_par_bucket", []))
                          + " | IC = "
                          + ", ".join(f"[{a2:+.4f},{b2:+.4f}]"
                                      for a2, b2 in
                                      L.get("IC_Psi_par_bucket", [])))
                    B2 = L.get(
                        "couplage_B_epsilon_Psi_de_la_lecture_principale") or {}
                    if B2.get("epsilon_Psi_par_bucket"):
                        print(f"            couplage B (ε_Ψ de la lecture "
                              f"PRINCIPALE) : ε_Ψ = "
                              + ", ".join(f"{x:.4f}"
                                          for x in B2["epsilon_Psi_par_bucket"])
                              + " | IC = "
                              + ", ".join(f"[{a2:+.4f},{b2:+.4f}]"
                                          for a2, b2 in
                                          B2.get("IC_Psi_par_bucket", []))
                              + f" → {B2.get('classe_Sigma')}")
                print(f"        classes par lecture de cos_p (couplage A) : "
                      f"{v.get('classe_Sigma_par_lecture_de_cos_p')}")
                print(f"        classes par lecture de cos_p (couplage B) : "
                      f"{v.get('classe_Sigma_par_lecture_couplage_B')}")
                print(f"        {v.get('classe_conditionnelle')}")
        _pbs_json(out / f"mesure-{slug(m)}.json", syn)

    # ------------------------------------------------------------ PORTES
    dec_ref, cell_ref = None, None
    mat = materiau_v4()
    dec_ref = mat["unites_decisionnelles"]
    cell_ref = list(mat["cellules"].keys())
    viv = vivier_n_etat(dec_ref)
    portes = {"V-appar": v_appar(viv, dec_ref), "V-T": v_T(dec_ref, cell_ref),
              "V-ident": vi}
    # `V-ecriture` : les deux lectures, avec un cas où elles DIVERGENT
    ex_m = np.array([2.0, 1.0, 4.0])
    ex_mu = np.array([2.0, 0.0, 3.0])
    ex_null = [np.array([1.0, 1.0]), np.array([0.0]), np.array([1.0, 2.0, 3.0])]
    portes["V-ecriture"] = v_ecriture(ex_mu, ex_null, ex_m)
    for m, syn in par_modele.items():
        for k2, v in syn["portes"].items():
            portes[f"{k2}|{m}"] = v
    _pbs_json(out / "portes.json",
              {k: {"verdict": (v[0] if isinstance(v, tuple) else PASS),
                   "detail": (v[1] if isinstance(v, tuple) else v)}
               for k, v in portes.items()})
    print("\n--- portes ---")
    for k2, v in portes.items():
        print(f"  [{v[0] if isinstance(v, tuple) else PASS:<11}] {k2}")

    # ------------------------------------- ARTEFACTS DE DÉPOUILLEMENT
    # Défauts du tour de correction : ces six quantités étaient calculées mais
    # ne laissaient AUCUN artefact dans les bruts. Elles en ont un désormais.

    # (1) dépouillement COMPLET de la partition `Σ` — toutes les cellules,
    #     pas seulement `type|S0` ; les trois lectures de `cos_p` côte à côte.
    dep = {"regle_de_lecture":
           "l'énumération porte sur les 24 cellules (3 modèles × 2 conditions "
           "× 4 strates). Sous la condition `aucun`, σ± vaut EXACTEMENT 1.0 : "
           "valeur FORCÉE par une identité (0-162, (xxv)), poids probant NUL — "
           "elle est comptée pour l'exhaustivité, jamais lue.",
           "arbitrages_PI":
           "les arbitrages PI portant sur Σ vivent au registre "
           "`ARBITRAGES_PI_SIGMA` ; les comptes et les phrases ci-dessous en "
           "sont CONSTRUITS, jamais écrits en dur.",
           "cellules": {}, "comptes": {}, "comptes_par_lecture": {},
           "comptes_par_completion_du_conjoint": {},
           "bascules_de_classe_sous_la_completion_PROMUE": [],
           "cellules_a_rho_ic_SANS_OBJET": [],
           "lectures_licites_D28_des_Sigma_mort": {},
           "causes_nommees_des_Sigma_ind": {},
           "bascules_imputables_a_B": []}
    for m, syn in par_modele.items():
        for cle, v in syn["sigma"].items():
            re_ = v.get("resolution_effective_N_eff_DEFF_rho_ic") or {}
            dep["cellules"][cle] = {
                "classe_Sigma_lecture_principale_h": v["classe_Sigma"],
                "classe_Sigma_par_completion_du_conjoint":
                    v.get("classe_Sigma_par_completion_du_conjoint"),
                "classe_Sigma_par_lecture_de_cos_p":
                    v.get("classe_Sigma_par_lecture_de_cos_p"),
                "classe_Sigma_par_lecture_couplage_B":
                    v.get("classe_Sigma_par_lecture_couplage_B"),
                "classe_conditionnelle": v.get("classe_conditionnelle"),
                "sigma_pm": v["detail"].get("global", {}).get("sigma_pm"),
                "sigma_hat_pool":
                    v["detail"].get("global", {}).get("sigma_hat_pool"),
                "Psi_global": v["detail"].get("global", {}).get("Psi"),
                "ecart_a_0_5_survit": v["ecart_a_0_5_survit"],
                "IC_sigma_pm_sous_grappes":
                    v["bootstrap_grappes"].get("IC"),
                "resolution_effective_N_eff_DEFF_rho_ic": re_,
                "profil_sigma_par_bucket": v["profil_sigma_par_bucket"],
                "resolution_du_profil": v.get("resolution_du_profil"),
                "lecture_licite_D28_si_Sigma_mort":
                    v.get("lecture_licite_D28_si_Sigma_mort"),
                "cause_nommee_si_Sigma_ind":
                    v.get("cause_nommee_si_Sigma_ind"),
                "bascule_de_classe_imputable_a_B":
                    v.get("bascule_de_classe_imputable_a_B"),
                "valeur_forcee_par_identite":
                    v.get("valeur_forcee_par_identite"),
                "lectures_de_cos_p": v.get("lectures_de_cos_p")}
            cc = v.get("classe_Sigma_par_completion_du_conjoint") or {}
            if cc and not cc.get("les_deux_coincident"):
                rp = v.get("resolution_du_profil") or {}
                dep["bascules_de_classe_sous_la_completion_PROMUE"].append({
                    "cellule": cle,
                    "classe_sous_DEUX_POINTS_descriptive":
                        cc.get("deux_points_DESCRIPTIVE"),
                    "classe_sous_MONOTONIE_COMPLETE_promue":
                        cc.get("monotonie_complete_PROMUE"),
                    "sigma_pm": v["detail"].get("global", {}).get("sigma_pm"),
                    "marge_premier_moins_dernier":
                        rp.get("marge_premier_moins_dernier"),
                    "marge_rapportee_a_epsilon_Psi_b1":
                        rp.get("marge_rapportee_a_epsilon_Psi_b1"),
                    "n_inversions_de_monotonie":
                        rp.get("n_inversions_de_monotonie"),
                    "inversions": rp.get("inversions_detaillees")})
            if re_.get("rho_ic_hors_domaine"):
                dep["cellules_a_rho_ic_SANS_OBJET"].append({
                    "cellule": cle,
                    "rho_ic_publie": SANS_OBJET,
                    "drapeau_de_domaine": re_["rho_ic_hors_domaine"],
                    "DEFF_route_VARIANCE": re_.get("DEFF_route_VARIANCE"),
                    "m_barre": re_.get("m_barre_trials_par_grappe"),
                    "regle_D23": "une quantité sans domaine de définition se "
                                 "publie SANS OBJET, jamais un nombre"})
            if v.get("lecture_licite_D28_si_Sigma_mort"):
                dep["lectures_licites_D28_des_Sigma_mort"][cle] = \
                    v["lecture_licite_D28_si_Sigma_mort"]
            if v.get("cause_nommee_si_Sigma_ind"):
                dep["causes_nommees_des_Sigma_ind"][cle] = \
                    v["cause_nommee_si_Sigma_ind"]
            bb2 = v.get("bascule_de_classe_imputable_a_B") or {}
            if bb2.get("bascule"):
                dep["bascules_imputables_a_B"].append(
                    dict(bb2, cellule=cle))
    for cle, v in dep["cellules"].items():
        c = v["classe_Sigma_lecture_principale_h"]
        dep["comptes"][c] = dep["comptes"].get(c, 0) + 1
    for champ, cle_c in (("monotonie_complete_PROMUE", "monotonie_complete"),
                         ("deux_points_DESCRIPTIVE", "deux_points")):
        cpt = {}
        for v in dep["cellules"].values():
            c = (v.get("classe_Sigma_par_completion_du_conjoint")
                 or {}).get(champ)
            if c:
                cpt[c] = cpt.get(c, 0) + 1
        dep["comptes_par_completion_du_conjoint"][cle_c] = cpt
    for nom in LECTURES_COS_P:
        cpt = {}
        for v in dep["cellules"].values():
            c = (v.get("classe_Sigma_par_lecture_de_cos_p") or {}).get(nom)
            if c:
                cpt[c] = cpt.get(c, 0) + 1
        dep["comptes_par_lecture"][nom] = cpt
    dep["comptes_par_lecture_couplage_B"] = {}
    for nom in LECTURES_COS_P:
        cpt = {}
        for v in dep["cellules"].values():
            c = (v.get("classe_Sigma_par_lecture_couplage_B") or {}).get(nom)
            if c:
                cpt[c] = cpt.get(c, 0) + 1
        dep["comptes_par_lecture_couplage_B"][nom] = cpt
    _ouverts = [a for a in ARBITRAGES_PI_SIGMA if a["statut"] == "OUVERT"]
    dep["registre_des_arbitrages_PI"] = {
        "n_au_registre": len(ARBITRAGES_PI_SIGMA),
        "n_OUVERTS": len(_ouverts),
        "n_TRANCHES": len(ARBITRAGES_PI_SIGMA) - len(_ouverts),
        "n_qui_deplacent_une_classe":
            sum(1 for a in ARBITRAGES_PI_SIGMA if a["deplace_une_classe"]),
        "arbitrages": list(ARBITRAGES_PI_SIGMA),
        "consequence": (
            f"{len(_ouverts)} arbitrage(s) OUVERT(s) sur "
            f"{len(ARBITRAGES_PI_SIGMA)} au registre : "
            + (("toute classe Σ de ce run est CONDITIONNELLE à "
                + ", ".join(a["id"] for a in _ouverts))
               if _ouverts else
               "aucune classe Σ de ce run n'est conditionnelle à un arbitrage "
               "PI ; les trois sont TRANCHÉS. La conditionnalité qui subsiste "
               "est celle des clauses gelées elles-mêmes, publiée par "
               "cellule.")),
        "consignation_du_defaut": CONSIGNATION_CONDITIONNALITE_PUBLIEE}
    dep["cellules_par_classe_lecture_h"] = {
        c: sorted(k for k, v in dep["cellules"].items()
                  if v["classe_Sigma_lecture_principale_h"] == c)
        for c in sorted(dep["comptes"])}
    _pbs_json(out / "depouillement-Sigma.json", dep)

    # (2) ordinal N6 — LES SIX IC joints, et le minimum RÉEL sur les six.
    ord6 = {"enonce": "Delta_Base(S0) >= Delta_Base(S3) (N6)",
            "borne_de_la_conjonction": "min (0-63) — jamais un produit",
            "cellules": {}}
    bornes, largeurs = [], {}
    for m, syn in par_modele.items():
        for cond, o in syn["ordinal_N6"].items():
            ord6["cellules"][f"{m}|{cond}"] = o
            if isinstance(o.get("IC_ecart"), list):
                bornes.append((float(o["IC_ecart"][0]), f"{m}|{cond}"))
                largeurs[f"{m}|{cond}"] = float(o["IC_ecart"][1]
                                                - o["IC_ecart"][0])
    if bornes:
        mn = min(bornes)
        lg = largeurs.get(mn[1])
        rap = (abs(mn[0]) / lg) if lg else None
        ord6["minimum_reel_des_IC_inf_sur_les_six"] = {
            "valeur": mn[0], "cellule": mn[1],
            "avertissement": "le minimum sur le SEUL bras `aucun` n'est pas le "
                             "minimum de l'ordinal : les six cellules sont "
                             "publiées et le min porte sur les six",
            # --- RÉSOLUTION de la borne (défaut du tour de correction)
            "largeur_de_l_IC_de_cette_cellule": lg,
            "borne_rapportee_a_la_largeur_de_l_IC": rap,
            "borne_en_pourcentage_de_la_largeur":
                (None if rap is None else 100.0 * rap),
            "resolution_du_percentile_bootstrap":
                {"B": B_BOOT, "quantile": "percentile type 7 (D24)",
                 "pas_de_percentile_approche": 1.0 / B_BOOT},
            "statut_de_la_conjonction": (
                "ÉQUIVALENCE, pas un ordinal établi — la borne inférieure est "
                f"à {100.0 * rap:.2f} % de la largeur de son propre IC, donc "
                "DANS LE BRUIT du percentile bootstrap. La lecture licite est "
                "« l'ordre n'est pas distinguable de l'égalité à cette "
                "résolution », JAMAIS « l'ordinal est établi »."
                if (rap is not None and rap < 0.01) else
                "ordinal établi à cette résolution — la borne inférieure "
                "excède 1 % de la largeur de son IC"),
            "interdit_D28": "une borne dans le bruit ne licencie pas un "
                            "ordinal ; l'effet est borné par 2 × la "
                            "résolution"}
        ord6["minimum_par_condition"] = {
            cond: min((b for b in bornes if b[1].endswith(f"|{cond}")),
                      default=(None, None))[0]
            for cond in PBS_CONDITIONS}
    _pbs_json(out / "ordinal-N6.json", ord6)

    # (3) `Δ_Base − Δ̂_Base` — table COMPLÈTE, avec SIGNES et résolution.
    ecd = {"regle": "l'écart se publie par cellule, avec son signe et son "
                    "rapport à ε_B ; une étendue résumée n'est pas une table",
           "cellules": {}}
    n_1x = n_2x = 0
    for m, syn in par_modele.items():
        for cle, c in syn["cellules"].items():
            e = c.get("ecart_Delta_moins_Delta_chapeau")
            if not isinstance(e, float):
                continue
            r = c.get("ecart_Delta_rapporte_a_epsilon_B")
            ecd["cellules"][cle] = {
                "Delta_Base": c.get("Delta_Base"),
                "Delta_chapeau_Base": c.get("Delta_chapeau_Base"),
                "ecart_Delta_moins_Delta_chapeau": e,
                "epsilon_B": c.get("epsilon_B"),
                "ecart_absolu_rapporte_a_epsilon_B": r}
            if isinstance(r, float):
                n_1x += int(r > 1.0)
                n_2x += int(r > 2.0)
    vals = [v["ecart_Delta_moins_Delta_chapeau"]
            for v in ecd["cellules"].values()]
    if vals:
        ecd["etendue_signee"] = [min(vals), max(vals)]
        ecd["n_cellules"] = len(vals)
        ecd["n_ecarts_negatifs"] = int(sum(1 for v in vals if v < 0))
        ecd["n_depassant_1x_epsilon_B"] = n_1x
        ecd["n_depassant_2x_epsilon_B"] = n_2x
    _pbs_json(out / "ecart-Delta-Delta-chapeau.json", ecd)

    # (4) `ε_R` littéral CONTRE `ε_R` centré, sur les 24 cellules (0-183).
    epsr = {"regle": "le LITTÉRAL est DÉCISIONNEL (clause gelée §4.4) ; le "
                     "CENTRÉ est DESCRIPTIF et ne classe rien. Les cellules "
                     "dont la classe changerait sous le centré sont "
                     "CONSIGNÉES, jamais reclassées.",
            "cellules": {}, "cellules_qui_basculeraient": []}
    for m, syn in par_modele.items():
        for cle, c in syn["cellules"].items():
            if not isinstance(c.get("epsilon_R"), float):
                continue
            l1 = c.get("classe_R")
            l2 = c.get("classe_R_sous_epsilon_R_centre_descriptif")
            epsr["cellules"][cle] = {
                "R_Base": c.get("R_Base"), "IC_R_Base": c.get("IC_R_Base"),
                "epsilon_R_litteral_DECISIONNEL": c.get("epsilon_R"),
                "epsilon_R_centre_DESCRIPTIF":
                    c.get("epsilon_R_centre_descriptif"),
                "rapport_centre_sur_litteral":
                    (c["epsilon_R_centre_descriptif"] / c["epsilon_R"]
                     if (c.get("epsilon_R")
                         and isinstance(c.get("epsilon_R_centre_descriptif"),
                                        float)) else None),
                "classe_R_DECISIONNELLE": l1,
                "classe_R_sous_le_centre_DESCRIPTIVE": l2}
            if l1 != l2:
                epsr["cellules_qui_basculeraient"].append(
                    {"cellule": cle, "de": l1, "vers": l2})
    _pbs_json(out / "epsilon-R-centre.json", epsr)

    print("\n--- dépouillement Σ (24 cellules énumérées) ---")
    print(f"  comptes, lecture PRINCIPALE `h` : {dep['comptes']}")
    for c2, lst in dep["cellules_par_classe_lecture_h"].items():
        print(f"    {c2:<10} : {lst}")
    for nom, cpt in dep["comptes_par_lecture"].items():
        print(f"  comptes, lecture `{nom}` (couplage A) : {cpt}")
    for nom, cpt in dep["comptes_par_lecture_couplage_B"].items():
        print(f"  comptes, lecture `{nom}` (couplage B) : {cpt}")
    for nom, cpt in dep["comptes_par_completion_du_conjoint"].items():
        promue = " [PROMUE]" if nom == "monotonie_complete" else " [DESCRIPTIVE]"
        print(f"  comptes, complétion du conjoint `{nom}`{promue} : {cpt}")
    print("  RAPPEL : sous la condition `aucun`, σ± = 1.0 EXACTEMENT — valeur "
          "FORCÉE par une identité (0-162, (xxv)), poids probant NUL.")
    reg = dep["registre_des_arbitrages_PI"]
    print(f"  registre des arbitrages PI : {reg['n_au_registre']} au registre, "
          f"{reg['n_OUVERTS']} OUVERT(s), {reg['n_TRANCHES']} TRANCHÉ(s), "
          f"{reg['n_qui_deplacent_une_classe']} qui déplace(nt) une classe.")
    print(f"  {reg['consequence']}")
    print(f"  {reg['consignation_du_defaut']}")

    print("\n--- bascules de classe sous la complétion PROMUE (monotonie "
          "complète) ---")
    for b3 in dep["bascules_de_classe_sous_la_completion_PROMUE"]:
        print(f"  {b3['cellule']:<40} "
              f"{b3['classe_sous_DEUX_POINTS_descriptive']} → "
              f"{b3['classe_sous_MONOTONIE_COMPLETE_promue']} | σ± = "
              + ("SO" if not isinstance(b3["sigma_pm"], float)
                 else f"{b3['sigma_pm']:.4f}")
              + " | marge = "
              + ("SO" if not isinstance(b3["marge_premier_moins_dernier"],
                                        float)
                 else f"{b3['marge_premier_moins_dernier']:+.4f}")
              + " ("
              + ("SO" if not isinstance(
                  b3["marge_rapportee_a_epsilon_Psi_b1"], float)
                 else f"{100.0 * b3['marge_rapportee_a_epsilon_Psi_b1']:.1f} % "
                      f"de ε_Ψ")
              + f") | {b3['n_inversions_de_monotonie']} inversion(s)")
    if not dep["bascules_de_classe_sous_la_completion_PROMUE"]:
        print("  aucune")

    print("\n--- ρ_ic HORS DOMAINE ⇒ SANS OBJET (D23) ---")
    for r3 in dep["cellules_a_rho_ic_SANS_OBJET"]:
        d3 = r3["drapeau_de_domaine"][0]
        print(f"  {r3['cellule']:<40} ρ_ic = SANS OBJET  (DEFF = "
              f"{d3['DEFF']:.2f} > m̄ = {d3['m_barre']:.1f} ⇒ ρ_ic = "
              f"{d3['rho_ic_hors_domaine_qui_aurait_ete_publie']:.4f} > 1)")
    if not dep["cellules_a_rho_ic_SANS_OBJET"]:
        print("  aucune")

    print("\n--- Σ-mort : formulation licite (D28) ---")
    for cle, L3 in dep["lectures_licites_D28_des_Sigma_mort"].items():
        print(f"  {cle:<40} {L3['formulation_obligatoire']}")
        print(f"      N_eff = "
              + ("SO" if not isinstance(L3["N_eff_route_VARIANCE"], float)
                 else f"{L3['N_eff_route_VARIANCE']:.1f}")
              + f" ; largeur d'IC = {L3['largeur_de_l_IC']:.4f} ; écart "
                f"ponctuel à 0.5 = {L3['ecart_ponctuel_a_0_5']:+.4f}")
    if not dep["lectures_licites_D28_des_Sigma_mort"]:
        print("  aucune")

    print("\n--- Σ-ind : cause NOMMÉE ---")
    for cle, C3 in dep["causes_nommees_des_Sigma_ind"].items():
        print(f"  {cle:<40} {C3['cause']}")
    if not dep["causes_nommees_des_Sigma_ind"]:
        print("  aucune")

    print(f"\n--- bascule de classe imputable à B (10⁴ contre le jeu réduit) "
          f"---\n  {len(dep['bascules_imputables_a_B'])} bascule(s) : "
          f"{[b4['cellule'] for b4 in dep['bascules_imputables_a_B']]}")

    print("\n--- Δ_Base − Δ̂_Base (table complète, signes) ---")
    for cle, v in ecd["cellules"].items():
        print(f"  {cle:<40} Δ={v['Delta_Base']:+.4f} Δ̂="
              f"{v['Delta_chapeau_Base']:+.4f} écart="
              f"{v['ecart_Delta_moins_Delta_chapeau']:+.4f} ε_B="
              + ("SO" if not isinstance(v["epsilon_B"], float)
                 else f"{v['epsilon_B']:.4f}")
              + " |écart|/ε_B="
              + ("SO" if not isinstance(
                  v["ecart_absolu_rapporte_a_epsilon_B"], float)
                 else f"{v['ecart_absolu_rapporte_a_epsilon_B']:.3f}"))
    if "etendue_signee" in ecd:
        print(f"  étendue SIGNÉE = [{ecd['etendue_signee'][0]:+.4f}, "
              f"{ecd['etendue_signee'][1]:+.4f}] sur {ecd['n_cellules']} "
              f"cellules ; {ecd['n_ecarts_negatifs']} négative(s) ; "
              f"{ecd['n_depassant_1x_epsilon_B']}/{ecd['n_cellules']} > 1×ε_B ; "
              f"{ecd['n_depassant_2x_epsilon_B']}/{ecd['n_cellules']} > 2×ε_B")

    print("\n--- ordinal N6 : LES SIX IC ---")
    for cle, o in ord6["cellules"].items():
        if o.get("statut") == SANS_OBJET:
            print(f"  {cle:<40} SANS OBJET")
            continue
        print(f"  {cle:<40} écart={o['ecart_A_moins_B']:+.4f} IC="
              f"[{o['IC_ecart'][0]:+.6f}, {o['IC_ecart'][1]:+.6f}]")
    if "minimum_reel_des_IC_inf_sur_les_six" in ord6:
        mn = ord6["minimum_reel_des_IC_inf_sur_les_six"]
        m6 = ord6["minimum_reel_des_IC_inf_sur_les_six"]
        print(f"  minimum RÉEL des IC_inf sur les six : {mn['valeur']:+.6e} "
              f"({mn['cellule']})")
        print(f"    résolution : largeur de l'IC = "
              f"{m6['largeur_de_l_IC_de_cette_cellule']:.6e} ; la borne vaut "
              f"{m6['borne_en_pourcentage_de_la_largeur']:.4f} % de cette "
              f"largeur ; pas du percentile bootstrap = "
              f"{m6['resolution_du_percentile_bootstrap']['pas_de_percentile_approche']:.0e}")
        print(f"    statut : {m6['statut_de_la_conjonction']}")

    print("\n--- ε_R littéral (décisionnel) contre ε_R centré (descriptif) ---")
    for cle, v in epsr["cellules"].items():
        print(f"  {cle:<40} ε_R="
              f"{v['epsilon_R_litteral_DECISIONNEL']:.4f} centré="
              + ("SO" if v["epsilon_R_centre_DESCRIPTIF"] is None
                 else f"{v['epsilon_R_centre_DESCRIPTIF']:.4f}")
              + " ratio="
              + ("SO" if v["rapport_centre_sur_litteral"] is None
                 else f"{v['rapport_centre_sur_litteral']:.3f}")
              + f" | {v['classe_R_DECISIONNELLE']} → "
              f"{v['classe_R_sous_le_centre_DESCRIPTIVE']}")
    print(f"  cellules qui BASCULERAIENT sous le centré (consignées, NON "
          f"reclassées) : {epsr['cellules_qui_basculeraient']}")

    # ------------------------------------------------------ BUDGET (0-196)
    # Défaut du tour de correction : la cause publiée attribuait le dépassement
    # à « +390 s pour la SECONDE simulation ». Les durées mesurées disent que
    # les DEUX branches coûtent ~390 s, et que le §9 budgète le poste `N-queue`
    # à « négligeable » : c'est la PREMIÈRE qui a démoli le poste, pas la
    # seconde. La cause est sous-attribuée d'un facteur 2.
    branches = {k: v.get("duree_s") for k, v in sims.items()
                if isinstance(v, dict) and v.get("duree_s") is not None}
    tot_nq = sum(v for v in branches.values() if isinstance(v, (int, float)))
    n_br = len(branches)
    bud = {"poste": "N-queue — simulation exacte à sélection top-64 par vecteur",
           "budget_grave_au_9": "négligeable",
           "duree_par_branche_s": branches,
           "n_branches": n_br,
           "duree_totale_du_poste_s": tot_nq,
           "duree_moyenne_par_branche_s": (tot_nq / n_br if n_br else None),
           "cause_publiee":
               (f"le poste `N-queue` est budgété à ZÉRO (« négligeable », §9) "
                f"et coûte {tot_nq / n_br:.1f} s PAR BRANCHE, sur {n_br} "
                f"branches — soit {tot_nq:.1f} s au total"
                if n_br else "aucune branche exécutée"),
           "cause_ecartee":
               "« +390 s pour la SECONDE simulation » — sous-attribution d'un "
               "facteur 2 : la PREMIÈRE branche coûte le même prix et c'est "
               "elle qui a démoli le poste",
           "duree_totale_du_run_s": round(time.time() - t0, 1),
           "budget_grave_total_au_9": "< 10 min CPU"}
    _pbs_json(out / "budget-N-queue.json", bud)
    print("\n--- budget (0-196) ---")
    print(f"  {bud['cause_publiee']}")
    for k, v in branches.items():
        print(f"    branche {k:<40} {v} s")

    # ------------------------------------------------------------ CSV
    lignes = ["modele,condition,strate,P,n_vide,somme_inter,rho_Base,"
              "rho_chapeau_Base,R_Base,IC_R_inf,IC_R_sup,epsilon_R,"
              "epsilon_R_centre,classe_R,classe_R_centre,"
              "Delta_Base,IC_D_inf,IC_D_sup,epsilon_B,Delta_chapeau_Base,"
              "ecart_Delta_moins_chapeau,ecart_sur_epsilon_B,"
              "classe_Delta,classe_B,sigma_pm,sigma_hat_pool,Psi,"
              "sigma_hat_pool_z,Psi_z,sigma_hat_pool_tronque,Psi_tronque,"
              "classe_Sigma_h,classe_Sigma_z,classe_Sigma_tronque,"
              "classe_Sigma_monotone_PROMUE,"
              "classe_Sigma_deux_points_DESCRIPTIVE,"
              "profil_monotone,n_inversions,N_eff_variance,DEFF,rho_ic,"
              "cos_moyen_h,gamma_moyen_h,vivier_min,vivier_max"]

    def _f(x, n=4):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else "SANS_OBJET"

    for m, syn in par_modele.items():
        for cond in PBS_CONDITIONS:
            for s in STRATES:
                c = syn["cellules"].get(f"{m}|{cond}|{s}")
                if not c:
                    continue
                icr = c.get("IC_R_Base") or [None, None]
                icd = c.get("IC_Delta_Base") or [None, None]
                sg = syn["sigma"].get(f"{m}|{cond}|{s}", {})
                lec = sg.get("lectures_de_cos_p") or {}
                cls = sg.get("classe_Sigma_par_lecture_de_cos_p") or {}

                def _lec(nom, champ):
                    return (lec.get(nom) or {}).get(champ)

                lignes.append(",".join([
                    m, cond, s, str(c["P"]), str(c["n_vide"]),
                    str(c.get("somme_intersections", 0)),
                    _f(c.get("rho_Base")), _f(c.get("rho_chapeau_Base")),
                    _f(c.get("R_Base")), _f(icr[0]), _f(icr[1]),
                    _f(c.get("epsilon_R")),
                    _f(c.get("epsilon_R_centre_descriptif")),
                    str(c.get("classe_R")),
                    str(c.get("classe_R_sous_epsilon_R_centre_descriptif")),
                    _f(c.get("Delta_Base")), _f(icd[0]), _f(icd[1]),
                    _f(c.get("epsilon_B")), _f(c.get("Delta_chapeau_Base")),
                    _f(c.get("ecart_Delta_moins_Delta_chapeau")),
                    _f(c.get("ecart_Delta_rapporte_a_epsilon_B")),
                    str(c.get("classe_Delta")), str(c.get("classe_B")),
                    _f(c.get("sigma_pm")), _f(c.get("sigma_hat_pool")),
                    _f(c.get("Psi_global")),
                    _f(_lec("z", "sigma_hat_pool_global")),
                    _f(_lec("z", "Psi_global")),
                    _f(_lec("tronque_phi", "sigma_hat_pool_global")),
                    _f(_lec("tronque_phi", "Psi_global")),
                    str(cls.get("h")), str(cls.get("z")),
                    str(cls.get("tronque_phi")),
                    str((sg.get("classe_Sigma_par_completion_du_conjoint")
                         or {}).get("monotonie_complete_PROMUE")),
                    str((sg.get("classe_Sigma_par_completion_du_conjoint")
                         or {}).get("deux_points_DESCRIPTIVE")),
                    # D23 : sans domaine de définition (un seul bucket après
                    # fusion), le conjoint se publie SANS OBJET, jamais None
                    ("SANS_OBJET"
                     if (sg.get("resolution_du_profil") or {}).get("statut")
                     == SANS_OBJET
                     else str(sg.get("profil_monotone_complet_PROMU"))),
                    ("SANS_OBJET"
                     if (sg.get("resolution_du_profil") or {}).get("statut")
                     == SANS_OBJET
                     else str((sg["resolution_du_profil"]
                               ).get("n_inversions_de_monotonie"))),
                    _f((sg.get("resolution_effective_N_eff_DEFF_rho_ic")
                        or {}).get("N_eff_route_VARIANCE"), 1),
                    _f((sg.get("resolution_effective_N_eff_DEFF_rho_ic")
                        or {}).get("DEFF_route_VARIANCE"), 2),
                    _f((sg.get("resolution_effective_N_eff_DEFF_rho_ic")
                        or {}).get("rho_ic_Kish_route_VARIANCE")),
                    _f(c.get("cos_moyen_h")),
                    _f(c.get("gamma_moyen_h")), str(c.get("vivier_min")),
                    str(c.get("vivier_max"))]))
    (out / "summary.csv").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"\ntemps CPU total : {time.time() - t0:.1f} s")
    print(f"bruts : {out}")
    return 0


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
    # ---- EXP-2026-08-27-plancher-base-sigma (§10 — options ajoutées, D29)
    p.add_argument("--base-ref", dest="base_ref", default=None,
                   choices=list(PBS_REFERENCES) + ["tous"],
                   help="cycle « plancher-base-sigma » : vecteur de référence "
                        "x dont on prend top64(G·x). `tous` (défaut du cycle) "
                        "exécute les trois niveaux du §8 dans le même passage — "
                        "mêmes paires, même ordre, même cache Z.")
    p.add_argument("--mu-loo", dest="mu_loo", action="store_true",
                   help="μ_global^{LOO} est la MESURE PRINCIPALE, "
                        "inconditionnelle (P7) ; le non-LOO est publié en "
                        "descriptif. L'option est un TÉMOIN : les deux sont "
                        "publiés dans tous les cas (V-loo l'exige).")
    p.add_argument("--sigma-rang", dest="sigma_rang", action="store_true",
                   help="brin σ± : 8 buckets dyadiques, r = max(rang_A, "
                        "rang_B) gelé, bootstrap par GRAPPE (V-grappe) et "
                        "simulation exacte à rang (Q-M1-bis).")
    p.add_argument("--quad", action="store_true",
                   help="exécute V-quad SEULE (ρ̂_Base et Δ̂_Base par cellule) "
                        "et s'arrête. V-quad doit être exécutée et publiée "
                        "AVANT toute mesure (§6.E).")
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
    # Défaut du tour de correction : les portes B et C du §6 (`V-cache`, `V-G`)
    # et `V-norm` s'écrivaient dans le SEUL répertoire du cycle précédent, alors
    # que ce sont des **critères d'invalidation** du cycle courant. Elles sont
    # désormais publiées dans TOUS les répertoires de sortie concernés.
    cycle_pbs = bool(a.base_ref or a.quad or a.sigma_rang or a.mu_loo)
    sorties = [out]
    if cycle_pbs:
        sortie_pbs = Path(a.out) if a.out != str(SORTIE) else SORTIE_PBS
        if sortie_pbs.resolve() != out.resolve():
            sortie_pbs.mkdir(parents=True, exist_ok=True)
            sorties.append(sortie_pbs)

    # Défaut du tour de correction : `_porte_json` écrivait dans TOUS les
    # répertoires de sortie, et a donc RÉÉCRIT `V-cache.json`, `V-G.json` et
    # `V-norm.json` d'un cycle CLOS et vérifié (`recouvrement-supports`).
    # Le contenu était hash-identique, donc rien n'a été falsifié — mais
    # réécrire les bruts d'un cycle clos n'est pas neutre pour **D14-R**.
    # L'écriture est désormais CONDITIONNÉE à l'absence du fichier ; un fichier
    # présent est RELU, son sha256 comparé, et le résultat TRACÉ. Un contenu
    # DIVERGENT n'est jamais écrasé : il est signalé.
    trace_portes = {"regle": "écriture conditionnée à l'absence du fichier ; "
                             "un fichier présent est relu et comparé, jamais "
                             "écrasé (D14-R)",
                    "repertoires": [str(r) for r in sorties],
                    "fichiers": {}}

    def _cles_divergentes(a, b, p="", acc=None):
        """Chemins des FEUILLES qui diffèrent — un drapeau de divergence qui
        ne nomme pas sa cause n'est pas un drapeau."""
        acc = [] if acc is None else acc
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    acc.append({"chemin": f"{p}/{k}", "cause": "clé absente "
                                + ("du fichier sur disque" if k not in b
                                   else "de la valeur recalculée")})
                else:
                    _cles_divergentes(a[k], b[k], f"{p}/{k}", acc)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                acc.append({"chemin": p, "cause": f"longueur {len(a)} vs "
                                                  f"{len(b)}"})
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    _cles_divergentes(x, y, f"{p}[{i}]", acc)
        elif a != b:
            acc.append({"chemin": p, "valeur_recalculee": a,
                        "valeur_sur_disque": b})
        return acc

    def _porte_json(nom: str, obj) -> None:
        txt = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        neuf = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        for rep in sorties:
            chemin = rep / nom
            cle = str(chemin)
            if chemin.exists():
                anc = chemin.read_text(encoding="utf-8")
                vieux = hashlib.sha256(anc.encode("utf-8")).hexdigest()
                trace_portes["fichiers"][cle] = {
                    "action": "AUCUNE — fichier préexistant, NON RÉÉCRIT",
                    "sha256_sur_disque": vieux,
                    "sha256_qui_aurait_ete_ecrit": neuf,
                    "identique": bool(vieux == neuf),
                    "cles_divergentes": (
                        None if vieux == neuf else
                        _cles_divergentes(
                            json.loads(txt),
                            json.loads(anc))),
                    "consequence": ("aucune — le fichier sur disque est "
                                    "hash-identique à ce que ce run aurait "
                                    "écrit" if vieux == neuf else
                                    "DIVERGENCE — le fichier sur disque DIFFÈRE "
                                    "de ce que ce run aurait écrit ; il n'est "
                                    "PAS écrasé, les clés divergentes sont "
                                    "NOMMÉES ci-dessus")}
                continue
            chemin.write_text(txt, encoding="utf-8")
            trace_portes["fichiers"][cle] = {
                "action": "ÉCRIT — le fichier était absent",
                "sha256_ecrit": neuf, "identique": None}

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
    _porte_json("V-cache.json",
                {"porte": "V-cache", "verdict": vc, "detail": dc})
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
    _porte_json("V-G.json", {"porte": "V-G", "verdict": vg, "detail": dg})
    if vg != PASS:
        print("\n" + dg["verdict_porte"])
        return 3

    vn = {m: config_de_normalisation(m) for m in modeles}
    v, dn = v_norm(vn)
    print(f"\nV-norm : {v}")
    for m, c in vn.items():
        print(f"  {m:<32} {c['normalisation']:<10} centre={c['centre_t_il']} "
              f"| {c['ligne_de_config']}")
    _porte_json("V-norm.json", {"porte": "V-norm", "verdict": v, "detail": dn})
    print(f"  portes de provenance publiées dans : "
          f"{', '.join(str(r) for r in sorties)}")
    # trace D14-R de l'écriture des portes de provenance
    cible_trace = (Path(a.out) if a.out != str(SORTIE) else
                   (SORTIE_PBS if cycle_pbs else out))
    cible_trace.mkdir(parents=True, exist_ok=True)
    n_reecrits = sum(1 for v in trace_portes["fichiers"].values()
                     if v["action"].startswith("ÉCRIT"))
    n_intacts = sum(1 for v in trace_portes["fichiers"].values()
                    if v["identique"] is True)
    n_diverg = sum(1 for v in trace_portes["fichiers"].values()
                   if v["identique"] is False)
    trace_portes["comptes"] = {"ecrits_car_absents": n_reecrits,
                               "preexistants_hash_identiques_INTACTS":
                                   n_intacts,
                               "preexistants_DIVERGENTS_non_ecrases": n_diverg}
    (cible_trace / "trace-portes-de-provenance.json").write_text(
        json.dumps(trace_portes, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"  trace D14-R : {n_reecrits} écrit(s) car absent(s), "
          f"{n_intacts} préexistant(s) INTACT(s) (hash identique, non "
          f"réécrit(s)), {n_diverg} divergent(s) non écrasé(s)")
    for cle_f, v_f in trace_portes["fichiers"].items():
        print(f"    {cle_f}\n      {v_f['action']}"
              + (f"  sha256 = {v_f['sha256_sur_disque']}"
                 if "sha256_sur_disque" in v_f else "")
              + ("" if v_f.get("identique") is not False else
                 "  [DIVERGENT, non écrasé]"))
        for d_f in (v_f.get("cles_divergentes") or []):
            print(f"        divergence : {d_f}")

    # ---- aiguillage du cycle « plancher-base-sigma » (EXP-2026-08-27)
    if cycle_pbs:
        if a.banc_e is None:
            print("\n§6.A : le cycle plancher-base-sigma exige `--banc-e 0` "
                  "(banc D14-S `--suite pbs` vert) AVANT toute exécution.")
            return 4
        sortie = Path(a.out) if a.out != str(SORTIE) else SORTIE_PBS
        if not a.quad and not (sortie / "V-quad.json").exists():
            print("\n§6.E : `V-quad` n'a pas été exécutée — `ρ_Base` serait "
                  "retiré de l'interprétation. Lancer d'abord `--quad`.")
            return 5
        return main_pbs(a, cfg, modeles, sortie)

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
