# SPDX-License-Identifier: AGPL-3.0-or-later
"""v4-matériel — noyau décisionnel et mesure (protocole `EXP-2026-08-23-v4-materiel.md`).

Deux moitiés, strictement séparées :

  * **le noyau décisionnel** (partitions, enveloppes nulles, portes) — CPU pur,
    aucune donnée, aucun modèle : c'est lui que le banc D14-S exerce clause par
    clause, **avant tout GPU** ;
  * **la mesure** (`run_mesure`) — un seul passage sur les trois modèles, exécutée
    **uniquement** si le banc rend `E = 0` et si la génération rend PASS.

Interdits structurels rappelés ici parce qu'ils sont vérifiables dans ce fichier :
`M` n'est **jamais** instanciée, aucune injection, aucun backprop, `engram/` non
modifié, `E3` sans objet. La NLL du token de capture est conservée en **scalaire**
(jamais le tenseur de logits : 870 × 151k fp32 ≈ 2 Gio sur Qwen).

Ordres d'évaluation **GRAVÉS** (exclusivité D18) — ils sont l'objet même des
défauts 0-68, 0-78, 0-87, 0-88 :
  calibrateur : `INVALIDE-INSTRUMENT → N-b → N-a → N-ind`
  bandes      : `famine → C+ → C− → C-0 → C-ind`
  maillons    : `+ → − → 0-résolu → ind`  (`ind` par COMPLÉMENTATION, en dernier)

Schéma **1× / 2×** (défaut 0-81) : marge de significativité à 1× l'enveloppe
nulle, couloir d'équivalence à 2×. Une classe d'équivalence ne dit **jamais**
« pas d'effet » : elle dit « effet borné par 2× la résolution ».
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import pool_v4 as p4  # noqa: E402
from engram.config import EngramConfig  # noqa: E402

PASS, FAIL = "PASS", "FAIL"

# =========================================================================
#  Constantes du protocole §7 — recopiées, jamais ajustées après mesure.
# =========================================================================

TAU = 0.15                 # §4.4, confirmé par Math (recalcul 0.229/√10 × 2 = 0.145)
EPS_MAX = 0.66             # §4.5-3 — plafond dérivé ; toute valeur MC au-dessus
                           # est une ERREUR DE PIPELINE, pas un résultat
B_MC = 10_000              # §4.5-1 et §4.6 — B = 10⁴
B_BOOT = 10_000            # bootstrap de TIGES
B_JOINT = 10_000           # `V-joint` — B ≥ 2000 exigé
ALPHA = 0.05               # alpha propre
Q_ENVELOPPE = 0.95         # `ε = q₀.₉₅(|D̄_null|)` — LIGNE CANONIQUE UNIQUE
SOMME_M_MIN = 60           # §4.3 — clause de famine globale (D19)
K_SUPPORT_MIN = 9          # §4.5 — `K_eff^support ≤ 8` ⇒ `C-ind` d'office
K_EFF = p4.K_EFF           # 10
M1_TIGE = p4.M1_ATTENDU    # 5
TAILLE_POOL = p4.TAILLE_POOL   # 36
P_TIGE = M1_TIGE / TAILLE_POOL  # 5/36
NLL_BANDES = 3             # `V-surprise` — dérivé, non préféré (0-52)
NLL_SEUIL_ABSOLU = 4.0     # superposition secondaire = `thr` du projet
DTYPE_FORWARD = "bfloat16"  # D21 — épinglé ; repli fp32 = CHEMIN NOMINAL

# Plancher lexical dérivé du matériau v4 (§3) : constantes SANS variance
# d'échantillonnage — elles se publient, elles ne se testent pas.
PLANCHER_LEXICAL = (36, 37)          # 36/37, en FRACTION
HASARD_TIGE_RESTREINT = (1, 6)       # 1/6, en FRACTION

MODELES = ("gpt2", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-1.5B")
L_ATTENDU = {"gpt2": 12, "HuggingFaceTB/SmolLM2-360M": 32, "Qwen/Qwen2.5-1.5B": 28}
COUCHE_REF = {"gpt2": 6, "HuggingFaceTB/SmolLM2-360M": 16, "Qwen/Qwen2.5-1.5B": 14}

# Vocabulaire INTERDIT (§2, entrées i à xiii). `V-plafond` et `V-calib` sont des
# portes de SCHÉMA : elles échouent si l'un de ces termes apparaît.
VOCABULAIRE_INTERDIT = (
    "séparation de patterns", "separation de patterns",
    "adressage", "adressable", "non lexical", "non-lexical",
    "récupération par la clé", "recuperation par la cle",
    "plan 2 × 2 exact", "plan 2x2 exact",
    "les ponts remplissent la cellule manquante",
    "distingue l'unité", "distingue l'unite",
    "tendance", "suggère", "suggere", "va dans le sens de",
)

# Phrases GRAVÉES, obligatoires, à recopier telles quelles.
PHRASE_XIII = ("compatible avec un encodage de surface ; le canal identité "
               "n'est pas adressé par cette primaire")
PHRASE_PERIMETRE = ("la primaire 1 ne peut, à elle seule, distinguer un "
                    "adressage représentationnel d'un transcript lexical ; "
                    "cette distinction n'est pas au périmètre de ce run")
PHRASE_G_BIS = ("l'instrument est sain, la question posée est restée sans "
                "réponse")

# Les trois phrases gravées ci-dessus sont exemptées du balayage : elles NOMMENT
# la limite (§4.9-2), elles ne l'emploient jamais comme évidence.
_PHRASES_GRAVEES = (PHRASE_PERIMETRE, PHRASE_XIII, PHRASE_G_BIS)

# Opérationnalisations DÉCLARÉES (le protocole fixe la clause, pas la
# statistique d'exécution). Publiées, hors E, jamais ajustées après lecture.
OPERATIONNALISATIONS = {
    "forme stratifiée-combinée de M1/M2/M3": (
        "le protocole grave « Σ_domaines (observé − attendu-dans-la-strate) », "
        "sans dire à quelle strate appartient une paire INTER-domaines. Déclaré : "
        "une moyenne de strate est la moyenne NON PONDÉRÉE des moyennes "
        "intra-strate, les strates étant les 4 DOMAINES pour les quantités à "
        "domaine identique (S3, S1) et les 6 COUPLES de domaines pour les "
        "quantités à domaine différent (S2, S0). Rien n'est poolé ; la nulle de "
        "permutation intra-tige est calculée sur la MÊME fonctionnelle."),
    "ΔR1_inv — R1": (
        "`R1` est l'indicatrice de rang 1 parmi les 37 candidats (36 du pool `C5` "
        "gelé + la cible), égalités en MID-RANK (condition E2) : à `g` ex æquo en "
        "tête, la cible compte `1/g`."),
    "clé nulle appariée": (
        "la clé nulle de la requête `i` est l'état de la paire de cadre "
        "`i mod 40` dans le MÊME moule — même position de capture, même longueur "
        "en tokens (0-58). Appariement déterministe, déclaré, sans aléa."),
    "clusters du calibrateur": (
        "§4.4 fixe `K_eff = 10` : les clusters sont les 10 tiges DÉCISIONNELLES, "
        "et la statistique porte sur les 60 unités décisionnelles × 3 types = 180 "
        "requêtes. `C5` (« 72/72 requêtes ») est une CONDITION de génération "
        "vérifiée sur les 72 entités ; elle ne définit pas le support de la "
        "statistique. Inclure la réserve donnerait `K_eff = 14` et resserrerait "
        "l'IC sous la dérivation de `τ`."),
    "unité de requête": (
        "une requête est (unité, cellule de requête) ; la cellule de requête est "
        "l'AUTRE variante du MÊME type (invariance mesurée à trois "
        "transformations nommées, §8)."),
}


# =========================================================================
#  Instrumentation du RUN (issue 3 du Vérifieur) — `V-pool` et `V-t1` ne sont
#  plus des AFFIRMATIONS mais des lectures de l'état réel du pipeline.
#
#  Un futur run qui recalculerait un rang à l'intérieur d'un rééchantillon
#  bootstrap, ou qui calculerait une statistique intra-tige à `t−1`, doit
#  ÉCHOUER — pas passer parce que l'argument codé en dur dit qu'il ne le fait
#  pas. Les compteurs ci-dessous sont alimentés au point d'exécution.
# =========================================================================

_INSTR = {"profondeur_bootstrap": 0, "rangs_calcules": 0,
          "rangs_calcules_dans_un_bootstrap": 0, "quantites_t1": []}


def reinitialiser_instrumentation() -> None:
    """À appeler en tête de chaque modèle : les compteurs sont PAR RUN."""
    _INSTR["profondeur_bootstrap"] = 0
    _INSTR["rangs_calcules"] = 0
    _INSTR["rangs_calcules_dans_un_bootstrap"] = 0
    _INSTR["quantites_t1"] = []


def instrumentation() -> dict:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in _INSTR.items()}


class _SousBootstrap:
    """Contexte marquant un rééchantillonnage. Tout rang calculé sous ce
    contexte est un rang recalculé DANS un échantillon — ce que `V-pool`
    interdit (inférence conditionnelle au pool gelé)."""

    def __enter__(self):
        _INSTR["profondeur_bootstrap"] += 1
        return self

    def __exit__(self, *exc):
        _INSTR["profondeur_bootstrap"] -= 1
        return False


def _noter_rang(n: int = 1) -> None:
    _INSTR["rangs_calcules"] += n
    if _INSTR["profondeur_bootstrap"] > 0:
        _INSTR["rangs_calcules_dans_un_bootstrap"] += n


def noter_quantite_t1(nom: str, paires, tige_par_indice) -> dict:
    """Enregistre une quantité calculée à `t−1` avec le fait — CONSTATÉ, non
    déclaré — que son **ensemble de comparaison** contient ou non deux unités de
    même tige. `V-t1` lit ce registre.

    `paires` est la liste des couples effectivement COMPARÉS : c'est la bonne
    granularité, car un sous-ensemble d'unités peut contenir deux membres d'une
    même tige sans jamais les comparer entre eux.
    """
    paires = [tuple(x) for x in paires]
    intra = any(tige_par_indice[a] == tige_par_indice[b] for a, b in paires)
    n_intra = sum(1 for a, b in paires if tige_par_indice[a] == tige_par_indice[b])
    q = {"nom": nom, "n_paires_comparees": len(paires),
         "n_paires_intra_tige": n_intra, "intra_tige": bool(intra)}
    _INSTR["quantites_t1"].append(q)
    return q


def paires_de_lensemble(indices):
    """Toutes les paires d'un ensemble de comparaison « tous contre tous »."""
    idx = list(indices)
    return [(idx[a], idx[b]) for a in range(len(idx))
            for b in range(a + 1, len(idx))]


# =========================================================================
#  Barrière d'information (`V-compo`) — scellement, publication, descellement
# =========================================================================

class Barriere:
    """D19 rendue EXÉCUTABLE : la quantité observée est **scellée** (hashée sans
    lecture), l'enveloppe nulle est **publiée**, puis la quantité est
    **descellée**. Le banc vérifie l'ORDRE ; un `ε` publié après lecture de `D`
    est un run invalide."""

    def __init__(self):
        self.journal = []
        self._valeur = None
        self._scelle = False

    def sceller(self, valeur) -> str:
        h = hashlib.sha256(repr(valeur).encode()).hexdigest()[:16]
        self._valeur = valeur
        self._scelle = True
        self.journal.append(("sceller", h))
        return h

    def publier(self, enveloppe):
        """`enveloppe = None` = SANS OBJET (nulle sans domaine de definition) :
        l'ordre de la barriere est enregistre quand meme, seule la VALEUR manque."""
        if not self._scelle:
            raise RuntimeError("publication d'enveloppe avant scellement")
        v = None if enveloppe is None else float(enveloppe)
        self.journal.append(("publier", v))
        return v

    def desceller(self):
        if [x[0] for x in self.journal] != ["sceller", "publier"]:
            raise RuntimeError(f"ordre de barrière violé : {self.journal}")
        self.journal.append(("desceller", None))
        self._scelle = False
        return self._valeur

    def ordre(self) -> list:
        return [x[0] for x in self.journal]


def v_compo(ordre, somme_m, exclusion_m0_declaree: bool, eps: float,
            eps_max: float = EPS_MAX) -> tuple:
    """`V-compo` : nulle MC seedée **exécutée et publiée avant** lecture de `D` ;
    `Σ_q m_q` publié ; exclusion `m = 0` conforme à la déclaration ; `ε ≤ ε_max`."""
    attendu = ["sceller", "publier", "desceller"]
    ok = (list(ordre) == attendu and exclusion_m0_declaree
          and somme_m is not None and eps is not None and eps <= eps_max)
    return (PASS if ok else FAIL), {
        "ordre_barriere": list(ordre), "ordre_attendu": attendu,
        "somme_m_publiee": somme_m, "exclusion_m0_predeclaree": exclusion_m0_declaree,
        "epsilon": eps, "epsilon_max": eps_max,
        "epsilon_sous_plafond": (eps is not None and eps <= eps_max)}


# =========================================================================
#  Partition 1 — CALIBRATEUR (§4.4). Ordre gravé, exclusivité D18.
# =========================================================================

def classe_calibrateur(ic, tau: float = TAU) -> str:
    """`INVALIDE-INSTRUMENT → N-b → N-a → N-ind` — la PREMIÈRE satisfaite emporte."""
    lo, hi = float(ic[0]), float(ic[1])
    if hi < -tau:                       # exigence de MAGNITUDE (défaut 0-86/D10)
        return "INVALIDE-INSTRUMENT"
    if lo > 0.0:                        # seuil d'EXISTENCE — verdict bénin
        return "N-b"
    if -2 * tau <= lo and hi <= 2 * tau:   # couloir d'ÉQUIVALENCE (2×, défaut 0-81)
        return "N-a"
    return "N-ind"


LECTURE_CALIBRATEUR = {
    "N-b": ("l'état porte AU MOINS de la surface encodée — " + PHRASE_XIII),
    "N-a": "écart BORNÉ par la résolution : |Δ| < 2τ établi",
    "N-ind": "indécidable ICI, matériel ou K_eff insuffisant ⇒ augmenter la résolution",
    "INVALIDE-INSTRUMENT": ("c'est l'INSTRUMENT qui est accusé, pas le cortex ⇒ "
                            "arrêt, retour au banc"),
}


# =========================================================================
#  Partition 2 — BANDES DE LA PRIMAIRE (§4.5). Ordre gravé.
# =========================================================================

def bande_primaire(ic, eps: float, somme_m: int, k_support: int) -> str:
    """`famine → C+ → C− → C-0 → C-ind`.

    Famine GLOBALE : `Σ_q m_q < 60`. Famine PARTIELLE : `K_eff^support ≤ 8`
    (défaut 0-74 — une somme qui saute les `q` indéfinis change son support)."""
    if somme_m < SOMME_M_MIN or k_support < K_SUPPORT_MIN:
        return "C-ind"
    lo, hi = float(ic[0]), float(ic[1])
    if lo > eps:
        return "C+"
    if hi < -eps:
        return "C−"
    if -2 * eps <= lo and hi <= 2 * eps:
        return "C-0"
    return "C-ind"


LECTURE_BANDE = {
    "C+": "excès d'intrusions tige-partagées — décrément lure-spécifique mesuré "
          "comme composition",
    "C−": "déplétion — la géométrie sépare activement les voisins de surface",
    "C-0": "composition BORNÉE par la résolution : |effet| < 2ε établi",
    "C-ind": "indécidable ICI — suite selon la cause (§6.G)",
}


def cause_c_ind(classe_calib: str, somme_m: int, k_support: int) -> dict:
    """§6.G — les SIX causes de `C-ind`, aucune muette (défauts 0-71, 0-80,
    0-85, 0-89, 0-99). Le discriminant est la CLASSE du calibrateur, jamais une
    dichotomie « bas / haut »."""
    if somme_m >= SOMME_M_MIN and k_support < K_SUPPORT_MIN:
        return {"cause": "famine PARTIELLE",
                "suite": "le support s'est effondré sur une minorité de clusters ; "
                         "la suite se lit sur la distribution des m_q publiée",
                "jamais": "« augmenter la résolution » par défaut"}
    if somme_m < SOMME_M_MIN:
        table = {
            "N-b": ("famine par SATURATION", "réduire la dominance de surface"),
            "N-a": ("famine par PUISSANCE", "augmenter la résolution"),
            "N-ind": ("ni la question ni l'instrument ne sont résolus",
                      "re-qualifier l'instrument AVANT de re-mesurer"),
            "INVALIDE-INSTRUMENT": ("la chaîne de mesure est en cause",
                                    "arrêt (§6.F-bis) ; aucune lecture de la primaire"),
        }
        c, s = table[classe_calib]
        return {"cause": c, "suite": s, "classe_calibrateur": classe_calib}
    return {"cause": "C-ind DE RÉSOLUTION (hors famine) — la classe du "
                     "calibrateur ne discrimine rien ici",
            "suite": "effet vrai dans la zone morte déclarée (≈ε à ≈3ε) ⇒ "
                     "augmenter la résolution (K_eff, requêtes contributives)"}


def phrase_g_bis_licenciee(classe_calib: str, bande: str) -> bool:
    """§6.G-bis, portée STRICTE (défaut D7) : la phrase n'est licenciée que par
    la seule cellule conjointe `N-b × C-ind`."""
    return classe_calib == "N-b" and bande == "C-ind"


# =========================================================================
#  Partition 3 — MAILLONS puis ORDRES (§4.6). `ind` par complémentation.
# =========================================================================

ETATS_MAILLON = ("+", "−", "0-résolu", "ind")


def etat_maillon(ic, eps_m: float) -> str:
    """`+ → − → 0-résolu → ind`. `ind` est défini par COMPLÉMENTATION et évalué
    en DERNIER (défaut 0-78) ; les seuils sont `±ε_M` / `±2ε_M` (défaut 0-88),
    jamais 0 (défaut 0-96)."""
    lo, hi = float(ic[0]), float(ic[1])
    if lo > eps_m:
        return "+"
    if hi < -eps_m:
        return "−"
    if -2 * eps_m <= lo and hi <= 2 * eps_m:
        return "0-résolu"
    return "ind"


def classe_ord(m1: str, m2: str, m3: str) -> str:
    """Six classes. `ORD-ind` absorbe tout maillon en état `ind` (règle de
    routage) ; l'espace résolu compte exactement 27 cellules."""
    if "ind" in (m1, m2, m3):
        return "ORD-ind"
    if m2 == "−":
        return "ORD-3"
    if m2 == "0-résolu":
        return "ORD-0"
    # m2 == "+" : le bloc est subordonné à M3, contrôle de manipulation de `C7`
    if m3 == "+":
        return "ORD-1" if m1 == "+" else "ORD-4"
    return "ORD-2"


VERDICT_ORD = {
    "ORD-1": ("chaîne complète S3 > S2 > S1 > S0",
              "les deux facteurs sont instanciés et ordonnés"),
    "ORD-2": ("le facteur domaine n'a pas été instancié — retour au matériau",
              "JAMAIS « le token domine, comme prédit »"),
    "ORD-3": ("le domaine domine la tige ⇒ géométrie sémantique, C5 insuffisante",
              "retour au banc ; la mesure ne s'interprète pas"),
    "ORD-0": ("l'effet de tige à la capture est BORNÉ par 2ε_M — lecture "
              "d'équivalence",
              "même SUITE qu'ORD-3, verdict DISTINCT : la cause candidate est le "
              "locus de capture ⇒ déplacer la capture OU resserrer ε_M en "
              "augmentant K_eff ; ne change pas le pool"),
    "ORD-4": ("domaine instancié, chaîne incomplète",
              "rapporter le maillon manquant ; pas d'ordre global"),
    "ORD-ind": ("ordre indécidable ICI",
                "augmenter la résolution — jamais « retour au matériau »"),
}


def espace_ord_resolu() -> dict:
    """Recompté PAR EXÉCUTION, jamais par affirmation (défaut 0-77)."""
    etats = ("+", "−", "0-résolu")
    c = {}
    for m1 in etats:
        for m2 in etats:
            for m3 in etats:
                c[classe_ord(m1, m2, m3)] = c.get(classe_ord(m1, m2, m3), 0) + 1
    c["total"] = sum(v for k, v in c.items() if k != "total")
    return c


# =========================================================================
#  Enveloppes nulles — `ε` (primaire) et `ε_M` (maillons)
# =========================================================================

def _quantile(x, q: float) -> float:
    return float(np.quantile(np.asarray(x, dtype=np.float64), q))


def d_observe(x_par_requete, m_par_requete) -> dict:
    """`D = Σ_q (X_q − m_q·5/36)` — une DIFFÉRENCE observé − attendu (D16).

    `E3` déclarée : les requêtes à `m = 0` sont EXCLUES, et aucune autre
    sélection sur `m` n'est permise (défaut 0-65)."""
    x = np.asarray(x_par_requete, dtype=np.float64)
    m = np.asarray(m_par_requete, dtype=np.float64)
    garde = m >= 1                                   # E3, PRÉ-DÉCLARÉE
    d = float(np.sum(x[garde] - m[garde] * P_TIGE))
    n_eff = int(garde.sum())
    return {"D": d, "D_normalise_par_requete": d / n_eff if n_eff else float("nan"),
            "n_eff": n_eff, "somme_m": int(m.sum()),
            "distribution_m": np.bincount(m.astype(int)).tolist()}


def eps_nulle_composition(m_par_requete, tige_par_requete, b: int = B_MC,
                          seed: int = 0) -> dict:
    """`ε = q₀.₉₅(|D̄_null|)` — LIGNE CANONIQUE UNIQUE (§4.5-1, §7).

    Enveloppe BILATÉRALE à 95 % de la nulle MC seedée de l'hypergéométrique
    **par requête**, agrégée **par tiges** (`K_eff = 10`, D19), `B = 10⁴`.
    Sous normalité `= 1.96·σ_null`.
    """
    rng = np.random.default_rng(seed)
    m = np.asarray(m_par_requete, dtype=np.int64)
    garde = m >= 1
    m = m[garde]
    tig = np.asarray(tige_par_requete)[garde]
    n_eff = len(m)
    if n_eff == 0:
        # Aucune enveloppe n'existe : la nulle est CONDITIONNELLE aux `m_q` et son
        # domaine de definition est vide. On rend `None`, jamais un NaN (D23).
        return {"epsilon": None, "n_eff": 0, "B": b, "seed": seed,
                "raison": "support vide"}
    # X | m ~ Hypergéom(36, 5, m) : 5 « succès » (tige-partagés) dans 36 tirages
    tirages = np.empty((b, n_eff), dtype=np.float64)
    for j in range(n_eff):
        tirages[:, j] = rng.hypergeometric(M1_TIGE, TAILLE_POOL - M1_TIGE,
                                           int(m[j]), size=b)
    exces = tirages - m[None, :] * P_TIGE
    d_bar = exces.sum(axis=1) / n_eff
    eps = _quantile(np.abs(d_bar), Q_ENVELOPPE)
    return {"epsilon": eps, "sigma_null": float(d_bar.std(ddof=1)),
            "n_eff": n_eff, "B": b, "seed": seed,
            "formule": "ε = q₀.₉₅(|D̄_null|), enveloppe bilatérale à 95 %",
            "clusters": sorted(set(map(str, tig))),
            "epsilon_max": EPS_MAX, "sous_plafond": eps <= EPS_MAX}


def permutations_intra_tige(n_membres: int = 6, n_famille: int = 3) -> int:
    """`C(6,3) = 20` partitions par tige (§4.6)."""
    return math.comb(n_membres, n_famille)


def eps_m_permutation(stat_null_echantillons, b: int = B_MC) -> dict:
    """`ε_M = q₀.₉₅(|Δcos̄_null|)` — enveloppe bilatérale à 95 % de la nulle de
    permutation intra-tige, PAR MAILLON et PAR MODÈLE (défaut 0-79 : aucune
    constante absolue n'existe, en poser une serait 0-52 sous une autre forme)."""
    e = np.asarray(stat_null_echantillons, dtype=np.float64)
    return {"epsilon_M": _quantile(np.abs(e), Q_ENVELOPPE),
            "sigma_null": float(e.std(ddof=1)), "B": len(e),
            "formule": "ε_M = q₀.₉₅(|Δcos̄_null|), permutation intra-tige "
                       "stratifiée par domaine, combinée"}


# =========================================================================
#  Inférence — bootstrap de TIGES, IC percentile, `V-joint` max-t
# =========================================================================

def bootstrap_tiges(valeurs_par_tige, stat=np.mean, b: int = B_BOOT,
                    seed: int = 0) -> dict:
    """L'unité d'échange est la TIGE (0-50) : `K_eff = 10`, jamais 20."""
    rng = np.random.default_rng(seed)
    v = np.asarray(valeurs_par_tige, dtype=np.float64)
    k = len(v)
    with _SousBootstrap():
        ech = np.array([stat(v[rng.integers(0, k, k)]) for _ in range(b)])
    lo, hi = np.quantile(ech, [ALPHA / 2, 1 - ALPHA / 2])
    return {"estime": float(stat(v)), "IC": [float(lo), float(hi)],
            "K_eff": k, "B": b, "seed": seed,
            "demi_largeur": float((hi - lo) / 2),
            "reserve": "à 10 clusters un bootstrap percentile SOUS-COUVRE "
                       "légèrement ; la clause « si l'IC déborde τ, le verdict "
                       "est N-ind, jamais un τ ajusté » est la protection"}


def v_joint(valeurs_par_tige_par_modele, b: int = B_JOINT, seed: int = 0) -> tuple:
    """`V-joint` — IC SIMULTANÉS par enveloppe bootstrap jointe (max-t).

    Ni min-p (qui réintroduit des p-valeurs par modèle, l'objet même de
    l'interdit 0-63), ni Bonferroni (qui ignore la dépendance que le bootstrap
    joint capture). **Un seul rééchantillonnage de tiges par réplique**, trois
    statistiques sur le MÊME rééchantillon.
    """
    noms = list(valeurs_par_tige_par_modele)
    V = {j: np.asarray(valeurs_par_tige_par_modele[j], dtype=np.float64) for j in noms}
    k = len(V[noms[0]])
    if any(len(V[j]) != k for j in noms):
        return FAIL, {"raison": "les trois séries n'ont pas le même nombre de tiges"}
    if k < 2 or any(not np.isfinite(V[j]).all() for j in noms):
        return "SANS OBJET", {"raison": f"K = {k} cluster(s) contributif(s), ou "
                                        f"valeurs non finies : l'enveloppe "
                                        f"bootstrap jointe n'a pas de domaine de "
                                        f"definition"}
    rng = np.random.default_rng(seed)
    obs = {j: float(V[j].mean()) for j in noms}
    ech = {j: np.empty(b) for j in noms}
    with _SousBootstrap():
        for r in range(b):
            idx = rng.integers(0, k, k)             # UN SEUL rééchantillonnage
            for j in noms:
                ech[j][r] = V[j][idx].mean()
    s = {j: float(ech[j].std(ddof=1)) for j in noms}
    M = np.max([np.abs(ech[j] - obs[j]) / (s[j] if s[j] > 0 else 1.0)
                for j in noms], axis=0)
    c = _quantile(M, 1 - ALPHA)
    ic = {j: [obs[j] - c * s[j], obs[j] + c * s[j]] for j in noms}
    conjonction = all(ic[j][0] > 0 or ic[j][1] < 0 for j in noms)
    return PASS, {"c_star": c, "IC_simultanes": ic, "estimes": obs,
                  "conjonction_declaree": conjonction, "B": b, "seed": seed,
                  "regle": "conjonction déclarée ssi 0 est hors des TROIS IC "
                           "simultanés ; le produit de p-valeurs par modèle est "
                           "INTERDIT (0-63, clause C-bis)"}


def v_joint_produit_de_p(p_valeurs) -> tuple:
    """Contre-exemple ÉCHOUANT gravé : « p₁·p₂·p₃ < 0.05 » déclaré significatif."""
    produit = 1.0
    for p in p_valeurs:
        produit *= p
    return FAIL, {"produit": produit, "declare_significatif": produit < ALPHA,
                  "raison": "produit de p-valeurs par modèle — INTERDIT (0-63)"}


# =========================================================================
#  Portes de mesure
# =========================================================================

def v_leak(ic_b0, ic_b0p) -> tuple:
    """`V-leak` : **`B0 ≤ 0` ET `B0′ = 0`** (IC de permutation contenant 0).

    `B0` est un détecteur FAIBLE — sa prédiction ≤ 0 est déjà garantie par `C7`.
    `B0′ = [cos̄(S3) − cos̄(S1)](ℓ=0)`, apparié en domaine, prédit EXACTEMENT 0 :
    c'est **le seul détecteur qui morde** (défaut 0-97).
    « couche 0 » = sortie des embeddings avant le premier bloc (`hidden_states[0]`).
    """
    b0_ok = float(ic_b0[0]) <= 0.0
    b0p_ok = float(ic_b0p[0]) <= 0.0 <= float(ic_b0p[1])
    ok = b0_ok and b0p_ok
    return (PASS if ok else FAIL), {
        "IC_B0": list(map(float, ic_b0)), "B0_inferieur_ou_egal_0": b0_ok,
        "IC_B0prime": list(map(float, ic_b0p)), "B0prime_contient_0": b0p_ok,
        "antipode": "B0 > 0 OU B0′ ≠ 0 ⇒ FUITE ⇒ arrêt, aucune interprétation "
                    "de M1/M2/M3"}


def v_dtype(marges_tete_par_tige, marges_coupure_par_tige, delta_chapeau,
            delta_mesure: bool = True, detail_mesure: dict | None = None) -> tuple:
    """`V-dtype` v2 : **deux marges** (tête et coupure `T`), unité = la TIGE.

    « famille » laisserait passer deux familles de la MÊME tige, c'est-à-dire un
    cluster entier corrompu, sans déclencher (Math, Q5 ; défaut 0-92).
    Le repli fp32 est le **chemin nominal**, pas une exception.

    **`δ̂` doit être MESURÉ** (§4.7, §7 : « bf16 épinglé (D21) ; `m = 60` états
    fp32 ; `δ̂ = max|Δcos|` »). Une CONSTANTE en dur — fût-elle l'ULP bf16 — n'est
    pas la quantité commandée : elle peut être anti-conservatrice, et la porte
    est BLOQUANTE (§6.I). `delta_mesure=False` fait donc échouer la porte.
    """
    if not delta_mesure:
        return FAIL, {"raison": "δ̂ n'a pas été MESURÉ : le protocole commande un "
                                "contrôle fp32 sur m = 60 états et "
                                "δ̂ = max|Δcos|, pas une constante en dur",
                      "delta_chapeau_fourni": float(delta_chapeau),
                      "porte": "§6.I, bloquante"}
    seuil = 2 * float(delta_chapeau)
    touchees = sorted({t for t, m in marges_tete_par_tige.items() if m < seuil}
                      | {t for t, m in marges_coupure_par_tige.items() if m < seuil})
    ok = len(touchees) <= 1
    det = {"delta_chapeau": float(delta_chapeau), "delta_mesure": True,
           "seuil_2delta": seuil,
           "tiges_touchees": touchees, "n_tiges_touchees": len(touchees),
           "unite": "TIGE (cluster), jamais la famille",
           "repli_fp32": "CHEMIN NOMINAL déclaré avant le run (D21 / 0-51)"}
    if detail_mesure:
        det["mesure_de_delta"] = detail_mesure
    return (PASS if ok else "INCONCLUSIF-précision"), det


def v_surprise(signe_m1_par_bande, n_bandes: int = NLL_BANDES) -> tuple:
    """`V-surprise` : M1 n'est créditée que si **son signe est stable sur les
    trois bandes** de NLL. Si M1 n'est présente que dans la bande la plus haute,
    **elle EST le confondant de surprise (0-55) et M1 est RETIRÉE**."""
    signes = list(signe_m1_par_bande)
    stable = len(signes) == n_bandes and len(set(signes)) == 1 and signes[0] != 0
    haute_seule = (len(signes) == n_bandes and signes[-1] != 0
                   and all(s == 0 for s in signes[:-1]))
    return (PASS if stable else FAIL), {
        "signes_par_bande": signes, "stable": stable,
        "M1_retiree": bool(haute_seule),
        "bandes": n_bandes, "coupures": "TERTILES par modèle sur la NLL MOYENNE "
                                        "DE PAIRE, avant toute lecture de M1",
        "superposition_absolue_nats": NLL_SEUIL_ABSOLU,
        "interdit": "toute repondération ou sélection après lecture ⇒ run invalide"}


def v_subst(substitutions, k_eff_republie) -> tuple:
    """`V-subst` (réécrite, défaut 0-69) : substitution d'une famille PONTÉE
    INTERDITE ; réparation au niveau UNITÉ ; `K_eff` recalculé et republié ;
    `K_eff ≤ 8` ⇒ retour au PI."""
    pontee = [s for s in substitutions if s.get("niveau") == "famille"
              and s.get("pontee")]
    ok = (not pontee and k_eff_republie is not None
          and k_eff_republie >= K_SUPPORT_MIN)
    return (PASS if ok else FAIL), {
        "substitutions": substitutions, "substitutions_de_famille_pontee": pontee,
        "K_eff_republie": k_eff_republie,
        "regle": "réserve = stock de SUFFIXES qualifiés par sous-vivier, pas des "
                 "familles ; tige morte ⇒ K_eff = 9 permis ; K_eff ≤ 8 ⇒ PI"}


def v_pool(hash_avant: str, hash_apres: str, rangs_recalcules=None,
           instr: dict | None = None) -> tuple:
    """`V-pool` : pool gelé, hash avant/après, **aucun rang recalculé dans un
    échantillon bootstrap** (inférence CONDITIONNELLE au pool gelé).

    Le troisième terme n'est plus une AFFIRMATION : passé `instr`, il est LU
    dans les compteurs d'exécution (`rangs_calcules_dans_un_bootstrap`). Un run
    qui recalculerait un rang sous `_SousBootstrap` échoue ici (issue 3).
    """
    if instr is not None:
        mesure = instr.get("rangs_calcules_dans_un_bootstrap", 0) > 0
        source = "instrumentation d'exécution"
    else:
        mesure = bool(rangs_recalcules)
        source = "argument (cas de banc synthétique)"
    ok = hash_avant == hash_apres and not mesure
    det = {"hash_avant": hash_avant, "hash_apres": hash_apres,
           "rangs_recalcules_dans_un_bootstrap": mesure, "source": source}
    if instr is not None:
        det["rangs_calcules"] = instr.get("rangs_calcules", 0)
        det["rangs_calcules_dans_un_bootstrap"] = instr.get(
            "rangs_calcules_dans_un_bootstrap", 0)
    return (PASS if ok else FAIL), det


# Les phrases GRAVÉES du protocole contiennent elles-mêmes des termes proscrits
# (« adressage représentationnel » dans la phrase de périmètre) : elles NOMMENT la
# limite, elles ne l'emploient pas comme évidence. Elles sont donc retirées du
# texte AVANT le balayage — sinon la porte échouerait sur la formulation que le
# protocole rend obligatoire.
PHRASES_GRAVEES = _PHRASES_GRAVEES


def _contient_terme_interdit(texte: str) -> list:
    bas = texte.lower()
    for ph in PHRASES_GRAVEES:
        bas = bas.replace(ph.lower(), " ")
    return [t for t in VOCABULAIRE_INTERDIT if t.lower() in bas]


def v_plafond(schema: dict) -> tuple:
    """`V-plafond` (remplace `V-lex`) — porte de SCHÉMA. Le rapport publie
    `36/37` et `1/6` **en fractions**, avec le plancher **stratifié par domaine**,
    et la phrase gravée du périmètre. Un plancher POOLÉ est un échec (0-67)."""
    manques = []
    if schema.get("plancher_lexical_fraction") != "36/37":
        manques.append("plancher 36/37 en fraction")
    if schema.get("hasard_tige_restreint_fraction") != "1/6":
        manques.append("hasard 1/6 en fraction")
    strat = schema.get("plancher_par_domaine")
    if not isinstance(strat, dict) or sorted(strat) != sorted(p4.DOMAINES):
        manques.append("plancher stratifié par domaine (jamais poolé)")
    if schema.get("phrase_perimetre") != PHRASE_PERIMETRE:
        manques.append("phrase gravée du périmètre")
    termes = _contient_terme_interdit(json.dumps(schema, ensure_ascii=False))
    ok = not manques and not termes
    return (PASS if ok else FAIL), {"champs_manquants": manques,
                                    "termes_interdits_detectes": termes}


def v_calib(schema: dict) -> tuple:
    """`V-calib` : la primaire 1 est classée en UNE ET UNE SEULE des 4 classes ;
    **aucun champ décisionnel** n'existe pour elle ; les deux descriptifs
    obligatoires sont présents."""
    classes = ("N-b", "N-a", "N-ind", "INVALIDE-INSTRUMENT")
    manques, fautes = [], []
    if schema.get("classe") not in classes:
        manques.append("classe du calibrateur")
    for champ in ("delta_r1_inv_vs_cle_nulle", "delta_r1_inv_vs_plancher_lexical"):
        if champ not in schema:
            manques.append(champ)
    for champ in schema:
        if champ.startswith("verdict") or champ.startswith("bande") \
                or champ == "decision":
            fautes.append(champ)
    termes = _contient_terme_interdit(json.dumps(schema, ensure_ascii=False))
    ok = not manques and not fautes and not termes
    return (PASS if ok else FAIL), {"champs_manquants": manques,
                                    "champs_decisionnels_presents": fautes,
                                    "termes_interdits_detectes": termes}


def v_perimetre(rapport: dict) -> tuple:
    """`V-perimetre` : les TROIS éléments du §4.9 sont présents — mécanisme,
    limite nommée, successeur désigné. Un seul manquant ⇒ échec du pipeline."""
    manques = [k for k in ("mecanisme", "limite_nommee", "successeur_designe")
               if not rapport.get(k)]
    return (PASS if not manques else FAIL), {"elements_manquants": manques}


def v_bindur(schema: dict) -> tuple:
    """`V-bindur` : le bin dur est marqué DESCRIPTIF et son champ décisionnel est
    **absent** du schéma de sortie."""
    ok = (schema.get("bin_dur_statut") == "DESCRIPTIF"
          and "bin_dur_verdict" not in schema)
    return (PASS if ok else FAIL), {"statut": schema.get("bin_dur_statut"),
                                    "champ_decisionnel_present":
                                        "bin_dur_verdict" in schema}


def v_t1(quantites_t1=None, instr: dict | None = None) -> tuple:
    """`V-t1` (porte, pas note — défaut 0-73) : toute quantité à `t−1` dont
    l'ensemble de comparaison contient deux unités de MÊME TIGE ⇒ échec du
    pipeline. Les états y sont bit-identiques et le cosinus dégénéré sort en
    `1.0` EXACT — cas qui **échappe à la clause NaN**."""
    if instr is not None:
        quantites_t1 = instr.get("quantites_t1", [])
        source = "registre d'exécution"
    else:
        quantites_t1 = quantites_t1 or []
        source = "argument (cas de banc synthétique)"
    fautes = [q for q in quantites_t1 if q.get("intra_tige")]
    return (PASS if not fautes else FAIL), {
        "quantites_t1": quantites_t1, "n_quantites_t1": len(quantites_t1),
        "fautes": fautes, "source": source, "cardinal_t1": p4.CARD_T1,
        "motif": "publier le cardinal 14 ne suffisait pas"}


def v_ordre_partitions() -> tuple:
    """`V-ord` : classification en UNE ET UNE SEULE des SIX classes ORD, et
    espace résolu à exactement 27 cellules — RECOMPTÉ PAR EXÉCUTION."""
    c = espace_ord_resolu()
    attendu = {"ORD-1": 1, "ORD-4": 2, "ORD-2": 6, "ORD-3": 9, "ORD-0": 9,
               "total": 27}
    ok = all(c.get(k) == v for k, v in attendu.items()) and "ORD-ind" not in c
    return (PASS if ok else FAIL), {"comptage_execute": c, "attendu": attendu,
                                    "n_classes": 6}


# =========================================================================
#  Probabilités d'atteinte sous la nulle — PUBLIÉES AVANT LE RUN
# =========================================================================

def probas_sous_nulle(b: int = 200_000, seed: int = 0, tau: float = TAU,
                      k_eff: int = K_EFF) -> dict:
    """Simulation seedée : probabilités d'atteinte des classes des TROIS
    partitions sous la nulle. Publiées **avant** le run (§4.4, §4.5, §4.6)."""
    rng = np.random.default_rng(seed)
    z = 1.959963984540054
    # calibrateur : Δ̄ ~ N(0, σ/√K), σ = √(2·(1/37)(36/37)) = 0.229
    sigma = math.sqrt(2 * (1 / 37) * (36 / 37))
    se = sigma / math.sqrt(k_eff)
    est = rng.normal(0.0, se, b)
    # σ̂ variable : loi du χ² à K−1 ddl
    s_hat = se * np.sqrt(rng.chisquare(k_eff - 1, b) / (k_eff - 1))
    hw = z * s_hat
    calib = {}
    for lo, hi in zip(est - hw, est + hw):
        c = classe_calibrateur((lo, hi), tau)
        calib[c] = calib.get(c, 0) + 1
    calib = {k: round(v / b, 5) for k, v in calib.items()}
    # bandes de la primaire : même forme, seuil ε = enveloppe nulle = z·σ_null
    eps = z * se
    bandes = {}
    for lo, hi in zip(est - hw, est + hw):
        c = bande_primaire((lo, hi), eps, SOMME_M_MIN, K_EFF)
        bandes[c] = bandes.get(c, 0) + 1
    bandes = {k: round(v / b, 5) for k, v in bandes.items()}
    # maillons : idem avec ε_M = enveloppe nulle
    maillons = {}
    for lo, hi in zip(est - hw, est + hw):
        c = etat_maillon((lo, hi), eps)
        maillons[c] = maillons.get(c, 0) + 1
    maillons = {k: round(v / b, 5) for k, v in maillons.items()}
    # classes ORD sous indépendance des trois maillons
    etats = list(maillons)
    ord_p = {}
    for m1 in etats:
        for m2 in etats:
            for m3 in etats:
                p = maillons[m1] * maillons[m2] * maillons[m3]
                c = classe_ord(m1, m2, m3)
                ord_p[c] = ord_p.get(c, 0.0) + p
    return {"B": b, "seed": seed, "tau": tau, "K_eff": k_eff,
            "sigma_par_requete": round(sigma, 5), "SE": round(se, 5),
            "calibrateur": calib, "bandes_primaire": bandes,
            "etats_de_maillon": maillons,
            "classes_ORD": {k: round(v, 5) for k, v in ord_p.items()},
            "note": "publiées AVANT le run ; aucune relecture a posteriori "
                    "n'est autorisée (§4.4, §4.5)"}


# =========================================================================
#  Schéma de sortie du rapport — construit, puis passé aux portes de schéma
# =========================================================================

def schema_de_sortie(classe_calib: str, planchers_par_domaine: dict) -> dict:
    return {
        "classe": classe_calib,
        "delta_r1_inv_vs_cle_nulle": PHRASE_XIII,
        "delta_r1_inv_vs_plancher_lexical": PHRASE_XIII,
        "plancher_lexical_fraction": "36/37",
        "hasard_tige_restreint_fraction": "1/6",
        "plancher_par_domaine": planchers_par_domaine,
        "phrase_perimetre": PHRASE_PERIMETRE,
        "bin_dur_statut": "DESCRIPTIF",
    }


def bloc_perimetre() -> dict:
    """§4.9 — les trois éléments obligatoires, écrits AVANT mesure."""
    return {
        "mecanisme": "les suffixes étant globalement uniques, aucun concurrent "
                     "ne partage le suffixe de la requête — ni la nulle 1/37 du "
                     "calibrateur ni la nulle hypergéométrique de la primaire ne "
                     "peuvent monitorer ce canal",
        "limite_nommee": "aucune formulation du rapport ne peut exclure que "
                         "l'effet mesuré transite par le canal suffixe",
        "successeur_designe": "successeur de v4-matériel, matériau PROPRE requis "
                              "(~20 tiges / 40 familles) : la cellule « suffixe "
                              "partagé × domaine différent » est structurellement "
                              "interdite sous C7, et la version intra-domaine "
                              "ferait tomber K_eff à 5",
    }


# =========================================================================
#  ==========================  MESURE (GPU)  =============================
#
#  Exécutée UNIQUEMENT après `E = 0` au banc et PASS intégral de génération
#  (§6.A, clause d'abandon). Un seul passage, trois modèles, aucun balayage.
#  `M` n'est jamais instanciée ; aucune injection ; `engram/` non modifié.
# =========================================================================

CELLULES_DE_REQUETE = {"T1": ("T1-B", "T1-A"), "T2": ("T2-B", "T2-A"),
                       "T3": ("T3-B", "T3-A")}
TYPE_DECISIONNEL = "T1"    # §8 : 60 requêtes décisionnelles ; T2 et T3 sont
                           # publiés en DESCRIPTIF (l'arithmétique du protocole —
                           # « Σ_q m_q ≥ 60 ≈ 1 gagnant par requête » — fixe le
                           # nombre de requêtes à 60, soit une par unité)


def _sequences_de_cellule(mat, cle):
    """132 séquences par cellule : 72 unités + 40 paires de cadre + 20 pseudo."""
    c = mat["cellules"][cle]
    pref = " ".join(c["mots"])
    icap = p4.indice_capture(cle, mat)
    lignes = []
    for i, u in enumerate(mat["unites"]):
        lignes.append({"role": "unite", "i": i, "capture": icap,
                       "texte": pref + " " + p4.surface_entite(u, c["capitalise"])})
    for j, (a, b) in enumerate(mat["paires_cadre"]):
        s = (a[0].upper() + a[1:].lower()) if c["capitalise"] else a.lower()
        lignes.append({"role": "cadre", "i": j, "capture": icap,
                       "texte": f"{pref} {s} {b}"})
    for k, w in enumerate(mat["pseudo_mots"]):
        lignes.append({"role": "pseudo", "i": k, "capture": None,
                       "texte": f"{pref} {w}"})
    return lignes


def _blocs(modele):
    for att in ("transformer", "model"):
        m = getattr(modele, att, None)
        if m is not None:
            for nom in ("h", "layers"):
                b = getattr(m, nom, None)
                if b is not None:
                    return b
    raise RuntimeError("blocs introuvables")


def capture_modele(nom_modele: str, mat, out: Path, verbeux: bool = True):
    """Un forward par cellule. Capture : états de TOUTES les couches en `t` et
    `t-1`, et **NLL SCALAIRE** du token de capture. Jamais le tenseur de logits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dispo = torch.cuda.is_available()
    device = torch.device("cuda" if dispo else "cpu")
    if dispo:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(nom_modele)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # D21 : dtype du forward EPINGLE bf16 ; repli fp32 = CHEMIN NOMINAL declare.
    voulu = torch.bfloat16 if dispo else torch.float32
    modele = AutoModelForCausalLM.from_pretrained(nom_modele, dtype=voulu)
    modele.to(device).eval().requires_grad_(False)
    dtype_effectif = str(next(modele.parameters()).dtype)
    L = len(_blocs(modele))
    res = {"modele": nom_modele, "L": L, "L_attendu": L_ATTENDU[nom_modele],
           "V-L": PASS if L == L_ATTENDU[nom_modele] else FAIL,
           "device": str(device), "cuda_disponible": bool(dispo),
           "dtype_demande": DTYPE_FORWARD, "dtype_effectif": dtype_effectif,
           "repli_fp32": dtype_effectif == "torch.float32",
           "duree_chargement_s": round(time.time() - t0, 2),
           "cellules": {}, "n_forwards": 0}

    etats, nll = {}, {}
    for cle in mat["cellules"]:
        t1 = time.time()
        lignes = _sequences_de_cellule(mat, cle)
        ids_l = [tok.encode(x["texte"], add_special_tokens=False) for x in lignes]
        lmax = max(len(s) for s in ids_l)
        ids = torch.full((len(ids_l), lmax), tok.pad_token_id, dtype=torch.long)
        msk = torch.zeros((len(ids_l), lmax), dtype=torch.long)
        for r, s in enumerate(ids_l):
            ids[r, :len(s)] = torch.tensor(s)
            msk[r, :len(s)] = 1
        pos = torch.tensor([x["capture"] if x["capture"] is not None
                            else len(ids_l[r]) - 1 for r, x in enumerate(lignes)])
        with torch.no_grad():
            o = modele(input_ids=ids.to(device), attention_mask=msk.to(device),
                       output_hidden_states=True)
        res["n_forwards"] += 1
        hs = o.hidden_states                      # L+1 tenseurs (B, T, d)
        ar = torch.arange(len(ids_l), device=device)
        idx = pos.to(device)
        E = np.stack([h[ar, idx].float().cpu().numpy() for h in hs], axis=0)
        Em1 = np.stack([h[ar, idx - 1].float().cpu().numpy() for h in hs], axis=0)
        # NLL SCALAIRE du token de capture (teacher-forcing, cortex gele)
        lg = o.logits[ar, idx - 1].float()
        cible = ids.to(device)[ar, idx]
        lp = torch.log_softmax(lg, dim=-1)
        nll[cle] = (-lp[ar, cible]).cpu().numpy().astype(np.float64)
        del o, hs, lg, lp
        etats[cle] = {"t": E, "t_moins_1": Em1}
        res["cellules"][cle] = {
            "n_sequences": len(ids_l), "longueur_max": int(lmax),
            "indice_capture": p4.indice_capture(cle, mat),
            "duree_s": round(time.time() - t1, 2),
            "dimension": int(E.shape[2]), "n_couches_capturees": int(E.shape[0]),
            "nan_ou_inf": bool(not np.isfinite(E).all()
                               or not np.isfinite(nll[cle]).all()),
        }
        if verbeux:
            print(f"  {nom_modele} / {cle} : {len(ids_l)} seq., "
                  f"{res['cellules'][cle]['duree_s']} s")
        if dispo:
            torch.cuda.empty_cache()

    out.mkdir(parents=True, exist_ok=True)
    slug = nom_modele.replace("/", "_")
    np.savez_compressed(out / f"etats-{slug}.npz",
                        **{f"{c}|{k}": etats[c][k] for c in etats for k in etats[c]},
                        **{f"nll|{c}": nll[c] for c in nll})
    res["vram_max_reservee_gio"] = (round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
                                    if dispo else 0.0)
    res["vram_max_allouee_gio"] = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                                   if dispo else 0.0)
    del modele
    if dispo:
        torch.cuda.empty_cache()

    # --- CONTRÔLE fp32 commandé par le protocole (§4.7 / §7, issue 1) ---------
    # Le forward décisionnel est épinglé bf16 (D21) ; `δ̂` est l'écart MESURÉ
    # entre les cosinus bf16 et les cosinus fp32 sur les états qui portent la
    # décision. Le repli fp32 est le CHEMIN NOMINAL : si le modèle fp32 ne tient
    # pas en VRAM, le contrôle passe sur CPU et le device est PUBLIÉ.
    t2 = time.time()
    if dispo:
        torch.cuda.reset_peak_memory_stats()
    cellules_ctrl = list(dict.fromkeys(CELLULES_DE_REQUETE[TYPE_DECISIONNEL]))
    etats_fp32, dev_ctrl, motif_ctrl = {}, None, ""
    for essai in (("cuda" if dispo else "cpu"), "cpu"):
        try:
            dev = torch.device(essai)
            mc = AutoModelForCausalLM.from_pretrained(nom_modele,
                                                      dtype=torch.float32)
            mc.to(dev).eval().requires_grad_(False)
            for cle in cellules_ctrl:
                lignes = _sequences_de_cellule(mat, cle)
                ids_l = [tok.encode(x["texte"], add_special_tokens=False)
                         for x in lignes]
                lmax = max(len(x) for x in ids_l)
                ids = torch.full((len(ids_l), lmax), tok.pad_token_id,
                                 dtype=torch.long)
                msk = torch.zeros((len(ids_l), lmax), dtype=torch.long)
                for r, x in enumerate(ids_l):
                    ids[r, :len(x)] = torch.tensor(x)
                    msk[r, :len(x)] = 1
                pos = torch.tensor([x["capture"] if x["capture"] is not None
                                    else len(ids_l[r]) - 1
                                    for r, x in enumerate(lignes)])
                with torch.no_grad():
                    oc = mc(input_ids=ids.to(dev), attention_mask=msk.to(dev),
                            output_hidden_states=True)
                arc = torch.arange(len(ids_l), device=dev)
                etats_fp32[cle] = np.stack(
                    [h[arc, pos.to(dev)].float().cpu().numpy()
                     for h in oc.hidden_states], axis=0)
                del oc
            dev_ctrl = essai
            del mc
            if dispo:
                torch.cuda.empty_cache()
            break
        except torch.OutOfMemoryError as e:      # repli fp32 sur CPU : NOMINAL
            motif_ctrl = f"OOM CUDA sur le contrôle fp32 ({type(e).__name__})"
            etats_fp32 = {}
            try:
                del mc
            except NameError:
                pass
            if dispo:
                torch.cuda.empty_cache()
    res["controle_fp32"] = {
        "cellules": cellules_ctrl, "device": dev_ctrl,
        "repli_cpu": dev_ctrl == "cpu" and dispo, "motif_repli": motif_ctrl,
        "duree_s": round(time.time() - t2, 2),
        "vram_max_reservee_gio": (round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
                                  if dispo and dev_ctrl == "cuda" else 0.0),
        "n_sequences": sum(len(_sequences_de_cellule(mat, c))
                           for c in cellules_ctrl),
        "regle": "repli fp32 = CHEMIN NOMINAL déclaré avant le run (D21 / 0-51)"}
    res["duree_totale_s"] = round(time.time() - t0, 2)
    return res, etats, nll, etats_fp32


# ------------------------------------------------------------------ analyse

def _cos(A, B=None):
    A = np.asarray(A, dtype=np.float32)
    B = A if B is None else np.asarray(B, dtype=np.float32)
    na = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    nb = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return (na @ nb.T).astype(np.float64)


def _r1_mid_rank(scores, cible: int) -> float:
    """R1 = indicatrice de rang 1, egalites en MID-RANK (E2) : a `g` ex aequo en
    tete, la cible compte `1/g`."""
    _noter_rang()
    s = np.asarray(scores, dtype=np.float64)
    mx = s.max()
    gagnants = int(np.sum(s == mx))
    return (1.0 / gagnants) if s[cible] == mx else 0.0


def calibrateur(mat, etats, couche: int, seed: int = 0) -> dict:
    """`DR1_inv = R1(variante correcte) - R1(cle nulle appariee)`, rang parmi 37."""
    u = mat["unites"]
    n_u = len(u)
    n_cadre = len(mat["paires_cadre"])
    # §4.4 : statistique de cluster par TIGE, `K_eff = 10`. Les clusters sont donc
    # les 10 tiges DECISIONNELLES ; les 12 unites de reserve (4 tiges simples)
    # porteraient `K_eff = 14` et resserreraient l'IC sous la derivation de `tau`.
    # `C5` reste verifiee sur 72/72 requetes (porte de generation) : c'est la
    # CONDITION qui porte sur 72, la STATISTIQUE qui porte sur 60.
    decisionnelles = [i for i, x in enumerate(u) if x["pontee"]]
    par_tige, lignes = {}, []
    for typ, (cle_q, cle_c) in CELLULES_DE_REQUETE.items():
        Q = etats[cle_q]["t"][couche]
        C = etats[cle_c]["t"][couche]
        for i in decisionnelles:
            pool = p4.pool_c5(i, u, seed)
            cand = pool + [i]
            M = _cos(Q[[i, n_u + (i % n_cadre)]], C[cand])
            r_ok = _r1_mid_rank(M[0], len(cand) - 1)
            r_nul = _r1_mid_rank(M[1], len(cand) - 1)
            d = r_ok - r_nul
            lignes.append({"unite": i, "type": typ, "R1": r_ok, "R1_nul": r_nul,
                           "delta": d, "tige": u[i]["tige"],
                           "domaine": u[i]["domaine"]})
            par_tige.setdefault(u[i]["tige"], []).append(d)
    tiges = sorted(par_tige)
    v = [float(np.mean(par_tige[t])) for t in tiges]
    boot = bootstrap_tiges(v, seed=seed)
    plancher = {d: f"{PLANCHER_LEXICAL[0]}/{PLANCHER_LEXICAL[1]}"
                for d in p4.DOMAINES}
    return {"delta_R1_inv": boot["estime"], "IC": boot["IC"], "K_eff": boot["K_eff"],
            "classe": classe_calibrateur(boot["IC"]),
            "R1_moyen": float(np.mean([x["R1"] for x in lignes])),
            "R1_nul_moyen": float(np.mean([x["R1_nul"] for x in lignes])),
            "n_requetes": len(lignes), "par_tige": dict(zip(tiges, v)),
            "plancher_par_domaine": plancher,
            "descriptif": {"vs_cle_nulle": PHRASE_XIII,
                           "vs_plancher_lexical": PHRASE_XIII}}


def primaire(mat, etats, couche: int, seed: int = 0,
             typ: str = TYPE_DECISIONNEL) -> dict:
    """Composition des intrusions - nulle hypergeometrique exacte, barriere D19."""
    u = mat["unites"]
    cle_q, cle_c = CELLULES_DE_REQUETE[typ]
    Q, C = etats[cle_q]["t"][couche], etats[cle_c]["t"][couche]
    x_l, m_l, tig_l = [], [], []
    for i, unite in enumerate(u):
        if not unite["pontee"]:
            continue
        p = p4.pool_p2(i, u, seed)
        pool = p["pool"]
        s = _cos(Q[[i]], C[pool + [i]])[0]
        _noter_rang()
        cible = s[-1]
        conc = s[:-1]
        bat = conc > cible
        m = int(bat.sum())
        partage = np.array([1 if j in p["tige_partages"] else 0 for j in pool])
        x_l.append(int(partage[bat].sum()))
        m_l.append(m)
        tig_l.append(unite["tige"])
    obs = d_observe(x_l, m_l)
    # --- barriere d'information EXECUTABLE : sceller -> publier -> desceller
    bar = Barriere()
    bar.sceller(obs["D"])
    env = eps_nulle_composition(m_l, tig_l, b=B_MC, seed=seed)
    eps = bar.publier(env["epsilon"])
    bar.desceller()
    # Support VIDE (famine globale) : `eps` n'est pas defini. On ne publie PAS un
    # NaN comme si c'etait une enveloppe (clause NaN, §6.B) — on declare que la
    # quantite est SANS OBJET, la famine ayant deja emporte la classification.
    support_vide = obs["n_eff"] == 0 or eps is None or not np.isfinite(eps)
    if support_vide:
        eps = None
    par_tige = {}
    for x, m, t in zip(x_l, m_l, tig_l):
        if m >= 1:
            par_tige.setdefault(t, []).append(x - m * P_TIGE)
    tiges = sorted(par_tige)
    v = [float(np.mean(par_tige[t])) for t in tiges]
    boot = (bootstrap_tiges(v, seed=seed) if v else
            {"estime": float("nan"), "IC": [float("nan")] * 2, "K_eff": 0})
    k_support = len(tiges)
    bande = (bande_primaire(boot["IC"], eps, obs["somme_m"], k_support)
             if (k_support and eps is not None) else "C-ind")
    if support_vide:
        v_c = "SANS OBJET"
        det_c = {"ordre_barriere": bar.ordre(),
                 "ordre_attendu": ["sceller", "publier", "desceller"],
                 "somme_m_publiee": obs["somme_m"],
                 "exclusion_m0_predeclaree": True,
                 "epsilon": None,
                 "raison": "support vide (n_eff = 0) : la nulle hypergeometrique "
                           "conditionnelle aux m_q n'a pas de domaine de "
                           "definition ; la famine globale (Sigma_q m_q < 60) a "
                           "deja emporte la classification en C-ind d'office (D19)"}
    else:
        v_c, det_c = v_compo(bar.ordre(), obs["somme_m"], True, eps)
    # Aucune quantite sans domaine de definition n'est publiee en NaN (D23).
    d_norm = None if support_vide else obs["D_normalise_par_requete"]
    ic_pub = None if support_vide else boot["IC"]
    est_pub = None if support_vide else boot["estime"]
    return {"D": obs["D"], "D_normalise": d_norm,
            "n_eff": obs["n_eff"], "somme_m": obs["somme_m"],
            "distribution_m": obs["distribution_m"],
            "K_eff_support": k_support, "epsilon": eps, "enveloppe": env,
            "IC": ic_pub, "estime_par_requete": est_pub,
            "bande": bande, "V-compo": v_c, "detail_V-compo": det_c,
            "X_par_requete": x_l, "m_par_requete": m_l,
            "courbe_X_vs_m": _courbe_x_m(x_l, m_l),
            "par_tige": dict(zip(tiges, v)), "type_de_requete": typ}


def _courbe_x_m(x_l, m_l) -> dict:
    """Descriptif obligatoire (§4.3) : l'exces doit etre NUL aux deux bords et
    maximal a `m` intermediaire - forme en cloche, signature du decrement lure."""
    out = {}
    for x, m in zip(x_l, m_l):
        out.setdefault(int(m), []).append(x - m * P_TIGE)
    return {str(k): {"n": len(v), "exces_moyen": round(float(np.mean(v)), 4)}
            for k, v in sorted(out.items())}


def _cos_moyen_stratifie(COS, paires) -> float:
    """Moyenne NON PONDEREE des moyennes intra-strate - rien n'est poole (C7/0-67)."""
    par = {}
    for (a, b, sid) in paires:
        par.setdefault(sid, []).append(COS[a, b])
    if not par:
        return float("nan")
    return float(np.mean([np.mean(v) for v in par.values()]))


def _cos_strates(mat, etats, couche: int):
    """Matrice de cosinus des 60 unités décisionnelles, moyennée sur les 6
    cellules (l'invariance aux trois transformations nommées est le sujet, §8).
    Cosinus en fp32 (§7)."""
    dec = mat["unites_decisionnelles"]
    n = len(dec)
    COS = np.zeros((n, n))
    idx = [i for i, u in enumerate(mat["unites"]) if u["pontee"]]
    for cle in mat["cellules"]:
        COS += _cos(etats[cle]["t"][couche][idx])
    return COS / len(mat["cellules"]), dec, n


def maillons(mat, etats, couche: int, b: int = B_MC, seed: int = 0,
             masque_paires=None, etiquette: str = "global") -> dict:
    """M1/M2/M3 **et `B0′`** par permutation intra-tige, stratifiée par domaine,
    combinée.

    La permutation redistribue les 6 unités d'une tige en 2 familles de 3
    (`C(6,3) = 20` par tige), ce qui **redéfinit les strates** : c'est la nulle
    substantielle « l'état ne distingue pas le contenu des sous-viviers » (§4.6).
    Les moyennes de strate sont des moyennes NON PONDÉRÉES de moyennes
    intra-strate : rien n'est poolé (`C7` / 0-67).

    `masque_paires` restreint l'ensemble de paires (bandes de NLL de
    `V-surprise`, issue 2). Le masque est une fonction des UNITÉS (leur NLL),
    donc **invariant par la permutation** : la nulle reste exacte sur le
    sous-ensemble.

    **`B0′ = cos̄(S3) − cos̄(S1)` reçoit sa PROPRE enveloppe de permutation**
    `ε_{B0′} = q₀.₉₅(|B0′_null|)` (issue 5) : l'enveloppe d'une somme n'est pas
    la moyenne des enveloppes de ses termes.
    """
    COS, dec, n = _cos_strates(mat, etats, couche)
    tige_de = np.array([u["tige"] for u in dec])
    dom0 = np.array([p4.DOMAINES.index(u["domaine"]) for u in dec])
    tiges = sorted(set(tige_de.tolist()))
    membres = {t: np.flatnonzero(tige_de == t) for t in tiges}
    doms_tige = {t: sorted(set(dom0[membres[t]].tolist())) for t in tiges}

    ia, ib = np.triu_indices(n, 1)
    if masque_paires is not None:
        sel0 = np.asarray(masque_paires, dtype=bool)
        ia, ib = ia[sel0], ib[sel0]
    cosv = COS[ia, ib]
    meme_tige = tige_de[ia] == tige_de[ib]
    nd = len(p4.DOMAINES)
    couples = {}
    for x in range(nd):
        for y in range(x + 1, nd):
            couples[(x, y)] = nd + len(couples)
    n_sid = nd + len(couples)
    cpl = np.zeros((nd, nd), dtype=np.int64)
    for x in range(nd):
        cpl[x, x] = x
        for y in range(x + 1, nd):
            cpl[x, y] = cpl[y, x] = couples[(x, y)]

    def stats(dom):
        da, db = dom[ia], dom[ib]
        meme_dom = da == db
        sid = cpl[da, db]
        cat = np.where(meme_tige & meme_dom, 0,
                       np.where(meme_tige, 1, np.where(meme_dom, 2, 3)))
        out = np.empty(4)
        cnt = np.empty(4, dtype=np.int64)
        for k in range(4):
            selk = cat == k
            cnt[k] = int(selk.sum())
            if not selk.any():
                out[k] = np.nan
                continue
            sm = np.bincount(sid[selk], weights=cosv[selk], minlength=n_sid)
            cn = np.bincount(sid[selk], minlength=n_sid)
            nz = cn > 0
            out[k] = float(np.mean(sm[nz] / cn[nz]))
        return out, cnt                              # [S3, S2, S1, S0]

    def quatre(m):
        """[M1, M2, M3, B0'] — B0' = S3 − S1, apparié en domaine."""
        return np.array([m[0] - m[1], m[1] - m[2], m[2] - m[3], m[0] - m[2]])

    m_obs, cnt = stats(dom0)
    obs = quatre(m_obs)
    moyennes = dict(zip(("S3", "S2", "S1", "S0"), map(float, m_obs)))
    effectifs = dict(zip(("S3", "S2", "S1", "S0"), map(int, cnt)))

    rng = np.random.default_rng(seed)
    nul = np.empty((b, 4))
    for r in range(b):
        dom = dom0.copy()
        for t in tiges:
            mm = membres[t].copy()
            rng.shuffle(mm)
            dA, dB = doms_tige[t]
            dom[mm[:3]] = dA
            dom[mm[3:]] = dB
        nul[r] = quatre(stats(dom)[0])

    noms = ("M1", "M2", "M3", "B0_prime")
    out = {"cos_moyen_par_strate": moyennes, "n_paires_par_strate": effectifs,
           "n_paires_total": int(len(ia)), "etiquette": etiquette,
           "C_6_3": permutations_intra_tige(), "B": b, "seed": seed,
           "couche": couche}
    etats_m = []
    for j, nom in enumerate(noms):
        col = nul[:, j]
        if not np.isfinite(obs[j]) or not np.isfinite(col).all():
            out[nom] = {"etat": "SANS OBJET",
                        "raison": "strate vide sur ce sous-ensemble de paires",
                        "observe": None, "centre": None, "epsilon_M": None,
                        "IC": None}
            if nom != "B0_prime":
                etats_m.append("ind")
            continue
        env = eps_m_permutation(col - col.mean())
        centre = float(obs[j]) - float(col.mean())
        ic = [centre - env["epsilon_M"], centre + env["epsilon_M"]]
        e = etat_maillon(ic, env["epsilon_M"])
        if nom != "B0_prime":
            etats_m.append(e)
        out[nom] = {"observe": float(obs[j]),
                    "attendu_sous_la_nulle": float(col.mean()),
                    "centre": centre, "epsilon_M": env["epsilon_M"],
                    "IC": ic, "etat": e,
                    "signe": (1 if e == "+" else -1 if e == "−" else 0)}
    out["B0_prime"]["derivation_enveloppe"] = (
        "nulle de permutation PROPRE, ε_{B0′} = q₀.₉₅(|B0′_null|), même "
        "machinerie MC seedée que ε_M — jamais la moyenne de ε_M1 et ε_M2 "
        "(l'enveloppe d'une somme n'est pas la moyenne des enveloppes)")
    out["classe_ORD"] = classe_ord(*etats_m)
    out["verdict_ORD"] = VERDICT_ORD[out["classe_ORD"]]
    return out


def fuite_couche_0(mat, etats, seed: int = 0, b: int = B_MC) -> dict:
    """`V-leak` : `B0` (détecteur FAIBLE, garanti ≤ 0 par `C7`) et **`B0′`**
    (détecteur FORT, le seul qui morde). « couche 0 » = `hidden_states[0]`.

    Chacun porte SA propre enveloppe de permutation (issue 5).
    """
    m = maillons(mat, etats, 0, b=b, seed=seed, etiquette="couche 0")
    b0, e0 = m["M2"]["centre"], m["M2"]["epsilon_M"]
    b0p, e0p = m["B0_prime"]["centre"], m["B0_prime"]["epsilon_M"]
    ic_b0 = [b0 - e0, b0 + e0]
    ic_b0p = [b0p - e0p, b0p + e0p]
    v, det = v_leak(ic_b0, ic_b0p)
    return {"B0": b0, "epsilon_B0": e0, "IC_B0": ic_b0,
            "B0_prime": b0p, "epsilon_B0_prime": e0p, "IC_B0_prime": ic_b0p,
            "B0_prime_observe": m["B0_prime"]["observe"],
            "B0_prime_attendu_sous_la_nulle": m["B0_prime"]["attendu_sous_la_nulle"],
            "derivation_enveloppe_B0_prime": m["B0_prime"]["derivation_enveloppe"],
            "V-leak": v, "detail": det, "couche": "hidden_states[0]",
            "cos_moyen_par_strate": m["cos_moyen_par_strate"]}


def bandes_nll(mat, nll) -> dict:
    """Bandes GRAVÉES de `V-surprise` : **3 bandes, tertiles calculés PAR MODÈLE
    sur la NLL MOYENNE DE PAIRE** (l'unité de bande est la PAIRE : M1 est une
    statistique de paire, stratifier par membre rendrait les paires inter-bandes
    indéfinies). Plus la superposition secondaire à 2 bandes, coupure ABSOLUE à
    `4.0 nats` = la constante `thr` du projet.
    """
    dec = [i for i, u in enumerate(mat["unites"]) if u["pontee"]]
    par_unite = np.mean([nll[c][dec] for c in mat["cellules"]], axis=0)
    u = mat["unites_decisionnelles"]
    n = len(u)
    ia, ib = np.triu_indices(n, 1)
    val = (par_unite[ia] + par_unite[ib]) / 2.0
    coupes = np.quantile(val, [1 / 3, 2 / 3])
    bande = np.digitize(val, coupes)                 # 0, 1, 2
    return {"nll_par_unite": par_unite, "nll_par_paire": val,
            "coupures_tertiles": [float(c) for c in coupes],
            "bande_par_paire": bande,
            "n_paires_par_bande": np.bincount(bande, minlength=3).tolist(),
            "superposition_absolue": {
                "seuil_nats": NLL_SEUIL_ABSOLU,
                "n_paires_au_dessus": int((val > NLL_SEUIL_ABSOLU).sum()),
                "n_paires_au_dessous": int((val <= NLL_SEUIL_ABSOLU).sum())}}


def m1_par_bande_nll(mat, etats, nll, couche: int, b: int = B_MC,
                     seed: int = 0) -> dict:
    """`V-surprise` APPLIQUÉE (issue 2) : M1 recalculé **dans chacune des trois
    bandes de NLL**, avec sa propre nulle de permutation et sa propre `ε_M`.

    Règle de décision gravée (§4.7) : *M1 n'est créditée que si son signe est
    stable sur les trois bandes. Si M1 n'est présente que dans la bande la plus
    haute, elle EST le confondant de surprise (0-55) et M1 est RETIRÉE.*
    **Aucune repondération, aucune sélection après lecture.**
    """
    bn = bandes_nll(mat, nll)
    par_bande, signes = {}, []
    for k in range(NLL_BANDES):
        masque = bn["bande_par_paire"] == k
        r = maillons(mat, etats, couche, b=b, seed=seed + k,
                     masque_paires=masque, etiquette=f"bande NLL {k}")
        m1 = r["M1"]
        signe = m1.get("signe", 0) if m1.get("etat") != "SANS OBJET" else None
        signes.append(0 if signe is None else signe)
        par_bande[f"bande_{k}"] = {
            "n_paires": r["n_paires_total"],
            "n_paires_par_strate": r["n_paires_par_strate"],
            "M1_centre": m1.get("centre"), "epsilon_M": m1.get("epsilon_M"),
            "IC": m1.get("IC"), "etat": m1.get("etat"), "signe": signe}
    v, det = v_surprise(signes)
    det["M1_par_bande"] = par_bande
    det["coupures_tertiles"] = bn["coupures_tertiles"]
    det["n_paires_par_bande"] = bn["n_paires_par_bande"]
    det["superposition_absolue"] = bn["superposition_absolue"]
    return {"V-surprise": v, "detail": det, "signes_par_bande": signes,
            "M1_retiree": bool(det.get("M1_retiree"))}


def surprise(mat, nll) -> dict:
    """`V-surprise` : NLL du token de capture (SCALAIRE), 3 bandes, tertiles."""
    dec = [i for i, u in enumerate(mat["unites"]) if u["pontee"]]
    par_unite = np.mean([nll[c][dec] for c in mat["cellules"]], axis=0)
    u = mat["unites_decisionnelles"]
    n = len(u)
    val, strat = [], []
    for a in range(n):
        for b in range(a + 1, n):
            val.append((par_unite[a] + par_unite[b]) / 2)
            strat.append(p4.strate(u[a], u[b]))
    val = np.asarray(val)
    coupes = np.quantile(val, [1 / 3, 2 / 3])
    bande = np.digitize(val, coupes)
    return {"NLL_moyenne_par_unite": float(par_unite.mean()),
            "NLL_min_max": [float(par_unite.min()), float(par_unite.max())],
            "coupures_tertiles": [float(c) for c in coupes],
            "n_paires_par_bande": np.bincount(bande, minlength=3).tolist(),
            "NLL_moyenne_par_strate": {s: float(np.mean(
                [v for v, k in zip(val, strat) if k == s]))
                for s in ("S3", "S2", "S1", "S0")},
            "superposition_absolue": {
                "seuil_nats": NLL_SEUIL_ABSOLU,
                "n_paires_au_dessus": int((val > NLL_SEUIL_ABSOLU).sum())},
            "unite_de_bande": "la PAIRE, par la moyenne des NLL de ses deux membres",
            "n_bandes": NLL_BANDES}


def precision_dtype(mat, etats, etats_fp32, nom_modele: str, couche: int,
                    seed: int = 0) -> dict:
    """`V-dtype` v2 : deux marges (tête et coupure `T`), unité = la TIGE, et
    **`δ̂ = max|Δcos|` MESURÉ** contre un contrôle fp32 (issue 1).

    `δ̂` est l'écart maximal, sur **l'ensemble de comparaison qui porte la
    décision** (`m = 60` requêtes × 37 candidats), entre le cosinus calculé sur
    les états du forward bf16 épinglé et celui calculé sur les états du forward
    fp32 de contrôle. Aucune constante n'est substituée à cette mesure.
    """
    u = mat["unites"]
    cle_q, cle_c = CELLULES_DE_REQUETE[TYPE_DECISIONNEL]
    Q, C = etats[cle_q]["t"][couche], etats[cle_c]["t"][couche]
    dispo_ctrl = bool(etats_fp32) and cle_q in etats_fp32 and cle_c in etats_fp32
    Qf = etats_fp32[cle_q][couche] if dispo_ctrl else None
    Cf = etats_fp32[cle_c][couche] if dispo_ctrl else None

    tete, coupure, ecarts, n_cos = {}, {}, [], 0
    for i, unite in enumerate(u):
        if not unite["pontee"]:
            continue
        p = p4.pool_p2(i, u, seed)
        cand = p["pool"] + [i]
        s = _cos(Q[[i]], C[cand])[0]
        if dispo_ctrl:
            sf = _cos(Qf[[i]], Cf[cand])[0]
            ecarts.append(float(np.max(np.abs(s - sf))))
            n_cos += len(cand)
        tri = np.sort(s)[::-1]
        marge_tete = float(tri[0] - tri[1])
        T = s[-1]
        marge_coupe = float(np.min(np.abs(s[:-1] - T)))
        t = unite["tige"]
        tete[t] = min(tete.get(t, 1e9), marge_tete)
        coupure[t] = min(coupure.get(t, 1e9), marge_coupe)

    if not dispo_ctrl:
        return {"V-dtype": FAIL, "modele": nom_modele,
                "detail": {"raison": "contrôle fp32 absent : δ̂ non mesurable",
                           "porte": "§6.I, bloquante"}}
    delta = float(max(ecarts))
    det_mesure = {"delta_chapeau_mesure": delta,
                  "m_requetes": len(ecarts), "n_cosinus_compares": n_cos,
                  "ecart_median_par_requete": float(np.median(ecarts)),
                  "cellules": [cle_q, cle_c], "couche": couche,
                  "constante_ULP_bf16_pour_reference_seulement": 2 ** -8,
                  "constante_conservatrice": bool(2 ** -8 >= delta)}
    v, det = v_dtype(tete, coupure, delta, delta_mesure=True,
                     detail_mesure=det_mesure)
    det["marges_de_tete_par_tige"] = {k: round(x, 6) for k, x in tete.items()}
    det["marges_a_la_coupure_par_tige"] = {k: round(x, 6) for k, x in coupure.items()}
    det["marge_de_tete_min"] = round(min(tete.values()), 6)
    det["marge_a_la_coupure_min"] = round(min(coupure.values()), 6)
    return {"V-dtype": v, "detail": det, "modele": nom_modele}


def descriptif_A3(mat, etats, couche: int, cfg) -> dict:
    """**A3** (§4.8) : `cos(topk(G·h))` relu sur les strates.

    `G` est GELÉE et seedée par `cfg.seed` (D9 : aucune `G` apprise) ; `dg_dim`
    et `dg_topk` viennent d'`EngramConfig`, aucune constante en dur. **Aucun
    backprop, `M` n'est pas instanciée** : on ne lit ni n'écrit de mémoire, on
    applique la seule projection `φ = topk(G·h)` aux états déjà capturés.

    Prédiction pré-déclarée : compression CROISSANTE avec le cosinus brut.
    Antipode : compression UNIFORME ⇒ `topk(G·h)` est un rééchelonnement et non
    un séparateur. **Descriptif : aucune décision n'en dépend.**
    """
    dec = mat["unites_decisionnelles"]
    n = len(dec)
    idx = [i for i, u in enumerate(mat["unites"]) if u["pontee"]]
    d = etats[next(iter(mat["cellules"]))]["t"][couche].shape[1]
    rng = np.random.default_rng(cfg.seed)
    G = rng.standard_normal((cfg.dg_dim, d)).astype(np.float32) / np.sqrt(d)
    COS_dg = np.zeros((n, n))
    for cle in mat["cellules"]:
        H = etats[cle]["t"][couche][idx].astype(np.float32)
        Z = H @ G.T                                   # (n, dg_dim)
        seuil = np.partition(np.abs(Z), -cfg.dg_topk, axis=1)[:, -cfg.dg_topk]
        PHI = np.where(np.abs(Z) >= seuil[:, None], Z, 0.0)
        COS_dg += _cos(PHI)
    COS_dg /= len(mat["cellules"])
    COS_brut, _, _ = _cos_strates(mat, etats, couche)
    par_strate = {}
    for a in range(n):
        for b in range(a + 1, n):
            par_strate.setdefault(p4.strate(dec[a], dec[b]), []).append(
                (COS_brut[a, b], COS_dg[a, b]))
    tab = {}
    for k in ("S3", "S2", "S1", "S0"):
        br = np.array([x[0] for x in par_strate[k]])
        dg = np.array([x[1] for x in par_strate[k]])
        tab[k] = {"cos_brut": float(br.mean()), "cos_dg": float(dg.mean()),
                  "compression": float(br.mean() - dg.mean()), "n": len(br)}
    comps = [tab[k]["compression"] for k in ("S3", "S2", "S1", "S0")]
    return {"table_par_strate": tab,
            "amplitude_de_compression": float(max(comps) - min(comps)),
            "G": {"gelee": True, "seed": cfg.seed, "dg_dim": cfg.dg_dim,
                  "dg_topk": cfg.dg_topk, "apprise": False,
                  "source_des_hyperparametres": "EngramConfig"},
            "M_instanciee": False, "backprop": False,
            "prediction": "compression croissante avec le cosinus brut",
            "antipode": "compression uniforme ⇒ rééchelonnement, pas séparateur",
            "statut": "DESCRIPTIF — aucune décision n'en dépend (§4.8)"}


def descriptif_A4_t1(mat, etats, couche: int) -> dict:
    """**A4** (§4.8) : profil à `t−1`, **INTER-TIGE SEULEMENT**.

    `V-t1` interdit toute statistique à `t−1` dont l'ensemble de comparaison
    contient deux unités de MÊME TIGE : à `t−1` le suffixe n'est pas dans le
    préfixe causal, les six tronquées d'une tige sont **bit-identiques** et le
    cosinus dégénère en `1.0` EXACT — cas qui **échappe à la clause NaN**
    (défauts 0-64, 0-73). Le cardinal D24-b à `t−1` vaut **14**, pas 72.

    La quantité est ENREGISTRÉE au registre `t−1` avec son ensemble de
    comparaison réel ; `V-t1` lit ce registre (issue 3).
    """
    dec = mat["unites_decisionnelles"]
    n = len(dec)
    idx = [i for i, u in enumerate(mat["unites"]) if u["pontee"]]
    tige_par_indice = {a: dec[a]["tige"] for a in range(n)}
    paires = [(a, b) for a in range(n) for b in range(a + 1, n)
              if dec[a]["tige"] != dec[b]["tige"]]        # INTER-TIGE seulement
    noter_quantite_t1("profil A4 à t−1 (inter-tige)", paires, tige_par_indice)
    out = {}
    for pos, cle_pos in (("t_moins_1", "t_moins_1"), ("t", "t")):
        C = np.zeros((n, n))
        for cle in mat["cellules"]:
            C += _cos(etats[cle][cle_pos][couche][idx])
        C /= len(mat["cellules"])
        s1 = [C[a, b] for a, b in paires
              if dec[a]["domaine"] == dec[b]["domaine"]]
        s0 = [C[a, b] for a, b in paires
              if dec[a]["domaine"] != dec[b]["domaine"]]
        out[pos] = {"cos_S1": float(np.mean(s1)), "cos_S0": float(np.mean(s0)),
                    "M3_brut": float(np.mean(s1) - np.mean(s0)),
                    "n_S1": len(s1), "n_S0": len(s0)}
    out["cardinal_D24b_a_t_moins_1"] = p4.CARD_T1
    out["paires_intra_tige_exclues"] = n * (n - 1) // 2 - len(paires)
    out["discriminant"] = ("confondant de COPIE DE TOKEN : la copie doit être "
                           "bien plus forte à la capture qu'à t−1 ; profil "
                           "identique aux deux positions ⇒ pas de signature de "
                           "copie")
    out["statut"] = "DESCRIPTIF — aucune décision n'en dépend (§4.8)"
    return out


def descriptif_P1_moins_P2(mat, etats, couche: int, seed: int = 0,
                           typ: str = TYPE_DECISIONNEL) -> dict:
    """**`P1 − P2`** (§4.8) : contraste inter-pools, **DESCRIPTIF par 0-49**.

    `P1` = `R1` dans le pool `C5` (36 concurrents à recouvrement de token NUL) ;
    `P2` = `R1` dans le pool surface-apparié (5 tige-partagés + 12 même-domaine
    + 19 autre-domaine). Lecture **ASYMÉTRIQUEMENT INFORMATIVE** gravée : sous
    monotonie stochastique du score en σ, `P1 ≥ P2` partout avec égalité aux
    deux extrémités ⇒ **une valeur positive ou nulle est NON INFORMATIVE**
    (compatible avec l'effet maximal comme avec l'absence d'effet) ; **seule une
    valeur strictement négative est informative**.
    """
    u = mat["unites"]
    cle_q, cle_c = CELLULES_DE_REQUETE[typ]
    Q, C = etats[cle_q]["t"][couche], etats[cle_c]["t"][couche]
    p1, p2 = [], []
    for i, unite in enumerate(u):
        if not unite["pontee"]:
            continue
        c5 = p4.pool_c5(i, u, seed) + [i]
        pp = p4.pool_p2(i, u, seed)["pool"] + [i]
        p1.append(_r1_mid_rank(_cos(Q[[i]], C[c5])[0], len(c5) - 1))
        p2.append(_r1_mid_rank(_cos(Q[[i]], C[pp])[0], len(pp) - 1))
    d = float(np.mean(p1) - np.mean(p2))
    return {"P1": float(np.mean(p1)), "P2": float(np.mean(p2)),
            "P1_moins_P2": d, "n_requetes": len(p1), "type_de_requete": typ,
            "lecture_gravee": ("valeur positive ou nulle : NON INFORMATIVE ; "
                               "seule une valeur strictement négative est "
                               "informative (violerait la monotonie "
                               "stochastique en σ)"),
            "informatif": bool(d < 0),
            "statut": "DESCRIPTIF par 0-49 — aucune décision n'en dépend"}


def run_mesure(out_dir=None, modeles=MODELES, seed: int = 0,
               b_perm: int = B_MC) -> dict:
    """Mesure complete. **Ne s'execute que si le banc rend `E = 0`.**"""
    t0 = time.time()
    out_dir = out_dir or (ROOT / "experiments" / "results" / "v4-materiel")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = EngramConfig(dataset="pool_v4")
    cfg.seed = seed
    encs = p4.encodeurs()
    mat = p4.construire(cfg, encs)
    gen = p4.garanties(mat, encs)
    echecs = [g["porte"] for g in gen if g["verdict"] != PASS]
    if echecs:
        return {"ARRET": "portes de generation en echec", "portes": echecs}

    hash_avant = mat["sha256"]
    reinitialiser_instrumentation()          # compteurs PAR RUN (issue 3)
    par_modele, series_pour_joint = {}, {}
    for nom in modeles:
        cap, etats, nll, etats_fp32 = capture_modele(nom, mat, out_dir / "raw")
        couche = COUCHE_REF[nom]
        cal = calibrateur(mat, etats, couche, seed)
        pri = primaire(mat, etats, couche, seed)
        mai = maillons(mat, etats, couche, b=b_perm, seed=seed)
        leak = fuite_couche_0(mat, etats, seed=seed, b=b_perm)
        sur = surprise(mat, nll)
        # `V-surprise` APPLIQUÉE aux données du run (issue 2)
        vsu = m1_par_bande_nll(mat, etats, nll, couche, b=b_perm, seed=seed)
        prec = precision_dtype(mat, etats, etats_fp32, nom, couche, seed)
        # descriptifs pré-déclarés du §4.8 (issue 6) — depuis les états déjà
        # capturés, AUCUN forward supplémentaire
        a3 = descriptif_A3(mat, etats, couche, cfg)
        a4 = descriptif_A4_t1(mat, etats, couche)
        p1p2 = descriptif_P1_moins_P2(mat, etats, couche, seed)
        par_modele[nom] = {
            "capture": cap, "couche_de_reference": couche,
            "calibrateur": cal, "primaire": pri, "maillons": mai,
            "fuite_couche_0": leak, "surprise": sur, "precision": prec,
            "V-surprise": vsu["V-surprise"], "detail_V-surprise": vsu["detail"],
            "M1_retiree_par_V_surprise": vsu["M1_retiree"],
            "descriptifs_4_8": {"A3": a3, "A4_t_moins_1": a4,
                                "P1_moins_P2": p1p2},
            "instrumentation_apres_ce_modele": instrumentation(),
            "cause_C_ind": (cause_c_ind(cal["classe"], pri["somme_m"],
                                        pri["K_eff_support"])
                            if pri["bande"] == "C-ind" else None),
            "phrase_G_bis_licenciee": phrase_g_bis_licenciee(cal["classe"],
                                                             pri["bande"]),
        }
        series_pour_joint[nom] = [pri["par_tige"][t] for t in sorted(pri["par_tige"])]
    tailles = {len(v) for v in series_pour_joint.values()}
    if len(tailles) != 1:
        vj, det_j = FAIL, {"raison": "series de tailles differentes",
                           "tailles": sorted(tailles)}
    elif tailles == {0}:
        vj, det_j = "SANS OBJET", {"raison": "aucune tige contributive sur aucun "
                                             "modele (famine globale)"}
    else:
        vj, det_j = v_joint(series_pour_joint, b=B_JOINT, seed=seed)

    schema = schema_de_sortie(
        par_modele[modeles[0]]["calibrateur"]["classe"],
        par_modele[modeles[0]]["calibrateur"]["plancher_par_domaine"])
    v_p, det_p = v_plafond(schema)
    v_cl, det_cl = v_calib(schema)
    v_pe, det_pe = v_perimetre(bloc_perimetre())
    v_bd, det_bd = v_bindur(schema)
    instr = instrumentation()
    v_po, det_po = v_pool(hash_avant, p4.sha256_materiau(mat), instr=instr)
    v_t, det_t = v_t1(instr=instr)
    v_su, det_su = v_subst([], K_EFF)
    v_or, det_or = v_ordre_partitions()
    v_sur_run = (PASS if all(r["V-surprise"] == PASS
                             for r in par_modele.values()) else FAIL)
    v_dt_run = (PASS if all(r["precision"]["V-dtype"] == PASS
                            for r in par_modele.values())
                else "INCONCLUSIF-précision")

    return {
        "protocole": "experiments/EXP-2026-08-23-v4-materiel.md",
        "config": cfg.summary(), "seed": seed,
        "sha256_materiau": hash_avant,
        "dtype_du_forward": DTYPE_FORWARD,
        "M_instanciee": False, "injection": False, "backprop": False,
        "E3": "sans objet dans ce run (aucune lecture, aucune injection)",
        "par_modele": par_modele,
        "V-joint": vj, "detail_V-joint": det_j,
        "portes_de_schema": {"V-plafond": v_p, "V-calib": v_cl,
                             "V-perimetre": v_pe, "V-bindur": v_bd,
                             "V-pool": v_po, "V-t1": v_t, "V-subst": v_su,
                             "V-ord": v_or, "V-surprise": v_sur_run,
                             "V-dtype": v_dt_run},
        "detail_portes": {"V-plafond": det_p, "V-calib": det_cl,
                          "V-perimetre": det_pe, "V-bindur": det_bd,
                          "V-pool": det_po, "V-t1": det_t, "V-subst": det_su,
                          "V-ord": det_or},
        "instrumentation_du_run": instr,
        "schema_de_sortie": schema, "perimetre": bloc_perimetre(),
        "probabilites_sous_la_nulle": probas_sous_nulle(200_000, seed),
        "operationnalisations_declarees": OPERATIONNALISATIONS,
        "clause_NaN_D23": {
            "etats_captures": {n: not any(c["nan_ou_inf"] for c in
                                          r["capture"]["cellules"].values())
                               for n, r in par_modele.items()},
            "quantites_publiees_non_finies": sorted({
                f"{n}.{k}" for n, r in par_modele.items()
                for k in ("calibrateur", "primaire", "maillons")
                for v in _valeurs_numeriques(r[k])
                if not np.isfinite(v)}),
            "regle": "reductions NaN-STRICTES ; une quantite sans domaine de "
                     "definition est declaree SANS OBJET, jamais publiee en NaN",
        },
        "duree_totale_s": round(time.time() - t0, 2),
    }


def _valeurs_numeriques(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _valeurs_numeriques(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _valeurs_numeriques(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="v4-materiel - mesure (§10)")
    ap.add_argument("--seed", type=int, default=0)
    # §4.6 et §7 gravent `B = 10⁴` : le DÉFAUT doit être la valeur gravée,
    # sans quoi la CLI invite un run non conforme (issue 4).
    ap.add_argument("--b-perm", type=int, default=B_MC)
    ap.add_argument("--modeles", default=",".join(MODELES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else (ROOT / "experiments" / "results"
                                           / "v4-materiel")
    print("=" * 78)
    print("v4-MATERIEL - MESURE (le banc D14-S doit avoir rendu E = 0)")
    print("M jamais instanciee - aucune injection - aucun backprop - E3 sans objet")
    print("=" * 78)
    rep = run_mesure(out, tuple(args.modeles.split(",")), args.seed, args.b_perm)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mesure.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    if "ARRET" in rep:
        print(f"ARRET : {rep['ARRET']} - {rep['portes']}")
        return 1
    for nom, r in rep["par_modele"].items():
        c, pr, m = r["calibrateur"], r["primaire"], r["maillons"]
        cap = r["capture"]
        print(f"\n--- {nom} (couche {r['couche_de_reference']}) ---")
        print(f"  device={cap['device']} dtype={cap['dtype_effectif']} "
              f"VRAM_reservee={cap.get('vram_max_reservee_gio')} Gio "
              f"duree={cap['duree_totale_s']} s forwards={cap['n_forwards']}")
        print(f"  calibrateur : DR1_inv = {c['delta_R1_inv']:.4f} "
              f"IC {[round(x, 4) for x in c['IC']]} -> {c['classe']} "
              f"(R1 {c['R1_moyen']:.4f} / R1_nul {c['R1_nul_moyen']:.4f})")
        eps_txt = ("SANS OBJET" if pr["epsilon"] is None
                   else f"{pr['epsilon']:.4f}")
        ic_txt = ("SANS OBJET" if pr["K_eff_support"] == 0
                  else str([round(x, 4) for x in pr["IC"]]))
        print(f"  primaire    : D = {pr['D']:.4f} ; somme_m = {pr['somme_m']} ; "
              f"n_eff = {pr['n_eff']} ; K_support = {pr['K_eff_support']} ; "
              f"eps = {eps_txt} ; IC {ic_txt} -> {pr['bande']}")
        if pr["bande"] == "C-ind":
            print(f"                cause (§6.G) : {r['cause_C_ind']}")
        for j in ("M1", "M2", "M3"):
            print(f"  {j} : centre {m[j]['centre']:+.5f} eps_M "
                  f"{m[j]['epsilon_M']:.5f} -> {m[j]['etat']}")
        lk = r["fuite_couche_0"]
        print(f"  ORD : {m['classe_ORD']} | V-leak : {lk['V-leak']} "
              f"(B0 {lk['B0']:+.5f} +/- {lk['epsilon_B0']:.5f} ; "
              f"B0' {lk['B0_prime']:+.5f} +/- {lk['epsilon_B0_prime']:.5f} "
              f"[enveloppe PROPRE])")
        pd_ = r["precision"]["detail"]
        dm = pd_.get("mesure_de_delta", {})
        print(f"  V-dtype : {r['precision']['V-dtype']} | delta_chapeau MESURE = "
              f"{dm.get('delta_chapeau_mesure', float('nan')):.6f} "
              f"(2delta = {pd_.get('seuil_2delta', float('nan')):.6f} ; "
              f"marge de tete min {pd_.get('marge_de_tete_min')} ; "
              f"marge a la coupure min {pd_.get('marge_a_la_coupure_min')} ; "
              f"tiges touchees {pd_.get('n_tiges_touchees')})")
        print(f"            controle fp32 : device={cap['controle_fp32']['device']}"
              f" repli_cpu={cap['controle_fp32']['repli_cpu']}"
              f" VRAM_reservee={cap['controle_fp32']['vram_max_reservee_gio']} Gio"
              f" duree={cap['controle_fp32']['duree_s']} s"
              f" | constante ULP conservatrice = "
              f"{dm.get('constante_conservatrice')}")
        print(f"  NLL capture : moyenne {r['surprise']['NLL_moyenne_par_unite']:.4f} "
              f"nats ; tertiles "
              f"{[round(x, 3) for x in r['surprise']['coupures_tertiles']]}")
        dv = r["detail_V-surprise"]
        print(f"  V-surprise : {r['V-surprise']} | signes de M1 par bande de NLL "
              f"{dv['signes_par_bande']} | M1 RETIREE = "
              f"{r['M1_retiree_par_V_surprise']}")
        for k in sorted(dv["M1_par_bande"]):
            bb = dv["M1_par_bande"][k]
            ctr = bb["M1_centre"]
            epsb = bb["epsilon_M"]
            print(f"             {k} : n={bb['n_paires']:5d} strates="
                  f"{bb['n_paires_par_strate']} centre="
                  f"{'None' if ctr is None else format(ctr, '+.5f')} eps_M="
                  f"{'None' if epsb is None else format(epsb, '.5f')} -> "
                  f"{bb['etat']}")
        d48 = r["descriptifs_4_8"]
        print(f"  A3 (descriptif) : compression par strate "
              + ", ".join(f"{k}={d48['A3']['table_par_strate'][k]['compression']:+.5f}"
                          for k in ('S3', 'S2', 'S1', 'S0'))
              + f" ; amplitude {d48['A3']['amplitude_de_compression']:.5f}")
        a4 = d48["A4_t_moins_1"]
        print(f"  A4 (descriptif, INTER-TIGE seulement) : M3_brut a t = "
              f"{a4['t']['M3_brut']:+.5f} ; a t-1 = "
              f"{a4['t_moins_1']['M3_brut']:+.5f} ; paires intra-tige exclues = "
              f"{a4['paires_intra_tige_exclues']} ; cardinal t-1 = "
              f"{a4['cardinal_D24b_a_t_moins_1']}")
        pp = d48["P1_moins_P2"]
        print(f"  P1-P2 (descriptif) : P1={pp['P1']:.4f} P2={pp['P2']:.4f} "
              f"difference={pp['P1_moins_P2']:+.4f} ; informatif="
              f"{pp['informatif']}")
    print(f"\nV-joint : {rep['V-joint']}")
    print(f"portes de schema : {rep['portes_de_schema']}")
    ins = rep["instrumentation_du_run"]
    print(f"instrumentation  : rangs calcules = {ins['rangs_calcules']} ; "
          f"dont dans un bootstrap = "
          f"{ins['rangs_calcules_dans_un_bootstrap']} ; quantites a t-1 = "
          f"{len(ins['quantites_t1'])} (intra-tige : "
          f"{sum(1 for q in ins['quantites_t1'] if q['intra_tige'])})")
    print(f"duree totale : {rep['duree_totale_s']} s")
    print(f"rapport : {out / 'mesure.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
