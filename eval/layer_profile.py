# SPDX-License-Identifier: AGPL-3.0-or-later
"""I2 — instrument de profil par couche (`layer_profile`).

Protocole : `experiments/EXP-2026-08-22-layer-profile.md` (statut **PROPOSE**,
version CONSOLIDÉE : la gate de pré-enregistrement n'est pas franchie ; ce
fichier est écrit, pas exécuté sur GPU).

INSTRUMENT, PAS MÉCANISME : aucune injection, aucune écriture, `M` jamais
instanciée, `engram/` non modifié (`engram.cortex._find_blocks` est lu, jamais
touché), aucun gradient (D8), **aucun nouveau champ `EngramConfig`**.

Ce que fait le module :

- **capture** — hooks sur TOUTES les couches simultanément, posés et retirés dans
  un `try/finally` (`V-hooks`) ; **un seul forward par (modèle, variante)**
  (`V-1pass`).
- **indexation** — `ℓ = 0` = sortie des embeddings ; `ℓ ∈ [1, L]` = sortie du
  bloc `ℓ`. **L'argmax décisionnel ne se cherche que sur `[1, L]`** (§4.1).
- **corpus** — (a) `pool.fact_pairs(80)` re-paraphrasé par les constantes gelées
  (§4.2) ; **`B-v3`**, bras DESCRIPTIF, jamais fusionné, jamais décisionnel.
- **partition par slot d'IDENTITÉ** — `P-0` / `P-own` / `P-ent` / `P-both`
  (§4.2). **Le verbe n'est pas un slot** : il est une covariable de ventilation.
- **quantités** — `AUC` (existence), `R1_36` et `ΔR1` (couloir, strate `P-own`),
  entropie Giraldo, λ₁/Σλ, ventilations.
- **cinq nulles** (§5) — `AUC_lex` / `R1_lex` (0 forward), **nulle de cadre**
  (verbe conservé, slots de contenu remplacés), `ℓ = 0`, corpus mélangé, nulle
  statistique (0.5 / 1/37 + bootstrap par **composante de slot** + permutation
  des étiquettes d'unité à couche fixée **avec recalcul de `max_ℓ`**).

Conventions numériques fixées (§7) : cosinus et Gram **fp32**, valeurs propres
**fp64**, bootstrap par **composante de slot** B = 10 000 avec **argmax
re-sélectionné**, **BCa** dès qu'une borne dépasse 0.95, égalités **½ crédit**.

Usage (APRÈS la gate de pré-enregistrement seulement) :
  .venv\\Scripts\\python eval\\layer_profile.py --model gpt2 --go
"""

from __future__ import annotations

import argparse
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

from pool import (  # noqa: E402  — tables et règles de paraphrase GELÉES
    ENTITIES, OWNER_OBJ, OWNERS, PARA1_VERB, PARA2_PREFIX, PARA3_HEAD,
    PARA3_MID, PARA3_TAIL, POOL_PARAPHRASE_TYPES, VERBS, _lower_first,
)

# =========================================================================
#  Constantes du protocole — §3, §4, §7. Aucune n'est ajustable après mesure.
# =========================================================================

N_UNITES = 80                 # §4.2 : N = 80 (M-10 / N-15)
N_UNITES_PRECEDENT = 30       # consigné : la valeur ÉCARTÉE (§E)
N_PARA = 3                    # trois types de paraphrase
N_A = 90                      # §3 : n_a = 90, sous-échantillonné parmi 240
N_INTRA_ATTENDU = 240         # §4.7 V-paires : 80 × C(3,2)
N_INTER_ATTENDU = 28_440      # §4.7 V-paires : C(80,2) × 9
B_BOOT = 10_000               # §4.3 : bootstrap par composante de slot
B_SOUS_ECH = 200              # §3 : sous-échantillonnage à n_a, B = 200
ALPHA_IC = 0.05               # IC 95 %
SEUIL_BCA = 0.95              # §4.3 : BCa dès qu'une borne dépasse 0.95
SEED = 0                      # §7

# ------------------------------------------------------------- couloir v4
S_LEURRES = 36                # §4.5 : 36 concurrents RÉELS
TAILLE_JEU_R1 = S_LEURRES + 1  # 1 cible + 36 concurrents = 37
HASARD_R1 = 1.0 / TAILLE_JEU_R1        # 1/37 = 0.02703
T_COULOIR = 0.25              # §4.5, décision PI : T = 0.25
T_COULOIR_PLUS = 0.50         # palier descriptif V⁺
# Repères de PUBLICATION en AUC (§4.5, « note historique ») — JAMAIS un critère.
AUC_REPERE_025 = 0.25 ** (1.0 / S_LEURRES)
AUC_REPERE_050 = 0.50 ** (1.0 / S_LEURRES)

# --------------------------------------------------------------- puissance
SIGMA0 = 0.5                  # §3 : borne de Bernoulli, conservatrice
MARGE_R1 = 0.25               # §3 : |θ − T| sous `R1_36`
Z_975 = 1.96                  # §3 : quantile normal, recopié tel quel
# §3/§4.7 : « sous décision AUC à T = 0.9622 : K ≥ 429 ». Le protocole PORTE le
# résultat mais PAS la valeur de |θ−T| qui le produit ; le banc recopie donc le
# K requis déclaré et publie EN PLUS sa propre dérivation candidate (marge =
# largeur du couloir en AUC), en signalant l'écart. Les deux rendent FAIL à tout
# N ≤ 80 : la conclusion de la porte ne dépend pas de la levée de cette
# sous-spécification.
K_REQUIS_AUC_PROTOCOLE = 429
MARGE_AUC_BANC = AUC_REPERE_050 - AUC_REPERE_025

# §7 — profondeurs attendues, re-lues du config par `V-L`.
L_ATTENDU = {"gpt2": 12, "HuggingFaceTB/SmolLM2-360M": 32, "Qwen/Qwen2.5-1.5B": 28}
MODELES = tuple(L_ATTENDU)
VARIANTES = ("a", "b_v3", "nulle_cadre", "nulle_melangee")

# Remplissage neutre GELÉ de la nulle de cadre (§5, maillon 2). Un seul mot,
# choisi hors des cinq tables indexées par l'unité : il ne porte aucun matériel.
REMPLISSAGE_NEUTRE = " thing"

# ---------------------------------------------------- partition d'identité
P_0, P_OWN, P_ENT, P_BOTH = "P-0", "P-own", "P-ent", "P-both"
PARTITIONS = (P_0, P_OWN, P_ENT, P_BOTH)
STRATE_DECISIONNELLE = P_OWN          # §D.2, nommée AVANT mesure
N_OWNERS, N_ENTITES, N_VERBES = len(OWNERS), len(ENTITIES), len(VERBS)
ESPACE_IDENTITE = N_OWNERS * N_ENTITES        # 320 — le verbe n'est pas un slot

# `V-source` (§4.7, défaut 0-10) : registre des équations citées, avec le SUPPORT
# de lecture. Lire du HTML pour citer une équation est un motif d'arrêt.
CITATIONS = (
    {"equation": "A_ij = K_ij / (n·√(K_ii·K_jj)), tr(A) = 1, H = −Σ λ log λ",
     "source_primaire": "Giraldo, Rao & Principe 2014 — Measures of entropy from "
                        "data using infinitely divisible kernels",
     "support": "PDF",
     "attribution_erronee_ecartee": "Skean et al. arXiv:2412.09563 Eq. 1 — telle "
                                    "qu'imprimée, elle NE PORTE PAS la "
                                    "normalisation des lignes"},
    {"equation": "A = P(cos_intra > cos_inter) + ½·P(=) (Mann-Whitney)",
     "source_primaire": "Mann & Whitney 1947 — On a test of whether one of two "
                        "random variables is stochastically larger than the other",
     "support": "PDF", "attribution_erronee_ecartee": None},
    {"equation": "IC BCa : α₁ = Φ(ẑ₀ + (ẑ₀+z_α)/(1−â(ẑ₀+z_α)))",
     "source_primaire": "Efron 1987 — Better bootstrap confidence intervals",
     "support": "PDF", "attribution_erronee_ecartee": None},
)


# =========================================================================
#  Corpus (§4.2)
# =========================================================================

def triplets_fact_pairs(n: int = N_UNITES) -> list[tuple[int, int, int]]:
    """Les triplets (owner, entity, verb) de `pool.fact_pairs`, en INDICES.

    Recopié de la règle de `fact_pairs` (`i mod len(table)`), pas de la fonction :
    `fact_pairs` rend des chaînes, I2 a besoin des slots. 16 owners, 20 entités,
    5 verbes ; `lcm(16, 20, 5) = 80` ⇒ 80 triplets distincts.
    """
    return [(i % N_OWNERS, i % N_ENTITES, i % N_VERBES) for i in range(n)]


def paraphrases_de_triplets(triples) -> list[tuple[str, str, str]]:
    """Les trois indices paraphrasés, par les règles GELÉES (`PARA1_VERB`,
    `PARA2_PREFIX`, `PARA3_*`, `OWNER_OBJ`). **Zéro contenu nouveau.**"""
    out = []
    for o, e, v in triples:
        owner, entity, verb = OWNERS[o], ENTITIES[e], VERBS[v]
        para1 = f"{owner} {entity} {PARA1_VERB}"
        para2 = f"{PARA2_PREFIX}{_lower_first(owner)} {entity} {verb}"
        para3 = f"{PARA3_HEAD}{entity}{PARA3_MID}{OWNER_OBJ[owner]}{PARA3_TAIL}"
        out.append((para1, para2, para3))
    return out


def corpus_a(n: int = N_UNITES) -> dict:
    """Corpus (a) PRIMAIRE : `pool.fact_pairs(80)` re-paraphrasé (§4.2)."""
    triples = triplets_fact_pairs(n)
    return {"nom": "a", "triplets": triples,
            "slots": [(OWNERS[o], ENTITIES[e], VERBS[v]) for o, e, v in triples],
            "paraphrases": paraphrases_de_triplets(triples)}


def corpus_b_v3(tokenize=None, n: int = 30) -> dict:
    """**`B-v3`** — bras DESCRIPTIF (§4.2 b) : le jeu d'unités de v3, tel quel.

    **Jamais fusionné, jamais décisionnel, jamais appelé « strate »** (M-12,
    défaut 0-11). Import paresseux : la table v3 est définie par le BPE de GPT-2.
    """
    from pool import v3_unit_triples
    triples = v3_unit_triples(n, tokenize)
    return {"nom": "B-v3", "triplets": triples,
            "slots": [(OWNERS[o], ENTITIES[e], VERBS[v]) for o, e, v in triples],
            "paraphrases": paraphrases_de_triplets(triples)}


def sha256_corpus(corpus: dict) -> str:
    """`V-hash` : SHA-256 du corpus (gelé), avant/après mesure."""
    blob = json.dumps({"nom": corpus["nom"],
                       "triplets": [list(t) for t in corpus["triplets"]],
                       "paraphrases": [list(p) for p in corpus["paraphrases"]]},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# =========================================================================
#  Partition par slot d'IDENTITÉ (§4.2) — le VERBE n'est pas un slot (N-9)
# =========================================================================

def classe_identite(slot_i, slot_j) -> str:
    """Classe d'une paire d'unités par les seuls slots d'IDENTITÉ (owner,
    entité). **Le verbe est ignoré** : cinq quasi-synonymes ne font pas un slot
    d'identité (N-9, défaut 0-12)."""
    meme_owner = slot_i[0] == slot_j[0]
    meme_entite = slot_i[1] == slot_j[1]
    if meme_owner and meme_entite:
        return P_BOTH
    if meme_owner:
        return P_OWN
    if meme_entite:
        return P_ENT
    return P_0


def partition_identite(slots) -> dict:
    """Recensement `P-0 / P-own / P-ent / P-both` + carte des paires + covariable
    de ventilation par verbe (§4.2). Fonction PURE, aucun GPU."""
    n = len(slots)
    paires, recens = {}, {c: 0 for c in PARTITIONS}
    verbe = {}
    for i in range(n):
        for j in range(i + 1, n):
            c = classe_identite(slots[i], slots[j])
            paires[(i, j)] = c
            recens[c] += 1
            verbe[(i, j)] = "verbe=" if slots[i][2] == slots[j][2] else "verbe≠"
    return {"n_unites": n, "paires": paires, "recensement": recens,
            "n_paires": n * (n - 1) // 2,
            "covariable_verbe": verbe,
            "paires_par_classe": {c: sorted(p for p, v in paires.items() if v == c)
                                  for c in PARTITIONS}}


def composantes(slots) -> dict:
    """Les composantes de slot : `owner` (16 blocs) et `entite` (20 blocs).

    Le CLUSTER de rééchantillonnage (§D.1) est la composante de slot — owner pour
    `P-own`, entité pour `P-ent` — seul cluster sous lequel les paires d'une
    strate sont indépendantes, et qui **préserve exactement** les effectifs.
    """
    par_owner, par_entite = {}, {}
    for i, (o, e, _v) in enumerate(slots):
        par_owner.setdefault(o, []).append(i)
        par_entite.setdefault(e, []).append(i)
    return {"owner": {k: tuple(v) for k, v in par_owner.items()},
            "entite": {k: tuple(v) for k, v in par_entite.items()}}


def clusters_de_strate(slots, strate: str = STRATE_DECISIONNELLE) -> dict:
    """Étiquette de cluster par UNITÉ et `K` pour la strate décisionnelle.

    **`K` = nombre de composantes PORTANT au moins une paire de la strate**
    (donc de taille ≥ 2) : une composante singleton ne contribue à aucune paire
    et n'entre pas dans la puissance. D'où `P-own` : K = 14 (N=30) → **16**
    (N=80) ; `P-ent` : K = 10 → **20**. Les composantes de taille ≥ 2 sont les
    clusters du rééchantillonnage (`membres`).
    """
    comp = composantes(slots)
    cle = "owner" if strate == P_OWN else "entite"
    ordre = sorted(comp[cle])
    etiq = {}
    for k, nom in enumerate(ordre):
        for i in comp[cle][nom]:
            etiq[i] = k
    porteuses = [nom for nom in ordre if len(comp[cle][nom]) >= 2]
    return {"strate": strate, "cle": cle, "K": len(porteuses),
            "K_composantes_totales": len(ordre),
            "n_singletons": len(ordre) - len(porteuses),
            "etiquette_par_unite": etiq,
            "membres": {k: comp[cle][nom] for k, nom in enumerate(porteuses)}}


def p_both_impossible(n_owners: int = N_OWNERS, n_entites: int = N_ENTITES,
                      n: int = N_UNITES) -> dict:
    """N-11, gravé : `P-both` exige `lcm(16, 20) = 80 | d`. La paire minimale sur
    l'ENTITÉ n'existe à aucun `N ≤ 80`."""
    l = math.lcm(n_owners, n_entites)
    return {"lcm": l, "N": n, "possible": l < n,
            "clause": "P-both exige lcm(16,20) = 80 | d : la paire minimale sur "
                      "l'ENTITÉ — le leurre canonique — n'existe à aucun N ≤ 80. "
                      "Le protocole n'a pas CHOISI le contraste d'owner ; "
                      "l'arithmétique du pool le lui a IMPOSÉ."}


# =========================================================================
#  Diversité (§4.7 `V-diversité`) et puissance (§4.7 `V-puissance`)
# =========================================================================

def diversite(slots) -> tuple[int, int, int]:
    """`(#owners, #entités, #verbes)` effectivement présents."""
    return (len({s[0] for s in slots}), len({s[1] for s in slots}),
            len({s[2] for s in slots}))


def diversite_attendue(n: int) -> tuple[int, int, int]:
    """`(min(N,16), min(N,20), min(N,5))` — combinatoire, ZÉRO bootstrap.

    Remplace `V-div`, qui mesurait un CARDINAL DE STRATE et **récompensait la
    dégénérescence** (défaut 0-7).
    """
    return (min(n, N_OWNERS), min(n, N_ENTITES), min(n, N_VERBES))


def k_requis(sigma0: float = SIGMA0, marge: float = MARGE_R1,
             z: float = Z_975) -> int:
    """`K_S ≥ (z·σ₀/|θ−T|)²`, arrondi à l'entier supérieur (§3)."""
    return int(math.ceil((z * sigma0 / abs(marge)) ** 2))


# =========================================================================
#  Paires intra / inter (§4.7 `V-paires`)
# =========================================================================

def paires_intra_inter(n_unites: int = N_UNITES, n_para: int = N_PARA):
    """Indices de lignes des paires intra et inter. La ligne `i*n_para + t` porte
    la paraphrase `t` de l'unité `i`. `n_intra = n·C(3,2)`, `n_inter = C(n,2)·9`."""
    intra, inter = [], []
    for i in range(n_unites):
        for s in range(n_para):
            for t in range(s + 1, n_para):
                intra.append((i * n_para + s, i * n_para + t))
    for i in range(n_unites):
        for j in range(i + 1, n_unites):
            for s in range(n_para):
                for t in range(n_para):
                    inter.append((i * n_para + s, j * n_para + t))
    return intra, inter


# =========================================================================
#  Jeu de candidats `R1_36` (§4.5) — DÉTERMINISTE, indépendant des données
# =========================================================================

def _ordre_decalage(i: int, candidats, n: int) -> list[int]:
    """Ordre par **décalage d'indice croissant** `d = (j − i) mod n`."""
    return sorted(candidats, key=lambda j: ((j - i) % n, j))


def jeu_candidats_R1(i: int, t: int, slots, strate: str = STRATE_DECISIONNELLE,
                     s: int = S_LEURRES, n_para: int = N_PARA) -> dict:
    """Jeu de candidats **pré-déclaré** du §4.5 : 1 cible + `s = 36` concurrents.

    - **cible unique** : même unité, type `t′` par la règle cyclique fixe
      para1 → para2 → para3 → para1 ;
    - **concurrents, le plus dur d'abord** : (1) les 12 états des 4 autres unités
      de la composante d'**owner** ; (2) les 9 états des 3 autres unités de la
      composante d'**entité** ; (3) 15 états de `P-0` par **décalage croissant**.
      Sur `P-ent`, les deux premiers blocs sont échangés (le plus dur d'abord
      reste la composante du slot partagé).

    **Aucune similarité mesurée n'entre ici** : toute sélection par proximité est
    un motif d'invalidation (§4.5, §6). La fonction ne reçoit aucun état.
    """
    n = len(slots)
    comp = composantes(slots)
    o, e = slots[i][0], slots[i][1]
    bloc_own = _ordre_decalage(i, [j for j in comp["owner"][o] if j != i], n)
    bloc_ent = _ordre_decalage(i, [j for j in comp["entite"][e] if j != i], n)
    exclus = set(bloc_own) | set(bloc_ent) | {i}
    bloc_p0 = _ordre_decalage(i, [j for j in range(n) if j not in exclus], n)
    ordre = ([bloc_own, bloc_ent, bloc_p0] if strate == P_OWN
             else [bloc_ent, bloc_own, bloc_p0])
    etats = [j * n_para + u for bloc in ordre for j in bloc for u in range(n_para)]
    cible = i * n_para + ((t + 1) % n_para)
    concurrents = etats[:s]
    return {"requete": i * n_para + t, "cible": cible,
            "concurrents": concurrents,
            "taille": 1 + len(concurrents),
            "blocs": {"owner": len(bloc_own) * n_para,
                      "entite": len(bloc_ent) * n_para,
                      "P-0": len(bloc_p0) * n_para},
            "strate": strate}


def r1_36(S, slots, strate: str = STRATE_DECISIONNELLE, s: int = S_LEURRES,
          n_para: int = N_PARA) -> dict:
    """`R1_36` : plus proche voisin en cosinus dans le jeu de candidats.

    `S` : matrice de similarité (n·3, n·3). **Égalités à ½ crédit** (§7). Rend
    aussi le vecteur de succès par requête (pour l'appariement de `ΔR1`) et
    l'étiquette de cluster de chaque requête.
    """
    S = np.asarray(S, dtype=np.float64)
    n = len(slots)
    clus = clusters_de_strate(slots, strate)["etiquette_par_unite"]
    succes, cluster, requetes = [], [], []
    for i in range(n):
        for t in range(n_para):
            j = jeu_candidats_R1(i, t, slots, strate, s, n_para)
            cand = [j["cible"]] + list(j["concurrents"])
            vals = S[j["requete"], cand]
            m = vals.max()
            gagnants = int((vals == m).sum())
            succes.append((1.0 / gagnants) if vals[0] == m else 0.0)
            cluster.append(clus[i])
            requetes.append(j["requete"])
    succes = np.asarray(succes, dtype=np.float64)
    return {"R1": float(succes.mean()), "succes": succes,
            "cluster": np.asarray(cluster), "requetes": np.asarray(requetes),
            "n_requetes": int(succes.size), "hasard": 1.0 / (1 + s),
            "taille_jeu": 1 + s, "strate": strate}


def delta_r1(succes_reel, succes_plancher) -> np.ndarray:
    """`ΔR1` **appariée par requête** (§4.3, D16) — différence, pas taux."""
    a = np.asarray(succes_reel, dtype=np.float64)
    b = np.asarray(succes_plancher, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("appariement impossible : formes différentes")
    return a - b


# =========================================================================
#  Quantités — fonctions PURES
# =========================================================================

def cosinus_matrice(X) -> np.ndarray:
    """Matrice des cosinus, **fp32** (§7). `X` : (n, d)."""
    A = np.asarray(X, dtype=np.float32)
    nrm = np.linalg.norm(A, axis=1, keepdims=True).astype(np.float32)
    nrm = np.where(nrm == 0, np.float32(1.0), nrm)
    U = (A / nrm).astype(np.float32)
    return (U @ U.T).astype(np.float32)


def auc_par_couche(cos_intra, cos_inter) -> float:
    """`A = P(cos_intra > cos_inter) + ½·P(=)` — Mann-Whitney (§3, §4.3).

    **Égalités à ½ crédit**, déclaré avant mesure. Invariante sous toute
    transformation strictement monotone appliquée **par couche** (M-1b, 0-1).
    """
    a = np.asarray(cos_intra, dtype=np.float64).ravel()
    b = np.sort(np.asarray(cos_inter, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")
    inf = np.searchsorted(b, a, side="left")
    sup = np.searchsorted(b, a, side="right")
    return float((inf.sum() + 0.5 * (sup - inf).sum()) / (a.size * b.size))


def compte_egalites(cos_intra, cos_inter) -> int:
    a = np.asarray(cos_intra, dtype=np.float64).ravel()
    b = np.sort(np.asarray(cos_inter, dtype=np.float64).ravel())
    return int((np.searchsorted(b, a, "right") - np.searchsorted(b, a, "left")).sum())


def recall_at_1(X, etiquettes, metrique: str = "cos") -> float:
    """`R1_full` : plus proche voisin (hors soi) parmi TOUS les états.

    **Indice ↔ indice UNIQUEMENT.** Mesurer une quantité indice↔fait est un motif
    d'invalidation du run (§4.4, §6).
    """
    A = np.asarray(X, dtype=np.float32)
    lab = np.asarray(etiquettes)
    if metrique == "cos":
        S = cosinus_matrice(A).astype(np.float64)
        np.fill_diagonal(S, -np.inf)
        voisin = np.argmax(S, axis=1)
    elif metrique == "l2":
        D = ((A[:, None, :].astype(np.float64) - A[None, :, :].astype(np.float64))
             ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        voisin = np.argmin(D, axis=1)
    else:
        raise ValueError(f"métrique inconnue : {metrique!r}")
    return float(np.mean(lab[voisin] == lab))


def _gram_normalisee(X, normalisation: str) -> np.ndarray:
    """Gram **fp32** puis normalisation.

    - `"giraldo"` (LA convention, §3, **Giraldo et al. 2014**) : `A_ij = K_ij /
      (n·√(K_ii·K_jj))`, `tr(A) = 1`, invariante sous mise à l'échelle des LIGNES.
    - `"trace"` : `A = K / tr(K)` — contre-exemple ÉCHOUANT (défaut 0-2).
    """
    A = np.asarray(X, dtype=np.float32)
    n = A.shape[0]
    K = (A @ A.T).astype(np.float32)
    if normalisation == "giraldo":
        d = np.sqrt(np.clip(np.diag(K).astype(np.float64), 1e-30, None))
        return (K.astype(np.float64) / (n * np.outer(d, d)))
    if normalisation == "trace":
        tr = float(np.trace(K).astype(np.float64))
        return K.astype(np.float64) / (tr if tr != 0 else 1.0)
    raise ValueError(f"normalisation inconnue : {normalisation!r}")


def valeurs_propres(X, normalisation: str = "giraldo") -> np.ndarray:
    A = _gram_normalisee(X, normalisation)
    return np.clip(np.linalg.eigvalsh(A.astype(np.float64)), 0.0, None)


def entropie_matricielle(X, normalisation: str = "giraldo") -> float:
    """Entropie matricielle, convention **Giraldo et al. 2014**, α → 1 (§3, §4.6).

    `H = −Σ λ log λ`. **La normalisation des lignes est REQUISE** (défaut 0-2).
    `H ≤ log min(n, d)` ⇒ `n_a = 90` fixé, comparaisons de NIVEAUX interdites.
    """
    lam = valeurs_propres(X, normalisation)
    s = lam.sum()
    if s > 0:
        lam = lam / s
    nz = lam[lam > 0]
    return float(-(nz * np.log(nz)).sum())


def lambda1_ratio(X, normalisation: str = "giraldo") -> float:
    """`λ₁/Σλ` — obligatoire par couche (§2 (v), `V-λ₁`)."""
    lam = valeurs_propres(X, normalisation)
    s = lam.sum()
    return float(lam.max() / s) if s > 0 else float("nan")


def sous_echantillonner_H(X, n_a: int = N_A, b: int = B_SOUS_ECH, seed: int = SEED,
                          normalisation: str = "giraldo") -> dict:
    """`H` et `λ₁/Σλ` sur `b` sous-échantillons de taille `n_a` (§3)."""
    A = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    if n < n_a:
        raise ValueError(f"n = {n} < n_a = {n_a} : sous-échantillonnage impossible")
    hs, l1 = [], []
    for _ in range(b):
        idx = rng.choice(n, size=n_a, replace=False)
        hs.append(entropie_matricielle(A[idx], normalisation))
        l1.append(lambda1_ratio(A[idx], normalisation))
    hs, l1 = np.array(hs), np.array(l1)
    q = lambda v: (float(np.percentile(v, 25)), float(np.percentile(v, 75)))  # noqa: E731
    return {"n": n_a, "B": b, "H_mediane": float(np.median(hs)), "H_IQR": q(hs),
            "lambda1_mediane": float(np.median(l1)), "lambda1_IQR": q(l1),
            "borne_log_min_n_d": float(np.log(min(n_a, A.shape[1])))}


# =========================================================================
#  Incertitude — cluster = composante de slot (§D.1), argmax re-sélectionné
# =========================================================================

def bootstrap_par_cluster(stat_fn, k_clusters: int, b: int = B_BOOT,
                          seed: int = SEED, alpha: float = ALPHA_IC) -> dict:
    """Bootstrap par **composante de slot** (§4.3, §D.1) : on rééchantillonne les
    CLUSTERS avec remise, jamais les paires ni les requêtes.

    `stat_fn(indices_de_clusters)` rend un scalaire.
    """
    rng = np.random.default_rng(seed)
    ech = np.empty(b, dtype=np.float64)
    for k in range(b):
        ech[k] = stat_fn(rng.integers(0, k_clusters, size=k_clusters))
    fini = ech[np.isfinite(ech)]
    lo = float(np.percentile(fini, 100 * alpha / 2)) if fini.size else float("nan")
    hi = float(np.percentile(fini, 100 * (1 - alpha / 2))) if fini.size else float("nan")
    return {"B": b, "K": k_clusters, "ic_bas": lo, "ic_haut": hi,
            "moyenne": float(fini.mean()) if fini.size else float("nan"),
            "n_non_fini": int(b - fini.size), "echantillons": ech}


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p):
    """Quantile normal (Acklam, précision ~1e-9) — aucune dépendance scipy."""
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0 else float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def ic_bca(echantillons, theta_obs: float, jackknife, alpha: float = ALPHA_IC) -> dict:
    """IC **BCa** (Efron 1987) — requis dès qu'une borne dépasse 0.95 (M-16 : le
    percentile SOUS-COUVRE près de la borne 1)."""
    ech = np.asarray(echantillons, dtype=np.float64)
    ech = ech[np.isfinite(ech)]
    jk = np.asarray(jackknife, dtype=np.float64)
    prop = float((ech < theta_obs).mean())
    z0 = _phi_inv(min(max(prop, 1e-12), 1 - 1e-12))
    jbar = jk.mean()
    num = ((jbar - jk) ** 3).sum()
    den = 6.0 * (((jbar - jk) ** 2).sum() ** 1.5)
    a = float(num / den) if den != 0 else 0.0
    out = []
    for p in (alpha / 2, 1 - alpha / 2):
        z = _phi_inv(p)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(ech, 100 * _phi(adj))))
    return {"ic_bas": out[0], "ic_haut": out[1], "z0": z0, "a": a, "methode": "BCa"}


def ic_du_max(echantillons, theta_obs: float, jackknife=None,
              alpha: float = ALPHA_IC, seuil_bca: float = SEUIL_BCA) -> dict:
    """IC du **max sur les couches**, argmax re-sélectionné dans chaque
    rééchantillon (M-15), **BCa** dès qu'une borne dépasse `seuil_bca` (M-16).

    Rend aussi l'**estimateur débiaisé** `θ̂ = 2·max_obs − mean_b(θ*_b)` : le biais
    de sélection du max (`E[max − moyenne] ≈ 0.033` d'AUC) dépasse le couloir
    entier, donc le max brut **fuit la bande N**.
    """
    ech = np.asarray(echantillons, dtype=np.float64)
    fini = ech[np.isfinite(ech)]
    lo = float(np.percentile(fini, 100 * alpha / 2))
    hi = float(np.percentile(fini, 100 * (1 - alpha / 2)))
    methode = "percentile"
    if max(lo, hi) > seuil_bca and jackknife is not None:
        b = ic_bca(fini, theta_obs, jackknife, alpha)
        lo, hi, methode = b["ic_bas"], b["ic_haut"], b["methode"]
    return {"ic_bas": lo, "ic_haut": hi, "methode": methode,
            "max_observe": float(theta_obs),
            "moyenne_bootstrap": float(fini.mean()),
            "theta_debiaise": estimateur_debiaise(theta_obs, fini),
            "B": int(ech.size)}


def estimateur_debiaise(max_obs: float, echantillons) -> float:
    """`θ̂ = 2·max_obs − mean_b(θ*_b)` (M-15). Peut sortir de [0, 1] : **non
    tronqué** ici, la troncature casserait la couverture du BCa (§12.3)."""
    ech = np.asarray(echantillons, dtype=np.float64)
    ech = ech[np.isfinite(ech)]
    return float(2.0 * max_obs - ech.mean())


def ic_max_des_ic_par_couche(ic_par_couche):
    """**MOTIF D'INVALIDATION** (§4.3, §6) : « max des IC par couche » ne construit
    pas un IC du max — il ignore la re-sélection de l'argmax. Toujours rejeté."""
    raise ValueError(
        "« max des IC par couche » = motif d'invalidation (§4.3, §6) : l'IC du "
        "max exige la RE-SÉLECTION de l'argmax dans chaque rééchantillon.")


def permutation_max_couches(stat_par_couche, couches, b: int = B_BOOT,
                            seed: int = SEED) -> dict:
    """Bande **N** par permutation des **étiquettes d'unité à couche fixée avec
    recalcul de `max_ℓ`** (M-15) — FWER exact.

    `stat_par_couche(ell, rng)` rend la statistique de la couche `ell` sous une
    permutation tirée avec `rng`. La permutation des étiquettes de COUCHE est
    invalide (couches non échangeables).
    """
    rng = np.random.default_rng(seed)
    ech = np.empty(b, dtype=np.float64)
    for k in range(b):
        ech[k] = max(stat_par_couche(e, rng) for e in couches)
    return {"B": b, "q_0.95": float(np.percentile(ech, 95)),
            "echantillons": ech}


def permutation_etiquettes_unite(cos_intra, cos_inter, b: int = B_BOOT,
                                 seed: int = SEED) -> dict:
    """Nulle statistique du §5 (maillon 5) à couche fixée."""
    a = np.asarray(cos_intra, dtype=np.float64).ravel()
    c = np.asarray(cos_inter, dtype=np.float64).ravel()
    tout = np.concatenate([a, c])
    n_i = a.size
    rng = np.random.default_rng(seed)
    ech = np.empty(b, dtype=np.float64)
    for k in range(b):
        p = rng.permutation(tout.size)
        ech[k] = auc_par_couche(tout[p[:n_i]], tout[p[n_i:]])
    obs = auc_par_couche(a, c)
    return {"B": b, "auc_observee": obs, "q_0.95": float(np.percentile(ech, 95)),
            "p_unilateral": float((ech >= obs).mean()), "echantillons": ech}


def auc_stat_fn(cos_intra_unite, cos_inter_paire, membres_par_cluster):
    """Fabrique un `stat_fn` d'AUC pour `bootstrap_par_cluster`.

    `cos_intra_unite` : (n, 3) ; `cos_inter_paire` : (n, n, 9) ;
    `membres_par_cluster[k]` : les unités du cluster `k`.
    """
    A = np.asarray(cos_intra_unite, dtype=np.float64)
    B = np.asarray(cos_inter_paire, dtype=np.float64)

    def stat(indices_clusters):
        unites = np.concatenate([np.asarray(membres_par_cluster[int(k)])
                                 for k in indices_clusters])
        intra = A[unites].ravel()
        i, j = np.triu_indices(unites.size, k=1)
        garde = unites[i] != unites[j]
        if not garde.any():
            return float("nan")
        inter = B[unites[i][garde], unites[j][garde]].ravel()
        return auc_par_couche(intra, inter)

    return stat


def v_plat(courbes_par_unite, b: int = B_BOOT, seed: int = SEED) -> dict:
    """`V-plat` (§4.7) : courbe **centrée par unité**, PLATE ssi
    `R_obs ≤ q_0.95(R*)`. **Aucune constante posée.**"""
    C = np.asarray(courbes_par_unite, dtype=np.float64)
    C = C - C.mean(axis=1, keepdims=True)
    moy = C.mean(axis=0)
    r_obs = float(moy.max() - moy.min())
    C0 = C - C.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    rs = np.empty(b, dtype=np.float64)
    for k in range(b):
        m = C0[rng.integers(0, n, size=n)].mean(axis=0)
        rs[k] = m.max() - m.min()
    q95 = float(np.percentile(rs, 95))
    return {"R_obs": r_obs, "q_0.95_R_etoile": q95, "B": b,
            "plate": bool(r_obs <= q95), "courbe_centree": moy}


# =========================================================================
#  Fenêtres et indexation (§3, §4.1)
# =========================================================================

def w_of_L(L: int) -> int:
    """`w(L) = max(1, ⌊L/12⌋)` ⇒ 1 / 2 / 2 pour L = 12 / 32 / 28."""
    return max(1, L // 12)


def fenetre_D3(L: int) -> tuple[int, int]:
    """Fenêtre D3 : `⌊L/2⌋ ± w(L)` ⇒ [5,7] / [14,18] / [12,16]."""
    c, w = L // 2, w_of_L(L)
    return (c - w, c + w)


def borne_multiplicite(L: int) -> float:
    """Borne conservatrice `(2w+1)/L` (§4.8)."""
    return (2 * w_of_L(L) + 1) / L


def couches_decisionnelles(L: int) -> list[int]:
    """`[1, L]` — `ℓ = 0` est publié comme nulle et **exclu de l'argmax** (§4.1)."""
    return list(range(1, L + 1))


def argmax_decisionnel(courbe) -> int:
    c = np.asarray(courbe, dtype=np.float64)
    return int(1 + np.argmax(c[1:]))


def argmin_decisionnel(courbe) -> int:
    c = np.asarray(courbe, dtype=np.float64)
    return int(1 + np.argmin(c[1:]))


# =========================================================================
#  Nulles du §5
# =========================================================================

def vecteurs_indicateurs_bpe(chaines, tokenize) -> np.ndarray:
    """Vecteurs indicateurs de tokens BPE (maillon 1, **0 forward**)."""
    toks = [set(tokenize(s)) for s in chaines]
    vocab = sorted(set().union(*toks)) if toks else []
    pos = {t: k for k, t in enumerate(vocab)}
    X = np.zeros((len(chaines), max(1, len(vocab))), dtype=np.float32)
    for r, ts in enumerate(toks):
        for t in ts:
            X[r, pos[t]] = 1.0
    return X


def auc_lex(chaines, tokenize, n_unites: int = N_UNITES,
            n_para: int = N_PARA) -> float:
    """`AUC_lex` (§5, maillon 1) : *combien le seul recouvrement lexical
    produit-il, sans cortex ?*"""
    X = vecteurs_indicateurs_bpe(chaines, tokenize)
    S = cosinus_matrice(X)
    intra, inter = paires_intra_inter(n_unites, n_para)
    return auc_par_couche([S[a, b] for a, b in intra], [S[a, b] for a, b in inter])


def r1_lex(chaines, slots, tokenize, strate: str = STRATE_DECISIONNELLE) -> float:
    """`R1_lex` (§5, maillon 1) — le même plancher, sur la statistique
    DÉCISIONNELLE."""
    X = vecteurs_indicateurs_bpe(chaines, tokenize)
    return r1_36(cosinus_matrice(X).astype(np.float64), slots, strate)["R1"]


def suffixe_commun(sequences) -> list:
    """Plus long suffixe de tokens commun (par exécution du tokenizer)."""
    if not sequences:
        return []
    k = 0
    m = min(len(s) for s in sequences)
    while k < m and len({tuple(s[len(s) - k - 1:len(s) - k]) for s in sequences}) == 1:
        k += 1
    return list(sequences[0][len(sequences[0]) - k:]) if k else []


CADRE, SLOT = "cadre", "slot"


def segments_par_type(owner: str, entity: str, verb: str, t: int) -> list[tuple]:
    """Décomposition d'un indice en segments `(texte, rôle)`.

    Rôle `SLOT` = slot de CONTENU (owner, entité). Rôle `CADRE` = préfixe,
    ponctuation **et le verbe** — qui n'est pas un slot d'identité (N-9) et est
    donc **conservé** par la nulle du maillon 2 (§5, re-spécifiée, 0-8).
    """
    if t == 0:
        return [(owner, SLOT), (" " + entity, SLOT), (" " + PARA1_VERB, CADRE)]
    if t == 1:
        return [(PARA2_PREFIX, CADRE), (_lower_first(owner), SLOT),
                (" " + entity, SLOT), (" " + verb, CADRE)]
    return [(PARA3_HEAD, CADRE), (entity, SLOT), (PARA3_MID, CADRE),
            (OWNER_OBJ[owner], SLOT), (PARA3_TAIL, CADRE)]


def spans_slots_par_type(owner: str, entity: str, verb: str, t: int,
                         inclure_verbe: bool = False) -> tuple:
    """Chaîne complète d'un indice + les **spans de caractères** de ses slots de
    contenu (owner, entité). Le cadre — préfixe, ponctuation, **verbe** — n'y est
    pas, sauf si `inclure_verbe` (construction FAUTIVE, contre-exemple du banc).
    """
    segs = segments_par_type(owner, entity, verb, t)
    texte, spans, pos = "", [], 0
    for k, (s, role) in enumerate(segs):
        if role == SLOT or (inclure_verbe and k in _INDEX_VERBE.get(t, ())):
            spans.append((pos, pos + len(s)))
        texte += s
        pos += len(s)
    return texte, spans


# Index du segment PORTANT LE VERBE dans `segments_par_type` : para1 → le verbe
# global gelé, para2 → le verbe de l'unité, para3 → aucun (le type n'en a pas).
_INDEX_VERBE = {0: (2,), 1: (3,)}


def span_verbe_par_type(owner: str, entity: str, verb: str, t: int):
    """Span de caractères du VERBE (cadre), ou `None` pour para3."""
    segs = segments_par_type(owner, entity, verb, t)
    idx = _INDEX_VERBE.get(t)
    if not idx:
        return None
    pos = sum(len(s) for s, _ in segs[:idx[0]])
    return (pos, pos + len(segs[idx[0]][0]))


def verbe_conserve(slots, tokenize, sequences, offsets) -> dict:
    """Vérifie PAR EXÉCUTION que la nulle du maillon 2 conserve le **verbe**
    verbatim (§5, re-spécifiée) : les tokens chevauchant le span du verbe sont
    identiques à ceux de l'indice réel."""
    if offsets is None:
        return {"conserve": None, "verifiable": False,
                "motif": "spans de caractères indisponibles (repli mot-à-mot)"}
    ok, n_verifies = True, 0
    for u, (owner, entity, verb) in enumerate(slots):
        for t in range(N_PARA):
            sp = span_verbe_par_type(owner, entity, verb, t)
            if sp is None:
                continue
            texte, _ = spans_slots_par_type(owner, entity, verb, t)
            ids, offs = offsets(texte)
            nul = sequences[u * N_PARA + t]
            for k, (s, e) in enumerate(offs):
                if s < sp[1] and sp[0] < e:
                    n_verifies += 1
                    ok = ok and (nul[k] == int(ids[k]))
    return {"conserve": bool(ok), "verifiable": True,
            "n_tokens_de_verbe_verifies": n_verifies}


def nulle_cadre(slots, tokenize, filler_id, offsets=None,
                effacer_le_verbe: bool = False) -> dict:
    """**Maillon 2, re-spécifiée (0-8)** : *même cadre de type, **verbe
    conservé**, slots de contenu remplacés par un remplissage neutre gelé,
    apparié en longueur et position.*

    L'ancienne clause « même suffixe » est ABANDONNÉE : le suffixe commun de
    para2 est **vide**, elle y était **vacuée**.

    Deux constructions, dans cet ordre de préférence :

    - `offsets(s) -> (ids, spans)` fourni (tokenizer rapide) : les tokens dont le
      span de caractères **chevauche** un slot de contenu sont remplacés par le
      remplissage ; tous les autres — cadre et **verbe** — sont conservés
      VERBATIM. Longueur et position sont préservées **par construction** ;
    - à défaut, découpage par segments, dont la fidélité au BPE de la chaîne
      entière est **vérifiée par exécution** et publiée.
    """
    seqs, longueurs_ok, fidele, n_remplaces = [], [], [], []
    for owner, entity, verb in slots:
        for t in range(N_PARA):
            texte, spans = spans_slots_par_type(owner, entity, verb, t,
                                                effacer_le_verbe)
            complet = list(tokenize(texte))
            if offsets is not None:
                ids, offs = offsets(texte)
                ids = list(ids)
                nul = [filler_id if any(a < e and s < b for (a, b) in spans)
                       else tid for tid, (s, e) in zip(ids, offs)]
                fidele.append(ids == complet)
            else:
                segs = segments_par_type(owner, entity, verb, t)
                iv = _INDEX_VERBE.get(t, ()) if effacer_le_verbe else ()
                reel, nul = [], []
                for k, (txt, role) in enumerate(segs):
                    tk = list(tokenize(txt))
                    reel.extend(tk)
                    nul.extend([filler_id] * len(tk)
                               if (role == SLOT or k in iv) else tk)
                fidele.append(reel == complet)
            n_remplaces.append(sum(1 for x in nul if x == filler_id))
            longueurs_ok.append(len(nul) == len(complet))
            seqs.append(nul)
    return {"sequences": seqs,
            "longueurs_appariees": bool(all(longueurs_ok)),
            "segmentation_fidele": bool(all(fidele)),
            "n_segmentations_infideles": int(len(fidele) - sum(fidele)),
            "tokens_remplaces_min_max": [int(min(n_remplaces)),
                                         int(max(n_remplaces))],
            "construction": "offsets" if offsets is not None else "segments",
            "verbe_conserve": True,
            "position_de_capture": "dernier token de l'indice, inchangée",
            "remplissage": REMPLISSAGE_NEUTRE}


def nulle_melangee(chaines, tokenize, seed: int = SEED) -> list[list[int]]:
    """Maillon 4 : corpus (a) **mélangé au niveau des tokens**, apparié en
    multiensemble, nombre d'items, longueur et position."""
    rng = np.random.default_rng(seed)
    out = []
    for s in chaines:
        t = list(tokenize(s))
        out.append([t[k] for k in rng.permutation(len(t))])
    return out


NULLE_PERMISSIVE_MARGE = 0.02      # §5, maillon 4 : `AUC < 0.5 − 0.02`


def nulle_melangee_permissive(auc_melangee: float) -> dict:
    """Clause pré-enregistrée du maillon 4 : si `AUC_mélangée < 0.5 − 0.02`, la
    nulle est **déclarée permissive** (collapse vers un attracteur), **exclue du
    plancher le plus haut**, et le fait est publié."""
    perm = bool(auc_melangee < 0.5 - NULLE_PERMISSIVE_MARGE)
    return {"auc_melangee": float(auc_melangee), "permissive": perm,
            "seuil": 0.5 - NULLE_PERMISSIVE_MARGE,
            "consequence": ("exclue du plancher le plus haut, fait publié"
                            if perm else "conservée dans les planchers")}


def partage_dernier_token(chaines, tokenize) -> float:
    """`V-suffixe` : fraction des paires (i<j) d'un même type dont le DERNIER
    token BPE coïncide."""
    derniers = [list(tokenize(s))[-1] for s in chaines]
    n = len(derniers)
    if n < 2:
        return float("nan")
    eg = sum(1 for i in range(n) for j in range(i + 1, n) if derniers[i] == derniers[j])
    return eg / (n * (n - 1) / 2)


def partage_suffixe_derive(n: int = N_UNITES) -> dict:
    """`V-suffixe` **entièrement dérivée** (§4.7, 0-8) :

    - **para1 = 1.0000** : tous les indices finissent par `PARA1_VERB`, chaîne
      globale gelée ⇒ 1 par construction ;
    - **para3 = 1.0000** : tous finissent par `PARA3_TAIL` ⇒ 1 ;
    - **para2 = `#{paires : 5 | d}/C(N,2)`** : para2 finit par le **verbe**, de
      période 5 ⇒ 5·C(⌈·⌉,2) sur C(N,2). À N = 80 : 600/3160 = **0.18987**.
    """
    q, r = divmod(n, N_VERBES)
    paires_5 = r * (q + 1) * q // 2 + (N_VERBES - r) * q * (q - 1) // 2
    total = n * (n - 1) // 2
    return {"para1": 1.0, "para2": paires_5 / total if total else float("nan"),
            "para3": 1.0, "n_paires_5_divise_d": paires_5, "C_n_2": total,
            "N": n}


# =========================================================================
#  Capture — hooks sur TOUTES les couches, UN forward (V-hooks, V-1pass)
# =========================================================================

def compte_hooks(module) -> int:
    """Nombre total de hooks survivants dans l'arbre (`V-hooks`)."""
    n = 0
    for m in module.modules():
        n += len(getattr(m, "_forward_hooks", {}))
        n += len(getattr(m, "_forward_pre_hooks", {}))
    return n


class CaptureToutesCouches:
    """Contexte : hooks sur les L blocs + un `forward_pre_hook` sur le bloc 0
    (`ℓ = 0`), retirés en `finally`. Un seul forward profile les L+1 couches."""

    def __init__(self, model, blocks):
        self.model, self.blocks = model, blocks
        self.handles = []
        self.etats: dict[int, np.ndarray] = {}
        self.positions = None
        self.n_forwards = 0

    def _extraire(self, hidden):
        import torch
        h = hidden[0] if isinstance(hidden, tuple) else hidden
        b = h.shape[0]
        idx = (self.positions if self.positions is not None
               else torch.full((b,), h.shape[1] - 1, dtype=torch.long,
                               device=h.device))
        sel = h[torch.arange(b, device=h.device), idx.to(h.device)]
        return sel.detach().float().cpu().numpy()

    def __enter__(self):
        def pre_bloc0(_m, inputs):
            self.etats[0] = self._extraire(inputs[0])

        def fait_hook(ell):
            def hook(_m, _i, output):
                self.etats[ell] = self._extraire(output)
            return hook

        def compte(_m, _i):
            self.n_forwards += 1

        self.handles.append(self.model.register_forward_pre_hook(compte))
        self.handles.append(self.blocks[0].register_forward_pre_hook(pre_bloc0))
        for ell, bloc in enumerate(self.blocks, start=1):
            self.handles.append(bloc.register_forward_hook(fait_hook(ell)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False

    def pile(self, L: int) -> np.ndarray:
        return np.stack([self.etats[ell] for ell in range(L + 1)], axis=0)


def capture_un_forward(model, blocks, input_ids, attention_mask, positions) -> dict:
    """UN forward, tous les états de couche (`V-1pass`, `V-hooks`, D8)."""
    import torch
    cap = CaptureToutesCouches(model, blocks)
    try:
        with cap:
            cap.positions = positions
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False)
            etats = {k: v.copy() for k, v in cap.etats.items()}
            n_fw = cap.n_forwards
    finally:
        cap.__exit__(None, None, None)
    return {"etats": etats, "n_forwards": n_fw,
            "hooks_restants": compte_hooks(model)}


# =========================================================================
#  Bandes (§4.5) et cellules (§4.8) — classifieurs PURS
# =========================================================================

BANDE_N, BANDE_M, BANDE_I, BANDE_V, BANDE_D = "N", "M", "I", "V", "D"
BANDE_HORS = "HORS-PARTITION"
BANDES = (BANDE_N, BANDE_M, BANDE_I, BANDE_V)


def bande_modele(ic_delta_r1, ic_r1, seuil: float = T_COULOIR) -> str:
    """Bande d'UN modèle (§4.5), conditions **dans cet ordre** :

    1. **N** — IC 95 % de `ΔR1` (appariée, argmax re-sélectionné) contient 0 ;
    2. **M** — `ΔR1` > 0 significatif **et** `IC_sup(R1_36) < T` ;
    3. **I** — `ΔR1` > 0 significatif **et** `IC_inf < T ≤ IC_sup` ;
    4. **V** — `IC_inf(R1_36) ≥ T`.

    Conventions de bord gravées : `IC_sup = T → I` ; `IC_inf = T → V` ; `ΔR1`
    dont l'IC touche 0 par la borne inférieure → **N**.

    Opérationnalisation DÉCLARÉE (le protocole ne nomme pas le cas `ΔR1`
    significativement NÉGATIF) : la bande **N** est l'absence de `ΔR1`
    significativement positif (`IC_inf(ΔR1) ≤ 0`), seule lecture qui rende la
    partition **exhaustive et mutuellement exclusive** sans amender une clause.
    """
    lo_d = float(ic_delta_r1[0])
    if lo_d <= 0.0:
        return BANDE_N
    lo, hi = float(ic_r1[0]), float(ic_r1[1])
    if hi < seuil:
        return BANDE_M
    if lo >= seuil:
        return BANDE_V
    return BANDE_I


def delta_r1_negatif(ic_delta_r1) -> bool:
    """Sous-étiquette DESCRIPTIVE du cas replié dans N : `ΔR1` significativement
    négatif (`IC_sup < 0`). Publiée, jamais décisionnelle."""
    return float(ic_delta_r1[1]) < 0.0


def mention_v_plus(ic_r1, seuil_plus: float = T_COULOIR_PLUS) -> bool:
    """Mention **V⁺** si `IC_inf ≥ 0.50` (§4.5)."""
    return float(ic_r1[0]) >= seuil_plus


def bande_gate(bande_gpt2: str, bande_smollm2: str) -> str:
    """**Overlay `D`, précédence gravée** (§4.5) : (1) si l'un des deux modèles
    rend `I`, le verdict global est **`I`** ; (2) sinon bandes différentes ⇒
    **`D`** ; (3) sinon la bande commune."""
    if BANDE_I in (bande_gpt2, bande_smollm2):
        return BANDE_I
    return bande_gpt2 if bande_gpt2 == bande_smollm2 else BANDE_D


def cellule(l_contrast: int, l_H: int, fenetre, L: int,
            plate_contrast: bool, plate_H: bool) -> str:
    """Cellule C1-C4 d'UN modèle (§4.8). **C4 d'abord**."""
    bord = lambda x: x == 1 or x == L                       # noqa: E731
    if plate_contrast or plate_H or bord(l_contrast) or bord(l_H):
        return "C4"
    dedans = lambda x: fenetre[0] <= x <= fenetre[1]         # noqa: E731
    a, b = dedans(l_contrast), dedans(l_H)
    if a and b:
        return "C1"
    if a or b:
        return "C2"
    return "C3"


# =========================================================================
#  Profil complet d'une variante (partie GPU — NON EXÉCUTÉE en statut PROPOSE)
# =========================================================================

def profil_depuis_etats(etats, slots, strate: str = STRATE_DECISIONNELLE) -> dict:
    """Toutes les quantités du §4.3 à partir des états capturés.

    `etats[ℓ]` : (n·3, d) — la ligne `i*3 + t` porte la paraphrase `t` de l'unité
    `i`. **Aucune quantité indice↔fait** n'est calculable ici.
    """
    L = max(etats)
    n = len(slots)
    intra, inter = paires_intra_inter(n, N_PARA)
    part = partition_identite(slots)
    etiquettes = np.repeat(np.arange(n), N_PARA)
    out = {"L": L, "n_intra": len(intra), "n_inter": len(inter),
           "recensement_identite": part["recensement"],
           "auc": {}, "auc_par_classe": {}, "auc_par_verbe": {},
           "R1_36": {}, "R1_36_P-ent": {}, "succes_R1": {}, "R1_full": {},
           "H": {}, "lambda1": {}, "s_intra": {}, "s_inter": {}, "ratio": {},
           "z": {}, "egalites": {}}
    for ell in range(L + 1):
        X = etats[ell]
        S = cosinus_matrice(X)
        Sd = S.astype(np.float64)
        ci = np.array([S[a, b] for a, b in intra], dtype=np.float64)
        ce = np.array([S[a, b] for a, b in inter], dtype=np.float64)
        out["auc"][ell] = auc_par_couche(ci, ce)
        out["egalites"][ell] = compte_egalites(ci, ce)
        out["s_intra"][ell] = float(ci.mean())
        out["s_inter"][ell] = float(ce.mean())
        out["ratio"][ell] = float(ci.mean() / ce.mean()) if ce.mean() else float("nan")
        sd = float(ce.std(ddof=1))
        out["z"][ell] = float((ci.mean() - ce.mean()) / sd) if sd else float("nan")
        par_classe = {}
        for c in PARTITIONS:
            sel = [(a, b) for (i, j), cc in part["paires"].items() if cc == c
                   for a in range(i * N_PARA, i * N_PARA + N_PARA)
                   for b in range(j * N_PARA, j * N_PARA + N_PARA)]
            par_classe[c] = (auc_par_couche(ci, [S[a, b] for a, b in sel])
                             if sel else float("nan"))
        out["auc_par_classe"][ell] = par_classe
        par_verbe = {}
        for v in ("verbe=", "verbe≠"):
            sel = [(a, b) for (i, j), vv in part["covariable_verbe"].items() if vv == v
                   for a in range(i * N_PARA, i * N_PARA + N_PARA)
                   for b in range(j * N_PARA, j * N_PARA + N_PARA)]
            par_verbe[v] = (auc_par_couche(ci, [S[a, b] for a, b in sel])
                            if sel else float("nan"))
        out["auc_par_verbe"][ell] = par_verbe
        r_own = r1_36(Sd, slots, P_OWN)
        r_ent = r1_36(Sd, slots, P_ENT)
        out["R1_36"][ell] = r_own["R1"]
        out["R1_36_P-ent"][ell] = r_ent["R1"]
        out["succes_R1"][ell] = r_own["succes"].tolist()
        out["R1_full"][ell] = {m: recall_at_1(X, etiquettes, m) for m in ("cos", "l2")}
        out["H"][ell] = entropie_matricielle(X)
        out["lambda1"][ell] = lambda1_ratio(X)
    courbe = np.array([out["auc"][e] for e in range(L + 1)])
    out["l_contrast"] = argmax_decisionnel(courbe)
    out["l_H"] = argmin_decisionnel(np.array([out["H"][e] for e in range(L + 1)]))
    out["fenetre_D3"] = list(fenetre_D3(L))
    out["w"] = w_of_L(L)
    return out


def _prompts_du_corpus(corpus) -> list[str]:
    """Les indices, disposés `i*3 + t`."""
    return [p for triple in corpus["paraphrases"] for p in triple]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2", choices=list(MODELES))
    ap.add_argument("--variante", default="a", choices=list(VARIANTES))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results"
                                         / "layer-profile"))
    ap.add_argument("--go", action="store_true",
                    help="exigé pour toute mesure : le protocole est en PROPOSE "
                         "tant que le banc n'est pas à E = 0")
    args = ap.parse_args()

    print("=" * 78)
    print("I2 — layer_profile — EXP-2026-08-22-layer-profile.md")
    print("INSTRUMENT : M jamais instanciée, aucun gradient, engram/ non modifié.")
    print("=" * 78)
    if not args.go:
        print("REFUS : --go absent. Statut du protocole = PROPOSE ; la gate de "
              "pré-enregistrement (banc à E = 0) n'est pas franchie.")
        return 2

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engram.cortex import _find_blocks          # LECTURE SEULE

    t0 = time.time()
    dispo = torch.cuda.is_available()
    device = torch.device("cuda" if dispo else "cpu")
    print(f"device = {device} (cuda disponible : {dispo})")
    if not dispo:
        print("ANOMALIE : repli CPU — à rapporter, pas à contourner (§6).")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval().requires_grad_(False)
    blocks = _find_blocks(model)
    L = len(blocks)
    verdict_L = "PASS" if L == L_ATTENDU[args.model] else "FAIL"
    print(f"L (config) = {L} ; attendu {L_ATTENDU[args.model]} → V-L {verdict_L}")
    if verdict_L == "FAIL":
        print("ARRÊT (§4.7 V-L).")
        return 3

    tokenize = lambda s: [int(t) for t in tok.encode(s)]        # noqa: E731
    corpus = corpus_b_v3(tokenize) if args.variante == "b_v3" else corpus_a()
    prompts = _prompts_du_corpus(corpus)
    def offsets(s):
        enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
        return [int(x) for x in enc["input_ids"]], list(enc["offset_mapping"])

    if args.variante == "nulle_cadre":
        filler = tokenize(REMPLISSAGE_NEUTRE)[-1]
        seqs = nulle_cadre(corpus["slots"], tokenize, filler,
                           offsets if tok.is_fast else None)["sequences"]
    elif args.variante == "nulle_melangee":
        seqs = nulle_melangee(prompts, tokenize)
    else:
        seqs = [tokenize(p) for p in prompts]

    hash_avant = sha256_corpus(corpus)
    lmax = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), lmax), tok.pad_token_id, dtype=torch.long)
    msk = torch.zeros((len(seqs), lmax), dtype=torch.long)
    pos = torch.zeros(len(seqs), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
        msk[r, :len(s)] = 1
        pos[r] = len(s) - 1

    cap = capture_un_forward(model, blocks, ids.to(device), msk.to(device),
                             pos.to(device))
    vram = (torch.cuda.max_memory_allocated() / 2 ** 30) if dispo else 0.0
    prof = profil_depuis_etats(cap["etats"], corpus["slots"])
    prof.update({
        "modele": args.model, "variante": args.variante,
        "n_forwards": cap["n_forwards"],
        "V-1pass": "PASS" if cap["n_forwards"] == 1 else "FAIL",
        "hooks_restants": cap["hooks_restants"],
        "V-hooks": "PASS" if cap["hooks_restants"] == 0 else "FAIL",
        "V-L": verdict_L, "sha256_avant": hash_avant,
        "sha256_apres": sha256_corpus(corpus),
        "vram_gio": round(vram, 3), "duree_s": round(time.time() - t0, 2),
        "device": str(device), "seed": SEED,
    })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nom = f"{args.model.replace('/', '_')}-{args.variante}"
    (out / f"{nom}.json").write_text(
        json.dumps(prof, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"AUC ℓ*={prof['l_contrast']} : {prof['auc'][prof['l_contrast']]:.4f} ; "
          f"R1_36 = {prof['R1_36'][prof['l_contrast']]:.4f} ; H ℓ*={prof['l_H']} ; "
          f"forwards={prof['n_forwards']} ; VRAM={prof['vram_gio']} Gio ; "
          f"{prof['duree_s']} s")
    print(f"écrit : {out / (nom + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
