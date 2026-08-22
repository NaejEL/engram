# SPDX-License-Identifier: AGPL-3.0-or-later
"""I2 — instrument de profil par couche (`layer_profile`).

Protocole : `experiments/EXP-2026-08-22-layer-profile.md` (statut **PROPOSE** :
la gate de pré-enregistrement n'est pas franchie ; ce fichier est écrit, pas
exécuté sur GPU).

INSTRUMENT, PAS MÉCANISME : aucune injection, aucune écriture, `M` jamais
instanciée, `engram/` non modifié (`engram.cortex._find_blocks` est lu, jamais
touché), aucun gradient (D8), **aucun nouveau champ `EngramConfig`**.

Ce que fait le module :

- **capture** — hooks sur TOUTES les couches simultanément, posés et retirés dans
  un `try/finally` (`V-hooks`) ; **un seul forward par (modèle, variante)**
  (`V-1pass`) : profiler L couches coûte UN passage, jamais L.
- **indexation** — `ℓ = 0` = sortie des embeddings (+ positions), capturée par un
  `forward_pre_hook` sur le bloc 0 ; `ℓ ∈ [1, L]` = sortie du bloc `ℓ`.
  **L'argmax décisionnel ne se cherche que sur `[1, L]`** ; `ℓ = 0` est la nulle
  « encodage » et en est **exclu** (§4.1).
- **quantités** — AUC (primaire), Recall@1 (cosinus et L2), entropie matricielle
  (convention Giraldo complète), λ₁/Σλ, ventilations `s_intra` / `s_inter` /
  ratio / z.
- **cinq nulles** (§5) — `AUC_lex` (0 forward), nulle suffixe, `ℓ = 0`, corpus
  mélangé au niveau des tokens, et la nulle statistique (0.5 + bootstrap +
  permutation des étiquettes d'unité à couche fixée).

Les fonctions de calcul sont **pures et importables** — c'est ce que le banc
(`eval/gate_bench.py`, suite `i2`) et les tests CPU (`tests/test_layer_profile.py`)
exercent : `auc_par_couche`, `recall_at_1`, `entropie_matricielle`,
`lambda1_ratio`, `stratifier`, `bootstrap_par_unite`.

Conventions numériques fixées (§7) : cosinus et Gram **fp32**, valeurs propres
**fp64**, bootstrap **par unité** (jamais par paire) B = 10 000, égalités à
**½ crédit**.

Usage (APRÈS la gate de pré-enregistrement seulement) :
  .venv\\Scripts\\python eval\\layer_profile.py --model gpt2 --go
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

N_UNITES = 30                 # §4.2 : 30 unités
N_PARA = 3                    # trois types de paraphrase
N_A = 90                      # §3 : n_a = 90 — dépendance à `n` de H
N_INTRA_ATTENDU = 90          # §4.7 V-paires : 30 × C(3,2)
N_INTER_ATTENDU = 3915        # §4.7 V-paires : C(30,2) × 9
B_BOOT = 10_000               # §4.3 : bootstrap par unité
B_SOUS_ECH = 200              # §3 : sous-échantillonnage à n_a, B = 200
ALPHA_IC = 0.05               # IC 95 % percentile
SEED = 0                      # §7
S_LEURRES = 36                # §4.5 : budget v4
# §4.5 — couloir de faisabilité, RE-DÉRIVÉ ici (jamais recopié) :
AUC_COULOIR_050 = 0.50 ** (1.0 / S_LEURRES)
AUC_COULOIR_025 = 0.25 ** (1.0 / S_LEURRES)
AUC_COULOIR_050_29 = 0.50 ** (1.0 / 29)
# §7 — profondeurs attendues, re-lues du config par `V-L`.
L_ATTENDU = {"gpt2": 12, "HuggingFaceTB/SmolLM2-360M": 32, "Qwen/Qwen2.5-1.5B": 28}
MODELES = tuple(L_ATTENDU)
VARIANTES = ("a", "a_prime", "nulle_suffixe", "nulle_melangee")

# Remplissage neutre GELÉ de la nulle suffixe (§5, maillon 2). Un seul mot,
# choisi hors des cinq tables indexées par l'unité (`OWNERS`, `ENTITIES`,
# `VERBS`, `SECRETS_80`, `OWNER_OBJ`) : il ne porte donc aucun matériel d'unité.
REMPLISSAGE_NEUTRE = " thing"

STRATES = ("S0", "S1", "S2")   # (a) ; le jeu v3 est le bras séparé S3


# =========================================================================
#  Corpus (§4.2) — (a) `fact_pairs(30)` RE-PARAPHRASÉ par les règles gelées
# =========================================================================

def triplets_fact_pairs(n: int = N_UNITES) -> list[tuple[int, int, int]]:
    """Les triplets (owner, entity, verb) de `pool.fact_pairs`, en INDICES.

    Recopié de la règle de `fact_pairs` (`i mod len(table)`), pas de la fonction :
    `fact_pairs` rend des chaînes, I2 a besoin des slots pour la stratification.
    16 owners, 20 entités, 5 verbes.
    """
    return [(i % len(OWNERS), i % len(ENTITIES), i % len(VERBS)) for i in range(n)]


def paraphrases_de_triplets(triples) -> list[tuple[str, str, str]]:
    """Les trois indices paraphrasés d'une liste de triplets, par les règles
    GELÉES de `POOL_PARAPHRASES` (§7 / §15 A-1 de v3) — recopiées à l'identique,
    seule l'indexation des slots change (§4.2 : re-paraphrasage de (a))."""
    out = []
    for o, e, v in triples:
        owner, entity, verb = OWNERS[o], ENTITIES[e], VERBS[v]
        para1 = f"{owner} {entity} {PARA1_VERB}"
        para2 = f"{PARA2_PREFIX}{_lower_first(owner)} {entity} {verb}"
        para3 = f"{PARA3_HEAD}{entity}{PARA3_MID}{OWNER_OBJ[owner]}{PARA3_TAIL}"
        out.append((para1, para2, para3))
    return out


def corpus_a(n: int = N_UNITES) -> dict:
    """Corpus (a) PRIMAIRE : `fact_pairs(30)` re-paraphrasé (§D, §4.2)."""
    triples = triplets_fact_pairs(n)
    return {"nom": "a", "triplets": triples,
            "slots": [(OWNERS[o], ENTITIES[e], VERBS[v]) for o, e, v in triples],
            "paraphrases": paraphrases_de_triplets(triples)}


def corpus_a_prime(tokenize=None) -> dict:
    """Corpus (a′) = strate **S3** : le jeu d'unités de v3, TEL QUEL (§4.2).

    Bras séparé, JAMAIS fusionné avec (a). Import paresseux : la table v3 est
    définie par le BPE de GPT-2 (C-3), que (a) n'a pas à payer.
    """
    from pool import v3_unit_triples
    triples = v3_unit_triples(N_UNITES, tokenize)
    return {"nom": "a_prime", "triplets": triples,
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
#  Stratification (§4.2) — recouvrement de surface entre unités
# =========================================================================

def stratifier(slots) -> dict:
    """Strate de chaque paire d'unités par **nombre de slots de contenu partagés**.

    `slots[i]` = (owner, entity, verb) de l'unité `i`. 0 partagé → **S0**,
    1 → **S1**, 2 → **S2**, 3 → **S3-dégénéré** (deux unités identiques : hors
    strates de (a), consigné à part).

    Rend le recensement (à publier AVANT mesure, §4.7 `V-div`) et la carte des
    paires. Fonction PURE, aucun GPU, décidable sur la seule combinatoire.
    """
    n = len(slots)
    paires, recens = {}, {"S0": 0, "S1": 0, "S2": 0, "S3-degenere": 0}
    for i in range(n):
        for j in range(i + 1, n):
            k = sum(1 for a, b in zip(slots[i], slots[j]) if a == b)
            nom = "S3-degenere" if k == 3 else f"S{k}"
            paires[(i, j)] = nom
            recens[nom] += 1
    return {"n_unites": n, "paires": paires, "recensement": recens,
            "n_paires": n * (n - 1) // 2,
            "paires_par_strate": {s: sorted(p for p, v in paires.items() if v == s)
                                  for s in ("S0", "S1", "S2", "S3-degenere")}}


def strates_non_vides(slots, indices) -> dict:
    """Strates non vides sur un sous-multiensemble d'unités (`indices` peut
    contenir des répétitions : c'est le rééchantillonnage par unité).

    Une paire de positions dont les DEUX unités sont la même unité tirée deux
    fois n'entre dans aucune strate de (a) : elle n'est pas une paire d'unités.
    """
    car = stratifier(slots)["paires"]
    vus = {s: 0 for s in STRATES}
    m = len(indices)
    for a in range(m):
        for b in range(a + 1, m):
            i, j = indices[a], indices[b]
            if i == j:
                continue
            s = car[(min(i, j), max(i, j))]
            if s in vus:
                vus[s] += 1
    return vus


# =========================================================================
#  Paires intra / inter (§4.7 `V-paires`)
# =========================================================================

def paires_intra_inter(n_unites: int = N_UNITES, n_para: int = N_PARA):
    """Indices de lignes (0..n_unites*n_para-1) des paires intra et inter.

    Convention de disposition : la ligne `i*n_para + t` porte la paraphrase `t`
    de l'unité `i`. `n_intra = n·C(3,2)`, `n_inter = C(n,2)·9`.
    """
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
    transformation strictement monotone appliquée **par couche** : c'est la
    propriété qui interdit à l'anisotropie de déplacer l'argmax (M-1b).
    """
    a = np.asarray(cos_intra, dtype=np.float64).ravel()
    b = np.sort(np.asarray(cos_inter, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")
    inf = np.searchsorted(b, a, side="left")          # # {b < a}
    sup = np.searchsorted(b, a, side="right")         # # {b <= a}
    eg = sup - inf                                     # # {b == a}
    return float((inf.sum() + 0.5 * eg.sum()) / (a.size * b.size))


def compte_egalites(cos_intra, cos_inter) -> int:
    """Comptes d'égalités exactes (ventilation obligatoire, §4.3)."""
    a = np.asarray(cos_intra, dtype=np.float64).ravel()
    b = np.sort(np.asarray(cos_inter, dtype=np.float64).ravel())
    return int((np.searchsorted(b, a, "right") - np.searchsorted(b, a, "left")).sum())


def recall_at_1(X, etiquettes, metrique: str = "cos") -> float:
    """Fraction des états dont le plus proche voisin (hors soi) porte la MÊME
    unité — `metrique` ∈ {"cos", "l2"} (§4.3).

    **Indice ↔ indice UNIQUEMENT.** Mesurer une quantité indice↔fait est un
    motif d'invalidation du run (§4.4, §6) : cette fonction ne reçoit qu'un seul
    jeu d'états et une étiquette d'unité par état — elle ne peut pas en produire.
    """
    A = np.asarray(X, dtype=np.float32)
    lab = np.asarray(etiquettes)
    n = A.shape[0]
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

    - `"giraldo"` (LA convention du protocole, §3) : `A_ij = K_ij /
      (n·√(K_ii·K_jj))`, donc `tr(A) = 1` par construction et `A` est invariante
      sous mise à l'échelle des LIGNES de `X`.
    - `"trace"` : `A = K / tr(K)`. Conservée UNIQUEMENT comme contre-exemple
      échouant du banc (défaut 0-2) — elle confond `H` avec le profil de normes.
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
    """Valeurs propres de la Gram normalisée, en **fp64** (§7)."""
    A = _gram_normalisee(X, normalisation)
    lam = np.linalg.eigvalsh(A.astype(np.float64))
    return np.clip(lam, 0.0, None)


def entropie_matricielle(X, normalisation: str = "giraldo") -> float:
    """Entropie matricielle, convention **Giraldo et al. 2014**, α → 1 (§3, §4.6).

    `H = −Σᵢ λᵢ log λᵢ` sur les valeurs propres de `A_ij = K_ij/(n√(K_ii K_jj))`
    (`tr(A) = 1`). **La normalisation des lignes est REQUISE** : sans elle, `H`
    est confondue avec le profil de normes par couche (défaut 0-2). Log naturel
    (nats) ; `H ≤ log min(n, d)`, d'où `n_a = 90` fixé et l'interdiction de
    comparer des NIVEAUX entre corpus (§3, M-3).
    """
    lam = valeurs_propres(X, normalisation)
    s = lam.sum()
    if s > 0:
        lam = lam / s
    nz = lam[lam > 0]
    return float(-(nz * np.log(nz)).sum())


def lambda1_ratio(X, normalisation: str = "giraldo") -> float:
    """`λ₁/Σλ` — **obligatoire par couche** (§2 (v), porte `V-λ₁`) : sans lui un
    minimum de `H` n'est PAS interprétable."""
    lam = valeurs_propres(X, normalisation)
    s = lam.sum()
    return float(lam.max() / s) if s > 0 else float("nan")


def sous_echantillonner_H(X, n_a: int = N_A, b: int = B_SOUS_ECH, seed: int = SEED,
                          normalisation: str = "giraldo") -> dict:
    """`H` et `λ₁/Σλ` sur `b` sous-échantillons de taille `n_a` (§3, M-3) :
    médiane ± IQR. Rend `n` explicite — les NIVEAUX ne se comparent qu'à `n` égal.
    """
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
#  Incertitude — bootstrap PAR UNITÉ (§4.3), jamais par paire
# =========================================================================

def bootstrap_par_unite(stat_fn, n_unites: int = N_UNITES, b: int = B_BOOT,
                        seed: int = SEED, alpha: float = ALPHA_IC) -> dict:
    """Bootstrap **par unité** : on rééchantillonne les UNITÉS factuelles avec
    remise, jamais les paires (§4.3, M-4).

    `stat_fn(indices)` reçoit un tableau d'indices d'unités (avec répétitions) et
    rend un scalaire. IC 95 % percentile.
    """
    rng = np.random.default_rng(seed)
    ech = np.empty(b, dtype=np.float64)
    for k in range(b):
        ech[k] = stat_fn(rng.integers(0, n_unites, size=n_unites))
    fini = ech[np.isfinite(ech)]
    lo = float(np.percentile(fini, 100 * alpha / 2)) if fini.size else float("nan")
    hi = float(np.percentile(fini, 100 * (1 - alpha / 2))) if fini.size else float("nan")
    return {"B": b, "ic_bas": lo, "ic_haut": hi,
            "moyenne": float(fini.mean()) if fini.size else float("nan"),
            "n_non_fini": int(b - fini.size), "echantillons": ech}


def auc_stat_fn(cos_intra_unite, cos_inter_paire):
    """Fabrique un `stat_fn` d'AUC pour `bootstrap_par_unite`.

    `cos_intra_unite` : (n, 3) — les 3 cosinus intra de chaque unité.
    `cos_inter_paire` : (n, n, 9) — les 9 cosinus croisés de chaque paire d'unités.
    Les paires (unité tirée deux fois) sont exclues du bras inter : ce ne sont
    pas des paires d'unités.
    """
    A = np.asarray(cos_intra_unite, dtype=np.float64)
    B = np.asarray(cos_inter_paire, dtype=np.float64)

    def stat(indices):
        idx = np.asarray(indices)
        intra = A[idx].ravel()
        i, j = np.triu_indices(idx.size, k=1)
        garde = idx[i] != idx[j]
        if not garde.any():
            return float("nan")
        inter = B[idx[i][garde], idx[j][garde]].ravel()
        return auc_par_couche(intra, inter)

    return stat


def permutation_etiquettes_unite(cos_intra, cos_inter, b: int = B_BOOT,
                                 seed: int = SEED) -> dict:
    """Nulle statistique du §5 (maillon 5) : permutation des étiquettes d'unité
    entre paires, **à couche fixée** — les paires sont échangeables sous H₀ à
    l'INTÉRIEUR d'une couche (la permutation des étiquettes de COUCHE, elle, est
    invalide : M-4)."""
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


def v_plat(courbes_par_unite, b: int = B_BOOT, seed: int = SEED) -> dict:
    """`V-plat` (§4.7, re-dérivée M-4) : courbe **centrée par unité**, PLATE ssi
    `R_obs ≤ q_0.95(R*)`, B = 10 000. **Aucune constante posée.**

    `courbes_par_unite` : (n_unites, n_couches).

    Opérationnalisation DÉCLARÉE par le banc (le protocole fixe la statistique
    `R = max_ℓ − min_ℓ` et le bootstrap par unité, pas la fabrique de `R*`) :
    `R*` est le range de la courbe moyenne de rééchantillons d'unités tirés de la
    courbe **doublement centrée** (par unité PUIS par couche) — c.-à-d. le bruit
    par unité sans le profil observé, la seule façon d'obtenir une distribution
    de `R` sous une vérité plate. Permutation des étiquettes de couche : INVALIDE
    (couches non échangeables, variances différentes).
    """
    C = np.asarray(courbes_par_unite, dtype=np.float64)
    C = C - C.mean(axis=1, keepdims=True)              # centrage PAR UNITÉ
    moy = C.mean(axis=0)
    r_obs = float(moy.max() - moy.min())
    C0 = C - C.mean(axis=0, keepdims=True)             # vérité plate + bruit
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
    """`w(L) = max(1, ⌊L/12⌋)` (M-8) ⇒ 1 / 2 / 2 pour L = 12 / 32 / 28."""
    return max(1, L // 12)


def fenetre_D3(L: int) -> tuple[int, int]:
    """Fenêtre D3 : `⌊L/2⌋ ± w(L)` ⇒ [5,7] / [14,18] / [12,16]."""
    c, w = L // 2, w_of_L(L)
    return (c - w, c + w)


def borne_multiplicite(L: int) -> float:
    """Borne conservatrice `(2w+1)/L` (M-5)."""
    return (2 * w_of_L(L) + 1) / L


def couches_decisionnelles(L: int) -> list[int]:
    """`[1, L]` — `ℓ = 0` est publié comme nulle et **exclu de l'argmax** (§4.1)."""
    return list(range(1, L + 1))


def argmax_decisionnel(courbe) -> int:
    """Argmax sur `[1, L]` : `courbe` est indexée `ℓ = 0..L`, l'indice 0 est exclu."""
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
    """`AUC_lex` (§5, maillon 1) : *combien d'AUC le seul recouvrement lexical
    produit-il, sans cortex ?*"""
    X = vecteurs_indicateurs_bpe(chaines, tokenize)
    S = cosinus_matrice(X)
    intra, inter = paires_intra_inter(n_unites, n_para)
    return auc_par_couche([S[a, b] for a, b in intra], [S[a, b] for a, b in inter])


def suffixe_commun(sequences) -> list:
    """Plus long suffixe de tokens commun à toutes les séquences (par exécution
    du tokenizer, aucun jugement) — sert la nulle suffixe et `V-suffixe`."""
    if not sequences:
        return []
    k = 0
    m = min(len(s) for s in sequences)
    while k < m and len({tuple(s[len(s) - k - 1:len(s) - k]) for s in sequences}) == 1:
        k += 1
    return list(sequences[0][len(sequences[0]) - k:]) if k else []


def nulle_suffixe(chaines_par_type, tokenize, filler_id) -> list[list[int]]:
    """Maillon 2 : contenu d'unité remplacé par un **remplissage neutre gelé**,
    **même suffixe, même longueur, même position**.

    `chaines_par_type[t]` = les 30 indices du type `t`. Le suffixe conservé est
    le plus long suffixe de tokens COMMUN aux 30 indices du type (déterminé par
    exécution) ; tout ce qui le précède est remplacé par `filler_id` répété, à
    longueur inchangée. La longueur propre de chaque prompt est PRÉSERVÉE : c'est
    ce qui fait de cette nulle un plancher de **position et de suffixe** et non
    une constante.
    """
    out = []
    for chaines in chaines_par_type:
        seqs = [list(tokenize(s)) for s in chaines]
        suf = suffixe_commun(seqs)
        for s in seqs:
            k = len(s) - len(suf)
            out.append([filler_id] * k + list(suf))
    return out


def nulle_melangee(chaines, tokenize, seed: int = SEED) -> list[list[int]]:
    """Maillon 4 : corpus (a) **mélangé au niveau des tokens**, apparié en
    multiensemble, nombre d'items, longueur et position (la permutation d'une
    séquence conserve exactement son multiensemble et sa longueur ; la position
    de capture reste le dernier indice)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in chaines:
        t = list(tokenize(s))
        out.append([t[k] for k in rng.permutation(len(t))])
    return out


def partage_dernier_token(chaines, tokenize) -> float:
    """`V-suffixe` (§4.7, rétrogradée en intégrité) — opérationnalisation
    DÉCLARÉE par le banc : fraction des paires (i<j) d'un même type dont le
    DERNIER token BPE est identique. 1.0 = tous les indices du type finissent sur
    le même token."""
    derniers = [list(tokenize(s))[-1] for s in chaines]
    n = len(derniers)
    if n < 2:
        return float("nan")
    eg = sum(1 for i in range(n) for j in range(i + 1, n) if derniers[i] == derniers[j])
    return eg / (n * (n - 1) / 2)


# =========================================================================
#  Capture — hooks sur TOUTES les couches, UN forward (V-hooks, V-1pass)
# =========================================================================

def compte_hooks(module) -> int:
    """Nombre total de hooks (forward, pre-forward) survivants dans l'arbre —
    `V-hooks` : un hook survivant contaminerait tout run ultérieur."""
    n = 0
    for m in module.modules():
        n += len(getattr(m, "_forward_hooks", {}))
        n += len(getattr(m, "_forward_pre_hooks", {}))
    return n


class CaptureToutesCouches:
    """Contexte : hooks sur les L blocs + un `forward_pre_hook` sur le bloc 0
    (pour `ℓ = 0` = sortie des embeddings + positions), retirés en `finally`.

    Un seul forward suffit à profiler les L+1 couches (`V-1pass`) ; le compteur
    `n_forwards` est incrémenté par un pre-hook sur le module racine.
    """

    def __init__(self, model, blocks):
        self.model, self.blocks = model, blocks
        self.handles = []
        self.etats: dict[int, np.ndarray] = {}
        self.positions = None            # LongTensor [B] : dernier token réel
        self.n_forwards = 0

    def _extraire(self, hidden):
        import torch
        h = hidden[0] if isinstance(hidden, tuple) else hidden
        b = h.shape[0]
        idx = (self.positions if self.positions is not None
               else torch.full((b,), h.shape[1] - 1, dtype=torch.long,
                               device=h.device))
        sel = h[torch.arange(b, device=h.device), idx.to(h.device)]
        return sel.detach().float().cpu().numpy()      # capture en fp32 (§7)

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
        return False                      # aucune exception n'est avalée

    def pile(self, L: int) -> np.ndarray:
        """(L+1, n, d) — `ℓ = 0..L`."""
        return np.stack([self.etats[ell] for ell in range(L + 1)], axis=0)


def capture_un_forward(model, blocks, input_ids, attention_mask, positions) -> dict:
    """UN forward, tous les états de couche (`V-1pass`, `V-hooks`, D8).

    `torch.no_grad()` : aucun gradient nulle part.
    """
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
        cap.__exit__(None, None, None)     # idempotent : les handles sont vidés
    return {"etats": etats, "n_forwards": n_fw,
            "hooks_restants": compte_hooks(model)}


# =========================================================================
#  Bandes (§4.5) et cellules (§4.8) — classifieurs PURS
# =========================================================================

BANDE_V, BANDE_M, BANDE_N, BANDE_D = "V", "M", "N", "D"
BANDE_HORS = "HORS-PARTITION"


def bande_modele(ic_max, planchers, ic_toutes_couches,
                 seuil: float = AUC_COULOIR_025) -> str:
    """Bande d'UN modèle sur `max_ℓ AUC(ℓ | S2 ∪ S3)` (§4.5).

    - **V** : IC 95 % inf ≥ `seuil` (0.9622 re-dérivé = `0.25^(1/36)`) ;
    - **N** : IC ∩ [planchers] ≠ ∅ **à toutes les couches** ;
    - **M** : IC inf > plancher le plus haut **et** IC sup < `seuil`.

    Rend `HORS-PARTITION` pour toute observation qu'aucune des trois clauses ne
    couvre — le classifieur ne comble AUCUN trou : combler serait amender.
    """
    lo, hi = float(ic_max[0]), float(ic_max[1])
    p_max = max(planchers)
    if lo >= seuil:
        return BANDE_V
    chevauche = lambda ic: not (ic[1] < min(planchers) or ic[0] > p_max)  # noqa: E731
    if ic_toutes_couches and all(chevauche(ic) for ic in ic_toutes_couches):
        return BANDE_N
    if lo > p_max and hi < seuil:
        return BANDE_M
    return BANDE_HORS


def bande_gate(bande_gpt2: str, bande_smollm2: str) -> str:
    """**D — dissocié** : bande différente entre GPT-2 et SmolLM2 (§4.5)."""
    return bande_gpt2 if bande_gpt2 == bande_smollm2 else BANDE_D


def cellule(l_contrast: int, l_H: int, fenetre, L: int,
            plate_contrast: bool, plate_H: bool) -> str:
    """Cellule C1-C4 d'UN modèle (§4.8). **C4 d'abord** : une courbe PLATE ou un
    argmax au BORD retire la quantité du test joint."""
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

def profil_depuis_etats(etats, slots) -> dict:
    """Toutes les quantités de §4.3 à partir des états capturés.

    `etats[ℓ]` : (90, d) — la ligne `i*3 + t` porte la paraphrase `t` de l'unité
    `i`. **Aucune quantité indice↔fait** n'est calculable ici : la fonction ne
    reçoit que des états d'indices.
    """
    L = max(etats)
    intra, inter = paires_intra_inter(len(slots), N_PARA)
    car = stratifier(slots)["paires"]
    etiquettes = np.repeat(np.arange(len(slots)), N_PARA)
    out = {"L": L, "n_intra": len(intra), "n_inter": len(inter),
           "auc": {}, "auc_par_strate": {}, "recall_at_1": {}, "H": {},
           "lambda1": {}, "s_intra": {}, "s_inter": {}, "ratio": {}, "z": {},
           "egalites": {}}
    for ell in range(L + 1):
        X = etats[ell]
        S = cosinus_matrice(X)
        ci = np.array([S[a, b] for a, b in intra], dtype=np.float64)
        ce = np.array([S[a, b] for a, b in inter], dtype=np.float64)
        out["auc"][ell] = auc_par_couche(ci, ce)
        out["egalites"][ell] = compte_egalites(ci, ce)
        out["s_intra"][ell] = float(ci.mean())
        out["s_inter"][ell] = float(ce.mean())
        out["ratio"][ell] = float(ci.mean() / ce.mean()) if ce.mean() else float("nan")
        sd = float(ce.std(ddof=1))
        out["z"][ell] = float((ci.mean() - ce.mean()) / sd) if sd else float("nan")
        par_strate = {}
        for s in STRATES:
            sel = [(a, b) for (i, j), st in car.items() if st == s
                   for a in range(i * N_PARA, i * N_PARA + N_PARA)
                   for b in range(j * N_PARA, j * N_PARA + N_PARA)]
            par_strate[s] = (auc_par_couche(ci, [S[a, b] for a, b in sel])
                             if sel else float("nan"))
        out["auc_par_strate"][ell] = par_strate
        out["recall_at_1"][ell] = {m: recall_at_1(X, etiquettes, m)
                                   for m in ("cos", "l2")}
        out["H"][ell] = entropie_matricielle(X)
        out["lambda1"][ell] = lambda1_ratio(X)
    courbe = np.array([out["auc"][e] for e in range(L + 1)])
    out["l_contrast"] = argmax_decisionnel(courbe)
    out["l_H"] = argmin_decisionnel(np.array([out["H"][e] for e in range(L + 1)]))
    out["fenetre_D3"] = list(fenetre_D3(L))
    out["w"] = w_of_L(L)
    return out


def _prompts_du_corpus(corpus) -> list[str]:
    """Les 90 indices, disposés `i*3 + t` (§ convention de `paires_intra_inter`)."""
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
    corpus = corpus_a() if args.variante in ("a", "nulle_suffixe",
                                             "nulle_melangee") else corpus_a_prime()
    prompts = _prompts_du_corpus(corpus)
    if args.variante == "nulle_suffixe":
        filler = tokenize(REMPLISSAGE_NEUTRE)[-1]
        par_type = [[corpus["paraphrases"][i][t] for i in range(N_UNITES)]
                    for t in range(N_PARA)]
        brut = nulle_suffixe(par_type, tokenize, filler)
        seqs = [brut[t * N_UNITES + i] for i in range(N_UNITES) for t in range(N_PARA)]
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
        "n_forwards": cap["n_forwards"], "V-1pass": "PASS" if cap["n_forwards"] == 1
        else "FAIL",
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
          f"H ℓ*={prof['l_H']} ; forwards={prof['n_forwards']} ; "
          f"VRAM={prof['vram_gio']} Gio ; {prof['duree_s']} s")
    print(f"écrit : {out / (nom + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
