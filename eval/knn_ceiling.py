# SPDX-License-Identifier: AGPL-3.0-or-later
"""V2-D(a) v2 — kNN-LM nu : borne de l'étage des logits.

Protocole pré-enregistré : experiments/EXP-2026-08-21-knn-borne-logits-v2.md
(antécédent : EXP-2026-08-21-knn-borne-logits.md — TERMINE, INVALIDE).

Ce fichier est l'AMENDEMENT du script du run 1, pas une réécriture. Ce qui change,
et uniquement cela (§10 du protocole v2) :

  1. `T_q = c · med_{j≥2}(d²_j − d²_min)` **PAR REQUÊTE**. La médiane-de-médianes
     par bras du run 1 (`calibrate_temperature`) est ABANDONNÉE : elle confondait
     profondeur d'indice et température, et surtout elle n'était pas la formule
     pré-enregistrée.
  2. `c = 0.0` ≡ **un-hot** ≡ limite c→0⁺ ≡ **uniforme sur l'argmin-set** (§10).
     La grille décisionnelle est c ∈ {un-hot, 0.03, 0.1, 0.3, 1, 3} et la cellule
     est **`sup_c`** — le c atteignant le sup est loggé par requête.
  3. Le bras G n'a **plus de τ fixé** : la règle « quantile 0.95 des d²_min des
     requêtes-FAIT » du run 1 était dégénérée (d²_min = 0.0 exact ⇒ τ = 0). G est
     une **courbe** E3(τ) / P1(τ) / h(τ) sur une grille de τ.
  4. `P5B_VAR_GATE` **SUPPRIMÉ**, remplacé par la porte **V-var** : var(D_t) = 0 à
     1e-12 sur le sous-ensemble `p_kNN = 0` (identité, pas un test).
  5. Portes réécrites et ORDONNÉES, avec **arrêt dur** à la première échouée
     (décision PI §13.10) : V-base → V-cap → V-drift → V0 → V-tie →
     V1a/V1b/V1c → V-var → V2 → P1.
  6. Nouveaux logs obligatoires : **R1v** (rang du premier voisin dont la VALEUR
     est le token cible), **R2z** (cos normalisé par la nulle empirique du
     distracteur 30 k, dans les DEUX conditions), **p₁₀** et **p_max** par
     position, le **c-du-sup**, le **compte d'ex-æquo** de d²_min.

ARCHITECTURE IMPOSÉE (conservée du run 1) : deux étages strictement séparés.

  `--phase gpu`       passes forward ; elles LOGGENT des bruts et ne calculent
                      AUCUNE métrique de la grille : par requête le vecteur
                      `p_LM` COMPLET (log-probas fp32), l'état final (pré-lm_head),
                      l'état de la couche `layer_index`, la cible ; par datastore
                      les clés (fp32) et les tokens-valeurs.
  `--phase analysis`  tout le reste, SANS GPU : la grille λ × c, `sup_c`, P3, P4,
                      P5, P5c-id, P5f, P6, P7, P8, la courbe G, R1/R1v/R2z/R3/R4,
                      la frontière complète.

`p_LM` ne dépend ni de λ ni de c : c'est ce qui rend le balayage gratuit.

CONTRAINTES NUMÉRIQUES (portes de tests CPU bloquantes, §7) :
  * distances en **fp32**, calculées comme ‖q − k‖² (jamais ‖q‖²+‖k‖²−2qᵀk : à
    ‖h‖² ~ 10⁴ l'annulation catastrophique efface un quasi-match) ;
  * softmax kNN **après soustraction de d²_min** (sinon underflow silencieux) ;
  * mélange en **log-espace** (`logaddexp`), jamais en probas brutes ;
  * `p_LM` en log-probs fp32, **exponentiation fp64**, rang par **comptage strict**.

INTERDITS PORTÉS PAR CE SCRIPT (§2, §5, §10) :
  * les clés kNN ne passent JAMAIS par G/DG (`FastWeightMemory.phi`) — ce serait
    détruire l'invariance à la paraphrase que P1 mesure (D9) ;
  * le datastore ne se remplit JAMAIS pendant une mesure de logprob (D7) : il est
    `freeze()`é avant la première requête et toute écriture ensuite lève ;
  * aucun backprop, aucun optimiseur, aucune G apprise (D8/D9) ;
  * `knn_lambda` par défaut = 0.0 ⇒ λ=0 doit être bit-exact vs le E1 courant ;
  * `raw/gpu_raw.npz` du run 1 n'est relu QUE par la porte V-drift (comparaison
    bit-à-bit, JAMAIS substitution) — la re-collecte est intégrale et à neuf.

Usage :
  python eval/knn_ceiling.py --phase gpu --stage core        # V-cap/V0/V-drift/A/D
  python eval/knn_ceiling.py --phase analysis --stage gates   # V-tie…V2, P1, h
  python eval/knn_ceiling.py --phase gpu --stage rest         # B/C/E + λ=0
  python eval/knn_ceiling.py --phase analysis --stage all     # tout hors-ligne
  python eval/knn_ceiling.py --phase gpu --stage esc          # escalade 30 gabarits
  python eval/knn_ceiling.py --phase analysis --stage esc
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------- constantes
# Toutes fixées par le protocole pré-enregistré v2 (§4, §7, §9). Aucune n'est
# ajustable après mesure (D14).

K_NEIGHBORS = 8                       # §7 : k = 8
LAMBDA_STAR = 1.0 - math.exp(-0.05)   # §3 : λ* = 1 − e^(−0.05) = 0.0487706
LAMBDA_GRID = [0.02, LAMBDA_STAR, 0.05, 0.10, 0.25]   # §4.4, balayage hors-ligne

# §4.1 : grille de températures. 0.0 = UN-HOT = limite c→0⁺ = uniforme sur
# l'argmin-set. La cellule décisionnelle est `sup_c` sur CETTE grille.
UNHOT = 0.0
C_GRID = [UNHOT, 0.03, 0.1, 0.3, 1.0, 3.0]
ARMS = ["final", "inject"]            # bras P6 : état final vs couche layer_index

RANK_TOP = 10                         # « top-10 »
P1_PARAPHRASE_MIN = 2                 # §4 P1 : ≥ 2/3 paraphrases

# --- seuils dérivés, §3 (vérifiés Math D-1..D-5) ---
E3_THRESHOLD = 0.05                   # nats/token
# top-10 à p_kNN = 1, p_LM(cible) ≈ 0 : λ > (1−λ)·p₁₀ ⟺ p₁₀ < λ/(1−λ)
P10_FEASIBLE = math.exp(0.05) - 1.0                 # = 0.0512711 à λ*
MASS_FACTOR = (1.0 - LAMBDA_STAR) / LAMBDA_STAR     # = 19.5041 (facteur (ii))
INV_LAMBDA_STAR = 1.0 / LAMBDA_STAR                 # = 20.5043
F_HARD_BOUND = 0.5 / LAMBDA_STAR                    # = 10.2521 (p_P − p_M ≤ 1)

# --- tolérances des portes (§4.3, §6) ---
VCAP_TOL = 1e-5      # V-cap : |lm_head(h) − logits|_max
V1A_TOL = 1e-6       # V1a : identité sur p_kNN(y_t) = 0
V1C_TOL = 1e-6       # V1c : E3 mesuré vs recomposé (nats)
VVAR_TOL = 1e-12     # V-var : var(D_t) sur p_kNN = 0
V2_MIN_FEASIBLE = 6  # § 6 : n_faisable ≥ 6 requis pour un REJETE ; ≤ 5 ⇒ INCONCLUSIF

# SHA-256 (préfixe 16) du brut archivé du run 1 — porte V-drift (§4.3, §5.12).
DRIFT_REFERENCE_SHA16 = "0ef6f9914fad18ec"
DRIFT_REFERENCE = Path(__file__).resolve().parents[1] / "experiments" / "results" \
    / "knn-borne-logits" / "raw" / "gpu_raw.npz"

BOOT_B = 10000                        # §7 : bootstrap par secret, 10 000 tirages
BOOT_SEED = 777
PERM_SEED = 12345                     # P3 : permutation des valeurs, déterministe
DISTRACTOR_TOKENS = 30000             # §7
CHUNK = 1024                          # fenêtre GPT-2 pour la construction du store
P5C_DECILES = 10
ESC_N = 30                            # §4.5 : escalade, 30 gabarits de eval/pool.py
ESC_TRUE = 12                         # H vraie si n ≥ 12/30
ESC_FALSE = 5                         # H fausse si n ≤ 5/30


# =========================================================================
#  Primitives numériques — CPU pur, testables sans modèle HF
# =========================================================================

def squared_distances(q, keys):
    """d²_j = ‖q − k_j‖², **en fp32 et par différence**.

    Ne JAMAIS remplacer par ‖q‖² + ‖k‖² − 2qᵀk : à ‖h‖² ~ 10⁴ cette forme perd
    l'information d'un quasi-match par annulation catastrophique (piège §4.4 sous
    une nouvelle forme, porte de test CPU). Le fp16 est refusé explicitement.
    """
    q = np.asarray(q)
    keys = np.asarray(keys)
    if q.dtype == np.float16 or keys.dtype == np.float16:
        raise TypeError(
            "chaîne de distance en fp16 : INTERDIT (protocole §7, critère "
            "d'invalidation §6). Caster en fp32 avant d'appeler."
        )
    q32 = q.astype(np.float32, copy=False)
    k32 = keys.astype(np.float32, copy=False)
    diff = k32 - q32[None, :]
    return np.einsum("ij,ij->i", diff, diff).astype(np.float32)


def topk_neighbors(d2, k):
    """Indices des k plus proches voisins, triés par d² croissant (ex æquo : par
    indice croissant — déterministe)."""
    k = min(int(k), len(d2))
    idx = np.argsort(np.asarray(d2, dtype=np.float64), kind="stable")[:k]
    return idx


def per_query_temperature(d2_k, c):
    """`T_q = c · med_{j≥2}(d²_j − d²_min)` — **PAR REQUÊTE** (protocole v2 §4.1).

    `j ≥ 2` = les voisins autres que le plus proche (le terme j=1 vaut 0 par
    construction et écraserait la médiane vers 0). `c = 0` ⇒ T_q = 0 ⇒ un-hot.
    Un T_q nul par dégénérescence (tous les voisins à égalité) retombe aussi sur
    l'un-hot, qui est sa limite continue.
    """
    d2 = np.sort(np.asarray(d2_k, dtype=np.float64))
    if float(c) <= 0.0 or d2.size < 2:
        return 0.0
    return float(c) * float(np.median(d2[1:] - d2[0]))


def knn_weights(d2_k, temp):
    """softmax(−(d² − d²_min)/T) sur les k voisins, ou **un-hot** si T ≤ 0.

    * La soustraction de d²_min est OBLIGATOIRE : sans elle, exp(−d²/T) avec
      d² ~ 10⁴ underflow à 0 partout et la distribution devient un one-hot
      invisible (ou un NaN par 0/0). Porte de test CPU.
    * `T ≤ 0` ≡ **un-hot** ≡ limite c→0⁺ ≡ **uniforme sur l'argmin-set** (§10) :
      spécifié par le protocole, pas inventé ici. Le compte d'ex-æquo est loggé
      par la porte V-tie.
    """
    d2_k = np.asarray(d2_k, dtype=np.float64)
    m = float(d2_k.min())
    if float(temp) <= 0.0:
        w = (d2_k == m).astype(np.float64)
        return w / w.sum()
    z = -(d2_k - m) / float(temp)
    w = np.exp(z)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("softmax kNN dégénéré (underflow)")
    return w / s


def knn_distribution(d2_k, values_k, temp):
    """Distribution kNN agrégée par token, à TEMPÉRATURE ABSOLUE T = `temp`."""
    w = knn_weights(d2_k, temp)
    out: dict[int, float] = {}
    for v, wi in zip(np.asarray(values_k).tolist(), w.tolist()):
        out[int(v)] = out.get(int(v), 0.0) + float(wi)
    # Les tokens à masse EXACTEMENT nulle (un-hot hors argmin-set) sont retirés :
    # « les valeurs recevant de la masse » est la partition exacte de P5c-id.
    return {t: m for t, m in out.items() if m > 0.0}


def knn_distribution_c(d2_k, values_k, c):
    """Distribution kNN agrégée par token, au paramètre **c** de la grille :
    T_q est recalculée PAR REQUÊTE (`per_query_temperature`)."""
    return knn_distribution(d2_k, values_k, per_query_temperature(d2_k, c))


def knn_entropy(pk):
    """R4 : entropie (nats) de la distribution kNN agrégée par token."""
    return float(-sum(p * math.log(p) for p in pk.values() if p > 0.0))


def count_argmin_ties(d2_all):
    """V-tie : nombre d'entrées EXACTEMENT à d²_min (fp32 bit-à-bit)."""
    d2 = np.asarray(d2_all)
    return int((d2 == d2.min()).sum())


def rank_of_index(d2_all, i):
    """Rang (1-based) de l'entrée `i` dans l'ordre des d² croissants, ex æquo
    départagés par indice croissant (identique à `argsort(kind='stable')`)."""
    d2 = np.asarray(d2_all, dtype=np.float64)
    v = d2[int(i)]
    return int((d2 < v).sum() + (d2[:int(i)] == v).sum()) + 1


def rank_of_value(d2_all, values, target):
    """**R1v** (nouveau, protocole v2 §4.2) : rang du PREMIER voisin dont la
    **VALEUR** est le token cible. Distinct de R1, qui est le rang d'une entrée
    désignée à l'avance. Retourne -1 si aucune entrée ne porte la cible."""
    d2 = np.asarray(d2_all, dtype=np.float64)
    vals = np.asarray(values)
    hit = np.nonzero(vals == int(target))[0]
    if hit.size == 0:
        return -1
    best = int(hit[int(np.argmin(d2[hit]))])   # argmin stable ⇒ plus petit indice
    return rank_of_index(d2, best)


def lm_prob_stats(logp_row, target, top=RANK_TOP):
    """`p_LM` en log-probs fp32 → **exponentiation fp64** → `p_max`, `p₁₀`,
    `p_LM(cible)` et le rang de base par **comptage strict** (§4.4, Math Q2)."""
    lp = np.asarray(logp_row, dtype=np.float32).astype(np.float64)
    p = np.exp(lp)
    t = int(target)
    k = min(int(top), p.size)
    part = np.partition(p, -k)[-k:]
    return {"p_max": float(p.max()), "p10": float(np.sort(part)[0]),
            "p_target": float(p[t]), "logp_target": float(lp[t]),
            "rank_base": int((p > p[t]).sum()) + 1}


def mix_logprob(logp_lm_target, p_knn_target, lam):
    """log p_mix(y) = logaddexp(log(1−λ) + log p_LM(y), log λ + log p_kNN(y)).

    Mélange en LOG-ESPACE : avec p_LM ≈ e^−15, la forme en probas brutes perd le
    terme LM dès que λ p_kNN le domine, et λ=0 cesse d'être bit-exact.
    """
    lam = float(lam)
    if lam == 0.0:                      # λ=0 ⇒ IDENTITÉ bit-exacte (§5.4)
        return float(logp_lm_target)
    a = math.log1p(-lam) + float(logp_lm_target)
    if p_knn_target <= 0.0:
        return a
    b = math.log(lam) + math.log(float(p_knn_target))
    return float(np.logaddexp(a, b))


def mix_delta_nll(logp_lm_target, p_knn_target, lam):
    """ΔNLL_t = −log[(1−λ) + λ·p_kNN(y)/p_LM(y)] = −log(1−λ) − log1p(λr/(1−λ)).

    Forme SANS ANNULATION. La forme naïve `−(log p_mix − log p_LM)` additionne
    puis retranche `log p_LM` : à p_kNN/p_LM ~ 1e-20 elle rend une valeur ~1 ULP
    AU-DESSUS du bord −log(1−λ), ce qui est algébriquement impossible et fait
    échouer V1b sur du bruit d'arrondi. Ici le rapport reste en log-espace
    (`log λ + log p_kNN − log p_LM`, jamais de division de probas brutes) et
    `log1p` préserve les décréments infinitésimaux.
    """
    lam = float(lam)
    if lam == 0.0:
        return 0.0
    edge = -math.log1p(-lam)
    if p_knn_target <= 0.0:
        return edge                      # identité exacte (porte V1a)
    t = math.exp(math.log(lam) - math.log1p(-lam)
                 + math.log(float(p_knn_target)) - float(logp_lm_target))
    return edge - math.log1p(t)


def mix_decrement_vec(logp_lm, p_knn, lam):
    """δ_t = log1p(λ·r/(1−λ)) avec r = p_kNN(y_t)/p_LM(y_t) — le **décrément
    exact** de ΔNLL_t sous le bord −log(1−λ).

    C'est LE prédicat testable en flottant pour V1b : la décroissance stricte en r
    s'énonce `δ_t > 0`, vraie dès que p_kNN(y_t) > 0. Sur `ΔNLL_t = bord − δ_t`
    l'inégalité stricte devient invisible quand δ_t < ulp(bord) (r ≲ 1e-16) — un
    plancher de représentation, pas une violation algébrique.
    """
    lp = np.asarray(logp_lm, dtype=np.float64)
    pk = np.asarray(p_knn, dtype=np.float64)
    lam = float(lam)
    if lam == 0.0:
        return np.zeros_like(lp)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        t = np.exp(math.log(lam) - math.log1p(-lam) + np.log(pk) - lp)
    t = np.where(pk > 0.0, t, 0.0)
    return np.log1p(t)


def mix_delta_nll_vec(logp_lm, p_knn, lam):
    """Version vectorisée de `mix_delta_nll` (même forme sans annulation)."""
    lam = float(lam)
    if lam == 0.0:
        return np.zeros_like(np.asarray(logp_lm, dtype=np.float64))
    return -math.log1p(-lam) - mix_decrement_vec(logp_lm, p_knn, lam)


def mix_rank(logp_lm_full, pk, lam, target):
    """Rang (1-based) de `target` sous le mélange, EXACT sur tout le vocabulaire,
    par **comptage strict** en fp64. Recalculé hors GPU depuis le `p_LM` complet
    loggé + les k couples (d², token)."""
    p = np.exp(np.asarray(logp_lm_full, dtype=np.float32).astype(np.float64))
    if lam == 0.0:
        pm = p
    else:
        pm = (1.0 - lam) * p
        if pk:
            idx = np.fromiter(pk.keys(), dtype=np.int64)
            val = np.fromiter(pk.values(), dtype=np.float64)
            pm[idx] += lam * val
    return int((pm > pm[int(target)]).sum()) + 1


def mix_argmax(logp_lm_full, pk, lam):
    """argmax de p_mix. Le candidat est soit l'argmax de p_LM, soit un token du
    support kNN : le balayage se limite donc à ≤ k+1 tokens (P5f)."""
    p = np.exp(np.asarray(logp_lm_full, dtype=np.float32).astype(np.float64))
    base = int(p.argmax())
    if lam == 0.0 or not pk:
        return base
    cands = [base] + [int(t) for t in pk]
    best, best_v = base, (1.0 - lam) * p[base] + lam * pk.get(base, 0.0)
    for t in cands:
        v = (1.0 - lam) * p[t] + lam * pk.get(t, 0.0)
        if v > best_v:
            best, best_v = t, v
    return best


def sup_over_c(logp_row, d2k, valsk, target, lam, c_grid=None):
    """**Cellule décisionnelle `sup_c`** (§4.1, B-1) : supremum de la famille de
    températures sur la grille DÉCLARÉE. Le sup est pris sur la décision (rang
    minimal) ; les ex æquo sont départagés par la masse p_kNN(cible) puis par
    l'ordre de la grille. Le **c atteignant le sup est retourné et loggé**.

    Ce n'est PAS un choix post-hoc : la grille est pré-enregistrée et la
    multiplicité est absorbée par P3, évalué au MÊME `sup_c`.
    """
    c_grid = C_GRID if c_grid is None else c_grid
    per_c, best_key, best_c = {}, None, None
    for c in c_grid:
        T = per_query_temperature(d2k, c)
        pk = knn_distribution(d2k, valsk, T)
        r = mix_rank(logp_row, pk, lam, target)
        mass = float(pk.get(int(target), 0.0))
        per_c[c] = {"T_q": T, "rank": r, "p_knn_target": mass,
                    "H_knn": knn_entropy(pk), "pk": pk}
        key = (r, -mass)
        if best_key is None or key < best_key:
            best_key, best_c = key, c
    b = per_c[best_c]
    return {"c_sup": best_c, "rank": b["rank"], "p_knn_target": b["p_knn_target"],
            "H_knn": b["H_knn"], "T_q": b["T_q"], "pk": b["pk"],
            "per_c": {c: {k: v for k, v in per_c[c].items() if k != "pk"}
                      for c in c_grid}}


def permute_values(values, seed=PERM_SEED):
    """P3 : permutation des VALEURS seules. Les clés — donc toutes les distances —
    sont inchangées par construction (porte de test CPU)."""
    v = np.asarray(values).copy()
    rng = np.random.default_rng(seed)
    return v[rng.permutation(len(v))]


def first_bpe_target(tokenizer, prompt, continuation):
    """Cible = **premier token BPE** de `" <secret>"`, dérivée PAR DIFFÉRENCE
    (identique à `EngramEngine.logprob_continuation`) — jamais par un encodage
    direct de la continuation, qui perdrait l'espace de début de mot."""
    ids_prompt = tokenizer.encode(prompt)
    ids_full = tokenizer.encode(prompt + continuation)
    assert ids_full[:len(ids_prompt)] == ids_prompt, "tokenisation non préfixe-stable"
    ids_cont = ids_full[len(ids_prompt):]
    assert ids_cont, "continuation vide"
    return int(ids_cont[0]), list(ids_cont)


# ------------------------------------------------------------------ datastore

class Datastore:
    """Datastore kNN NU : clés = états du cortex gelé, valeurs = tokens suivants.

    Deux invariants portés par la classe (tests CPU (vi) et (vii)) :
      * `add()` copie les clés TELLES QUELLES — aucune projection G/DG, aucun `phi`.
        La classe n'a ni G ni référence à `FastWeightMemory`.
      * `freeze()` est appelé avant la première requête ; toute écriture ensuite
        lève. C'est la garantie mécanique que le datastore ne se remplit jamais
        pendant une mesure de logprob (D7).
    """

    def __init__(self, name=""):
        self.name = name
        self._keys: list[np.ndarray] = []
        self._values: list[int] = []
        self.keys: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.frozen = False

    def add(self, keys, values):
        if self.frozen:
            raise RuntimeError(
                "datastore gelé : écriture interdite (D7 — le datastore ne se "
                "remplit jamais pendant une mesure)"
            )
        keys = np.asarray(keys, dtype=np.float32)
        values = np.asarray(values, dtype=np.int64)
        assert keys.ndim == 2 and keys.shape[0] == values.shape[0]
        self._keys.append(keys)
        self._values.append(values)

    def freeze(self):
        self.keys = (np.concatenate(self._keys, axis=0) if self._keys
                     else np.zeros((0, 0), dtype=np.float32))
        self.values = (np.concatenate(self._values, axis=0) if self._values
                       else np.zeros((0,), dtype=np.int64))
        self.frozen = True
        return self

    def __len__(self):
        return 0 if self.values is None else int(len(self.values))

    def query(self, q, k=K_NEIGHBORS, values=None):
        """Lecture seule : (indices, d², tokens-valeurs, d² complet) des k plus
        proches voisins. `values` permet d'injecter un vecteur de valeurs PERMUTÉES
        (P3) sans toucher aux clés — les distances sont donc rigoureusement
        identiques."""
        assert self.frozen, "query() sur un datastore non gelé"
        d2 = squared_distances(q, self.keys)
        idx = topk_neighbors(d2, k)
        vals = self.values if values is None else np.asarray(values)
        return idx, d2[idx].astype(np.float64), vals[idx], d2


def d2_matrix(Q, K):
    """Matrice [n_requêtes, n_entrées] de d², **toujours par différence et en fp32**
    (jamais ‖q‖²+‖k‖²−2qᵀk). Une ligne à la fois : à 30 000 entrées × 768 dims,
    l'intermédiaire pèse déjà 92 Mo."""
    Q = np.asarray(Q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    out = np.empty((Q.shape[0], K.shape[0]), dtype=np.float32)
    for i in range(Q.shape[0]):
        diff = K - Q[i][None, :]
        out[i] = np.einsum("ij,ij->i", diff, diff)
    return out


def neighbors_from_matrix(d2mat, values, k=K_NEIGHBORS):
    """(idx, d2k, valsk) des k plus proches par LIGNE — argpartition puis tri.
    Les d² ne dépendent ni de λ ni de c : ce cache rend toute la grille gratuite."""
    d2mat = np.asarray(d2mat)
    n, m = d2mat.shape
    kk = min(int(k), m)
    if kk < m:
        part = np.argpartition(d2mat, kk - 1, axis=1)[:, :kk]
    else:
        part = np.tile(np.arange(m), (n, 1))
    d2k = np.take_along_axis(d2mat, part, axis=1)
    order = np.argsort(d2k, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    d2k = np.take_along_axis(d2k, order, axis=1).astype(np.float64)
    return idx, d2k, np.asarray(values)[idx]


def knn_weights_batch(d2k, c):
    """Poids kNN pour un LOT de requêtes, `T_q` recalculée par ligne (§4.1)."""
    d2k = np.asarray(d2k, dtype=np.float64)
    m = d2k.min(axis=1, keepdims=True)
    if float(c) <= 0.0:
        w = (d2k == m).astype(np.float64)
        return w / w.sum(axis=1, keepdims=True)
    srt = np.sort(d2k, axis=1)
    T = float(c) * np.median(srt[:, 1:] - srt[:, :1], axis=1, keepdims=True)
    w = np.where(T > 0.0, np.exp(-(d2k - m) / np.where(T > 0.0, T, 1.0)),
                 (d2k == m).astype(np.float64))
    return w / w.sum(axis=1, keepdims=True)


def target_mass_batch(d2k, valsk, targets, c):
    """p_kNN(y_t) par position, en lot. Retourne (masse, poids)."""
    w = knn_weights_batch(d2k, c)
    hit = (np.asarray(valsk) == np.asarray(targets, dtype=np.int64)[:, None])
    return (w * hit).sum(axis=1), w


# ------------------------------------------------------------- statistiques

def pearson(x, y):
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2:
        return math.nan
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / den if den > 0 else math.nan


def rankdata(xs):
    xs = list(xs)
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def sign_test_exact(n_success, n_total):
    """§4 P1 : test des signes exact, p = P(X ≥ n_success) sous X ~ Bin(n, 1/2)."""
    return float(sum(math.comb(n_total, i) for i in range(n_success, n_total + 1))
                 / 2.0 ** n_total)


def binom_tail_ge(n_success, n_total, p):
    """P(X ≥ n_success) sous X ~ Bin(n_total, p) — escalade (§4.5)."""
    return float(sum(math.comb(n_total, i) * p ** i * (1 - p) ** (n_total - i)
                     for i in range(n_success, n_total + 1)))


def bootstrap_mean(values, b=BOOT_B, seed=BOOT_SEED):
    """§7 : bootstrap PAR SECRET (l'unité est le secret, jamais la graine)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "B": b, "n": 0}
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(b, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)), "B": b, "n": int(len(v))}


def bootstrap_median_log(values, b=BOOT_B, seed=BOOT_SEED):
    """§7 : bootstrap **EN LOG**, par secret, 10 000 tirages.

    `median(log x) = log(median x)` (la médiane commute avec toute transformation
    monotone) : l'IC bootstrap calculé en log s'exponentie donc exactement. Les
    zéros (log = −∞) sont admis et propres tant qu'ils sont minoritaires ; leur
    compte est rapporté. Couverture réelle 88-92 % à N=10 ⇒ **« indicatif, N=10 »**
    (Math Q4).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"median": math.nan, "ci_lo": math.nan, "ci_hi": math.nan,
                "B": b, "n": 0, "n_zeros": 0, "note": "indicatif, N=0"}
    with np.errstate(divide="ignore"):
        lv = np.log(v)
    rng = np.random.default_rng(seed)
    draws = np.median(lv[rng.integers(0, len(v), size=(b, len(v)))], axis=1)
    return {"median": float(np.median(v)),
            "ci_lo": float(np.exp(np.percentile(draws, 2.5))),
            "ci_hi": float(np.exp(np.percentile(draws, 97.5))),
            "B": b, "n": int(len(v)), "n_zeros": int((v == 0).sum()),
            "note": "IC bootstrap EN LOG — indicatif, N=%d" % len(v)}


def mean_sd(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return math.nan, math.nan
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(out_dir, gate, detail, res=None):
    """Arrêt dur (§13.10) : la porte échouée est journalisée COMME RÉSULTAT
    (porte + dérivation + chiffre mesuré), sans proposer d'amendement."""
    payload = {"gate": gate, "detail": detail, "verdict": "ARRET DUR — porte échouée",
               "note": "journalisée comme résultat (décision PI §13.10) ; "
                       "aucun amendement de porte n'est proposé"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"GATE_FAILURE_{gate}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res is not None:
        res["gate_failure"] = payload
        _dump(out_dir, res, [])
    print(f"\n  !! PORTE {gate} ÉCHOUÉE — ARRÊT DUR (§13.10). Détail : {detail}")
    sys.exit(2)


# =========================================================================
#  Passe GPU — LOG BRUT UNIQUEMENT
# =========================================================================

def _gpu_phase(args, out_dir):
    import torch
    from collateral import NEUTRAL_TEXT
    from fact_injection import FACT_TEMPLATE, QUESTIONS, SECRETS
    from read_gate import E1C_FACT, E1C_QUESTION
    from engram import EngramConfig, EngramEngine

    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stage = args.stage

    # §7 : variables fixées. Défauts EngramConfig (gpt2, layer 6, λ=2, cap=0.5,
    # η=0.2, decay=1e-3, thr=4, dg=8192/64, gate keysim, prune 512/0.10, seed 0)
    # + le second point de capture (inerte par défaut, armé ici).
    cfg = EngramConfig(capture_final_state=True)
    print(f"[knn_ceiling v2 / V2-D(a) — passe GPU, stage={stage}] {cfg.summary()}")
    print(f"  capture_final_state={cfg.capture_final_state} "
          f"knn_lambda(défaut)={cfg.knn_lambda} knn_k={cfg.knn_k} "
          f"knn_temp_c(défaut)={cfg.knn_temp_c} (0.0 ≡ un-hot ≡ uniforme sur l'argmin-set)")

    engine = EngramEngine(cfg)
    device = str(engine.cortex.device)
    print(f"  device : {device}")
    if device == "cpu":
        print("  !! ANOMALIE : repli CPU (CUDA indisponible) — à rapporter")
    model = engine.cortex.model
    L = cfg.layer_index
    tok = engine.tokenizer

    arrays: dict[str, np.ndarray] = {}
    meta: dict = {}
    if stage in ("rest", "esc") and (raw / "gpu_raw.npz").exists():
        arrays = dict(np.load(raw / "gpu_raw.npz"))
        meta = json.loads((raw / "gpu_meta.json").read_text(encoding="utf-8"))

    @torch.no_grad()
    def states_of(ids):
        """Un seul forward (sans cache) → (final[T,d], layerL[T,d]) fp32 CPU.
        `final` = sortie de la normalisation finale = entrée de lm_head."""
        x = torch.tensor([ids], device=engine.cortex.device)
        o = model(input_ids=x, use_cache=False, output_hidden_states=True)
        hs = o.hidden_states
        return (hs[-1][0].float().cpu().numpy().astype(np.float32),
                hs[L + 1][0].float().cpu().numpy().astype(np.float32))

    def build_store(text):
        """Traces égalisées (§5.7) : UNE entrée par token du texte, exactement comme
        M sous force_write=True (clé = état à t−1, valeur = token t)."""
        ids = tok.encode(text)
        fin, l6 = states_of(ids)
        return {"ids": np.asarray(ids, dtype=np.int64), "keys_final": fin[:-1],
                "keys_inject": l6[:-1], "values": np.asarray(ids[1:], dtype=np.int64)}

    @torch.no_grad()
    def score_request(prompt, continuation, *, read):
        """Mesure D7-conforme : clear_context AVANT, écriture coupée, tokenisation
        dérivée par différence (identique à `logprob_continuation`). Logge par
        position : p_LM COMPLET, état final, état couche L, cible, NLL, entropie."""
        _, ids_cont = first_bpe_target(tok, prompt, continuation)
        engine.clear_context()
        assert engine._context_len == 0 and engine._past is None, "clear_context inopérant"
        w0 = engine.memory.write_count
        engine.stream(prompt, read=read, write=False)
        logps, fins, injs, tgts, nlls, ents = [], [], [], [], [], []
        for tid in ids_cont:
            lp = torch.log_softmax(engine._last_logits.float(), dim=-1)
            logps.append(lp.cpu().numpy().astype(np.float32))
            fins.append(engine.cortex.last_h_final.cpu().numpy().astype(np.float32))
            injs.append(engine.cortex.last_h_pre.cpu().numpy().astype(np.float32))
            tgts.append(int(tid))
            nlls.append(-float(lp[tid]))
            ents.append(float(-(lp.exp() * lp).sum()))
            engine._consume(tid, read=read, write=False, force_write=False)
        assert engine.memory.write_count == w0, "D7/D8 : write pendant une mesure"
        return {"logp": np.stack(logps), "final": np.stack(fins),
                "inject": np.stack(injs), "target": np.asarray(tgts, dtype=np.int64),
                "nll_base": np.asarray(nlls, dtype=np.float32),
                "H": np.asarray(ents, dtype=np.float32)}

    def pass_fact(prefix, sec_list, *, load_M, template=None, questions=None,
                  per_secret=False, quiet=False):
        """Une passe de rappel : store du fait + 4 requêtes (exact + 3 paraphrases).
        `per_secret=True` (escalade) : `template`/`questions` sont indexés par secret."""
        template = template if template is not None else FACT_TEMPLATE
        questions = questions if questions is not None else QUESTIONS
        writes_log = {}
        for si, secret in enumerate(sec_list):
            tpl = template[si] if per_secret else template
            qs = questions[si] if per_secret else questions
            fact = tpl.format(secret=secret)
            store = build_store(fact)
            arrays[f"{prefix}_ids_{si}"] = store["ids"]
            arrays[f"{prefix}_keys_final_{si}"] = store["keys_final"]
            arrays[f"{prefix}_keys_inject_{si}"] = store["keys_inject"]
            arrays[f"{prefix}_values_{si}"] = store["values"]

            engine.reset_memory()
            engine.clear_context()
            w = 0
            if load_M:
                recs = engine.stream(fact, force_write=True)
                w = sum(r.wrote for r in recs)
                if w == 0:
                    print(f"  !! 0 write sur {secret} : run INVALIDE")
                    sys.exit(2)
            writes_log[secret] = w
            for qi, (qname, qprompt) in enumerate(qs):
                r = score_request(qprompt, f" {secret}", read=load_M)
                for key in ("logp", "final", "inject", "target", "nll_base", "H"):
                    arrays[f"{prefix}_q{qi}_{key}_{si}"] = r[key]
            if not quiet:
                print(f"  [{prefix}] {secret:<14} store={len(store['values'])} entrées "
                      f"writes={w}")
        return writes_log

    # ===================================================== stage « esc » (§4.5)
    if stage == "esc":
        from pool import SECRETS_80, fact_pairs
        print(f"\n--- ESCALADE : {ESC_N} gabarits de eval/pool.py (déclenchement "
              f"AUTOMATIQUE, autorisé §13.4) ---")
        t0 = time.time()
        pairs = fact_pairs(ESC_N)
        esc_secrets = SECRETS_80[:ESC_N]
        templates, qsets = [], []
        for i, (tpl, qexact) in enumerate(pairs):
            # Les gabarits de pool.py ne fournissent qu'UNE question (le préfixe
            # exact). Les 3 paraphrases sont dérivées DÉTERMINISTEMENT par
            # substitution du syntagme verbal (même propriétaire, même entité,
            # autre verbe) — la seule famille de paraphrases que la structure
            # combinatoire de pool.py définit sans ambiguïté.
            from pool import VERBS
            head = qexact
            verb = None
            for v in sorted(VERBS, key=len, reverse=True):
                if head.endswith(" " + v):
                    verb, head = v, head[: -(len(v) + 1)]
                    break
            assert verb is not None, f"verbe non identifié dans « {qexact} »"
            others = [v for v in VERBS if v != verb][:3]
            templates.append(tpl)
            qsets.append([("exact", qexact)]
                         + [(f"para{j+1}", f"{head} {v}") for j, v in enumerate(others)])
        esc_arrays: dict[str, np.ndarray] = {}
        saved = arrays
        arrays = esc_arrays
        writes = pass_fact("S", esc_secrets, load_M=False, template=templates,
                           questions=qsets, per_secret=True, quiet=True)
        arrays = saved
        esc_meta = {"config": cfg.summary(), "device": device,
                    "secrets": esc_secrets,
                    "templates": templates,
                    "questions": [[q for q, _ in qs] for qs in qsets],
                    "question_prompts": [[p for _, p in qs] for qs in qsets],
                    "writes": writes, "duration_s": time.time() - t0,
                    "n_gabarits": ESC_N}
        np.savez(raw / "esc_raw.npz", **esc_arrays)
        (raw / "esc_meta.json").write_text(
            json.dumps(esc_meta, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print(f"  {ESC_N} gabarits × 4 requêtes en {esc_meta['duration_s']:.1f}s "
              f"→ {raw / 'esc_raw.npz'}")
        return

    # ===================================================== stage « core »
    if stage in ("core", "all"):
        # ------------------------------- porte V-cap : hidden_states[-1] = entrée lm_head
        print("\n================ PORTE V-cap ================")
        with torch.no_grad():
            probe = tok.encode("The password is swordfish.")
            x = torch.tensor([probe], device=engine.cortex.device)
            o = model(input_ids=x, use_cache=False, output_hidden_states=True)
            relog = model.lm_head(o.hidden_states[-1])
            cap_dev = float((relog - o.logits).abs().max())
        print(f"  |lm_head(hidden_states[-1]) − logits|_max = {cap_dev:.3e} "
              f"(seuil ≤ {VCAP_TOL:.0e} fp32)")
        print(f"  V-cap → {'OK' if cap_dev <= VCAP_TOL else 'ÉCHEC'}")
        if cap_dev > VCAP_TOL:
            _fail(out_dir, "V-cap",
                  {"measured": cap_dev, "threshold": VCAP_TOL,
                   "derivation": "hidden_states[-1] doit être l'entrée exacte de "
                                 "lm_head ; sinon les clés ne sont pas l'état final"})

        secrets = SECRETS[:args.secrets]
        meta = {"config": cfg.summary(), "device": device,
                "d_model": engine.cortex.d_model, "layer_index": L,
                "secrets": secrets, "questions": [q for q, _ in QUESTIONS],
                "question_prompts": {q: p for q, p in QUESTIONS},
                "fact_template": FACT_TEMPLATE, "vcap_dev": cap_dev,
                "vcap_ok": bool(cap_dev <= VCAP_TOL), "capture_final_state": True,
                "protocol": "experiments/EXP-2026-08-21-knn-borne-logits-v2.md"}

        # ------------------------------- V0 (porte, 1 secret) — AVANT tout le reste
        print("\n================ PORTE V0 (1 secret, indice EXACT) ================")
        t0 = time.time()
        pass_fact("A", secrets[:1], load_M=False)
        st0 = Datastore("V0")
        st0.add(arrays["A_keys_final_0"], arrays["A_values_0"])
        st0.freeze()
        key0 = arrays["A_q0_final_0"][0]
        tgt0 = int(arrays["A_q0_target_0"][0])
        d2_all0 = squared_distances(key0, st0.keys)
        hit0 = np.nonzero(st0.values == tgt0)[0]
        ci0 = int(hit0[0]) if len(hit0) else -1
        r1_0 = rank_of_index(d2_all0, ci0) if ci0 >= 0 else -1
        r1v_0 = rank_of_value(d2_all0, st0.values, tgt0)
        ties0 = count_argmin_ties(d2_all0)
        print(f"  secret {secrets[0]} | store {len(st0)} entrées | "
              f"d²_min = {float(d2_all0.min()):.6f}")
        print(f"  R1 (rang de l'entrée correcte) = {r1_0} (attendu 1) | "
              f"R1v = {r1v_0} | ex-æquo à d²_min = {ties0}")
        print(f"  PORTE V0 : R1 = 1 — aucune clause R3 (protocole v2 §4.3)")
        print(f"  V0 → {'OK' if r1_0 == 1 else 'ÉCHEC'} ({time.time() - t0:.1f}s)")
        meta["V0_gpu_inline"] = {"R1": r1_0, "R1v": r1v_0, "ties_at_d2min": ties0,
                                 "d2_min": float(d2_all0.min()),
                                 "gate_clause": "R1 = 1 (aucune clause R3)",
                                 "ok": bool(r1_0 == 1)}
        if r1_0 != 1:
            np.savez(raw / "gpu_raw_V0.npz", **arrays)
            _fail(out_dir, "V0", {"R1": r1_0, "expected": 1,
                                  "derivation": "d²_min = 0.0 sur indice exact ⇒ "
                                                "l'entrée correcte est le 1er voisin ; "
                                                "sinon store/indexation cassés"})

        # ------------------------------- V-drift : contrôle croisé bit-à-bit
        print("\n================ PORTE V-drift (contrôle croisé bit-à-bit) ================")
        drift = {"reference": str(DRIFT_REFERENCE).replace("\\", "/"),
                 "reference_sha16_expected": DRIFT_REFERENCE_SHA16,
                 "note": "VÉRIFICATION D'ENVIRONNEMENT SEULEMENT (§13.5) — un écart "
                         "est une ANOMALIE à signaler, jamais une autorisation de "
                         "réutiliser les bruts archivés. La re-collecte est intégrale."}
        if not DRIFT_REFERENCE.exists():
            drift["status"] = "référence absente"
            print(f"  !! ANOMALIE : {DRIFT_REFERENCE} absent — contrôle croisé impossible")
        else:
            sha = sha256_file(DRIFT_REFERENCE)
            drift["reference_sha256"] = sha
            drift["reference_sha16_match"] = bool(sha[:16] == DRIFT_REFERENCE_SHA16)
            ref = np.load(DRIFT_REFERENCE)
            names = ["A_ids_0", "A_keys_final_0", "A_keys_inject_0", "A_values_0"]
            for qi in range(len(QUESTIONS)):
                for key in ("logp", "final", "inject", "target", "nll_base", "H"):
                    names.append(f"A_q{qi}_{key}_0")
            cmp = {}
            for nme in names:
                if nme not in ref.files or nme not in arrays:
                    cmp[nme] = {"status": "absent"}
                    continue
                a, b = np.asarray(arrays[nme]), np.asarray(ref[nme])
                if a.shape != b.shape:
                    cmp[nme] = {"status": "forme différente",
                                "shape_new": list(a.shape), "shape_ref": list(b.shape)}
                    continue
                identical = bool(np.array_equal(a, b))
                if a.dtype.kind == "f":
                    md = float(np.max(np.abs(a.astype(np.float64)
                                             - b.astype(np.float64)))) if a.size else 0.0
                else:
                    md = 0.0 if identical else float("nan")
                cmp[nme] = {"bitwise_identical": identical, "max_abs_dev": md,
                            "n": int(a.size)}
            drift["arrays"] = cmp
            ok = [v for v in cmp.values() if v.get("bitwise_identical") is not None]
            n_id = sum(1 for v in ok if v["bitwise_identical"])
            devs = [v["max_abs_dev"] for v in ok if not v["bitwise_identical"]
                    and not math.isnan(v["max_abs_dev"])]
            drift["n_arrays_compared"] = len(ok)
            drift["n_bitwise_identical"] = n_id
            drift["max_abs_dev_overall"] = float(max(devs)) if devs else 0.0
            drift["status"] = "aucun écart" if n_id == len(ok) else "ÉCART — ANOMALIE"
            print(f"  SHA-256 de la référence : {sha[:16]} "
                  f"(attendu {DRIFT_REFERENCE_SHA16}) → "
                  f"{'concordant' if drift['reference_sha16_match'] else 'DISCORDANT'}")
            print(f"  tableaux comparés : {n_id}/{len(ok)} identiques bit-à-bit | "
                  f"écart absolu max = {drift['max_abs_dev_overall']:.3e}")
            print(f"  V-drift → {drift['status']} "
                  f"(vérification d'environnement ; ne bloque pas)")
        meta["V_drift"] = drift

        # -------------------------------------------- passe A : store fait-seul, M=0
        print(f"\n--- passe A : store fait-seul, M=0, {len(secrets)} secrets ---")
        t0 = time.time()
        meta["writes_A"] = pass_fact("A", secrets, load_M=False)
        meta["duration_A"] = time.time() - t0
        print(f"  passe A : {meta['duration_A']:.1f}s")

        # ---------------------------------------- passe D : E3 par position, M=0
        # Avec M = 0, l'état du cortex sur NEUTRAL_TEXT ne dépend PAS du secret :
        # une seule passe suffit, les 10 stores sont appliqués hors-ligne.
        print("\n--- passe D : NEUTRAL_TEXT par position (M=0) ---")
        t0 = time.time()
        engine.reset_memory()
        ids_n = tok.encode(NEUTRAL_TEXT)
        engine.clear_context()
        w0 = engine.memory.write_count
        n_fin, n_inj, n_nll, n_H, n_tgt, n_lp = [], [], [], [], [], []
        with torch.no_grad():
            for tid in ids_n:
                if engine._last_logits is not None:
                    lp = torch.log_softmax(engine._last_logits.float(), dim=-1)
                    n_lp.append(lp.cpu().numpy().astype(np.float32))
                    n_fin.append(engine.cortex.last_h_final.cpu().numpy().astype(np.float32))
                    n_inj.append(engine.cortex.last_h_pre.cpu().numpy().astype(np.float32))
                    n_nll.append(-float(lp[tid]))
                    n_H.append(float(-(lp.exp() * lp).sum()))
                    n_tgt.append(int(tid))
                engine._consume(tid, read=False, write=False, force_write=False)
        assert engine.memory.write_count == w0, "D7 : write pendant la passe D"
        arrays["D_logp"] = np.stack(n_lp)
        arrays["D_final"] = np.stack(n_fin)
        arrays["D_inject"] = np.stack(n_inj)
        arrays["D_nll_base"] = np.asarray(n_nll, dtype=np.float32)
        arrays["D_H"] = np.asarray(n_H, dtype=np.float32)
        arrays["D_target"] = np.asarray(n_tgt, dtype=np.int64)
        meta["duration_D"] = time.time() - t0
        meta["neutral_tokens"] = len(ids_n)
        meta["neutral_positions"] = len(n_tgt)
        print(f"  {len(ids_n)} tokens, {len(n_tgt)} positions valides, "
              f"{meta['duration_D']:.1f}s")

    # ===================================================== stage « rest »
    if stage in ("rest", "all"):
        secrets = meta["secrets"]
        # ------------------------- passe B : M au point courant + kNN (descriptif)
        print("\n--- passe B : M chargée (point courant) + kNN ---")
        t0 = time.time()
        meta["writes_B"] = pass_fact("B", secrets, load_M=True)
        meta["duration_B"] = time.time() - t0
        print(f"  passe B : {meta['duration_B']:.1f}s")

        # ------------------------------------------------------- passe C : E1c
        print("\n--- passe C : E1c (M=0 et M chargée) ---")
        t0 = time.time()
        store_c = build_store(E1C_FACT)
        arrays["C_ids"] = store_c["ids"]
        arrays["C_keys_final"] = store_c["keys_final"]
        arrays["C_keys_inject"] = store_c["keys_inject"]
        arrays["C_values"] = store_c["values"]
        for tag, load_M in (("m0", False), ("mload", True)):
            engine.reset_memory()
            engine.clear_context()
            w = 0
            if load_M:
                recs = engine.stream(E1C_FACT, force_write=True)
                w = sum(r.wrote for r in recs)
            meta[f"writes_C_{tag}"] = w
            for wi, word in enumerate((" Marseille", " Paris")):
                r = score_request(E1C_QUESTION, word, read=load_M)
                for key in ("logp", "final", "inject", "target", "nll_base", "H"):
                    arrays[f"C_{tag}_w{wi}_{key}"] = r[key]
        meta["duration_C"] = time.time() - t0
        meta["e1c_fact"] = E1C_FACT
        meta["e1c_question"] = E1C_QUESTION
        print(f"  passe C : {meta['duration_C']:.1f}s "
              f"(writes M chargée = {meta['writes_C_mload']})")

        # -------------------------------------- passe E : store distracteur 30 k
        if not args.skip_distractor:
            print(f"\n--- passe E : store distracteur {args.distractor_tokens} tokens ---")
            t0 = time.time()
            root = Path(__file__).resolve().parents[1]
            rfc = root / "data" / "rfc9293.txt"
            if not rfc.exists():
                print(f"  !! ANOMALIE : {rfc} absent — bras distracteur NON exécuté")
                meta["distractor"] = None
            else:
                meta["rfc_sha256"] = sha256_file(rfc)
                meta["rfc_bytes"] = rfc.stat().st_size
                txt = rfc.read_text(encoding="utf-8", errors="replace")
                ids_r: list[int] = []
                for line in txt.splitlines(keepends=True):
                    ids_r.extend(tok.encode(line))
                    if len(ids_r) >= args.distractor_tokens + CHUNK:
                        break
                ids_r = ids_r[:args.distractor_tokens]
                dk, dv = [], []
                for start in range(0, len(ids_r), CHUNK):
                    chunk = ids_r[start:start + CHUNK]
                    if len(chunk) < 2:
                        continue
                    fin, _ = states_of(chunk)
                    dk.append(fin[:-1])
                    dv.append(np.asarray(chunk[1:], dtype=np.int64))
                arrays["E_keys_final"] = np.concatenate(dk, axis=0)
                arrays["E_values"] = np.concatenate(dv, axis=0)
                meta["distractor"] = {"tokens": len(ids_r), "chunk": CHUNK,
                                      "entries": int(len(arrays["E_values"])),
                                      "sha256": meta["rfc_sha256"],
                                      "sha16": meta["rfc_sha256"][:16]}
                meta["duration_E"] = time.time() - t0
                print(f"  {len(ids_r)} tokens → {len(arrays['E_values'])} entrées, "
                      f"{meta['duration_E']:.1f}s | SHA-16 rfc9293 = "
                      f"{meta['rfc_sha256'][:16]}")
        else:
            meta["distractor"] = None

        # ------------------------------------- contrôle §5.4 : λ=0 bit-exact
        print("\n--- contrôle §5.4 : λ=0 bit-exact vs le E1 courant (fin de run) ---")
        lam0 = {}
        for secret in secrets:
            engine.reset_memory()
            engine.clear_context()
            engine.stream(FACT_TEMPLATE.format(secret=secret), force_write=True)
            engine.clear_context()
            lp_ref, rk_ref = engine.logprob_continuation(QUESTIONS[0][1], f" {secret}")
            engine.reset_memory()
            engine.clear_context()
            lp_b, rk_b = engine.logprob_continuation(QUESTIONS[0][1], f" {secret}")
            lam0[secret] = {"lp_mem": lp_ref, "rank_mem": rk_ref,
                            "lp_base": lp_b, "rank_base": rk_b,
                            "delta": lp_ref - lp_b}
        meta["lambda0_replay"] = lam0
        d_mean, d_sd = mean_sd([v["delta"] for v in lam0.values()])
        meta["lambda0_replay_E1"] = {"mean": d_mean, "sd": d_sd, "n": len(lam0)}
        print(f"  E1 exact rejoué (M chargée, λ=0) : {d_mean:+.3f} ± {d_sd:.3f} "
              f"(N={len(lam0)})")

    meta["duration_gpu_s"] = meta.get("duration_gpu_s", 0.0) + (time.time() - t_start)
    np.savez(raw / "gpu_raw.npz", **arrays)
    (raw / "gpu_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n  passe GPU (stage={stage}) : {time.time() - t_start:.1f}s → "
          f"{raw / 'gpu_raw.npz'} "
          f"({(raw / 'gpu_raw.npz').stat().st_size / 1e6:.1f} Mo)")


# =========================================================================
#  Analyse hors-ligne — ZÉRO GPU
# =========================================================================

class Query:
    """Une requête loggée : p_LM complet + les deux clés candidates + la cible."""

    __slots__ = ("logp", "final", "inject", "target", "nll_base", "H", "tag")

    def __init__(self, logp, final, inject, target, nll_base, H, tag):
        self.logp, self.final, self.inject = logp, final, inject
        self.target, self.nll_base, self.H, self.tag = target, nll_base, H, tag

    def key(self, arm, p=0):
        return (self.final if arm == "final" else self.inject)[p]


def _escalation_analysis(out_dir):
    """§4.5 — escalade sur 30 gabarits, MÊME règle « ≥ 2/3 paraphrases »,
    unité = le secret. H vraie si n ≥ 12/30 ; H fausse si n ≤ 5/30."""
    raw = out_dir / "raw"
    z = np.load(raw / "esc_raw.npz")
    meta = json.loads((raw / "esc_meta.json").read_text(encoding="utf-8"))
    secrets = meta["secrets"]
    n_s = len(secrets)
    print("\n================ ESCALADE (30 gabarits) ================")
    per_secret, cases = [], []
    for si in range(n_s):
        st = Datastore(f"S{si}")
        st.add(z[f"S_keys_final_{si}"], z[f"S_values_{si}"])
        st.freeze()
        hits = 0
        for qi in (1, 2, 3):
            q = Query(z[f"S_q{qi}_logp_{si}"], z[f"S_q{qi}_final_{si}"],
                      z[f"S_q{qi}_inject_{si}"], z[f"S_q{qi}_target_{si}"],
                      z[f"S_q{qi}_nll_base_{si}"], z[f"S_q{qi}_H_{si}"], f"para{qi}")
            tgt = int(q.target[0])
            _, d2k, valsk, d2all = st.query(q.key("final", 0), K_NEIGHBORS)
            s = sup_over_c(q.logp[0], d2k, valsk, tgt, LAMBDA_STAR)
            lm = lm_prob_stats(q.logp[0], tgt)
            top = s["rank"] <= RANK_TOP
            hits += int(top)
            cases.append({"secret": secrets[si], "question": f"para{qi}",
                          "rank_base": lm["rank_base"], "rank_mix": s["rank"],
                          "top10": bool(top), "c_sup": s["c_sup"],
                          "R1v": rank_of_value(d2all, st.values, tgt),
                          "R3": s["p_knn_target"], "p10": lm["p10"]})
        per_secret.append({"secret": secrets[si], "hits": hits,
                           "h": hits / 3.0, "success": hits >= P1_PARAPHRASE_MIN})
    n = sum(1 for s in per_secret if s["success"])
    verdict = ("H vraie (n ≥ 12/30)" if n >= ESC_TRUE else
               "H fausse (n ≤ 5/30)" if n <= ESC_FALSE else
               "zone grise persistante [6, 11] ⇒ INCONCLUSIF")
    print(f"  n = {n}/{ESC_N} secrets avec ≥ 2/3 paraphrases en top-10")
    print(f"  seuils pré-enregistrés : H vraie ≥ {ESC_TRUE} (puissance "
          f"{binom_tail_ge(ESC_TRUE, ESC_N, 0.5):.3f} à p=0.5 ; risque I "
          f"{binom_tail_ge(ESC_TRUE, ESC_N, 0.1):.2e} sous p=0.1) | "
          f"H fausse ≤ {ESC_FALSE}")
    print(f"  ⇒ {verdict}")
    print(f"  h (moyenne des taux par secret) = "
          f"{float(np.mean([s['h'] for s in per_secret])):.4f}")
    payload = {"n_secrets": n, "N": ESC_N, "verdict": verdict,
               "per_secret": per_secret, "cases": cases,
               "thresholds": {"true": ESC_TRUE, "false": ESC_FALSE},
               "power_at_p0.5": binom_tail_ge(ESC_TRUE, ESC_N, 0.5),
               "type_I_at_p0.1": binom_tail_ge(ESC_TRUE, ESC_N, 0.1),
               "h_mean": float(np.mean([s["h"] for s in per_secret])),
               "gpu_meta": meta}
    (out_dir / "escalation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"  → {out_dir / 'escalation.json'}")
    return payload


def _analysis_phase(args, out_dir):
    raw = out_dir / "raw"
    z = np.load(raw / "gpu_raw.npz")
    meta = json.loads((raw / "gpu_meta.json").read_text(encoding="utf-8"))
    secrets = meta["secrets"]
    qnames = meta["questions"]
    n_s = len(secrets)
    n_q = len(qnames)
    gates_only = (args.stage == "gates")
    res: dict = {"command": f"python eval/knn_ceiling.py --phase analysis "
                            f"--stage {args.stage}",
                 "protocol": "experiments/EXP-2026-08-21-knn-borne-logits-v2.md",
                 "config": meta["config"], "device_gpu_pass": meta["device"],
                 "lambda_star": LAMBDA_STAR, "k": K_NEIGHBORS,
                 "lambda_grid": LAMBDA_GRID, "c_grid": C_GRID,
                 "p10_feasible_threshold": P10_FEASIBLE,
                 "mass_factor": MASS_FACTOR, "F_hard_bound": F_HARD_BOUND,
                 "gpu_meta": {k: v for k, v in meta.items()
                              if k not in ("lambda0_replay",)}}
    csv_rows = []
    t_start = time.time()

    # -------------------------------------------------- reconstruction des objets
    def make_store(prefix, si, arm):
        s = Datastore(f"{prefix}{si}-{arm}")
        s.add(z[f"{prefix}_keys_{arm}_{si}"], z[f"{prefix}_values_{si}"])
        return s.freeze()

    stores = {arm: [make_store("A", si, arm) for si in range(n_s)] for arm in ARMS}
    queries: dict[str, list[list[Query]]] = {}
    for prefix in ("A", "B"):
        if f"A_q0_logp_0" not in z.files:
            continue
        if prefix == "B" and "B_q0_logp_0" not in z.files:
            continue
        queries[prefix] = []
        for si in range(n_s):
            per_q = []
            for qi, qname in enumerate(qnames):
                pre = f"{prefix}_q{qi}"
                per_q.append(Query(z[f"{pre}_logp_{si}"], z[f"{pre}_final_{si}"],
                                   z[f"{pre}_inject_{si}"], z[f"{pre}_target_{si}"],
                                   z[f"{pre}_nll_base_{si}"], z[f"{pre}_H_{si}"],
                                   qname))
            queries[prefix].append(per_q)

    # entrée CORRECTE du store = celle dont la valeur est le 1er token de « secret »
    correct_idx = []
    for si in range(n_s):
        tgt0 = int(queries["A"][si][0].target[0])
        hit = np.nonzero(z[f"A_values_{si}"] == tgt0)[0]
        correct_idx.append(int(hit[0]) if len(hit) else -1)
    res["correct_entry_index"] = correct_idx
    assert all(i >= 0 for i in correct_idx), "entrée correcte introuvable dans le store"
    res["store_sizes"] = [int(len(z[f"A_values_{si}"])) for si in range(n_s)]

    D_nll = z["D_nll_base"].astype(np.float64)
    D_H = z["D_H"].astype(np.float64)
    D_tgt = z["D_target"]
    D_logp = z["D_logp"]
    D_key = {"final": z["D_final"], "inject": z["D_inject"]}
    n_pos = len(D_tgt)

    # d² NEUTRAL_TEXT × store fait-seul : indépendants de λ et de c ⇒ une seule fois
    t_d2 = time.time()
    d2_neutral = {si: d2_matrix(D_key["final"], z[f"A_keys_final_{si}"])
                  for si in range(n_s)}
    nbr_neutral = {si: neighbors_from_matrix(d2_neutral[si], z[f"A_values_{si}"])
                   for si in range(n_s)}
    d2min_neutral = {si: d2_neutral[si].min(axis=1).astype(np.float64)
                     for si in range(n_s)}
    print(f"\n  d² NEUTRAL_TEXT × store fait-seul : {n_pos}×"
          f"{d2_neutral[0].shape[1]} × {n_s} secrets en {time.time() - t_d2:.1f}s")

    # p_LM par position sur NEUTRAL_TEXT : p_max, p₁₀ (§4.2)
    D_stats = [lm_prob_stats(D_logp[t], int(D_tgt[t])) for t in range(n_pos)]
    D_logp_target = np.asarray([s["logp_target"] for s in D_stats], dtype=np.float64)
    D_p10 = np.asarray([s["p10"] for s in D_stats], dtype=np.float64)
    D_pmax = np.asarray([s["p_max"] for s in D_stats], dtype=np.float64)

    def e3_vector(si, lam, c, tau=None, nbr=None, d2min=None):
        """ΔNLL par position sur NEUTRAL_TEXT, vectorisé. `tau` arme le gate G."""
        idx, d2k, valsk = nbr if nbr is not None else nbr_neutral[si]
        dmin = d2min if d2min is not None else d2min_neutral[si]
        mass, _ = target_mass_batch(d2k, valsk, D_tgt, c)
        lam_eff = np.full(n_pos, float(lam))
        if tau is not None:
            lam_eff = np.where(dmin <= tau, float(lam), 0.0)
        out = np.empty(n_pos, dtype=np.float64)
        uniq = np.unique(lam_eff)
        for lv in uniq:
            m_ = lam_eff == lv
            out[m_] = mix_delta_nll_vec(D_logp_target[m_], mass[m_], float(lv))
        return out, mass

    # =====================================================================
    #  PORTES, DANS L'ORDRE — arrêt dur à la première échouée (§13.10)
    # =====================================================================

    # ------------------------------------------------------------- V-tie
    print("\n================ PORTE V-tie (ex-æquo de d²_min) ================")
    ties_recall, ties_neutral = [], []
    for si in range(n_s):
        for qi in range(n_q):
            q = queries["A"][si][qi]
            d2 = squared_distances(q.key("final", 0), stores["final"][si].keys)
            ties_recall.append(count_argmin_ties(d2))
    for si in range(n_s):
        mn = d2_neutral[si].min(axis=1, keepdims=True)
        ties_neutral.extend((d2_neutral[si] == mn).sum(axis=1).tolist())
    v_tie = {"recall_queries": {"n": len(ties_recall), "max": int(max(ties_recall)),
                                "n_gt1": int(sum(1 for t in ties_recall if t > 1)),
                                "counts": ties_recall},
             "neutral_positions": {"n": len(ties_neutral),
                                   "max": int(max(ties_neutral)),
                                   "n_gt1": int(sum(1 for t in ties_neutral if t > 1))},
             "rule": "si > 0 ex-æquo : un-hot appliqué UNIFORME sur l'argmin-set "
                     "(§4.1, implémenté dans knn_weights)",
             "expected": 0}
    print(f"  requêtes de rappel ({len(ties_recall)}) : ex-æquo > 1 sur "
          f"{v_tie['recall_queries']['n_gt1']} cas, max = {v_tie['recall_queries']['max']}")
    print(f"  positions NEUTRAL_TEXT ({len(ties_neutral)}) : ex-æquo > 1 sur "
          f"{v_tie['neutral_positions']['n_gt1']} cas, max = "
          f"{v_tie['neutral_positions']['max']}")
    print("  V-tie → RAPPORTÉ (porte descriptive : aucun seuil, l'un-hot est "
          "uniforme sur l'argmin-set par construction)")
    res["V_tie"] = v_tie

    # -------------------------------------------------- V1a / V1b / V1c / V-var
    print("\n================ PORTES V1a / V1b / V1c / V-var ================")
    v1a, v1b, v1c, vvar = {}, {}, {}, {}
    for lam in LAMBDA_GRID:
        edge = -math.log1p(-lam)
        for c in C_GRID:
            cell = f"lam={lam:.6f}|c={c}"
            devs, viol, n_zero, n_pos_mass, rec_devs, var_zero = [], 0, 0, 0, [], []
            n_above, n_ulp_tie = 0, 0
            for si in range(n_s):
                D, mass = e3_vector(si, lam, c)
                zero = mass <= 0.0
                n_zero += int(zero.sum())
                n_pos_mass += int((~zero).sum())
                if zero.any():
                    devs.append(float(np.max(np.abs(D[zero] - edge))))
                    var_zero.append(float(np.var(D[zero])))
                if (~zero).any():
                    # V1b : décroissance STRICTE en r. Le décrément exact est
                    # δ_t = log1p(λr/(1−λ)) > 0 dès que p_kNN(y_t) > 0 ; c'est
                    # LUI le prédicat testable en flottant. `ΔNLL_t == bord` peut
                    # survenir sans violation algébrique quand δ_t < ulp(bord)
                    # (r ~ 1e-20) : ces cas sont comptés SÉPARÉMENT et publiés.
                    delta = mix_decrement_vec(D_logp_target, mass, lam)[~zero]
                    viol += int((delta <= 0.0).sum())
                    n_above += int((D[~zero] > edge).sum())
                    n_ulp_tie += int((D[~zero] == edge).sum())
                # V1c : E3 mesuré vs RECOMPOSÉ (identité de partition)
                f_relief = float((~zero).mean())
                recomposed = (1.0 - f_relief) * edge + (
                    float(D[~zero].sum()) / n_pos if (~zero).any() else 0.0)
                rec_devs.append(abs(float(D.mean()) - recomposed))
            v1a[cell] = {"max_abs_dev": float(max(devs)) if devs else 0.0,
                         "n_positions_pknn_zero": n_zero, "tol": V1A_TOL}
            v1b[cell] = {"n_violations": viol, "n_positions_pknn_pos": n_pos_mass,
                         "n_delta_nll_strictly_above_edge": n_above,
                         "n_delta_nll_equal_edge_at_ulp_floor": n_ulp_tie}
            v1c[cell] = {"max_abs_dev_nats": float(max(rec_devs)), "tol": V1C_TOL}
            vvar[cell] = {"max_var_D_on_pknn_zero": float(max(var_zero)) if var_zero else 0.0,
                          "tol": VVAR_TOL}
    v1a_max = max(v["max_abs_dev"] for v in v1a.values())
    v1b_viol = sum(v["n_violations"] for v in v1b.values())
    v1b_above = sum(v["n_delta_nll_strictly_above_edge"] for v in v1b.values())
    v1b_tie = sum(v["n_delta_nll_equal_edge_at_ulp_floor"] for v in v1b.values())
    v1c_max = max(v["max_abs_dev_nats"] for v in v1c.values())
    vvar_max = max(v["max_var_D_on_pknn_zero"] for v in vvar.values())
    print(f"  V1a  sur p_kNN(y_t) = 0 : max_t |ΔNLL_t + log(1−λ)| = {v1a_max:.3e} "
          f"(seuil ≤ {V1A_TOL:.0e}, sur {len(v1a)} cellules λ×c)")
    print(f"  V1b  sur p_kNN(y_t) > 0 : violations de ΔNLL_t < −log(1−λ) = "
          f"{v1b_viol} (exigé : 0, strict)")
    print(f"       décomposition : ΔNLL_t STRICTEMENT au-dessus du bord = {v1b_above} "
          f"(violation algébrique) | ΔNLL_t == bord au PLANCHER ULP (δ_t < ulp(bord), "
          f"r ~ 1e-20) = {v1b_tie} — publié, sans violation algébrique")
    print(f"  V1c  E3 mesuré vs recomposé : écart max = {v1c_max:.3e} nats "
          f"(seuil ≤ {V1C_TOL:.0e})")
    print(f"  V-var var(D_t) sur p_kNN = 0 : max = {vvar_max:.3e} "
          f"(seuil = 0 à {VVAR_TOL:.0e})")
    res["V1a"] = {"max_over_cells": v1a_max, "tol": V1A_TOL, "cells": v1a,
                  "ok": bool(v1a_max <= V1A_TOL)}
    res["V1b"] = {"n_violations_total": v1b_viol,
                  "n_delta_nll_strictly_above_edge_total": v1b_above,
                  "n_delta_nll_equal_edge_at_ulp_floor_total": v1b_tie,
                  "criterion": "décrément δ_t = log1p(λr/(1−λ)) > 0 sur 100 % des "
                               "positions à p_kNN(y_t) > 0 ; le compte de ΔNLL_t "
                               "égaux au bord au plancher ULP est publié à part",
                  "cells": v1b, "ok": bool(v1b_viol == 0)}
    res["V1c"] = {"max_over_cells": v1c_max, "tol": V1C_TOL, "cells": v1c,
                  "ok": bool(v1c_max <= V1C_TOL)}
    res["V_var"] = {"max_over_cells": vvar_max, "tol": VVAR_TOL, "cells": vvar,
                    "ok": bool(vvar_max <= VVAR_TOL)}
    for name, ok, detail in (("V1a", v1a_max <= V1A_TOL,
                              {"measured": v1a_max, "tol": V1A_TOL,
                               "derivation": "ΔNLL_t = −log(1−λ) exactement quand "
                                             "p_kNN(y_t) = 0 — identité algébrique"}),
                             ("V1b", v1b_viol == 0,
                              {"violations": v1b_viol,
                               "derivation": "ΔNLL strictement décroissant en "
                                             "p_kNN(y_t) ⇒ < −log(1−λ) dès que la "
                                             "masse est > 0"}),
                             ("V1c", v1c_max <= V1C_TOL,
                              {"measured": v1c_max, "tol": V1C_TOL,
                               "derivation": "E3 = (1−f)·(−log(1−λ)) + moyenne du "
                                             "soulagement — recomposition exacte"}),
                             ("V-var", vvar_max <= VVAR_TOL,
                              {"measured": vvar_max, "tol": VVAR_TOL,
                               "derivation": "D_t constant sur p_kNN = 0 ⇒ var = 0 "
                                             "(remplace la porte de corrélation, NaN)"})):
        print(f"  {name} → {'OK' if ok else 'ÉCHEC'}")
        if not ok:
            _fail(out_dir, name, detail, res)

    # ------------------------------------------------------------------- V2
    print("\n================ PORTE V2 (faisabilité arithmétique) ================")
    feas_rows, n_feasible = [], 0
    for si in range(n_s):
        per = []
        for qi in range(1, n_q):
            q = queries["A"][si][qi]
            lm = lm_prob_stats(q.logp[0], int(q.target[0]))
            per.append({"question": qnames[qi], "p10": lm["p10"],
                        "p_max": lm["p_max"], "feasible": bool(lm["p10"] < P10_FEASIBLE),
                        "H": float(q.H[0]), "logp_base": lm["logp_target"]})
        ok = sum(1 for p in per if p["feasible"]) >= P1_PARAPHRASE_MIN
        n_feasible += int(ok)
        feas_rows.append({"secret": secrets[si], "feasible": ok, "detail": per})
    print(f"  seuil : p₁₀ < λ*/(1−λ*) = e^0.05 − 1 = {P10_FEASIBLE:.7f}")
    print(f"  n_faisable (≥ 2/3 paraphrases sous le seuil) = {n_feasible}/{n_s}  "
          f"(Math : ≥ 7 ; Neuro : [7,10] ; ≤ 5 ⇒ INCONCLUSIF budget arithmétique)")
    for r in feas_rows:
        print(f"    {r['secret']:<12} faisable={str(r['feasible']):<5} "
              + " ".join(f"{d['question']}:p₁₀={d['p10']:.5f}"
                         f"{'✓' if d['feasible'] else '✗'}" for d in r["detail"]))
    res["V2"] = {"n_feasible": n_feasible, "N": n_s, "threshold": P10_FEASIBLE,
                 "per_secret": feas_rows, "ok": bool(n_feasible > 5)}
    csv_rows.append(["V2", "faisabilite", "n_feasible", n_feasible, "", n_s])
    if n_feasible <= 5:
        _fail(out_dir, "V2",
              {"n_feasible": n_feasible, "threshold_secrets": V2_MIN_FEASIBLE,
               "p10_threshold": P10_FEASIBLE,
               "verdict": "INCONCLUSIF — budget arithmétique",
               "derivation": "top-10 exige λ·p_kNN > (1−λ)·p₁₀ ; à p_kNN ≤ 1 le "
                             "canal ne peut rien faire si p₁₀ ≥ λ/(1−λ)"}, res)

    # =====================================================================
    #  P1 — cellule décisionnelle `sup_c` (§4.1)
    # =====================================================================
    print("\n================ P1 (DÉCISIONNELLE — sup_c, λ*, ITT sur les 10) ================")

    def eval_case(store, q, arm, lam, p=0, values=None, tau=None, correct=None,
                  null=None):
        """Recalcul hors-ligne complet d'une requête sous λ, au **sup_c**."""
        _, d2k, valsk, d2all = store.query(q.key(arm, p), K_NEIGHBORS, values=values)
        tgt = int(q.target[p])
        lm = lm_prob_stats(q.logp[p], tgt)
        d2min = float(d2all.min())
        lam_eff = lam if (tau is None or d2min <= tau) else 0.0
        s = sup_over_c(q.logp[p], d2k, valsk, tgt, lam_eff)
        vals_eff = store.values if values is None else np.asarray(values)
        out = {"target": tgt, "rank_base": lm["rank_base"], "rank_mix": s["rank"],
               "top10": bool(s["rank"] <= RANK_TOP), "c_sup": s["c_sup"],
               "T_q": s["T_q"], "R3": s["p_knn_target"], "R4": s["H_knn"],
               "p10": lm["p10"], "p_max": lm["p_max"],
               "logp_base": lm["logp_target"], "H": float(q.H[p]),
               "d2_min": d2min, "ties": count_argmin_ties(d2all),
               "R1v": rank_of_value(d2all, vals_eff, tgt),
               "lam_eff": lam_eff,
               "logp_mix": mix_logprob(lm["logp_target"], s["p_knn_target"], lam_eff),
               "per_c": s["per_c"]}
        out["delta_logp"] = out["logp_mix"] - lm["logp_target"]
        if correct is not None and correct >= 0:
            out["R1"] = rank_of_index(d2all, correct)
            ck = store.keys[correct].astype(np.float64)
            kq = np.asarray(q.key(arm, p), dtype=np.float64)
            cos = float(ck @ kq / (np.linalg.norm(ck) * np.linalg.norm(kq) + 1e-12))
            out["R2_cos_correct"] = cos
            out["R2_d2_correct"] = float(d2all[correct])
            if null is not None:
                out["R2z"] = (cos - null["mu"]) / null["sigma"] if null["sigma"] > 0 \
                    else math.nan
        return out

    def p1_table(store_list, arm, lam, prefix="A", tau=None, values_fn=None,
                 cross=False, idx_map=None):
        """(n secrets succès, les 30 cas, détail par secret) sous `sup_c`."""
        per_secret, cases = [], []
        for si in range(n_s):
            sj = (si + 1) % n_s if cross else si
            st = store_list[sj]
            vals = values_fn(sj) if values_fn else None
            ci = (idx_map[sj] if idx_map else correct_idx[sj]) if not cross else -1
            hits, det = 0, []
            for qi in range(1, n_q):          # les 3 PARAPHRASES
                q = queries[prefix][si][qi]
                e = eval_case(st, q, arm, lam, values=vals, tau=tau, correct=ci)
                e.update({"secret": secrets[si], "question": qnames[qi]})
                e.pop("per_c", None)
                hits += int(e["top10"])
                det.append(e)
                cases.append(e)
            per_secret.append({"secret": secrets[si], "hits": hits,
                               "h": hits / (n_q - 1),
                               "success": hits >= P1_PARAPHRASE_MIN, "detail": det})
        n = sum(1 for s in per_secret if s["success"])
        return n, cases, per_secret

    n1, cases1, per1 = p1_table(stores["final"], "final", LAMBDA_STAR)
    h_values = [s["h"] for s in per1]
    h_boot = bootstrap_median_log(h_values)
    p1 = {"n_secrets": n1, "N": n_s, "lambda": LAMBDA_STAR, "cell": "sup_c",
          "n_cases_top10": sum(c["top10"] for c in cases1), "n_cases": len(cases1),
          "sign_test_p": sign_test_exact(n1, n_s),
          "per_secret": [{k: v for k, v in s.items() if k != "detail"} for s in per1],
          "cases": cases1}
    print(f"  n (secrets avec ≥ 2/3 paraphrases en top-10) = {n1}/{n_s}   "
          f"(H vraie ≥ 5 ; H fausse ≤ 1 ; zone grise 2–4 ⇒ escalade automatique)")
    print(f"  test des signes exact : p = {p1['sign_test_p']:.4g}")
    print(f"  les 30 cas descriptifs : {p1['n_cases_top10']}/{p1['n_cases']} en top-10")
    print("\n  les 30 cas (bras final, λ*, sup_c) :")
    print("    secret       question  rang_base → rang_mix  top10  c_sup   R1  R1v  "
          "R3        R4      p₁₀       p_max     d²_min   Δlogp")
    for cc in cases1:
        print(f"    {cc['secret']:<12} {cc['question']:<8} {cc['rank_base']:>9} → "
              f"{cc['rank_mix']:<8} {str(cc['top10']):<6} {cc['c_sup']:<6} "
              f"{cc.get('R1', -1):>3} {cc['R1v']:>4}  {cc['R3']:.6f}  "
              f"{cc['R4']:.4f}  {cc['p10']:.6f}  {cc['p_max']:.6f}  "
              f"{cc['d2_min']:8.2f} {cc['delta_logp']:+.4f}")
    print(f"\n  h (CO-PRIMAIRE — taux de récupération par paraphrase, par secret) :")
    print("    " + "  ".join(f"{s['secret']}={s['h']:.3f}" for s in per1))
    print(f"  médiane h = {h_boot['median']:.4f} | IC95 bootstrap EN LOG "
          f"[{h_boot['ci_lo']:.4f}, {h_boot['ci_hi']:.4f}] — {h_boot['note']}")
    print(f"  moyenne h = {float(np.mean(h_values)):.4f} | seuils : H vraie ≥ 0.50 ; "
          f"H fausse ≤ 0.20 (dérivés de P1 par q = r²(3−2r))")
    res["P1"] = p1
    res["h"] = {"values_per_secret": h_values, "secrets": secrets,
                "median": h_boot["median"], "mean": float(np.mean(h_values)),
                "min": float(np.min(h_values)), "max": float(np.max(h_values)),
                "bootstrap_log": h_boot,
                "thresholds": {"H_true": 0.50, "H_false": 0.20},
                "derivation": "q = r²(3−2r) : r=0.50 ⇒ q=0.500 (n≥5/10) ; "
                              "r=0.20 ⇒ q=0.104 (n≤1/10)"}
    csv_rows.append(["P1", "final|sup_c|lam*", "n_secrets_top10", n1, "", n_s])
    csv_rows.append(["h", "final|sup_c|lam*", "h_mean", f"{np.mean(h_values):.6f}",
                     f"{mean_sd(h_values)[1]:.6f}", n_s])

    # ---- ANTIPODE + clause « cellule dégénérée » (§4.3) ----
    succ = [cc for cc in cases1 if cc["top10"]]
    fail = [cc for cc in cases1 if not cc["top10"]]
    edge_c = {C_GRID[0], C_GRID[-1]}
    degenerate = bool(fail and
                      sum(1 for cc in fail if cc["R1v"] > 1) > len(fail) / 2 and
                      sum(1 for cc in fail if cc["c_sup"] in edge_c) > len(fail) / 2)
    res["P1_antipode"] = {
        "n_success_cases": len(succ),
        "n_success_with_R1_gt_1": sum(1 for cc in succ if cc.get("R1", 1) > 1),
        "n_success_with_R1v_gt_1": sum(1 for cc in succ if cc["R1v"] > 1),
        "majority_success_R1v_gt_1": bool(succ and sum(
            1 for cc in succ if cc["R1v"] > 1) > len(succ) / 2),
        "n_fail_with_R1v_gt_1": sum(1 for cc in fail if cc["R1v"] > 1),
        "n_fail_c_sup_at_grid_edge": sum(1 for cc in fail if cc["c_sup"] in edge_c),
        "degenerate_cell": degenerate}
    print(f"  ANTIPODE : succès avec R1v > 1 : "
          f"{res['P1_antipode']['n_success_with_R1v_gt_1']}/{len(succ)} | "
          f"échecs avec R1v > 1 : {res['P1_antipode']['n_fail_with_R1v_gt_1']}/{len(fail)} "
          f"| c_sup au bord de grille sur les échecs : "
          f"{res['P1_antipode']['n_fail_c_sup_at_grid_edge']}/{len(fail)}")
    print(f"  clause « cellule dégénérée » : "
          f"{'DÉCLENCHÉE ⇒ INCONCLUSIF' if degenerate else 'non déclenchée'}")

    # ---- distribution du c-du-sup ----
    c_hist = {str(c): sum(1 for cc in cases1 if cc["c_sup"] == c) for c in C_GRID}
    res["c_sup_histogram_paraphrases"] = c_hist
    print(f"  distribution du c-du-sup (30 cas) : "
          + " ".join(f"c={k}:{v}" for k, v in c_hist.items()))

    if gates_only:
        res["duration_analysis_s"] = time.time() - t_start
        _dump(out_dir, res, csv_rows)
        print(f"\n  [stage=gates] portes franchies jusqu'à P1 incluse. "
              f"{res['duration_analysis_s']:.1f}s → {out_dir}")
        return res

    # =====================================================================
    #  Le reste — descriptif, lu APRÈS le verdict
    # =====================================================================

    # -------------------------------------------------------- P1-exact (ancrage)
    print("\n================ P1-exact (ancrage, question exacte) ================")
    ex_cases = []
    for si in range(n_s):
        e = eval_case(stores["final"][si], queries["A"][si][0], "final", LAMBDA_STAR,
                      correct=correct_idx[si])
        e.update({"secret": secrets[si], "question": "exact"})
        e.pop("per_c", None)
        ex_cases.append(e)
    n_exact = sum(1 for e in ex_cases if e["top10"])
    print(f"  top-10 sur indice exact : {n_exact}/{n_s} (attendu 10/10 si H vraie ; "
          f"< 10/10 avec p₁₀ conforme = bug)")
    for e in ex_cases:
        print(f"    {e['secret']:<12} rang {e['rank_base']:>7} → {e['rank_mix']:<6} "
              f"top10={str(e['top10']):<5} c_sup={e['c_sup']:<5} R1={e.get('R1')} "
              f"R1v={e['R1v']} R3={e['R3']:.6f} p₁₀={e['p10']:.6f} "
              f"d²_min={e['d2_min']:.6f}")
    res["P1_exact"] = {"n_top10": n_exact, "N": n_s, "cases": ex_cases}

    # ------------------------------------------------------------------ F₁₀
    print("\n================ F₁₀ (DIAGNOSTIC, non décisionnel — triplet) ================")
    f10_cases = []
    for cc in cases1:
        finite = cc["R3"] > 0.0
        f = (INV_LAMBDA_STAR * cc["p10"] / (1.0 + cc["p10"])) if finite else math.inf
        f10_cases.append({"secret": cc["secret"], "question": cc["question"],
                          "finite": bool(finite), "F10": f, "p10": cc["p10"]})
    per_secret_f10 = []
    for si, s in enumerate(secrets):
        vals = [c["F10"] for c in f10_cases if c["secret"] == s and c["finite"]]
        per_secret_f10.append(float(np.exp(np.mean(np.log(vals)))) if vals else math.nan)
    n_fin = sum(1 for c in f10_cases if c["finite"])
    fin_vals = [c["F10"] for c in f10_cases if c["finite"]]
    f10 = {"success_rate_cases": n_fin / len(f10_cases),
           "n_finite_cases": n_fin, "n_infinite_cases": len(f10_cases) - n_fin,
           "per_secret_geomean": per_secret_f10,
           "log_F10_per_secret": [math.log(v) if v == v and v > 0 else None
                                  for v in per_secret_f10],
           "min_case": float(min(fin_vals)) if fin_vals else math.nan,
           "max_case": float(max(fin_vals)) if fin_vals else math.nan,
           "bootstrap_log": bootstrap_median_log(
               [v for v in per_secret_f10 if v == v]),
           "n_gt_1_cases": sum(1 for v in fin_vals if v > 1.0),
           "prediction": "[0.10, 0.62] < 1 ⇒ le budget n'est pas la contrainte mordante",
           "note": "bimodale (+∞ sur échec) et CONSTANTE entre bras ⇒ jamais une "
                   "médiane seule"}
    print(f"  (1) taux de succès (cas à p_kNN(cible) > 0 au sup_c) : "
          f"{n_fin}/{len(f10_cases)} = {f10['success_rate_cases']:.3f}")
    print(f"  (2) F₁₀ conditionnelle aux succès, 10 valeurs par secret "
          f"(moyenne géométrique) :")
    print("      " + "  ".join(f"{s}={v:.4f}" if v == v else f"{s}=n/a"
                               for s, v in zip(secrets, per_secret_f10)))
    print(f"      min/max sur les cas : {f10['min_case']:.4f} / {f10['max_case']:.4f} "
          f"| F₁₀ > 1 sur {f10['n_gt_1_cases']}/{n_fin} cas (ANTIPODE si majorité)")
    print(f"  (3) nombre d'échecs (F₁₀ = +∞) : {f10['n_infinite_cases']}")
    res["F10"] = f10

    # ------------------------------------------------------------------- P2
    print("\n================ P2 (E1c — SANS fourchette ; borne dure F ≤ 10.252) ================")
    p2 = None
    if "C_keys_final" in z.files:
        store_c = {}
        for arm in ARMS:
            s = Datastore(f"C-{arm}")
            s.add(z[f"C_keys_{arm}"], z["C_values"])
            store_c[arm] = s.freeze()
        p2 = {"label": "au un-hot avec récupération correcte, P2 ne dépend que de "
                       "p_LM — mesure du prior de GPT-2, PAS du canal ; légitime "
                       "comme chiffre de rachat, nulle comme information mécaniste",
              "F_hard_bound": F_HARD_BOUND, "conditions": {}}
        grid = np.unique(np.round(np.concatenate(
            [np.linspace(0.0, 0.999, 1000), np.asarray(LAMBDA_GRID)]), 6)).tolist()
        for tag in ("m0", "mload"):
            for arm in ARMS:
                qm = Query(z[f"C_{tag}_w0_logp"], z[f"C_{tag}_w0_final"],
                           z[f"C_{tag}_w0_inject"], z[f"C_{tag}_w0_target"],
                           z[f"C_{tag}_w0_nll_base"], z[f"C_{tag}_w0_H"], "Marseille")
                qp = Query(z[f"C_{tag}_w1_logp"], z[f"C_{tag}_w1_final"],
                           z[f"C_{tag}_w1_inject"], z[f"C_{tag}_w1_target"],
                           z[f"C_{tag}_w1_nll_base"], z[f"C_{tag}_w1_H"], "Paris")
                lam_renv = None
                for lam in grid:
                    if eval_case(store_c[arm], qm, arm, lam)["rank_mix"] == 1:
                        lam_renv = lam
                        break
                at = eval_case(store_c[arm], qm, arm, LAMBDA_STAR)
                atp = eval_case(store_c[arm], qp, arm, LAMBDA_STAR)
                F = (lam_renv / LAMBDA_STAR) if lam_renv is not None else None
                p2["conditions"][f"{tag}|{arm}"] = {
                    "lambda_renv": lam_renv, "F": F,
                    "F_within_hard_bound": (F is None or F <= F_HARD_BOUND),
                    "rank_marseille_base": at["rank_base"],
                    "rank_marseille_at_lambda_star": at["rank_mix"],
                    "rank_paris_at_lambda_star": atp["rank_mix"],
                    "delta_marseille": at["delta_logp"], "delta_paris": atp["delta_logp"],
                    "R3_marseille": at["R3"], "c_sup": at["c_sup"],
                    "d2_min": at["d2_min"], "p10": at["p10"], "p_max": at["p_max"],
                    "rank1_at_or_below_lambda_star": bool(
                        lam_renv is not None and lam_renv <= LAMBDA_STAR)}
                e = p2["conditions"][f"{tag}|{arm}"]
                print(f"  [{tag} / bras {arm}] λ_renv = "
                      f"{('%.6f' % lam_renv) if lam_renv is not None else 'jamais (≤0.999)'}"
                      f" | F = {('%.3f' % F) if F is not None else 'n/a'} "
                      f"(borne dure ≤ {F_HARD_BOUND:.3f} ; ancre de régime 6.77)")
                print(f"      rang Marseille : base {at['rank_base']} → "
                      f"{at['rank_mix']} à λ* | Paris à λ* : {atp['rank_mix']} | "
                      f"ΔMars {at['delta_logp']:+.4f} ΔParis {atp['delta_logp']:+.4f}")
        print(f"  ÉTIQUETTE OBLIGATOIRE : {p2['label']}")
    else:
        print("  !! passe C absente — P2 non exécuté")
    res["P2"] = p2

    # ------------------------------------------------------------------- P3
    print("\n================ P3 (BLOQUANT — valeurs permutées, au MÊME sup_c) ================")
    perm_vals = {si: permute_values(z[f"A_values_{si}"], PERM_SEED + si)
                 for si in range(n_s)}
    q0 = queries["A"][0][0]
    _, d2a, _, _ = stores["final"][0].query(q0.key("final", 0), K_NEIGHBORS)
    _, d2b, _, _ = stores["final"][0].query(q0.key("final", 0), K_NEIGHBORS,
                                            values=perm_vals[0])
    dist_identical = bool(np.array_equal(d2a, d2b))
    multiset_ok = all(sorted(perm_vals[si].tolist())
                      == sorted(z[f"A_values_{si}"].tolist()) for si in range(n_s))
    n3, cases3, per3 = p1_table(stores["final"], "final", LAMBDA_STAR,
                                values_fn=lambda si: perm_vals[si])
    # identité exacte de la baisse, sur la condition EXACTE de Math Q2 :
    # `valeur(argmin) ≠ cible` (plus large que « cible ∉ k valeurs », qui reste
    # un SOUS-ENSEMBLE CONSERVATEUR VALIDE — les deux comptes sont loggés).
    exp_drop = math.log1p(-LAMBDA_STAR)
    strict_dev, cons_dev, n_strict, n_cons = [], [], 0, 0
    for si in range(n_s):
        for qi in range(n_q):
            q = queries["A"][si][qi]
            _, d2k, valsk, d2all = stores["final"][si].query(
                q.key("final", 0), K_NEIGHBORS, values=perm_vals[si])
            tgt = int(q.target[0])
            s = sup_over_c(q.logp[0], d2k, valsk, tgt, LAMBDA_STAR)
            lm = lm_prob_stats(q.logp[0], tgt)
            drop = mix_logprob(lm["logp_target"], s["p_knn_target"],
                               LAMBDA_STAR) - lm["logp_target"]
            argmin_val = int(valsk[0])
            if argmin_val != tgt:
                n_strict += 1
                if s["c_sup"] == UNHOT:
                    strict_dev.append(abs(drop - exp_drop))
            if tgt not in set(int(v) for v in valsk):
                n_cons += 1
                cons_dev.append(abs(drop - exp_drop))
    p3 = {"distances_identical_under_permutation": dist_identical,
          "multiset_preserved": multiset_ok,
          "n_secrets_top10": n3, "n_cases_top10": sum(c["top10"] for c in cases3),
          "expected_drop_log1m_lambda": exp_drop,
          "exact_condition_valeur_argmin_ne_cible": {
              "n_cases": n_strict, "n_checked_unhot": len(strict_dev),
              "max_abs_dev": float(max(strict_dev)) if strict_dev else 0.0},
          "conservative_subset_cible_absente_des_k_valeurs": {
              "n_cases": n_cons,
              "max_abs_dev": float(max(cons_dev)) if cons_dev else 0.0},
          "blocking_ok": bool(dist_identical and multiset_ok and n3 <= 1)}
    print(f"  distances inchangées sous permutation : {dist_identical} | "
          f"multiensemble conservé : {multiset_ok}")
    print(f"  top-10 sous permutation, au MÊME sup_c : {n3}/{n_s} secrets, "
          f"{p3['n_cases_top10']}/30 cas   (H vraie ≤ 1/10 ; H fausse ≥ 5/10)")
    print(f"  identité de la baisse = log(1−λ*) = {exp_drop:.6f} :")
    print(f"    condition EXACTE « valeur(argmin) ≠ cible » : {n_strict}/40 cas, "
          f"écart max (sous-cellule un-hot, {len(strict_dev)} cas) = "
          f"{p3['exact_condition_valeur_argmin_ne_cible']['max_abs_dev']:.3e}")
    print(f"    sous-ensemble CONSERVATEUR « cible ∉ k valeurs » : {n_cons}/40 cas, "
          f"écart max = "
          f"{p3['conservative_subset_cible_absente_des_k_valeurs']['max_abs_dev']:.3e}")
    print(f"  P3 → {'OK' if p3['blocking_ok'] else 'ÉCHEC ⇒ INCONCLUSIF'}")
    res["P3"] = p3

    # ------------------------------------------------------------------- P4
    print("\n================ P4 (spécificité : store A / prompt B) ================")
    n4, cases4, _ = p1_table(stores["final"], "final", LAMBDA_STAR, cross=True)
    ex4 = 0
    for si in range(n_s):
        e = eval_case(stores["final"][(si + 1) % n_s], queries["A"][si][0], "final",
                      LAMBDA_STAR)
        ex4 += int(e["top10"])
    res["P4"] = {"n_secrets_top10_paraphrases": n4,
                 "n_cases_top10": sum(c["top10"] for c in cases4),
                 "n_exact_top10": ex4, "N": n_s,
                 "criterion": "H vraie ≤ 1/10 ; H fausse ≥ 5/10"}
    print(f"  paraphrases : {n4}/{n_s} secrets, {res['P4']['n_cases_top10']}/30 cas | "
          f"question exacte : {ex4}/{n_s}   (critère : ≤ 1/10)")

    # ------------------------------------- stores distracteurs (P5, P7, R2z)
    dist_store, offset, d2_neutral_dist, nbr_dist = None, 0, None, None
    null_stats = {}
    if "E_keys_final" in z.files:
        offset = int(len(z["E_values"]))
        dist_store = {}
        for si in range(n_s):
            s = Datastore(f"distractor+fact{si}")
            s.add(z["E_keys_final"], z["E_values"])
            s.add(z[f"A_keys_final_{si}"], z[f"A_values_{si}"])
            dist_store[si] = s.freeze()
        print(f"\n  store distracteur : {offset} entrées + "
              f"{len(z['A_values_0'])} du fait")
        t_d2 = time.time()
        d2_neutral_dist = d2_matrix(D_key["final"], z["E_keys_final"])
        print(f"  d² NEUTRAL_TEXT × distracteur : {time.time() - t_d2:.1f}s")

        # ---- nulle empirique d'anisotropie pour R2z (§5.10) ----
        EK = z["E_keys_final"].astype(np.float32)
        EKn = EK / (np.linalg.norm(EK, axis=1, keepdims=True) + 1e-12)

        def null_of(key):
            k = np.asarray(key, dtype=np.float32)
            k = k / (np.linalg.norm(k) + 1e-12)
            cs = EKn @ k
            return {"mu": float(cs.mean()), "sigma": float(cs.std()),
                    "n": int(cs.size)}
    else:
        print("\n  !! store distracteur absent : P7, R2z et le bras distracteur "
              "de P5 non exécutés")

        def null_of(key):
            return None

    # ------------------------------------------------- R1 / R1v / R2z / R3 / R4
    print("\n================ INSTRUMENTATION R1 / R1v / R2z / R3 / R4 ================")
    instr = {"exact": [], "paraphrase": []}
    for si in range(n_s):
        for qi in range(n_q):
            q = queries["A"][si][qi]
            nl = null_of(q.key("final", 0))
            e = eval_case(stores["final"][si], q, "final", LAMBDA_STAR,
                          correct=correct_idx[si], null=nl)
            e.update({"secret": secrets[si], "question": qnames[qi]})
            e.pop("per_c", None)
            if nl:
                null_stats[f"{secrets[si]}|{qnames[qi]}"] = nl
            instr["exact" if qi == 0 else "paraphrase"].append(e)
    for cond in ("exact", "paraphrase"):
        rows = instr[cond]
        agg = {"R1_mean": float(np.mean([r["R1"] for r in rows])),
               "R1_eq1_frac": float(np.mean([r["R1"] == 1 for r in rows])),
               "R1v_mean": float(np.mean([r["R1v"] for r in rows])),
               "R1v_eq1_frac": float(np.mean([r["R1v"] == 1 for r in rows])),
               "R1v_le_k_frac": float(np.mean([0 < r["R1v"] <= K_NEIGHBORS
                                               for r in rows])),
               "R2_cos_mean": float(np.mean([r["R2_cos_correct"] for r in rows])),
               "R2z_mean": (float(np.mean([r["R2z"] for r in rows]))
                            if "R2z" in rows[0] else math.nan),
               "R2z_median": (float(np.median([r["R2z"] for r in rows]))
                              if "R2z" in rows[0] else math.nan),
               "R2_d2_min_mean": float(np.mean([r["d2_min"] for r in rows])),
               "R3_mean": float(np.mean([r["R3"] for r in rows])),
               "R4_mean": float(np.mean([r["R4"] for r in rows])),
               "n": len(rows)}
        res.setdefault("instrumentation", {})[cond] = {"aggregate": agg, "rows": rows}
        print(f"  [{cond:<10}] N={agg['n']:>2} | R1 {agg['R1_mean']:6.2f} "
              f"(=1 : {agg['R1_eq1_frac']*100:5.1f} %) | R1v {agg['R1v_mean']:7.2f} "
              f"(=1 : {agg['R1v_eq1_frac']*100:5.1f} % ; ≤k : "
              f"{agg['R1v_le_k_frac']*100:5.1f} %) | cos {agg['R2_cos_mean']:+.4f} "
              f"| z {agg['R2z_mean']:+.3f} (méd {agg['R2z_median']:+.3f}) | "
              f"R3 {agg['R3_mean']:.6f} | R4 {agg['R4_mean']:.4f}")
    res["R2z_null"] = {"reference": "distracteur 30 k (cos de la clé de requête "
                                    "contre les 30 k clés distractrices)",
                       "per_query": null_stats}

    # ---- déclencheur multi-clé (§6), écrit avant les données ----
    trig = None
    if dist_store is not None:
        armed_a, armed_b, armed_c, per_sec_trig = 0, 0, 0, []
        for si in range(n_s):
            rows = [r for r in instr["paraphrase"] if r["secret"] == secrets[si]]
            ex = [r for r in instr["exact"] if r["secret"] == secrets[si]][0]
            a = sum(1 for r in rows if r["R1"] > 1) >= P1_PARAPHRASE_MIN
            zs = [r["R2z"] for r in rows]
            # (b) a DEUX moitiés : médiane paraphrase ≤ 3.0 ET z(exact) hors
            # échelle. « Hors échelle » est opérationnalisé sur la MÊME borne de
            # l'énoncé (z > 3.0) — les deux nombres sont loggés séparément.
            b = float(np.median(zs)) <= 3.0 and float(ex["R2z"]) > 3.0
            zpp = []
            for i in range(len(rows)):
                for j in range(len(rows)):
                    if i == j:
                        continue
                    ki = queries["A"][si][i + 1].key("final", 0)
                    kj = queries["A"][si][j + 1].key("final", 0)
                    nl = null_of(ki)
                    cs = float(np.dot(ki, kj) / (np.linalg.norm(ki)
                                                 * np.linalg.norm(kj) + 1e-12))
                    zpp.append((cs - nl["mu"]) / nl["sigma"])
            c_clause = float(np.median(zpp)) >= float(np.median(zs)) + 2.0
            armed_a += int(a)
            armed_b += int(b)
            armed_c += int(c_clause)
            per_sec_trig.append({"secret": secrets[si], "a_R1_gt1": bool(a),
                                 "b_full": bool(b),
                                 "b_z_para_le_3": bool(float(np.median(zs)) <= 3.0),
                                 "b_z_exact_off_scale": bool(float(ex["R2z"]) > 3.0),
                                 "z_para_median": float(np.median(zs)),
                                 "z_exact": ex["R2z"],
                                 "z_para_para_median": float(np.median(zpp)),
                                 "c_discriminant": bool(c_clause)})
        armed = (armed_a >= 6 and armed_b >= 6 and armed_c >= 6)
        trig = {"armed": bool(armed), "n_a": armed_a, "n_b": armed_b, "n_c": armed_c,
                "per_secret": per_sec_trig,
                "rule": "ARMÉ ssi (a) ∧ (b) ∧ (c) sur ≥ 6/10 secrets ; "
                        "(c) FAUSSE ⇒ NE PAS ARMER (renvoie vers P6 ou Fast-KV)"}
        print(f"  déclencheur multi-clé (§6) : (a) {armed_a}/10, (b) {armed_b}/10, "
              f"(c) {armed_c}/10 → {'ARMÉ' if armed else 'NON ARMÉ'}")
    res["multikey_trigger"] = trig

    # ------------------------------------------------------------------- P5
    print("\n================ P5 (porte d'IMPLÉMENTATION : E3 ≤ 0.05 est un THÉORÈME) ================")
    p5 = {"n_positions": n_pos, "conditions": {}}
    for lam_label, lam in (("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
        for c_label, c in (("un-hot", UNHOT), ("c=1", 1.0)):
            fact = [e3_vector(si, lam, c)[0] for si in range(n_s)]
            entry = {"lambda": lam, "c": c,
                     "edge_-log(1-lambda)": -math.log1p(-lam),
                     "E3_per_secret": [float(s.mean()) for s in fact],
                     "E3_mean": float(np.mean([s.mean() for s in fact])),
                     "E3_max_secret": float(np.max([s.mean() for s in fact])),
                     "boot_by_secret": bootstrap_mean([float(s.mean()) for s in fact]),
                     "max_delta": float(np.max([s.max() for s in fact])),
                     "min_delta": float(np.min([s.min() for s in fact]))}
            entry["within_theorem"] = bool(entry["E3_mean"] <= E3_THRESHOLD) \
                if lam <= LAMBDA_STAR else None
            p5["conditions"][f"fait-seul|{lam_label}|{c_label}"] = entry
            if dist_store is not None:
                dc = {si: np.concatenate([d2_neutral_dist, d2_neutral[si]], axis=1)
                      for si in (0,)}
                pass
    if dist_store is not None:
        vals_comb = {si: np.concatenate([z["E_values"], z[f"A_values_{si}"]])
                     for si in range(n_s)}
        nbr_comb, d2min_comb = {}, {}
        for si in range(n_s):
            comb = np.concatenate([d2_neutral_dist, d2_neutral[si]], axis=1)
            nbr_comb[si] = neighbors_from_matrix(comb, vals_comb[si])
            d2min_comb[si] = comb.min(axis=1).astype(np.float64)
        for lam_label, lam in (("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
            for c_label, c in (("un-hot", UNHOT), ("c=1", 1.0)):
                dist = [e3_vector(si, lam, c, nbr=nbr_comb[si],
                                  d2min=d2min_comb[si])[0] for si in range(n_s)]
                p5["conditions"][f"distracteur|{lam_label}|{c_label}"] = {
                    "lambda": lam, "c": c, "edge_-log(1-lambda)": -math.log1p(-lam),
                    "E3_per_secret": [float(s.mean()) for s in dist],
                    "E3_mean": float(np.mean([s.mean() for s in dist])),
                    "E3_max_secret": float(np.max([s.mean() for s in dist])),
                    "boot_by_secret": bootstrap_mean([float(s.mean()) for s in dist]),
                    "max_delta": float(np.max([s.max() for s in dist])),
                    "min_delta": float(np.min([s.min() for s in dist]))}
    for name, e in p5["conditions"].items():
        flag = ""
        if e["lambda"] <= LAMBDA_STAR:
            flag = ("  !! E3(λ*) > 0.05 ⇒ BUG (§4bis-7)" if e["E3_mean"] > E3_THRESHOLD
                    else "  (bande prédite [0.046, 0.050) ; point Math 0.0464-0.0469)")
        print(f"  [{name:<34}] E3 = {e['E3_mean']:+.6f} nats/token "
              f"(max secret {e['E3_max_secret']:+.6f}) | bord "
              f"{e['edge_-log(1-lambda)']:.6f} | max ΔNLL {e['max_delta']:.6f}{flag}")
        csv_rows.append(["P5", name, "E3_mean", f"{e['E3_mean']:.6f}",
                         f"{mean_sd(e['E3_per_secret'])[1]:.6f}", n_s])
    # ordinal Neuro
    for lam_label in ("lambda_star", "0.25"):
        for c_label in ("un-hot", "c=1"):
            ka = f"fait-seul|{lam_label}|{c_label}"
            kb = f"distracteur|{lam_label}|{c_label}"
            if kb in p5["conditions"]:
                a, b = p5["conditions"][ka]["E3_mean"], p5["conditions"][kb]["E3_mean"]
                p5[f"ordinal_fait_gt_distracteur|{lam_label}|{c_label}"] = bool(a > b)
                print(f"  ordinal Neuro ({lam_label}, {c_label}) : E3(fait-seul) "
                      f"{a:+.6f} {'>' if a > b else '≤'} E3(distracteur) {b:+.6f} → "
                      f"{'conforme' if a > b else 'INVERSE'}")
    e3_bug = any(e["lambda"] <= LAMBDA_STAR and e["E3_mean"] > E3_THRESHOLD
                 for e in p5["conditions"].values())
    p5["E3_lambda_star_exceeds_theorem"] = bool(e3_bug)
    res["P5"] = p5

    # --------------------------------------------------------------- P5c-id
    print("\n================ P5c-id (T2 par IDENTITÉ — partition exacte) ================")
    p5c = {}
    for c_label, c in (("un-hot", UNHOT), ("c=0.03", 0.03), ("c=0.1", 0.1),
                       ("c=0.3", 0.3), ("c=1", 1.0), ("c=3", 3.0)):
        fr, dt_relief, dt_full = [], [], []
        for si in range(n_s):
            D, mass = e3_vector(si, LAMBDA_STAR, c)
            rel = mass > 0.0
            fr.append(float(rel.mean()))
            if rel.any():
                dt_relief.extend(D[rel].tolist())
            dt_full.extend(D[~rel].tolist())
        p5c[c_label] = {
            "f_relief_mean": float(np.mean(fr)), "f_relief_per_secret": fr,
            "n_relief_positions_total": len(dt_relief),
            "D_relief_mean": float(np.mean(dt_relief)) if dt_relief else math.nan,
            "D_relief_min": float(np.min(dt_relief)) if dt_relief else math.nan,
            "D_relief_max": float(np.max(dt_relief)) if dt_relief else math.nan,
            "D_complement_mean": float(np.mean(dt_full)) if dt_full else math.nan,
            "D_complement_var": float(np.var(dt_full)) if dt_full else math.nan}
        e = p5c[c_label]
        print(f"  [{c_label:<7}] f_relief = {e['f_relief_mean']*100:6.3f} % "
              f"({e['n_relief_positions_total']} positions sur {n_pos*n_s}) | "
              f"D_t soulagé : moy {e['D_relief_mean']:+.6f} "
              f"[{e['D_relief_min']:+.6f}, {e['D_relief_max']:+.6f}] | "
              f"complément : moy {e['D_complement_mean']:+.6f} "
              f"var {e['D_complement_var']:.2e}")
    unhot_f = p5c["un-hot"]["f_relief_mean"]
    finite_f = [p5c[k]["f_relief_mean"] for k in p5c if k != "un-hot"]
    p5c["prediction_check"] = {
        "f_relief_unhot": unhot_f,
        "in_predicted_band_0.005_0.05": bool(0.005 <= unhot_f <= 0.05),
        "strictly_less_than_all_finite_c": bool(all(unhot_f < f for f in finite_f)),
        "antipode_normalisation_fausse": bool(any(unhot_f >= f for f in finite_f))}
    print(f"  prédiction : f_relief(un-hot) ∈ [0.5 %, 5 %] → "
          f"{p5c['prediction_check']['in_predicted_band_0.005_0.05']} | "
          f"STRICTEMENT < f_relief(c fini) → "
          f"{p5c['prediction_check']['strictly_less_than_all_finite_c']}")
    res["P5c_id"] = p5c

    # ----------------------------------------------------------------- P5f
    print("\n================ P5f (D11 comportemental — bascules d'argmax, semi-analytique) ================")
    p5f = {"note": "critère DÉRIVÉ : bascule ⟺ p_max − p_LM(v_nn) < λ/(1−λ) ⇒ "
                   "« concentrées aux marges fines » est quasi tautologique ; "
                   "seuls le TAUX et l'ANTIPODE sont informatifs"}
    dec_edges = np.quantile(D_nll, np.linspace(0, 1, P5C_DECILES + 1)[1:-1])
    dec = np.clip(np.digitize(D_nll, dec_edges), 0, P5C_DECILES - 1)
    for lam_label, lam in (("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
        flips = np.zeros(n_pos, dtype=np.float64)
        for si in range(n_s):
            idx, d2k, valsk = nbr_neutral[si]
            w = knn_weights_batch(d2k, UNHOT)
            for t in range(n_pos):
                pk = {}
                for v, wi in zip(valsk[t].tolist(), w[t].tolist()):
                    pk[int(v)] = pk.get(int(v), 0.0) + float(wi)
                if mix_argmax(D_logp[t], pk, lam) != mix_argmax(D_logp[t], {}, 0.0):
                    flips[t] += 1.0
        rate = flips / n_s
        by_dec = [float(rate[dec == d].mean()) for d in range(P5C_DECILES)]
        p5f[lam_label] = {"lambda": lam, "flip_rate_overall": float(rate.mean()),
                          "margin_threshold": lam / (1 - lam),
                          "flip_rate_by_nll_decile": by_dec,
                          "flip_rate_decile_1_most_confident": by_dec[0],
                          "flip_rate_decile_10_least_confident": by_dec[-1],
                          "antipode_flips_at_confident_positions":
                              bool(by_dec[0] > by_dec[-1])}
        e = p5f[lam_label]
        print(f"  λ={lam_label:<12} taux de bascule = {e['flip_rate_overall']*100:.3f} % "
              f"(prédit < 2 % à λ*) | seuil de marge λ/(1−λ) = "
              f"{e['margin_threshold']:.5f}")
        print("      par décile de NLL_base (1 = plus confiant) : "
              + " ".join(f"{v*100:.2f}" for v in by_dec))
        print(f"      ANTIPODE (bascules aux positions CONFIANTES) : "
              f"{e['antipode_flips_at_confident_positions']}")
    res["P5f"] = p5f

    # ------------------------------------------------------------------- P6
    print("\n================ P6 (prédiction ordinale signée : couche 6 vs état final) ================")
    p6 = {}
    for arm in ARMS:
        n_p, cases_p, per_p = p1_table(stores[arm], arm, LAMBDA_STAR)
        ex_hits, r1s, r1vs = 0, [], []
        for si in range(n_s):
            e = eval_case(stores[arm][si], queries["A"][si][0], arm, LAMBDA_STAR,
                          correct=correct_idx[si])
            ex_hits += int(e["top10"])
            r1s.append(e["R1"])
            r1vs.append(e["R1v"])
        p6[arm] = {"n_secrets_paraphrase": n_p,
                   "n_cases_top10": sum(c["top10"] for c in cases_p),
                   "h_per_secret": [s["h"] for s in per_p],
                   "h_mean": float(np.mean([s["h"] for s in per_p])),
                   "exact_top10": ex_hits, "R1_exact_mean": float(np.mean(r1s)),
                   "R1v_exact_mean": float(np.mean(r1vs)),
                   "R1_paraphrase_mean": float(np.mean([c["R1"] for c in cases_p])),
                   "R1v_paraphrase_mean": float(np.mean([c["R1v"] for c in cases_p]))}
        e = p6[arm]
        print(f"  bras {arm:<7} exact top-10 {ex_hits}/{n_s} | paraphrases "
              f"{n_p}/{n_s} secrets, {e['n_cases_top10']}/30 cas | h = "
              f"{e['h_mean']:.4f} | R1 exact {e['R1_exact_mean']:.2f} / paraphrase "
              f"{e['R1_paraphrase_mean']:.2f} | R1v paraphrase "
              f"{e['R1v_paraphrase_mean']:.2f}")
    n_better = sum(1 for a, b in zip(p6["inject"]["h_per_secret"],
                                     p6["final"]["h_per_secret"]) if a > b)
    p6["n_secrets_layer6_strictly_better_paraphrase"] = n_better
    p6["prediction_met"] = bool(n_better >= 6)
    p6["antipode_N2_falsified"] = bool(
        sum(1 for a, b in zip(p6["inject"]["h_per_secret"],
                              p6["final"]["h_per_secret"]) if a <= b) >= 6)
    print(f"  couche 6 STRICTEMENT meilleure sous paraphrase (h) sur "
          f"{n_better}/{n_s} secrets — prédiction : ≥ 6/10 → {p6['prediction_met']}")
    print(f"  ANTIPODE (couche 6 ≤ état final sur ≥ 6/10 ⇒ N2 falsifiée) : "
          f"{p6['antipode_N2_falsified']}")
    print("  AUCUNE portée sur X7 : P6 localise l'invariance, rien d'autre.")
    res["P6"] = p6

    # ------------------------------------------------------------------- P7
    print("\n================ P7 (sélectivité : store 30 k + le fait) ================")
    if dist_store is not None:
        idx_map = {si: correct_idx[si] + offset for si in range(n_s)}
        n7, cases7, per7 = p1_table([dist_store[si] for si in range(n_s)], "final",
                                    LAMBDA_STAR, idx_map=idx_map)
        h7 = [s["h"] for s in per7]
        h_base = [s["h"] for s in per1]
        deg = [(hb - hd) / hb if hb > 0 else math.nan
               for hb, hd in zip(h_base, h7)]
        # densité locale : nb d'entrées DISTRACTRICES plus proches que l'entrée correcte
        dens = []
        for si in range(n_s):
            per = []
            for qi in range(1, n_q):
                q = queries["A"][si][qi]
                d2 = squared_distances(q.key("final", 0), dist_store[si].keys)
                per.append(float((d2[:offset] < d2[idx_map[si]]).sum()))
            dens.append(float(np.median(per)))
        ok_pairs = [(d, x) for d, x in zip(deg, dens) if not math.isnan(d)]
        res["P7"] = {
            "n_secrets": n7, "n_cases_top10": sum(c["top10"] for c in cases7),
            "baseline_n_secrets": n1,
            "baseline_n_cases": sum(c["top10"] for c in cases1),
            "h_distractor_per_secret": h7, "h_fact_only_per_secret": h_base,
            "h_distractor_mean": float(np.mean(h7)),
            "h_fact_only_mean": float(np.mean(h_base)),
            "degradation_h_per_secret": deg,
            "degradation_h_mean": float(np.nanmean(deg)) if any(
                not math.isnan(d) for d in deg) else math.nan,
            "local_density_per_secret": dens,
            "degradation_variance": float(np.var([d for d, _ in ok_pairs]))
            if ok_pairs else math.nan,
            "corr_degradation_local_density": (pearson([d for d, _ in ok_pairs],
                                                       [x for _, x in ok_pairs])
                                               if len(ok_pairs) > 1 else math.nan),
            "corr_note": "corrélation NaN si la dégradation est CONSTANTE entre "
                         "secrets (variance nulle) — le cas est rapporté tel quel",
            "corr_degradation_store_size": math.nan,
            "corr_store_size_note": "UNE seule taille de store dans ce run ⇒ la "
                                    "corrélation avec la taille est INDÉFINIE (NaN), "
                                    "rapportée comme telle (Neuro N6 non testable ici)",
            "R1_paraphrase_mean": float(np.mean([c["R1"] for c in cases7])),
            "R1v_paraphrase_mean": float(np.mean([c["R1v"] for c in cases7])),
            "criterion": "dégradation ≤ 30 %"}
        e = res["P7"]
        print(f"  P1 sous distracteur : {n7}/{n_s} secrets ({n1} fait-seul) | "
              f"{e['n_cases_top10']}/30 cas ({e['baseline_n_cases']} fait-seul)")
        print(f"  h : {e['h_distractor_mean']:.4f} (distracteur) vs "
              f"{e['h_fact_only_mean']:.4f} (fait-seul) | dégradation moyenne = "
              f"{e['degradation_h_mean']*100:.1f} % (critère ≤ 30 %)")
        print(f"  R1 moyen parmi {offset + len(z['A_values_0'])} entrées = "
              f"{e['R1_paraphrase_mean']:.1f} | R1v moyen = "
              f"{e['R1v_paraphrase_mean']:.1f}")
        print(f"  corr(dégradation, densité locale) = "
              f"{e['corr_degradation_local_density']:+.4f} | corr(dégradation, "
              f"taille du store) = NaN ({e['corr_store_size_note']})")
    else:
        res["P7"] = None

    # ------------------------------------------------------------------- P8
    print("\n================ P8 (forme : rang vs d²_min, poolé) ================")
    pts = []
    for si in range(n_s):
        for qi in range(n_q):
            q = queries["A"][si][qi]
            e = eval_case(stores["final"][si], q, "final", LAMBDA_STAR)
            pts.append({"secret": secrets[si], "question": qnames[qi],
                        "d2min": e["d2_min"], "rank": e["rank_mix"],
                        "rank_base": e["rank_base"],
                        "log_rank": math.log(e["rank_mix"])})
    d2v = [p["d2min"] for p in pts]
    rkv = [p["rank"] for p in pts]
    order = np.argsort(d2v)
    binned, nb_bins = [], 8
    for b in range(nb_bins):
        sl = order[b * len(order) // nb_bins:(b + 1) * len(order) // nb_bins]
        if len(sl) == 0:
            continue
        binned.append({"n": int(len(sl)),
                       "d2min_mean": float(np.mean([d2v[i] for i in sl])),
                       "rank_median": float(np.median([rkv[i] for i in sl])),
                       "log_rank_mean": float(np.mean([math.log(rkv[i]) for i in sl]))})
    diffs = [binned[i + 1]["log_rank_mean"] - binned[i]["log_rank_mean"]
             for i in range(len(binned) - 1)]
    res["P8"] = {"n_points": len(pts), "spearman_d2min_rank": spearman(d2v, rkv),
                 "pearson_d2min_logrank": pearson(d2v, [p["log_rank"] for p in pts]),
                 "bins": binned, "consecutive_log_rank_diffs": diffs,
                 "monotone_increasing_bins": bool(all(d >= 0 for d in diffs)),
                 "max_abs_diff": float(max(abs(d) for d in diffs)) if diffs else math.nan,
                 "points": pts}
    print(f"  N={len(pts)} | Spearman(d²_min, rang) = "
          f"{res['P8']['spearman_d2min_rank']:+.4f} | Pearson(d²_min, log rang) = "
          f"{res['P8']['pearson_d2min_logrank']:+.4f}")
    print("    bin  n   d²_min moyen   rang médian   log-rang moyen")
    for i, b in enumerate(binned):
        print(f"    {i:<4} {b['n']:<3} {b['d2min_mean']:12.2f} {b['rank_median']:13.1f} "
              f"{b['log_rank_mean']:16.4f}")
    print("  écarts consécutifs de log-rang : " + " ".join(f"{d:+.3f}" for d in diffs)
          + f" | monotone croissant : {res['P8']['monotone_increasing_bins']}")

    # ------------------------------------------------- bras G (COURBE, aucun τ)
    print("\n================ BRAS G (courbe E3(τ)/P1(τ)/h(τ) — AUCUN τ fixé) ================")
    pooled = np.concatenate([d2min_neutral[si] for si in range(n_s)])
    fact_d2min = np.asarray([c["d2_min"] for c in cases1
                             + [e for e in ex_cases]], dtype=np.float64)
    qs = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    tau_grid = [0.0] + [float(np.quantile(pooled, q)) for q in qs] + [float("inf")]
    tau_grid = sorted(set(tau_grid))
    g_curve = []
    for tau in tau_grid:
        row = {"tau": tau,
               "neutral_pass_rate": float(np.mean(pooled <= tau)),
               "fact_query_pass_rate": float(np.mean(fact_d2min <= tau))}
        for lam_label, lam in (("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
            e3 = float(np.mean([e3_vector(si, lam, UNHOT, tau=tau)[0].mean()
                                for si in range(n_s)]))
            n_g, cases_g, per_g = p1_table(stores["final"], "final", lam, tau=tau)
            row[f"E3|{lam_label}"] = e3
            row[f"E3_within_budget|{lam_label}"] = bool(e3 <= E3_THRESHOLD)
            row[f"P1|{lam_label}"] = n_g
            row[f"h|{lam_label}"] = float(np.mean([s["h"] for s in per_g]))
        g_curve.append(row)
    promo = [r for r in g_curve
             if r["P1|0.25"] >= n1 and r["E3_within_budget|0.25"]]
    res["G"] = {"tau_grid": tau_grid, "curve": g_curve,
                "rule": "AUCUN τ fixé (la règle v1 « quantile 0.95 des d²_min FAIT » "
                        "est dégénérée : d²_min = 0.0 exact ⇒ τ = 0)",
                "promotion_trigger": "P1 conservée à λ = 0.25 AVEC E3 ≤ 0.05",
                "promotion_taus": [r["tau"] for r in promo],
                "promotion_fired": bool(promo),
                "note": "l'un-hot est aveugle à la force ⇒ G est le SEUL endroit du "
                        "run où une pondération par la force intervient ; question de "
                        "design de (b), jamais un résultat de l'instrument"}
    print("    τ            passage neutre  passage fait   E3(λ*)     P1(λ*)  h(λ*)   "
          "E3(0.25)   P1(0.25)  h(0.25)")
    for r in g_curve:
        print(f"    {r['tau']:<12.4g} {r['neutral_pass_rate']:>13.4f} "
              f"{r['fact_query_pass_rate']:>13.4f}  {r['E3|lambda_star']:+.6f} "
              f"{r['P1|lambda_star']:>7}  {r['h|lambda_star']:.4f}  "
              f"{r['E3|0.25']:+.6f} {r['P1|0.25']:>8}  {r['h|0.25']:.4f}")
    print(f"  déclencheur de promotion « G → organe » (P1 conservée à λ=0.25 AVEC "
          f"E3 ≤ 0.05) : {'DÉCLENCHÉ' if promo else 'non déclenché'}"
          + (f" aux τ = {promo[0]['tau']:.4g}…" if promo else ""))

    # ------------------------ bras descriptif : M au point courant + kNN
    if "B" in queries:
        print("\n================ BRAS DESCRIPTIF : M au point courant + kNN (additivité) ================")
        addit = {}
        for lam_label, lam in (("0.0", 0.0), ("lambda_star", LAMBDA_STAR),
                               ("0.25", 0.25)):
            rows = []
            for si in range(n_s):
                for qi in range(n_q):
                    qa, qb = queries["A"][si][qi], queries["B"][si][qi]
                    ea = eval_case(stores["final"][si], qa, "final", lam)
                    eb = eval_case(stores["final"][si], qb, "final", lam)
                    la = ea["logp_base"]
                    rows.append({"secret": secrets[si], "question": qnames[qi],
                                 "knn_only": ea["logp_mix"] - la,
                                 "M_only": eb["logp_base"] - la,
                                 "M_plus_knn": eb["logp_mix"] - la,
                                 "additive_gap": (eb["logp_mix"] - la)
                                 - ((ea["logp_mix"] - la) + (eb["logp_base"] - la))})
            for key in ("knn_only", "M_only", "M_plus_knn", "additive_gap"):
                for qname in qnames:
                    vals = [r[key] for r in rows if r["question"] == qname]
                    m_, s_ = mean_sd(vals)
                    addit.setdefault(lam_label, {}).setdefault(qname, {})[key] = \
                        {"mean": m_, "sd": s_, "n": len(vals)}
            ex = addit[lam_label]["exact"]
            print(f"  λ={lam_label:<12} exact : kNN seul {ex['knn_only']['mean']:+.4f} | "
                  f"M seule {ex['M_only']['mean']:+.4f} ± {ex['M_only']['sd']:.4f} | "
                  f"M+kNN {ex['M_plus_knn']['mean']:+.4f} | écart d'additivité "
                  f"{ex['additive_gap']['mean']:+.4f}")
        res["arm_M_plus_knn"] = addit

    # ----------------------------------------- décomposition à trois facteurs
    print("\n================ DÉCOMPOSITION À TROIS FACTEURS (sous sup_c) ================")
    dec_rows = []
    for cc in cases1:
        f_i = bool(0 < cc["R1v"] <= K_NEIGHBORS)
        f_ii = bool(cc["R3"] > MASS_FACTOR * cc["p10"])
        f_iii = bool(cc["p10"] < P10_FEASIBLE)
        dec_rows.append({"secret": cc["secret"], "question": cc["question"],
                         "presence_R1v_le_k": f_i, "mass_sufficient": f_ii,
                         "feasible_p10": f_iii, "conjunction": bool(f_i and f_ii and f_iii),
                         "top10": cc["top10"], "c_sup": cc["c_sup"], "R1v": cc["R1v"],
                         "R3": cc["R3"], "p10": cc["p10"]})
    agree = sum(1 for r in dec_rows if r["conjunction"] == r["top10"])
    unhot_rows = [r for r in dec_rows if r["c_sup"] == UNHOT]
    dec3 = {"n_cases": len(dec_rows),
            "n_presence": sum(r["presence_R1v_le_k"] for r in dec_rows),
            "n_mass_sufficient": sum(r["mass_sufficient"] for r in dec_rows),
            "n_feasible": sum(r["feasible_p10"] for r in dec_rows),
            "n_conjunction": sum(r["conjunction"] for r in dec_rows),
            "n_top10": sum(r["top10"] for r in dec_rows),
            "agreement_conjunction_vs_top10": agree,
            "mass_factor": MASS_FACTOR, "p10_threshold": P10_FEASIBLE,
            "n_cases_in_unhot_subcell": len(unhot_rows),
            "unhot_subcell_two_factor_identity":
                "dans la sous-cellule un-hot, (i)∧(ii) se réduit à "
                "`valeur(argmin) = cible` — identité exacte à deux facteurs (Math Q5)",
            "rows": dec_rows}
    print(f"  (i)   présence  R1v ≤ k = {K_NEIGHBORS} : {dec3['n_presence']}/30")
    print(f"  (ii)  masse suffisante p_kNN(cible) > {MASS_FACTOR:.4f}·p₁₀ : "
          f"{dec3['n_mass_sufficient']}/30")
    print(f"  (iii) faisabilité p₁₀ < {P10_FEASIBLE:.7f} : {dec3['n_feasible']}/30")
    print(f"  conjonction (i)∧(ii)∧(iii) : {dec3['n_conjunction']}/30 | top-10 observé "
          f"{dec3['n_top10']}/30 | accord {agree}/30")
    print(f"  sous-cellule un-hot : {len(unhot_rows)}/30 cas")
    res["decomposition_three_factors"] = dec3

    # table de covariables : faisable vs infaisable
    feas_flags = {(r["secret"], r["question"]): r["feasible_p10"] for r in dec_rows}
    cov = {}
    for label, want in (("faisable", True), ("infaisable", False)):
        sel = [cc for cc in cases1 if feas_flags[(cc["secret"], cc["question"])] == want]
        cov[label] = {"n": len(sel),
                      "median_H": float(np.median([c["H"] for c in sel])) if sel else math.nan,
                      "median_logp_base": float(np.median([c["logp_base"] for c in sel]))
                      if sel else math.nan,
                      "median_p10": float(np.median([c["p10"] for c in sel])) if sel else math.nan,
                      "median_p_max": float(np.median([c["p_max"] for c in sel])) if sel else math.nan}
        print(f"  covariables [{label:<10}] N={cov[label]['n']:>2} | médiane H = "
              f"{cov[label]['median_H']:.4f} | médiane logp_base = "
              f"{cov[label]['median_logp_base']:+.4f} | médiane p₁₀ = "
              f"{cov[label]['median_p10']:.6f}")
    res["covariates_feasible_vs_not"] = cov

    # ------------------------------------------------------ frontière complète
    print("\n================ FRONTIÈRE COMPLÈTE (λ × c) ================")
    frontier = []
    print("    λ         c       E3 analyt.  E3 mesuré   P1 n/10  cas/30  h       "
          "R1v méd  R3 moy     R4 moy   F₁₀ méd")
    for lam in LAMBDA_GRID:
        for c in C_GRID:
            e3 = float(np.mean([e3_vector(si, lam, c)[0].mean() for si in range(n_s)]))
            per_secret, cases_lc = [], []
            for si in range(n_s):
                hits = 0
                for qi in range(1, n_q):
                    q = queries["A"][si][qi]
                    _, d2k, valsk, d2all = stores["final"][si].query(
                        q.key("final", 0), K_NEIGHBORS)
                    tgt = int(q.target[0])
                    T = per_query_temperature(d2k, c)
                    pk = knn_distribution(d2k, valsk, T)
                    rk = mix_rank(q.logp[0], pk, lam, tgt)
                    lm = lm_prob_stats(q.logp[0], tgt)
                    top = rk <= RANK_TOP
                    hits += int(top)
                    cases_lc.append({"top10": bool(top),
                                     "R1v": rank_of_value(d2all, stores["final"][si].values, tgt),
                                     "R3": float(pk.get(tgt, 0.0)),
                                     "R4": knn_entropy(pk), "p10": lm["p10"],
                                     "F10": (INV_LAMBDA_STAR * lm["p10"] / (1 + lm["p10"]))
                                     if pk.get(tgt, 0.0) > 0 else math.inf})
                per_secret.append(hits)
            n_lc = sum(1 for hh in per_secret if hh >= P1_PARAPHRASE_MIN)
            f10v = [x["F10"] for x in cases_lc if math.isfinite(x["F10"])]
            row = {"lambda": lam, "c": c, "E3_analytic": -math.log1p(-lam),
                   "E3_measured": e3, "E3_within_budget": bool(e3 <= E3_THRESHOLD),
                   "P1_n_secrets": n_lc,
                   "P1_n_cases": sum(x["top10"] for x in cases_lc),
                   "h_mean": float(np.mean([hh / (n_q - 1) for hh in per_secret])),
                   "R1v_median": float(np.median([x["R1v"] for x in cases_lc])),
                   "R3_mean": float(np.mean([x["R3"] for x in cases_lc])),
                   "R4_mean": float(np.mean([x["R4"] for x in cases_lc])),
                   "F10_median_finite": float(np.median(f10v)) if f10v else math.nan,
                   "n_F10_infinite": len(cases_lc) - len(f10v)}
            frontier.append(row)
            csv_rows.append(["frontier", f"lam={lam:.6f}|c={c}", "E3_measured",
                             f"{e3:.6f}", "", n_s])
            csv_rows.append(["frontier", f"lam={lam:.6f}|c={c}", "P1_n_secrets",
                             n_lc, "", n_s])
            print(f"    {lam:.6f}  {c:<6} {row['E3_analytic']:.6f}    "
                  f"{e3:+.6f}  {n_lc:>6}  {row['P1_n_cases']:>6}  "
                  f"{row['h_mean']:.4f}  {row['R1v_median']:>7.1f}  "
                  f"{row['R3_mean']:.6f}  {row['R4_mean']:.4f}  "
                  f"{row['F10_median_finite']:.4f}")
    res["frontier"] = frontier

    # ---------------------------------------------------- table d'attribution
    print("\n================ TABLE D'ATTRIBUTION (§4.3) ================")
    par = instr["paraphrase"]
    ex = instr["exact"]
    r1_eq1 = float(np.mean([r["R1"] == 1 for r in par]))
    r1v_eq1 = float(np.mean([r["R1v"] == 1 for r in par]))
    r3_mean = float(np.mean([r["R3"] for r in par]))
    z_par = float(np.median([r["R2z"] for r in par])) if "R2z" in par[0] else math.nan
    z_ex = float(np.median([r["R2z"] for r in ex])) if "R2z" in ex[0] else math.nan
    p1_ok = n1 >= 5
    if res["P1_antipode"]["degenerate_cell"]:
        row, diag = ("P1 échoue ; R1v > 1 majoritaire sur les échecs, sup au bord "
                     "de grille", "cellule dégénérée ⇒ INCONCLUSIF — cellule dégénérée")
    elif p1_ok:
        row = "P1 réussit"
        diag = ("adressage effectif (R1=1, R1v=1, R3 élevée)"
                if (r1_eq1 > 0.5 and r1v_eq1 > 0.5)
                else "ANTIPODE : succès sans l'entrée correcte ⇒ ININTERPRÉTABLE")
    elif r1_eq1 > 0.5 and r1v_eq1 > 0.5 and r3_mean >= 0.5:
        row = "P1 échoue ; R1 = 1 ; R1v = 1 ; R3 élevée"
        diag = ("Branche B — échec côté INJECTION ; SEUL résultat qui parle contre "
                "le candidat (b)")
    else:
        row = "P1 échoue ; R1 > 1 ; R1v > 1 ; R3 faible"
        diag = ("Branche A — échec côté CLÉ ; V2-D non réfuté ; évaluer le "
                "déclencheur multi-clé (§6)")
    attribution = {
        "P1_n_secrets": n1, "P1_n_cases": p1["n_cases_top10"],
        "h_mean": float(np.mean(h_values)), "h_median": h_boot["median"],
        "R1_paraphrase_mean": float(np.mean([r["R1"] for r in par])),
        "R1_paraphrase_eq1_frac": r1_eq1,
        "R1_exact_mean": float(np.mean([r["R1"] for r in ex])),
        "R1_exact_eq1_frac": float(np.mean([r["R1"] == 1 for r in ex])),
        "R1v_paraphrase_mean": float(np.mean([r["R1v"] for r in par])),
        "R1v_paraphrase_eq1_frac": r1v_eq1,
        "R1v_paraphrase_le_k_frac": float(np.mean([0 < r["R1v"] <= K_NEIGHBORS
                                                   for r in par])),
        "R1v_exact_mean": float(np.mean([r["R1v"] for r in ex])),
        "R2z_paraphrase_median": z_par, "R2z_exact_median": z_ex,
        "R3_paraphrase_mean": r3_mean,
        "R3_exact_mean": float(np.mean([r["R3"] for r in ex])),
        "R4_paraphrase_mean": float(np.mean([r["R4"] for r in par])),
        "row": row, "diagnostic": diag}
    res["attribution"] = attribution
    print(f"  R1  : exact {attribution['R1_exact_mean']:.2f} "
          f"(=1 : {attribution['R1_exact_eq1_frac']*100:.0f} %) | paraphrase "
          f"{attribution['R1_paraphrase_mean']:.2f} (=1 : {r1_eq1*100:.0f} %)")
    print(f"  R1v : exact {attribution['R1v_exact_mean']:.2f} | paraphrase "
          f"{attribution['R1v_paraphrase_mean']:.2f} (=1 : {r1v_eq1*100:.0f} % ; "
          f"≤ k : {attribution['R1v_paraphrase_le_k_frac']*100:.0f} %)")
    print(f"  R2z : exact (médiane) {z_ex:+.3f} | paraphrase (médiane) {z_par:+.3f}")
    print(f"  R3  : exact {attribution['R3_exact_mean']:.6f} | paraphrase "
          f"{r3_mean:.6f}   R4 paraphrase {attribution['R4_paraphrase_mean']:.4f}")
    print(f"  ⇒ ligne « {row} » → {diag}")

    # ------------------------------------------- confondants loggés (§5.11)
    conf = {"lambda0_replay_E1": meta.get("lambda0_replay_E1"),
            "store_values": {}, "corr_rank_mix_logp_base_per_secret": {}}
    for si in range(n_s):
        vals = z[f"A_values_{si}"].tolist()
        conf["store_values"][secrets[si]] = {
            "n_entries": len(vals), "n_distinct_values": len(set(vals)),
            "frac_neutral_positions_covered": float(
                np.mean([int(t) in set(vals) for t in D_tgt.tolist()]))}
        rows = [cc for cc in cases1 if cc["secret"] == secrets[si]]
        conf["corr_rank_mix_logp_base_per_secret"][secrets[si]] = spearman(
            [r["rank_mix"] for r in rows], [r["logp_base"] for r in rows])
    conf["ties_reported"] = res["V_tie"]["recall_queries"]["n_gt1"]
    conf["c_sup_histogram"] = c_hist
    res["confounders"] = conf
    print(f"\n  confondants : λ=0 rejoué E1 = {conf['lambda0_replay_E1']} | "
          f"part moyenne des positions neutres couvertes par les valeurs du store = "
          f"{float(np.mean([v['frac_neutral_positions_covered'] for v in conf['store_values'].values()])):.4f}")

    # ---------------------------------------------------- manifeste + sorties
    res["duration_analysis_s"] = time.time() - t_start
    _dump(out_dir, res, csv_rows)
    root = Path(__file__).resolve().parents[1]
    man_paths = [raw / "gpu_raw.npz", raw / "gpu_meta.json", raw / "esc_raw.npz",
                 raw / "esc_meta.json", out_dir / "analysis.json",
                 out_dir / "summary.csv",
                 root / "eval" / "knn_ceiling.py", root / "tests" / "test_knn_ceiling.py",
                 root / "engram" / "config.py", root / "engram" / "cortex.py",
                 root / "experiments" / "EXP-2026-08-21-knn-borne-logits-v2.md",
                 DRIFT_REFERENCE]
    rfc = root / "data" / "rfc9293.txt"
    if rfc.exists():
        man_paths.append(rfc)
    manifest = {"files": {}}
    for p in man_paths:
        if p.exists():
            manifest["files"][str(p).replace("\\", "/")] = {
                "sha256": sha256_file(p), "sha16": sha256_file(p)[:16],
                "bytes": p.stat().st_size}
    (out_dir / "manifest_sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  manifeste SHA-256 : {len(manifest['files'])} fichiers")
    print(f"  analyse : {res['duration_analysis_s']:.1f}s → {out_dir}")
    return res


def _dump(out_dir, res, csv_rows):
    with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bloc", "condition", "metrique", "moyenne", "ecart_type", "N"])
        w.writerows(csv_rows)


# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all", choices=["gpu", "analysis", "all"])
    ap.add_argument("--stage", default="all",
                    choices=["all", "core", "rest", "gates", "esc"])
    ap.add_argument("--secrets", type=int, default=10)
    ap.add_argument("--distractor-tokens", type=int, default=DISTRACTOR_TOKENS)
    ap.add_argument("--skip-distractor", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else \
        root / "experiments" / "results" / "knn-borne-logits-v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("gpu", "all"):
        _gpu_phase(args, out_dir)
    if args.phase in ("analysis", "all"):
        if args.stage == "esc":
            _escalation_analysis(out_dir)
        else:
            _analysis_phase(args, out_dir)


if __name__ == "__main__":
    main()
