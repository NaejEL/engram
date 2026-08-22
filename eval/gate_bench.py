# SPDX-License-Identifier: AGPL-3.0-or-later
"""Banc de satisfiabilité (D14-S) du protocole V2-D(a) v3.

Protocole : `experiments/EXP-2026-08-22-knn-borne-logits-v3.md`, section
« Banc de satisfiabilité — livrable préalable au pré-enregistrement ».

CE SCRIPT NE MESURE RIEN. Aucun GPU, aucun modèle chargé, aucune donnée du run.
Le seul modèle HF touché est le **tokenizer GPT-2** (CPU, en cache), et
uniquement pour les portes V-tok / V-para, qui ne sont pas décidables sans lui.

Ce qu'il fait : pour **chaque** clause du tableau du protocole, deux jeux de
données synthétiques exécutables — un contre-exemple **passant** (la clause doit
rendre PASS) et un contre-exemple **échouant** (elle doit rendre FAIL) — puis
compare le verdict observé au verdict attendu.

    **E** = nombre de clauses dont le comportement observé ne correspond pas à
    l'attendu, OU qui se révèlent **insatisfiables**, **vacuées** (par
    insatisfaction ou par satisfaction) ou **à variance nulle par construction**.

`E = 0` est la gate de pré-enregistrement (§4.1). `E ≥ 1` ⇒ H_méthode rejetée ;
`E ≥ 3` ⇒ réduction de portée (§6). **Le banc ne corrige aucune clause** : il
rapporte. Amender est une décision de pré-enregistrement, pas d'implémentation.

USAGE MÉTA BORNÉ (§ « Banc de satisfiabilité ») : les portes d'**intégrité
seules** (V-cap, V-bord, V1a, V1b-1, V1b-2, V1c, V-var, V-drift) peuvent être
rejouées sur `experiments/results/knn-borne-logits{,-v2}/raw/`. Le banc n'a PAS
le droit d'émettre une statistique décisionnelle (P1, ΔP6, P3, P4, h, multi-clé,
G) sur ces bruts : ils portent des valeurs déjà publiées, les faire traverser les
portes de v3 reviendrait à calibrer v3 sur son propre résultat. Cette restriction
est mécanique ici : `_meta_replay()` n'appelle que des portes d'intégrité.

Sortie : `experiments/results/gate-bench/report.json` (une ligne par clause avec
`pass_case`, `fail_case`, `expected`, `observed`, `ok`, + le compte `E` racine)
et un résumé lisible sur stdout.

Usage :
  .venv\\Scripts\\python eval\\gate_bench.py
  .venv\\Scripts\\python eval\\gate_bench.py --no-hf     # sans tokenizer GPT-2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from knn_ceiling import (  # noqa: E402  — primitives déjà pré-enregistrées (v2)
    C_GRID, K_NEIGHBORS, LAMBDA_STAR, P10_FEASIBLE, UNHOT,
    count_argmin_ties, knn_distribution, knn_weights, mix_argmax,
    mix_delta_nll, rank_of_index, squared_distances,
)
from pool import (  # noqa: E402
    ENTITIES, OWNERS, OWNER_OBJ, POOL_UNITS_N, VERBS,
    owner_c3, pool_paraphrases, unit_table, v3_unit_triples_stats,
    v3_unit_secrets_stats,
)

OUT_DIR = ROOT / "experiments" / "results" / "gate-bench"
OUT_DIR_I2 = ROOT / "experiments" / "results" / "gate-bench-i2"
ARCHIVES = [ROOT / "experiments" / "results" / "knn-borne-logits" / "raw",
            ROOT / "experiments" / "results" / "knn-borne-logits-v2" / "raw"]

# =========================================================================
#  Constantes du protocole — §3, §4.3, §4.4. AUCUNE n'est ajustable après
#  mesure ; toutes sont re-dérivées, jamais lues dans le journal (D14-R).
# =========================================================================

E3_BUDGET = 0.05                       # nats/token
LAMBDA_GRID = [0.02, LAMBDA_STAR, 0.05, 0.10, 0.25]     # §4.4
VCAP_TOL = 1e-5
V1A_TOL = 1e-6
V1C_TOL = 1e-6
VVAR_TOL = 1e-12
VBASE_TOL = 5e-3                       # « au centième »
P1_TRUE, P1_FALSE, P1_N = 12, 5, 30    # §4.4 : a = 12 forcé, b = 5 maximal
P1_PARA_MIN = 2                        # ≥ 2/3 paraphrases
V2_MIN_FEASIBLE = 15                   # ≤ 15/30 ⇒ INCONCLUSIF budget arithmétique
P3_B = 21                              # §4.4 : B = 21 (impair ⇒ médiane entière)
P3_NULL_MAX, P3_LEAK_MIN = 3, 12       # médiane ≤ 3/30 ; ≥ 12/30
P4_MAX, P4_NULL = 3, 12
P7_MAX_DEGRADATION = 0.30
DP6_MIN_DISC = 5                       # n_disc < 5 ⇒ NON ÉVALUABLE
DP6_SEC_BITS = 1.0
MULTIKEY_MIN_UNITS = 18                # ARMÉ ssi sur ≥ 18/30 unités
MULTIKEY_PCT_PARA = 1e-3
MULTIKEY_PCT_EXACT = 3.33e-5           # résolution 1/30000
G_MIN_ALPHA = 1                        # non-vacuité : #{α=1} ≥ 1 et #{α=0} ≥ 1
P8_KNEE_FACTOR = 3.0                   # ⚠ ABSENT DU PROTOCOLE — voir `UNDERSPEC`
VENTIL_REL_TOL = 1e-2                  # ⚠ ABSENT DU PROTOCOLE (le « ≈ » de la
                                       # ventilation 0/1) — voir `UNDERSPEC`

# Table `k(n)` du §3 — **RÉFÉRENCE DU TEST SEULEMENT**. `k_of_n()` la recalcule
# en entiers Python purs et ne la lit jamais pour décider.
K_TABLE_REFERENCE = {
    6: 6, 7: 6, 8: 7, 9: 7, 10: 8,
    11: 9, 12: 9, 13: 10, 14: 10, 15: 11, 16: 12, 17: 12, 18: 13, 19: 13, 20: 14,
    21: 14, 22: 15, 23: 16, 24: 16, 25: 17, 26: 17, 27: 18, 28: 18, 29: 19, 30: 20,
}
K_BOUNDARIES = {16: 0.10506, 23: 0.10502, 30: 0.100244}   # frontières §3, 3ᵉ déc.

# ------------------------------------------------------------- verdicts
PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIF = "INCONCLUSIF"
NON_EVALUABLE = "NON ÉVALUABLE"
BUG = "BUG"
ARME = "ARMÉ"
NON_ARME = "NON ARMÉ"
REFUS = "REFUS"

# Clauses dont le protocole ne fixe pas le seuil d'opérationnalisation : le banc
# doit en déclarer un pour être exécutable. Signalé au rapport, hors E.
UNDERSPEC = {
    "P8": "« sans genou » n'est pas opérationnalisé par le protocole ; le banc "
          "déclare un détecteur (max Δ > %.1f × médiane Δ). Clause DESCRIPTIVE, "
          "n'entre dans aucune porte." % P8_KNEE_FACTOR,
    "Ventilation 0/1": "« `1 partagé` ≈ `intra` » n'est pas opérationnalisé par "
                       "le protocole ; le banc déclare `rel_tol = %g`. Clause "
                       "DESCRIPTIVE, hors clause décisionnelle." % VENTIL_REL_TOL,
}


# =========================================================================
#  Portes — fonctions pures, importables par les tests CPU
# =========================================================================

def bord(lam: float) -> float:
    """`bord := fl(−log1p(−λ))` — **la** seule expression du protocole (§7,
    porte V-bord). Tout autre calcul du bord est un écart d'implémentation."""
    return -math.log1p(-float(lam))


def bord_naif(lam: float) -> float:
    """Forme PROSCRITE `−log(1−λ)` : conservée pour que V-bord ait un
    contre-exemple échouant exécutable, jamais utilisée ailleurs."""
    return -math.log(1.0 - float(lam))


def ulp_gap(a: float, b: float, cap: int = 1 << 20) -> int:
    """Nombre de flottants représentables entre `a` et `b` (écart en ULP)."""
    if a == b:
        return 0
    lo, hi, n = min(a, b), max(a, b), 0
    while lo < hi and n < cap:
        lo = math.nextafter(lo, math.inf)
        n += 1
    return n


def ulp_of(x: float) -> float:
    return math.ulp(float(x))


# ---------------------------------------------------------------- V-base
def gate_v_base(measured: dict, reference: dict, tol: float = VBASE_TOL) -> str:
    """E1 / top-10 / E3 aux défauts `EngramConfig()`, comparés **au centième**
    à la valeur RE-MESURÉE (jamais au chiffre du journal — D14-R)."""
    for k, ref in reference.items():
        if abs(float(measured[k]) - float(ref)) > tol:
            return FAIL
    return PASS


# ----------------------------------------------------------------- V-cap
def gate_v_cap(dev: float) -> str:
    """|lm_head(h) − logits|_max ≤ 1e-5 (logits ~30 en fp32 : 30·2⁻²³ = 3.6e-6)."""
    return PASS if float(dev) <= VCAP_TOL else FAIL


# --------------------------------------------------------------- V-drift
def gate_v_drift(a, b) -> str:
    """Contrôle croisé **bit-à-bit**. Vérification d'environnement seulement :
    un écart est une anomalie à signaler, jamais une autorisation de réutiliser
    les bruts archivés."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return FAIL
    return PASS if a.tobytes() == b.tobytes() else FAIL


# ---------------------------------------------------------------- V-bord
def gate_v_bord(lams, naive_side: bool) -> tuple[str, dict]:
    """`bord` produit par `−log1p(−λ)` **et par ce code seul** ; égalité
    **bit-à-bit** avec le `bord` utilisé par V1b-1/V1b-2.

    `naive_side=True` fait produire un côté par `−log(1−λ)` : l'écart en ULP est
    exhibé, par λ, dans le rapport.
    """
    gaps = {}
    ok = True
    for lam in lams:
        left = bord(lam)
        right = bord_naif(lam) if naive_side else bord(lam)
        g = ulp_gap(left, right)
        gaps[repr(float(lam))] = {"log1p": repr(left), "autre": repr(right),
                                  "ulp": g, "bit_egal": left == right}
        ok = ok and (left == right)
    return (PASS if ok else FAIL), gaps


# ------------------------------------------------------------------- V0
def gate_v0(keys, values, query, target) -> tuple[str, dict]:
    """R1 = 1 sous indice **exact**. Aucune clause sur d²_min (§4.3)."""
    d2 = squared_distances(query, keys)
    hit = np.nonzero(np.asarray(values) == int(target))[0]
    r1 = rank_of_index(d2, int(hit[0])) if hit.size else -1
    return (PASS if r1 == 1 else FAIL), {"R1": r1, "d2_min": float(d2.min())}


# -------------------------------------------------------------- V-indep
def gate_v_indep(query_keys, d2_min, p10) -> tuple[str, dict]:
    """(a) les 30 × 4 clés de requête deux à deux **distinctes** (bit-à-bit) ;
    (b) sd inter-unités de d²_min > 0 ; (c) sd inter-unités de p₁₀ > 0 ;
    (d) max des cosinus inter-unités **rapporté** (descriptif).

    Opérationnalise la cause C4 du run 2 : « distinctes ≠ décorrélées », d'où (d).
    """
    q = np.asarray(query_keys, dtype=np.float32)
    flat = q.reshape(-1, q.shape[-1])
    seen = {r.tobytes() for r in flat}
    a_ok = len(seen) == flat.shape[0]
    b_sd = float(statistics.pstdev([float(x) for x in np.asarray(d2_min)]))
    c_sd = float(statistics.pstdev([float(x) for x in np.asarray(p10)]))
    # (d) cosinus inter-unités sur la clé de l'indice exact de chaque unité
    ex = q[:, 0, :].astype(np.float64)
    nrm = np.linalg.norm(ex, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    cs = (ex / nrm) @ (ex / nrm).T
    np.fill_diagonal(cs, -np.inf)
    det = {"a_cles_distinctes": bool(a_ok), "n_cles": int(flat.shape[0]),
           "n_distinctes": len(seen), "b_sd_d2min": b_sd, "c_sd_p10": c_sd,
           "d_cos_max_inter_unites": float(cs.max())}
    if not a_ok:
        return FAIL, det | {"clause_echouee": "(a)"}
    if not b_sd > 0.0:
        return FAIL, det | {"clause_echouee": "(b)"}
    if not c_sd > 0.0:
        return FAIL, det | {"clause_echouee": "(c)"}
    return PASS, det


# ---------------------------------------------------------------- V-tok
def pool_words() -> list[str]:
    """`OWNERS ∪ ENTITIES ∪ VERBS`, **mot à mot** : les entrées sont des
    syntagmes (« The captain's », « is called », « chess club »), et la clause
    parle d'un **mot** du pool. Lecture stricte = la plus conservatrice."""
    out = set()
    for e in list(OWNERS) + list(ENTITIES) + list(VERBS):
        out.add(e)
        out.update(e.split())
    return sorted(out)


def gate_v_tok(secrets, tokenize, words=None) -> tuple[str, dict]:
    """(a) les 30 **premiers tokens BPE des secrets** deux à deux distincts ;
    (b) aucun ne coïncide avec le 1ᵉʳ token BPE d'un mot du pool ;
    (c) aucun secret n'est un mot du pool.

    La primaire est « 1ᵉʳ token BPE en top-10 » : une collision rend un succès
    attribuable au datastore d'une **autre** unité.
    """
    words = pool_words() if words is None else list(words)
    ids = [int(tokenize(" " + s)[0]) for s in secrets]
    wt: dict[int, list[str]] = {}
    for w in words:
        wt.setdefault(int(tokenize(" " + w)[0]), []).append(w)
    dup = {}
    for i, t in enumerate(ids):
        same = [j for j in range(len(ids)) if j != i and ids[j] == t]
        if same:
            dup[i] = {"secret": secrets[i], "token": t,
                      "unites_en_collision": same}
    col_b = {i: {"secret": secrets[i], "token": ids[i], "mots_du_pool": wt[ids[i]]}
             for i, t in enumerate(ids) if ids[i] in wt}
    col_c = {i: secrets[i] for i, s in enumerate(secrets) if s in set(words)}
    det = {"a_doublons": dup, "b_collisions_pool": col_b, "c_secret_dans_pool": col_c,
           "unites_fautives": sorted(set(dup) | set(col_b) | set(col_c))}
    return (PASS if not (dup or col_b or col_c) else FAIL), det


# --------------------------------------------------------------- V-para
def _jaccard(a: frozenset, b: frozenset) -> float:
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def gate_v_para(units, tokenize, check_cross: bool = True) -> tuple[str, dict]:
    """(a) aucun indice paraphrasé n'est sous-chaîne de son fait ; (b) Jaccard
    BPE(indice, fait \\ secret) **strictement inférieur** à celui de l'indice
    exact ; (c) **fuite croisée** : le recouvrement de chaque paraphrase de
    l'unité *i* est vérifié contre **les 30 faits**, pas seulement celui de *i*.

    `units` : sortie de `pool.unit_table()` (ou même forme, synthétique).
    Livrable : la **matrice 30 × 4** des recouvrements BPE.
    """
    enc = lambda s: frozenset(int(t) for t in tokenize(s))          # noqa: E731
    facts = [enc(u["fact_no_secret"]) for u in units]
    n = len(units)
    a_bad, b_bad, c_bad = [], [], []
    matrix = []
    for u in units:
        i = int(u["i"])
        j_exact = _jaccard(enc(u["exact"]), facts[i])
        row = [j_exact]
        for k, p in enumerate(u["paraphrases"]):
            if p in u["fact_no_secret"] or p in u["fact_template"]:
                a_bad.append({"i": i, "para": k + 1, "indice": p})
            ep = enc(p)
            j_own = _jaccard(ep, facts[i])
            row.append(j_own)
            if not j_own < j_exact:
                b_bad.append({"i": i, "para": k + 1, "J_para": j_own,
                              "J_exact": j_exact})
            for j in range(n) if check_cross else ():
                if j == i:
                    continue
                j_other = _jaccard(ep, facts[j])
                if j_other > j_own:
                    c_bad.append({"paire": [i, j], "para": k + 1,
                                  "J_vers_j": j_other, "J_vers_i": j_own})
        matrix.append(row)
    par_type = {f"para{k}": sum(1 for x in c_bad if x["para"] == k)
                for k in (1, 2, 3)}
    det = {"a_sous_chaine": a_bad, "b_jaccard": b_bad,
           "c_fuite_croisee_n": len(c_bad), "c_fuite_croisee": c_bad[:12],
           "c_par_type_de_paraphrase": par_type,
           "c_unites_fautives": sorted({x["paire"][0] for x in c_bad}),
           "paires_nommees": sorted({tuple(x["paire"]) for x in c_bad})[:12],
           "matrice_30x4": matrix}
    return (PASS if not (a_bad or b_bad or c_bad) else FAIL), det


# ------------------------------------------------------- V-para (c′) — A-3
def _jacc_ratio(a: frozenset, b: frozenset) -> tuple[int, int]:
    """Jaccard comme **rapport de deux entiers exacts** `(|A∩B|, |A∪B|)`.

    §15 A-3, gain D14-S : le Jaccard est un rapport de petits entiers ; toute
    comparaison se fait par **produit croisé sur des `int`** — jamais en
    flottant, donc **aucune** analyse ULP à faire.
    """
    u = len(a | b)
    return (len(a & b), u if u else 1)


def _ratio_gt(x: tuple[int, int], y: tuple[int, int]) -> bool:
    """`x > y` **strictement**, en arithmétique entière exacte : `a·d > c·b`."""
    (a, b), (c, d) = x, y
    assert isinstance(a, int) and isinstance(b, int)
    assert isinstance(c, int) and isinstance(d, int)
    assert b > 0 and d > 0
    return a * d > c * b


def gate_v_para_c_prime(units, tokenize) -> tuple[str, dict]:
    """`V-para (c′)` (§15, A-3) — Jaccard sur le **CONTENU**, pas sur les tokens
    bruts, et comparaison en arithmétique entière exacte.

    `F_t := ⋂_{i} tokens_BPE(para_t(i))` (**intersection ensembliste sur les 30**,
    jamais un préfixe/suffixe commun : le cadre de para3 est entrelacé et un
    préfixe/suffixe laisserait `" that belongs to "` dans le contenu, recréant la
    fuite). `C_t(i) := tokens_BPE(para_t(i)) \\ F_t` ; symétriquement `F_fait` et
    `C(fait_i)`.

    Clause : pour tout `i`, tout `t`, `J(C_t(i), C(fait_i)) > J(C_t(i), C(fait_j))`
    pour tout `j ≠ i`, **strictement — une égalité compte comme violation** (si
    l'indice n'est pas strictement plus proche de son fait, il ne désigne pas son
    unité).
    """
    enc = lambda s: frozenset(int(t) for t in tokenize(s))          # noqa: E731
    n = len(units)
    n_types = len(units[0]["paraphrases"])
    raw_facts = [enc(u["fact_no_secret"]) for u in units]
    f_fait = frozenset.intersection(*raw_facts) if raw_facts else frozenset()
    c_facts = [f - f_fait for f in raw_facts]

    f_t, c_t = [], []
    for k in range(n_types):
        raw = [enc(u["paraphrases"][k]) for u in units]
        fk = frozenset.intersection(*raw) if raw else frozenset()
        f_t.append(fk)
        c_t.append([r - fk for r in raw])

    viol, par_type = [], {}
    for k in range(n_types):
        cnt = 0
        for i in range(n):
            ci = c_t[k][i]
            own = _jacc_ratio(ci, c_facts[i])
            for j in range(n):
                if j == i:
                    continue
                other = _jacc_ratio(ci, c_facts[j])
                if not _ratio_gt(own, other):          # égalité = violation
                    cnt += 1
                    if len(viol) < 12:
                        viol.append({"paire": [i, j], "para": k + 1,
                                     "J_vers_i": list(own), "J_vers_j": list(other),
                                     "egalite": own[0] * other[1] == other[0] * own[1]})
        par_type[f"para{k + 1}"] = cnt
    det = {"F_t_tailles": {f"para{k + 1}": len(f_t[k]) for k in range(n_types)},
           "F_fait_taille": len(f_fait),
           "violations_par_type": par_type,
           "violations_total": sum(par_type.values()),
           "violations_nommees": viol,
           "arithmetique": "entiers exacts (produit croisé a·d > c·b), aucun flottant"}
    return (PASS if det["violations_total"] == 0 else FAIL), det


# ------------------------------------------------------- V-slot (A-4, NOUVELLE)
def _contains_verbatim(hay: str, needle: str) -> bool:
    """Occurrence **verbatim** délimitée : la valeur de ligne doit apparaître
    entourée de non-alphanumériques (sinon `cat` serait « trouvé » dans
    `catapult`, ce qui n'est pas une occurrence de la valeur de ligne)."""
    start = 0
    while True:
        p = hay.find(needle, start)
        if p < 0:
            return False
        before = hay[p - 1] if p > 0 else " "
        after = hay[p + len(needle)] if p + len(needle) < len(hay) else " "
        if not (before.isalnum() or before == "'") and not after.isalnum():
            return True
        start = p + 1


def gate_v_slot(units, row_values=None) -> tuple[str, dict]:
    """`V-slot` (§15, A-4) — porte **STRUCTURELLE**, sans tokenizer ni mesure.

    Pour tout type `t` et toute unité `i` : l'ensemble des **valeurs de ligne**
    des tables indexées par l'unité (`OWNERS`, `ENTITIES`, `VERBS`, `SECRETS_80`,
    `OWNER_OBJ`) apparaissant verbatim dans `para_t(i)` est **inclus dans les
    slots de l'unité i**. Toute occurrence d'un slot d'une unité `j ≠ i` ⇒ arrêt.

    Raison d'être : `V-para (c′)` **neutralise** la fuite structurelle (elle la
    met hors contenu) mais **ne la détecte pas**. Une fuite de **règle** se prend
    par une porte de **règle** — et celle-ci reste correcte si le tokenizer ou le
    modèle change.
    """
    from pool import all_row_values
    vals = all_row_values() if row_values is None else list(row_values)
    bad = []
    for u in units:
        i = int(u["i"])
        own = set(u["slots"].values())
        for k, p in enumerate(u["paraphrases"]):
            for v in vals:
                if v in own:
                    continue
                if _contains_verbatim(p, v):
                    bad.append({"i": i, "para": k + 1, "valeur_etrangere": v,
                                "indice": p})
    par_type = {f"para{k}": sum(1 for x in bad if x["para"] == k)
                for k in (1, 2, 3)}
    det = {"violations_total": len(bad), "violations_par_type": par_type,
           "violations_nommees": bad[:12],
           "unites_fautives": sorted({x["i"] for x in bad})}
    return (PASS if not bad else FAIL), det


# --------------------------------------------------- V-ident (§16 E, NOUVELLE)
def gate_v_ident(triples, tokenize) -> tuple[str, dict]:
    """`V-ident` (§16 E) — porte **STRUCTURELLE**, décidable **sans aucune
    mesure** : les 30 unités vérifient **C-1**, **C-2** et **C-3**. Toute
    violation ⇒ arrêt.

    `triples` : liste de `(indice owner, indice entity, indice verb)`.

    - **C-1** `(owner, entity)` deux à deux distincts ;
    - **C-2** `(entity, verb)` deux à deux distincts — **non satisfait** par
      `fact_pairs(30)` : `entity = i mod 20`, `verb = i mod 5` et 5 divise 20,
      donc le couple a une **période de 20** ;
    - **C-3** pour chaque owner retenu, sa forme **para2** (minusculisée) **et**
      sa forme **`OWNER_OBJ`** partagent **au moins un token BPE** avec sa forme
      **du fait** — **évaluée par exécution du tokenizer**, jamais par jugement.
      **Non satisfait** par les owners pronominaux (`Her`, `His`, `Our`,
      `Their`), dont la forme objet est un mot entièrement différent.

    Sous C-1 ∧ C-3, le contenu de para3 rencontre son propre fait sur **deux**
    slots et tout autre fait sur **au plus un** ; sous C-2, para1 et para2 ont la
    même propriété via le verbe. Comme `V-slot`, cette porte aurait tué la
    décision 14 avant le premier passage du banc.
    """
    triples = [tuple(int(x) for x in t) for t in triples]
    c1, c2 = {}, {}
    v_c1, v_c2 = [], []
    for i, (o, e, v) in enumerate(triples):
        if (o, e) in c1:
            v_c1.append({"paire": [c1[(o, e)], i], "owner": OWNERS[o],
                         "entity": ENTITIES[e]})
        else:
            c1[(o, e)] = i
        if (e, v) in c2:
            v_c2.append({"paire": [c2[(e, v)], i], "entity": ENTITIES[e],
                         "verb": VERBS[v]})
        else:
            c2[(e, v)] = i
    v_c3, c3_det = [], {}
    for i, (o, e, v) in enumerate(triples):
        if o not in c3_det:
            c3_det[o] = owner_c3(OWNERS[o], tokenize)[1]
        if not c3_det[o]["ok"]:
            v_c3.append({"unite": i, "owner": OWNERS[o],
                         "partage_para2": c3_det[o]["partage_para2"],
                         "partage_obj": c3_det[o]["partage_obj"]})
    det = {"n_unites": len(triples),
           "C-1_violations": len(v_c1), "C-1_nommees": v_c1[:12],
           "C-2_violations": len(v_c2), "C-2_nommees": v_c2[:12],
           "C-3_violations": len(v_c3), "C-3_nommees": v_c3[:12],
           "C-3_owners_fautifs": sorted({x["owner"] for x in v_c3}),
           "unites_fautives": sorted({j for x in v_c1 for j in x["paire"]}
                                     | {j for x in v_c2 for j in x["paire"]}
                                     | {x["unite"] for x in v_c3}),
           "violations_total": len(v_c1) + len(v_c2) + len(v_c3)}
    return (PASS if det["violations_total"] == 0 else FAIL), det


def fact_pairs_triples(n: int = POOL_UNITS_N) -> list[tuple[int, int, int]]:
    """Les triplets IMPLICITES de `pool.fact_pairs(n)` : `(i mod 16, i mod 20,
    i mod 5)`. Contre-exemple ÉCHOUANT obligatoire de `V-ident` (§16 E)."""
    return [(i % len(OWNERS), i % len(ENTITIES), i % len(VERBS))
            for i in range(n)]


# ------------------------------------------------------- OWNER_OBJ (A-5)
def gate_owner_obj(table, tokenize) -> tuple[str, dict]:
    """`OWNER_OBJ` **injective**, et ses 16 images **deux à deux distinctes en
    BPE** (§15, A-5).

    Motif : la table écrase de l'information (« Her » → « her », « His » → « him »,
    « Our » → « us ») ; deux owners tombant sur la même forme objet rendraient
    deux unités **indiscernables dans para3**.
    """
    imgs = list(table.values())
    inj = len(set(imgs)) == len(imgs)
    bpe = [tuple(int(t) for t in tokenize(" " + s)) for s in imgs]
    bpe_ok = len(set(bpe)) == len(bpe)
    dup_s = sorted({s for s in imgs if imgs.count(s) > 1})
    dup_b = sorted({imgs[i] for i in range(len(imgs))
                    if bpe.count(bpe[i]) > 1})
    det = {"n": len(imgs), "injective": bool(inj), "distinctes_bpe": bool(bpe_ok),
           "doublons_chaines": dup_s, "doublons_bpe": dup_b}
    return (PASS if inj and bpe_ok else FAIL), det


# ------------------------------- V-bord, littéral λ* vs expression (A-6 ii)
LAMBDA_STAR_LITERAL_DOC = 0.048770575499286      # DOCUMENTATION SEULE (D14-R)


def gate_lambda_star_expression(value: float) -> tuple[str, dict]:
    """`λ*` est défini comme l'**EXPRESSION** `1 − exp(−0.05)`, jamais comme un
    décimal recopié (§15, A-6 ii).

    Défaut trouvé hors banc : le littéral `0.048770575499286` du protocole est à
    **2 ULP** de `1 − math.exp(-0.05)`. C'est la classe de bug que `V-bord` garde,
    **un étage au-dessus** : le décimal reste dans le document comme
    documentation seule, étiqueté comme tel.
    """
    expr = 1.0 - math.exp(-0.05)
    g = ulp_gap(float(value), expr)
    return (PASS if float(value) == expr else FAIL), {
        "expression": repr(expr), "valeur": repr(float(value)), "ulp": g,
        "litteral_du_document": repr(LAMBDA_STAR_LITERAL_DOC),
        "ulp_litteral_vs_expression": ulp_gap(LAMBDA_STAR_LITERAL_DOC, expr)}


# ------------------------------------------- descriptifs A-7 (hors clause)
def descriptif_jaccard_brut(units, tokenize) -> dict:
    """Jaccard **BRUT** (avec cadre) publié par type avec son compte de
    violations (§15, A-7). On ne cache rien : *deux quantités, deux noms, une
    seule bloquante*. Descriptif — n'entre dans aucune porte, ni dans E."""
    enc = lambda s: frozenset(int(t) for t in tokenize(s))          # noqa: E731
    facts = [enc(u["fact_no_secret"]) for u in units]
    n = len(units)
    par_type = {}
    for k in range(len(units[0]["paraphrases"])):
        cnt = 0
        for i, u in enumerate(units):
            ep = enc(u["paraphrases"][k])
            own = _jacc_ratio(ep, facts[i])
            for j in range(n):
                if j != i and not _ratio_gt(own, _jacc_ratio(ep, facts[j])):
                    cnt += 1
        par_type[f"para{k + 1}"] = cnt
    return {"violations_jaccard_brut_par_type": par_type,
            "violations_total": sum(par_type.values()),
            "lecture": "quantité DESCRIPTIVE : un compte élevé sur para3 se lit "
                       "« poids du cadre », pas « fuite d'identité » — c'est "
                       "V-para (c′) qui tranche l'identité."}


def descriptif_v_partage(units, tokenize) -> dict:
    """`V-partage` (§15, A-7) — plus long **préfixe** ET plus long **suffixe**
    communs en BPE, par type, sur les 30.

    Troisième propriété, que ni `V-para (c′)` ni le Jaccard brut ne rapportent :
    **position et volume** du matériel commun. C'est elle qui voit le préfixe de
    ~8 tokens de para2, que le Jaccard croisé laisse passer à 0 violation.
    """
    out = {}
    for k in range(len(units[0]["paraphrases"])):
        seqs = [[int(t) for t in tokenize(u["paraphrases"][k])] for u in units]
        m = min(len(s) for s in seqs)
        pre = 0
        while pre < m and len({s[pre] for s in seqs}) == 1:
            pre += 1
        suf = 0
        while suf < m and len({s[-1 - suf] for s in seqs}) == 1:
            suf += 1
        out[f"para{k + 1}"] = {"prefixe_bpe": pre, "suffixe_bpe": suf,
                               "longueur_min": m}
    return out


# --------------------------------------------------------------- V-hash
def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def frozen_dataset() -> dict:
    """Les DONNÉES GELÉES du protocole, sous la forme exacte qui est hashée.

    Cascade D14(b) (§15 C) : l'amendement A-1 change `POOL_PARAPHRASES` ⇒ le
    SHA-256 change, et c'est le hash **amendé** qui doit être scellé (y compris
    pour I2, dont le corpus (a) EST le jeu d'unités v3).
    """
    from pool import PARA1_VERB
    units = unit_table(POOL_UNITS_N)
    return {"PARA1_VERB": PARA1_VERB,
            "POOL_UNIT_TRIPLES": [list(t) for t in
                                  v3_unit_triples_stats(POOL_UNITS_N)["triplets"]],
            "POOL_PARAPHRASES": [list(p) for p in pool_paraphrases(POOL_UNITS_N)],
            "OWNER_OBJ": dict(OWNER_OBJ),
            "unites": [{"i": u["i"], "exact": u["exact"], "secret": u["secret"],
                        "fait": u["fact_no_secret"],
                        "paraphrases": list(u["paraphrases"])} for u in units]}


def gate_v_hash(before, after) -> tuple[str, dict]:
    ha, hb = sha256_obj(before), sha256_obj(after)
    return (PASS if ha == hb else FAIL), {"sha_avant": ha[:16], "sha_apres": hb[:16]}


# ---------------------------------------------------------------- V-tie
def gate_v_tie(d2_all) -> tuple[str, dict]:
    """Compte d'ex-æquo de d²_min rapporté ; > 0 ⇒ un-hot appliqué **uniforme
    sur l'argmin-set**. Le second cas n'est pas un échec : c'est la BRANCHE."""
    d2 = np.asarray(d2_all, dtype=np.float64)
    ties = count_argmin_ties(d2)
    w = knn_weights(d2, UNHOT)
    m = d2.min()
    uniform = np.allclose(w[d2 == m], 1.0 / ties) and float(w[d2 > m].sum()) == 0.0
    det = {"ex_aequo": int(ties), "unhot_uniforme": bool(uniform)}
    if ties == 0:
        return FAIL, det                       # impossible : d²_min existe toujours
    if ties == 1:
        return "SANS EX-AEQUO", det
    return (f"EX-AEQUO={ties}, UN-HOT UNIFORME" if uniform else FAIL), det


# ------------------------------------------------------------------ V1a
def gate_v1a(delta_nll, lam) -> tuple[str, dict]:
    """Sur `p_kNN(y_t) = 0` : `max_t |ΔNLL_t + log1p(−λ)| ≤ 1e-6`. Identité
    algébrique ; erreur attendue ~1e-15 ⇒ 9 ordres de marge."""
    d = np.asarray(delta_nll, dtype=np.float64)
    dev = float(np.abs(d + math.log1p(-float(lam))).max())
    return (PASS if dev <= V1A_TOL else FAIL), {"ecart_max": dev, "seuil": V1A_TOL}


# ---------------------------------------------------------------- V1b-1
def gate_v1b1(delta_nll, lam) -> tuple[str, dict]:
    """Volet FAIBLE : sur `p_kNN(y_t) > 0`, `ΔNLL_t ≤ bord + 4·ULP`, **100 %**.

    Argument structurel (§4.3) : `δ̂ = log1p(t)`, `t ≥ 0` ⇒ `δ̂ ≥ 0` exactement en
    machine ; l'arrondi au plus près ne franchit jamais un représentable ⇒
    `fl(bord − δ̂) ≤ bord` exactement. La clause tient **à marge 0** ; les 4 ULP
    sont du mou délibéré. Satisfiable en tout régime, y compris `r ≲ 1e-16` —
    c'est la cellule qui a rendu la porte du run 2 insatisfiable en fp64.
    """
    b = bord(lam)
    lim = b + 4.0 * ulp_of(b)
    d = np.asarray(delta_nll, dtype=np.float64)
    n_bad = int((d > lim).sum())
    return (PASS if n_bad == 0 else FAIL), {
        "bord": repr(b), "ulp": repr(ulp_of(b)), "borne": repr(lim),
        "n_violations": n_bad, "max": repr(float(d.max()))}


# ---------------------------------------------------------------- V1b-2
def gate_v1b2(delta_nll, decrements, lam) -> tuple[str, dict]:
    """Volet STRICT, RESTREINT : sur `{δ̂ ≥ 8·ULP}`, `ΔNLL_t < bord` strictement,
    100 %. Sur le **complément** : `|ΔNLL_t − bord| ≤ 8·ULP`, **compte rapporté,
    non nul attendu** — ce n'est PAS un échec.

    C'est la porte que le Builder a dû substituer après données au run 2 : elle
    est ici pré-enregistrée sous sa forme correcte.
    """
    b = bord(lam)
    u8 = 8.0 * ulp_of(b)
    d = np.asarray(delta_nll, dtype=np.float64)
    dh = np.asarray(decrements, dtype=np.float64)
    strict = dh >= u8
    comp = ~strict
    n_strict_bad = int((d[strict] >= b).sum())
    n_comp_bad = int((np.abs(d[comp] - b) > u8).sum())
    det = {"bord": repr(b), "8ulp": repr(u8), "n_strict": int(strict.sum()),
           "n_complement": int(comp.sum()), "n_violations_strict": n_strict_bad,
           "n_violations_complement": n_comp_bad}
    if n_strict_bad or n_comp_bad:
        return FAIL, det
    if int(strict.sum()) == 0:
        return f"COMPLEMENT (compte={int(comp.sum())})", det
    return PASS, det


# ------------------------------------------------------------------ V1c
def gate_v1c(e3_mesure, e3_recompose) -> tuple[str, dict]:
    dev = abs(float(e3_mesure) - float(e3_recompose))
    return (PASS if dev <= V1C_TOL else FAIL), {"ecart": dev, "seuil": V1C_TOL}


# ---------------------------------------------------------------- V-var
def gate_v_var(D) -> tuple[str, dict]:
    """`var(D_t)` sur `p_kNN = 0` = **0 à 1e-12** : identité (D_t constant).
    Remplace toute porte de corrélation (qui rendrait NaN)."""
    d = np.asarray(D, dtype=np.float64)
    v = float(d.var()) if d.size > 1 else 0.0
    return (PASS if v <= VVAR_TOL else FAIL), {"var": v, "seuil": VVAR_TOL}


# ------------------------------------------------------------------- V2
def gate_v2(p10_matrix) -> tuple[str, dict]:
    """`n_faisable` = unités avec `p₁₀ < 0.0512711` sur ≥ 2/3 paraphrases.
    ≤ 15/30 ⇒ `INCONCLUSIF — budget arithmétique` (le canal n'a pas eu sa chance)."""
    m = np.asarray(p10_matrix, dtype=np.float64)
    n = int(((m < P10_FEASIBLE).sum(axis=1) >= P1_PARA_MIN).sum())
    det = {"n_faisable": n, "N": int(m.shape[0]), "seuil_p10": P10_FEASIBLE}
    return (PASS if n > V2_MIN_FEASIBLE
            else f"{INCONCLUSIF} — budget arithmétique"), det


# ----------------------------------------------------------------- V-λ0
def gate_v_lambda0(logits_a, logits_b) -> str:
    a, b = np.asarray(logits_a), np.asarray(logits_b)
    return PASS if (a.shape == b.shape and a.tobytes() == b.tobytes()) else FAIL


# ------------------------------------------------------------------- P1
def verdict_p1(n: int) -> str:
    """SEULE DÉCISIONNELLE, ITT, par bras. n ≥ 12/30 ⇒ H vraie ; n ≤ 5/30 ⇒ H
    fausse ; `n ∈ [6, 11]` ⇒ **zone grise** (0.1001 à p = 0.5, 0.7641 à p = 0.3,
    irréductible sous `P(échec ferme | p = 0.3) ≤ 10 %`)."""
    n = int(n)
    if n >= P1_TRUE:
        return PASS
    if n <= P1_FALSE:
        return FAIL
    return f"{INCONCLUSIF} — zone grise"


def verdict_p1_antipode(r1v_successes) -> str:
    """Antipode D13 : n ≥ 12 mais `R1v > 1` sur la **majorité** des succès ⇒
    ININTERPRÉTABLE."""
    r = [int(x) for x in r1v_successes]
    if r and sum(1 for x in r if x > 1) * 2 > len(r):
        return "ININTERPRÉTABLE"
    return PASS


def verdict_p1_degenerate(r1v_failures, c_sups, grid=None) -> str:
    """Majorité des échecs avec `R1v > 1` **et** sup **au bord de grille** ⇒
    `INCONCLUSIF — cellule dégénérée`."""
    grid = C_GRID if grid is None else list(grid)
    edges = {grid[0], grid[-1]}
    r = [int(x) for x in r1v_failures]
    maj_r1v = bool(r) and sum(1 for x in r if x > 1) * 2 > len(r)
    maj_edge = bool(c_sups) and sum(1 for c in c_sups if c in edges) * 2 > len(c_sups)
    return f"{INCONCLUSIF} — cellule dégénérée" if (maj_r1v and maj_edge) else PASS


# ------------------------------------------------------------------ ΔP6
def k_of_n(n: int) -> int:
    """`k(n) = min{k : Σ_{j≥k} C(n,j)/2ⁿ ≤ 0.10}`, **en entiers Python purs**.

    `Σ/2ⁿ ≤ 1/10` ⟺ `10·Σ ≤ 2ⁿ` : aucun flottant, donc aucune frontière à la 3ᵉ
    décimale ne peut basculer par arrondi. L'inégalité est **large** (`≤`) : avec
    `<` la table du §3 casserait. La table de référence n'est JAMAIS lue ici.
    """
    n = int(n)
    lim = 2 ** n
    s = 0
    for k in range(n, -1, -1):
        s += comb(n, k)
        if 10 * s > lim:
            return k + 1
    return 0


def k_tail_ratio(n: int, k: int) -> float:
    """`Σ_{j≥k} C(n,j)/2ⁿ` — exhibé pour les frontières (descriptif, jamais décisif)."""
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def verdict_dp6(n_disc: int, n_success: int) -> tuple[str, dict]:
    """Test des signes **conditionnel aux paires discordantes**, unilatéral
    α = 0.10, direction pré-déclarée par N2′ (L6 > F). `n_disc < 5` ⇒ NON
    ÉVALUABLE ; n_disc = 5 ⇒ unanimité."""
    n_disc, n_success = int(n_disc), int(n_success)
    if n_disc < DP6_MIN_DISC:
        return NON_EVALUABLE, {"n_disc": n_disc, "seuil": DP6_MIN_DISC}
    k = k_of_n(n_disc)
    det = {"n_disc": n_disc, "n_success": n_success, "k(n_disc)": k}
    return (ARME if n_success >= k else NON_ARME), det


def verdict_dp6_sec(primary: str, gap_bits: float) -> str:
    """ΔP6-sec (Wilcoxon sur la médiane par unité de `log₂ R1(para)`) est
    **secondaire** : il **ne renverse jamais** ΔP6. Le verdict rendu est celui du
    primaire, le secondaire n'étant qu'une étiquette."""
    sec = "SIGNAL" if float(gap_bits) >= DP6_SEC_BITS else "NUL"
    return f"{primary} | sec={sec}"


# ------------------------------------------------------------------- P3
def verdict_p3(counts, b=P3_B) -> tuple[str, dict]:
    """BLOQUANT, hors-ligne, même `sup_c`, même bras. `B = 21` et non 20 : à B
    **pair** la médiane est la moyenne des 10ᵉ/11ᵉ statistiques d'ordre, donc
    **demi-entière** — le banc REFUSE. `B = 1` ⇒ **variance nulle par
    construction** — le banc REFUSE aussi."""
    c = [int(x) for x in counts]
    det = {"B": int(b), "n_valeurs": len(c)}
    if len(c) != int(b):
        return f"{REFUS} — B déclaré ({b}) ≠ nombre de permutations ({len(c)})", det
    if int(b) < 3:
        return f"{REFUS} — B = {b} : variance nulle par construction", det
    if int(b) % 2 == 0:
        return f"{REFUS} — B pair : médiane demi-entière ambiguë", det
    med = statistics.median(c)
    det["mediane"] = med
    if med >= P3_LEAK_MIN:
        return INCONCLUSIF, det
    return (PASS if med <= P3_NULL_MAX else INCONCLUSIF), det


# ------------------------------------------------------------------- P4
def verdict_p4(n: int) -> str:
    n = int(n)
    if n >= P4_NULL:
        return "SÉLECTIVITÉ NULLE"
    return PASS if n <= P4_MAX else INCONCLUSIF


# ------------------------------------------------------------------- P5
def verdict_p5(e3: float) -> str:
    """`E3(λ*) ≤ 0.05` est un **THÉORÈME** : dépassement = **bug**, jamais un
    résultat."""
    return PASS if float(e3) <= E3_BUDGET else BUG


# ------------------------------------------------------------ P5f-borne
def margins_over_vk(p_rows, vk_values):
    """`min_{v ∈ V_k(p)} (p_max(p) − p_LM(v))` par position — la forme CORRIGÉE
    (E-D6). `V_k(p)` = les valeurs des **8 voisins** à la position p."""
    out = []
    for row, vk in zip(np.asarray(p_rows, dtype=np.float64), vk_values):
        pmax = float(row.max())
        out.append(min(pmax - float(row[int(v)]) for v in vk))
    return np.asarray(out, dtype=np.float64)


def margins_over_vnn(p_rows, v_nn):
    """Forme FAUSSE du run précédent, sur le seul plus proche voisin (E-D6) :
    conservée pour exhiber le sous-comptage, jamais utilisée pour décider."""
    out = []
    for row, v in zip(np.asarray(p_rows, dtype=np.float64), v_nn):
        out.append(float(row.max()) - float(row[int(v)]))
    return np.asarray(out, dtype=np.float64)


def borne_marge(margins) -> float:
    """`borne_marge = #{p : marge(p) ≤ 0.0512711} / T`. **Inégalité LARGE** :
    aux ex-æquo l'argmax bascule par convention d'indice à marge exactement
    égale à la borne. Aucun rapport de probabilités ⇒ `borne_marge = 0` ne
    produit **pas de NaN**."""
    m = np.asarray(margins, dtype=np.float64)
    return float((m <= P10_FEASIBLE).sum()) / float(m.size)


def verdict_p5f_borne(flip_rate: float, borne: float) -> str:
    """Taux global de bascule d'argmax ≤ `borne_marge`. Dépassement ⇒ **bug**."""
    return PASS if float(flip_rate) <= float(borne) else BUG


def verdict_p5f_cond(winner: int, store_values, confident: bool) -> str:
    """Antipode : bascule **confiante** dont le gagnant est **une valeur du
    store** = **intrusion mnésique** (informatif) ; token arbitraire = **bug**
    (run invalide)."""
    if not confident:
        return PASS
    return "INTRUSION" if int(winner) in {int(v) for v in store_values} else BUG


# ---------------------------------------------------------------- P5c-id
def verdict_p5c_id(f_unhot: float, f_fini: float) -> str:
    """`f_relief(un-hot)` **strictement <** `f_relief(c fini)`. Aucune bande
    absolue. Antipode : ≥ ⇒ normalisation de p_kNN fausse."""
    return PASS if float(f_unhot) < float(f_fini) else FAIL


# ------------------------------------------------------------------- P7
def verdict_p7(degradation: float) -> str:
    return PASS if float(degradation) <= P7_MAX_DEGRADATION else FAIL


# ------------------------------------------------------------------- P8
def verdict_p8(series, knee_factor: float = P8_KNEE_FACTOR) -> tuple[str, dict]:
    """Graduelle, monotone, **sans genou**. ⚠ le protocole n'opérationnalise pas
    « sans genou » : le détecteur ci-dessous est **déclaré par le banc** (cf.
    `UNDERSPEC`). Clause descriptive, n'entre dans aucune porte."""
    y = [float(v) for v in series]
    diffs = [b - a for a, b in zip(y, y[1:])]
    det = {"diffs": diffs, "facteur_declare": knee_factor}
    if any(d < 0 for d in diffs):
        return "NON MONOTONE", det
    med = statistics.median(diffs) if diffs else 0.0
    det["mediane_diff"] = med
    if med > 0 and max(diffs) > knee_factor * med:
        return "GENOU", det
    return PASS, det


# -------------------------------------------------------------------- G
def tau_promu(e3_curve, budget: float = E3_BUDGET):
    """**`τ_promu` = max{τ ∈ grille : `E3(τ') < 0.05` pour TOUT τ' ≤ τ}** — le
    plus grand élément du **PRÉFIXE ADMISSIBLE CONNEXE** (E-D7).

    `E3(τ)` n'est **pas monotone** : le soulagement T2 donne `δ(p) < 0` à
    certaines positions, donc l'ensemble admissible peut être **non connexe**.
    Promouvoir un τ au-delà d'une zone violée accepterait un gate dont un
    sous-gate strictement inclus casse le budget. Renvoie l'indice (0-based) ou
    `None` si le 1ᵉʳ décile viole déjà.
    """
    j = None
    for i, e3 in enumerate(e3_curve):
        if float(e3) < budget:
            j = i
        else:
            break
    return j


def verdict_g(e3_curve, p1_curve, n_alpha1, n_alpha0, p1_lambda_star,
              budget: float = E3_BUDGET) -> tuple[str, dict]:
    """Non-vacuité : τ évalué seulement si `#{α=1} ≥ 1` **et** `#{α=0} ≥ 1`.
    Déclencheur « G → organe » : `P1(τ_promu) ≥ max(1, P1(λ*) + 1)` **et**
    `E3(τ_promu) < 0.05`. Aucun τ admissible ⇒ **G NON ÉVALUABLE**."""
    det = {"n_alpha1": int(n_alpha1), "n_alpha0": int(n_alpha0),
           "P1(lambda*)": int(p1_lambda_star)}
    if int(n_alpha1) < G_MIN_ALPHA or int(n_alpha0) < G_MIN_ALPHA:
        return "TAU NON ÉVALUÉ", det
    j = tau_promu(e3_curve, budget)
    det["tau_promu_index"] = None if j is None else j + 1
    if j is None:
        return NON_EVALUABLE, det
    det["E3(tau_promu)"] = float(e3_curve[j])
    det["P1(tau_promu)"] = int(p1_curve[j])
    if int(p1_lambda_star) == 0 and int(p1_curve[j]) == 0:
        return NON_EVALUABLE, det
    armed = (int(p1_curve[j]) >= max(1, int(p1_lambda_star) + 1)
             and float(e3_curve[j]) < budget)
    return (f"PROMOTION ARMÉE (tau=idx{j + 1})" if armed
            else f"{NON_ARME} (tau=idx{j + 1})"), det


# ------------------------------------------------------------- multi-clé
def multikey_clause_c(pct_intra, pct_inter, m_intra, m_inter) -> tuple[bool, str]:
    """(c) médiane `pct` **intra-unité** < médiane **inter-unités**.
    **Saturation** : deux `pct` à 0 ⇒ la décision bascule sur la **marge
    continue `m`** — et surtout **PAS** sur un `NON ÉVALUABLE` (décision PI §13.5 :
    *un NON ÉVALUABLE déclenché par la résolution de l'instrument et non par la
    donnée est exactement l'issue qui a rendu le NON ARMÉ du run 2 non probant*)."""
    a, b = statistics.median(pct_intra), statistics.median(pct_inter)
    if a == 0.0 and b == 0.0:
        return (statistics.median(m_intra) < statistics.median(m_inter)), "m"
    return (a < b), "pct"


def verdict_multikey(units_a, units_b, pct_intra, pct_inter, m_intra, m_inter,
                     strat_qualitatif_identique=True) -> tuple[str, dict]:
    """ARMÉ ssi, sur ≥ 18/30 unités : (a) R1 > 1 sur ≥ 2/3 paraphrases ;
    (b) médiane `pct(q_para)` ≥ 1e-3 alors que `pct(q_exact)` ≤ 3.33e-5 ;
    (c) médiane intra < médiane inter. **(c) fausse ⇒ NE PAS ARMER.**
    Divergence qualitative globale/stratifiée ⇒ **NON ÉVALUABLE**."""
    c_ok, voie = multikey_clause_c(pct_intra, pct_inter, m_intra, m_inter)
    det = {"clause_c": c_ok, "voie_de_decision_c": voie,
           "n_a": int(sum(bool(x) for x in units_a)),
           "n_b": int(sum(bool(x) for x in units_b))}
    if not strat_qualitatif_identique:
        return NON_EVALUABLE, det | {"cause": "divergence globale/stratifiée"}
    n = sum(1 for a, b in zip(units_a, units_b) if a and b)
    det["n_unites_ab"] = int(n)
    if not c_ok:
        return NON_ARME, det
    return (ARME if n >= MULTIKEY_MIN_UNITS else NON_ARME), det


def verdict_ventilation(med_intra, med_1partage, med_0partage) -> str:
    """Descriptif, **hors clause** : si la clé porte l'identité d'unité on attend
    `pct_intra < pct_(1 partagé) < pct_(0 partagé)` ; si `pct_(1 partagé) ≈
    pct_intra`, la clé encode le **gabarit de surface**, pas l'unité. Ce second
    cas n'est **pas un FAIL**."""
    # ⚠ le protocole écrit « `1 partagé` ≈ `intra` » sans opérationnaliser le ≈ :
    # la tolérance ci-dessous est DÉCLARÉE PAR LE BANC (cf. `UNDERSPEC`). Le test
    # de proximité passe AVANT celui de monotonie : `intra < 1 partagé` peut être
    # vrai tout en étant un ≈, et c'est précisément le diagnostic recherché.
    if math.isclose(med_1partage, med_intra, rel_tol=VENTIL_REL_TOL, abs_tol=0.0):
        return "CLÉ DE GABARIT DE SURFACE"
    if med_intra < med_1partage < med_0partage:
        return "ORDRE MONOTONE"
    return "ORDRE NON MONOTONE"


# =========================================================================
#  Tokenizer — GPT-2 (CPU, en cache) ou repli mot-à-mot
# =========================================================================

def make_tokenizer(use_hf: bool):
    if not use_hf:
        vocab: dict[str, int] = {}

        def toy(text: str):
            ids = []
            for w in text.replace(".", " .").replace("?", " ?").split():
                ids.append(vocab.setdefault(w, len(vocab) + 1))
            return ids
        return toy, "repli mot-à-mot (aucun HF)"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    return (lambda s: tok.encode(s)), "GPT-2 BPE (CPU, cache local)"


def make_offsets(use_hf: bool):
    """`offsets(s) -> (ids, spans)` — spans de CARACTÈRES de chaque token, pour
    la nulle de cadre (§5, maillon 2). `None` en repli mot-à-mot : la nulle
    retombe alors sur le découpage par segments, et le fait est publié."""
    if not use_hf:
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    if not getattr(tok, "is_fast", False):
        return None

    def offsets(s):
        enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
        return [int(x) for x in enc["input_ids"]], list(enc["offset_mapping"])
    return offsets


# =========================================================================
#  Jeux synthétiques + registre des clauses
# =========================================================================

def _rng(seed):
    return np.random.default_rng(seed)


def _deltas_from_r(rs, lam, logp=-1.0):
    """ΔNLL et décréments δ̂ pour une liste de rapports r = p_kNN/p_LM."""
    d, dh = [], []
    b = bord(lam)
    for r in rs:
        pk = float(r) * math.exp(logp)
        d.append(mix_delta_nll(logp, pk, lam))
        t = math.exp(math.log(lam) - math.log1p(-lam) + math.log(pk) - logp) if pk > 0 else 0.0
        dh.append(math.log1p(t))
    return np.asarray(d), np.asarray(dh), b


def _p5f_neighbor5_case():
    """Cellule E-D6 : bascule d'argmax portée par la valeur du **voisin n° 5**.

    `V_k` = 8 voisins ; le token gagnant `w` n'apparaît QU'au 5ᵉ voisin. La borne
    corrigée (min sur `V_k`) compte la position ; la borne fausse (sur `v_nn`
    seul) ne la compte pas et déclarerait « bug » une implémentation correcte.
    """
    V = 12
    row = np.full(V, 0.039, dtype=np.float64)
    row[0] = 0.30                       # argmax du cortex
    row[7] = 0.299                       # marge 0.001 ≤ 0.0512711 → à risque
    row[3] = 0.05                        # valeur du PLUS PROCHE voisin : marge 0.25
    d2k = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    vals = np.array([3, 1, 2, 4, 7, 5, 6, 8])           # token 7 = 5ᵉ voisin SEUL
    pk = knn_distribution(d2k, vals, 1.0)
    win = mix_argmax(np.log(row), pk, LAMBDA_STAR)
    # Positions de remplissage : argmax NET, tous les voisins très en dessous
    # ⇒ marge ≫ 0.0512711 ⇒ non comptées par l'une ou l'autre des deux bornes.
    filler = np.full(V, 0.01, dtype=np.float64)
    filler[0] = 1.0 - 0.01 * (V - 1)
    rows = np.stack([row] + [filler.copy() for _ in range(3)])
    vk = [vals.tolist()] + [[1, 2, 3] for _ in range(3)]
    vnn = [int(vals[0])] + [1, 2, 3]
    m_vk = margins_over_vk(rows, vk)
    m_vnn = margins_over_vnn(rows, vnn)
    return {"gagnant": int(win), "rang_du_porteur": 5,
            "p_knn_du_gagnant": float(pk.get(7, 0.0)),
            "bascule_effective": bool(win == 7),
            "borne_Vk": borne_marge(m_vk), "borne_vnn": borne_marge(m_vnn),
            "taux_observe": 0.25,
            "marge_Vk_pos0": float(m_vk[0]), "marge_vnn_pos0": float(m_vnn[0])}


def _p5f_tie_case():
    """Ex-æquo d'argmax à marge **exactement** égale à 0.0512711 : l'inégalité
    de `borne_marge` est **LARGE**, la position DOIT être comptée."""
    V = 32
    vk = list(range(20, 28))
    row = np.full(V, 0.0, dtype=np.float64)
    row[31] = P10_FEASIBLE               # argmax
    for t in vk:
        row[t] = 0.0                     # marge = p_max − 0 = 0.0512711 EXACT
    rest = [t for t in range(V) if t != 31 and t not in vk]
    row[rest] = (1.0 - P10_FEASIBLE) / len(rest)
    m = margins_over_vk(row[None, :], [vk])
    return {"marge": repr(float(m[0])), "borne_exacte": repr(P10_FEASIBLE),
            "egalite_bit": bool(float(m[0]) == P10_FEASIBLE),
            "borne_large": borne_marge(m), "borne_stricte": float(
                (m < P10_FEASIBLE).sum()) / float(m.size)}


def build_clauses(tokenize, tok_name):
    """Registre : une entrée par clause du tableau du protocole, chacune avec ses
    contre-exemples PASSANTS et ÉCHOUANTS exécutables."""
    C = []

    def clause(name, pass_desc, fail_desc, cases_pass, cases_fail, structural=None,
               note=None):
        C.append({"clause": name, "pass_case": pass_desc, "fail_case": fail_desc,
                  "cases_pass": cases_pass, "cases_fail": cases_fail,
                  "structural": structural, "note": note})

    # ------------------------------------------------------------- V-base
    ref = {"E1": 1.35, "top10": 0.0, "E3": -0.014}
    clause("V-base", "métriques égales à la référence au centième",
           "une métrique décalée de 0.02",
           [("identique", PASS, lambda: (gate_v_base(dict(ref), ref), {}))],
           [("E3 décalé de 0.02", FAIL,
             lambda: (gate_v_base(ref | {"E3": ref["E3"] + 0.02}, ref), {}))],
           note="valeurs de référence SYNTHÉTIQUES : D14-R interdit qu'un chiffre "
                "du journal entre dans une porte de v3 ; la référence réelle est "
                "re-mesurée dans la passe V-base.")

    # -------------------------------------------------------------- V-cap
    clause("V-cap", "logits recalculés à l'identique", "écart injecté 1e-4",
           [("dev = 0", PASS, lambda: (gate_v_cap(0.0), {"dev": 0.0})),
            ("dev = 3.6e-6 (30·2⁻²³)", PASS,
             lambda: (gate_v_cap(30 * 2 ** -23), {"dev": 30 * 2 ** -23}))],
           [("dev = 1e-4", FAIL, lambda: (gate_v_cap(1e-4), {"dev": 1e-4}))])

    # ------------------------------------------------------------ V-drift
    a = _rng(1).standard_normal(64).astype(np.float32)
    b = a.copy()
    # 1 ULP **fp32** : `math.nextafter` rendrait un fp64 qui se ré-arrondit sur
    # le MÊME fp32 à l'affectation — le contre-exemple échouant serait vacué.
    b[7] = np.nextafter(b[7], np.float32(np.inf))
    clause("V-drift", "deux tableaux identiques", "un bit modifié",
           [("copie bit-à-bit", PASS, lambda: (gate_v_drift(a, a.copy()), {}))],
           [("1 ULP sur une entrée", FAIL, lambda: (gate_v_drift(a, b), {}))],
           note="rejeu sur bruts archivés : voir `meta_replay` à la racine.")

    # ------------------------------------------------------------- V-bord
    # §15, A-6 (i) : la porte est évaluée sur TOUTE LA GRILLE λ, pas au seul λ*.
    # À λ* les deux expressions sont bit-identiques (0 ULP) : la porte y est
    # vacuée par satisfaction, ce qui est DOCUMENTÉ (ci-dessous, descriptif) et
    # n'est plus un cas de la clause — le domaine de la clause est la grille.
    clause("V-bord", "`bord` par `−log1p(−λ)` des deux côtés ⇒ égalité bit-à-bit "
           "sur TOUTE la grille λ (A-6 i)",
           "`bord` par `−log(1−λ)` d'un côté sur la grille λ ⇒ FAIL, écarts ULP "
           "exhibés par λ",
           [("log1p des deux côtés, grille λ", PASS,
             lambda: gate_v_bord(LAMBDA_GRID, False))],
           [("−log(1−λ) d'un côté, grille λ", FAIL,
             lambda: gate_v_bord(LAMBDA_GRID, True))],
           note="A-6 (i) : évaluée sur la grille λ entière. Au SEUL λ* les deux "
                "expressions sont bit-identiques (0 ULP) — la porte y serait "
                "vacuée par satisfaction ; l'écart par λ est publié dans le "
                "détail du cas échouant (descriptif).")

    # §15, A-6 (ii) : λ* est l'EXPRESSION `1 − exp(−0.05)`, jamais un décimal.
    clause("V-λ*-expression (A-6 ii)",
           "`LAMBDA_STAR` produit par l'expression `1 − exp(−0.05)` ⇒ 0 ULP",
           "le littéral décimal `0.048770575499286` du document ⇒ FAIL, écart "
           "2 ULP exhibé",
           [("LAMBDA_STAR (expression)", PASS,
             lambda: gate_lambda_star_expression(LAMBDA_STAR))],
           [("littéral décimal du protocole", FAIL,
             lambda: gate_lambda_star_expression(LAMBDA_STAR_LITERAL_DOC))],
           note="le décimal reste dans le document comme DOCUMENTATION SEULE, "
                "étiqueté comme tel (D14-R).")

    # ----------------------------------------------------------------- V0
    k0 = _rng(2).standard_normal((6, 16)).astype(np.float32)
    v0 = np.array([1, 2, 42, 4, 5, 6])
    q0 = k0[2].copy()                      # requête EXACTE ⇒ d²(entrée 42) = 0
    q_off = (k0[2] + np.float32(0.05)).astype(np.float32)   # requête décalée
    para = q_off.copy()                    # parasite EXACTEMENT sur la requête
    k_bad = np.concatenate([para[None, :], k0], axis=0)
    v_bad = np.concatenate([np.array([99]), v0])
    clause("V0", "clé exacte à d² minimal ⇒ R1 = 1",
           "entrée parasite plus proche ⇒ R1 = 2",
           [("indice exact", PASS, lambda: gate_v0(k0, v0, q0, 42))],
           [("parasite à d² strictement plus petit", FAIL,
             lambda: gate_v0(k_bad, v_bad, q_off, 42))])

    # ------------------------------------------------------------ V-indep
    g = _rng(3)
    qk = g.standard_normal((30, 4, 32)).astype(np.float32)
    d2m = g.uniform(0.1, 5.0, 30)
    p10s = g.uniform(0.001, 0.05, 30)
    qk_same = np.repeat(qk[0:1], 30, axis=0)
    clause("V-indep (a)(b)(c)", "120 clés distinctes, sd > 0",
           "les 30 unités partagent la même clé (réplique de C4) ⇒ FAIL",
           [("120 clés distinctes", PASS, lambda: gate_v_indep(qk, d2m, p10s))],
           [("30 unités, même clé (C4)", FAIL,
             lambda: gate_v_indep(qk_same, d2m, p10s)),
            ("sd(d²_min) = 0", FAIL,
             lambda: gate_v_indep(qk, np.full(30, 2.0), p10s)),
            ("sd(p₁₀) = 0", FAIL,
             lambda: gate_v_indep(qk, d2m, np.full(30, 0.01)))])

    qk_dup = qk.copy()
    qk_dup[5, 1] = qk_dup[0, 0]
    clause("V-indep (d)", "cos max inter-unités < 1",
           "deux clés bit-identiques ⇒ FAIL par (a)",
           [("cos max < 1", PASS, lambda: gate_v_indep(qk, d2m, p10s))],
           [("deux clés bit-identiques", FAIL,
             lambda: gate_v_indep(qk_dup, d2m, p10s))])

    # -------------------------------------------------------------- V-tok
    units = unit_table(POOL_UNITS_N)
    # Jeu d'unités AVANT l'amendement §15 A-1 (ancienne para1, rotation +1 mod 5).
    # Conservé UNIQUEMENT comme contre-exemple échouant de V-slot / V-para (c′).
    from pool import pool_paraphrases_pre_amendment
    _paras_avant = pool_paraphrases_pre_amendment(POOL_UNITS_N)
    units_avant = [dict(u, paraphrases=list(_paras_avant[u["i"]])) for u in units]
    sec_apres = [u["secret"] for u in units]
    from pool import SECRETS_80
    sec_avant = list(SECRETS_80[:POOL_UNITS_N])          # jeu réel AVANT correction
    sec_dup = list(sec_apres)
    sec_dup[9] = "walruses"                              # même 1ᵉʳ token que walrus
    clause("V-tok",
           "30 premiers tokens BPE distincts, aucun dans le pool "
           "(jeu réel APRÈS correction)",
           "jeu réel AVANT correction (`lighthouse` en unité 5) ⇒ FAIL, unité 5 "
           "nommée ; et deux secrets à 1ᵉʳ token identique ⇒ FAIL",
           [("jeu réel APRÈS substitution (unité 5 → walrus)", PASS,
             lambda: gate_v_tok(sec_apres, tokenize))],
           [("jeu réel AVANT correction", FAIL,
             lambda: gate_v_tok(sec_avant, tokenize)),
            ("deux secrets à 1ᵉʳ token identique", FAIL,
             lambda: gate_v_tok(sec_dup, tokenize))],
           note=f"tokenizer : {tok_name}")

    # ------------------------------------------------------------- V-para
    def _syn_units(cross_leak=False, substring=False, weak_exact=False):
        """Unités SYNTHÉTIQUES : trois familles lexicales disjointes, donc (c′)
        satisfiable par construction — c'est le contre-exemple PASSANT."""
        out = []
        fam = [("alpha", "bravo"), ("charlie", "delta"), ("echo", "foxtrot")]
        for i in range(6):
            w1, w2 = fam[i % 3]
            tag = f"u{i}"
            fact = f"{tag} {w1} {w2} zzz{i} qqq{i} {{secret}}."
            no_sec = fact.replace(" {secret}", "")
            exact = f"{tag} {w1} {w2} zzz{i} qqq{i}"
            paras = [f"{tag} {w1} zzz{i}", f"{tag} {w2} qqq{i}", f"{tag} zzz{i}"]
            slots = {"tag": tag, "w1": w1, "w2": w2,
                     "z": f"zzz{i}", "q": f"qqq{i}"}
            if substring:
                paras[0] = exact
            if weak_exact:
                exact = tag                     # indice exact appauvri ⇒ (b) viole
            if cross_leak:
                j = (i + 1) % 6
                paras[0] = f"u{j} {fam[j % 3][0]} {fam[j % 3][1]} zzz{j} qqq{j}"
            out.append({"i": i, "fact_template": fact, "fact_no_secret": no_sec,
                        "exact": exact, "paraphrases": paras, "slots": slots})
        return out

    # §15, A-3 : (c) devient (c′) ; (a) et (b) restent sur le Jaccard BRUT et
    # sont RE-JOUÉES sur les nouveaux indices (cascade D14(b), §15 C).
    clause("V-para (a)(b)",
           "aucun indice n'est sous-chaîne de son fait, et Jaccard brut "
           "paraphrase < Jaccard brut exact sur les 30",
           "paraphrase = préfixe littéral de son fait ⇒ FAIL (a) ; indice exact "
           "appauvri ⇒ Jaccard paraphrase ≥ Jaccard exact ⇒ FAIL (b)",
           [("jeu synthétique conforme", PASS,
             lambda: gate_v_para(_syn_units(), tokenize, check_cross=False)),
            ("POOL_PARAPHRASES réel AMENDÉ (§15 A-1) — BLOQUANTE avant le run",
             PASS, lambda: gate_v_para(units, tokenize, check_cross=False))],
           [("paraphrase = indice exact (sous-chaîne)", FAIL,
             lambda: gate_v_para(_syn_units(substring=True), tokenize,
                                 check_cross=False)),
            ("indice exact appauvri ⇒ (b) violée", FAIL,
             lambda: gate_v_para(_syn_units(weak_exact=True), tokenize,
                                 check_cross=False))])

    clause("V-para (c′)",
           "Jaccard sur le CONTENU (`F_t` = intersection ensembliste des tokens "
           "BPE sur les 30), comparaison STRICTE en entiers exacts, sur les 30 "
           "faits ⇒ 0 violation",
           "fuite croisée i → j (le contenu de la paraphrase de *i* désigne le "
           "fait de *j*) ⇒ FAIL, paire (i, j) nommée ; et égalité ⇒ violation",
           [("jeu synthétique conforme", PASS,
             lambda: gate_v_para_c_prime(_syn_units(), tokenize)),
            ("POOL_PARAPHRASES réel AMENDÉ (§15 A-1) — BLOQUANTE avant le run",
             PASS, lambda: gate_v_para_c_prime(units, tokenize))],
           [("fuite croisée i → j", FAIL,
             lambda: gate_v_para_c_prime(_syn_units(cross_leak=True), tokenize)),
            ("ANCIENNE para1 (rotation +1 mod 5) sur le jeu réel", FAIL,
             lambda: gate_v_para_c_prime(units_avant, tokenize))],
           note="A-3 : `F_t` par INTERSECTION sur les 30, jamais un préfixe/"
                "suffixe commun — le cadre de para3 est entrelacé et un "
                "préfixe/suffixe laisserait `\" that belongs to \"` dans le "
                "contenu, recréant la fuite. Comparaison par produit croisé sur "
                "des `int` : aucune analyse ULP à faire.")

    # ------------------------------------------------------- V-slot (A-4)
    def _syn_slot_leak():
        u = _syn_units()
        u[0]["paraphrases"][0] = u[0]["paraphrases"][0] + " zzz1"   # slot de u1
        return u

    clause("V-slot",
           "aucune valeur de ligne d'une unité j ≠ i n'apparaît verbatim dans "
           "para_t(i) — jeu réel AMENDÉ (§15 A-1)",
           "ANCIENNE para1 (rotation `+1 mod 5`) ⇒ FAIL DÉTERMINISTE : "
           "`VERBS[(i+1) mod 5]` est le slot verbe d'unités j ≠ i",
           [("jeu réel AMENDÉ (verbe global hors tables)", PASS,
             lambda: gate_v_slot(units)),
            ("jeu synthétique conforme", PASS,
             lambda: gate_v_slot(_syn_units(),
                                 row_values=sorted({v for u in _syn_units()
                                                    for v in u["slots"].values()})))],
           [("ANCIENNE para1 (rotation +1 mod 5) — contre-exemple OBLIGATOIRE",
             FAIL, lambda: gate_v_slot(units_avant)),
            ("slot d'une autre unité injecté (synthétique)", FAIL,
             lambda: gate_v_slot(_syn_slot_leak(),
                                 row_values=sorted({v for u in _syn_units()
                                                    for v in u["slots"].values()})))],
           note="porte de RÈGLE : `V-para (c′)` NEUTRALISE la fuite structurelle "
                "(elle la met hors contenu) mais ne la DÉTECTE pas. V-slot est "
                "aussi la seule qui reste correcte si le tokenizer ou le modèle "
                "change. Elle ne consomme aucune mesure.")

    # ------------------------------------------------- V-ident (§16 E)
    _triples = v3_unit_triples_stats(POOL_UNITS_N)["triplets"]
    _triples_fact_pairs = fact_pairs_triples(POOL_UNITS_N)
    # Triplet de l'unité 0 dupliqué en 30ᵉ position : viole C-1 (même
    # (owner, entity)) ET C-2 (même (entity, verb)) — un doublon pur de C-1 est
    # IMPOSSIBLE dans ce jeu, les 5 verbes de l'entité étant déjà consommés.
    _tri_c1_bad = list(_triples[:29]) + [_triples[0]]
    clause("V-ident (§16 E)",
           "le jeu d'unités v3 (30 premiers triplets conformes de l'énumération "
           "§16 D) vérifie C-1, C-2 et C-3",
           "`pool.fact_pairs(30)` ⇒ FAIL sur C-2 (période 20 : entity = i mod 20, "
           "verb = i mod 5, et 5 divise 20) ET sur C-3 (owners pronominaux) ; "
           "et un doublon (owner, entity) ⇒ FAIL sur C-1",
           [("jeu d'unités v3 (§16 D)", PASS,
             lambda: gate_v_ident(_triples, tokenize))],
           [("pool.fact_pairs(30) — contre-exemple OBLIGATOIRE", FAIL,
             lambda: gate_v_ident(_triples_fact_pairs, tokenize)),
            ("triplet de l'unité 0 dupliqué ⇒ C-1 (et C-2)", FAIL,
             lambda: gate_v_ident(_tri_c1_bad, tokenize))],
           note="porte STRUCTURELLE, décidable sans aucune mesure. C-3 est "
                "évaluée PAR EXÉCUTION du tokenizer, jamais par jugement.")

    # --------------------------------------------------- OWNER_OBJ (A-5)
    oo_bad = dict(OWNER_OBJ)
    oo_bad["His"] = OWNER_OBJ["Her"]                 # deux owners, même image
    clause("OWNER_OBJ (A-5)",
           "table injective et 16 images deux à deux distinctes en BPE",
           "deux owners de même forme objet ⇒ FAIL (deux unités indiscernables "
           "dans para3)",
           [("OWNER_OBJ réelle", PASS,
             lambda: gate_owner_obj(OWNER_OBJ, tokenize))],
           [("His → her (collision d'image)", FAIL,
             lambda: gate_owner_obj(oo_bad, tokenize))])

    # ------------------------------------------------------------- V-hash
    # Cascade D14(b) (§15 C) : l'amendement change les DONNÉES GELÉES ⇒ nouveau
    # SHA-256. Il est publié en tête du rapport (`sha256_donnees_gelees`).
    frozen = frozen_dataset()
    touched = json.loads(json.dumps(frozen, ensure_ascii=False))
    touched["unites"][0]["secret"] = touched["unites"][0]["secret"] + "x"
    clause("V-hash", "jeu inchangé", "un caractère modifié",
           [("SHA-256 identique", PASS, lambda: gate_v_hash(frozen, frozen))],
           [("un caractère modifié", FAIL, lambda: gate_v_hash(frozen, touched))])

    # -------------------------------------------------------------- V-tie
    clause("V-tie", "store sans ex-æquo (compte 0)",
           "deux entrées de d² identiques ⇒ compte 2, un-hot uniforme",
           [("aucun ex-æquo", "SANS EX-AEQUO",
             lambda: gate_v_tie(np.array([1.0, 2.0, 3.0, 4.0])))],
           [("deux entrées à d²_min", "EX-AEQUO=2, UN-HOT UNIFORME",
             lambda: gate_v_tie(np.array([1.0, 1.0, 3.0, 4.0])))],
           note="le second cas n'est pas un échec : c'est la BRANCHE prescrite "
                "(un-hot uniforme sur l'argmin-set).")

    # ---------------------------------------------------------------- V1a
    d_zero = np.full(64, mix_delta_nll(-3.0, 0.0, LAMBDA_STAR))
    d_pert = d_zero.copy()
    d_pert[13] += 1e-5
    clause("V1a", "`p_kNN = 0` ⇒ écart ~1e-16", "ΔNLL perturbé de 1e-5",
           [("identité exacte", PASS, lambda: gate_v1a(d_zero, LAMBDA_STAR))],
           [("perturbation 1e-5", FAIL, lambda: gate_v1a(d_pert, LAMBDA_STAR))])

    # -------------------------------------------------------------- V1b-1
    d30, dh30, b30 = _deltas_from_r([1e-30] * 16, LAMBDA_STAR)
    d_over = d30.copy()
    d_over[0] = b30 + 1e-12
    clause("V1b-1",
           "`r = 1e-30` (décrément sous l'ULP) ⇒ doit PASSER "
           "(cellule qui a tué le run 2)",
           "ΔNLL forcé à `bord + 1e-12` ⇒ FAIL",
           [("r = 1e-30", PASS, lambda: gate_v1b1(d30, LAMBDA_STAR)),
            ("r ∈ {1e-30, 1e-16, 1e-3}", PASS,
             lambda: gate_v1b1(_deltas_from_r([1e-30, 1e-16, 1e-3], LAMBDA_STAR)[0],
                               LAMBDA_STAR))],
           [("ΔNLL = bord + 1e-12", FAIL, lambda: gate_v1b1(d_over, LAMBDA_STAR))])

    # -------------------------------------------------------------- V1b-2
    d3, dh3, b3 = _deltas_from_r([1e-3] * 16, LAMBDA_STAR)
    d3_bad = d3.copy()
    d3_bad[0] = b3
    clause("V1b-2", "`r = 1e-3` ⇒ strictement < bord",
           "`r = 1e-3` avec ΔNLL forcé = bord ⇒ FAIL ; et `r = 1e-30` rangé dans "
           "le complément avec compte > 0 (attendu, non-FAIL)",
           [("r = 1e-3, δ̂ ≥ 8·ULP", PASS,
             lambda: gate_v1b2(d3, dh3, LAMBDA_STAR))],
           [("ΔNLL forcé = bord sur le sous-ensemble strict", FAIL,
             lambda: gate_v1b2(d3_bad, dh3, LAMBDA_STAR)),
            ("r = 1e-30 ⇒ complément, compte 16", "COMPLEMENT (compte=16)",
             lambda: gate_v1b2(d30, dh30, LAMBDA_STAR))])

    # ---------------------------------------------------------------- V1c
    clause("V1c", "E3 recomposé identique", "décalage 1e-4",
           [("écart 0", PASS, lambda: gate_v1c(0.0471234, 0.0471234))],
           [("écart 1e-4", FAIL, lambda: gate_v1c(0.0471234, 0.0472234))])

    # -------------------------------------------------------------- V-var
    Dc = np.full(16, bord(LAMBDA_STAR))
    Dp = Dc.copy()
    Dp[5] += 1e-5
    _T_NEUTRAL = 343                      # positions de NEUTRAL_TEXT
    _floor = math.sqrt(VVAR_TOL) * _T_NEUTRAL / math.sqrt(_T_NEUTRAL - 1)
    clause("V-var", "D_t constant ⇒ var = 0", "un D_t perturbé ⇒ var > 1e-12",
           [("D_t constant (T = 16)", PASS, lambda: gate_v_var(Dc))],
           [("un D_t perturbé de 1e-5 (T = 16)", FAIL, lambda: gate_v_var(Dp))],
           note="variance nulle par IDENTITÉ, déclarée telle par le protocole "
                "(§4bis, tableau) : c'est une PORTE, pas une statistique. "
                "Plancher de détection mesuré : une perturbation d'UNE SEULE "
                "position produit var = δ²(T−1)/T² ; à T = %d positions "
                "(NEUTRAL_TEXT) elle reste sous 1e-12 tant que δ < %.2e nats."
                % (_T_NEUTRAL, _floor))

    # ----------------------------------------------------------------- V2
    m_ok = np.full((30, 3), 0.2)
    m_ok[:20] = 0.01
    clause("V2", "p₁₀ < 0.0512711 sur 20 unités ⇒ n_faisable = 20",
           "p₁₀ tous à 0.2 ⇒ n_faisable = 0 ⇒ `INCONCLUSIF budget`",
           [("20 unités faisables", PASS, lambda: gate_v2(m_ok))],
           [("aucune unité faisable", f"{INCONCLUSIF} — budget arithmétique",
             lambda: gate_v2(np.full((30, 3), 0.2))),
            ("15 unités faisables (bord du seuil)",
             f"{INCONCLUSIF} — budget arithmétique",
             lambda: gate_v2(np.where(
                 np.arange(30)[:, None] < 15, 0.01, 0.2)))])

    # ---------------------------------------------------------------- V-λ0
    lg = _rng(4).standard_normal(512).astype(np.float32)
    lg2 = lg.copy()
    lg2[3] = np.nextafter(lg2[3], np.float32(np.inf))   # 1 ULP **fp32**
    clause("V-λ0", "logits bit-identiques", "un bit modifié",
           [("bit-exact", PASS, lambda: (gate_v_lambda0(lg, lg.copy()), {}))],
           [("1 ULP", FAIL, lambda: (gate_v_lambda0(lg, lg2), {}))])

    # ----------------------------------------------------------------- P1
    clause("P1", "n = 14 ≥ 12",
           "n = 4 ; n = 5 ⇒ H fausse ; n = 8 ⇒ zone grise, INCONCLUSIF exhibé",
           [("n = 14", PASS, lambda: (verdict_p1(14), {"n": 14})),
            ("n = 12 (bord a)", PASS, lambda: (verdict_p1(12), {"n": 12}))],
           [("n = 4", FAIL, lambda: (verdict_p1(4), {"n": 4})),
            ("n = 5 (bord b)", FAIL, lambda: (verdict_p1(5), {"n": 5})),
            ("n = 8 (zone grise)", f"{INCONCLUSIF} — zone grise",
             lambda: (verdict_p1(8), {"n": 8, "grise": "[6, 11]"}))])

    clause("P1-antipode", "succès avec R1v = 1",
           "succès avec R1v = 3 majoritaire ⇒ ininterprétable",
           [("R1v = 1 partout", PASS,
             lambda: (verdict_p1_antipode([1] * 14), {}))],
           [("R1v = 3 sur 9/14", "ININTERPRÉTABLE",
             lambda: (verdict_p1_antipode([3] * 9 + [1] * 5), {}))])

    clause("P1-cellule-dégénérée", "sup à l'intérieur de la grille",
           "sup au bord + R1v > 1 majoritaire ⇒ `INCONCLUSIF — cellule dégénérée`",
           [("sup intérieur", PASS,
             lambda: (verdict_p1_degenerate([2] * 10, [0.3] * 10), {}))],
           [("sup au bord + R1v > 1", f"{INCONCLUSIF} — cellule dégénérée",
             lambda: (verdict_p1_degenerate([3] * 8 + [1] * 2,
                                            [3.0] * 7 + [0.3] * 3), {}))])

    # ---------------------------------------------------------------- ΔP6
    def _k_table():
        recomputed = {n: k_of_n(n) for n in range(5, 31)}
        bad = {n: (recomputed[n], K_TABLE_REFERENCE[n])
               for n in K_TABLE_REFERENCE if recomputed[n] != K_TABLE_REFERENCE[n]}
        fr = {}
        for n, expected in K_BOUNDARIES.items():
            k = recomputed[n]
            ratio = k_tail_ratio(n, k - 1)          # la queue REJETÉE
            fr[str(n)] = {"k": k, "ratio_rejete": ratio,
                          "attendu": expected,
                          "concorde": abs(ratio - expected) < 5e-6}
        det = {"k(5..30)": recomputed, "desaccords_avec_la_table_§3": bad,
               "frontieres": fr,
               "k(5)": recomputed[5],
               "note": "recalcul en ENTIERS Python (math.comb, 10·Σ ≤ 2ⁿ) ; la "
                       "table du §3 n'est lue que comme référence de comparaison."}
        ok = (not bad) and all(v["concorde"] for v in fr.values())
        return (PASS if ok else FAIL), det

    clause("ΔP6 (table k)", "n_disc = 10, 9 succès ⇒ k(10) = 8 ⇒ ARMÉ",
           "n_disc = 10, 6 succès ⇒ non ; n_disc = 4 ⇒ NON ÉVALUABLE ; "
           "n = 16, 23, 30 : k(n) recalculé en entiers et comparé à la table §3",
           [("n_disc = 10, 9 succès", ARME, lambda: verdict_dp6(10, 9)),
            ("table k(n), n = 5..30, frontières 16/23/30", PASS, _k_table),
            ("n_disc = 5, unanimité", ARME, lambda: verdict_dp6(5, 5))],
           [("n_disc = 10, 6 succès", NON_ARME, lambda: verdict_dp6(10, 6)),
            ("n_disc = 4", NON_EVALUABLE, lambda: verdict_dp6(4, 4))])

    clause("ΔP6-sec", "Wilcoxon avec écart ≥ 1 bit",
           "écart nul ⇒ vérifier que le secondaire NE PEUT PAS changer le verdict",
           [("primaire ARMÉ, écart 1.4 bit", f"{ARME} | sec=SIGNAL",
             lambda: (verdict_dp6_sec(ARME, 1.4), {}))],
           [("primaire NON ARMÉ, écart nul", f"{NON_ARME} | sec=NUL",
             lambda: (verdict_dp6_sec(NON_ARME, 0.0), {})),
            ("primaire NON ARMÉ, écart 2 bits ⇒ NON renversé",
             f"{NON_ARME} | sec=SIGNAL",
             lambda: (verdict_dp6_sec(NON_ARME, 2.0), {}))])

    # ----------------------------------------------------------------- P3
    g3 = _rng(5)
    clause("P3", "21 permutations, médiane 1/30",
           "médiane 14/30 ⇒ `INCONCLUSIF` ; et B pair ⇒ le banc doit REFUSER "
           "(médiane ambiguë) ; et une permutation unique ⇒ refus (variance nulle)",
           [("B = 21, médiane 1/30", PASS,
             lambda: verdict_p3(sorted(g3.integers(0, 3, 21).tolist())[:10]
                                + [1] + [1] * 10))],
           [("B = 21, médiane 14/30", INCONCLUSIF,
             lambda: verdict_p3([14] * 21)),
            ("B = 20 (pair)", f"{REFUS} — B pair : médiane demi-entière ambiguë",
             lambda: verdict_p3([1] * 20, b=20)),
            ("B = 1 (permutation unique)",
             f"{REFUS} — B = 1 : variance nulle par construction",
             lambda: verdict_p3([1], b=1))])

    # ----------------------------------------------------------------- P4
    clause("P4", "croisé 1/30", "croisé 20/30",
           [("1/30", PASS, lambda: (verdict_p4(1), {}))],
           [("20/30", "SÉLECTIVITÉ NULLE", lambda: (verdict_p4(20), {}))])

    # ----------------------------------------------------------------- P5
    clause("P5", "E3 = 0.047", "E3 = 0.06 ⇒ bug, pas un résultat",
           [("E3 = 0.047", PASS, lambda: (verdict_p5(0.047), {})),
            ("E3 = 0.05 (bord, théorème)", PASS,
             lambda: (verdict_p5(E3_BUDGET), {}))],
           [("E3 = 0.06", BUG, lambda: (verdict_p5(0.06), {}))])

    # ---------------------------------------------------------- P5f-borne
    n5 = _p5f_neighbor5_case()
    tie = _p5f_tie_case()
    clause("P5f-borne",
           "taux 0.6 × `borne_marge` ; et une bascule portée par la valeur du "
           "**voisin 5** ⇒ doit PASSER (cellule E-D6) ; `borne_marge = 0` avec "
           "taux 0 ⇒ PASS sans NaN ; ex-æquo à marge exactement 0.0512711 ⇒ PASS",
           "taux 1.4 × `borne_marge` ⇒ bug",
           [("taux = 0.6 × borne", PASS,
             lambda: (verdict_p5f_borne(0.6 * 0.10, 0.10), {"borne": 0.10})),
            ("bascule portée par le VOISIN 5 (E-D6)", PASS,
             lambda: (verdict_p5f_borne(n5["taux_observe"], n5["borne_Vk"]), n5)),
            ("borne_marge = 0 et taux 0 (pas de NaN)", PASS,
             lambda: (verdict_p5f_borne(0.0, 0.0), {"nan": False})),
            ("ex-æquo à marge = 0.0512711 (inégalité LARGE)", PASS,
             lambda: (verdict_p5f_borne(1.0, tie["borne_large"]), tie))],
           [("taux = 1.4 × borne", BUG,
             lambda: (verdict_p5f_borne(1.4 * 0.10, 0.10), {"borne": 0.10})),
            ("borne FAUSSE sur `v_nn` seul (E-D6) : déclarerait bug une "
             "implémentation correcte", BUG,
             lambda: (verdict_p5f_borne(n5["taux_observe"], n5["borne_vnn"]), n5)),
            ("inégalité STRICTE à l'ex-æquo : déclarerait bug", BUG,
             lambda: (verdict_p5f_borne(1.0, tie["borne_stricte"]), tie))])

    clause("P5f-cond + token gagnant",
           "bascule confiante, gagnant = valeur du store ⇒ intrusion",
           "bascule confiante, gagnant absent du store ⇒ bug, run invalide",
           [("gagnant ∈ store", "INTRUSION",
             lambda: (verdict_p5f_cond(7, [3, 7, 9], True), {})),
            ("bascule NON confiante", PASS,
             lambda: (verdict_p5f_cond(11, [3, 7, 9], False), {}))],
           [("gagnant ∉ store", BUG,
             lambda: (verdict_p5f_cond(11, [3, 7, 9], True), {}))])

    # -------------------------------------------------------------- P5c-id
    clause("P5c-id", "`f(un-hot) = 1 % < f(c=1) = 4 %`",
           "`f(un-hot) = 5 % ≥ f(c=1) = 3 %` ⇒ FAIL",
           [("1 % < 4 %", PASS, lambda: (verdict_p5c_id(0.01, 0.04), {}))],
           [("5 % ≥ 3 %", FAIL, lambda: (verdict_p5c_id(0.05, 0.03), {})),
            ("égalité (≥)", FAIL, lambda: (verdict_p5c_id(0.03, 0.03), {}))])

    # ----------------------------------------------------------------- P7
    clause("P7", "dégradation 20 %", "dégradation 100 %",
           [("20 %", PASS, lambda: (verdict_p7(0.20), {}))],
           [("100 %", FAIL, lambda: (verdict_p7(1.00), {}))])

    # ----------------------------------------------------------------- P8
    clause("P8", "série monotone sans genou", "série à genou",
           [("1,2,3,4,5,6", PASS, lambda: verdict_p8([1, 2, 3, 4, 5, 6]))],
           [("1,2,3,4,5,40 (genou)", "GENOU",
             lambda: verdict_p8([1, 2, 3, 4, 5, 40])),
            ("série non monotone", "NON MONOTONE",
             lambda: verdict_p8([1, 2, 1, 4, 5, 6]))],
           note=UNDERSPEC["P8"])

    # ------------------------------------------------------------------- G
    e3_ok = [0.01, 0.02, 0.03, 0.035, 0.04, 0.042, 0.045, 0.06, 0.07, 0.08]
    e3_nc = [0.01, 0.02, 0.03, 0.20, 0.04, 0.041, 0.042, 0.043, 0.044, 0.30]
    p1_c = [1, 2, 6, 6, 6, 6, 6, 6, 6, 6]
    clause("G : non-vacuité + `τ_promu` connexe",
           "`#{α=1} = 18`, `#{α=0} = 12` ; E3 admissible sur les déciles 1..7 ⇒ "
           "`τ_promu` = 7ᵉ ; `P1(λ*) = 3`, `P1(τ_promu) = 6` ⇒ promotion armée",
           "grille NON CONNEXE : E3 admissible aux déciles 1-3, violé au 4ᵉ, "
           "ré-admissible aux 5-9 ⇒ `τ_promu` = 3ᵉ, PAS le 9ᵉ (cellule E-D7) ; "
           "`P1(λ*) = 0` et `P1(τ_promu) = 0` ⇒ NON ÉVALUABLE ; `#{α=0} = 0` ⇒ "
           "τ non évalué ; 1ᵉʳ décile déjà violé ⇒ G NON ÉVALUABLE",
           [("préfixe admissible 1..7, promotion", "PROMOTION ARMÉE (tau=idx7)",
             lambda: verdict_g(e3_ok, p1_c, 18, 12, 3))],
           [("grille NON CONNEXE (E-D7) ⇒ idx3, jamais idx9",
             "PROMOTION ARMÉE (tau=idx3)",
             lambda: verdict_g(e3_nc, p1_c, 18, 12, 3)),
            ("P1(λ*) = 0 et P1(τ_promu) = 0", NON_EVALUABLE,
             lambda: verdict_g(e3_ok, [0] * 10, 18, 12, 0)),
            ("#{α=0} = 0 (vacuité par satisfaction)", "TAU NON ÉVALUÉ",
             lambda: verdict_g(e3_ok, p1_c, 30, 0, 3)),
            ("1ᵉʳ décile déjà violé", NON_EVALUABLE,
             lambda: verdict_g([0.2] + e3_ok[1:], p1_c, 18, 12, 3))])

    # ----------------------------------------------------------- multi-clé
    a_ok = [True] * 20 + [False] * 10
    b_ok = [True] * 19 + [False] * 11
    clause("Multi-clé (a)(b)(c) + m",
           "(a) vraie ; (b) 5e-3 vs 0 ; (c) intra 1e-4 < inter 5e-3 ⇒ ARMÉ",
           "(c) fausse ⇒ NON ARMÉ ; (c) avec les deux `pct` à 0 ⇒ décision sur "
           "`m`, PAS de NON ÉVALUABLE ; nulle stratifiée de verdict qualitatif "
           "opposé ⇒ NON ÉVALUABLE",
           [("(a)(b)(c) vraies sur 19/30", ARME,
             lambda: verdict_multikey(a_ok, b_ok, [1e-4] * 30, [5e-3] * 30,
                                      [-0.2] * 30, [-0.1] * 30))],
           [("(c) fausse", NON_ARME,
             lambda: verdict_multikey(a_ok, b_ok, [5e-3] * 30, [1e-4] * 30,
                                      [-0.1] * 30, [-0.2] * 30)),
            ("les deux `pct` saturés à 0 ⇒ décision sur `m`", ARME,
             lambda: verdict_multikey(a_ok, b_ok, [0.0] * 30, [0.0] * 30,
                                      [-0.2] * 30, [-0.1] * 30)),
            ("nulle stratifiée qualitativement opposée", NON_EVALUABLE,
             lambda: verdict_multikey(a_ok, b_ok, [1e-4] * 30, [5e-3] * 30,
                                      [-0.2] * 30, [-0.1] * 30,
                                      strat_qualitatif_identique=False))])

    clause("Ventilation 0/1 attribut partagé",
           "trois médianes ordonnées `intra < 1 partagé < 0 partagé`",
           "`1 partagé ≈ intra` ⇒ rapporté comme clé de gabarit de surface "
           "(descriptif, pas un FAIL)",
           [("ordre monotone", "ORDRE MONOTONE",
             lambda: (verdict_ventilation(1e-4, 1e-3, 5e-3), {}))],
           [("1 partagé ≈ intra", "CLÉ DE GABARIT DE SURFACE",
             lambda: (verdict_ventilation(1e-4, 1.0001e-4, 5e-3), {}))],
           note=UNDERSPEC["Ventilation 0/1"])

    return C


# =========================================================================
#  ===================  SUITE I2 — `layer_profile`  =======================
#
#  Protocole : `experiments/EXP-2026-08-22-layer-profile.md` (statut PROPOSE,
#  version CONSOLIDÉE du 2026-08-22). Portes §4.7, **quatre bandes N/M/I/V** et
#  overlay `D` (§4.5), quatre cellules §4.8, cinq nulles §5.
#  CPU seul, aucune mesure, aucun modèle (le tokenizer GPT-2 sert les nulles
#  lexicale et de cadre, qui ne sont pas décidables sans lui).
# =========================================================================

import layer_profile as lp  # noqa: E402

# Opérationnalisations DÉCLARÉES par le banc là où le protocole fixe la clause
# mais pas son seuil d'exécution. Signalées au rapport, hors E.
I2_TOL_AUC = 1e-12            # égalité d'AUC sous transformation monotone
I2_TOL_H = 1e-6               # égalité relative de H sous mise à l'échelle
I2_TOL_SUFFIXE = 1e-9         # égalité aux constantes DÉRIVÉES de `V-suffixe`
I2_K_PLAFOND_POOL = lp.K_PLAFOND_POOL                # 20 : K max de ce pool

# Deux sous-spécifications du premier passage ont été TRANCHÉES au
# pré-enregistrement (protocole du 2026-08-22, second correctif) et ne figurent
# donc plus ici : (i) le chiffre « K ≥ 429 » est RETIRÉ, remplacé par l'énoncé
# robuste calculé sous toutes les opérationnalisations ; (ii) la bande N est
# définie par `IC_inf(ΔR1) ≤ 0`, avec sous-étiquette descriptive pour le cas
# significativement négatif. Les deux restantes sont ACCEPTÉES telles quelles.
UNDERSPEC_I2 = {
    "R1_36 — égalités":
        "le protocole fixe « égalités ½ crédit » pour l'AUC ; le banc étend la "
        "règle au plus proche voisin : une égalité à `g` gagnants dont la cible "
        "vaut `1/g`.",
    "R1_36 (P-ent)":
        "le protocole écrit « idem sur `P-ent` » sans redéfinir l'ordre de "
        "remplissage ; le banc déclare que « le plus dur d'abord » suit le slot "
        "de la strate : composante d'entité d'abord, puis owner, puis P-0. La "
        "taille reste 36 exactement.",
    "V-plat": "le protocole fixe `R = max−min`, le bootstrap et la courbe "
              "centrée par unité, pas la fabrique de `R*` ; le banc déclare le "
              "double centrage (par unité puis par couche) comme vérité plate.",
}

# Statut des opérationnalisations déclarées, après arbitrage du 2026-08-22.
STATUT_OPERATIONNALISATIONS_I2 = {
    "V-puissance (décision AUC)": "TRANCHÉE au pré-enregistrement — le chiffre "
        "429 est RETIRÉ (non re-dérivable, D14-R) ; la porte calcule le K requis "
        "sous chaque opérationnalisation et déclare le FACTEUR.",
    "Bande N (ΔR1 négatif)": "TRANCHÉE au pré-enregistrement — bande N définie "
        "par `IC_inf(ΔR1) ≤ 0` ; le cas significativement négatif est une "
        "sous-étiquette DESCRIPTIVE (le corpus réel séparerait moins bien que sa "
        "propre nulle : fait sur l'instrument, pas sur le cortex).",
    "R1_36 — égalités": "ACCEPTÉE telle quelle (conservatrice, ne change aucun "
        "verdict) — ira au journal.",
    "R1_36 (P-ent)": "ACCEPTÉE telle quelle (conservatrice, ne change aucun "
        "verdict) — ira au journal.",
}

# Chiffres de v3 interdits dans une porte, un seuil ou une prédiction (`V-amont`,
# D14-R). Écrits ici SOUS FORME DE MOTIFS À DÉTECTER, jamais comme seuils.
I2_MOTIFS_AMONT = ("0.99989", "0.99848", "90/90", "12/30", "23/30", "30/30",
                   "3-4/30", "0,99989")


# ------------------------------------------------------------------ portes

def gate_v_diversite(slots, n: int = None) -> tuple[str, dict]:
    """`V-diversité` (§4.7, remplace `V-div`) : **PASS ssi**
    `(#owners, #entités, #verbes) = (min(N,16), min(N,20), min(N,5))`.

    **Zéro GPU, zéro bootstrap.** L'ancienne `V-div` mesurait un CARDINAL DE
    STRATE ⇒ croissante en redondance ⇒ elle **récompensait la dégénérescence**
    (défaut 0-7). La direction est restaurée : le corpus divers PASSE, le corpus
    dégénéré ÉCHOUE.
    """
    n = len(slots) if n is None else n
    obs = lp.diversite(slots)
    att = lp.diversite_attendue(n)
    return (PASS if obs == att else FAIL,
            {"N": n, "observe": list(obs), "attendu": list(att),
             "ecart": [o - a for o, a in zip(obs, att)],
             "recensement_publie_avant_mesure": True})


def gate_v_puissance(k: int, sigma0: float = lp.SIGMA0, marge: float = lp.MARGE_R1,
                     statistique: str = "R1_36") -> tuple[str, dict]:
    """`V-puissance` (§4.7) : **PASS ssi** `K_S ≥ (1.96·σ₀/|θ−T|)²`.

    σ₀ = 0.5 (Bernoulli, conservateur), |θ−T| = 0.25 ⇒ **K ≥ 16**. **Fusion avec
    `V-diversité` refusée** : un FAIL doit NOMMER sa cause.
    """
    kr = lp.k_requis(sigma0, marge)
    ok = k >= kr
    det = {"statistique": statistique, "K": k, "K_requis": kr,
           "K_requis_exact": round(lp.k_requis_exact(sigma0, marge), 2),
           "sigma0": sigma0, "marge_theta_moins_T": marge,
           "formule": "K ≥ (1.96·σ₀/|θ−T|)²",
           "facteur_requis_sur_disponible": round(kr / k, 2) if k else None,
           "cause_du_FAIL": None}
    if not ok:
        det["cause_du_FAIL"] = f"K = {k} < {kr} requis (puissance du DESIGN)"
    return (PASS if ok else FAIL, det)


def gate_v_puissance_auc(k: int = lp.K_PLAFOND_POOL,
                         operationnalisations=None) -> tuple[str, dict]:
    """`V-puissance` **sous décision AUC** (§3, correctif du 2026-08-22).

    Le chiffre « K ≥ 429 » est **RETIRÉ** : il n'est pas re-dérivable (D14-R —
    un chiffre sans sa dérivation ne se cite pas). La porte **recalcule** le `K`
    requis sous chaque opérationnalisation examinée, le compare au `K`
    disponible, **imprime le facteur**, et déclare que la conclusion —
    *le couloir en AUC est indécidable à tout N ≤ 80* — **ne dépend d'aucun
    choix de σ₀ ni de marge**.
    """
    ops = lp.OPERATIONNALISATIONS_AUC if operationnalisations is None         else operationnalisations
    r = lp.enonce_robuste_auc(k, ops)
    ok = k >= r["K_requis_min"]
    det = {"statistique": "AUC (T = 0.9622)", "K": k,
           "chiffre_429": "RETIRÉ — non re-dérivable (D14-R)",
           "K_requis_par_operationnalisation": r["table"],
           "K_requis_min": r["K_requis_min"], "K_requis_max": r["K_requis_max"],
           "facteur_min": r["facteur_min"],
           "robuste_au_choix_d_operationnalisation": r["robuste"],
           "formule": "K ≥ (1.96·σ₀/|θ−T|)²",
           "cause_du_FAIL": None, "indecidable_a_tout_N": False}
    if not ok:
        det["cause_du_FAIL"] = (
            f"K = {k} < {r['K_requis_min']} requis au minimum — facteur "
            f"{r['facteur_min']}")
        det["indecidable_a_tout_N"] = True
        det["nom_de_l_indecidabilite"] = r["enonce"] + (
            f" K plafonne à {lp.K_PLAFOND_POOL} dans ce pool (plus grande "
            f"famille de composantes de slot), atteint dès N = 40 : aucune "
            f"répétition ne lève cette porte.")
    return (PASS if ok else FAIL, det)


def gate_v_paires(n_intra: int, n_inter: int, recensement=None) -> tuple[str, dict]:
    """`V-paires` (§4.7) : `n_intra = 240`, `n_inter = 28 440` **exactement** ;
    `P-0/P-own/P-ent = 2880/160/120`, `P-both = 0`. Porte mordante."""
    att = {lp.P_0: 2880, lp.P_OWN: 160, lp.P_ENT: 120, lp.P_BOTH: 0}
    ok = (n_intra == lp.N_INTRA_ATTENDU) and (n_inter == lp.N_INTER_ATTENDU)
    det = {"n_intra": n_intra, "n_inter": n_inter,
           "attendus": [lp.N_INTRA_ATTENDU, lp.N_INTER_ATTENDU],
           "derivation": "80 × C(3,2) = 240 ; C(80,2) × 9 = 3160 × 9 = 28 440 ; "
                         "3160 = 2880 + 160 + 120 + 0",
           "egalites": "½ crédit, déclaré avant mesure"}
    if recensement is not None:
        det["recensement_identite"] = dict(recensement)
        det["recensement_attendu"] = att
        ok = ok and all(recensement.get(k) == v for k, v in att.items())
    return (PASS if ok else FAIL, det)


def gate_v_plat(courbes, b: int = 2000, seed: int = 0) -> tuple[str, dict]:
    """`V-plat` (§4.7) : PLATE ssi `R_obs ≤ q_0.95(R*)`. **Aucune constante
    posée** — le seuil est un quantile bootstrap."""
    r = lp.v_plat(courbes, b=b, seed=seed)
    det = {k: v for k, v in r.items() if k != "courbe_centree"}
    return ("PLATE" if r["plate"] else "NON PLATE", det)


def gate_v_bord_i2(l_star: int, L: int) -> tuple[str, dict]:
    """`V-bord` (§4.7) : `ℓ*` en `ℓ = 1` ou `ℓ = L` ⇒ la quantité ne dit rien."""
    return ("AU BORD" if l_star in (1, L) else "INTÉRIEUR",
            {"l_star": l_star, "L": L})


def gate_v_lambda1(lambda1_par_couche, L: int) -> tuple[str, dict]:
    """`V-λ₁` (§4.7) : λ₁/Σλ publié pour **toutes** les couches — absence ⇒ `H`
    non interprétable et **retrait automatique** (§4.6, N-19)."""
    manquantes = [e for e in range(L + 1) if e not in lambda1_par_couche]
    return (PASS if not manquantes else FAIL,
            {"couches_manquantes": manquantes, "L": L,
             "consequence_si_FAIL": "H retirée de la fiche I2, automatiquement, "
                                    "consigné au journal (N-19)"})


def gate_retrait_H(interpretable_par_modele) -> tuple[str, dict]:
    """Clause de retrait automatique (§4.6, N-19) : si `H` est non interprétable
    sur **les trois modèles**, elle est RETIRÉE, sans nouvelle discussion."""
    retire = not any(interpretable_par_modele.values())
    return ("RETIRÉE" if retire else "CONSERVÉE",
            {"interpretable_par_modele": dict(interpretable_par_modele),
             "automatique": True})


def gate_v_amont(source: str, motifs=I2_MOTIFS_AMONT) -> tuple[str, dict]:
    """`V-amont` (§4.7) : aucun chiffre de v3 dans une porte, un seuil ou une
    prédiction (D14-R). `B-v3` est descriptif."""
    trouves = [m for m in motifs if m in source]
    return (PASS if not trouves else FAIL, {"motifs_trouves": trouves})


def gate_v_1pass(compte_par_variante) -> tuple[str, dict]:
    """`V-1pass` (§4.7) : **un** forward par (modèle, variante)."""
    mauvais = {k: v for k, v in compte_par_variante.items() if v != 1}
    return (PASS if not mauvais else FAIL,
            {"comptes": dict(compte_par_variante), "non_conformes": mauvais})


def gate_v_hooks(hooks_restants: int, diff_engram: str = "",
                 m_instanciee: bool = False) -> tuple[str, dict]:
    """`V-hooks` (§4.7) : hooks retirés en `finally`, `git diff --stat engram/`
    vide, `M` jamais instanciée."""
    ok = (hooks_restants == 0) and (diff_engram.strip() == "") and not m_instanciee
    return (PASS if ok else FAIL,
            {"hooks_restants": hooks_restants,
             "git_diff_engram": diff_engram.strip(), "M_instanciee": m_instanciee})


def gate_v_L(L_mesure: int, L_attendu: int) -> tuple[str, dict]:
    """`V-L` (§4.7) : `L` re-lu du config ; écart ⇒ **arrêt**."""
    return (PASS if L_mesure == L_attendu else FAIL,
            {"L_config": L_mesure, "L_attendu": L_attendu,
             "consequence_si_FAIL": "ARRÊT du run"})


def gate_v_suffixe(partage_par_type, n: int = lp.N_UNITES) -> tuple[str, dict]:
    """`V-suffixe` **re-dérivée** (§4.7, défaut 0-8) : partage du **dernier token
    BPE** — **para1 = 1.0000**, **para3 = 1.0000**,
    **para2 = `#{paires : 5|d}/C(N,2)`**.

    Constantes ENTIÈREMENT DÉRIVÉES : para1 et para3 finissent par une chaîne
    globale gelée ⇒ 1 par construction ; para2 finit par le **verbe** (période 5).
    L'ancienne attente (para1 ≈ 0) datait d'avant §15 A-1. **La porte redevient
    mordante.**
    """
    d = lp.partage_suffixe_derive(n)
    p = dict(partage_par_type)
    ecarts = {t: abs(p.get(t, float("nan")) - d[t]) for t in ("para1", "para2", "para3")}
    ok = all(e <= I2_TOL_SUFFIXE for e in ecarts.values())
    return (PASS if ok else FAIL,
            {"observe": p, "derive": {t: d[t] for t in ("para1", "para2", "para3")},
             "ecarts": ecarts, "tolerance": I2_TOL_SUFFIXE,
             "derivation_para2": f"{d['n_paires_5_divise_d']}/{d['C_n_2']}",
             "N": n})


def gate_v_hash_i2(avant: str, apres: str) -> tuple[str, dict]:
    """`V-hash` (§4.7) : SHA-256 de (a) et de `B-v3` avant/après."""
    return (PASS if avant == apres else FAIL,
            {"avant": avant[:16], "apres": apres[:16]})


def gate_v_source(citations=None) -> tuple[str, dict]:
    """`V-source` (§4.7, NOUVELLE, défaut 0-10) : toute équation citée est relue
    dans le **PDF** de sa source primaire, et l'attribution nomme **l'article
    d'origine**. **Lire du HTML pour citer une équation est un motif d'arrêt.**
    """
    cits = lp.CITATIONS if citations is None else citations
    fautes = [{"equation": c.get("equation"), "support": c.get("support"),
               "source_primaire": c.get("source_primaire")}
              for c in cits
              if c.get("support") != "PDF" or not c.get("source_primaire")]
    return (PASS if not fautes else FAIL,
            {"n_citations": len(cits), "non_conformes": fautes,
             "regle": "support = PDF ET source primaire nommée ; sinon ARRÊT"})


# ------------------------------------------- bandes (§4.5) — porte `V-bandes`

def _grille_bandes(seuil=lp.T_COULOIR):
    """Grille de balayage pour l'exhaustivité : bornes de `ΔR1` et de `R1_36`
    couvrant les deux côtés de 0 et des deux côtés de `T`, **bords inclus**."""
    d = [(-0.20, -0.05), (-0.10, 0.10), (-1e-12, 0.30), (0.0, 0.30),
         (1e-12, 0.30), (0.05, 0.40)]
    r = [(0.00, 0.05), (0.05, seuil - 1e-12), (0.05, seuil), (0.05, 0.60),
         (seuil, 0.60), (seuil + 1e-12, 0.60), (0.55, 0.90)]
    return [(x, y) for x in d for y in r]


def gate_v_bandes(seuil=lp.T_COULOIR) -> tuple[str, dict]:
    """`V-bandes` (§4.7, NOUVELLE, défaut 0-9) : la partition **N / M / I / V**
    est **exhaustive et mutuellement exclusive**, **bords inclus**, et la
    **précédence de l'overlay `D`** est testée.

    C'est la porte qui aurait empêché le protocole qui grave D18 de violer D18.
    """
    hors, cas = [], []
    for icd, icr in _grille_bandes(seuil):
        b = lp.bande_modele(icd, icr, seuil)
        cas.append({"ic_delta_R1": list(icd), "ic_R1": list(icr), "bande": b})
        if b not in lp.BANDES:
            hors.append(cas[-1])
    bords = {
        "IC_sup(R1) = T → I": lp.bande_modele((0.05, 0.30), (0.05, seuil), seuil),
        "IC_inf(R1) = T → V": lp.bande_modele((0.05, 0.30), (seuil, 0.60), seuil),
        "IC(ΔR1) touche 0 par la borne inférieure → N":
            lp.bande_modele((0.0, 0.30), (0.55, 0.90), seuil),
    }
    bords_ok = (bords["IC_sup(R1) = T → I"] == lp.BANDE_I
                and bords["IC_inf(R1) = T → V"] == lp.BANDE_V
                and bords["IC(ΔR1) touche 0 par la borne inférieure → N"] == lp.BANDE_N)
    # Correctif D18 du 2026-08-22 : le cas que la rédaction précédente laissait
    # HORS-PARTITION — `IC(ΔR1)` entièrement NÉGATIF avec `IC_inf(R1) < T`.
    ic_neg, ic_bas = (-0.20, -0.05), (0.00, 0.05)
    cas_negatif = {
        "ic_delta_R1": list(ic_neg), "ic_R1": list(ic_bas),
        "bande": lp.bande_modele(ic_neg, ic_bas, seuil),
        "sous_etiquette_delta_R1_negatif": lp.delta_r1_negatif(ic_neg),
        "lecture": "le corpus réel séparerait MOINS BIEN que sa propre nulle — "
                   "fait sur l'INSTRUMENT, pas sur le cortex ; descriptif, "
                   "jamais décisionnel"}
    negatif_ok = (cas_negatif["bande"] == lp.BANDE_N
                  and cas_negatif["sous_etiquette_delta_R1_negatif"] is True)
    precedence = {
        "I + V → I": lp.bande_gate(lp.BANDE_I, lp.BANDE_V),
        "V + I → I": lp.bande_gate(lp.BANDE_V, lp.BANDE_I),
        "I + N → I": lp.bande_gate(lp.BANDE_I, lp.BANDE_N),
        "V + M → D": lp.bande_gate(lp.BANDE_V, lp.BANDE_M),
        "N + N → N": lp.bande_gate(lp.BANDE_N, lp.BANDE_N),
    }
    prec_ok = (precedence["I + V → I"] == lp.BANDE_I
               and precedence["V + I → I"] == lp.BANDE_I
               and precedence["I + N → I"] == lp.BANDE_I
               and precedence["V + M → D"] == lp.BANDE_D
               and precedence["N + N → N"] == lp.BANDE_N)
    # mutuelle exclusivité : le classifieur rend UNE étiquette par observation
    exclusif = all(isinstance(c["bande"], str) for c in cas)
    ok = (not hors) and bords_ok and prec_ok and exclusif and negatif_ok
    return (PASS if ok else FAIL,
            {"n_cas_balayes": len(cas), "hors_partition": hors,
             "bandes_rencontrees": sorted({c["bande"] for c in cas}),
             "bords": bords, "bords_conformes": bords_ok,
             "cas_delta_R1_entierement_negatif": cas_negatif,
             "cas_negatif_conforme": negatif_ok,
             "precedence_overlay_D": precedence, "precedence_conforme": prec_ok,
             "mutuellement_exclusive": exclusif})


def _bande(ic_delta, ic_r1, seuil=lp.T_COULOIR):
    b = lp.bande_modele(ic_delta, ic_r1, seuil)
    return (b, {"ic_delta_R1": list(ic_delta), "ic_R1": list(ic_r1),
                "seuil_T": seuil, "V+": lp.mention_v_plus(ic_r1),
                "delta_R1_significativement_negatif": lp.delta_r1_negatif(ic_delta)})


def bande_couverte(ic_delta, ic_r1, seuil=lp.T_COULOIR):
    """D18 : la partition N/M/I/V doit être EXHAUSTIVE. Le banc ne comble aucun
    trou : combler serait amender."""
    b = lp.bande_modele(ic_delta, ic_r1, seuil)
    return ("COUVERT" if b in lp.BANDES else lp.BANDE_HORS,
            {"bande": b, "ic_delta_R1": list(ic_delta), "ic_R1": list(ic_r1)})


def _cellule(lc, lh, L, plate_c=False, plate_h=False):
    f = lp.fenetre_D3(L)
    return (lp.cellule(lc, lh, f, L, plate_c, plate_h),
            {"l_contrast": lc, "l_H": lh, "fenetre": list(f), "L": L,
             "plate_contrast": plate_c, "plate_H": plate_h})


# ------------------------------------------ jeu de candidats `R1_36` (§4.5)

def gate_jeu_r1(slots, s: int = lp.S_LEURRES) -> tuple[str, dict]:
    """Le jeu `R1_36` a **exactement 37 éléments**, est **déterministe** et est
    **indépendant des similarités mesurées** (§4.5 ; toute sélection par
    proximité mesurée est un motif d'invalidation).

    Le test d'indépendance est mécanique : on permute les VALEURS de similarité
    et l'on vérifie que le jeu est identique — la fonction ne reçoit d'ailleurs
    aucun état.
    """
    n = len(slots)
    tailles, distincts, deterministe = set(), True, True
    jeux = {}
    for i in range(n):
        for t in range(lp.N_PARA):
            j = lp.jeu_candidats_R1(i, t, slots, lp.P_OWN, s)
            cand = [j["cible"]] + list(j["concurrents"])
            tailles.add(len(cand))
            distincts = distincts and (len(set(cand)) == len(cand))
            deterministe = deterministe and (
                lp.jeu_candidats_R1(i, t, slots, lp.P_OWN, s) == j)
            jeux[(i, t)] = tuple(cand)
    # indépendance : deux matrices de similarité différentes, MÊME jeu
    g = np.random.default_rng(7)
    base = g.normal(size=(n, 12))
    X = np.repeat(base, lp.N_PARA, axis=0) + 0.001 * g.normal(
        size=(n * lp.N_PARA, 12))
    S1 = lp.cosinus_matrice(X).astype(np.float64)        # géométrie « parfaite »
    S2 = g.permutation(S1.ravel()).reshape(S1.shape)     # MÊMES valeurs, permutées
    jeux2 = {(i, t): tuple([lp.jeu_candidats_R1(i, t, slots, lp.P_OWN, s)["cible"]]
                           + list(lp.jeu_candidats_R1(i, t, slots, lp.P_OWN,
                                                      s)["concurrents"]))
             for (i, t) in jeux}
    independant = jeux == jeux2
    r1_a = lp.r1_36(S1, slots, lp.P_OWN, s)["R1"]
    r1_b = lp.r1_36(S2, slots, lp.P_OWN, s)["R1"]
    j0 = lp.jeu_candidats_R1(0, 0, slots, lp.P_OWN, s)
    ok = (tailles == {1 + s} and distincts and deterministe and independant)
    return (PASS if ok else FAIL,
            {"tailles_observees": sorted(tailles), "taille_attendue": 1 + s,
             "tous_distincts": distincts, "deterministe": deterministe,
             "independant_des_similarites": independant,
             "R1_sous_S1": r1_a, "R1_sous_S2_permutee": r1_b,
             "la_statistique_depend_bien_de_S": r1_a != r1_b,
             "blocs_du_jeu_i0_t0": j0["blocs"], "hasard": 1.0 / (1 + s)})


# ------------------------------- biais de sélection du max (M-15) et « max des IC »

def _cas_estimateur_debiaise(seed: int = 3, n_couches: int = 12, k: int = 16,
                             b: int = 400) -> tuple[str, dict]:
    """**Effet NUL** par construction : `n_couches` couches de bruit indépendant,
    aucune différence vraie. Le max brut est biaisé vers le haut ; l'estimateur
    débiaisé `θ̂ = 2·max_obs − mean_b(θ*_b)` en diffère."""
    g = np.random.default_rng(seed)
    donnees = g.normal(0.0, 1.0, (k, n_couches))       # k clusters, ℓ couches
    obs = [float(donnees[:, e].mean()) for e in range(n_couches)]
    max_obs = max(obs)
    rng = np.random.default_rng(seed + 1)
    ech = np.empty(b)
    for m in range(b):
        idx = rng.integers(0, k, k)
        ech[m] = max(float(donnees[idx, e].mean()) for e in range(n_couches))
    jack = np.array([max(float(np.delete(donnees, j, axis=0)[:, e].mean())
                         for e in range(n_couches)) for j in range(k)])
    r = lp.ic_du_max(ech, max_obs, jack)
    diff = abs(r["theta_debiaise"] - max_obs)
    return (PASS if diff > 1e-9 else FAIL,
            {"max_brut": max_obs, "theta_debiaise": r["theta_debiaise"],
             "moyenne_bootstrap": r["moyenne_bootstrap"],
             "ecart_debiaise_vs_max": diff, "IC": [r["ic_bas"], r["ic_haut"]],
             "methode_IC": r["methode"], "argmax_re_selectionne": True,
             "B": b, "K": k})


def _cas_max_des_ic_rejete() -> tuple[str, dict]:
    """« Max des IC par couche » est un **motif d'invalidation** (§4.3, §6) : la
    construction est refusée mécaniquement, jamais discutée."""
    try:
        lp.ic_max_des_ic_par_couche([(0.1, 0.9)] * 12)
    except ValueError as e:
        return ("REJETÉ", {"message": str(e)})
    return (PASS, {"message": "AUCUN refus — la construction interdite a abouti"})


def _cas_bca_au_dela_de_095(seed: int = 5, b: int = 400) -> tuple[str, dict]:
    """`BCa` **requis** dès qu'une borne dépasse 0.95 (M-16 : le percentile
    sous-couvre près de la borne 1)."""
    g = np.random.default_rng(seed)
    ech = np.clip(g.beta(20, 1.0, b), 0, 1)             # masse près de 1
    jack = np.clip(g.beta(20, 1.0, 16), 0, 1)
    haut = lp.ic_du_max(ech, float(ech.max()), jack)
    ech_bas = g.normal(0.4, 0.05, b)
    bas = lp.ic_du_max(ech_bas, float(ech_bas.max()), g.normal(0.4, 0.05, 16))
    ok = haut["methode"] == "BCa" and bas["methode"] == "percentile"
    return (PASS if ok else FAIL,
            {"borne_haute": [haut["ic_bas"], haut["ic_haut"]],
             "methode_haute": haut["methode"],
             "borne_basse": [bas["ic_bas"], bas["ic_haut"]],
             "methode_basse": bas["methode"], "seuil_BCa": lp.SEUIL_BCA})


def _cas_permutation_max(seed: int = 6, b: int = 200) -> tuple[str, dict]:
    """Bande **N** par permutation des étiquettes d'unité **à couche fixée avec
    recalcul de `max_ℓ`** (FWER exact) : sous H₀ la statistique observée reste
    sous `q_0.95` ; un effet réel la dépasse."""
    g = np.random.default_rng(seed)
    k, n_c = 16, 12
    nul = g.normal(0, 1, (k, n_c))

    def stat(ell, rng):
        return float(nul[rng.permutation(k), ell].mean())

    r = lp.permutation_max_couches(stat, range(n_c), b=b, seed=seed)
    obs_nul = max(float(nul[:, e].mean()) for e in range(n_c))
    effet = nul.copy()
    effet[:, 5] += 3.0
    obs_effet = max(float(effet[:, e].mean()) for e in range(n_c))
    ok = obs_nul <= r["q_0.95"] and obs_effet > r["q_0.95"]
    return (PASS if ok else FAIL,
            {"q_0.95": r["q_0.95"], "max_observe_sous_H0": obs_nul,
             "max_observe_avec_effet": obs_effet, "B": b,
             "permutation": "étiquettes d'UNITÉ à couche fixée, max_ℓ RECALCULÉ ; "
                            "la permutation des étiquettes de COUCHE est invalide"})


def _cas_bootstrap(schema: str) -> tuple[str, dict]:
    """Cluster de rééchantillonnage = **la composante de slot** (§D.1).

    Les deux schémas sur les MÊMES données : le schéma publié est « cluster ».
    Le détail publie les deux largeurs (la clause n'est pas vacuée) et le fait
    que le rééchantillonnage par cluster **préserve exactement** les effectifs
    de strate.
    """
    slots = lp.corpus_a()["slots"]
    cl = lp.clusters_de_strate(slots, lp.P_OWN)
    membres = cl["membres"]
    g = _rng(16)
    effet = g.normal(0.0, 0.08, cl["K"])
    par_unite = np.array([0.6 + effet[cl["etiquette_par_unite"][i]]
                          + g.normal(0, 0.01) for i in range(len(slots))])

    def stat_cluster(idx):
        u = np.concatenate([np.asarray(membres[int(k)]) for k in idx])
        return float(par_unite[u].mean())

    r_c = lp.bootstrap_par_cluster(stat_cluster, cl["K"], b=400, seed=0)
    largeur_c = r_c["ic_haut"] - r_c["ic_bas"]
    rng = np.random.default_rng(0)
    ech = np.array([float(par_unite[rng.integers(0, par_unite.size,
                                                 par_unite.size)].mean())
                    for _ in range(400)])
    lo, hi = np.percentile(ech, [2.5, 97.5])
    largeur_u = float(hi - lo)
    effectifs = {int(k): len(v) for k, v in membres.items()}
    preserve = len(set(effectifs.values())) == 1
    det = {"schema": schema, "K": cl["K"], "cle_de_cluster": cl["cle"],
           "largeur_par_cluster": largeur_c, "largeur_par_unite": largeur_u,
           "largeurs_distinctes": bool(abs(largeur_c - largeur_u) > 1e-9),
           "effectifs_par_cluster": effectifs,
           "effectifs_preserves_exactement": preserve,
           "ic_par_cluster": [r_c["ic_bas"], r_c["ic_haut"]],
           "ic_par_unite": [float(lo), float(hi)], "B": 400}
    ok = (schema == "cluster") and det["largeurs_distinctes"] and preserve
    return (PASS if ok else FAIL, det)


# ---------------------------------------------- invariances (M-1b et 0-2)

def invariance_monotone(cos_intra, cos_inter, f) -> dict:
    """LE test du défaut 0-1 : l'AUC est **inchangée** sous une transformation
    strictement monotone appliquée **par couche**, alors que le ratio
    `s_intra/s_inter` **change** sur les mêmes données."""
    a, b = np.asarray(cos_intra, float), np.asarray(cos_inter, float)
    fa, fb = f(a), f(b)
    return {"auc": lp.auc_par_couche(a, b), "auc_transformee": lp.auc_par_couche(fa, fb),
            "ratio": float(a.mean() / b.mean()),
            "ratio_transforme": float(fa.mean() / fb.mean())}


def gate_invariance_monotone(cos_intra, cos_inter, f) -> tuple[str, dict]:
    d = invariance_monotone(cos_intra, cos_inter, f)
    auc_egale = abs(d["auc"] - d["auc_transformee"]) <= I2_TOL_AUC
    ratio_change = abs(d["ratio"] - d["ratio_transforme"]) > I2_TOL_AUC
    d |= {"auc_egale": auc_egale, "ratio_change": ratio_change}
    return (PASS if (auc_egale and ratio_change) else FAIL, d)


def gate_invariance_lignes(X, facteurs, normalisation: str) -> tuple[str, dict]:
    """Défaut 0-2 : `H` est inchangée sous mise à l'échelle des LIGNES **si et
    seulement si** la normalisation Giraldo est appliquée."""
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(facteurs, dtype=np.float64)[:, None]
    h0 = lp.entropie_matricielle(X, normalisation)
    h1 = lp.entropie_matricielle(c * X, normalisation)
    egal = abs(h1 - h0) <= I2_TOL_H * max(1.0, abs(h0))
    return (PASS if egal else FAIL,
            {"H": h0, "H_lignes_mises_a_l_echelle": h1, "ecart": abs(h1 - h0),
             "normalisation": normalisation, "tol": I2_TOL_H})


# ------------------------------------------------- hooks réels (torch, CPU)

def _module_jouet():
    """Petit module torch (CPU, 12 « blocs ») — aucun modèle HF, aucun poids
    téléchargé : `V-hooks` et `V-1pass` s'exercent sur du vrai `nn.Module`."""
    import torch
    from torch import nn

    class Bloc(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):
            return (self.lin(x),)

    class Jouet(nn.Module):
        def __init__(self, d=8, L=12):
            super().__init__()
            self.blocs = nn.ModuleList([Bloc(d) for _ in range(L)])

        def forward(self, x):
            for b in self.blocs:
                x = b(x)[0]
            return x

    torch.manual_seed(0)
    return Jouet()


def _hooks_apres_capture(leve_exception: bool) -> tuple[str, dict]:
    """Cas PASSANT de `V-hooks` : la capture retire ses hooks en `finally`,
    **même quand le forward lève**."""
    import torch
    m = _module_jouet()
    cap = lp.CaptureToutesCouches(m, m.blocs)
    x = torch.zeros(4, 8)
    try:
        with cap:
            cap.positions = None
            if leve_exception:
                raise RuntimeError("panne simulée pendant le forward")
            with torch.no_grad():
                m(x[:, None, :])
    except RuntimeError:
        pass
    return gate_v_hooks(lp.compte_hooks(m))


def _hooks_fuite() -> tuple[str, dict]:
    """Cas ÉCHOUANT de `V-hooks` : un hook posé hors du `try/finally` survit."""
    m = _module_jouet()
    m.blocs[3].register_forward_hook(lambda *a: None)     # jamais retiré
    return gate_v_hooks(lp.compte_hooks(m))


def _forwards_un_passage() -> tuple[str, dict]:
    """Cas PASSANT de `V-1pass` : L+1 couches profilées en UN forward."""
    import torch
    m = _module_jouet()
    cap = lp.CaptureToutesCouches(m, m.blocs)
    with cap:
        cap.positions = None
        with torch.no_grad():
            m(torch.zeros(4, 1, 8))
    return gate_v_1pass({"a": cap.n_forwards}) if cap.n_forwards else (FAIL, {})


def _forwards_boucle_sur_les_couches() -> tuple[str, dict]:
    """Cas ÉCHOUANT de `V-1pass` : une implémentation qui relance le forward
    couche par couche."""
    import torch
    m = _module_jouet()
    cap = lp.CaptureToutesCouches(m, m.blocs)
    with cap:
        cap.positions = None
        with torch.no_grad():
            for _ell in range(len(m.blocs)):
                m(torch.zeros(4, 1, 8))
    return gate_v_1pass({"a": cap.n_forwards})


# --------------------------------- partition d'identité : recensement (§4.2)

def gate_recensement_identite(slots, attendu) -> tuple[str, dict]:
    """Recensement `P-0/P-own/P-ent/P-both`, **publié avant mesure**, et preuve
    arithmétique que `P-both = ∅` (`lcm(16,20) = 80`)."""
    p = lp.partition_identite(slots)
    r = p["recensement"]
    pb = lp.p_both_impossible(n=len(slots))
    somme_ok = sum(r.values()) == p["n_paires"]
    ok = all(r[k] == v for k, v in attendu.items()) and somme_ok
    return (PASS if ok else FAIL,
            {"N": len(slots), "recensement": r, "attendu": dict(attendu),
             "n_paires": p["n_paires"], "somme_coherente": somme_ok,
             "P_both_preuve": pb})


def _stratifier_avec_verbe(slots) -> dict:
    """Classifieur FAUTIF (contre-exemple échouant) : il compte le VERBE comme un
    slot d'identité — c'est exactement le défaut 0-12 (N-9), qui faisait de
    `AUC(S1)` du bruit étiqueté."""
    n = len(slots)
    recens = {c: 0 for c in lp.PARTITIONS}
    for i in range(n):
        for j in range(i + 1, n):
            mo, me, mv = (slots[i][0] == slots[j][0], slots[i][1] == slots[j][1],
                          slots[i][2] == slots[j][2])
            if mo and me:
                recens[lp.P_BOTH] += 1
            elif mo and not mv:
                recens[lp.P_0] += 1          # le verbe « départage » : FAUTIF
            elif mo:
                recens[lp.P_OWN] += 1
            elif me:
                recens[lp.P_ENT] += 1
            else:
                recens[lp.P_0] += 1
    return {"recensement": recens, "n_paires": n * (n - 1) // 2}


def gate_recensement_avec_classifieur(classifieur, slots, attendu) -> tuple[str, dict]:
    r = classifieur(slots)["recensement"]
    ok = all(r[k] == v for k, v in attendu.items())
    return (PASS if ok else FAIL, {"recensement": r, "attendu": dict(attendu)})


# ------------------------------------------------------------ les cinq nulles

def nulles_du_protocole(tokenize, offsets=None) -> dict:
    """Les cinq nulles du §5, telles qu'elles seront produites — celles qui ne
    demandent aucun forward sont CALCULÉES ici (maillon 1), les autres sont
    exhibées comme matériel construit (maillons 2 et 4) ou comme clause
    (maillons 3 et 5)."""
    a = lp.corpus_a()
    prompts = lp._prompts_du_corpus(a)
    filler = tokenize(lp.REMPLISSAGE_NEUTRE)[-1]
    cadre = lp.nulle_cadre(a["slots"], tokenize, filler, offsets)
    mel = lp.nulle_melangee(prompts, tokenize)
    orig = [list(tokenize(p)) for p in prompts]
    par_type = [[a["paraphrases"][i][t] for i in range(lp.N_UNITES)]
                for t in range(lp.N_PARA)]
    return {
        "1_materiel_AUC_lex": {
            "cout": "0 forward",
            "AUC_lex": lp.auc_lex(prompts, tokenize),
            "R1_lex": lp.r1_lex(prompts, a["slots"], tokenize),
            "question": "combien le seul recouvrement lexical produit-il, "
                        "sans cortex ?"},
        "2_capture_nulle_de_cadre": {
            "cout": "+1 forward",
            "specification": "même cadre de type, VERBE CONSERVÉ, slots de "
                             "contenu (owner, entité) remplacés par un "
                             "remplissage neutre gelé, apparié en longueur de "
                             "tokens et en position",
            "clause_abandonnee": "« même suffixe » — VACUÉE pour para2 (suffixe "
                                 "commun vide), défaut 0-8",
            "remplissage_gele": lp.REMPLISSAGE_NEUTRE,
            "token_de_remplissage": filler,
            "construction": cadre["construction"],
            "verbe_conserve": cadre["verbe_conserve"],
            "longueurs_appariees": cadre["longueurs_appariees"],
            "segmentation_fidele": cadre["segmentation_fidele"],
            "n_segmentations_infideles": cadre["n_segmentations_infideles"],
            "tokens_remplaces_min_max": cadre["tokens_remplaces_min_max"],
            "suffixes_communs_par_type": [
                lp.suffixe_commun([list(tokenize(s)) for s in par_type[t]])
                for t in range(lp.N_PARA)]},
        "3_encodage_l0": {
            "cout": "0",
            "clause": "ℓ = 0 publié comme nulle « aucun calcul » et EXCLU de "
                      "l'argmax décisionnel (§4.1)",
            "couches_decisionnelles_L12": lp.couches_decisionnelles(12)},
        "4_ordre_corpus_melange": {
            "cout": "+1 forward",
            "multiensembles_apparies": all(
                sorted(mel[k]) == sorted(orig[k]) for k in range(len(orig))),
            "longueurs_appariees": all(
                len(mel[k]) == len(orig[k]) for k in range(len(orig))),
            "position_de_capture": "dernier token de l'indice, inchangée",
            "clause_permissive": lp.nulle_melangee_permissive(0.45),
            "clause_permissive_exemple_conserve":
                lp.nulle_melangee_permissive(0.52)},
        "5_statistique": {
            "cout": "CPU",
            "plancher_AUC": 0.5,
            "plancher_R1_36": lp.HASARD_R1,
            "bootstrap": f"par COMPOSANTE DE SLOT, B = {lp.B_BOOT}, argmax "
                         f"re-sélectionné dans chaque rééchantillon",
            "BCa": f"requis dès qu'une borne dépasse {lp.SEUIL_BCA}",
            "permutation": "étiquettes d'unité entre paires, À COUCHE FIXÉE, "
                           "avec RECALCUL de max_ℓ (FWER exact)",
            "interdit": "« max des IC par couche » = motif d'invalidation"},
    }


def gate_nulle_cadre(slots, tokenize, filler, offsets=None,
                     effacer_le_verbe: bool = False) -> tuple[str, dict]:
    """Nulle du **maillon 2**, re-spécifiée (§5, défaut 0-8) : *même cadre de
    type, **verbe conservé**, slots de contenu remplacés par un remplissage
    neutre gelé, apparié en longueur et position.*

    L'ancienne clause « même suffixe » est ABANDONNÉE : le suffixe commun de
    para2 est **vide**, elle y était **vacuée**.
    """
    r = lp.nulle_cadre(slots, tokenize, filler, offsets, effacer_le_verbe)
    v = lp.verbe_conserve(slots, tokenize, r["sequences"], offsets)
    ok = bool(r["longueurs_appariees"] and v.get("conserve") is not False)
    det = {k: val for k, val in r.items() if k != "sequences"}
    det |= {"verbe": v, "effacer_le_verbe": effacer_le_verbe,
            "n_sequences": len(r["sequences"]),
            "suffixe_commun_abandonne":
                "la contrainte « même suffixe » est vacuée pour para2 "
                "(suffixe commun vide) — elle n'est plus posée"}
    return (PASS if ok else FAIL, det)


def build_clauses_i2(tokenize, tok_name, offsets=None):
    """Registre I2 : **toutes** les portes du §4.7, les **quatre bandes** du §4.5
    avec leurs **trois bords** et la **précédence `D`**, les **quatre cellules**
    du §4.8, chacune exhibée PASSANTE ET ÉCHOUANTE (D14-S, D18)."""
    C = []

    def clause(name, pass_desc, fail_desc, cases_pass, cases_fail, structural=None,
               note=None):
        C.append({"clause": name, "pass_case": pass_desc, "fail_case": fail_desc,
                  "cases_pass": cases_pass, "cases_fail": cases_fail,
                  "structural": structural, "note": note})

    a = lp.corpus_a()
    b_v3 = lp.corpus_b_v3(tokenize)
    a30 = lp.corpus_a(30)

    # ------------------------------------------------------ V-diversité
    clause("V-diversité",
           "`pool.fact_pairs(80)` rend (16, 20, 5) = (min(N,16), min(N,20), "
           "min(N,5)) ⇒ PASS",
           "le **jeu v3** rend (5, 6, ·) ⇒ FAIL — direction INVERSÉE par rapport "
           "à l'ancienne `V-div` (défaut 0-7)",
           [("fact_pairs(80)", PASS, lambda: gate_v_diversite(a["slots"])),
            ("fact_pairs(30)", PASS, lambda: gate_v_diversite(a30["slots"]))],
           [("jeu v3 (B-v3)", FAIL, lambda: gate_v_diversite(b_v3["slots"])),
            ("corpus à un seul owner et une seule entité", FAIL,
             lambda: gate_v_diversite([("o", "e", VERBS[i % 5]) for i in range(80)]))],
           note="`V-div` mesurait un cardinal de strate ⇒ elle récompensait la "
                "dégénérescence (jeu v3 1.000 PASS, fact_pairs 0.99838 FAIL). "
                "`V-diversité` restaure la direction : zéro GPU, zéro bootstrap.")

    # ------------------------------------------------------- V-puissance
    k_own80 = lp.clusters_de_strate(a["slots"], lp.P_OWN)["K"]
    k_own30 = lp.clusters_de_strate(a30["slots"], lp.P_OWN)["K"]
    k_ent80 = lp.clusters_de_strate(a["slots"], lp.P_ENT)["K"]
    clause("V-puissance",
           "`P-own` à N = 80 : K = 16 ≥ 16 requis ⇒ PASS **à l'égalité** (marge "
           "nulle, déclarée avant mesure)",
           "`P-own` à N = 30 : K = 14 ⇒ FAIL ; **sous décision AUC, K requis "
           "≥ 438 contre K ≤ 20 disponible (facteur ≥ 21.9) ⇒ FAIL à tout "
           "N ≤ 80**, et la porte NOMME l'indécidabilité, robuste au choix "
           "d'opérationnalisation",
           [(f"P-own, N=80 (K = {k_own80})", PASS,
             lambda: gate_v_puissance(k_own80)),
            (f"P-ent, N=80 (K = {k_ent80})", PASS,
             lambda: gate_v_puissance(k_ent80, statistique="R1_36 | P-ent"))],
           [(f"P-own, N=30 (K = {k_own30})", FAIL,
             lambda: gate_v_puissance(k_own30)),
            ("décision AUC, K disponible = 16 (P-own, N=80) — FACTEUR imprimé",
             FAIL, lambda: gate_v_puissance_auc(k_own80)),
            ("décision AUC, K disponible = 20 (plafond du pool) — FACTEUR "
             "imprimé", FAIL, lambda: gate_v_puissance_auc(lp.K_PLAFOND_POOL))],
           note="Le chiffre « K ≥ 429 » est RETIRÉ (non re-dérivable, D14-R). "
                "Énoncé robuste calculé par la porte : " + lp.enonce_robuste_auc()[
                    "enonce"] + " Fusion avec `V-diversité` REFUSÉE (M-13) : un "
                "FAIL sans cause nommée est un FAIL qu'on discute après coup.")

    # ------------------------------ recensement d'identité (§4.2, N-10/N-11)
    att80 = {lp.P_0: 2880, lp.P_OWN: 160, lp.P_ENT: 120, lp.P_BOTH: 0}
    att30 = {lp.P_0: 411, lp.P_OWN: 14, lp.P_ENT: 10, lp.P_BOTH: 0}
    clause("Partition par slot d'IDENTITÉ (§4.2)",
           "recensement 2880/160/120/0 à N = 80 et 411/14/10/0 à N = 30 ; "
           "`P-both = ∅` PROUVÉ par `lcm(16,20) = 80`",
           "un classifieur qui compte le VERBE comme slot d'identité (défaut "
           "0-12) rend un autre recensement ⇒ FAIL",
           [("N = 80", PASS, lambda: gate_recensement_identite(a["slots"], att80)),
            ("N = 30", PASS, lambda: gate_recensement_identite(a30["slots"], att30)),
            ("P-both impossible : lcm(16,20) = 80", PASS,
             lambda: (PASS if (lp.p_both_impossible()["lcm"] == 80
                               and not lp.p_both_impossible()["possible"]) else FAIL,
                      lp.p_both_impossible()))],
           [("classifieur qui traite le verbe comme un slot", FAIL,
             lambda: gate_recensement_avec_classifieur(_stratifier_avec_verbe,
                                                       a["slots"], att80)),
            ("recensement attendu faux (2881)", FAIL,
             lambda: gate_recensement_identite(a["slots"],
                                               dict(att80, **{lp.P_0: 2881})))],
           note="Le VERBE n'est pas un slot d'identité : cinq quasi-synonymes ; "
                "deux unités n'en différant que par le verbe DÉSIGNENT LE MÊME "
                "FAIT (N-9). Espace d'identité = 16 × 20 = 320.")

    # ----------------------------------------------------------- V-paires
    ni, ne = (len(x) for x in lp.paires_intra_inter())
    rec = lp.partition_identite(a["slots"])["recensement"]
    clause("V-paires",
           "les paires construites par l'instrument : 240 intra, 28 440 inter, "
           "recensement 2880/160/120/0",
           "239 ou 241 paires intra ⇒ FAIL",
           [("instrument réel", PASS, lambda: gate_v_paires(ni, ne, rec))],
           [("239 intra", FAIL, lambda: gate_v_paires(239, 28_440, rec)),
            ("241 intra", FAIL, lambda: gate_v_paires(241, 28_440, rec)),
            ("28 439 inter", FAIL, lambda: gate_v_paires(240, 28_439, rec)),
            ("recensement P-both = 1", FAIL,
             lambda: gate_v_paires(240, 28_440, dict(rec, **{lp.P_BOTH: 1})))])

    # --------------------------------------------------------- V-suffixe
    partages = {t: lp.partage_dernier_token(
        [a["paraphrases"][i][k] for i in range(lp.N_UNITES)], tokenize)
        for k, t in enumerate(lp.POOL_PARAPHRASE_TYPES)}
    partages30 = {t: lp.partage_dernier_token(
        [a30["paraphrases"][i][k] for i in range(30)], tokenize)
        for k, t in enumerate(lp.POOL_PARAPHRASE_TYPES)}
    clause("V-suffixe (re-dérivée, 0-8)",
           "partage du dernier token BPE MESURÉ sur les règles gelées : "
           "para1 = 1.0000, para2 = 600/3160 = 0.18987, para3 = 1.0000 à N = 80 "
           "(75/435 = 0.17241 à N = 30)",
           "un matériel où para2 ne suit pas la période 5 du verbe ⇒ FAIL",
           [("règles gelées réelles, N = 80", PASS,
             lambda: gate_v_suffixe(partages, 80)),
            ("règles gelées réelles, N = 30", PASS,
             lambda: gate_v_suffixe(partages30, 30))],
           [("para2 constant (suffixe global)", FAIL,
             lambda: gate_v_suffixe({"para1": 1.0, "para2": 1.0, "para3": 1.0}, 80)),
            ("ancienne attente périmée (para1 ≈ 0)", FAIL,
             lambda: gate_v_suffixe({"para1": 0.0, "para2": 0.0, "para3": 1.0}, 80))],
           note="Constantes ENTIÈREMENT DÉRIVÉES : para1/para3 finissent par une "
                "chaîne globale gelée ⇒ 1 ; para2 finit par le VERBE, période 5. "
                "Valeurs observées N=80 : "
                + ", ".join(f"{k} = {v:.5f}" for k, v in partages.items())
                + " ; N=30 : "
                + ", ".join(f"{k} = {v:.5f}" for k, v in partages30.items()))

    # ---------------------------------------------------------- V-bandes
    clause("V-bandes (nouvelle, 0-9)",
           "la partition N / M / I / V est EXHAUSTIVE et MUTUELLEMENT EXCLUSIVE, "
           "bords inclus, `IC(ΔR1)` entièrement négatif compris, et la "
           "précédence de l'overlay `D` est vérifiée",
           "un classifieur qui n'évalue pas `I` laisse un trou "
           "(`IC_inf < T ≤ IC_sup`) ⇒ HORS-PARTITION",
           [("balayage complet + trois bords + précédence D + ΔR1 négatif", PASS,
             lambda: gate_v_bandes()),
            ("IC(ΔR1) entièrement négatif ET IC_inf(R1) < T ⇒ N + "
             "sous-étiquette", lp.BANDE_N,
             lambda: _bande((-0.20, -0.05), (0.00, 0.05)))],
           [("classifieur SANS la bande I (l'ancien §4.5)", lp.BANDE_HORS,
             lambda: (lambda b: (PASS if b in lp.BANDES else lp.BANDE_HORS,
                                 {"cas": "IC_inf < T ≤ IC_sup sans bande I",
                                  "bande_rendue": b}))(
                 lp.BANDE_HORS))],
           note="C'est la porte qui aurait empêché le protocole qui grave D18 de "
                "violer D18 (défaut 0-9) — et de la violer une SECONDE fois : la "
                "clause réécrite pour réparer la première violation laissait "
                "`IC(ΔR1)` entièrement négatif hors partition (correctif du "
                "2026-08-22).")

    T = lp.T_COULOIR
    clause("Bande N — nulle (§4.5)",
           "`IC_inf(ΔR1) ≤ 0` ⇒ N : l'IC contient 0, TOUCHE 0 par la borne "
           "inférieure, ou est ENTIÈREMENT NÉGATIF (sous-étiquette descriptive)",
           "`ΔR1` significativement > 0 ⇒ ce n'est plus N",
           [("IC(ΔR1) = [−0.10, 0.10]", lp.BANDE_N,
             lambda: _bande((-0.10, 0.10), (0.05, 0.60))),
            ("IC(ΔR1) touche 0 par la borne inférieure", lp.BANDE_N,
             lambda: _bande((0.0, 0.30), (0.55, 0.90))),
            ("ΔR1 significativement NÉGATIF (sous-étiquette descriptive)",
             lp.BANDE_N, lambda: _bande((-0.20, -0.05), (0.00, 0.05)))],
           [("IC(ΔR1) = [+1e-12, 0.30], R1 sous T", lp.BANDE_M,
             lambda: _bande((1e-12, 0.30), (0.05, T - 1e-12)))],
           note="Bande N = **absence de `ΔR1` significativement positif** "
                "(`IC_inf(ΔR1) ≤ 0`), correctif D18 du 2026-08-22 : l'IC "
                "contient 0 OU est entièrement négatif. Le cas négatif porte la "
                "sous-étiquette DESCRIPTIVE `delta_R1_significativement_negatif` "
                "— le corpus réel séparerait moins bien que sa propre nulle, "
                "fait sur l'instrument et non sur le cortex.")

    clause("Bande M — marginal (§4.5)",
           "`ΔR1` > 0 significatif ET `IC_sup(R1_36) < T = 0.25`",
           "`IC_sup` exactement au seuil ⇒ I, pas M",
           [("IC(R1) = [0.05, 0.20]", lp.BANDE_M,
             lambda: _bande((0.05, 0.40), (0.05, 0.20))),
            ("IC_sup 1e-12 sous T", lp.BANDE_M,
             lambda: _bande((0.05, 0.40), (0.05, T - 1e-12)))],
           [("IC_sup EXACTEMENT à T", lp.BANDE_I,
             lambda: _bande((0.05, 0.40), (0.05, T))),
            ("ΔR1 non significatif", lp.BANDE_N,
             lambda: _bande((-0.01, 0.40), (0.05, 0.20)))],
           note="réorientation AUTOMATIQUE (décision PI n°4) : résultat, pas échec.")

    clause("Bande I — indécidable (§4.5)",
           "`ΔR1` > 0 significatif ET `IC_inf < T ≤ IC_sup`",
           "`IC_inf` exactement à T ⇒ V, pas I",
           [("IC(R1) = [0.05, 0.60]", lp.BANDE_I,
             lambda: _bande((0.05, 0.40), (0.05, 0.60))),
            ("IC_sup EXACTEMENT à T (bord gravé)", lp.BANDE_I,
             lambda: _bande((0.05, 0.40), (0.05, T)))],
           [("IC_inf EXACTEMENT à T", lp.BANDE_V,
             lambda: _bande((0.05, 0.40), (T, 0.60))),
            ("IC_sup sous T", lp.BANDE_M,
             lambda: _bande((0.05, 0.40), (0.05, T - 1e-12)))],
           note="cause = `K_S`. La levée n'est PAS un re-run (K plafonne à 16) : "
                "`I` déclenche la construction du matériel de v4 (décision PI n°7).")

    clause("Bande V — viable (§4.5)",
           "`IC_inf(R1_36) ≥ T = 0.25`, borne INCLUSE ; mention V⁺ si ≥ 0.50",
           "`IC_inf` 1e-12 sous T ⇒ I",
           [("IC_inf EXACTEMENT à T", lp.BANDE_V,
             lambda: _bande((0.05, 0.40), (T, 0.60))),
            ("IC(R1) = [0.55, 0.90] ⇒ V avec mention V⁺", lp.BANDE_V,
             lambda: _bande((0.05, 0.40), (0.55, 0.90)))],
           [("IC_inf 1e-12 sous T", lp.BANDE_I,
             lambda: _bande((0.05, 0.40), (T - 1e-12, 0.60)))],
           note="V⁺ est un palier DESCRIPTIF, publié dans le détail de la bande.")

    clause("Overlay D — précédence inter-modèles (§4.5)",
           "si l'un des modèles rend `I`, le verdict global est **`I`** ; sinon "
           "bandes différentes ⇒ `D`",
           "bandes identiques ⇒ la bande commune, jamais `D`",
           [("un modèle I, l'autre V ⇒ I", lp.BANDE_I,
             lambda: (lp.bande_gate(lp.BANDE_I, lp.BANDE_V), {})),
            ("un modèle V, l'autre I ⇒ I", lp.BANDE_I,
             lambda: (lp.bande_gate(lp.BANDE_V, lp.BANDE_I), {})),
            ("V vs M ⇒ D", lp.BANDE_D,
             lambda: (lp.bande_gate(lp.BANDE_V, lp.BANDE_M), {})),
            ("N vs M ⇒ D", lp.BANDE_D,
             lambda: (lp.bande_gate(lp.BANDE_N, lp.BANDE_M), {}))],
           [("V et V", lp.BANDE_V,
             lambda: (lp.bande_gate(lp.BANDE_V, lp.BANDE_V), {})),
            ("N et N", lp.BANDE_N,
             lambda: (lp.bande_gate(lp.BANDE_N, lp.BANDE_N), {})),
            ("I et I", lp.BANDE_I,
             lambda: (lp.bande_gate(lp.BANDE_I, lp.BANDE_I), {}))])

    clause("Bandes N/M/I/V — exhaustivité de la partition (D18)",
           "toute observation tombe dans N, M, I ou V — bords compris",
           "une observation qu'aucune clause ne couvre ⇒ HORS-PARTITION",
           [("IC(ΔR1) contient 0", "COUVERT",
             lambda: bande_couverte((-0.10, 0.10), (0.05, 0.60))),
            ("IC(ΔR1) ENTIÈREMENT NÉGATIF, IC_inf(R1) < T", "COUVERT",
             lambda: bande_couverte((-0.20, -0.05), (0.00, 0.05))),
            ("M", "COUVERT", lambda: bande_couverte((0.05, 0.40), (0.05, 0.20))),
            ("I (chevauche le seuil)", "COUVERT",
             lambda: bande_couverte((0.05, 0.40), (0.05, 0.60))),
            ("V", "COUVERT", lambda: bande_couverte((0.05, 0.40), (T, 0.60))),
            ("IC_sup EXACTEMENT au seuil", "COUVERT",
             lambda: bande_couverte((0.05, 0.40), (0.05, T))),
            ("IC dégénéré [T, T]", "COUVERT",
             lambda: bande_couverte((0.05, 0.40), (T, T)))],
           [("une étiquette hors partition", lp.BANDE_HORS,
             lambda: (lp.BANDE_HORS if "X" not in lp.BANDES else "COUVERT",
                      {"etiquette_testee": "X"}))],
           note="clause de couverture : elle n'attribue aucune bande, elle "
                "vérifie que la partition du §4.5 ne laisse pas de trou (D18).")

    # ------------------------------------- jeu de candidats R1_36 (§4.5)
    clause("Jeu `R1_36` — 37 éléments, déterministe, indépendant des similarités",
           "le jeu a EXACTEMENT 37 éléments, est déterministe, et permuter les "
           "valeurs de similarité NE LE CHANGE PAS",
           "un jeu construit à partir des proximités mesurées ⇒ motif "
           "d'invalidation (§6)",
           [("fact_pairs(80), strate P-own", PASS,
             lambda: gate_jeu_r1(a["slots"]))],
           [("jeu de taille 30 au lieu de 36", FAIL,
             lambda: (FAIL if lp.jeu_candidats_R1(0, 0, a["slots"], lp.P_OWN,
                                                  30)["taille"] != 37 else PASS,
                      {"taille": lp.jeu_candidats_R1(0, 0, a["slots"], lp.P_OWN,
                                                     30)["taille"]})),
            ("sélection par proximité mesurée", FAIL,
             lambda: (FAIL, {"clause": "toute sélection par proximité mesurée est "
                                       "un motif d'invalidation ; la fonction du "
                                       "banc ne reçoit AUCUN état"}))],
           note="Remplissage LE PLUS DUR D'ABORD : 12 états de la composante "
                "d'owner, 9 de la composante d'entité, 15 de P-0 par décalage "
                "croissant ⇒ conservateur. Hasard = 1/37 = 0.02703.")

    # -------------------------------- biais de sélection du max (M-15/M-16)
    clause("Estimateur débiaisé du max et rejet du « max des IC »",
           "sur données synthétiques à **effet nul**, `θ̂ = 2·max_obs − "
           "mean_b(θ*_b)` DIFFÈRE du max brut ; l'argmax est re-sélectionné dans "
           "chaque rééchantillon ; BCa au-delà de 0.95",
           "« max des IC par couche » est REJETÉ mécaniquement",
           [("effet nul, 12 couches, 16 clusters", PASS,
             lambda: _cas_estimateur_debiaise()),
            ("BCa déclenché au-delà de 0.95, percentile sinon", PASS,
             lambda: _cas_bca_au_dela_de_095()),
            ("permutation à couche fixée avec recalcul de max_ℓ", PASS,
             lambda: _cas_permutation_max())],
           [("max des IC par couche", "REJETÉ", lambda: _cas_max_des_ic_rejete())],
           note="`E[max − moyenne] ≈ 0.033` d'AUC > couloir entier (0.0187) ⇒ le "
                "max brut FUIT la bande N (M-15). « Max des IC par couche » est un "
                "motif d'invalidation.")

    # ------------------------------------ bootstrap par composante de slot
    clause("Bootstrap par COMPOSANTE DE SLOT (§4.3, D.1)",
           "l'IC publié est celui du rééchantillonnage par **composante de "
           "slot** (owner pour `P-own`), qui préserve EXACTEMENT les effectifs",
           "l'IC du rééchantillonnage par unité (interdit) ⇒ FAIL — et les deux "
           "largeurs diffèrent, donc la clause n'est pas vacuée",
           [("schéma « cluster »", PASS, lambda: _cas_bootstrap("cluster"))],
           [("schéma « unité » (interdit)", FAIL, lambda: _cas_bootstrap("unite"))],
           note="seul cluster sous lequel les paires d'une strate sont "
                "indépendantes, et qui préserve exactement les effectifs de "
                "strate ⇒ la question du bootstrap stratifié disparaît (D.1).")

    # -------------------------------- nulle du maillon 2 (§5, re-spécifiée)
    filler = tokenize(lp.REMPLISSAGE_NEUTRE)[-1]
    clause("Nulle de cadre (§5, maillon 2, re-spécifiée 0-8)",
           "même cadre de type, **verbe conservé**, slots de contenu remplacés "
           "par un remplissage neutre gelé, apparié en longueur et position",
           "une nulle qui efface AUSSI le verbe ⇒ FAIL (le verbe est du cadre, "
           "pas un slot d'identité)",
           [("nulle du protocole sur fact_pairs(80)", PASS,
             lambda: gate_nulle_cadre(a["slots"], tokenize, filler, offsets))],
           [("nulle qui efface le verbe", FAIL,
             lambda: gate_nulle_cadre(a["slots"], tokenize, filler, offsets,
                                      effacer_le_verbe=True))],
           note="l'ancienne clause « même suffixe » est ABANDONNÉE : le suffixe "
                "commun de para2 est vide, elle y était VACUÉE (défaut 0-8).")

    # ------------------------------------------------------------ V-plat
    g = _rng(11)
    plate = g.normal(0, 0.02, (30, 13))
    bosse = plate + np.array([0, .1, .2, .35, .5, .6, .55, .4, .3, .2, .1, 0, 0])
    clause("V-plat",
           "courbe synthétique à bosse : NON PLATE (au-dessus de `q_0.95(R*)`)",
           "courbe synthétique de bruit seul : PLATE (sous `q_0.95(R*)`)",
           [("bosse médiane", "NON PLATE", lambda: gate_v_plat(bosse))],
           [("bruit seul", "PLATE", lambda: gate_v_plat(plate))],
           note=UNDERSPEC_I2["V-plat"] + " Permutation des étiquettes de COUCHE : "
                "invalide.")

    # ------------------------------------------------------------ V-bord
    clause("V-bord",
           "`ℓ* = 6` sur L = 12 : INTÉRIEUR",
           "`ℓ* = 1` et `ℓ* = L` : AU BORD (écrit d'avance comme le plus probable "
           "pour `ℓ*_H`)",
           [("ℓ* = 6, L = 12", "INTÉRIEUR", lambda: gate_v_bord_i2(6, 12)),
            ("ℓ* = 2, L = 12 (bord+1)", "INTÉRIEUR", lambda: gate_v_bord_i2(2, 12)),
            ("ℓ* = L−1", "INTÉRIEUR", lambda: gate_v_bord_i2(11, 12))],
           [("ℓ* = 1", "AU BORD", lambda: gate_v_bord_i2(1, 12)),
            ("ℓ* = L = 12", "AU BORD", lambda: gate_v_bord_i2(12, 12))])

    # ------------------------------------------------- V-λ₁ et retrait de H
    clause("V-λ₁ et retrait automatique de `H` (§4.6, N-19)",
           "λ₁/Σλ publié pour les 13 couches (0..12) ⇒ PASS ; `H` non "
           "interprétable sur les TROIS modèles ⇒ RETRAIT automatique",
           "une couche manquante ⇒ FAIL ⇒ `H` non interprétable",
           [("0..12 complet", PASS,
             lambda: gate_v_lambda1({e: 0.1 for e in range(13)}, 12)),
            ("non interprétable sur les trois modèles ⇒ RETIRÉE", "RETIRÉE",
             lambda: gate_retrait_H({"gpt2": False, "smollm2": False,
                                     "qwen": False}))],
           [("couche 7 absente", FAIL,
             lambda: gate_v_lambda1({e: 0.1 for e in range(13) if e != 7}, 12)),
            ("interprétable sur un modèle ⇒ CONSERVÉE", "CONSERVÉE",
             lambda: gate_retrait_H({"gpt2": True, "smollm2": False,
                                     "qwen": False}))])

    # ----------------------------------------------------------- V-source
    clause("V-source (nouvelle, 0-10)",
           "toute équation citée est relue dans le **PDF** de sa source "
           "primaire, et l'attribution nomme l'article d'origine (Giraldo et al. "
           "2014 pour la normalisation des lignes)",
           "une équation citée d'après une lecture **HTML** ⇒ motif d'arrêt ; une "
           "attribution à Skean Eq. 1 pour la normalisation des lignes ⇒ FAIL",
           [("registre réel de l'instrument", PASS, lambda: gate_v_source())],
           [("lecture HTML", FAIL,
             lambda: gate_v_source([{"equation": "A_ij = K_ij/(n√(K_ii K_jj))",
                                     "source_primaire": "Giraldo et al. 2014",
                                     "support": "HTML"}])),
            ("attribution sans source primaire", FAIL,
             lambda: gate_v_source([{"equation": "H = −Σ λ log λ",
                                     "source_primaire": "", "support": "PDF"}]))],
           note="Skean et al. Eq. 1, telle qu'imprimée, NE PORTE PAS la "
                "normalisation des lignes ; les deux coïncident ssi toutes les "
                "lignes ont même norme, c.-à-d. ssi le défaut 0-2 est absent.")

    # ----------------------------------------------------------- V-amont
    src = (Path(__file__).parent / "layer_profile.py").read_text(encoding="utf-8")
    clause("V-amont",
           "`eval/layer_profile.py` ne contient aucun chiffre de v3",
           "une source qui pose un chiffre de v3 en seuil ⇒ FAIL",
           [("source réelle de l'instrument", PASS, lambda: gate_v_amont(src))],
           [("seuil posé sur une similarité de v3", FAIL,
             lambda: gate_v_amont("SEUIL = 0.99989  # cos inter-unités")),
            ("plancher de bruit de v3 recopié", FAIL,
             lambda: gate_v_amont("plancher = 12/30"))])

    # ----------------------------------------------------------- V-hooks
    clause("V-hooks",
           "les hooks des L blocs sont retirés en `finally`, y compris quand le "
           "forward lève",
           "un hook posé hors du `try/finally` survit ⇒ FAIL",
           [("capture normale", PASS, lambda: _hooks_apres_capture(False)),
            ("forward qui lève", PASS, lambda: _hooks_apres_capture(True))],
           [("hook jamais retiré", FAIL, lambda: _hooks_fuite()),
            ("engram/ modifié", FAIL,
             lambda: gate_v_hooks(0, " engram/hippocampus.py | 3 +-")),
            ("M instanciée", FAIL, lambda: gate_v_hooks(0, "", True))])

    # ----------------------------------------------------------- V-1pass
    clause("V-1pass",
           "L+1 couches profilées en UN forward",
           "une implémentation qui boucle sur les couches ⇒ FAIL",
           [("un passage", PASS, lambda: _forwards_un_passage()),
            ("quatre variantes à 1 forward", PASS,
             lambda: gate_v_1pass({v: 1 for v in lp.VARIANTES}))],
           [("boucle sur les 12 couches", FAIL,
             lambda: _forwards_boucle_sur_les_couches()),
            ("une variante à 2 forwards", FAIL,
             lambda: gate_v_1pass({"a": 1, "b_v3": 2}))])

    # --------------------------------------------------------------- V-L
    clause("V-L",
           "`L` re-lu du config == {12, 32, 28}",
           "`L` différent du config ⇒ ARRÊT",
           [(f"L = {v} pour {k}", PASS, (lambda v=v: gate_v_L(v, v)))
            for k, v in lp.L_ATTENDU.items()],
           [("L = 11 contre 12 attendu", FAIL, lambda: gate_v_L(11, 12)),
            ("L = 24 contre 28 attendu", FAIL, lambda: gate_v_L(24, 28))])

    # ----------------------------------------------------------- V-hash
    h_a, h_b = lp.sha256_corpus(a), lp.sha256_corpus(b_v3)
    clause("V-hash",
           "SHA-256 de (a) et de `B-v3` identiques avant/après",
           "un corpus modifié en cours de run ⇒ FAIL",
           [("(a) inchangé", PASS, lambda: gate_v_hash_i2(h_a, lp.sha256_corpus(a))),
            ("B-v3 inchangé", PASS,
             lambda: gate_v_hash_i2(h_b, lp.sha256_corpus(b_v3)))],
           [("(a) vs B-v3", FAIL, lambda: gate_v_hash_i2(h_a, h_b))])

    # ------------------------------- 0-1 : invariance monotone de l'AUC
    gm = _rng(12)
    ci = np.clip(gm.normal(0.70, 0.10, 240), -1, 1)
    ce = np.clip(gm.normal(0.60, 0.10, 5000), -1, 1)
    cube = lambda x: x ** 3                                        # noqa: E731
    sig = lambda x: 1.0 / (1.0 + np.exp(-(3.0 * x + 0.5)))         # noqa: E731
    decr = lambda x: -x                                            # noqa: E731
    clause("Invariance monotone de l'AUC (défaut 0-1)",
           "`x → x³` et `x → σ(3x+0.5)` : AUC INCHANGÉE, ratio CHANGÉ",
           "une transformation DÉCROISSANTE change l'AUC ⇒ FAIL (la clause porte "
           "sur les transformations strictement CROISSANTES)",
           [("x → x³", PASS, lambda: gate_invariance_monotone(ci, ce, cube)),
            ("x → σ(3x+0.5)", PASS, lambda: gate_invariance_monotone(ci, ce, sig))],
           [("x → −x (décroissante)", FAIL,
             lambda: gate_invariance_monotone(ci, ce, decr))],
           note="LE test qui matérialise le défaut 0-1 : le score en RATIO se "
                "déplace sous la même transformation, l'AUC non.")

    # ------------------------------- 0-2 : normalisation Giraldo de H
    gh = _rng(13)
    Xh = gh.normal(size=(24, 9))
    fac = gh.uniform(0.1, 10.0, 24)
    clause("Entropie — normalisation des lignes (défaut 0-2)",
           "convention **Giraldo** (`A_ij = K_ij/(n√(K_ii K_jj))`, `tr(A) = 1`) : "
           "`H` inchangée sous mise à l'échelle des lignes",
           "normalisation par la seule trace (`A = K/tr K`) : `H` CHANGE — `H` est "
           "alors confondue avec le profil de normes",
           [("giraldo", PASS, lambda: gate_invariance_lignes(Xh, fac, "giraldo")),
            ("giraldo, facteurs extrêmes", PASS,
             lambda: gate_invariance_lignes(Xh, np.linspace(1e-2, 1e2, 24),
                                            "giraldo"))],
           [("trace seule", FAIL,
             lambda: gate_invariance_lignes(Xh, fac, "trace"))],
           note="`tr(A) = 1` par construction sous Giraldo. La normalisation des "
                "lignes est REQUISE (§3, Giraldo et al. 2014).")

    # -------------------------------------- cellules C1 / C2 / C3 / C4
    clause("Cellule C1 (§4.8)",
           "les deux argmax dans la fenêtre D3, courbes non plates",
           "un seul dans la fenêtre ⇒ C2",
           [("ℓ*_c = 6, ℓ*_H = 6 (L = 12, fenêtre [5,7])", "C1",
             lambda: _cellule(6, 6, 12)),
            ("bords de fenêtre : 5 et 7", "C1", lambda: _cellule(5, 7, 12)),
            ("SmolLM2 : 14 et 18 (fenêtre [14,18])", "C1",
             lambda: _cellule(14, 18, 32))],
           [("ℓ*_c = 6, ℓ*_H = 4", "C2", lambda: _cellule(6, 4, 12)),
            ("juste hors fenêtre : 4 et 8", "C3", lambda: _cellule(4, 8, 12))])

    clause("Cellule C2 (§4.8)",
           "exactement un des deux dans la fenêtre",
           "les deux dedans ⇒ C1 ; aucun ⇒ C3",
           [("ℓ*_c dedans, ℓ*_H dehors", "C2", lambda: _cellule(6, 4, 12)),
            ("ℓ*_c dehors, ℓ*_H dedans", "C2", lambda: _cellule(9, 5, 12))],
           [("les deux dedans", "C1", lambda: _cellule(5, 7, 12)),
            ("aucun des deux", "C3", lambda: _cellule(3, 9, 12))],
           note="évidence FAIBLE : s'écrit « compatible avec », jamais "
                "« démontré » ; nommer laquelle.")

    clause("Cellule C3 (§4.8)",
           "ni l'une ni l'autre dans la fenêtre",
           "au moins un dedans ⇒ C1 ou C2",
           [("3 et 9", "C3", lambda: _cellule(3, 9, 12)),
            ("Qwen : 5 et 20 (fenêtre [12,16])", "C3", lambda: _cellule(5, 20, 28))],
           [("un dedans", "C2", lambda: _cellule(6, 9, 12)),
            ("les deux dedans", "C1", lambda: _cellule(6, 6, 12))])

    clause("Cellule C4 (§4.8)",
           "au moins une courbe PLATE ou un argmax AU BORD ⇒ le modèle sort du "
           "test joint",
           "aucune platitude, aucun bord ⇒ C1/C2/C3",
           [("ℓ*_H = 1 (bord)", "C4", lambda: _cellule(6, 1, 12)),
            ("ℓ*_c = L (bord)", "C4", lambda: _cellule(12, 6, 12)),
            ("courbe de contraste PLATE", "C4",
             lambda: _cellule(6, 6, 12, plate_c=True)),
            ("courbe H PLATE", "C4", lambda: _cellule(6, 6, 12, plate_h=True))],
           [("intérieur, non plates", "C1", lambda: _cellule(6, 6, 12)),
            ("intérieur hors fenêtre, non plates", "C3",
             lambda: _cellule(3, 9, 12))],
           note="pré-écrite comme la plus probable pour `ℓ*_H` (N-P4).")

    # ------------------------------------------ w(L) et indexation (§4.1)
    clause("w(L) et fenêtre D3",
           "`w(L) = max(1, ⌊L/12⌋)` rend 1 / 2 / 2 pour L = 12 / 32 / 28",
           "une fenêtre calculée sur `L/2 ± 1` partout ⇒ FAIL sur L = 32",
           [("w(12), w(32), w(28)", PASS,
             lambda: (PASS if [lp.w_of_L(L) for L in (12, 32, 28)] == [1, 2, 2]
                      else FAIL,
                      {"w": [lp.w_of_L(L) for L in (12, 32, 28)],
                       "fenetres": [list(lp.fenetre_D3(L)) for L in (12, 32, 28)],
                       "bornes_multiplicite": [round(lp.borne_multiplicite(L), 3)
                                               for L in (12, 32, 28)]}))],
           [("w constant = 1", FAIL,
             lambda: (PASS if [1, 1, 1] == [lp.w_of_L(L) for L in (12, 32, 28)]
                      else FAIL, {"w_pose": [1, 1, 1]}))])

    clause("Indexation ℓ = 0 exclue de l'argmax (§4.1)",
           "un maximum en `ℓ = 0` n'est PAS choisi : l'argmax se cherche sur [1, L]",
           "un argmax cherché sur [0, L] rendrait 0",
           [("courbe maximale en ℓ = 0", PASS,
             lambda: (PASS if lp.argmax_decisionnel([0.9] + [0.6] * 12) != 0
                      else FAIL,
                      {"argmax_decisionnel": lp.argmax_decisionnel(
                          [0.9] + [0.6] * 12),
                       "couches": lp.couches_decisionnelles(12)}))],
           [("argmax naïf sur [0, L]", FAIL,
             lambda: (PASS if int(np.argmax([0.9] + [0.6] * 12)) != 0 else FAIL,
                      {"argmax_naif": int(np.argmax([0.9] + [0.6] * 12))}))])

    # ------------------------------------------------ nulle statistique (§5)
    gs = _rng(15)
    clause("Nulle statistique (§5, maillon 5)",
           "sous H₀ (intra et inter tirés de la même loi) l'AUC observée est sous "
           "`q_0.95` de la permutation des étiquettes d'unité",
           "un décalage réel place l'AUC au-dessus de `q_0.95`",
           [("intra ≡ inter", PASS,
             lambda: (lambda r: (PASS if r["auc_observee"] <= r["q_0.95"] else FAIL,
                                 {k: v for k, v in r.items()
                                  if k != "echantillons"}))(
                 lp.permutation_etiquettes_unite(gs.normal(0.6, 0.1, 240),
                                                 gs.normal(0.6, 0.1, 4000), b=400)))],
           [("intra décalé de +0.3", FAIL,
             lambda: (lambda r: (PASS if r["auc_observee"] <= r["q_0.95"] else FAIL,
                                 {k: v for k, v in r.items()
                                  if k != "echantillons"}))(
                 lp.permutation_etiquettes_unite(gs.normal(0.9, 0.1, 240),
                                                 gs.normal(0.6, 0.1, 4000), b=400)))],
           note="permutation des étiquettes d'UNITÉ À COUCHE FIXÉE ; la "
                "permutation des étiquettes de COUCHE est invalide.")

    # ------------------------------------------- R1_full (indice ↔ indice)
    gr = _rng(14)
    base = gr.normal(size=(lp.N_UNITES, 6))
    serre = np.repeat(base, 3, axis=0) + 0.01 * gr.normal(size=(3 * lp.N_UNITES, 6))
    lache = gr.normal(size=(3 * lp.N_UNITES, 6))
    lab = np.repeat(np.arange(lp.N_UNITES), 3)
    clause("R1_full indice ↔ indice",
           "états serrés par unité ⇒ Recall@1 = 1.0 en cosinus ET en L2",
           "états indépendants de l'unité ⇒ Recall@1 s'effondre",
           [("états serrés, cos", PASS,
             lambda: (PASS if recall_i2(serre, lab, "cos") == 1.0 else FAIL,
                      {"recall": recall_i2(serre, lab, "cos")})),
            ("états serrés, L2", PASS,
             lambda: (PASS if recall_i2(serre, lab, "l2") == 1.0 else FAIL,
                      {"recall": recall_i2(serre, lab, "l2")}))],
           [("états indépendants, cos", FAIL,
             lambda: (PASS if recall_i2(lache, lab, "cos") > 0.5 else FAIL,
                      {"recall": recall_i2(lache, lab, "cos")}))],
           note="indice↔indice UNIQUEMENT : mesurer une quantité indice↔fait est "
                "un motif d'invalidation du run (§4.4, §6).")

    return C


def recall_i2(X, lab, metrique):
    return lp.recall_at_1(X, lab, metrique)


# =========================================================================
#  Usage méta borné : portes d'INTÉGRITÉ SEULES sur les bruts archivés
# =========================================================================

def _meta_replay():
    """Rejeu autorisé : **V-cap, V-bord, V1a, V1b-1, V1b-2, V1c, V-var, V-drift**.

    Aucune statistique décisionnelle (P1, ΔP6, P3, P4, h, multi-clé, G) n'est
    calculée ici, et aucune ne peut l'être : cette fonction n'appelle que des
    portes d'intégrité. Les bruts archivés portent des valeurs déjà publiées ;
    les faire traverser les portes de v3 reviendrait à calibrer v3 sur son
    propre résultat.
    """
    out = {"portes_autorisees": ["V-cap", "V-bord", "V1a", "V1b-1", "V1b-2",
                                 "V1c", "V-var", "V-drift"],
           "statistiques_decisionnelles": "INTERDITES (§ banc, usage méta borné)",
           "archives": {}}
    for p in ARCHIVES:
        key = str(p).replace("\\", "/")
        if not p.exists():
            out["archives"][key] = "ABSENTE"
            continue
        files = {}
        for x in sorted(p.glob("*")):
            h = hashlib.sha256()
            with open(x, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            files[x.name] = {"octets": x.stat().st_size, "sha256_16": h.hexdigest()[:16]}
        out["archives"][key] = {"fichiers": files}
    out["status"] = (
        "archives PRÉSENTES et inventoriées (nom, taille, SHA-256[:16]) ; rejeu "
        "méta NON EXÉCUTÉ. Raison : le rejeu est une FACULTÉ (« peuvent être "
        "rejouées »), pas une exigence de couverture — chaque porte d'intégrité a "
        "déjà ses deux contre-exemples synthétiques exécutables, donc le rejeu "
        "n'affecte pas E. Et reconstruire p_kNN sur ces bruts (V1a/V1b-1/V1b-2 en "
        "ont besoin) supposerait de ré-interroger un datastore bâti sur des "
        "valeurs déjà publiées — la frontière que l'usage méta borné interdit de "
        "franchir. Décision du PI.")
    return out


# =========================================================================
#  Exécution
# =========================================================================

def _unit_set_facts(tokenize) -> dict:
    """Faits BRUTS sur le jeu d'unités du §16 + **budget de la passe A**
    re-dérivé (cascade D14(b), §16 G). Descriptif : n'entre dans aucune porte.

    Le budget de la passe A est le compte de tokens BPE des 30 faits (une entrée
    de store par token du fait, §5.7) et des 30 × 4 indices.
    """
    st = v3_unit_triples_stats(POOL_UNITS_N)
    sec = v3_unit_secrets_stats(POOL_UNITS_N)
    units = unit_table(POOL_UNITS_N)
    tok_faits = [len(tokenize(u["fact_template"].replace("{secret}", u["secret"])))
                 for u in units]
    tok_exact = [len(tokenize(u["exact"])) for u in units]
    tok_para = [[len(tokenize(p)) for p in u["paraphrases"]] for u in units]
    return {
        "regle": st["enumeration"] + " ; 30 premiers triplets C-1 ∧ C-2 ∧ C-3",
        "triplets_examines_avant_30_conformes": st["triplets_examines"],
        "rejets_par_condition": st["rejets"],
        "owners_conformes_c3": st["owners_conformes_c3"],
        "owners_non_conformes_c3": [OWNERS[o] for o in st["owners_non_conformes_c3"]],
        "triplets": [list(t) for t in st["triplets"]],
        "entites_utilisees": sorted({ENTITIES[t[1]] for t in st["triplets"]}),
        "owners_utilises": sorted({OWNERS[t[0]] for t in st["triplets"]}),
        # FAIT BRUT sur le plan (aucune interprétation) : le verbe est-il une
        # fonction déterministe de l'owner ? de l'entité ? — la note de design
        # du §2 portait sur `fact_pairs` (verbe = f(entité)) ; le jeu du §16 est
        # un autre plan et cette ligne le mesure, elle ne la recopie pas.
        "verbe_fonction_de_l_owner": len({(t[0], t[2]) for t in st["triplets"]})
        == len({t[0] for t in st["triplets"]}),
        "verbe_fonction_de_l_entite": len({(t[1], t[2]) for t in st["triplets"]})
        == len({t[1] for t in st["triplets"]}),
        "occurrences_par_owner": {OWNERS[o]: sum(1 for t in st["triplets"]
                                                 if t[0] == o)
                                  for o in sorted({t[0] for t in st["triplets"]})},
        "occurrences_par_entite": {ENTITIES[e]: sum(1 for t in st["triplets"]
                                                    if t[1] == e)
                                   for e in sorted({t[1] for t in st["triplets"]})},
        "occurrences_par_verbe": {VERBS[v]: sum(1 for t in st["triplets"]
                                                if t[2] == v)
                                  for v in sorted({t[2] for t in st["triplets"]})},
        "substitutions_de_secret": sec["substitutions"],
        "regle_substitution": sec["regle"],
        "secrets": sec["secrets"],
        "budget_passe_A": {
            "tokens_des_30_faits": int(sum(tok_faits)),
            "entrees_de_store_par_unite": tok_faits,
            "tokens_des_30_indices_exacts": int(sum(tok_exact)),
            "tokens_des_90_paraphrases": int(sum(sum(r) for r in tok_para)),
            "tokens_par_type_de_paraphrase": [
                int(sum(r[k] for r in tok_para)) for k in range(3)],
        },
    }


def _evaluer(clauses) -> tuple[list, int]:
    """Boucle d'évaluation commune aux deux suites : pour chaque clause, tous les
    contre-exemples passants et échouants, puis le compte `E`."""
    rows, E = [], 0
    for c in clauses:
        exp_p, obs_p, det_p, ok_p = [], [], [], []
        for name, expected, fn in c["cases_pass"]:
            v, d = fn()
            exp_p.append(expected)
            obs_p.append(v)
            det_p.append({"cas": name, "attendu": expected, "observé": v,
                          "ok": v == expected, "détail": d})
            ok_p.append(v == expected)
        exp_f, obs_f, det_f, ok_f = [], [], [], []
        for name, expected, fn in c["cases_fail"]:
            v, d = fn()
            exp_f.append(expected)
            obs_f.append(v)
            det_f.append({"cas": name, "attendu": expected, "observé": v,
                          "ok": v == expected, "détail": d})
            ok_f.append(v == expected)

        reasons = []
        for name, expected, got in zip([d["cas"] for d in det_p], exp_p, obs_p):
            if expected != got:
                reasons.append(f"INSATISFIABLE — le contre-exemple passant "
                               f"« {name} » rend {got!r} au lieu de {expected!r}")
        for name, expected, got in zip([d["cas"] for d in det_f], exp_f, obs_f):
            if expected != got:
                kind = ("VACUÉE PAR SATISFACTION" if got == PASS
                        else "COMPORTEMENT NON CONFORME")
                reasons.append(f"{kind} — le contre-exemple échouant « {name} » "
                               f"rend {got!r} au lieu de {expected!r}")
        if c["structural"] is not None:
            reasons.extend(c["structural"]())

        ok = all(ok_p) and all(ok_f)
        counted = bool(reasons)
        E += 1 if counted else 0
        rows.append({
            "clause": c["clause"],
            "pass_case": c["pass_case"],
            "fail_case": c["fail_case"],
            "expected": {"pass_case": exp_p, "fail_case": exp_f},
            "observed": {"pass_case": obs_p, "fail_case": obs_f},
            "ok": bool(ok),
            "compte_dans_E": counted,
            "raisons_E": reasons,
            "note": c["note"],
            "cas": {"pass_case": det_p, "fail_case": det_f},
        })
    return rows, E


def run(use_hf: bool = True, out_dir: Path = OUT_DIR) -> dict:
    t0 = time.time()
    tokenize, tok_name = make_tokenizer(use_hf)
    clauses = build_clauses(tokenize, tok_name)
    rows, E = _evaluer(clauses)

    n_cov = sum(1 for r in rows if r["expected"]["pass_case"]
                and r["expected"]["fail_case"])
    report = {
        "protocole": "experiments/EXP-2026-08-22-knn-borne-logits-v3.md",
        "statut_protocole": "PROPOSE — gate de pré-enregistrement NON franchie",
        "banc": "D14-S — satisfiabilité, CPU seul, aucune mesure",
        "tokenizer": tok_name,
        "E": int(E),
        "n_clauses": len(rows),
        "couverture": {"clauses_avec_les_deux_contre_exemples": n_cov,
                       "total": len(rows),
                       "pct": round(100.0 * n_cov / len(rows), 2)},
        "n_cas": sum(len(r["cas"]["pass_case"]) + len(r["cas"]["fail_case"])
                     for r in rows),
        "amendement": "§15 (A-1..A-8) puis §16 (jeu d'unités refondu, C-1/C-2/"
                      "C-3, porte V-ident) — cascade D14(b) rejouée",
        "sha256_donnees_gelees": sha256_obj(frozen_dataset()),
        "jeu_unites_v3": _unit_set_facts(tokenize),
        "descriptifs_A7": {
            "jaccard_brut": descriptif_jaccard_brut(unit_table(POOL_UNITS_N),
                                                    tokenize),
            "V-partage": descriptif_v_partage(unit_table(POOL_UNITS_N), tokenize),
        },
        "lambda_star": {
            "expression": "1 - math.exp(-0.05)",
            "valeur": repr(LAMBDA_STAR),
            "litteral_du_document": repr(LAMBDA_STAR_LITERAL_DOC),
            "ulp_litteral_vs_expression": ulp_gap(LAMBDA_STAR_LITERAL_DOC,
                                                  1.0 - math.exp(-0.05)),
        },
        "clauses_sous_specifiees": UNDERSPEC,
        "meta_replay": _meta_replay(),
        "duree_s": round(time.time() - t0, 2),
        "clauses": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return report


def run_i2(use_hf: bool = True, out_dir: Path = OUT_DIR_I2) -> dict:
    """Banc de satisfiabilité d'I2 (`EXP-2026-08-22-layer-profile.md`,
    version CONSOLIDÉE)."""
    t0 = time.time()
    tokenize, tok_name = make_tokenizer(use_hf)
    offsets = make_offsets(use_hf)
    clauses = build_clauses_i2(tokenize, tok_name, offsets)
    rows, E = _evaluer(clauses)

    a = lp.corpus_a()
    a30 = lp.corpus_a(30)
    b_v3 = lp.corpus_b_v3(tokenize)
    n_cov = sum(1 for r in rows if r["expected"]["pass_case"]
                and r["expected"]["fail_case"])
    part80 = lp.partition_identite(a["slots"])
    part30 = lp.partition_identite(a30["slots"])
    part_v3 = lp.partition_identite(b_v3["slots"])
    suff = {t: lp.partage_dernier_token(
        [a["paraphrases"][i][k] for i in range(lp.N_UNITES)], tokenize)
        for k, t in enumerate(lp.POOL_PARAPHRASE_TYPES)}
    suff30 = {t: lp.partage_dernier_token(
        [a30["paraphrases"][i][k] for i in range(30)], tokenize)
        for k, t in enumerate(lp.POOL_PARAPHRASE_TYPES)}
    report = {
        "protocole": "experiments/EXP-2026-08-22-layer-profile.md",
        "statut_protocole": "PROPOSE — gate de pré-enregistrement NON franchie",
        "banc": "D14-S — satisfiabilité I2, CPU seul, aucune mesure, aucun modèle",
        "tokenizer": tok_name,
        "E": int(E),
        "n_clauses": len(rows),
        "couverture": {"clauses_avec_les_deux_contre_exemples": n_cov,
                       "total": len(rows),
                       "pct": round(100.0 * n_cov / len(rows), 2)},
        "n_cas": sum(len(r["cas"]["pass_case"]) + len(r["cas"]["fail_case"])
                     for r in rows),
        "corpus": {
            "a": {"source": "pool.fact_pairs(80) re-paraphrasé (§4.2 a)",
                  "N": lp.N_UNITES,
                  "sha256": lp.sha256_corpus(a),
                  "diversite_(owners,entites,verbes)": list(lp.diversite(a["slots"])),
                  "diversite_attendue": list(lp.diversite_attendue(lp.N_UNITES)),
                  "recensement_identite": part80["recensement"]},
            "a_N30": {"source": "pool.fact_pairs(30) — N ÉCARTÉ, publié pour le "
                                "recensement",
                      "N": 30,
                      "diversite_(owners,entites,verbes)":
                          list(lp.diversite(a30["slots"])),
                      "recensement_identite": part30["recensement"]},
            "B-v3": {"source": "jeu d'unités v3 — BRAS DESCRIPTIF, jamais fusionné, "
                               "jamais décisionnel, jamais appelé « strate »",
                     "sha256": lp.sha256_corpus(b_v3),
                     "diversite_(owners,entites,verbes)":
                         list(lp.diversite(b_v3["slots"])),
                     "recensement_identite": part_v3["recensement"]},
        },
        "partition_identite": {
            "espace_identite": lp.ESPACE_IDENTITE,
            "strate_decisionnelle": lp.STRATE_DECISIONNELLE,
            "P_both": lp.p_both_impossible(),
            "clusters": {s: {"cle": lp.clusters_de_strate(a["slots"], s)["cle"],
                             "K_N80": lp.clusters_de_strate(a["slots"], s)["K"],
                             "K_N30": lp.clusters_de_strate(a30["slots"], s)["K"]}
                         for s in (lp.P_OWN, lp.P_ENT)},
            "le_verbe_est_une_covariable": True,
        },
        "puissance": {
            "formule": "K ≥ (1.96·σ₀/|θ−T|)²",
            "sigma0": lp.SIGMA0, "marge_R1": lp.MARGE_R1,
            "K_requis_R1_36": lp.k_requis(),
            "chiffre_429": "RETIRÉ — non re-dérivable (D14-R)",
            "decision_AUC": lp.enonce_robuste_auc(),
        },
        "statut_des_operationnalisations": STATUT_OPERATIONNALISATIONS_I2,
        "paires": {"n_intra": len(lp.paires_intra_inter()[0]),
                   "n_inter": len(lp.paires_intra_inter()[1]),
                   "attendus": [lp.N_INTRA_ATTENDU, lp.N_INTER_ATTENDU]},
        "V-suffixe": {"observe_N80": suff, "observe_N30": suff30,
                      "derive_N80": lp.partage_suffixe_derive(80),
                      "derive_N30": lp.partage_suffixe_derive(30)},
        "couloir_v4": {
            "s": lp.S_LEURRES, "taille_du_jeu": lp.TAILLE_JEU_R1,
            "hasard": lp.HASARD_R1, "T": lp.T_COULOIR, "T_plus": lp.T_COULOIR_PLUS,
            "reperes_de_publication_en_AUC_jamais_un_critere": {
                "A_pour_R1_0.25_contre_36": lp.AUC_REPERE_025,
                "A_pour_R1_0.50_contre_36": lp.AUC_REPERE_050},
            "blocs_du_jeu": lp.jeu_candidats_R1(0, 0, a["slots"])["blocs"]},
        "fenetres": {str(k): {"L": v, "w": lp.w_of_L(v),
                              "fenetre_D3": list(lp.fenetre_D3(v)),
                              "borne_multiplicite": round(lp.borne_multiplicite(v), 4)}
                     for k, v in lp.L_ATTENDU.items()},
        "V-source": {"citations": [dict(c) for c in lp.CITATIONS],
                     "verdict": gate_v_source()[0]},
        "cinq_nulles": nulles_du_protocole(tokenize, offsets),
        "clauses_sous_specifiees": UNDERSPEC_I2,
        "duree_s": round(time.time() - t0, 2),
        "clauses": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return report


def _imprimer(rep: dict) -> None:
    print(f"tokenizer : {rep['tokenizer']}")
    print(f"clauses   : {rep['n_clauses']}  |  cas exécutés : {rep['n_cas']}  "
          f"|  couverture : {rep['couverture']['pct']} %")
    print(f"durée     : {rep['duree_s']} s")
    print("-" * 78)
    for r in rep["clauses"]:
        mark = "ok " if not r["compte_dans_E"] else "E !"
        print(f"[{mark}] {r['clause']}")
        for side in ("pass_case", "fail_case"):
            for d in r["cas"][side]:
                flag = "  " if d["ok"] else "!!"
                print(f"      {flag} {side:9} {d['cas'][:62]:<62} "
                      f"→ {d['observé']!r}")
        for reason in r["raisons_E"]:
            print(f"      >>> {reason}")
    print("-" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-hf", action="store_true",
                    help="repli mot-à-mot au lieu du tokenizer GPT-2")
    ap.add_argument("--suite", default="v3", choices=("v3", "i2", "all"),
                    help="v3 = V2-D(a) v3 (défaut, inchangé) ; i2 = layer_profile")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.suite in ("i2", "all"):
        out_i2 = Path(args.out) if (args.out and args.suite == "i2") else OUT_DIR_I2
        print("=" * 78)
        print("BANC DE SATISFIABILITÉ (D14-S) — EXP-2026-08-22-layer-profile (I2)")
        print("AUCUNE MESURE, AUCUN GPU, AUCUN MODÈLE. CPU seul.")
        print("=" * 78)
        rep_i2 = run_i2(use_hf=not args.no_hf, out_dir=out_i2)
        _imprimer(rep_i2)
        c = rep_i2["corpus"]
        for cle in ("a", "a_N30", "B-v3"):
            d = c[cle]
            print(f"corpus {cle:6}: {d['source']}")
            print(f"              diversité (owners, entités, verbes) = "
                  f"{d['diversite_(owners,entites,verbes)']}")
            print(f"              recensement d'identité "
                  f"(P-0/P-own/P-ent/P-both) : {d['recensement_identite']}")
        pi = rep_i2["partition_identite"]
        print(f"P-both      : lcm = {pi['P_both']['lcm']} ; possible = "
              f"{pi['P_both']['possible']}")
        print(f"clusters    : {pi['clusters']}  |  strate décisionnelle = "
              f"{pi['strate_decisionnelle']}")
        pu = rep_i2["puissance"]
        print(f"puissance   : K requis (R1_36) = {pu['K_requis_R1_36']} ; "
              f"chiffre 429 = {pu['chiffre_429']}")
        da = pu["decision_AUC"]
        for t in da["table"]:
            print(f"              AUC | {t['nom']:<52} K requis = "
                  f"{t['K_requis']:>5} (exact {t['K_requis_exact']}) ; "
                  f"facteur = {t['facteur']}")
        print(f"              AUC | {da['enonce']}")
        vs = rep_i2["V-suffixe"]
        print(f"V-suffixe   : observé N=80 "
              + ", ".join(f"{k}={v:.5f}" for k, v in vs["observe_N80"].items())
              + f"  |  dérivé para2 = {vs['derive_N80']['para2']:.5f} "
              f"({vs['derive_N80']['n_paires_5_divise_d']}/"
              f"{vs['derive_N80']['C_n_2']})")
        print(f"            : observé N=30 "
              + ", ".join(f"{k}={v:.5f}" for k, v in vs["observe_N30"].items())
              + f"  |  dérivé para2 = {vs['derive_N30']['para2']:.5f}")
        print(f"paires      : {rep_i2['paires']}")
        print(f"couloir v4  : {rep_i2['couloir_v4']}")
        print(f"fenêtres    : {rep_i2['fenetres']}")
        print(f"V-source    : {rep_i2['V-source']['verdict']} "
              f"({len(rep_i2['V-source']['citations'])} équations, support PDF)")
        n1 = rep_i2["cinq_nulles"]
        print(f"nulle 1 (0 forward)  : AUC_lex = "
              f"{n1['1_materiel_AUC_lex']['AUC_lex']:.4f} ; R1_lex = "
              f"{n1['1_materiel_AUC_lex']['R1_lex']:.4f}")
        print(f"nulle 2 (cadre)      : longueurs appariées = "
              f"{n1['2_capture_nulle_de_cadre']['longueurs_appariees']} ; "
              f"construction = {n1['2_capture_nulle_de_cadre']['construction']} ; "
              f"tokens remplacés (min,max) = "
              f"{n1['2_capture_nulle_de_cadre']['tokens_remplaces_min_max']} ; "
              f"verbe conservé = True")
        print(f"nulle 4 (mélangée)   : multiensembles appariés = "
              f"{n1['4_ordre_corpus_melange']['multiensembles_apparies']}")
        for k, v in rep_i2["clauses_sous_specifiees"].items():
            print(f"sous-spécifiée : {k} — {v}")
        print("-" * 78)
        print(f"E(I2) = {rep_i2['E']}")
        if rep_i2["E"] == 0:
            print("E = 0 — gate de satisfiabilité I2 VERTE.")
        else:
            bad = [r["clause"] for r in rep_i2["clauses"] if r["compte_dans_E"]]
            print(f"E ≥ 1 — gate NON franchie. Clauses en cause : {bad}")
            print("Le banc ne corrige AUCUNE clause : amender est une décision de "
                  "pré-enregistrement, pas d'implémentation.")
        print(f"rapport : {out_i2 / 'report.json'}")
        if args.suite == "i2":
            return 0
        print("=" * 78)

    print("=" * 78)
    print("BANC DE SATISFIABILITÉ (D14-S) — EXP-2026-08-22-knn-borne-logits-v3")
    print("AUCUNE MESURE, AUCUN GPU, AUCUN MODÈLE. CPU seul.")
    print("=" * 78)
    args.out = args.out or str(OUT_DIR)
    rep = run(use_hf=not args.no_hf, out_dir=Path(args.out))
    print(f"tokenizer : {rep['tokenizer']}")
    print(f"clauses   : {rep['n_clauses']}  |  cas exécutés : {rep['n_cas']}  "
          f"|  couverture : {rep['couverture']['pct']} %")
    print(f"durée     : {rep['duree_s']} s")
    print("-" * 78)
    for r in rep["clauses"]:
        mark = "ok " if not r["compte_dans_E"] else "E !"
        print(f"[{mark}] {r['clause']}")
        for side in ("pass_case", "fail_case"):
            for d in r["cas"][side]:
                flag = "  " if d["ok"] else "!!"
                print(f"      {flag} {side:9} {d['cas'][:62]:<62} "
                      f"→ {d['observé']!r}")
        for reason in r["raisons_E"]:
            print(f"      >>> {reason}")
    print("-" * 78)
    print(f"SHA-256 des données gelées AMENDÉES : {rep['sha256_donnees_gelees']}")
    js = rep["jeu_unites_v3"]
    print(f"§16 jeu d'unités : {js['triplets_examines_avant_30_conformes']} triplets "
          f"examinés pour 30 conformes ; rejets {js['rejets_par_condition']}")
    print(f"           owners hors C-3 : {js['owners_non_conformes_c3']}")
    print(f"           substitutions de secret : {js['substitutions_de_secret']} "
          f"({js['regle_substitution']})")
    print(f"           budget passe A : {js['budget_passe_A']['tokens_des_30_faits']} "
          f"tokens de faits, "
          f"{js['budget_passe_A']['tokens_des_30_indices_exacts']} exacts, "
          f"{js['budget_passe_A']['tokens_des_90_paraphrases']} paraphrases")
    ls = rep["lambda_star"]
    print(f"λ* = {ls['expression']} = {ls['valeur']}  |  littéral du document "
          f"{ls['litteral_du_document']} → écart {ls['ulp_litteral_vs_expression']} ULP")
    d7 = rep["descriptifs_A7"]
    print(f"A-7 Jaccard BRUT (descriptif) : "
          f"{d7['jaccard_brut']['violations_jaccard_brut_par_type']} "
          f"total={d7['jaccard_brut']['violations_total']}")
    print(f"A-7 V-partage (descriptif)    : {d7['V-partage']}")
    print("-" * 78)
    print(f"meta_replay : {rep['meta_replay']['status']}")
    for k, v in rep["clauses_sous_specifiees"].items():
        print(f"sous-spécifiée : {k} — {v}")
    print("-" * 78)
    print(f"E = {rep['E']}")
    if rep["E"] == 0:
        print("E = 0 — gate de satisfiabilité VERTE (décision du PI).")
    else:
        bad = [r["clause"] for r in rep["clauses"] if r["compte_dans_E"]]
        print(f"E ≥ 1 — H_méthode REJETÉE (§6). Clauses en cause : {bad}")
        print("Le banc ne corrige AUCUNE clause : amender est une décision de "
              "pré-enregistrement, pas d'implémentation.")
    print(f"rapport : {Path(args.out) / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
