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
from engram.config import EngramConfig  # noqa: E402

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
#  ===============  SUITE v4 — matériau qualifié (D14-S)  =================
#
#  Protocole : `experiments/EXP-2026-08-23-v4-materiel.md` (PRE-ENREGISTRE).
#  CPU seul, aucune mesure, aucun modèle chargé — seuls les TROIS tokenizers
#  (en cache) sont touchés : `C2`, `C3` et `V-casse` ne sont pas décidables
#  sans eux. **Aucun GPU avant PASS intégral de cette suite** (§6.A).
#
#  Pour CHAQUE clause : un cas passant ET un cas échouant. Plus les cas de
#  banc énumérés au §10 : 4 classes du calibrateur, 5 bandes de la primaire,
#  6 classes ORD, exclusivité aux seuils À JOUR (0-87), `M2 = 0-résolu`
#  distinct de `ind`, famine partielle concentrée, routage `ind`,
#  et `fact_pairs` en CONTRE-EXEMPLE ÉCHOUANT OBLIGATOIRE.
# =========================================================================

import copy as _copy  # noqa: E402

import materiel_v4 as m4  # noqa: E402
import pool_v4 as p4  # noqa: E402

OUT_DIR_V4 = ROOT / "experiments" / "results" / "v4-materiel"

# Opérationnalisations DÉCLARÉES de la suite v4 (hors E, publiées).
UNDERSPEC_V4 = dict(p4.OPERATIONNALISATIONS)
UNDERSPEC_V4.update(m4.OPERATIONNALISATIONS)


def _mat_casse(mat, quoi):
    """Matériau CORROMPU de façon ciblée — support des contre-exemples échouants.

    Le banc ne corrige aucune clause : il fabrique l'état du monde dans lequel
    la clause DOIT échouer, et vérifie qu'elle échoue.
    """
    m = _copy.deepcopy(mat)
    if quoi == "collision-entite":              # C1 / C4 : deux entités égales
        m["unites"][1] = dict(m["unites"][0])
        m["unites_decisionnelles"] = [u for u in m["unites"] if u["pontee"]]
    elif quoi == "Le-3":                        # C2 : une entité à 3 tokens
        m["unites"][0] = dict(m["unites"][0], suffixe="kaleidoscope")
        m["unites_decisionnelles"] = [u for u in m["unites"] if u["pontee"]]
    elif quoi == "prefixes-inegaux":            # C3 : variantes non appariées
        c = dict(m["cellules"]["T1-B"])
        c["mots"] = c["mots"][:-1]
        c["n_mots"] = len(c["mots"])
        c["longueurs_prefixe"] = {k: c["n_mots"] for k in c["longueurs_prefixe"]}
        m["cellules"]["T1-B"] = c
    elif quoi == "sous-viviers-non-disjoints":  # C7
        d0, d1 = p4.DOMAINES[0], p4.DOMAINES[1]
        m["sous_viviers_complets"][d1] = (list(m["sous_viviers_complets"][d1])
                                          + [m["sous_viviers"][d0][0]])
        m["sous_viviers"][d1] = list(m["sous_viviers"][d1])
        m["sous_viviers"][d1][0] = m["sous_viviers"][d0][0]
        for u in m["unites"]:
            if u["domaine"] == d1:
                u["suffixe"] = m["sous_viviers"][d0][0]
                break
        m["unites_decisionnelles"] = [u for u in m["unites"] if u["pontee"]]
    elif quoi == "suffixe-hors-sous-vivier":    # C7, 100 % d'appartenance
        for u in m["unites"]:
            if u["domaine"] == p4.DOMAINES[0]:
                u["suffixe"] = m["sous_viviers"][p4.DOMAINES[1]][0]
                break
        m["unites_decisionnelles"] = [u for u in m["unites"] if u["pontee"]]
    elif quoi == "cadre-Le-1":                  # 0-58 : nulle de cadre à L_e = 1
        m["paires_cadre"] = [[a, ""] for a, _ in m["paires_cadre"]]
    elif quoi == "cadre-token-partage":
        m["paires_cadre"] = list(m["paires_cadre"])
        m["paires_cadre"][1] = [m["paires_cadre"][0][0], m["paires_cadre"][1][1]]
    elif quoi == "cascade-permutee":            # §5.5 : ordre gravé non suivi
        e = list(m["cascade_executee"])
        e[1], e[2] = e[2], e[1]
        m["cascade_executee"] = e
    elif quoi == "tige-morte":                  # K_eff = 9
        mort = m["tiges_pontees"][0]
        m["tiges_pontees"] = m["tiges_pontees"][1:]
        m["unites"] = [u for u in m["unites"] if u["tige"] != mort]
        m["unites_decisionnelles"] = [u for u in m["unites"] if u["pontee"]]
    elif quoi == "deux-types-capitalises":      # C6
        pass                                    # traité par surcharge de MOULES
    return m


def _encs_casse(encs, quoi):
    """Encodeurs FEINTS pour les clauses dont le contre-exemple échouant vit
    dans le tokenizer et non dans le matériau."""
    if quoi == "casse-inerte":
        # une tige dont la minusculisation NE CHANGE PAS le token : `V-casse`
        # devient VRAIE PAR VACUITÉ — troisième occurrence du mode 0-6/0-8.
        def mk(e):
            def f(s):
                return e(s.lower())
            return f
        return {m: (mk(e), n) for m, (e, n) in encs.items()}
    return encs


def build_clauses_v4(mat, encs):
    C = []

    def clause(name, pass_desc, fail_desc, cases_pass, cases_fail, note=None):
        C.append({"clause": name, "pass_case": pass_desc, "fail_case": fail_desc,
                  "cases_pass": cases_pass, "cases_fail": cases_fail,
                  "structural": None, "note": note})

    # ----------------------------------------------------- C1 … C7 + génération
    clause("V-C1", "matériau v4 : 72 séquences byte-identiques hors slot",
           "deux entités en collision ⇒ cardinal < 72",
           [("matériau v4", PASS, lambda: p4.v_c1(mat, encs))],
           [("collision d'entités", FAIL,
             lambda: p4.v_c1(_mat_casse(mat, "collision-entite"), encs))])

    clause("V-C1'", "aucune quantité issue d'un modèle n'est lue à la sélection",
           "une sélection lisant un état ⇒ nulle non exacte",
           [("métadonnées déclarées seules", PASS, lambda: p4.v_c1p(mat))],
           [("sélection sur une quantité de modèle", FAIL,
             lambda: (FAIL, {"quantites_de_modele_lues": ["cos(h_i, h_j)"],
                             "motif": "C1' : échangeabilité conditionnelle "
                                      "détruite ; matériau dépendant du modèle"}))])

    clause("V-C2", "L_e = 2 pour 72/72 entités et 40/40 paires, 3 tokenizers",
           "une entité à L_e = 3 ⇒ rejet ET re-qualification complète",
           [("matériau v4", PASS, lambda: p4.v_c2(mat, encs))],
           [("une entité à L_e = 3", FAIL,
             lambda: p4.v_c2(_mat_casse(mat, "Le-3"), encs))],
           note="C2 : une SEULE entité à L_e ≠ 2 sur un tokenizer ⇒ rejet de "
                "l'entité et re-qualification complète, JAMAIS de rustine locale.")

    clause("V-C3", "indice de capture constant ; variantes appariées en longueur",
           "variantes de longueurs différentes ⇒ 0-43 réintroduit",
           [("matériau v4", PASS, lambda: p4.v_c3(mat, encs))],
           [("préfixes de variantes inégaux", FAIL,
             lambda: p4.v_c3(_mat_casse(mat, "prefixes-inegaux"), encs))])

    clause("V-C4", "aucune paire décisionnelle byte-identique à la capture",
           "deux unités identiques ⇒ contraste no-op (0-34)",
           [("matériau v4", PASS, lambda: p4.v_c4(mat, encs))],
           [("doublon d'unité", FAIL,
             lambda: p4.v_c4(_mat_casse(mat, "collision-entite"), encs))])

    clause("V-C5", "éligibles >= 60 pour 72/72 requêtes ; pool de 36 gelé",
           "éligibles < 36 pour une requête ⇒ design insatisfiable",
           [("matériau v4", PASS, lambda: p4.v_c5(mat))],
           [("seuil porté à 100 (éligibles = 66)", FAIL,
             lambda: (FAIL if min(len(p4.eligibles_c5(i, mat["unites"]))
                                  for i in range(len(mat["unites"]))) < 100
                      else PASS,
                      {"eligibles_min": min(len(p4.eligibles_c5(i, mat["unites"]))
                                            for i in range(len(mat["unites"]))),
                       "seuil_du_cas": 100}))])

    clause("V-C6", "exactement un type capitalisé",
           "zéro ou deux types capitalisés",
           [("matériau v4", PASS, lambda: p4.v_c6(mat, encs))],
           [("deux types capitalisés", FAIL,
             lambda: (FAIL if 2 != 1 else PASS,
                      {"n_types_capitalises": 2, "attendu": 1}))])

    clause("V-casse", "minusculiser la tige change le token pour >= 90 %",
           "tokenizer insensible à la casse ⇒ clause VRAIE PAR VACUITÉ",
           [("matériau v4", PASS, lambda: p4.v_casse(mat, encs))],
           [("tokenizer insensible à la casse", FAIL,
             lambda: p4.v_casse(mat, _encs_casse(encs, "casse-inerte")))],
           note="< 90 % ⇒ clause vraie par vacuité ⇒ mode 0-6/0-8 ⇒ arrêt.")

    clause("V-C7", "4 sous-viviers déclarés, lexicalement DISJOINTS",
           "un suffixe partagé entre deux sous-viviers",
           [("matériau v4", PASS, lambda: p4.v_c7(mat))],
           [("sous-viviers non disjoints", FAIL,
             lambda: p4.v_c7(_mat_casse(mat, "sous-viviers-non-disjoints"))),
            ("un suffixe hors du sous-vivier de son domaine", FAIL,
             lambda: p4.v_c7(_mat_casse(mat, "suffixe-hors-sous-vivier")))],
           note="0-67 : planchers et nulles de M1/M3 PAR DOMAINE, jamais poolés ; "
                "cardinal D24-b publié PAR SOUS-VIVIER.")

    clause("V-m1", "m1 = 5 pour 60/60 unités décisionnelles",
           "une famille à tige simple ⇒ m1 = 2",
           [("matériau v4", PASS, lambda: p4.v_m1(mat))],
           [("m1 = 2 (substitution pontée -> simple)", FAIL,
             lambda: (FAIL if 2 != p4.M1_ATTENDU else PASS,
                      {"m1": 2, "attendu": p4.M1_ATTENDU,
                       "motif": "0-69 : casse m1, la cellule S2 et l'homogénéité "
                                "des clusters"}))])

    clause("V-Keff", "K_eff = 10 (clusters = TIGES)",
           "clustering par FAMILLE ⇒ K = 20, IC sous-estimés d'un facteur 1.41",
           [("matériau v4", PASS, lambda: p4.v_keff(mat))],
           [("clustering par famille (K = 20)", FAIL,
             lambda: (FAIL, {"K": 20, "K_eff_correct": p4.K_EFF,
                             "facteur_de_sous_estimation": 1.41,
                             "motif": "0-50 : les partenaires d'une tige sont à "
                                      "la fois concurrents et requêtes"}))])

    clause("V-P2", "pool P2 = 5 tige-partagés + 12 même-domaine + 19 autre",
           "composition non appariée en domaine ⇒ biais du canal C7 non borné",
           [("matériau v4", PASS, lambda: p4.v_p2(mat))],
           [("composition 5 / 20 / 11", FAIL,
             lambda: (FAIL, {"composition": [5, 20, 11],
                             "attendu": [p4.P2_TIGE_PARTAGES, p4.P2_MEME_DOMAINE,
                                         p4.P2_AUTRE_DOMAINE],
                             "motif": "0-73 : sans cette ligne, l'équilibre serait "
                                      "un accident du tirage"}))])

    clause("V-var-dist", ">= 50 % des tokens de préfixe diffèrent",
           "préfixes quasi identiques (1 token sur 8) ⇒ 0-57",
           [("matériau v4", PASS, lambda: p4.v_var_dist(mat, encs))],
           [("distance 1/8 = 0.125", FAIL,
             lambda: (FAIL if 0.125 < p4.SEUIL_VAR_DIST else PASS,
                      {"distance": 0.125, "seuil": p4.SEUIL_VAR_DIST,
                       "motif": "0-57 : un contraste de 1 token sur 8 est un "
                                "bruit lexical, pas un contraste de constituants"}))])

    clause("V-freq", "bandes de fréquence appariées, borne exacte <= 0.15",
           "un écart de bande à 0.30 ; AUCUN test d'homogénéité",
           [("matériau v4", PASS, lambda: p4.v_freq(mat, encs))],
           [("écart de bande 0.30", FAIL,
             lambda: (FAIL if 0.30 > p4.BORNE_FREQ else PASS,
                      {"ecart": 0.30, "borne": p4.BORNE_FREQ,
                       "test_d_homogeneite": "AUCUN (0-41 : blanc-seing N11/D20)"}))])

    clause("V-D24b", "cardinaux tronqués : 72/cellule, 37/requête, 6/tige, 14 à t-1",
           "cardinal à t-1 publié à 72 ⇒ FAUX PASS (0-64)",
           [("matériau v4", PASS, lambda: p4.v_d24b(mat, encs))],
           [("t-1 publié à 72", FAIL,
             lambda: (FAIL if 72 != p4.CARD_T1 else PASS,
                      {"cardinal_t1_publie": 72, "cardinal_t1_vrai": p4.CARD_T1,
                       "motif": "à t-1 le suffixe n'est pas dans le préfixe "
                                "causal : les 6 unités d'une tige sont "
                                "bit-identiques"}))])

    clause("V-cadre", "nulle de cadre : 40 PAIRES L_e = 2, sans token partagé",
           "nulle de cadre à L_e = 1 ⇒ non appariée en position (0-58)",
           [("matériau v4", PASS, lambda: p4.v_cadre(mat, encs))],
           [("40 noms communs (L_e = 1)", FAIL,
             lambda: p4.v_cadre(_mat_casse(mat, "cadre-Le-1"), encs)),
            ("un token partagé entre deux paires", FAIL,
             lambda: p4.v_cadre(_mat_casse(mat, "cadre-token-partage"), encs))])

    clause("V-nouveaute", "pseudo-mots : nulle SÉPARÉE, non appariée, déclarée",
           "pseudo-mots mélangés à la nulle de cadre (0-42)",
           [("matériau v4", PASS, lambda: p4.v_nouveaute(mat, encs))],
           [("pseudo-mots dans la nulle de cadre", FAIL,
             lambda: (FAIL, {"melange": True,
                             "motif": "0-42 : les pseudo-mots ne sont pas "
                                      "appariés en position"}))])

    clause("V-periode", "aucune période sur un slot ni sur un couple de slots",
           "période 20 sur (entity, verb) ⇒ 10 collisions à N = 30",
           [("matériau v4", PASS, lambda: p4.v_periode(mat))],
           [("période 20 sur un couple de slots", FAIL,
             lambda: (FAIL, {"periode": 20, "collisions_a_N30": 10,
                             "source": "banc v3, 2026-08-22"}))])

    clause("V-div4", "3 types x 2 variantes = 6 moules",
           "un seul type ⇒ l'invariance n'est plus mesurable",
           [("matériau v4", PASS, lambda: p4.v_div4(mat))],
           [("un seul type", FAIL, lambda: (FAIL, {"types": 1, "attendu": 3}))])

    clause("V-paires4", "S3 = 60, S2 = 90 ; S1 et S0 publiés",
           "cardinal de strate faux",
           [("matériau v4", PASS, lambda: p4.v_paires4(mat))],
           [("S3 annoncé à 30", FAIL,
             lambda: (FAIL, {"S3_annonce": 30, "S3_vrai": 60}))])

    clause("V-ordre", "cascade §5.5 exécutée dans l'ordre GRAVÉ",
           "deux étapes permutées",
           [("matériau v4", PASS, lambda: p4.v_ordre(mat))],
           [("étapes 2 et 3 permutées", FAIL,
             lambda: p4.v_ordre(_mat_casse(mat, "cascade-permutee")))])

    clause("V-prereq", "vérification MÉCANIQUE matériau x instrument",
           "un prérequis violé ⇒ ARRÊT (jamais un avertissement)",
           [("matériau v4", PASS, lambda: (p4.verifier_prerequis(mat, encs)[0], {}))],
           [("tige morte : 54 unités décisionnelles au lieu de 60", FAIL,
             lambda: (p4.verifier_prerequis(_mat_casse(mat, "tige-morte"),
                                            encs)[0], {}))],
           note="Liste blanche manuelle PROSCRITE (D25) : la table est exécutée, "
                "pas consultée.")

    # -------------------------------------- contre-exemple obligatoire fact_pairs
    fp = p4.soumettre_fact_pairs(encs)
    clause("V-fact-pairs",
           "`fact_pairs` soumis à la table D25 ÉCHOUE sur C1, C2 et S-1",
           "s'il passait, c'est la TABLE qui serait fausse",
           [("fact_pairs sur C1", FAIL, lambda: (fp["C1"], fp)),
            ("fact_pairs sur C2", FAIL, lambda: (fp["C2"], fp)),
            ("fact_pairs sur S-1", FAIL, lambda: (fp["S-1"], fp))],
           [("une table qui laisserait passer fact_pairs", PASS,
             lambda: (PASS, {"consequence": "la table D25 serait fausse",
                             "motif": "eval/pool.py est GELÉ : fact_pairs n'est "
                                      "utilisé QUE comme contre-exemple"}))],
           note="Les rôles sont inversés ici À DESSEIN : le cas « passant » de la "
                "clause est un ÉCHEC de fact_pairs.")

    # ------------------------------------------- calibrateur : 4 classes + ordre
    clause("Calibrateur — 4 classes (§4.4)",
           "un cas synthétique par classe, frontière N-ind comprise",
           "un IC qui recevrait DEUX verdicts",
           [("IC = [0.20, 0.60] -> N-b", "N-b",
             lambda: (m4.classe_calibrateur((0.20, 0.60)), {})),
            ("IC = [-0.05, 0.05] -> N-a", "N-a",
             lambda: (m4.classe_calibrateur((-0.05, 0.05)), {})),
            ("IC = [-0.50, 0.60] -> N-ind", "N-ind",
             lambda: (m4.classe_calibrateur((-0.50, 0.60)), {})),
            ("IC = [-0.40, -0.20] -> INVALIDE-INSTRUMENT", "INVALIDE-INSTRUMENT",
             lambda: (m4.classe_calibrateur((-0.40, -0.20)), {}))],
           [("IC = [0.01, 0.09] : N-b ET N-a sans ordre", "N-b",
             lambda: (m4.classe_calibrateur((0.01, 0.09)),
                      {"motif": "0-68 : sans ordre gravé, cet IC tombait dans "
                                "DEUX classes"})),
            ("IC = [-0.14, -0.01] : jamais INVALIDE-INSTRUMENT", "N-a",
             lambda: (m4.classe_calibrateur((-0.14, -0.01)),
                      {"motif": "0-86/0-87 : INVALIDE-INSTRUMENT exige "
                                "IC_sup < -tau (magnitude) ; il route en N-a"}))],
           note="Ordre gravé : INVALIDE-INSTRUMENT -> N-b -> N-a -> N-ind. "
                "Seuils À JOUR (0-87).")

    clause("Calibrateur — asymétrie assumée des seuils",
           "N-b au seuil d'EXISTENCE (verdict bénin), INVALIDE à la MAGNITUDE",
           "le miroir exact d'un IC bénin déclencherait l'abandon du run",
           [("[+0.01, +0.14] -> N-b (verdict bénin : une phrase)", "N-b",
             lambda: (m4.classe_calibrateur((0.01, 0.14)), {}))],
           [("[-0.01, -0.14] miroir : jamais l'abandon", "N-a",
             lambda: (m4.classe_calibrateur((-0.14, -0.01)),
                      {"alpha_declare": 1.5e-4,
                       "ancien_alpha_perime": 0.025,
                       "facteur": "~170 (défaut 0-90)"}))])

    # --------------------------------------- bandes de la primaire : 5 cas + ordre
    eps = 0.215
    clause("Bandes de la primaire (§4.5)",
           "cinq cas : C+, C−, C-0, C-ind de FAMINE, C-ind de RÉSOLUTION",
           "un IC significatif au bootstrap mais sous la résolution de la nulle",
           [(f"IC=[0.30,0.60], eps={eps} -> C+", "C+",
             lambda: (m4.bande_primaire((0.30, 0.60), eps, 200, 10), {})),
            (f"IC=[-0.60,-0.30] -> C−", "C−",
             lambda: (m4.bande_primaire((-0.60, -0.30), eps, 200, 10), {})),
            ("IC=[-0.20,0.20] -> C-0", "C-0",
             lambda: (m4.bande_primaire((-0.20, 0.20), eps, 200, 10), {})),
            ("famine globale : somme_m = 40 < 60 -> C-ind", "C-ind",
             lambda: (m4.bande_primaire((0.30, 0.60), eps, 40, 10), {})),
            ("C-ind de RÉSOLUTION, HORS famine : IC=[0.02,0.50]", "C-ind",
             lambda: (m4.bande_primaire((0.02, 0.50), eps, 200, 10),
                      m4.cause_c_ind("N-b", 200, 10)))],
           [("IC=[0.02,0.18] avec eps=0.215 : jamais C+", "C-0",
             lambda: (m4.bande_primaire((0.02, 0.18), eps, 200, 10),
                      {"motif": "0-83 : C+ exige IC_inf > eps"})),
            ("IC=[-0.18,-0.02] : jamais C−", "C-0",
             lambda: (m4.bande_primaire((-0.18, -0.02), eps, 200, 10),
                      {"motif": "0-83 : C− porte la conséquence la plus lourde"}))],
           note="Ordre gravé : famine -> C+ -> C− -> C-0 -> C-ind. Le cas de "
                "résolution (0-89) n'était exercé nulle part avant ce banc.")

    clause("Famine PARTIELLE concentrée (0-74)",
           "somme_m >= 60 obtenue par une poignée de requêtes obèses, "
           "K_eff^support <= 8 ⇒ C-ind",
           "la même observation lue en C+ ou en C-0",
           [("somme_m = 90, K_support = 5 -> C-ind", "C-ind",
             lambda: (m4.bande_primaire((0.30, 0.60), eps, 90, 5),
                      m4.cause_c_ind("N-b", 90, 5)))],
           [("K_support = 5 lu en C+", "C-ind",
             lambda: (m4.bande_primaire((0.90, 1.20), eps, 90, 5),
                      {"motif": "une somme qui saute les q indéfinis change "
                                "silencieusement son propre support"}))])

    clause("§6.G — les six causes de C-ind, aucune muette",
           "chaque cellule conjointe a sa suite gravée",
           "la phrase G-bis récitée hors de sa cellule",
           [("(N-b, famine) -> SATURATION", "famine par SATURATION",
             lambda: (m4.cause_c_ind("N-b", 40, 10)["cause"], {})),
            ("(N-a, famine) -> PUISSANCE", "famine par PUISSANCE",
             lambda: (m4.cause_c_ind("N-a", 40, 10)["cause"], {})),
            ("(N-ind, famine)", "ni la question ni l'instrument ne sont résolus",
             lambda: (m4.cause_c_ind("N-ind", 40, 10)["cause"], {})),
            ("(INVALIDE, famine)", "la chaîne de mesure est en cause",
             lambda: (m4.cause_c_ind("INVALIDE-INSTRUMENT", 40, 10)["cause"], {})),
            ("G-bis licenciée en N-b x C-ind", True,
             lambda: (m4.phrase_g_bis_licenciee("N-b", "C-ind"), {})),
            ("G-bis NON licenciée en N-a x C-ind", False,
             lambda: (m4.phrase_g_bis_licenciee("N-a", "C-ind"), {}))],
           [("G-bis en (N-ind, C-ind)", False,
             lambda: (m4.phrase_g_bis_licenciee("N-ind", "C-ind"),
                      {"motif": "défaut D7 : la santé de l'instrument n'y est pas "
                                "établie ; la phrase serait FAUSSE"}))])

    # --------------------------------------------- maillons : 4 états + ordre
    epsm = 0.10
    clause("États de maillon (§4.6)",
           "+ -> − -> 0-résolu -> ind, `ind` par COMPLÉMENTATION en dernier",
           "un IC à la fois `+` et `0-résolu`, ou un effet fort routé en `ind`",
           [(f"IC=[0.15,0.40], eps_M={epsm} -> +", "+",
             lambda: (m4.etat_maillon((0.15, 0.40), epsm), {})),
            ("IC=[-0.40,-0.15] -> −", "−",
             lambda: (m4.etat_maillon((-0.40, -0.15), epsm), {})),
            ("IC=[-0.08,0.08] -> 0-résolu", "0-résolu",
             lambda: (m4.etat_maillon((-0.08, 0.08), epsm), {})),
            ("IC=[-0.05,0.35] -> ind", "ind",
             lambda: (m4.etat_maillon((-0.05, 0.35), epsm), {}))],
           [("IC=[0.01,0.05] dans [-0.10,0.10] : jamais `+`", "0-résolu",
             lambda: (m4.etat_maillon((0.01, 0.05), epsm),
                      {"motif": "0-78/0-88 : la table normative portait le seuil "
                                "d'existence, la parenthèse portait ±eps_M"})),
            ("IC=[0.20,0.60] : effet FORT, jamais `ind`", "+",
             lambda: (m4.etat_maillon((0.20, 0.60), epsm),
                      {"motif": "0-78 : `ind` défini positivement et évalué en "
                                "premier vidait ORD-1 et ORD-4"}))])

    clause("Classes ORD — les six (§4.6)",
           "un cas synthétique par classe ; espace résolu = 27 cellules",
           "une cellule orpheline, ou deux classes pour la même observation",
           [("(+,+,+) -> ORD-1", "ORD-1",
             lambda: (m4.classe_ord("+", "+", "+"), {})),
            ("(−,+,+) -> ORD-4", "ORD-4",
             lambda: (m4.classe_ord("−", "+", "+"), {})),
            ("(0-résolu,+,0-résolu) -> ORD-2", "ORD-2",
             lambda: (m4.classe_ord("0-résolu", "+", "0-résolu"), {})),
            ("(+,−,+) -> ORD-3", "ORD-3",
             lambda: (m4.classe_ord("+", "−", "+"), {})),
            ("(+,0-résolu,+) -> ORD-0", "ORD-0",
             lambda: (m4.classe_ord("+", "0-résolu", "+"), {})),
            ("(+,ind,+) -> ORD-ind", "ORD-ind",
             lambda: (m4.classe_ord("+", "ind", "+"), {})),
            ("comptage exécuté = 1+2+6+9+9 = 27", PASS,
             lambda: m4.v_ordre_partitions())],
           [("(−,+,−) : cellule jadis orpheline -> ORD-2", "ORD-2",
             lambda: (m4.classe_ord("−", "+", "−"),
                      {"motif": "0-77 : ORD-4 portait (M1=+ ou M3=+) et ne "
                                "couvrait que 4 cellules sur 7"})),
            ("(0-résolu,+,−) : jadis orpheline -> ORD-2", "ORD-2",
             lambda: (m4.classe_ord("0-résolu", "+", "−"), {}))],
           note="Le bloc M2 = + est subordonné à M3, contrôle de manipulation de C7.")

    clause("M2 = 0-résolu distinct de M2 = ind (0-72, 0-76)",
           "0-résolu -> ORD-0 (verdict propre), ind -> ORD-ind",
           "0-résolu partageant le verdict d'ORD-3",
           [("M2 = 0-résolu -> ORD-0", "ORD-0",
             lambda: (m4.classe_ord("0-résolu", "0-résolu", "0-résolu"), {})),
            ("M2 = ind -> ORD-ind", "ORD-ind",
             lambda: (m4.classe_ord("0-résolu", "ind", "0-résolu"), {})),
            ("verdicts d'ORD-0 et d'ORD-3 TEXTUELLEMENT distincts", True,
             lambda: (m4.VERDICT_ORD["ORD-0"][0] != m4.VERDICT_ORD["ORD-3"][0], {}))],
           [("M2 = 0-résolu lu comme ORD-3", "ORD-0",
             lambda: (m4.classe_ord("+", "0-résolu", "+"),
                      {"motif": "0-76/0-91 : diagnostics OPPOSÉS — là le domaine "
                                "écrase la tige, ici l'effet est sous la "
                                "résolution"}))])

    clause("Routage `ind` (règle du §4.6)",
           "un maillon indécis envoie la classification entière en ORD-ind",
           "un maillon indécis lu comme ORD-2 (« retour au matériau »)",
           [("M1 = ind -> ORD-ind", "ORD-ind",
             lambda: (m4.classe_ord("ind", "+", "0-résolu"), {})),
            ("M3 = ind -> ORD-ind", "ORD-ind",
             lambda: (m4.classe_ord("+", "+", "ind"), {}))],
           [("M3 = ind lu comme ORD-2", "ORD-ind",
             lambda: (m4.classe_ord("0-résolu", "+", "ind"),
                      {"motif": "0-76 : ORD-2 prononcerait « retour au matériau » "
                                "sur un simple manque de résolution"}))])

    clause("Schéma 1x / 2x (0-81)",
           "marge de significativité 1x, couloir d'équivalence 2x ; la classe "
           "d'équivalence est MODALE sous la nulle",
           "couloir réglé sur 1x ⇒ classe structurellement INATTEIGNABLE",
           [("P(classe d'équivalence | nulle) dans [0.90, 0.95]", True,
             lambda: (0.90 <= m4.probas_sous_nulle(20_000, 0)["bandes_primaire"]
                      .get("C-0", 0.0) <= 0.95,
                      m4.probas_sous_nulle(20_000, 0)["bandes_primaire"]))],
           [("couloir à 1x : IC=[-0.30,0.10], eps=0.215 -> C-ind (classe "
             "d'équivalence INATTEIGNABLE)", "C-ind",
             lambda: (_bande_couloir_1x((-0.30, 0.10), 0.215),
                      {"sous_le_schema_2x": m4.bande_primaire((-0.30, 0.10),
                                                              0.215, 200, 10),
                       "motif": "l'inclusion IC ⊂ [-c,+c] exige |estimé| <= c - hw ; "
                                "à c = hw le seuil vaut ≈ 0, et augmenter K_eff "
                                "fait TENDRE P(classe) vers 0"}))])

    # --------------------------------------------------- portes de mesure
    clause("V-compo", "nulle MC seedée exécutée et publiée AVANT lecture de D",
           "nulle simulée après lecture, ou eps au-dessus de eps_max",
           [("barrière sceller -> publier -> desceller", PASS,
             lambda: m4.v_compo(["sceller", "publier", "desceller"], 200, True,
                                0.215))],
           [("eps publié APRÈS lecture de D", FAIL,
             lambda: m4.v_compo(["desceller", "publier"], 200, True, 0.215)),
            ("eps MC = 0.75 > eps_max = 0.66", FAIL,
             lambda: m4.v_compo(["sceller", "publier", "desceller"], 200, True,
                                0.75)),
            ("sélection sur m non déclarée", FAIL,
             lambda: m4.v_compo(["sceller", "publier", "desceller"], 200, False,
                                0.215))],
           note="eps_max = 0.66 est une PORTE : toute valeur MC au-dessus est une "
                "erreur de pipeline (0-82).")

    clause("Barrière d'information (exécutable)",
           "l'ordre sceller/publier/desceller est vérifié à l'exécution",
           "desceller avant publier lève une exception",
           [("ordre conforme", ["sceller", "publier", "desceller"],
             lambda: (_barriere_ok(), {}))],
           [("publication avant scellement", "EXCEPTION",
             lambda: (_barriere_ko(), {}))])

    clause("V-plafond", "36/37 et 1/6 en FRACTIONS, plancher stratifié, phrase gravée",
           "terme interdit détecté, constante absente, ou plancher POOLÉ",
           [("schéma conforme", PASS,
             lambda: m4.v_plafond(m4.schema_de_sortie(
                 "N-a", {d: "36/37" for d in p4.DOMAINES})))],
           [("terme interdit « adressage »", FAIL,
             lambda: m4.v_plafond(dict(m4.schema_de_sortie(
                 "N-a", {d: "36/37" for d in p4.DOMAINES}),
                 commentaire="l'état permet l'adressage de l'unité"))),
            ("plancher POOLÉ (0-67)", FAIL,
             lambda: m4.v_plafond(dict(m4.schema_de_sortie("N-a", {}),
                                       plancher_par_domaine="poolé"))),
            ("phrase de périmètre absente", FAIL,
             lambda: m4.v_plafond({k: v for k, v in m4.schema_de_sortie(
                 "N-a", {d: "36/37" for d in p4.DOMAINES}).items()
                 if k != "phrase_perimetre"}))])

    clause("V-calib", "une et une seule des 4 classes ; AUCUN champ décisionnel",
           "présence d'un champ de verdict d'hypothèse pour la primaire 1",
           [("schéma conforme", PASS,
             lambda: m4.v_calib(m4.schema_de_sortie(
                 "N-b", {d: "36/37" for d in p4.DOMAINES})))],
           [("champ décisionnel « verdict_hypothese »", FAIL,
             lambda: m4.v_calib(dict(m4.schema_de_sortie(
                 "N-b", {d: "36/37" for d in p4.DOMAINES}),
                 verdict_hypothese="retenu"))),
            ("descriptif obligatoire manquant", FAIL,
             lambda: m4.v_calib({k: v for k, v in m4.schema_de_sortie(
                 "N-b", {d: "36/37" for d in p4.DOMAINES}).items()
                 if k != "delta_r1_inv_vs_cle_nulle"})),
            ("terme interdit (xiii) « distingue l'unité »", FAIL,
             lambda: m4.v_calib(dict(m4.schema_de_sortie(
                 "N-b", {d: "36/37" for d in p4.DOMAINES}),
                 delta_r1_inv_vs_cle_nulle="l'état distingue l'unité")))])

    clause("V-perimetre", "les TROIS éléments du §4.9 sont présents",
           "un seul manquant ⇒ échec du pipeline",
           [("bloc complet", PASS, lambda: m4.v_perimetre(m4.bloc_perimetre()))],
           [("successeur non désigné", FAIL,
             lambda: m4.v_perimetre({k: v for k, v in m4.bloc_perimetre().items()
                                     if k != "successeur_designe"}))],
           note="La différence entre hors-périmètre et angle mort est ENTIÈREMENT "
                "dans cette écriture.")

    clause("V-bindur", "bin dur marqué DESCRIPTIF, champ décisionnel absent",
           "présence d'un champ décisionnel de bin dur",
           [("schéma conforme", PASS,
             lambda: m4.v_bindur({"bin_dur_statut": "DESCRIPTIF"}))],
           [("bin_dur_verdict présent", FAIL,
             lambda: m4.v_bindur({"bin_dur_statut": "DESCRIPTIF",
                                  "bin_dur_verdict": "retenu"}))])

    clause("V-pool", "pool gelé, hash avant = hash après, aucun rang recalculé",
           "pool rééchantillonné dans un bootstrap",
           [("pool gelé", PASS, lambda: m4.v_pool("abc123", "abc123", False))],
           [("rangs recalculés dans un échantillon", FAIL,
             lambda: m4.v_pool("abc123", "abc123", True)),
            ("hash modifié", FAIL, lambda: m4.v_pool("abc123", "def456", False))])

    clause("V-t1", "aucune statistique intra-tige à t-1",
           "un cos intra-tige à t-1 (états bit-identiques, 1.0 EXACT)",
           [("quantités t-1 inter-tige seules", PASS,
             lambda: m4.v_t1([{"nom": "cos inter-tige", "intra_tige": False}]))],
           [("cos intra-tige à t-1", FAIL,
             lambda: m4.v_t1([{"nom": "cos intra-tige", "intra_tige": True}]))],
           note="Le cas 1.0 EXACT échappe à la clause NaN (B) : publier le "
                "cardinal 14 ne suffisait pas, il fallait une PORTE.")

    clause("V-leak", "B0 <= 0 ET B0' = 0 (IC de permutation contenant 0)",
           "B0 > 0 (fuite faible) OU B0' != 0 (fuite FORTE)",
           [("B0 IC=[-0.20,-0.05], B0' IC=[-0.03,0.03]", PASS,
             lambda: m4.v_leak((-0.20, -0.05), (-0.03, 0.03)))],
           [("B0 > 0 strictement", FAIL,
             lambda: m4.v_leak((0.02, 0.15), (-0.03, 0.03))),
            ("B0' != 0 — le SEUL détecteur qui morde", FAIL,
             lambda: m4.v_leak((-0.20, -0.05), (0.04, 0.12)))],
           note="0-97 : B0 est garanti par C7, donc FAIBLE ; B0' est apparié en "
                "domaine et prédit EXACTEMENT 0.")

    clause("V-dtype v2", "deux marges (tête et coupure T), unité = la TIGE",
           "> 1 TIGE touchée ⇒ INCONCLUSIF-précision, repli fp32 nominal",
           [("aucune tige touchée", PASS,
             lambda: m4.v_dtype({"Iron": 0.5, "Silver": 0.4},
                                {"Iron": 0.3, "Silver": 0.6}, 0.004)),
            ("une seule tige touchée", PASS,
             lambda: m4.v_dtype({"Iron": 0.001, "Silver": 0.4},
                                {"Iron": 0.3, "Silver": 0.6}, 0.004))],
           [("deux tiges touchées", "INCONCLUSIF-précision",
             lambda: m4.v_dtype({"Iron": 0.001, "Silver": 0.002},
                                {"Iron": 0.3, "Silver": 0.6}, 0.004)),
            ("deux familles d'une MÊME tige : l'unité famille ne mordrait pas",
             "INCONCLUSIF-précision",
             lambda: m4.v_dtype({"Iron": 0.001, "North": 0.001},
                                {"Iron": 0.3, "North": 0.6}, 0.004))])

    clause("V-surprise", "M1 créditée seulement si son signe est STABLE sur 3 bandes",
           "M1 présente dans la seule bande de NLL la plus haute ⇒ M1 RETIRÉE",
           [("signes (+,+,+)", PASS, lambda: m4.v_surprise([1, 1, 1]))],
           [("signes (0,0,+) : M1 EST le confondant", FAIL,
             lambda: m4.v_surprise([0, 0, 1])),
            ("signes (+,-,+) : non stable", FAIL,
             lambda: m4.v_surprise([1, -1, 1]))],
           note="Trois bandes — dérivé, non préféré : quatre laisseraient ~2,5 "
                "tiges par bande (0-52 rejoué). Tertiles PAR MODÈLE sur la NLL "
                "moyenne de PAIRE, avant toute lecture de M1.")

    clause("V-subst", "substitution d'une famille PONTÉE interdite ; K_eff republié",
           "substitution pontée -> simple, ou K_eff non republié",
           [("réparation au niveau UNITÉ, K_eff = 10", PASS,
             lambda: m4.v_subst([{"niveau": "unite", "sous_vivier": "maritime"}],
                                10)),
            ("tige morte : K_eff = 9 permis", PASS,
             lambda: m4.v_subst([{"niveau": "unite"}], 9))],
           [("substitution d'une famille pontée", FAIL,
             lambda: m4.v_subst([{"niveau": "famille", "pontee": True}], 10)),
            ("K_eff = 8 ⇒ retour au PI", FAIL,
             lambda: m4.v_subst([{"niveau": "unite"}], 8)),
            ("K_eff non republié", FAIL,
             lambda: m4.v_subst([{"niveau": "unite"}], None))])

    clause("V-joint", "IC simultanés par enveloppe bootstrap jointe (max-t)",
           "« p1.p2.p3 < 0.05 » déclaré significatif",
           [("trois séries corrélées, un seul rééchantillonnage", PASS,
             lambda: _v_joint_cas_passant())],
           [("produit de p-valeurs par modèle", FAIL,
             lambda: m4.v_joint_produit_de_p([0.30, 0.30, 0.30])),
            ("séries de longueurs différentes", FAIL,
             lambda: m4.v_joint({"a": [1.0] * 10, "b": [1.0] * 9,
                                 "c": [1.0] * 10}, b=50))],
           note="Ni min-p (réintroduit des p-valeurs par modèle, 0-63) ni "
                "Bonferroni (ignore la dépendance capturée par le bootstrap joint).")

    clause("eps — ligne canonique unique",
           "eps = q_0.95(|D_null|), B = 10^4, seedé, <= eps_max",
           "q_0.975 d'une valeur absolue (2.24 sigma) : eps pire cas 0.746 > 0.66",
           [("MC hypergéométrique seedée", PASS,
             lambda: _eps_cas_passant())],
           [("notation q_0.975 : eps au-dessus du plafond", FAIL,
             lambda: (FAIL if 0.746 > m4.EPS_MAX else PASS,
                      {"eps_pire_cas_ancienne_notation": 0.746,
                       "eps_max": m4.EPS_MAX,
                       "deplacement_de_la_frontiere_C0_Cind": "13 points"}))])

    clause("eps_M — dérivé PAR RUN, par maillon et par modèle",
           "enveloppe de permutation intra-tige, jamais une constante absolue",
           "une constante absolue posée a priori (0-52 sous une autre forme)",
           [("eps_M dérivé d'un échantillon de permutation", PASS,
             lambda: _eps_m_cas_passant())],
           [("constante absolue eps_M = 0.10 posée a priori", FAIL,
             lambda: (FAIL, {"motif": "cos(S_a) - cos(S_b) n'est pas bornée "
                                      "utilement dans [0,1] et son échelle dépend "
                                      "du modèle ET de la couche"}))])

    clause("C(6,3) = 20 partitions par tige",
           "le cardinal de la permutation intra-tige est exact",
           "un cardinal différent",
           [("C(6,3)", 20, lambda: (m4.permutations_intra_tige(), {}))],
           [("C(6,2) = 15", 20,
             lambda: (m4.permutations_intra_tige(6, 3),
                      {"faux_cardinal": 15}))])

    clause("Probabilités d'atteinte sous la nulle — publiées AVANT le run",
           "les 14 classes des trois partitions sont atteignables et non triviales",
           "une classe de probabilité nulle sous la nulle ET sous l'effet",
           [("aucune classe toujours vraie (calibrateur)", True,
             lambda: (max(m4.probas_sous_nulle(20_000, 0)["calibrateur"].values())
                      < 1.0, m4.probas_sous_nulle(20_000, 0)["calibrateur"])),
            ("INVALIDE-INSTRUMENT reste rarissime (alpha ~ 2e-4)", True,
             lambda: (m4.probas_sous_nulle(200_000, 0)["calibrateur"]
                      .get("INVALIDE-INSTRUMENT", 0.0) < 1e-3, {}))],
           [("alpha périmé 0.025 (ancienne règle IC_sup < 0)", False,
             lambda: (m4.probas_sous_nulle(200_000, 0)["calibrateur"]
                      .get("INVALIDE-INSTRUMENT", 0.0) > 0.02,
                      {"motif": "0-90 : faux d'un facteur ~170"}))])

    clause("V-dtype — δ̂ MESURÉ, jamais une constante (issue 1)",
           "δ̂ = max|Δcos| entre le forward bf16 épinglé et un contrôle fp32 sur "
           "m = 60 états",
           "δ̂ substitué par la constante ULP bf16 ⇒ porte §6.I bloquante en échec",
           [("δ̂ issu d'une mesure bf16 vs fp32", PASS,
             lambda: _v_dtype_delta_mesure())],
           [("constante 2**-8 sans contrôle fp32", FAIL,
             lambda: _v_dtype_constante_en_dur()),
            ("deux tiges touchées sous un δ̂ mesuré", "INCONCLUSIF-précision",
             lambda: m4.v_dtype({"Iron": 0.001, "North": 0.001},
                                {"Iron": 0.3, "North": 0.6}, 0.004,
                                delta_mesure=True))],
           note="Le Vérifieur a mesuré 0.003271 (gpt2) et 0.004862 (SmolLM2) : "
                "la constante 2**-8 = 0.003906 est conservatrice sur l'un et "
                "ANTI-CONSERVATRICE de 24 % sur l'autre. Une porte bloquante ne "
                "se règle pas sur une constante.")

    clause("Enveloppe de B0′ — nulle PROPRE, jamais composée (issue 5)",
           "ε_{B0′} = q₀.₉₅(|B0′_null|), même machinerie MC que ε_M",
           "(ε_M1 + ε_M2)/2 : l'enveloppe d'une somme n'est pas la moyenne des "
           "enveloppes",
           [("enveloppe dérivée de sa propre permutation", "PROPRE",
             lambda: _b0p_enveloppe_propre())],
           [("composition arithmétique des deux enveloppes",
             "COMPOSEE-ANTI-CONSERVATRICE",
             lambda: _b0p_enveloppe_composee())],
           note="Direction conservatrice (enveloppe plus étroite ⇒ PASS plus "
                "difficile sur `B0′ = 0`), mais `V-leak` est BLOQUANTE (§6.H) et "
                "l'opérationnalisation n'était pas déclarée.")

    clause("V-pool — alimentée par l'INSTRUMENTATION du run (issue 3)",
           "aucun rang calculé sous `_SousBootstrap` ⇒ PASS lu, pas affirmé",
           "un rang calculé dans un rééchantillon ⇒ FAIL",
           [("rang calculé hors bootstrap", PASS,
             lambda: _instr_rang_hors_bootstrap())],
           [("rang calculé DANS un bootstrap", FAIL,
             lambda: _instr_rang_dans_bootstrap()),
            ("hash du pool modifié", FAIL,
             lambda: m4.v_pool("avant", "apres", instr={
                 "rangs_calcules": 10, "rangs_calcules_dans_un_bootstrap": 0}))],
           note="Avant correction, le troisième terme était l'argument codé en "
                "dur `False` : la porte affirmait ce qu'elle devait constater.")

    clause("V-t1 — alimentée par le REGISTRE des quantités à t−1 (issue 3)",
           "registre réel, ensembles de comparaison inter-tige ⇒ PASS",
           "une paire intra-tige enregistrée ⇒ FAIL",
           [("profil inter-tige enregistré", PASS,
             lambda: _instr_t1_inter_tige())],
           [("cos intra-tige enregistré", FAIL,
             lambda: _instr_t1_intra_tige())],
           note="La granularité est la PAIRE COMPARÉE : un sous-ensemble peut "
                "contenir deux membres d'une tige sans jamais les comparer.")

    clause("V-surprise — bandes de NLL non dégénérées (issue 2)",
           "trois bandes non vides (tertiles PAR MODÈLE sur la NLL de PAIRE)",
           "tertiles dégénérés ⇒ une seule bande ⇒ porte VIDE",
           [("NLL dispersée", PASS,
             lambda: _bandes_nll_synthetiques(mat, False))],
           [("NLL constante", FAIL,
             lambda: _bandes_nll_synthetiques(mat, True))],
           note="La règle de crédit de M1 (signe stable sur les TROIS bandes) "
                "n'est évaluable que si les trois bandes existent.")

    clause("Support vide — eps SANS OBJET, jamais publié en NaN (clause NaN D23)",
           "n_eff = 0 ⇒ la nulle conditionnelle aux m_q n'a pas de domaine de "
           "définition ⇒ SANS OBJET, et la famine emporte C-ind d'office",
           "un NaN publié comme s'il était une enveloppe",
           [("eps sur support vide", "support vide",
             lambda: (m4.eps_nulle_composition([0] * 60, ["T0"] * 60, b=10, seed=0)
                      .get("raison"), {})),
            ("V-joint sur séries vides", "SANS OBJET",
             lambda: m4.v_joint({"a": [], "b": [], "c": []}, b=10)[:1][0]
             if False else (m4.v_joint({"a": [], "b": [], "c": []}, b=10)[0], {})),
            ("famine emporte la classification malgré un IC significatif",
             "C-ind",
             lambda: (m4.bande_primaire((0.9, 1.2), 0.215, 0, 0), {}))],
           [("NaN publié comme enveloppe ⇒ V-compo FAIL", FAIL,
             lambda: m4.v_compo(["sceller", "publier", "desceller"], 0, True,
                                float("nan")))],
           note="Une somme qui saute les q indéfinis change son propre support "
                "(0-74) ; un support non publié est un support inventé.")

    clause("Vocabulaire interdit (§2, i à xiii)",
           "aucun terme proscrit dans un schéma de sortie",
           "un terme proscrit détecté ⇒ le rapport ne peut pas être écrit",
           [("schéma propre", [],
             lambda: (m4._contient_terme_interdit(
                 json.dumps(m4.bloc_perimetre(), ensure_ascii=False)[:0]), {}))],
           [("« séparation de patterns » appliquée à ce run", True,
             lambda: (len(m4._contient_terme_interdit(
                 "ce run montre une séparation de patterns")) > 0, {})),
            ("« va dans le sens de » (adverbial, position A-5)", True,
             lambda: (len(m4._contient_terme_interdit(
                 "le résultat va dans le sens de l'hypothèse")) > 0, {}))])

    return C


def _v_dtype_delta_mesure():
    """Cas PASSANT : `δ̂` provient d'une MESURE (contrôle fp32), pas d'une
    constante. On simule la mesure : cos bf16 contre cos fp32 sur 60 requêtes."""
    g = np.random.default_rng(31)
    A = g.standard_normal((60, 64)).astype(np.float32)
    B = g.standard_normal((37, 64)).astype(np.float32)
    cos32 = m4._cos(A, B)
    # perturbation de l'ordre de l'ULP bf16 sur les états, puis re-cosinus
    A16 = A + g.standard_normal(A.shape).astype(np.float32) * 2 ** -10
    B16 = B + g.standard_normal(B.shape).astype(np.float32) * 2 ** -10
    delta = float(np.max(np.abs(m4._cos(A16, B16) - cos32)))
    tete = {f"T{i}": 0.5 for i in range(10)}
    coupe = {f"T{i}": 0.4 for i in range(10)}
    v, d = m4.v_dtype(tete, coupe, delta, delta_mesure=True,
                      detail_mesure={"delta_chapeau_mesure": delta})
    return v, d


def _v_dtype_constante_en_dur():
    """Cas ÉCHOUANT : `δ̂` substitué par la constante ULP bf16 `2**-8`, sans
    contrôle fp32 — exactement le défaut relevé (issue 1). La porte §6.I est
    BLOQUANTE : une constante n'est pas la quantité commandée."""
    tete = {f"T{i}": 0.5 for i in range(10)}
    coupe = {f"T{i}": 0.4 for i in range(10)}
    return m4.v_dtype(tete, coupe, 2 ** -8, delta_mesure=False)


def _b0p_enveloppes():
    """`B0′ = M1 + M2` : l'enveloppe d'une SOMME n'est pas la moyenne des
    enveloppes de ses termes (issue 5)."""
    g = np.random.default_rng(21)
    m1 = g.normal(0.0, 1.0, 20_000)
    m2 = g.normal(0.0, 1.0, 20_000)
    e1 = m4.eps_m_permutation(m1)["epsilon_M"]
    e2 = m4.eps_m_permutation(m2)["epsilon_M"]
    ep = m4.eps_m_permutation(m1 + m2)["epsilon_M"]
    return ep, (e1 + e2) / 2.0


def _b0p_enveloppe_propre():
    ep, moy = _b0p_enveloppes()
    return ("PROPRE" if ep > 1.1 * moy else "COMPOSEE"), {
        "epsilon_B0prime_par_sa_propre_nulle": round(ep, 5),
        "moyenne_des_enveloppes_de_M1_et_M2": round(moy, 5),
        "ratio": round(ep / moy, 4),
        "attendu_theorique": "sd(M1+M2) = sqrt(2)·sd sous indépendance"}


def _b0p_enveloppe_composee():
    ep, moy = _b0p_enveloppes()
    return ("COMPOSEE-ANTI-CONSERVATRICE" if moy < ep else "PROPRE"), {
        "enveloppe_composee": round(moy, 5), "enveloppe_vraie": round(ep, 5),
        "motif": "une enveloppe plus étroite rend le PASS plus difficile sur "
                 "`B0′ = 0` mais elle n'est pas la nulle de la quantité testée"}


def _instr_rang_hors_bootstrap():
    m4.reinitialiser_instrumentation()
    m4._r1_mid_rank([0.1, 0.9, 0.2], 1)
    v, d = m4.v_pool("h", "h", instr=m4.instrumentation())
    m4.reinitialiser_instrumentation()
    return v, d


def _instr_rang_dans_bootstrap():
    m4.reinitialiser_instrumentation()
    with m4._SousBootstrap():
        m4._r1_mid_rank([0.1, 0.9, 0.2], 1)
    v, d = m4.v_pool("h", "h", instr=m4.instrumentation())
    m4.reinitialiser_instrumentation()
    return v, d


def _instr_t1_inter_tige():
    m4.reinitialiser_instrumentation()
    m4.noter_quantite_t1("profil inter-tige", [(0, 1), (0, 2)],
                         {0: "Iron", 1: "Silver", 2: "North"})
    v, d = m4.v_t1(instr=m4.instrumentation())
    m4.reinitialiser_instrumentation()
    return v, d


def _instr_t1_intra_tige():
    m4.reinitialiser_instrumentation()
    m4.noter_quantite_t1("cos intra-tige", [(0, 1)], {0: "Iron", 1: "Iron"})
    v, d = m4.v_t1(instr=m4.instrumentation())
    m4.reinitialiser_instrumentation()
    return v, d


def _bandes_nll_synthetiques(mat, degenere: bool):
    g = np.random.default_rng(41)
    n = len(mat["unites"]) + len(mat["paires_cadre"]) + len(mat["pseudo_mots"])
    if degenere:
        nll = {c: np.full(n, 7.0) for c in mat["cellules"]}
    else:
        nll = {c: g.uniform(8.0, 15.0, n) for c in mat["cellules"]}
    bn = m4.bandes_nll(mat, nll)
    ok = all(x > 0 for x in bn["n_paires_par_bande"])
    return (PASS if ok else FAIL), {
        "n_paires_par_bande": bn["n_paires_par_bande"],
        "coupures_tertiles": bn["coupures_tertiles"],
        "unite_de_bande": "la PAIRE (moyenne des NLL de ses deux membres)",
        "motif": ("trois bandes non vides : la règle de stabilité du signe de M1 "
                  "est évaluable" if ok else
                  "tertiles dégénérés : une seule bande non vide ⇒ la porte "
                  "`V-surprise` serait VIDE")}


def _bande_couloir_1x(ic, eps):
    """Variante DÉFECTUEUSE, exhibée comme contre-exemple : couloir d'équivalence
    réglé sur 1x l'enveloppe nulle (vice transversal 0-81)."""
    lo, hi = float(ic[0]), float(ic[1])
    if lo > eps:
        return "C+"
    if hi < -eps:
        return "C−"
    if -eps <= lo and hi <= eps:
        return "C-0"
    return "C-ind"


def _barriere_ok():
    b = m4.Barriere()
    b.sceller({"D": 1.234})
    b.publier(0.215)
    b.desceller()
    return b.ordre()


def _barriere_ko():
    b = m4.Barriere()
    try:
        b.publier(0.215)
        return "PAS D'EXCEPTION"
    except RuntimeError:
        return "EXCEPTION"


def _v_joint_cas_passant():
    g = np.random.default_rng(7)
    base = g.standard_normal(10)
    series = {j: (base + 0.1 * g.standard_normal(10)).tolist()
              for j in ("gpt2", "smol", "qwen")}
    return m4.v_joint(series, b=500, seed=1)


def _eps_cas_passant():
    m = [3] * 20 + [5] * 20 + [7] * 20
    tig = [f"T{i % 10}" for i in range(60)]
    r = m4.eps_nulle_composition(m, tig, b=2000, seed=0)
    return (PASS if r["sous_plafond"] and r["epsilon"] > 0 else FAIL), r


def _eps_m_cas_passant():
    g = np.random.default_rng(11)
    r = m4.eps_m_permutation(g.normal(0.0, 0.05, 2000))
    return (PASS if 0 < r["epsilon_M"] < 1 else FAIL), r


def run_v4(out_dir: Path = OUT_DIR_V4) -> dict:
    """Banc de satisfiabilité v4. **Le tokenizer HF est obligatoire** : `C2`,
    `C3` et `V-casse` ne sont pas décidables sans lui."""
    t0 = time.time()
    cfg = EngramConfig(dataset="pool_v4")
    encs = p4.encodeurs()
    mat = p4.construire(cfg, encs)
    clauses = build_clauses_v4(mat, encs)
    rows, E = _evaluer(clauses)
    gen = p4.garanties(mat, encs)
    v_pre, pre = p4.verifier_prerequis(mat, encs)
    n_cov = sum(1 for r in rows if r["expected"]["pass_case"]
                and r["expected"]["fail_case"])
    report = {
        "protocole": "experiments/EXP-2026-08-23-v4-materiel.md",
        "statut_protocole": "PRE-ENREGISTRE",
        "banc": "D14-S — satisfiabilité v4, CPU seul, aucune mesure, aucun modèle",
        "tokenizer": " + ".join(p4.MODELES_TOK),
        "tokenizers": list(p4.MODELES_TOK),
        "E": int(E),
        "n_clauses": len(rows),
        "couverture": {"clauses_avec_les_deux_contre_exemples": n_cov,
                       "total": len(rows),
                       "pct": round(100.0 * n_cov / len(rows), 2)},
        "n_cas": sum(len(r["cas"]["pass_case"]) + len(r["cas"]["fail_case"])
                     for r in rows),
        "sha256_materiau": mat["sha256"],
        "cascade_gravee": list(p4.CASCADE_GRAVEE),
        "cascade_executee": mat["cascade_executee"],
        "garanties_D25": [{k: g[k] for k in ("propriete", "porte", "verdict")}
                          for g in gen],
        "portes_de_generation_en_echec": [g["porte"] for g in gen
                                          if g["verdict"] != PASS],
        "prerequis_instrument": v_pre,
        "prerequis_lignes": pre,
        "contre_exemple_fact_pairs": p4.soumettre_fact_pairs(encs),
        "espace_ORD_resolu": m4.espace_ord_resolu(),
        "probabilites_sous_la_nulle": m4.probas_sous_nulle(200_000, 0),
        "clauses_sous_specifiees": UNDERSPEC_V4,
        "duree_s": round(time.time() - t0, 2),
        "clauses": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_bench_v4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return report


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


# =========================================================================
#  ======  SUITE dgov — recouvrement des supports de `topk(G·h)` (D14-S)  ==
#
#  Protocole : `experiments/EXP-2026-08-23-recouvrement-supports.md`
#  (PRE-ENREGISTRE ; correction de provenance du 2026-08-26, §15).
#
#  CPU seul, aucune mesure, aucun GPU, aucun poids de modèle chargé. Les seuls
#  fichiers touchés sont les `config.json` en cache (porte `V-norm`, deux
#  lignes) et les `.npz` d'états déjà sur disque (portes de provenance).
#
#  **AUCUNE MESURE AVANT `E = 0`** (§6.E, critère d'abandon).
#
#  Contenu exigé au §10 : **12 cellules énumérées**, **10 objets simulés**
#  (4 classes de `N`, 4 états de `C`, 2 issues de `Core`), et **cas ÉCHOUANTS
#  OBLIGATOIRES** pour `V-borne`, `V-P8` (7 paires), `V-core-S` (deux unités
#  d'une même tige), `V-ulp` (états centrés et placebo), `V-seed`, `V-t1`.
#  **Le cardinal est recompté par ÉNUMÉRATION, jamais par affirmation.**
# =========================================================================

import support_overlap as so  # noqa: E402

OUT_DIR_DGOV = ROOT / "experiments" / "results" / "recouvrement-supports"

UNDERSPEC_DGOV = dict(so.OPERATIONNALISATIONS)


def _dgov_paires_conformes(n=40, seed=0):
    """Paires synthétiques dont l'identité `|A∩B| ≥ p_pair` tient **par
    construction** : on tire deux `z` et on lit les quantités réelles."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        za = rng.standard_normal(so.D_DG)
        zb = 0.7 * za + 0.7 * rng.standard_normal(so.D_DG)
        out.append(so.quantites_de_paire(za, zb))
    return out


def _dgov_paire_violante():
    """Contre-exemple ÉCHOUANT obligatoire de `V-borne` : une paire dont
    `|A∩B| < p_sym`. **Impossible en arithmétique exacte** — c'est précisément
    pourquoi toute violation est un **BUG DE MESURE** et jamais un résultat
    (§6.B). Le banc la fabrique pour vérifier que la porte mord."""
    return [{"inter": 2, "p_sym": 9, "p_pair": 5, "cos": 0.42}]


_CUM_CACHE = {}


def _dgov_cum():
    """Table `cum` gravée en fp64, mise en cache pour le banc."""
    if "d" not in _CUM_CACHE:
        _CUM_CACHE["d"] = so.charger_cum()
    return _CUM_CACHE["d"]


def _dgov_cum_decale():
    """Contre-exemple ÉCHOUANT : une table de contrôle décalée de 1e−3, soit
    ~7 × la tolérance gravée. Le banc doit la refuser, pas s'y ajuster."""
    import tempfile
    d = _dgov_cum()
    faux = {"_source": "contrôle fabriqué pour le banc",
            "cum_theorique_Z569_6": [d["cum"][j] + 1e-3
                                     for j in range(1, so.CUM_J_MAX + 1)]}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "faux.json"
        p.write_text(json.dumps(faux), encoding="utf-8")
        r = so.charger_cum(p)
    return (FAIL if not r["controle"]["conforme"] else PASS), {
        "ecart_max": r["controle"]["ecart_max_gravure_moins_controle"],
        "tolerance": so.CUM_TOLERANCE,
        "regle": "un écart au-dessus de la tolérance est un DÉFAUT À PUBLIER, "
                 "pas un ajustement"}


def _dgov_douze_psym():
    """Les douze `p_sym` de §16.1 recalculés depuis la gravure fp64."""
    d = _dgov_cum()
    cos = {("Qwen/Qwen2.5-1.5B", "S0"): 0.1471,
           ("Qwen/Qwen2.5-1.5B", "S2"): 0.1636,
           ("Qwen/Qwen2.5-1.5B", "S1"): 0.1883,
           ("gpt2", "S0"): 0.1886, ("gpt2", "S2"): 0.2070,
           ("Qwen/Qwen2.5-1.5B", "S3"): 0.2184, ("gpt2", "S1"): 0.2479,
           ("HuggingFaceTB/SmolLM2-360M", "S2"): 0.2566,
           ("HuggingFaceTB/SmolLM2-360M", "S0"): 0.2568,
           ("gpt2", "S3"): 0.2766,
           ("HuggingFaceTB/SmolLM2-360M", "S1"): 0.3174,
           ("HuggingFaceTB/SmolLM2-360M", "S3"): 0.3368}
    lignes, hors = {}, []
    for cle, c in cos.items():
        r = so.p_sym_theorique(c, d)
        lo, hi = so.P_SYM_ATTENDUS[cle]
        ok = r["statut"] == PASS and lo <= r["p_sym"] <= hi
        lignes[f"{cle[0]}|{cle[1]}"] = {"cos": c, "p_sym_grave": r["p_sym"],
                                        "attendu": [lo, hi], "conforme": ok}
        if not ok:
            hors.append(f"{cle[0]}|{cle[1]}")
    return (PASS if not hors else FAIL), {
        "par_cellule": lignes, "hors": hors,
        "prediction": "O ≥ 7 à 18 indices sur 64, soit 14 à 36 × la nulle"}


def _dgov_encadrement(conforme=True):
    """Encadrement synthétique : distribution de `p_sym` réalisé par cellule."""
    e = {cle: {"min": lo - 3, "q1": lo - 1, "median": lo, "q3": hi + 1,
               "max": hi + 3}
         for cle, (lo, hi) in so.P_SYM_ATTENDUS.items()}
    if not conforme:                   # IQR entièrement au-dessous du théorique
        e[("gpt2", "S3")] = {"min": 1, "q1": 1, "median": 2, "q3": 3, "max": 4}
    return e


def _dgov_psym_intersection_vide():
    """Intersection VIDE : `p_sym` doit valoir **0**, pas 1.

    Défaut trouvé **par la porte `V-borne` sur les données réelles** : un
    plancher à `p = 1` rend la borne PLUS FORTE QUE VRAIE et fabrique une
    violation sur chaque paire à `|A∩B| = 0` — 30 516 sur 669 060 au premier
    passage, concentrées dans `glob` et `type`, les conditions où le cosinus
    s'effondre vers 0.
    """
    za = np.concatenate([np.ones(so.K_TOPK) * 9.0, np.zeros(so.D_DG - so.K_TOPK)])
    zb = np.concatenate([np.zeros(so.D_DG - so.K_TOPK), np.ones(so.K_TOPK) * 9.0])
    s = so.quantites_de_paire(za, zb, so.K_TOPK)
    q = so.quantites_batch(za[None, :], zb[None, :], so.K_TOPK)
    ok = (s["inter"] == 0 and s["p_sym"] == 0 and int(q["p_sym"][0]) == 0
          and s["p_pair"] == 0 and so.p_sym_theorique(0.0)["p_sym"] == 0)
    return (PASS if ok else FAIL), {
        "inter": s["inter"], "p_sym_scalaire": s["p_sym"],
        "p_sym_batche": int(q["p_sym"][0]), "p_pair": s["p_pair"],
        "p_sym_theorique_a_cos_0": so.p_sym_theorique(0.0)["p_sym"],
        "regle": "|A∩B| ≥ p_sym doit rester satisfiable à q = 0 : g(0) = 0 "
                 "appartient à la recherche"}


def _dgov_psym_plancher_a_un():
    """Contre-exemple ÉCHOUANT : le plancher `p_sym ≥ 1` viole l'identité sur
    toute paire à intersection vide."""
    return FAIL, {"q": 0, "p_sym_avec_plancher": 1,
                  "violation": "0 < 1",
                  "n_violations_au_premier_passage": 30516,
                  "n_paires_verifiees": 669060,
                  "motif": "une borne plus forte que vraie fabrique des "
                           "violations et ferait déclarer BUG un run sain"}


def _dgov_psym_cos_negatif():
    """`p_sym` doit traiter `cos < 0` par sa **valeur absolue** : la borne
    porte sur `|cos|`, un cosinus négatif n'est pas une borne plus faible."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(so.K_TOPK)
    b = -a * 0.8 + 0.2 * rng.standard_normal(so.K_TOPK)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c = float(np.dot(a, b))
    p_neg = so.p_sym_realise(a, b, c)
    p_abs = so.p_sym_realise(a, b, abs(c))
    return (PASS if (c < 0 and p_neg == p_abs) else FAIL), {
        "cos": c, "p_sym_avec_cos_signe": p_neg,
        "p_sym_avec_valeur_absolue": p_abs}


def _dgov_batch_equiv(n=32, seed=7):
    """Les noyaux **batchés** (ceux que la mesure exécute) doivent coïncider
    avec les noyaux **lus** (ceux que le banc exerce partout ailleurs)."""
    rng = np.random.default_rng(seed)
    Za = rng.standard_normal((n, so.D_DG))
    Zb = 0.6 * Za + 0.8 * rng.standard_normal((n, so.D_DG))
    q = so.quantites_batch(Za, Zb, so.K_TOPK)
    ecarts = {"inter": 0, "cos": 0.0, "f": 0.0, "p_sym": 0, "p_pair": 0,
              "support": 0, "sigma": 0}
    for i in range(n):
        s = so.quantites_de_paire(Za[i], Zb[i], so.K_TOPK)
        ecarts["inter"] = max(ecarts["inter"], abs(s["inter"] - int(q["inter"][i])))
        ecarts["cos"] = max(ecarts["cos"], abs(s["cos"] - float(q["cos"][i])))
        ecarts["f"] = max(ecarts["f"], abs(s["f"] - float(q["f"][i])))
        ecarts["p_sym"] = max(ecarts["p_sym"], abs(s["p_sym"] - int(q["p_sym"][i])))
        ecarts["p_pair"] = max(ecarts["p_pair"],
                               abs(s["p_pair"] - int(q["p_pair"][i])))
        ecarts["support"] += int(not np.array_equal(
            so.support_topk(Za[i], so.K_TOPK),
            so._topk_batch(Za[i:i + 1], so.K_TOPK)[0]))
        if s["sigma_pm"] is not None:
            ecarts["sigma"] = max(ecarts["sigma"], abs(
                float(s["sigma_pm"]) - int(q["accord"][i]) / int(q["inter"][i])))
    ok = (ecarts["inter"] == 0 and ecarts["p_sym"] == 0 and ecarts["p_pair"] == 0
          and ecarts["support"] == 0 and ecarts["cos"] < 1e-12
          and ecarts["f"] < 1e-12 and ecarts["sigma"] < 1e-12)
    return (PASS if ok else FAIL), {"n_paires": n, "ecarts_max": ecarts}


def _dgov_poids_equiv(seed=5):
    """`_poids_vect` (mesure) ≡ `poids_bootstrap` (règle déclarée, banc)."""
    rng = np.random.default_rng(seed)
    tiges = [f"T{i}" for i in range(10)]
    tdp = [(tiges[int(rng.integers(0, 10))], tiges[int(rng.integers(0, 10))])
           for _ in range(200)]
    tdp += [(t, t) for t in tiges]
    mult_arr = rng.integers(0, 4, 10)
    mult = {t: int(mult_arr[i]) for i, t in enumerate(tiges)}
    ta, tb = so._indices_de_tige(tdp, tiges)
    w1 = so.poids_bootstrap(tdp, mult)
    w2 = so._poids_vect(ta, tb, mult_arr)
    return (PASS if np.array_equal(w1, w2) else FAIL), {
        "n_paires": len(tdp), "ecart_max": float(np.abs(w1 - w2).max()),
        "n_paires_intra_tige": int((ta == tb).sum())}


def _dgov_egalite_exacte():
    """Paire à `|A∩B| = 1` dont l'indice commun porte la plus grande magnitude
    des DEUX clés : la borne y est atteinte **avec égalité**, `|cos| = g(1)`.
    Sans tolérance, fp64 la fait basculer et fabrique une violation."""
    D, k = so.D_DG, so.K_TOPK
    rng = np.random.default_rng(11)
    za = np.zeros(D)
    zb = np.zeros(D)
    za[0] = 10.0
    zb[0] = 7.0                                   # indice commun, magnitude max
    za[1:k] = rng.uniform(0.5, 1.5, k - 1)
    zb[k:2 * k - 1] = rng.uniform(0.5, 1.5, k - 1)
    s = so.quantites_de_paire(za, zb, k)
    q = so.quantites_batch(za[None, :], zb[None, :], k)
    ok = (s["inter"] == 1 and s["inter"] >= s["p_sym"]
          and int(q["inter"][0]) >= int(q["p_sym"][0]))
    return (PASS if ok else FAIL), {
        "inter": s["inter"], "p_sym_scalaire": s["p_sym"],
        "p_sym_batche": int(q["p_sym"][0]),
        "p_sym_sans_tolerance": int(q["p_sym_strict"][0]),
        "decidee_par_la_tolerance": bool(q["decidee_par_la_tolerance"][0]),
        "cos": float(s["cos"]), "tolerance_ULP": so.TOL_ULP_BORNE}


def _dgov_tolerance_trop_large():
    """Contre-exemple ÉCHOUANT : un écart de `1e−6` — environ `10⁹` ULP à
    cette échelle — ne doit PAS être absorbé. La tolérance vaut 8 ULP."""
    c = 0.03
    ulp = float(np.spacing(c))
    return FAIL, {"ecart_teste": 1e-6, "ulp_a_cette_echelle": ulp,
                  "ecart_en_ulp": 1e-6 / ulp,
                  "tolerance_ULP": so.TOL_ULP_BORNE,
                  "absorbe": 1e-6 <= so.TOL_ULP_BORNE * ulp,
                  "motif": "une tolérance qui absorberait un écart réel serait "
                           "un masque, pas une correction de comparaison"}


def _dgov_supports_core(n_unites=10, commun=7, seed=0):
    """Supports synthétiques partageant `commun` indices sur toutes les unités."""
    rng = np.random.default_rng(seed)
    base = np.arange(commun)
    out = []
    for _ in range(n_unites):
        reste = rng.choice(np.arange(commun, so.D_DG), so.K_TOPK - commun,
                           replace=False)
        out.append(np.sort(np.concatenate([base, reste])))
    return out


def _dgov_marges(completes=True):
    m = {c: {"marge_ulp_min": 12.0 + i} for i, c in enumerate(so.CENTRAGES)}
    if not completes:            # 0-125 : la porte n'existait que pour les bruts
        for c in ("glob", "type", "auto", "plac"):
            m.pop(c)
    return m


def _dgov_spec_seed(complete=True):
    s = {"seed": 0,
         "generateur": "torch.randn(R, d, generator=manual_seed(seed))",
         "regle_de_tirage": "normalisation L2 par ligne ; indices 0..R−1",
         "indices": "0..R−1",
         "cardinal_par_modele": {m: 20 for m in so.MODELES},
         "appariement_inter_modeles": "par la NORME, jamais par le vecteur"}
    if not complete:             # 0-133 : la clause inexécutable
        s["appariement_inter_modeles"] = "mêmes vecteurs sur les trois modèles"
        s["cardinal_par_modele"] = {so.MODELES[0]: 20}
    return s


def _dgov_schema_diag(complet=True):
    cel = {}
    for m in so.MODELES:
        for x in so.CENTRAGES:
            for s in so.STRATES:
                cel[f"{m}|{x}|{s}"] = {
                    "f": 0.31, "sigma_pm": 0.52,
                    "O_cos_conjoint": {"O_moyen": 0.05, "cos_moyen": 0.24,
                                       "corr_O_cos": 0.4}}
    if not complet:              # 0-120 : sans `f`, le cycle ne peut pas donner
        cel[f"{so.MODELES[0]}|aucun|S3"]["f"] = None    # tort à son propre expert
    return {"modeles": list(so.MODELES), "conditions": list(so.CENTRAGES),
            "cellules": cel}


def _dgov_enumerer_cellules():
    """**12 cellules décisionnelles**, produites une à une par des IC
    synthétiques, puis recomptées par ÉNUMÉRATION."""
    n = float(so.N_HASARD)
    ics_n = {
        "HAUT": (( 0.20,  0.30), ( 0.20,  0.30)),
        "−":    ((-0.30, -0.20), (-0.30, -0.20)),
        "BAS":  ((-0.30, -0.20), (-0.004, 0.004)),
        "ind_L": ((-0.30, 0.30), (-0.30,  0.30)),
    }
    ics_c = {"c-mec": (-0.004, 0.004), "c-cent": (0.20, 0.30),
             "ind_Δ": (-0.30, 0.30)}
    eps_l, eps_e = 0.01, 0.01
    vues = []
    for cn, (icl, ico) in ics_n.items():
        obtenu_n = so.classe_n(icl, ico, eps_l, so.N_HASARD)
        for cc, icd in ics_c.items():
            obtenu_c = so.classe_c(icd, eps_e, so.N_HASARD)
            vues.append((obtenu_n, obtenu_c, cn, cc))
    attendu = so.cellules_decisionnelles()
    produites = sorted({(a, b) for a, b, _, _ in vues})
    conformes = all(a == cn and b == cc for a, b, cn, cc in vues)
    ok = conformes and produites == sorted(attendu["cellules"]) \
        and attendu["cardinal"] == 12 and len(produites) == 12
    return (PASS if ok else FAIL), {
        "cardinal_enumere": len(produites), "cardinal_attendu": 12,
        "cellules_produites": [list(x) for x in produites],
        "cellules_enumerees_par_le_noyau": [list(x)
                                            for x in attendu["cellules"]],
        "chaque_cellule_atteinte_par_un_IC": conformes,
        "n_hasard": str(so.N_HASARD), "corridor_2n": str(so.CORRIDOR),
        "n_flottant": n,
        "regle": "cardinal recompté PAR ÉNUMÉRATION, jamais par affirmation"}


def _dgov_enumerer_objets():
    """**10 objets simulés** : 4 classes de `N`, 4 états de `C`, 2 issues de
    `Core`. Chacun doit être **atteint** par une entrée synthétique."""
    eps = 0.01
    atteints = []
    for icl, ico in (((0.2, 0.3), (0.2, 0.3)), ((-0.3, -0.2), (-0.3, -0.2)),
                     ((-0.3, -0.2), (-0.004, 0.004)), ((-0.3, 0.3), (-0.3, 0.3))):
        atteints.append(("N", so.classe_n(icl, ico, eps, so.N_HASARD)))
    for icd in ((-0.3, -0.2), (-0.004, 0.004), (0.2, 0.3), (-0.3, 0.3)):
        atteints.append(("C", so.classe_c(icd, eps, so.N_HASARD)))
    vide = so.core(_dgov_supports_core(10, 0, seed=1), seuil=so.CORE_SEUIL)
    plein = so.core(_dgov_supports_core(10, 7, seed=2), seuil=so.CORE_SEUIL)
    atteints.append(("Core", "vide" if vide["taille"] == 0 else "non vide"))
    atteints.append(("Core", "non vide" if plein["taille"] > 0 else "vide"))
    attendu = so.objets_du_banc()
    ok = (sorted(atteints) == sorted(attendu["objets"])
          and attendu["cardinal"] == 10 and len(set(atteints)) == 10)
    return (PASS if ok else FAIL), {
        "objets_atteints": [list(x) for x in atteints],
        "objets_enumeres_par_le_noyau": [list(x) for x in attendu["objets"]],
        "cardinal_enumere": len(set(atteints)), "cardinal_attendu": 10,
        "Core_vide": vide["taille"], "Core_non_vide": plein["taille"],
        "regle": "cardinal recompté PAR ÉNUMÉRATION, jamais par affirmation"}


def _dgov_unite_de_n():
    """`n` et le corridor dans l'unité de publication de `O` — **fraction
    `p/64`**, pas l'indice.

    Défaut trouvé **par ce banc**, sur quatre clauses à la fois : `N_HASARD`
    valait `k²/D = 1/2` (l'unité INDICE) là où `O`, publié en `p/64`, appelle
    `n = 0.5/64 = 1/128` et `2n = 1/64`. Le corridor était **64 fois trop
    large** ⇒ `BAS` et `c-mec` vraies presque partout.
    """
    o = so.o_fraction(32 * 40, 40)            # 32 indices communs sur 40 paires
    centre = float(o - so.N_HASARD)
    ic_o = (centre - 1e-6, centre + 1e-6)
    juste = so.classe_n((-0.3, -0.2), ic_o, 0.01, so.N_HASARD)
    ok = (o == so.frac(1, 2) and so.N_HASARD == so.frac(1, 128)
          and so.CORRIDOR == so.frac(1, 64)
          and so.N_HASARD_INDICES == so.frac(1, 2) and juste != "BAS")
    return (PASS if ok else FAIL), {
        "O_de_32_indices": so.texte_fraction(o),
        "n_fraction": str(so.N_HASARD), "corridor_2n": str(so.CORRIDOR),
        "n_indices": str(so.N_HASARD_INDICES),
        "classe_sous_l_unite_juste": juste,
        "regle": "O est une fraction p/64 (0-132) : n et 2n le sont aussi"}


def _dgov_unite_de_n_fausse():
    """Contre-exemple ÉCHOUANT : le corridor pris en INDICES classe `BAS` une
    cellule où les supports partagent **32 indices sur 64**, soit 32 × le
    corridor."""
    o = so.o_fraction(32 * 40, 40)
    centre = float(o - so.N_HASARD)
    ic_o = (centre - 1e-6, centre + 1e-6)
    faux = so.classe_n((-0.3, -0.2), ic_o, 0.01, so.N_HASARD_INDICES)
    return (FAIL if faux == "BAS" else PASS), {
        "classe_sous_l_unite_fausse": faux,
        "O_de_32_indices": so.texte_fraction(o),
        "corridor_faux": str(2 * so.N_HASARD_INDICES),
        "motif": "32 indices sur 64 déclarés « bornés par 2 × le hasard »"}


def _dgov_exclusivite_ordres():
    """Ordres d'évaluation **GRAVÉS** (D18) : `N` = HAUT → − → BAS → ind_L,
    `C` = c-anti (court-circuitante) → c-mec → c-cent → ind_Δ ; la
    complémentation est évaluée **EN DERNIER**."""
    eps = 0.01
    # une entrée satisfaisant à la fois `HAUT` et `BAS` doit rendre `HAUT`
    prio_n = so.classe_n((0.20, 0.30), (-0.004, 0.004), eps, so.N_HASARD)
    # une entrée satisfaisant à la fois `c-anti` et `c-mec` doit rendre `c-anti`
    prio_c = so.classe_c((-0.02, -0.011), eps, so.N_HASARD)
    # `ind` par complémentation : ni l'une ni l'autre
    ind_n = so.classe_n((-0.3, 0.3), (-0.3, 0.3), eps, so.N_HASARD)
    ind_c = so.classe_c((-0.3, 0.3), eps, so.N_HASARD)
    ok = (prio_n == "HAUT" and prio_c == "c-anti"
          and ind_n == "ind_L" and ind_c == "ind_Δ")
    return (PASS if ok else FAIL), {
        "HAUT_prime_sur_BAS": prio_n, "c_anti_court_circuitante": prio_c,
        "ind_L_par_complementation": ind_n, "ind_delta_par_complementation": ind_c,
        "ordre_N": list(so.CLASSES_N), "ordre_C": list(so.CLASSES_C)}


def _dgov_tiges_inter(n_rep=4):
    """Jeu de paires INTER-tige équilibré sur `K_eff = 10` tiges."""
    tiges = [f"T{i}" for i in range(10)]
    tdp = [(tiges[a], tiges[b]) for a in range(10) for b in range(a + 1, 10)
           for _ in range(n_rep)]
    return tiges, tdp


def _dgov_deux_regles_eps():
    """Les DEUX règles d'affectation du signe sont calculables et publiées
    côte à côte, avec leur statut ; **aucune n'est promue**."""
    tiges, tdp = _dgov_tiges_inter()
    rng = np.random.default_rng(0)
    d = rng.normal(0.30, 0.04, len(tdp))
    r = {reg: so.eps_etoile(d, tdp, tiges, b=1000, seed=0, regle=reg)
         for reg in so.EPS_REGLES}
    ok = (len(so.EPS_REGLES) == 2
          and all(r[x]["epsilon_etoile"] is not None for x in so.EPS_REGLES)
          and r["min"]["statut_de_la_regle"].startswith("IMPLÉMENTÉE")
          and r["produit"]["statut_de_la_regle"].startswith("ALTERNATIVE"))
    return (PASS if ok else FAIL), {
        "cardinal_des_regles_enumere": len(so.EPS_REGLES),
        "par_regle": {x: {"epsilon_etoile": r[x]["epsilon_etoile"],
                          "statut": r[x]["statut_de_la_regle"],
                          "fraction_signe_non_retournable":
                              r[x]["fraction_de_paires_a_signe_non_retournable"]}
                      for x in so.EPS_REGLES},
        "regle": "les deux sont publiées ; AUCUNE n'est promue — arbitrage PI"}


def _dgov_regle_inconnue():
    """Contre-exemple ÉCHOUANT : une troisième règle inventée à l'exécution."""
    tiges, tdp = _dgov_tiges_inter(1)
    try:
        so.eps_etoile(np.ones(len(tdp)), tdp, tiges, b=10, seed=0,
                      regle="mediane")
    except ValueError as e:
        return FAIL, {"exception": str(e),
                      "motif": "poser une règle non déclarée à l'exécution "
                               "serait 0-52 ; le noyau la refuse"}
    return PASS, {"motif": "une règle non déclarée a été acceptée"}


def _dgov_plancher_structurel():
    """Le **plancher structurel** — puissance maximale disponible du test,
    indépendante de la donnée — est calculé, jamais supposé nul."""
    tiges, tdp = _dgov_tiges_inter()
    pmin = so.plancher_epsilon(tdp, tiges, b=1000, seed=0, regle="min")
    pprod = so.plancher_epsilon(tdp, tiges, b=1000, seed=0, regle="produit")
    intra = [(t, t) for t in tiges for _ in range(36)]
    pdeg = so.plancher_epsilon(intra, tiges, b=1000, seed=0, regle="produit")
    ok = (0.0 < pmin["plancher_relatif"] <= 1.0
          and 0.0 < pprod["plancher_relatif"] <= 1.0
          and abs(pdeg["plancher_relatif"] - 1.0) < 1e-12
          and pdeg["fraction_signe_non_retournable"] == 1.0)
    return (PASS if ok else FAIL), {
        "plancher_relatif_min": pmin["plancher_relatif"],
        "plancher_relatif_produit": pprod["plancher_relatif"],
        "plancher_structure_degeneree_100pct_intra_regle_produit":
            pdeg["plancher_relatif"],
        "fraction_signe_non_retournable_degeneree":
            pdeg["fraction_signe_non_retournable"],
        "definition": pmin["definition"],
        "regle": "sur une structure où AUCUN signe ne peut se retourner, le "
                 "plancher vaut EXACTEMENT 1 : la nulle de permutation est "
                 "l'observation elle-même"}


def _dgov_plancher_suppose_nul():
    """Contre-exemple ÉCHOUANT : un plancher supposé nul rend `c-cent`
    atteignable partout et masque la dégénérescence du test."""
    tiges = [f"T{i}" for i in range(10)]
    intra = [(t, t) for t in tiges for _ in range(36)]
    d = np.full(len(intra), 0.30)
    ic = so.bootstrap_tiges(d, intra, tiges, b=500, seed=0)
    faux = so.classe_c(ic["IC"], 0.0, so.N_HASARD)
    return (FAIL if faux == "c-cent" else PASS), {
        "classe_avec_epsilon_suppose_nul": faux,
        "motif": "sans plancher publié, un couloir réglé sur l'enveloppe nulle "
                 "de son propre estimateur (D28) ne se voit pas"}


def _dgov_puissance_partition_C():
    """**Clause de puissance / vacuité de la partition `C`** (tour de
    correction, B1-3).

    Quatre jeux synthétiques, chacun avec une réponse connue **par
    construction** :

    * `Δ*` **constant et non nul**, sur une structure où le signe PEUT se
      retourner ⇒ la classe **informative** `c-cent` doit rester
      **atteignable** (test de puissance maximale) ;
    * `Δ* = 0` par construction ⇒ **`c-cent` ne sort pas** ;
    * `Δ*` centré et bruité large ⇒ **`ind_Δ` atteignable** ;
    * structure **dégénérée** (plancher = 1) ⇒ le **DÉTECTEUR** doit annoncer
      `classe informative INATTEIGNABLE` — c'est lui qu'on teste ici, et le
      cas échouant obligatoire l'exerce en verdict.

    **Ambiguïtés DÉCLARÉES, non tranchées par le Builder :**

    1. la demande écrit « vérifier que `ind_Δ` reste atteignable » à `Δ*`
       **constant**, alors qu'à effet constant la classe attendue est `c-cent`
       (`ind_Δ` est la complémentation, atteinte par défaut). La clause vérifie
       **les deux lectures** et échoue si l'une tombe ;
    2. la demande écrit *« si `ind_Δ` est inatteignable même à `Δ*` constant,
       la clause doit le faire échouer et le compteur le publier »*, mais le
       même tour exige **`E = 0` sur `dgov` avant toute mesure**. Prises à la
       lettre ensemble, les deux rendent le protocole **inexécutable**. Lecture
       implémentée — celle qui le laisse exécutable : la clause teste le
       **détecteur** (il doit annoncer l'inatteignabilité, cas échouant
       obligatoire à l'appui) et **le compteur publie**, cellule par cellule,
       `classe_informative_inatteignable`. Arbitrage dû.
    """
    tiges = [f"T{i}" for i in range(10)]
    # structure où le signe PEUT se retourner sous la règle `min` : intra-tige.
    # (Sous `min`, une structure 100 % INTER-tige a un plancher de 1.0 — la
    # nulle de permutation y est l'observation elle-même. C'est mesuré, pas
    # supposé, et c'est le quatrième jeu.)
    tdp = [(t, t) for t in tiges for _ in range(36)]
    c = 0.30
    dc = np.full(len(tdp), c)
    eps_c = so.eps_etoile(dc, tdp, tiges, b=1000, seed=0)
    ic_c = so.bootstrap_tiges(dc, tdp, tiges, b=1000, seed=0)
    cl_c = so.classe_c(ic_c["IC"], eps_c["epsilon_etoile"], so.N_HASARD)
    z = np.zeros(len(tdp))
    eps_0 = so.eps_etoile(z, tdp, tiges, b=1000, seed=0)
    ic_0 = so.bootstrap_tiges(z, tdp, tiges, b=1000, seed=0)
    cl_0 = so.classe_c(ic_0["IC"], eps_0["epsilon_etoile"], so.N_HASARD)
    br = np.random.default_rng(1).normal(0.0, 0.30, len(tdp))
    eps_b = so.eps_etoile(br, tdp, tiges, b=1000, seed=0)
    ic_b = so.bootstrap_tiges(br, tdp, tiges, b=1000, seed=0)
    cl_b = so.classe_c(ic_b["IC"], eps_b["epsilon_etoile"], so.N_HASARD)
    # 4e jeu — DÉTECTEUR : structure dégénérée pour la règle `min` (inter-tige)
    _, inter = _dgov_tiges_inter()
    pl_deg = so.plancher_epsilon(inter, tiges, b=1000, seed=0, regle="min")
    ic_deg = so.bootstrap_tiges(np.full(len(inter), c), inter, tiges,
                                b=1000, seed=0)
    detecte = not (float(ic_deg["IC"][0])
                   > pl_deg["plancher_relatif"] * c)
    pl_ok = so.plancher_epsilon(tdp, tiges, b=1000, seed=0, regle="min")
    ok = (cl_c == "c-cent" and cl_0 != "c-cent" and cl_b == "ind_Δ"
          and detecte and pl_deg["plancher_relatif"] >= 1.0
          and pl_ok["plancher_relatif"] < 1.0)
    return (PASS if ok else FAIL), {
        "detecteur_d_inatteignabilite": {
            "structure": "100 % INTER-tige sous la règle `min`",
            "plancher_relatif": pl_deg["plancher_relatif"],
            "IC_inf_a_effet_constant": ic_deg["IC"][0],
            "classe_informative_inatteignable": bool(detecte),
            "structure_saine": "100 % INTRA-tige sous la règle `min`",
            "plancher_relatif_sain": pl_ok["plancher_relatif"]},
        "effet_constant_non_nul": {"Delta": c, "epsilon_etoile":
                                   eps_c["epsilon_etoile"],
                                   "IC": ic_c["IC"], "classe": cl_c,
                                   "attendu": "c-cent (classe informative)"},
        "effet_nul_par_construction": {"epsilon_etoile": eps_0["epsilon_etoile"],
                                       "IC": ic_0["IC"], "classe": cl_0,
                                       "attendu": "PAS c-cent"},
        "bruit_centre_large": {"epsilon_etoile": eps_b["epsilon_etoile"],
                               "IC": ic_b["IC"], "classe": cl_b,
                               "attendu": "ind_Δ"},
        "ambiguite_declaree": "la demande écrit « ind_Δ atteignable » à effet "
                              "CONSTANT ; à effet constant la classe attendue "
                              "est c-cent. LES DEUX lectures sont vérifiées. "
                              "Arbitrage dû."}


def _dgov_puissance_degeneree():
    """Contre-exemple ÉCHOUANT **obligatoire** : sur une structure où aucun
    signe ne peut se retourner (100 % intra-tige, règle `produit`), la classe
    informative `c-cent` est **inatteignable même à `Δ*` constant** — mode de
    vacuité 0-47 / 0-66. La clause doit le faire ÉCHOUER et le compteur le
    publier."""
    tiges = [f"T{i}" for i in range(10)]
    intra = [(t, t) for t in tiges for _ in range(36)]
    d = np.full(len(intra), 0.30)
    eps = so.eps_etoile(d, intra, tiges, b=1000, seed=0, regle="produit")
    ic = so.bootstrap_tiges(d, intra, tiges, b=1000, seed=0)
    cl = so.classe_c(ic["IC"], eps["epsilon_etoile"], so.N_HASARD)
    return (FAIL if cl != "c-cent" else PASS), {
        "fraction_signe_non_retournable":
            eps["fraction_de_paires_a_signe_non_retournable"],
        "epsilon_etoile": eps["epsilon_etoile"], "Delta": 0.30,
        "IC": ic["IC"], "classe_obtenue": cl,
        "motif": "ε* ≥ Δ* par construction du test : la classe informative de "
                 "la partition C est VIDE sur cette structure, quel que soit "
                 "l'effet. Vacuité 0-47 / 0-66, publiée et non masquée."}


def _dgov_couverture(complete=True):
    pub = {}
    for m in so.MODELES:
        for s in so.STRATES:
            pub[f"{m}|{s}"] = {"eps": {r: 0.3 for r in so.EPS_REGLES},
                               "mesures": {"LOO": 0.35, "split-half": 0.34}}
    if not complete:
        pub[f"{so.MODELES[0]}|S3"]["eps"].pop("produit")
        pub[f"{so.MODELES[1]}|S0"]["mesures"].pop("split-half")
    return so.couverture_publiee(pub)


def _dgov_split_half_sans_fuite():
    """La partition split-half ne fuit **jamais** : pour toute paire retenue,
    `μ` est estimée sur un ensemble qui ne contient **ni `a` ni `b`**."""
    n = 60
    moit = so.moities_de_cellule(n)
    retenues, cheval, fuites = 0, 0, 0
    for a in range(n):
        for b in range(a + 1, n):
            idx = so.indices_estimation_split_half(moit, a, b)
            if idx is None:
                cheval += 1
                continue
            retenues += 1
            if a in idx or b in idx:
                fuites += 1
    ok = (fuites == 0 and retenues > 0 and cheval > 0
          and retenues + cheval == n * (n - 1) // 2
          and int((moit == 0).sum()) == int((moit == 1).sum()) == n // 2)
    return (PASS if ok else FAIL), {
        "n_unites": n, "paires_totales": n * (n - 1) // 2,
        "paires_retenues": retenues, "paires_a_cheval_exclues": cheval,
        "fuites": fuites, "tailles_des_moities": [int((moit == 0).sum()),
                                                  int((moit == 1).sum())],
        "regle": "μ estimée sur la moitié 1−h : elle ne contient ni a ni b"}


def _dgov_split_half_fuite():
    """Contre-exemple ÉCHOUANT : estimer `μ` sur la moitié qui CONTIENT la
    paire — le split-half devient une fuite pure."""
    moit = so.moities_de_cellule(60)
    a, b = 0, 2                     # même moitié
    meme = np.flatnonzero(moit == int(moit[a]))
    return (FAIL if (a in meme and b in meme) else PASS), {
        "a_dans_l_ensemble_d_estimation": bool(a in meme),
        "b_dans_l_ensemble_d_estimation": bool(b in meme),
        "motif": "une estimation sur la moitié qui contient la paire n'est ni "
                 "un split-half ni un LOO : c'est une fuite"}


def _dgov_core_distribution():
    """`Core` publie sa **distribution adjacente** des comptes : un `|Core| = 0`
    ne se lit jamais en zéro plat (un indice à 8/10 annule déjà la nulle)."""
    r = so.core(_dgov_supports_core(10, 0, seed=4), seuil=so.CORE_SEUIL)
    d = r.get("distribution_adjacente") or {}
    ok = (r["taille"] == 0 and len(d) == 10 and ">=1" in d and ">=9" in d
          and "compte_max_sur_S" in r and d[">=1"] >= d[">=9"])
    return (PASS if ok else FAIL), {
        "taille_Core": r["taille"], "distribution_adjacente": d,
        "compte_max_sur_S": r.get("compte_max_sur_S"),
        "seuil": r["seuil"],
        "regle": "le seuil ≥ 9 est PRÉ-ENREGISTRÉ et ne bouge pas ; seule la "
                 "publication s'enrichit"}


def _dgov_core_zero_plat():
    """Contre-exemple ÉCHOUANT : `|Core| = 0` publié sans sa distribution."""
    return FAIL, {"Core": 0, "distribution_adjacente": None,
                  "motif": "un indice à 8/10 annule déjà la nulle Bin(10, "
                           "1/128) ; publier « 0 » sans la distribution le "
                           "masque"}


def _dgov_cardinal_unite():
    """Le cardinal intra-tige se publie **avec son unité**, et dans les DEUX
    unités : par cellule de capture, et dans l'unité de `P`."""
    par_cellule, n_cellules = 150, 6
    total = par_cellule * n_cellules
    ok = (total == 900 and 60 * n_cellules == 360 and 90 * n_cellules == 540)
    return (PASS if ok else FAIL), {
        "cardinal_par_cellule_de_capture": par_cellule,
        "n_cellules_de_capture": n_cellules,
        "cardinal_total_unite_de_P": total,
        "par_strate_unite_de_P": {"S3": 60 * n_cellules, "S2": 90 * n_cellules,
                                  "S1": 0, "S0": 0},
        "regle": "P est agrégé sur les 6 cellules : un cardinal PAR CELLULE "
                 "n'est pas dans l'unité de P"}


def _dgov_cardinal_sans_unite():
    """Contre-exemple ÉCHOUANT : un cardinal publié sans son unité."""
    return FAIL, {"cardinal_publie": 150, "unite": None,
                  "motif": "150 par cellule ou 150 au total ? Sans unité "
                           "nommée, le lecteur ne peut pas le savoir — et "
                           "150/900 n'est pas 100 % des paires S3 et S2"}


def _dgov_anomalie_non_expliquee():
    """L'anomalie `σ± ≪ 0.5` se consigne **NON EXPLIQUÉE**, en quantité
    diagnostique, **sans mécanisme proposé**."""
    cel = {"gpt2|type|S0": {"accord_numerateur": 1994, "accord_denominateur": 5083,
                            "sigma_pm": "1994/5083",
                            "sigma_pm_par_intersection": {"|A∩B|=1": 0.387}},
           "gpt2|aucun|S3": {"accord_numerateur": 5993, "accord_denominateur": 5993,
                             "sigma_pm": "1/1"}}
    b = so.bloc_anomalie_non_expliquee(cel)
    txt = json.dumps(b, ensure_ascii=False).lower()
    interdits = ["parce que", "s'explique par", "cause probable", "sans doute"]
    ok = (b["statut"] == "NON EXPLIQUÉ" and b["n_cellules_signalees"] == 1
          and "non décisionnelle" in b["nature"]
          and not any(t in txt for t in interdits))
    return (PASS if ok else FAIL), {
        "statut": b["statut"], "nature": b["nature"],
        "n_cellules_signalees": b["n_cellules_signalees"],
        "cellules": b["cellules"], "termes_de_mecanisme_detectes":
            [t for t in interdits if t in txt]}


def _dgov_anomalie_expliquee():
    """Contre-exemple ÉCHOUANT : nommer un mécanisme au lieu de nommer
    l'inconnu."""
    return FAIL, {"redaction_refusee": "σ± < 0.5 s'explique par la dépendance "
                                       "LOO entre les deux membres de la paire",
                  "motif": "la dépendance LOO est POSITIVE (Cov = σ²/58) et ne "
                           "peut pas produire cet écart ; nommer un mécanisme "
                           "faux est pire que nommer l'inconnu"}


def _dgov_cellules_diag(complet=True, conditions=None, sans_colonne=None,
                        sans_objet=False):
    """Les **60 cellules** de `V-diag` (3 modèles × 5 conditions du §8 ×
    4 strates), portant **toutes** les colonnes exigées, `p_sym` de cellule
    compris. Tour additif — L1/L2."""
    conds = tuple(conditions or so.CENTRAGES)
    cel = {}
    for m in so.MODELES:
        for x in conds:
            for s in so.STRATES:
                c = {"P": 360, "O_en_indices": 16.647222, "cos_moyen": 0.276588,
                     "f": 0.279088, "sigma_pm": "1/1",
                     "sigma_pm_flottant": 1.0, "mediane_indices": "16/1",
                     "IQR_indices": ["15/1", "18/1"],
                     "O_cos_conjoint": {"O_moyen": 0.260113,
                                        "cos_moyen": 0.276588,
                                        "corr_O_cos": 0.41},
                     "p_sym_median": 14}
                c["p_sym_cellule"] = so.p_sym_cellule(c)
                cel[f"{m}|{x}|{s}"] = c
    if not complet:
        # une cellule dont l'excès `O − p_sym` n'est pas publié : la table de
        # `V-diag` ne peut pas être écrite (0-120, étendu au tour additif)
        cel[f"{so.MODELES[0]}|aucun|S3"]["p_sym_cellule"][
            "exces_O_moins_p_sym_realise_median"] = None
    if sans_colonne:
        cel[f"{so.MODELES[0]}|aucun|S3"][sans_colonne] = None
    if sans_objet:
        # `SANS OBJET` est une valeur PUBLIÉE, pas un manque (D23)
        cel[f"{so.MODELES[0]}|aucun|S3"]["sigma_pm"] = so.SANS_OBJET
    return cel


def _dgov_decomposition(perturbation=0.0, n_rep=4, seed=3):
    """`Δ* = Δ_rem + Δ_add` sur des accumulateurs par paire synthétiques.
    `perturbation` casse l'identité — cas échouant MORDANT."""
    tiges, tdp = _dgov_tiges_inter(n_rep)
    rng = np.random.default_rng(seed)
    a = rng.normal(0.260, 0.03, len(tdp))
    t = a - rng.normal(0.225, 0.02, len(tdp))
    p = a + rng.normal(0.150, 0.01, len(tdp))
    d = so.decomposition_delta(a, t, p, tdp, tiges, b=200, seed=0)
    if perturbation:
        # on casse l'identité en déplaçant `Δ*` SEUL, sans toucher aux crans
        d["Delta_etoile_estime"] = float(d["Delta_etoile_estime"]) + perturbation
    v, det = so.identite_decomposition(
        d["Delta_rem_estime"], d["Delta_add_estime"], d["Delta_etoile_estime"],
        echelle=max(abs(float(d["Delta_rem_estime"])),
                    abs(float(d["Delta_add_estime"])),
                    abs(float(d["Delta_etoile_estime"])), 1e-300))
    ok = (v == PASS and d["identite_par_paire"]["verdict"] == PASS)
    return (PASS if ok else FAIL), {
        "identite_sur_les_estimes": det,
        "identite_par_paire": d["identite_par_paire"],
        "Delta_rem_indices": d["Delta_rem_en_indices"],
        "Delta_add_indices": d["Delta_add_en_indices"],
        "Delta_etoile_indices": d["Delta_etoile_en_indices"],
        "IC_rem": d["IC_Delta_rem"], "IC_add": d["IC_Delta_add"],
        "IC_Delta_etoile": d["IC_Delta_etoile"],
        "statut": d["statut"], "perturbation_injectee": perturbation}


def build_clauses_dgov(paires_ok, marges_ok, spec_ok, diag_ok):
    C = []

    def clause(name, pass_desc, fail_desc, cases_pass, cases_fail,
               structural=None, note=None):
        C.append({"clause": name, "pass_case": pass_desc, "fail_case": fail_desc,
                  "cases_pass": cases_pass, "cases_fail": cases_fail,
                  "structural": structural, "note": note})

    # ------------------------------------------------------- provenance
    clause("V-cache", "les trois `.npz` de v4 sont lisibles, hash publié",
           "un modèle absent du cache ⇒ re-forward, jamais un repli silencieux",
           [("cache v4 complet", PASS, lambda: so.v_cache(so.MODELES))],
           [("modèle inexistant", FAIL,
             lambda: so.v_cache(("modele/qui-n-existe-pas",)))],
           note="échec ⇒ re-forward autorisé (§14-4) : 53,82 s, VRAM ≤ 4,688 "
                "Gio en RÉSERVÉ (0-142).")

    clause("V-G (v2)",
           "G du projet instanciée et hashée ; cos fp32/fp64 sous 1e−6 ; "
           "divergence avec A3 publiée",
           "divergence fp32/fp64 au-dessus de la tolérance ⇒ arrêt de "
           "provenance (D14-R)",
           [("tolérance 1e−6", PASS,
             lambda: so.v_g((so.MODELES[0],), tol_precision=1e-6))],
           [("tolérance 0 — aucune divergence tolérée", FAIL,
             lambda: so.v_g((so.MODELES[0],), tol_precision=0.0))],
           note="v1 (« reproduire A3 bit-à-bit ») était INEXÉCUTABLE PAR "
                "CONSTRUCTION : A3 n'a jamais été calculé avec la G du projet "
                "(0-135). Aucune reproduction d'A3 n'est exigée ni possible.")

    clause("V-iid", "sanité de moments de G (prouvée par provenance, M3)",
           "une matrice non gaussienne ⇒ FAIL ; PAS de KS",
           [("G du projet, 512 lignes", PASS,
             lambda: so.v_iid(np.random.default_rng(0).standard_normal((512, 768))))],
           [("matrice uniforme [0,1]", FAIL,
             lambda: so.v_iid(np.random.default_rng(0).random((512, 768))))])

    # --------------------------------------------- portes exigées au §10
    clause("V-borne",
           "|A∩B| ≥ p_sym (borne SERRÉE, §16) sur 100 % des paires",
           "une paire à |A∩B| < p_sym ⇒ BUG DE MESURE, jamais un résultat",
           [("40 paires + encadrement conforme aux douze p_sym", PASS,
             lambda: so.v_borne(paires_ok, _dgov_encadrement(True))),
            ("40 paires, réalisée non encore mesurée", so.EN_ATTENTE,
             lambda: so.v_borne(paires_ok, None))],
           [("paire fabriquée inter=2 < p_sym=9", FAIL,
             lambda: so.v_borne(_dgov_paire_violante(),
                                _dgov_encadrement(True))),
            ("une cellule hors encadrement théorique", FAIL,
             lambda: so.v_borne(paires_ok, _dgov_encadrement(False)))],
           note="§16, D30 alinéa 2 : la borne gelée n'était pas fausse, elle "
                "était CORRECTE ET LÂCHE — majorer m_ψ ≤ 1 jette un facteur "
                "cos. Le seuil est ≥ cos, pas ≥ cos². membre (b) "
                "« la réalisée encadre la théorique » = EN-ATTENTE de Q-M6 ; "
                "poser un seuil ici serait 0-52, troisième occurrence.")

    clause("p_sym vs p_pair (double mesure D26)",
           "p_sym ≥ p_pair sur 100 % des paires ; p_pair reste DESCRIPTIF",
           "p_pair traité comme décisionnel ⇒ la borne lâche décide",
           [("40 paires réelles synthétiques", PASS,
             lambda: (PASS if all(p["p_sym"] >= p["p_pair"] for p in paires_ok)
                      else FAIL,
                      {"n": len(paires_ok),
                       "ecart_min_max": [min(p["ecart_psym_moins_ppair"]
                                             for p in paires_ok),
                                         max(p["ecart_psym_moins_ppair"]
                                             for p in paires_ok)],
                       "p_sym_min_max": [min(p["p_sym"] for p in paires_ok),
                                         max(p["p_sym"] for p in paires_ok)],
                       "p_pair_min_max": [min(p["p_pair"] for p in paires_ok),
                                          max(p["p_pair"] for p in paires_ok)]}))],
           [("borne lâche décisionnelle ⇒ prédiction plus FACILE", FAIL,
             lambda: (FAIL, {"motif": "D30 alinéa 2 n'autorise le resserrement "
                                      "que parce qu'il rend la prédiction plus "
                                      "DURE (O ≥ 7 à 18 au lieu de 1 à 5) ; "
                                      "revenir à la lâche inverserait le "
                                      "critère de direction"}))])

    clause("cum : gravure fp64 vs table de contrôle (§16.1)",
           "recalcul fp64 conforme au contrôle de lab-math à ±1.5e−4",
           "un contrôle décalé au-delà de la tolérance ⇒ défaut à publier",
           [("écart gravure − contrôle ≤ 1.5e−4", PASS,
             lambda: (PASS if _dgov_cum()["controle"]["conforme"] else FAIL,
                      {k: v for k, v in _dgov_cum()["controle"].items()
                       if k != "ecarts"})),
            ("croissante, concave, contiguë depuis 1, j_max = 20", PASS,
             lambda: (PASS if (_dgov_cum()["concave"]
                               and _dgov_cum()["croissante"]
                               and _dgov_cum()["contigue_depuis_1"]
                               and _dgov_cum()["j_max"] == 20) else FAIL,
                      {k: v for k, v in _dgov_cum().items()
                       if k not in ("cum", "controle")})),
            ("inverse vérifié par aller-retour math.erfc", PASS,
             lambda: (PASS if _dgov_cum()["verification_de_l_inverse"][
                 "residu_max_|Phi(ndtri(p))-p|"] < 1e-12 else FAIL,
                 _dgov_cum()["verification_de_l_inverse"]))],
           [("contrôle décalé de 1e−3", FAIL, _dgov_cum_decale)],
           note="scipy n'est PAS une dépendance du projet (requirements.txt : "
                "torch, transformers, pytest) : Φ⁻¹ est implémentée en fp64 "
                "(AS 241, Wichura 1988) et vérifiée par un aller-retour à "
                "travers math.erfc, qui ne partage aucune ligne avec elle. "
                "Substitution DÉCLARÉE à scipy.special.ndtri.")

    clause("Les douze p_sym (§16.1)",
           "la gravure fp64 reproduit les douze p_sym pré-enregistrés",
           "une valeur de cum extrapolée au lieu d'être calculée",
           [("douze cellules, frontières franches comprises", PASS,
             _dgov_douze_psym)],
           [("extrapolation linéaire de cum(8)", FAIL,
             lambda: (FAIL, {"cum_7": so.CUM_ANCRES[7],
                             "extrapolation_lineaire": round(
                                 2 * so.CUM_ANCRES[7] - so.CUM_ANCRES[6], 5),
                             "cum_8_grave": round(_dgov_cum()["cum"][8], 6),
                             "motif": "cum est CONCAVE : l'interpolation "
                                      "linéaire SOUS-ESTIME toujours, et elle "
                                      "a déjà déplacé une cellule"}))],
           note="les deux frontières franches (gpt2 S1 à 0.04 %, SmolLM2 S1 à "
                "0.08 %) sont sous le plancher numérique : elles s'écrivent "
                "12-13 et 16-17, et basculeraient vers le p INFÉRIEUR — sens "
                "CONSERVATEUR, jamais plus facile qu'annoncé.")

    clause("p_sym réalisé : |cos| et moyenne géométrique",
           "|cos_pair| et moyenne géométrique des masses top-p des DEUX clés",
           "la table MOYENNE des deux clés au lieu de la géométrique",
           [("40 paires, borne exacte vérifiée", PASS,
             lambda: (PASS if all(p["inter"] >= p["p_sym"] for p in paires_ok)
                      else FAIL,
                      {"n": len(paires_ok),
                       "regle": "|cos| et moyenne géométrique PAR CLÉ",
                       "n_violations": 0})),
            ("cos négatif traité par sa valeur absolue", PASS,
             _dgov_psym_cos_negatif)],
           [("table moyenne au lieu de géométrique", FAIL,
             lambda: (FAIL, {"motif": "la dérivation borne CHAQUE masse "
                                      "séparément ; la moyenne arithmétique "
                                      "des deux tables n'est pas √(T_φ·T_ψ) et "
                                      "n'est pas une borne"}))])

    clause("p_sym à intersection vide",
           "|A∩B| = 0 ⇒ p_sym = 0 (scalaire, batché et théorique)",
           "un plancher p_sym ≥ 1 fabrique une violation sur chaque paire vide",
           [("supports disjoints, cos = 0", PASS, _dgov_psym_intersection_vide)],
           [("plancher à 1 sur intersection vide", FAIL,
             _dgov_psym_plancher_a_un)],
           note="défaut trouvé PAR LA PORTE V-borne SUR LES DONNÉES RÉELLES : "
                "30 516 violations sur 669 060 paires, concentrées dans `glob` "
                "et `type`. Une borne plus forte que vraie aurait fait "
                "déclarer BUG un run sain — mode inverse du faux PASS.")

    clause("V-borne : tolérance ULP à l'égalité exacte",
           "|cos| = g(1) exactement ⇒ p_sym = 1, et le compteur de paires "
           "décidées par la tolérance est publié",
           "un écart réel (1e−6, soit ~10⁹ ULP) absorbé par la tolérance",
           [("égalité exacte, |A∩B| = 1 sur la plus grande magnitude", PASS,
             _dgov_egalite_exacte),
            ("compteur de paires décidées par la tolérance exposé", PASS,
             lambda: (PASS if "decidee_par_la_tolerance" in so.quantites_batch(
                 np.random.default_rng(1).standard_normal((4, so.D_DG)),
                 np.random.default_rng(2).standard_normal((4, so.D_DG)),
                 so.K_TOPK) else FAIL,
                 {"tolerance_ULP": so.TOL_ULP_BORNE,
                  "regle": "sans compteur publié, une tolérance est un masque"}))],
           [("écart de 1e−6 absorbé", FAIL, _dgov_tolerance_trop_large)],
           note="mesuré : 2 paires sur 669 060 tombaient du mauvais côté au "
                "dernier bit (0.0 et −1.0 ULP), sur des paires à |A∩B| = 1 où "
                "la borne est atteinte AVEC ÉGALITÉ. La tolérance ne desserre "
                "pas la borne : elle rend la comparaison fidèle à "
                "l'arithmétique réelle que la borne suppose.")

    clause("Encadrement : les TROIS lectures publiées",
           "l'IQR réalisé rencontre l'intervalle théorique (lecture B) ; A et C "
           "publiées à côté, avec leurs compteurs",
           "un IQR réalisé entièrement au-dessous du théorique",
           [("douze cellules synthétiques conformes", PASS,
             lambda: so._encadrement_theorique(_dgov_encadrement(True))),
            ("les trois lectures publiées par cellule et comptées", PASS,
             lambda: (PASS if (all(
                 "lecture_A_mediane_dans_theorique" in v
                 and "lecture_B_IQR_realise_rencontre_theorique" in v
                 and "lecture_C_contenance_stricte_q1_lo_hi_q3" in v
                 for v in so._encadrement_theorique(
                     _dgov_encadrement(True))[1]["par_cellule"].values())
                 and len(so._encadrement_theorique(
                     _dgov_encadrement(True))[1]["trois_lectures"]) == 4)
                 else FAIL,
                 so._encadrement_theorique(
                     _dgov_encadrement(True))[1]["trois_lectures"]))],
           [("IQR réalisé [1,3] contre théorique [14,14]", FAIL,
             lambda: so._encadrement_theorique(_dgov_encadrement(False)))],
           note="ambiguïté TRANCHÉE au tour de correction : « X encadre Y » = "
                "Y contenu dans X, ce qui désigne la lecture B (rôles "
                "syntaxiques respectés) ; la lecture C est la contenance "
                "STRICTE, version la plus forte de B. Les TROIS sont publiées "
                "côte à côte ; leur divergence est SANS EFFET sur le verdict.")

    clause("Noyaux batchés ≡ noyaux scalaires",
           "quantites_batch et support_topk batché coïncident avec les lus",
           "une version rapide qui diverge de la version lue",
           [("32 paires aléatoires", PASS, _dgov_batch_equiv),
            ("poids vectorisés ≡ poids_bootstrap", PASS, _dgov_poids_equiv)],
           [("divergence d'un indice sur le support", FAIL,
             lambda: (FAIL, {"motif": "une optimisation qui change le résultat "
                                      "est un défaut, pas une optimisation"}))],
           note="la mesure passe par les noyaux batchés ; le banc n'exerce que "
                "les noyaux lus. Sans cette clause, tout le banc porterait sur "
                "un code que la mesure n'exécute pas.")

    clause("V-P8", "toute cellule a ≥ 8 paires",
           "une cellule à 7 paires ⇒ exclue, cardinal publié (CAS OBLIGATOIRE)",
           [("4 cellules à 40, 90, 360, 1260 paires", PASS,
             lambda: so.v_p8({"gpt2|S3": 360, "gpt2|S2": 540,
                              "gpt2|S1": 2160, "gpt2|S0": 7560}))],
           [("une cellule à 7 paires", FAIL,
             lambda: so.v_p8({"gpt2|S3": 7, "gpt2|S2": 540,
                              "gpt2|S1": 2160, "gpt2|S0": 7560}))],
           note="0-119 : q95 par paire = 2 indices = le DOUBLE du corridor ; "
                "une paire isolée le dépasse 7.6 % du temps sous la nulle.")

    clause("V-t1", "aucune quantité n'est calculée à t−1 ; cardinal intra-tige "
                   "publié",
           "une quantité à t−1 dont l'ensemble contient deux unités d'une même "
           "tige (CAS OBLIGATOIRE)",
           [("registre t−1 vide, Core 1/tige, cardinal publié", PASS,
             lambda: so.v_t1([], {"S3": 360, "S2": 540, "S1": 0, "S0": 0},
                             core_S=list(range(10)),
                             tige_de={i: f"T{i}" for i in range(10)}))],
           [("quantité à t−1 avec paires intra-tige (membre i)", FAIL,
             lambda: so.v_t1([{"nom": "profil à t−1", "intra_tige": True,
                               "n_paires_intra_tige": 360}],
                             {"S3": 360}, list(range(10)),
                             {i: f"T{i}" for i in range(10)})),
            ("Core avec deux unités d'une même tige (membre ii)", FAIL,
             lambda: so.v_t1([], {"S3": 360}, list(range(10)),
                             {i: ("T0" if i == 9 else f"T{i}")
                              for i in range(10)})),
            ("cardinal intra-tige non publié (membre iii)", FAIL,
             lambda: so.v_t1([], None, list(range(10)),
                             {i: f"T{i}" for i in range(10)}))],
           note="0-143 (critique) : la clause d'origine VIDAIT S3 et S2, qui "
                "sont DÉFINIES par le partage de tige. Son motif était "
                "doublement faux — les états bit-identiques valent à t−1, pas "
                "à t (à t le suffixe EST le token de capture) ; et le "
                "bootstrap PAR TIGE ABSORBE la corrélation intra-tige au lieu "
                "d'en souffrir (0-50).")

    clause("V-t1 — S3 et S2 mesurées à `t`",
           "les paires intra-tige ne sont PAS exclues à t ; cardinal publié",
           "exclure les paires intra-tige vide S3 et S2 (0-143)",
           [("cardinal intra-tige > 0 et strates non vides", PASS,
             lambda: (PASS if so.v_t1([], {"S3": 360, "S2": 540},
                                      list(range(10)),
                                      {i: f"T{i}" for i in range(10)})[0] == PASS
                      else FAIL,
                      {"S3_et_S2_definies_par_le_partage_de_tige": True,
                       "portee": "à t, les paires intra-tige ne sont PAS "
                                 "exclues"}))],
           [("exclusion à t ⇒ S3 et S2 vides", FAIL,
             lambda: (FAIL, {"S3_apres_exclusion": 0, "S2_apres_exclusion": 0,
                             "motif": "auto-contradictoire avec le §7 et le "
                                      "§4.2, qui mesurent et chiffrent les "
                                      "quatre strates"}))])

    clause("V-core-S", "|S| = 10, une unité par tige, appartenance publiée",
           "deux unités d'une même tige dans S (CAS OBLIGATOIRE)",
           [("10 unités, 10 tiges", PASS,
             lambda: so.v_core_s(list(range(10)),
                                 {i: f"T{i}" for i in range(10)}))],
           [("deux unités de la tige T0", FAIL,
             lambda: so.v_core_s(
                 list(range(10)),
                 {i: ("T0" if i == 9 else f"T{i}") for i in range(10)}))],
           note="0-124 : la signature de Core vaut p < 1e-13 SOUS "
                "INDÉPENDANCE ; deux états d'une même tige rendent Core > 0 "
                "ATTENDU.")

    clause("V-ulp", "marge à la coupure publiée pour les CINQ conditions",
           "les états centrés et placebo sans marge (CAS OBLIGATOIRE)",
           [("cinq conditions", PASS, lambda: so.v_ulp(_dgov_marges(True)))],
           [("états bruts seuls (0-125)", FAIL,
             lambda: so.v_ulp(_dgov_marges(False)))],
           note="0-125 : les états centrés et placebo sont ceux dont les "
                "coordonnées sont RAPPROCHÉES DE ZÉRO par soustraction — les "
                "plus exposés au basculement de rang. La porte manquait "
                "exactement là où le risque est maximal.")

    clause("V-seed", "seed, générateur, règle de tirage gelés ; cardinal par "
                     "modèle ; appariement par la NORME",
           "appariement par le VECTEUR et cardinal partiel (CAS OBLIGATOIRE)",
           [("spécification complète, seed = 0", PASS,
             lambda: so.v_seed(_dgov_spec_seed(True)))],
           [("« mêmes vecteurs sur les trois modèles »", FAIL,
             lambda: so.v_seed(_dgov_spec_seed(False))),
            ("champ `seed` absent (distinct de seed = 0)", FAIL,
             lambda: so.v_seed({k: v for k, v in _dgov_spec_seed(True).items()
                                if k != "seed"}))],
           note="0-133 : d vaut 768 / 960 / 1536 — un vecteur de ℝ⁷⁶⁸ n'est "
                "pas un vecteur de ℝ¹⁵³⁶. Cinquième occurrence de la famille "
                "« cardinal périmé ».")

    # ----------------------------------------------- portes de publication
    clause("V-norm", "LayerNorm vs RMSNorm cité par ligne de config des trois "
                     "modèles",
           "une normalisation affirmée sans ligne de config",
           [("trois config.json en cache", PASS,
             lambda: so.v_norm({m: so.config_de_normalisation(m)
                                for m in so.MODELES}))],
           [("normalisation par croyance", FAIL,
             lambda: so.v_norm({m: {"normalisation": "LayerNorm",
                                    "ligne_de_config": None}
                                for m in so.MODELES}))],
           note="0-123 : sans `auto`, C-mod était une classe SANS CAUSE "
                "CANDIDATE — une classe qui se consigne et ne s'explique "
                "jamais.")

    clause("V-diag", "f, σ± et (O, cos) conjoint publiés par strate, condition "
                     "et modèle",
           "une cellule sans `f` ⇒ le rapport ne peut pas être écrit (§6.D)",
           [("schéma complet, 60 cellules", PASS, lambda: so.v_diag(diag_ok))],
           [("une cellule sans f", FAIL,
             lambda: so.v_diag(_dgov_schema_diag(False)))],
           note="0-120 : sans `f`, « les supports coïncident » et « quelques "
                "coordonnées géantes portent tout le cosinus » rendent le MÊME "
                "O — et le second est la branche de tort de Neuro (N7).")

    # ------------------------- tour additif L1/L2/L3 (clauses NEUVES)
    clause("V-diag-colonnes",
           "les 60 cellules (3 × 5 × 4) portent TOUTES les colonnes de "
           "V-diag — cardinal ÉNUMÉRÉ, jamais supposé",
           "une cellule sans excès `O − p_sym`, ou une condition manquante "
           "au cardinal (CAS OBLIGATOIRES)",
           [("60 cellules complètes", PASS,
             lambda: so.v_diag_colonnes(_dgov_cellules_diag(True))),
            ("SANS OBJET publié sur σ± — valeur, pas manque (D23)", PASS,
             lambda: so.v_diag_colonnes(
                 _dgov_cellules_diag(True, sans_objet=True)))],
           [("une cellule sans excès O − p_sym", FAIL,
             lambda: so.v_diag_colonnes(_dgov_cellules_diag(False))),
            ("48 cellules au lieu de 60 (condition `plac` absente)", FAIL,
             lambda: so.v_diag_colonnes(_dgov_cellules_diag(
                 True, conditions=[c for c in so.CENTRAGES if c != "plac"]))),
            ("une cellule sans `f`", FAIL,
             lambda: so.v_diag_colonnes(
                 _dgov_cellules_diag(True, sans_colonne="f")))],
           note="V-diag (l. 383) exige f, σ± et le couple (O, cos) conjoint "
                "par strate, par condition et par modèle. La quantité "
                "EXISTAIT ; elle n'était pas TABULÉE — le compte rendu ne "
                "portait `cos` que sous `aucun`. Cette clause compte les "
                "cellules × colonnes au lieu de les supposer (0-144).")

    clause("V-psym-conditions",
           "p_sym de cellule défini sur les CINQ conditions du §8 — cardinal "
           "COMPTÉ par condition",
           "quatre conditions au lieu de cinq, ou une cellule sans p_sym "
           "(CAS OBLIGATOIRES)",
           [("5 conditions × 12 cellules", PASS,
             lambda: so.v_psym_conditions(_dgov_cellules_diag(True)))],
           [("condition `type` absente ⇒ 4/5", FAIL,
             lambda: so.v_psym_conditions(_dgov_cellules_diag(
                 True, conditions=[c for c in so.CENTRAGES if c != "type"]))),
            ("une cellule sans p_sym de cellule", FAIL,
             lambda: so.v_psym_conditions(
                 _dgov_cellules_diag(True, sans_colonne="p_sym_cellule")))],
           note="La demande écrit « p_sym défini sur les 5 conditions, "
                "cardinal compté, jamais supposé ». Le cardinal par "
                "condition est publié.")

    clause("V-identite-decomposition",
           "Δ_rem + Δ_add = Δ* à la tolérance ULP DÉJÀ EN VIGUEUR "
           "(TOL_ULP_BORNE), par paire ET sur les estimés",
           "un Δ* déplacé de 1e−9 ⇒ FAIL (CAS OBLIGATOIRE)",
           [("360 paires synthétiques, bootstrap de tiges", PASS,
             lambda: _dgov_decomposition(0.0)),
            ("identité sur trois scalaires publiés", PASS,
             lambda: so.identite_decomposition(
                 0.2601128472222222 - 0.0353, 0.4105 - 0.2601128472222222,
                 0.4105 - 0.0353))],
           [("Δ* déplacé de 1e−9 sans toucher aux crans", FAIL,
             lambda: _dgov_decomposition(1e-9)),
            ("Δ* déplacé de 1e−6", FAIL,
             lambda: so.identite_decomposition(0.2248, 0.1504,
                                               0.3752 + 1e-6))],
           note="Δ_rem = O_aucun − O_type (cran RETIRÉ), Δ_add = O_plac − "
                "O_aucun (cran AJOUTÉ). DESCRIPTIFS : ils ne modifient aucune "
                "classe, aucun seuil, aucune borne. Δ* reste la primaire "
                "gelée ; cette clause ne vérifie qu'une IDENTITÉ.")

    clause("V-perimetre", "les trois éléments du §4.9 présents",
           "le successeur désigné manquant",
           [("bloc complet", PASS, lambda: so.v_perimetre(bloc_perimetre_dgov()))],
           [("sans successeur désigné", FAIL,
             lambda: so.v_perimetre({k: v for k, v in bloc_perimetre_dgov().items()
                                     if k != "successeur_designe"}))])

    clause("V-schéma", "aucun terme interdit ; O en fraction exacte ; phrase "
                       "gravée (xvi) verbatim",
           "un terme du vocabulaire interdit dans le schéma de sortie",
           [("schéma conforme", PASS, lambda: so.v_schema(schema_dgov()))],
           [("« séparation de patterns » dans le schéma", FAIL,
             lambda: so.v_schema(dict(schema_dgov(),
                                      commentaire="séparation de patterns")))],
           note="la phrase gravée (xvi) contient elle-même un terme proscrit "
                "(« chemin d'écriture ») : elle NOMME la limite et est retirée "
                "du balayage — sinon la porte échouerait sur la formulation "
                "que le protocole rend obligatoire.")

    # ---------------------------------------- partitions, cardinaux, ordres
    clause("Cellules décisionnelles",
           "12 cellules, chacune atteinte par un IC, recomptées par énumération",
           "un cardinal AFFIRMÉ au lieu d'être recompté",
           [("énumération exécutée", PASS, _dgov_enumerer_cellules)],
           [("cardinal affirmé à 16 (4 × 4, c-anti comptée)", FAIL,
             lambda: (FAIL if so.cellules_decisionnelles()["cardinal"] != 16
                      else PASS,
                      {"cardinal_affirme": 16,
                       "cardinal_enumere": so.cellules_decisionnelles()["cardinal"],
                       "motif": "c-anti est court-circuitante : elle ne croise "
                                "pas N"}))],
           note="famille 0-76(i) / 0-86 / 0-94 / 0-101 / 0-133 — cinq "
                "occurrences de « cardinal périmé ».")

    clause("Unité de `n` et du corridor",
           "n = 1/128 et 2n = 1/64 dans l'unité de publication de O (p/64)",
           "le corridor pris en INDICES classe BAS 32 indices partagés sur 64",
           [("échelle p/64", PASS, _dgov_unite_de_n)],
           [("échelle en indices (n = 1/2, 2n = 1)", FAIL,
             _dgov_unite_de_n_fausse)],
           note="défaut trouvé PAR CE BANC, sur quatre clauses à la fois : le "
                "corridor était 64 × trop large ⇒ BAS et c-mec vraies presque "
                "partout. Aucune donnée du run n'a été nécessaire pour le voir.")

    clause("ε_Λ dérivé de l'enveloppe nulle (§4.3)",
           "ε_Λ = 1.96 × 0.01097/√P, décroissant en P ; corridor 2n ABSOLU",
           "ε_Λ posé à 0, ou réglé sur l'enveloppe de son propre estimateur",
           [("P = 8, 360, 7560", PASS,
             lambda: (PASS if (so.epsilon_lambda(8) > so.epsilon_lambda(360)
                               > so.epsilon_lambda(7560) > 0
                               and abs(so.epsilon_lambda(8) - 0.0076) < 1e-3)
                      else FAIL,
                      {"eps_P8": so.epsilon_lambda(8),
                       "eps_P360": so.epsilon_lambda(360),
                       "eps_P7560": so.epsilon_lambda(7560),
                       "sd_nulle_par_paire": so.SD_NULLE_PAR_PAIRE,
                       "corridor_2n": str(so.CORRIDOR)})),
            ("le corridor reste 2n, indépendant de P", PASS,
             lambda: (PASS if so.CORRIDOR == so.frac(1, 64) else FAIL,
                      {"corridor": str(so.CORRIDOR),
                       "regle": "couloir d'équivalence ABSOLU (0-81 / D28)"}))],
           [("ε_Λ = 0 ⇒ HAUT dès que IC_inf(Λ) > 0", FAIL,
             lambda: (FAIL if so.classe_n((1e-9, 0.3), (0.2, 0.3), 0.0,
                                          so.N_HASARD) == "HAUT" else PASS,
                      {"motif": "sans marge 1×, une significativité de "
                                "10⁻⁹ suffirait à prononcer HAUT",
                       "eps_lambda_correct_P360": so.epsilon_lambda(360)}))],
           note="§4.3 pré-enregistre l'enveloppe ; ε_Λ n'a jamais été laissé au "
                "choix de l'implémentation.")

    clause("Objets du banc",
           "10 objets, chacun ATTEINT par une entrée synthétique",
           "un objet inatteignable ⇒ classe vide par construction",
           [("énumération exécutée", PASS, _dgov_enumerer_objets)],
           [("cardinal affirmé à 8", FAIL,
             lambda: (FAIL if so.objets_du_banc()["cardinal"] != 8 else PASS,
                      {"cardinal_affirme": 8,
                       "cardinal_enumere": so.objets_du_banc()["cardinal"]}))])

    clause("Ordres gravés (D18)",
           "HAUT prime sur BAS ; c-anti court-circuitante ; ind par "
           "complémentation EN DERNIER",
           "une entrée doublement satisfaite classée par la seconde clause",
           [("priorités exécutées", PASS, _dgov_exclusivite_ordres)],
           [("ordre inversé : BAS avant HAUT", FAIL,
             lambda: (FAIL if so.classe_n((0.20, 0.30), (-0.004, 0.004),
                                          0.01, so.N_HASARD) != "BAS" else PASS,
                      {"observe": so.classe_n((0.20, 0.30), (-0.004, 0.004),
                                              0.01, so.N_HASARD),
                       "attendu_du_cas_faux": "BAS"}))])

    clause("Deux écritures obligatoires (§4.6)",
           "ε* ≥ 2n ⇒ recouvrement VIDE déclaré ; ε* < 2n ⇒ région masquée "
           "publiée",
           "une région masquée passée sous silence",
           [("ε* = 0.05 ≥ 2n", PASS,
             lambda: (PASS if so.ecritures_obligatoires(0.05)["cas"] == 1
                      else FAIL, so.ecritures_obligatoires(0.05))),
            ("ε* = 0.002 < 2n, région publiée", PASS,
             lambda: (PASS if so.ecritures_obligatoires(
                 0.002, region_masquee=0.11)["region_masquee"] != so.SANS_OBJET
                 else FAIL, so.ecritures_obligatoires(0.002, region_masquee=0.11)))],
           [("ε* < 2n sans région publiée", FAIL,
             lambda: (FAIL if so.ecritures_obligatoires(0.002)["region_masquee"]
                      == so.SANS_OBJET else PASS,
                      so.ecritures_obligatoires(0.002)))],
           note="0-131 : sous l'alternative, un effet entre ε* et 2n est "
                "SIGNIFICATIF et classé NÉGLIGEABLE — étouffement assumé, "
                "jamais silencieux.")

    clause("R dérivé (§4.5)",
           "R = ⌈10·σ̂_dir²/σ̂_Δ²⌉, plafond 300 ; dépassement PUBLIÉ",
           "un R tronqué en silence au plafond",
           [("σ_dir = 0.02, σ_Δ = 0.015 ⇒ R sous plafond", PASS,
             lambda: (PASS if not so.r_derive(0.02, 0.015)["depassement"]
                      else FAIL, so.r_derive(0.02, 0.015))),
            ("σ_dir = 0.30, σ_Δ = 0.015 ⇒ dépassement publié", PASS,
             lambda: (PASS if so.r_derive(0.30, 0.015)["depassement"]
                      else FAIL, so.r_derive(0.30, 0.015)))],
           [("σ_Δ = 0 ⇒ SANS OBJET, jamais un R inventé", FAIL,
             lambda: (FAIL if so.r_derive(0.02, 0.0)["R"] is None else PASS,
                      so.r_derive(0.02, 0.0)))],
           note="0-126-iii / 0-52 : R choisi à la main est 0-52 rejoué.")

    # ---------------------------------------------- noyaux de la mesure
    clause("support_topk", "exactement k indices, ex æquo départagés par indice",
           "une coupure par SEUIL rend plus de k indices sur ex æquo",
           [("z gaussien, k = 64", PASS,
             lambda: (PASS if len(so.support_topk(
                 np.random.default_rng(0).standard_normal(so.D_DG))) == so.K_TOPK
                 else FAIL, {"k": so.K_TOPK})),
            ("z avec 70 ex æquo à la coupure", PASS,
             lambda: (PASS if len(so.support_topk(
                 np.concatenate([np.ones(70), np.zeros(so.D_DG - 70)]))) == so.K_TOPK
                 else FAIL, {"ex_aequo": 70, "k": so.K_TOPK}))],
           [("coupure par seuil |z| ≥ q sur 70 ex æquo", FAIL,
             lambda: (FAIL, {"n_indices_rendus_par_le_seuil": 70,
                             "k_attendu": so.K_TOPK,
                             "motif": "le seuil `|z| >= q_(k)` du chemin d'A3 "
                                      "peut dépasser k ; `topk` ne le peut pas"}))],
           note="0-135, quatrième différence — la seule INERTE : à G égale, "
                "seuil et topk donnent 0.0 EXACT sur 4/4 (aucun ex æquo dans "
                "le matériau réel).")

    clause("σ± sur intersection vide",
           "σ± SANS OBJET quand |A∩B| = 0 — jamais 0",
           "σ± publié à 0 sur une intersection vide",
           [("deux supports disjoints", PASS,
             lambda: (PASS if so.quantites_de_paire(
                 np.concatenate([np.ones(64) * 9, np.zeros(so.D_DG - 64)]),
                 np.concatenate([np.zeros(so.D_DG - 64), np.ones(64) * 9])
             )["sigma_pm"] is None else FAIL, {"regle": "SANS OBJET, jamais 0"}))],
           [("σ± = 0 sur intersection vide", FAIL,
             lambda: (FAIL, {"motif": "une quantité sans domaine de définition "
                                      "se publie SANS OBJET, jamais 0"}))])

    clause("O en fraction exacte",
           "O agrégé en p/64 exact ; cellule vide ⇒ SANS OBJET",
           "O publié en flottant tronqué (0-132)",
           [("128 intersections sur 40 paires", PASS,
             lambda: (PASS if so.o_fraction(128, 40) == so.frac(128, 2560)
                      else FAIL, {"O": so.texte_fraction(so.o_fraction(128, 40))})),
            ("cellule vide", PASS,
             lambda: (PASS if so.o_fraction(0, 0) is None else FAIL,
                      {"regle": "SANS OBJET, jamais 0"}))],
           [("flottant tronqué à 4 décimales", FAIL,
             lambda: (FAIL, {"valeur_tronquee": round(128 / 2560, 4),
                             "valeur_exacte": so.texte_fraction(
                                 so.o_fraction(128, 40)),
                             "motif": "détruit l'exactitude dont dépendent "
                                      "V-borne et Core"}))])

    clause("p_pair réalisé",
           "p_pair = max(p_a, p_b) ; |A∩B| ≥ p_pair sur des paires réelles",
           "p_pair calculé sur un seul membre (borne plus faible)",
           [("40 paires synthétiques", PASS,
             lambda: (PASS if all(p["inter"] >= p["p_pair"] for p in paires_ok)
                      else FAIL,
                      {"n": len(paires_ok),
                       "p_pair_min_max": [min(p["p_pair"] for p in paires_ok),
                                          max(p["p_pair"] for p in paires_ok)],
                       "inter_min_max": [min(p["inter"] for p in paires_ok),
                                         max(p["inter"] for p in paires_ok)]}))],
           [("max(p_a,p_b) ignoré ⇒ borne desserrée", FAIL,
             lambda: (FAIL, {"motif": "la dérivation vaut pour CHAQUE membre : "
                                      "|A∩B| ≥ max(p_a, p_b) est exact"}))])

    clause("Bascule split-half (§14-2)",
           "n_cell = 60 ⇒ LOO principal ; n_cell = 20 ⇒ split-half principal",
           "une bascule décidée APRÈS lecture des données",
           [("n_cell = 60", PASS,
             lambda: (PASS if so.bascule_split_half(60)["mesure_principale"]
                      == "LOO" else FAIL, so.bascule_split_half(60))),
            ("n_cell = 20", PASS,
             lambda: (PASS if so.bascule_split_half(20)["mesure_principale"]
                      == "split-half" else FAIL, so.bascule_split_half(20)))],
           [("bascule à n_cell = 20 refusée", FAIL,
             lambda: (FAIL if so.bascule_split_half(20)["bascule"] else PASS,
                      {"motif": "0-117 : le biais résiduel +1/(n_cell−2) ≈ 0.05 "
                                "est DE LA TAILLE de la quantité décidée — il "
                                "fabriquerait c-cent"}))])

    clause("Core / Core-G",
           "Core non vide ⇒ Core-G en fraction ; Core vide ⇒ SANS OBJET",
           "Core-G publié à 0 sur un Core vide",
           [("7 indices communs sur 10 unités", PASS,
             lambda: (PASS if so.core(_dgov_supports_core(10, 7, 2))["taille"] == 7
                      else FAIL, so.core(_dgov_supports_core(10, 7, 2)))),
            ("Core vide ⇒ Core-G SANS OBJET", PASS,
             lambda: (PASS if so.core_g([], np.arange(64))["Core_G"]
                      == so.SANS_OBJET else FAIL, so.core_g([], np.arange(64))))],
           [("Core-G = 0 sur Core vide", FAIL,
             lambda: (FAIL, {"motif": "support vide ≠ zéro : la fraction n'a "
                                      "pas de domaine de définition"}))])

    # ------- tour de correction : sensibilité `ε*`, plancher, split-half
    clause("ε* : les DEUX règles de signe, publiées côte à côte",
           "règles `min` (implémentée) et `produit` (alternative) calculées et "
           "publiées avec leur statut ; AUCUNE promue",
           "une troisième règle inventée à l'exécution, ou une règle promue",
           [("deux règles sur un jeu inter-tige", PASS, _dgov_deux_regles_eps)],
           [("règle non déclarée `mediane`", FAIL, _dgov_regle_inconnue)],
           note="le §4.5 ne dit pas ce que devient une paire à DEUX tiges "
                "(S1, S0). L'affectation à min(t_a, t_b) N'EST PAS "
                "pré-enregistrée : c'est une opérationnalisation déclarée. La "
                "règle `produit` est aussi naturelle et donne d'autres ε*. Les "
                "deux sont publiées ; l'arbitrage est PI. Publication de "
                "SENSIBILITÉ — ne dépend d'aucun résultat, ne peut que durcir "
                "la lecture (D30 alinéa 2).")

    clause("Plancher structurel de ε* (puissance maximale du test)",
           "plancher calculé par simulation à effet CONSTANT ; vaut EXACTEMENT "
           "1 quand aucun signe ne peut se retourner",
           "un plancher supposé nul ⇒ c-cent atteignable partout",
           [("K_eff = 10, règles min et produit, structure dégénérée", PASS,
             _dgov_plancher_structurel)],
           [("ε* supposé nul sur une structure dégénérée", FAIL,
             _dgov_plancher_suppose_nul)],
           note="D28 : le couloir d'équivalence est ABSOLU (2n) et ne se règle "
                "jamais sur l'enveloppe nulle de son propre estimateur. Le "
                "plancher dit la puissance MAXIMALE disponible du test, "
                "indépendamment de la donnée ; il se publie par cellule et par "
                "règle.")

    clause("Puissance / vacuité de la partition C",
           "Δ* constant non nul ⇒ c-cent atteignable ; Δ* = 0 ⇒ c-cent ne sort "
           "pas ; bruit centré large ⇒ ind_Δ atteignable",
           "structure où AUCUN signe ne se retourne ⇒ classe informative "
           "INATTEIGNABLE même à Δ* constant (CAS OBLIGATOIRE)",
           [("trois jeux à réponse connue par construction", PASS,
             _dgov_puissance_partition_C)],
           [("100 % intra-tige sous la règle `produit`", FAIL,
             _dgov_puissance_degeneree)],
           note="mode de vacuité 0-47 / 0-66 : une classe vide PAR "
                "CONSTRUCTION du test se publie, elle ne se découvre pas après "
                "coup. Ambiguïté DÉCLARÉE : la demande écrit « ind_Δ "
                "atteignable » à effet constant, où la classe attendue est "
                "c-cent ; les DEUX lectures sont vérifiées. Arbitrage dû.")

    clause("Couverture énumérée : deux règles ε* × deux mesures",
           "sur chaque cellule, les deux règles ε* ET les deux mesures "
           "(LOO, split-half) sont présentes ; cardinal COMPTÉ",
           "une règle ou une mesure manquante sur une cellule",
           [("36 = 3 modèles × 4 strates × 3 conditions ; 12 cellules "
             "décisionnelles", PASS, lambda: _dgov_couverture(True))],
           [("règle `produit` et mesure split-half retirées", FAIL,
             lambda: _dgov_couverture(False))],
           note="ambiguïté DÉCLARÉE : « 12 cellules × 3 modèles » et « 36 "
                "cellules » ne désignent pas le même objet. Les DEUX cardinaux "
                "sont comptés et publiés avec leur unité nommée. Arbitrage dû. "
                "Famille « cardinal périmé », sixième occurrence évitée par "
                "comptage.")

    clause("Split-half : partition sans fuite (§5-7, D26)",
           "μ estimée sur la moitié qui ne contient NI a NI b ; paires à cheval "
           "exclues, cardinal publié",
           "μ estimée sur la moitié qui CONTIENT la paire (CAS OBLIGATOIRE)",
           [("60 unités, 1770 paires énumérées", PASS,
             _dgov_split_half_sans_fuite)],
           [("estimation sur la moitié contenant a et b", FAIL,
             _dgov_split_half_fuite)],
           note="le §5-7 grave « double mesure (D26) : LOO ET split-half, "
                "publiés côte à côte » et le §4.8 le liste. Le premier tour "
                "n'avait implémenté QUE la bascule (§14-2) : annoncé et "
                "absent. Le LOO reste PRINCIPAL (n_cell = 60 ≥ 30) ; le "
                "split-half est le CONTRÔLE, avec un LOO refait sur le même "
                "sous-ensemble pour que l'écart ne se confonde pas avec un "
                "changement de jeu de paires.")

    clause("Core : distribution adjacente des comptes",
           "|Core| et la distribution ≥1..≥10 et le max publiés",
           "|Core| = 0 publié en zéro plat, sans sa distribution",
           [("Core vide, distribution complète", PASS, _dgov_core_distribution)],
           [("zéro plat", FAIL, _dgov_core_zero_plat)],
           note="le seuil ≥ 9/10 est PRÉ-ENREGISTRÉ et NE BOUGE PAS ; seule la "
                "publication s'enrichit. Un indice à 8/10 annule déjà la nulle "
                "Bin(10, 1/128) — publier « 0 » sans la distribution le masque.")

    clause("Cardinal intra-tige : unité NOMMÉE",
           "publié dans les deux unités — par cellule de capture ET dans "
           "l'unité de P (× 6 cellules)",
           "un cardinal publié sans son unité",
           [("150 par cellule = 900 dans l'unité de P", PASS,
             _dgov_cardinal_unite)],
           [("« 150 » sans unité", FAIL, _dgov_cardinal_sans_unite)],
           note="150 par cellule de capture, mais P est agrégé sur 6 cellules : "
                "900 au total, S3 360 et S2 540 — soit 100 % des paires S3 et "
                "S2. Les deux nombres sont vrais, dans deux unités.")

    clause("Anomalie σ± ≪ 0.5 : consignée NON EXPLIQUÉE",
           "statut NON EXPLIQUÉ, nature diagnostique non décisionnelle, aucun "
           "mécanisme proposé",
           "un mécanisme nommé à la place de l'inconnu",
           [("bloc d'anomalie sur une cellule à ~15 σ", PASS,
             _dgov_anomalie_non_expliquee)],
           [("« s'explique par la dépendance LOO »", FAIL,
             _dgov_anomalie_expliquee)],
           note="σ± = 0.392288 (gpt2|type|S0, 1994/5083) et 0.4945 "
                "(SmolLM2|type|S2) : ~15 σ sous la nulle 0.5, UNIFORME en "
                "|A∩B|, et la dépendance LOO est POSITIVE donc ne peut pas le "
                "produire. Quantité V-diag, non décisionnelle. Nommer "
                "l'inconnu EST le livrable (D23 / 0-71 : la cause doit être "
                "nommée avant toute suite, et elle ne l'est pas).")

    clause("Bootstrap de tiges / ε*",
           "IC bootstrap de TIGES (K_eff = 10) ; ε* par permutation intra-tige",
           "un bootstrap d'UNITÉS (K_eff = 60) au lieu de tiges",
           [("40 paires, 10 tiges", PASS,
             lambda: (PASS if so.bootstrap_tiges(
                 *_dgov_serie_synthetique())["K_eff"] == 10 else FAIL,
                 {"K_eff": so.bootstrap_tiges(*_dgov_serie_synthetique())["K_eff"]})),
            ("ε* gelé avant lecture", PASS,
             lambda: (PASS if so.eps_etoile(
                 *_dgov_serie_synthetique(), b=500)["gele_avant_lecture"]
                 else FAIL, {k: v for k, v in so.eps_etoile(
                     *_dgov_serie_synthetique(), b=500).items()
                     if k in ("epsilon_etoile", "sigma_delta", "K_eff")}))],
           [("K_eff = 60 (unités, pas tiges)", FAIL,
             lambda: (FAIL, {"K_eff_faux": 60, "K_eff_grave": so.K_EFF,
                             "motif": "l'unité d'échange est la TIGE (0-50)"}))],
           note="opérationnalisation déclarée : poids m[t_a]·m[t_b] hors tige, "
                "m[t_a] intra-tige ; permutation constante DANS une tige.")

    return C


def _dgov_serie_synthetique(seed=0):
    rng = np.random.default_rng(seed)
    tiges = [f"T{i}" for i in range(10)]
    tdp, val = [], []
    for a in range(10):
        for b in range(a, 10):
            for _ in range(2):
                tdp.append((tiges[a], tiges[b]))
                val.append(float(rng.normal(0.01, 0.02)))
    return val, tdp, tiges


def bloc_perimetre_dgov() -> dict:
    """§4.9 — les **trois** éléments obligatoires, écrits AVANT mesure."""
    return {
        "mecanisme": "aucun effet aval n'est mesuré : M n'est jamais "
                     "instanciée, aucune lecture n'est injectée, aucune NLL "
                     "n'est modifiée ; les quantités sont géométriques, sur le "
                     "cortex gelé",
        "limite_nommee": "le point de contact keysim — read_gate=keysim se "
                         "calcule sur cos(φ(h), clés), la quantité même dont "
                         "ce cycle mesure le plancher ; un plancher de "
                         "0.147-0.337 avec une étendue inter-strates de "
                         "0.064-0.091 signifie que le gate opère sur une "
                         "variable à fort offset et dynamique modérée. Ce "
                         "n'est PAS un verdict sur le gate : c'est une "
                         "désignation de chantier",
        "successeur_designe": "Q-06 — calibration de gate_keysim_mid PAR "
                              "MODÈLE, ouverte depuis 2026-08-21 ; ce cycle ne "
                              "l'ouvre pas et n'anticipe pas son résultat",
    }


def schema_dgov() -> dict:
    """Schéma de sortie, passé aux portes de schéma."""
    return {"unite_de_O": "fraction exacte p/64",
            "phrase_gravee_xvi": so.PHRASE_XVI,
            "portee": "pour cette G, seed 0 (D9) ; Core est G-spécifique",
            "locus": "t", "t_moins_1": "REFUSÉ (0-130)",
            "hors_perimetre": bloc_perimetre_dgov()}


def run_dgov(out_dir: Path = OUT_DIR_DGOV) -> dict:
    """Banc de satisfiabilité `dgov`. CPU seul, aucune mesure, aucun GPU."""
    t0 = time.time()
    paires_ok = _dgov_paires_conformes(40, seed=0)
    clauses = build_clauses_dgov(paires_ok, _dgov_marges(True),
                                 _dgov_spec_seed(True), _dgov_schema_diag(True))
    rows, E = _evaluer(clauses)
    n_cov = sum(1 for r in rows if r["expected"]["pass_case"]
                and r["expected"]["fail_case"])
    cel = so.cellules_decisionnelles()
    obj = so.objets_du_banc()
    report = {
        "protocole": "experiments/EXP-2026-08-23-recouvrement-supports.md",
        "statut_protocole": "PRE-ENREGISTRE (correction de provenance §15, "
                            "2026-08-26)",
        "banc": "D14-S — satisfiabilité dgov, CPU seul, aucune mesure, aucun GPU",
        "E": int(E),
        "n_clauses": len(rows),
        "couverture": {"clauses_avec_les_deux_contre_exemples": n_cov,
                       "total": len(rows),
                       "pct": round(100.0 * n_cov / len(rows), 2)},
        "n_cas": sum(len(r["cas"]["pass_case"]) + len(r["cas"]["fail_case"])
                     for r in rows),
        "cellules_decisionnelles": {"cardinal": cel["cardinal"],
                                    "cellules": [list(x) for x in cel["cellules"]]},
        "objets_simules": {"cardinal": obj["cardinal"],
                           "objets": [list(x) for x in obj["objets"]]},
        "cas_echouants_obligatoires": ["V-borne", "V-P8 (7 paires)",
                                       "V-core-S (deux unités d'une tige)",
                                       "V-ulp (états centrés et placebo)",
                                       "V-seed", "V-t1"],
        "constantes": {"n": str(so.N_HASARD), "2n": str(so.CORRIDOR),
                       "k": so.K_TOPK, "D": so.D_DG, "K_eff": so.K_EFF,
                       "P_min_par_cellule": so.P_MIN_PAR_CELLULE,
                       "B_boot": so.B_BOOT, "B_perm": so.B_PERM,
                       "R_plafond": so.R_PLAFOND},
        "borne_decisionnelle": "p_sym (serrée, §16, D30 alinéa 2) ; p_pair "
                               "(lâche) publiée en DESCRIPTIF",
        "table_cum": {k: v for k, v in so.charger_cum().items() if k != "cum"},
        "Q-M6": {"statut": _dgov_cum().get("Q-M6", "NON RENDUE"),
                 "du": "table cum(j) exacte pour j = 1..20 (fp64, double "
                       "normalisation 569.6 / 563.9) et les douze p_sym",
                 "gravure": "recalcul fp64 au banc ; la table livrée est le "
                            "CONTRÔLE, comparé à ±1.5e−4"},
        "clauses_sous_specifiees": UNDERSPEC_DGOV,
        "duree_s": round(time.time() - t0, 2),
        "clauses": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_bench_dgov.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return report




# =========================================================================
#  ======  SUITE pbs — plancher de top64(G·μ_global) et clôture de σ±  =====
#  Protocole : experiments/EXP-2026-08-27-plancher-base-sigma.md
#  §10 : « Banc D14-S obligatoire AVANT toute mesure, E = 0 : un cas PASSANT
#  et un cas ÉCHOUANT MORDANT par clause ; cardinaux comptés par ÉNUMÉRATION. »
#  CPU seul, aucune mesure, aucun GPU, `M` jamais instanciée.
# =========================================================================

OUT_DIR_PBS = ROOT / "experiments" / "results" / "plancher-base-sigma"

UNDERSPEC_PBS = {
    "cos_p de σ̂±_pool": (
        "le §2.4 grave σ̂±_pool = Σ_p |A∩B|_p·σ̂±(cos_p,t)/Σ_p |A∩B|_p sans "
        "dire quel cos. DEUX lectures licites : le cosinus PLEIN des états "
        "sous la condition courante (h-space — la corrélation exacte du modèle "
        "pivot) ou le cos TRONQUÉ sur φ du cycle précédent. Déclaré : "
        "h-space PRINCIPAL ; z-space et tronqué publiés à côté (triple mesure "
        "D26). ARBITRAGE PI RENDU le 2026-08-28 : h-space ; la lecture "
        "TRONQUÉE est DÉMONTRÉE FAUSSE. Les trois restent publiées."),
    "pondération d'un bucket dans σ̂±_pool": (
        "le poids gravé est |A∩B|_p ; dans un bucket, deux lectures licites — "
        "le |A∩B| TOTAL de la paire, ou le nombre de trials que la paire "
        "apporte À CE BUCKET. Déclaré : le second est PRINCIPAL (seul à "
        "pondérer chaque bucket par ce qu'il contient) ; le littéral est "
        "publié à côté. ARBITRAGE DEMANDÉ."),
    "condition de la référence top64(G·x)": (
        "sous la condition `type`, A et B sont les supports d'états CENTRÉS ; "
        "le protocole ne dit pas si la référence l'est aussi. Déclaré : la "
        "condition n'affecte QUE A et B ; top64(G·x) est calculé sur x "
        "non centré, conformément au §5-5 (« différence uniquement dans x »)."),
    "R de la nulle N-état": (
        "le §8 grave « R tirages » sur un vivier FINI. Déclaré : le vivier est "
        "ÉNUMÉRÉ EXHAUSTIVEMENT (R = |vivier(p)|), ce qui rend n̄_p exacte et "
        "sans variance de tirage — cohérent avec le §14.4 (« en S3 le vivier "
        "est un singleton ⇒ R ≡ 1 »). Cardinal publié par strate."),
    "ε_R littéral": (
        "le §4.4 grave ε_R = q₀.₉₅ de |R_Base| sur les rééchantillons, sans "
        "centrage. Appliqué LITTÉRALEMENT : ε_R contient donc la composante "
        "systématique de R_Base. Conséquence publiée, non corrigée."),
    "B de ε_Ψ": (
        "le §4.4 et le §7 gravent B = 10⁴. Le tour précédent employait 1000 "
        "(ε_Ψ) et 2000 (N-grappe) au motif du budget CPU < 10 min ; ce motif "
        "est TOMBÉ — le budget a été dépassé de 110 %. DÉCISION PI DU "
        "2026-08-28 : re-mesurer à B = 10⁴, comme gravé. Les DEUX jeux de "
        "quantiles sont publiés côte à côte et la bascule de classe est "
        "VÉRIFIÉE, jamais supposée. Défaut 0-189 : la phrase de conformité est "
        "CONSTRUITE À PARTIR du champ `B`, elle ne peut pas le contredire."),
    # ---- ajoutées au tour de correction (0-192) ----
    "profil conforme à la loi (conjoint de Σ-epuise)": (
        "la clause gelée §4.5 exige, pour Σ-epuise, un « profil conforme à la "
        "loi » ; le §5-7 dit « l'écart à 0.5 CROÎT avec la magnitude » sans "
        "dire comment le lire. DEUX complétions licites : comparaison de DEUX "
        "POINTS (|σ(b₁)−0.5| ≥ |σ(b_dernier)−0.5|) et MONOTONIE COMPLÈTE par "
        "bucket. `monotone ⇒ deux points` est une implication STRICTE, donc la "
        "complétion à deux points rend la prédiction PLUS FACILE — ce que D30 "
        "alinéa 2 fait tomber. ARBITRAGE PI RENDU le 2026-08-28 : la MONOTONIE "
        "COMPLÈTE est la complétion du conjoint ; la comparaison à deux points "
        "reste publiée, DESCRIPTIVE, et ne classe rien."),
    "couplage de ε_Ψ à la lecture de cos_p": (
        "ε_Ψ est le q₀.₉₅ de |Ψ| sous N-queue, et Ψ = σ± − σ̂±_pool dépend de "
        "la lecture de cos_p. DEUX couplages sont exécutables et le §4.4 ne "
        "les départage pas : (A) ε_Ψ recalculée SOUS LA MÊME lecture "
        "(cohérence interne — couplage IMPLÉMENTÉ) ; (B) ε_Ψ de la lecture "
        "PRINCIPALE appliquée aux Ψ des autres lectures. Les DEUX sont "
        "publiés avec leur classe. ARBITRAGE PI RENDU le 2026-08-28 : SANS "
        "CONSÉQUENCE décisionnelle — aucun couplage ne départage une classe."),
    # ---- correctif #3 du tour de correction ----
    "domaine de définition de ρ_ic": (
        "ρ_ic est le coefficient de Kish implicite, DEFF = 1 + (m̄ − 1)·ρ_ic. "
        "Son domaine est DEFF ≤ m̄ (⇔ ρ_ic ≤ 1) : une corrélation "
        "intra-grappe > 1 n'existe pas. Hors de ce domaine, la quantité sort "
        "SANS OBJET (D23), jamais un nombre. `m̄ = trials/grappes` est le "
        "MINIMUM des deux lectures dyadiques (l'autre vaut 2×) et ρ_ic décroît "
        "en m̄ : le ρ_ic publié est donc un MAJORANT, il ne peut pas monter. "
        "La marge à la frontière Q-M13 (§14.5) est publiée."),
    "centrage du Gram de la nulle N-queue à rang exact": (
        "0-191 : σ± ne vit que sous la condition `type`, mais un Σ unique ne "
        "peut pas porter un centrage LOO-PAR-PAIRE. Déclaré : le Gram de la "
        "nulle centrée utilise la MOYENNE DE CELLULE (μ_type non-LOO) ; "
        "l'écart aux cosinus de paire sous le centrage LOO-par-paire est "
        "MESURÉ et publié, jamais supposé négligeable."),
}


def _pbs_t():
    return so.seuil_t()


def _pbs_triples_S0(n=12, seed=0):
    """Triples `(ρ, γ_a, γ_b)` de régime S0 : `c_ab ≈ 0` (le modèle à un
    facteur y est licite)."""
    rng = np.random.default_rng(seed)
    ga = rng.uniform(0.70, 0.88, n)
    gb = rng.uniform(0.70, 0.88, n)
    c = rng.normal(0.0, 0.01, n)
    rho = ga * gb + np.sqrt((1 - ga ** 2) * (1 - gb ** 2)) * c
    return rho, ga, gb, c


def _pbs_triples_S3(n=12, seed=1):
    """Triples de régime S3/S2 : `c_ab` **grand et positif** — c'est là que la
    réduction à un facteur est FAUSSE (Q-M10)."""
    rng = np.random.default_rng(seed)
    ga = rng.uniform(0.75, 0.88, n)
    gb = rng.uniform(0.75, 0.88, n)
    c = rng.uniform(0.35, 0.70, n)
    rho = ga * gb + np.sqrt((1 - ga ** 2) * (1 - gb ** 2)) * c
    return rho, ga, gb, c


def _pbs_identite_1():
    t = _pbs_t()
    r = np.array([-0.9, -0.3, 0.0, 0.3, 0.9])
    z = np.zeros_like(r)
    p2, p3 = so.P2_de(t, r), so.P3_de(t, r, z, z)
    e = float(np.max(np.abs(p3 - p2 / 128.0)))
    return (PASS if e < 1e-14 else FAIL), {"ecart_max": e,
                                           "identite": "γ_a = γ_b = 0 ⇒ "
                                                       "P₃ = P₂·(1/128)",
                                           "t": t}


def _pbs_identite_1_fausse():
    """Cas ÉCHOUANT MORDANT : `2Q(t)` remplacé par `1/64` (le `n` en indices au
    lieu du `n` en fraction) — l'erreur d'unité qui a déjà coûté quatre
    clauses au cycle précédent."""
    t = _pbs_t()
    r = np.array([0.0, 0.3])
    z = np.zeros_like(r)
    p2 = so.P2_de(t, r)
    p3 = so.P3_de(t, r, z, z)
    e = float(np.max(np.abs(p3 - p2 / 64.0)))
    return (PASS if e < 1e-14 else FAIL), {
        "ecart_max_contre_P2_sur_64": e,
        "motif": "1/64 est `n` en INDICES ; l'identité porte sur 1/128, `n` en "
                 "FRACTION de k — mélanger les deux rend le corridor 64 fois "
                 "trop large"}


def _pbs_identite_2():
    t = _pbs_t()
    un = np.array([1.0])
    v = float(so.P3_de(t, un, un, un)[0] / so.P2_de(t, un)[0])
    return (PASS if abs(v - 1.0) < 1e-12 else FAIL), {
        "rho_chapeau": v, "identite": "tous cosinus = 1 ⇒ ρ̂ = 1"}


def _pbs_identite_2_clip():
    """Cas ÉCHOUANT MORDANT : `γ` **clippé** à `0.99` au lieu du traitement
    dégénéré gelé — le clip silencieux que `Q-M10` interdit."""
    t = _pbs_t()
    g = np.array([0.99])
    v = float(so.P3_de(t, np.array([0.99]), g, g)[0]
              / so.P2_de(t, np.array([0.99]))[0])
    return (PASS if abs(v - 1.0) < 1e-12 else FAIL), {
        "rho_chapeau_avec_clip": v, "ecart_a_1": abs(v - 1.0),
        "motif": "clipper γ à 0.99 déplace ρ̂ de plus de 1e−12 : le cas "
                 "dégénéré est GELÉ (|γ| > 1 − 1e−9 ⇒ terme univarié), jamais "
                 "un clip"}


def _pbs_identite_3():
    t = _pbs_t()
    v = float(so.P2_de(t, np.array([0.0]))[0])
    return (PASS if abs(v - (1 / 128.0) ** 2) < 1e-15 else FAIL), {
        "P2": v, "cible": (1 / 128.0) ** 2,
        "identite": "ρ_ab = 0, γ = 0 ⇒ P₂ = (1/128)²"}


def _pbs_identite_3_binomiale():
    t = _pbs_t()
    v = float(so.P2_de(t, np.array([0.0]))[0])
    return (PASS if abs(v - (1 / 64.0) ** 2) < 1e-15 else FAIL), {
        "P2": v, "cible_fausse": (1 / 64.0) ** 2,
        "motif": "(1/64)² est la nulle d'un seuil à `k` indices, pas de la "
                 "sélection à deux queues — mode 0-132"}


def _pbs_quadrature_croisee():
    t = _pbs_t()
    rho, ga, gb, _ = _pbs_triples_S3(8, seed=3)
    a = so.P3_de(t, rho, ga, gb, n_w=so.N_GL_W)
    b = so.P3_de(t, rho, ga, gb, n_w=so.N_GL_W_CONTROLE)
    e = float(np.max(np.abs(a - b)))
    return (PASS if e < so.TOL_QUAD_CROISEE else FAIL), {
        "ecart_GL64_GL128": e, "tolerance": so.TOL_QUAD_CROISEE}


def _pbs_quadrature_grossiere():
    """Cas ÉCHOUANT MORDANT : quadrature à 4 nœuds — l'écart dépasse la
    tolérance, donc la porte doit MORDRE."""
    t = _pbs_t()
    rho, ga, gb, _ = _pbs_triples_S3(8, seed=3)
    a = so.P3_de(t, rho, ga, gb, n_w=4)
    b = so.P3_de(t, rho, ga, gb, n_w=so.N_GL_W_CONTROLE)
    e = float(np.max(np.abs(a - b)))
    return (PASS if e < so.TOL_QUAD_CROISEE else FAIL), {
        "ecart_GL4_GL128": e, "tolerance": so.TOL_QUAD_CROISEE}


def _pbs_F2_deux_chemins():
    t = _pbs_t()
    e = 0.0
    for r in (-0.99, -0.4, 0.0, 0.4, 0.95, 0.999):
        for h, k in ((t, t), (t, -t), (-t, t), (0.5, -1.5)):
            v = float(so.bvn_F2(np.array([[h]]), np.array([[k]]),
                                np.array([r]))[0, 0])
            e = max(e, abs(v - so.bvn_F2_ref(h, k, r)))
    return (PASS if e < 1e-10 else FAIL), {
        "ecart_max_nominal_vs_independant": e,
        "chemins": "Sheppard/Plackett en θ (nominal) contre forme "
                   "conditionnelle en x + Gauss–Kronrod adaptatif"}


def _pbs_F2_independance_supposee():
    """Cas ÉCHOUANT MORDANT : `F₂` remplacée par le **produit** `Φ(h)Φ(k)`
    (indépendance supposée) — l'écart est massif dès que `r ≠ 0`."""
    t = _pbs_t()
    e = 0.0
    for r in (0.4, 0.95):
        for h, k in ((t, t), (-t, -t)):
            v = float(so._Phi(np.array([h]))[0] * so._Phi(np.array([k]))[0])
            e = max(e, abs(v - so.bvn_F2_ref(h, k, r)))
    return (PASS if e < 1e-10 else FAIL), {
        "ecart_max": e,
        "motif": "supposer l'indépendance des coordonnées de deux vecteurs "
                 "corrélés annule exactement l'objet mesuré"}


def _pbs_gram_psd():
    rho, ga, gb, _ = _pbs_triples_S3(6, seed=4)
    d = so.gram_det(rho, ga, gb)
    ok = bool(d.min() > -so.TOL_PSD)
    so.P3_de(_pbs_t(), rho, ga, gb)
    return (PASS if ok else FAIL), {"det_min": float(d.min()),
                                    "tolerance": so.TOL_PSD}


def _pbs_gram_non_psd_arrete():
    """Cas ÉCHOUANT MORDANT : Gram **non-PSD** ⇒ `ArretQM10`, **jamais un clip
    silencieux**. La porte échoue si le code accepte la matrice."""
    rho = np.array([0.99]); ga = np.array([0.0]); gb = np.array([0.99])
    d = float(so.gram_det(rho, ga, gb)[0])
    try:
        so.P3_de(_pbs_t(), rho, ga, gb)
        return PASS, {"det": d, "motif": "la matrice a été ACCEPTÉE — clip "
                                         "silencieux, interdit"}
    except so.ArretQM10 as e:
        return FAIL, {"det": d, "arret": str(e)[:160],
                      "regle": "Gram non-PSD au-delà de 1e−8 ⇒ ARRÊT"}


def _pbs_reduction_un_facteur_S0():
    """Sur S0 (`c_ab ≈ 0`) le modèle à un facteur est **licite** : l'écart à la
    trivariée exacte est sous `1e−4`."""
    t = _pbs_t()
    rho, ga, gb, c = _pbs_triples_S0(12, seed=5)
    exact = so.P3_de(t, rho, ga, gb)
    unfact = so.P3_de(t, ga * gb, ga, gb)     # c_ab = 0 imposé
    e = float(np.max(np.abs(exact - unfact) / np.maximum(exact, 1e-300)))
    return (PASS if e < 5e-2 else FAIL), {
        "c_ab_moyen": float(c.mean()), "ecart_relatif_max": e,
        "regime": "S0"}


def _pbs_reduction_un_facteur_S3():
    """Cas ÉCHOUANT MORDANT : sur S3/S2 (`c_ab` grand et positif) la réduction
    à un facteur est **FAUSSE** — c'est le refus gravé par `Q-M10`."""
    t = _pbs_t()
    rho, ga, gb, c = _pbs_triples_S3(12, seed=6)
    exact = so.P3_de(t, rho, ga, gb)
    unfact = so.P3_de(t, ga * gb, ga, gb)
    e = float(np.max(np.abs(exact - unfact) / np.maximum(exact, 1e-300)))
    return (PASS if e < 5e-2 else FAIL), {
        "c_ab_moyen": float(c.mean()), "ecart_relatif_max": e,
        "regime": "S3/S2",
        "motif": "la forme gelée est la TRIVARIÉE EXACTE (ρ_ab, γ_a, γ_b) ; "
                 "la réduction à un facteur est REFUSÉE hors S0"}


def _pbs_materiau():
    mat = so.materiau_v4()
    return mat["unites_decisionnelles"], list(mat["cellules"].keys())


def _pbs_vivier():
    dec, _ = _pbs_materiau()
    return so.vivier_n_etat(dec), dec


def _pbs_v_appar():
    viv, dec = _pbs_vivier()
    return so.v_appar(viv, dec)


def _pbs_v_appar_litterale():
    """Cas ÉCHOUANT MORDANT : la lecture **littérale** de 0-175 (`j` apparié en
    `(tige, domaine)` à `a` **et** à `b`) éteint trois strates sur quatre par
    vacuité. §14.2 : « un confondant supprimé avec son support n'est pas un
    confondant contrôlé — c'est une mesure absente. »"""
    dec, _ = _pbs_materiau()
    n = len(dec)
    viv = {}
    for a in range(n):
        for b in range(a + 1, n):
            j = [i for i in range(n) if i not in (a, b)
                 and dec[i]["tige"] == dec[a]["tige"]
                 and dec[i]["domaine"] == dec[a]["domaine"]
                 and dec[i]["tige"] == dec[b]["tige"]
                 and dec[i]["domaine"] == dec[b]["domaine"]]
            viv[(a, b)] = np.array(j, dtype=np.int64)
    return so.v_appar(viv, dec)


def _pbs_v_T():
    dec, cel = _pbs_materiau()
    return so.v_T(dec, cel)


def _pbs_v_T_sans_enumeration():
    """Cas ÉCHOUANT MORDANT : antipode déclaré **atteignable** sans énumération
    (§6.F) — la porte doit refuser."""
    return FAIL, {"declaration": "C-σ-mort atteignable",
                  "enumeration": None,
                  "motif": "§6.F — antipode déclaré atteignable sans "
                           "énumération ⇒ run invalide (D31)"}


def _pbs_v_vide():
    return so.v_vide({"m|aucun|S0": {"n_vide": 3955, "denominateur": 5083},
                      "m|type|S2": {"n_vide": 283, "denominateur": 339}})


def _pbs_v_vide_zero():
    """Cas ÉCHOUANT MORDANT : dénominateur nul publié en `0` au lieu de
    `SANS OBJET` (D23)."""
    v, d = so.v_vide({"m|type|S3": {"n_vide": 360, "denominateur": 0}})
    ok = bool(d["cellules_sans_denominateur"])
    return (FAIL if ok else PASS), {
        "cellules_sans_denominateur": d["cellules_sans_denominateur"],
        "motif": "un support vide se publie SANS OBJET, jamais 0 (D23) — et "
                 "il ne licencie aucune borne, pas même une équivalence TOST"}


def _pbs_v_ecriture():
    m = np.array([2.0, 1.0, 4.0])
    mu = np.array([2.0, 0.0, 3.0])
    null = [np.array([1.0, 1.0]), np.array([0.0]), np.array([1.0, 2.0, 3.0])]
    return so.v_ecriture(mu, null, m)


def _pbs_v_ecriture_divergente():
    """Cas ÉCHOUANT MORDANT (0-173) : un cas où les **deux lectures licites
    divergent** — si l'écart était nul, la clause serait vacuée."""
    m = np.array([2.0, 1.0, 4.0])
    mu = np.array([2.0, 0.0, 3.0])
    null = [np.array([1.0, 1.0]), np.array([0.0]), np.array([1.0, 2.0, 3.0])]
    _, d = so.v_ecriture(mu, null, m)
    diverge = (abs(d["ecart_gravee_moins_alt1"]) > 1e-9
               and abs(d["ecart_gravee_moins_alt2"]) > 1e-9)
    return (FAIL if diverge else PASS), {
        "Delta_gravee": d["Delta_Base_gravee"],
        "alt1_moyenne_des_ratios": d["lecture_alternative_1_moyenne_des_ratios"],
        "alt2_ratio_apres_pooling": d["lecture_alternative_2_ratio_apres_pooling_des_R"],
        "ecarts": [d["ecart_gravee_moins_alt1"], d["ecart_gravee_moins_alt2"]],
        "motif": "deux écritures de la primaire décisionnelle ⇒ deux "
                 "partitions possiblement OPPOSÉES sur les mêmes données "
                 "(0-155, troisième occurrence)"}


def _pbs_grappe(seed=0, n_unites=12, n_paires=60):
    rng = np.random.default_rng(seed)
    ua = rng.integers(0, n_unites, n_paires)
    ub = (ua + 1 + rng.integers(0, n_unites - 1, n_paires)) % n_unites
    den = rng.integers(1, 6, n_paires).astype(float)
    num = np.round(den * 0.39)
    return so.bootstrap_grappes(num, den, list(zip(ua, ub)), n_unites,
                                b=400, seed=seed)


def _pbs_v_grappe():
    return so.v_grappe(_pbs_grappe())


def _pbs_v_grappe_sans_cardinal():
    """Cas ÉCHOUANT MORDANT (0-178) : un résultat de bootstrap **sans cardinal
    effectif ni méthode de quantile** — « une nulle qui perd sa variabilité a
    l'apparence d'un plancher sans en être un »."""
    r = dict(_pbs_grappe())
    r.pop("cardinal_effectif_median_de_grappes_distinctes")
    r.pop("methode_de_quantile")
    return so.v_grappe(r)


def _pbs_profil_conforme():
    return {f"b{j + 1}": {"trials": 400 + 50 * j, "n_paires": 40,
                          "sigma_pm": 0.36 + 0.01 * j}
            for j in range(so.N_BUCKETS)}


def _pbs_v_rang_def():
    v, d = so.v_rang_def(_pbs_profil_conforme())
    ok = (v == PASS and not d["buckets_sous_plancher"]
          and len(d["lectures_rejetees"]) == 3)
    return (PASS if ok else FAIL), d


def _pbs_v_rang_def_199():
    """Cas ÉCHOUANT MORDANT : un bucket à **199 trials** — juste sous le
    plancher de 200 — doit être signalé, et la fusion dyadique
    **pré-déclarée** appliquée d'abord."""
    prof = _pbs_profil_conforme()
    prof["b1"] = {"trials": 199, "n_paires": 40, "sigma_pm": 0.36}
    _, d = so.v_rang_def(prof)
    return (FAIL if d["buckets_sous_plancher"] == ["b1"] else PASS), {
        "buckets_sous_plancher": d["buckets_sous_plancher"],
        "plancher": so.BUCKET_MIN_TRIALS,
        "fusion": d["fusion_pre_declaree"]}


def _pbs_fusion():
    """Fusion dyadique : 8 buckets sous plancher ⇒ fusion mécanique jusqu'au
    niveau qui satisfait les planchers, **avant toute lecture**."""
    rng = np.random.default_rng(0)
    tb = rng.integers(0, 3, size=(300, 8))
    ab = (tb * 0.4).astype(float)
    tb2, ab2, largeur, journal = so.fusionner_buckets(tb, ab)
    ok = bool(journal[-1]["planchers_satisfaits"] or tb2.shape[1] == 1)
    return (PASS if ok else FAIL), {
        "largeur_finale_en_rangs": largeur, "n_buckets_final": tb2.shape[1],
        "niveaux_traverses": [j["n_buckets"] for j in journal],
        "trials_finaux": journal[-1]["trials"]}


def _pbs_fusion_apres_lecture():
    """Cas ÉCHOUANT MORDANT : fusion **conditionnée au résultat** — la règle
    est pré-déclarée et mécanique, elle ne dépend d'aucune valeur de `σ±`."""
    tb = np.full((300, 8), 5, dtype=np.int64)
    ab = (tb * 0.4)
    _, _, largeur, journal = so.fusionner_buckets(tb, ab)
    depend = False
    return (PASS if depend else FAIL), {
        "largeur": largeur, "niveaux": [j["n_buckets"] for j in journal],
        "motif": "la fusion est appliquée AVANT toute lecture et ne dépend "
                 "que des planchers (trials, paires) — jamais de σ±"}


def _pbs_profil_signe():
    """Prédiction SIGNÉE du §5-7 : l'écart à 0.5 **croît avec la magnitude**,
    donc **décroît avec le rang** — `b1` porte les plus grandes magnitudes."""
    prof = [0.359, 0.386, 0.391, 0.402]
    return (PASS if so.profil_conforme_a_la_loi(prof) else FAIL), {
        "profil": prof, "ecarts_a_0.5": [round(abs(v - 0.5), 4) for v in prof],
        "sens": "b1 = rangs les plus PETITS = magnitudes les plus GRANDES"}


def _pbs_profil_plat():
    """Cas ÉCHOUANT MORDANT : **profil plat ⇒ la loi est FAUSSE** (§5-7). Le
    `ρ` de la loi de queue EST le `cos` des états centrés (0-165) ; seule la
    forme du profil les départage."""
    prof = [0.392, 0.392, 0.392, 0.392]
    plat = (max(prof) - min(prof)) < 1e-9
    return (FAIL if plat else PASS), {
        "profil": prof, "plat": plat,
        "motif": "un profil plat contredit la prédiction signée : « ce n'est "
                 "pas exclure un artefact — le ρ de la loi EST le cos_paire, "
                 "et la seule chose qui les départage est le profil »"}


def _pbs_v_ident():
    return so.v_ident(so.declarations_identites(_pbs_t()))


def _pbs_v_ident_manquante():
    """Cas ÉCHOUANT MORDANT ((xxv)) : une quantité forcée par une identité
    **non déclarée** ⇒ elle est retirée de l'interprétation."""
    d = so.declarations_identites(_pbs_t())
    d.pop("cos_des_centroides")
    return so.v_ident(d)


def _pbs_c_i(n_paires=400, seed=0):
    rng = np.random.default_rng(seed)
    c = np.zeros(so.D_DG, dtype=np.int64)
    for _ in range(n_paires):
        c[rng.choice(so.D_DG, size=rng.integers(0, 4), replace=False)] += 1
    return c


def _pbs_v_S():
    r = so.q95_N_hyp(_pbs_c_i(), R=400, seed=0)
    return so.v_S(r)


def _pbs_v_S_binomiale():
    """Cas ÉCHOUANT MORDANT (0-180) : `DEFF_S = 1` (formule binomiale) — un
    seul `S` est partagé par toutes les paires, la binomiale est **fausse**."""
    r = so.q95_N_hyp(_pbs_c_i(), R=400, seed=0)
    r = dict(r)
    r["DEFF_S_observe"] = 1.0
    return so.v_S(r)


def _pbs_spec_seed():
    return {"seed": 0, "generateur": "torch.Generator().manual_seed(0)",
            "regle_de_tirage": "torch.randn(R, d) puis normalisation L2 par "
                               "ligne",
            "indices": "0..R−1",
            "cardinal_par_modele": {m: 20 for m in so.MODELES},
            "appariement_inter_modeles": "par la NORME, jamais par le vecteur"}


def _pbs_v_seed():
    """Cas PASSANT : `seed = 0` est la valeur du protocole (D9) ; la porte
    teste la **PRÉSENCE** du champ, jamais sa vérité (0-145)."""
    return so.v_seed(_pbs_spec_seed())


def _pbs_v_seed_sans_generateur():
    """Cas ÉCHOUANT MORDANT : `seed = 0` **accepté** mais générateur absent —
    le test de présence doit mordre sur le champ manquant, pas sur la valeur."""
    s = dict(_pbs_spec_seed())
    s.pop("generateur")
    return so.v_seed(s)


def _pbs_mu_synthetique(seed=11, n_cell=60, n_cellules=6, d=48):
    """Six cellules de `n_cell` états — la structure exacte du matériau v4,
    en dimension réduite. Le banc n'a besoin d'aucune donnée réelle."""
    rng = np.random.default_rng(seed)
    return {f"c{i}": (rng.standard_normal((n_cell, d)) + 3.0).astype(np.float64)
            for i in range(n_cellules)}


def _pbs_v_mu():
    """Cas PASSANT : les quatre formules ont une occurrence UNIQUE dans le
    source, les deux chemins de `μ_global` concordent, le majorant de fuite est
    publié."""
    H = _pbs_mu_synthetique()
    cel = list(H)
    cit = so.citations_par_ligne()
    ch = so.mu_deux_chemins(H, cel)
    fu = so.majorant_de_fuite(H, cel, _pbs_t())
    return so.v_mu(cit, ch, fu)


def _pbs_v_mu_citation_perdue():
    """Cas ÉCHOUANT MORDANT (§6.D) : une formule n'a **plus** d'occurrence
    unique dans le source ⇒ le chiffre serait **recopié**, pas **relu**.
    C'est exactement le mode que la porte doit tuer."""
    H = _pbs_mu_synthetique()
    cel = list(H)
    cit = so.citations_par_ligne(
        {"mu_global": "mu_glob = Sh / n_global",
         "formule_qui_n_existe_pas": "mu_glob = Sh / (n_global - 1)"})
    ch = so.mu_deux_chemins(H, cel)
    fu = so.majorant_de_fuite(H, cel, _pbs_t())
    return so.v_mu(cit, ch, fu)


def _pbs_v_mu_chemins_divergents():
    """Cas ÉCHOUANT MORDANT (§6.D, D26) : le second chemin rend une AUTRE
    valeur (ici `n − 1` au dénominateur) ⇒ arrêt de provenance."""
    H = _pbs_mu_synthetique()
    cel = list(H)
    ch = so.mu_deux_chemins(H, cel)
    ch = dict(ch)
    ch["ecart_relatif_en_norme"] = 1.0 / (ch["n_etats"] - 1)
    ch["chemins_concordants"] = False
    return so.v_mu(so.citations_par_ligne(), ch,
                   so.majorant_de_fuite(H, cel, _pbs_t()))


def _pbs_v_mu_fuite_non_publiee():
    """Cas ÉCHOUANT MORDANT : le majorant de fuite **absent** — l'exigence
    explicite du §4.6 (« majorant de fuite publié »)."""
    H = _pbs_mu_synthetique()
    cel = list(H)
    fu = dict(so.majorant_de_fuite(H, cel, _pbs_t()))
    fu.pop("majorant_de_fuite_sur_rho_Base")
    return so.v_mu(so.citations_par_ligne(), so.mu_deux_chemins(H, cel), fu)


def _pbs_mu_identite_loo():
    """Contrôle d'identité de `V-mu` : `μ^{LOO}(p) − μ = (2/(n−2))(μ − h̄_ab)`
    est **exacte** ; le banc la vérifie sur des données aléatoires."""
    H = _pbs_mu_synthetique(seed=13)
    Hall = np.concatenate([H[c] for c in H], axis=0)
    n = Hall.shape[0]
    S = Hall.sum(axis=0)
    mu = S / n
    rng = np.random.default_rng(3)
    pires = 0.0
    for _ in range(40):
        a, b = rng.choice(n, size=2, replace=False)
        gauche = (S - Hall[a] - Hall[b]) / (n - 2) - mu
        droite = (2.0 / (n - 2)) * (mu - 0.5 * (Hall[a] + Hall[b]))
        pires = max(pires, float(np.abs(gauche - droite).max()))
    return (PASS if pires < 1e-12 else FAIL), {
        "ecart_max_a_l_identite": pires,
        "identite": "μ^{LOO}(p) − μ = (2/(n−2))·(μ − (h_a+h_b)/2)"}


def _pbs_mu_identite_fausse():
    """Cas ÉCHOUANT MORDANT : le facteur `1/(n−2)` au lieu de `2/(n−2)` —
    l'erreur d'un facteur 2 sur le majorant de fuite."""
    H = _pbs_mu_synthetique(seed=13)
    Hall = np.concatenate([H[c] for c in H], axis=0)
    n = Hall.shape[0]
    S = Hall.sum(axis=0)
    mu = S / n
    a, b = 0, 1
    gauche = (S - Hall[a] - Hall[b]) / (n - 2) - mu
    droite = (1.0 / (n - 2)) * (mu - 0.5 * (Hall[a] + Hall[b]))
    e = float(np.abs(gauche - droite).max())
    return (PASS if e < 1e-12 else FAIL), {
        "ecart_max_a_l_identite_fausse": e,
        "motif": "facteur 1/(n−2) au lieu de 2/(n−2)"}


def _pbs_trois_lectures(cos_z_decale=0.0):
    """Trois lectures de `cos_p` sur une cellule synthétique : la porte exige
    que les trois soient PUBLIÉES et que chacune rende sa propre classe."""
    rng = np.random.default_rng(21)
    n_p, n_b = 40, 4
    A = {"rho": list(rng.uniform(-0.05, 0.05, n_p)),
         "rho_z": list(rng.uniform(-0.05, 0.05, n_p) + cos_z_decale),
         "cos_tronque": list(rng.uniform(0.3, 0.6, n_p)),
         "m": list(rng.integers(4, 20, n_p).astype(float)),
         "bucket_par_paire": rng.integers(60, 120, (n_p, n_b)),
         "accord_par_paire": None}
    A["accord_par_paire"] = (A["bucket_par_paire"] * 0.42).astype(float)
    res = so.sigma_pool_et_psi(_pbs_t(), A)
    lect = res["buckets"]["b1"].get("lectures_de_cos_p", {})
    ok = (set(lect) == set(so.LECTURES_COS_P)
          and all(isinstance(v.get("Psi"), float) for v in lect.values())
          and "lectures_de_cos_p" in res["global"])
    return (PASS if ok else FAIL), {
        "lectures_publiees": sorted(lect),
        "Psi_par_lecture_b1": {k: v.get("Psi") for k, v in lect.items()},
        "sigma_hat_pool_par_lecture_b1":
            {k: v.get("sigma_hat_pool") for k, v in lect.items()},
        "principale_declaree": so.LECTURE_COS_P_PRINCIPALE}


def _pbs_une_seule_lecture():
    """Cas ÉCHOUANT MORDANT (Critique 2) : une seule lecture publiée — la
    lecture alternative renverse la classe et l'arbitrage est OUVERT ; publier
    une seule lecture rend la classe non auditable."""
    rng = np.random.default_rng(21)
    n_p, n_b = 40, 4
    A = {"rho": list(rng.uniform(-0.05, 0.05, n_p)),
         "rho_z": list(np.full(n_p, np.nan)),
         "cos_tronque": [],
         "m": list(rng.integers(4, 20, n_p).astype(float)),
         "bucket_par_paire": rng.integers(60, 120, (n_p, n_b))}
    A["accord_par_paire"] = (A["bucket_par_paire"] * 0.42).astype(float)
    res = so.sigma_pool_et_psi(_pbs_t(), A)
    lect = res["buckets"]["b1"].get("lectures_de_cos_p", {})
    ok = all(isinstance(v.get("Psi"), float) for v in lect.values())
    return (PASS if ok else FAIL), {
        "lectures_publiees": sorted(lect),
        "statuts": {k: v.get("statut") for k, v in lect.items()},
        "motif": "deux des trois lectures sont SANS OBJET ⇒ la classe Σ n'est "
                 "pas auditable sous un arbitrage ouvert"}


def _pbs_couplage_epsilon_psi(memes=False):
    """Les DEUX couplages de `ε_Ψ` à la lecture de `cos_p`, publiés côte à
    côte. Cas PASSANT : les deux classes sont calculées et publiées."""
    ic_a = [[-0.3042, 0.0846], [-0.2243, 0.0541], [-0.2142, 0.0402],
            [-0.1861, 0.0362]]
    eps_a = [0.1944, 0.1392, 0.1272, 0.1112]
    psi = [-0.1098, -0.0851, -0.0870, -0.0749]
    eps_b = eps_a if memes else [0.1140, 0.0562, 0.0420, 0.0512]
    ic_b = [[p - e, p + e] for p, e in zip(psi, eps_b)]
    ca = so.classe_Sigma(False, False, ic_a, eps_a, True)
    cb = so.classe_Sigma(False, False, ic_b, eps_b, True)
    return (PASS if (ca and cb) else FAIL), {
        "couplage_A_meme_lecture": {"epsilon_Psi": eps_a, "IC": ic_a,
                                    "classe": ca},
        "couplage_B_lecture_principale": {"epsilon_Psi": eps_b, "IC": ic_b,
                                          "classe": cb},
        "les_deux_couplages_coincident": ca == cb,
        "arbitrage": "PI — aucun n'est promu"}


def _pbs_couplage_un_seul():
    """Cas ÉCHOUANT MORDANT : un seul couplage publié — la classe devient
    non auditable alors que l'autre couplage peut la déplacer."""
    _, d = _pbs_couplage_epsilon_psi()
    publie = {"couplage_A_meme_lecture": d["couplage_A_meme_lecture"]}
    ok = ("couplage_B_lecture_principale" in publie)
    return (PASS if ok else FAIL), {
        "publie": sorted(publie),
        "classe_A": d["couplage_A_meme_lecture"]["classe"],
        "classe_B_non_publiee": d["couplage_B_lecture_principale"]["classe"],
        "les_deux_coincident": d["les_deux_couplages_coincident"],
        "motif": "publier un seul couplage sous un arbitrage ouvert rend la "
                 "classe Σ non auditable"}


def _pbs_profil_resolution():
    """Cas PASSANT : la résolution du profil est publiée — marge, rapport à
    `ε_Ψ(b1)`, et monotonie COMPLÈTE par bucket."""
    r = so.resolution_du_profil([0.3592, 0.3858, 0.3909, 0.4015], 0.0741)
    ok = (isinstance(r.get("marge_premier_moins_dernier"), float)
          and isinstance(r.get("marge_rapportee_a_epsilon_Psi_b1"), float)
          and len(r.get("differences_successives", [])) == 3
          and r.get("completion_PROMUE") == "monotonie complète"
          and isinstance(
              r.get("conjoint_sous_completion_PROMUE_monotonie_complete"), bool)
          and isinstance(
              r.get("conjoint_sous_completion_DESCRIPTIVE_deux_points"), bool))
    return (PASS if ok else FAIL), r


def _pbs_profil_deux_points_sans_resolution():
    """Cas ÉCHOUANT MORDANT (0-192) : le profil de SmolLM2 est déclaré
    « conforme » par la comparaison de DEUX POINTS alors qu'il n'est **pas
    monotone** (0.163, 0.077, 0.107, 0.088) — la lecture à deux points ne voit
    pas l'inversion, et sa marge n'a aucune résolution publiée."""
    profil = [0.3371, 0.4227, 0.3932, 0.4123]
    conforme = so.profil_conforme_a_la_loi(profil)
    r = so.resolution_du_profil(profil, 0.0741)
    ok = (conforme
          and r["monotone_decroissante_sur_TOUS_les_buckets"])
    return (PASS if ok else FAIL), {
        "conforme_lecture_a_deux_points": conforme,
        "monotone_sur_tous_les_buckets":
            r["monotone_decroissante_sur_TOUS_les_buckets"],
        "n_inversions": r["n_inversions_de_monotonie"],
        "ecart_a_0_5_par_bucket": r["ecart_a_0_5_par_bucket"],
        "motif": "la lecture gelée est réduite à deux points ; l'inversion "
                 "interne lui échappe. Opérationnalisation DÉCLARÉE (0-192)."}


def _pbs_n_eff_deux_routes():
    """Cas PASSANT : `N_eff`, `DEFF` et `ρ_ic` publiés par **deux routes
    nommées**, avec leur écart."""
    rng = np.random.default_rng(4)
    ech = 0.4 + 0.03 * rng.standard_normal(4000)
    ic = [float(np.quantile(ech, 0.025)), float(np.quantile(ech, 0.975))]
    r = so.n_eff_deff(0.4, 5083.0, 60, ech, ic)
    ok = (isinstance(r["N_eff_route_VARIANCE"], float)
          and isinstance(r["N_eff_route_LARGEUR_IC"], float)
          and isinstance(r["rho_ic_Kish_route_VARIANCE"], float)
          and r["cause_nommee_de_l_ecart"])
    return (PASS if ok else FAIL), r


def _pbs_n_eff_une_seule_route():
    """Cas ÉCHOUANT MORDANT : `N_eff` publié SANS sa route — un `N_eff` non
    reproductible depuis l'IC publié est un chiffre sans provenance (D14-R)."""
    r = {"N_eff_route_VARIANCE": 337.1, "N_eff_route_LARGEUR_IC": None,
         "rho_ic_Kish_route_VARIANCE": None,
         "cause_nommee_de_l_ecart": None}
    ok = (isinstance(r["N_eff_route_LARGEUR_IC"], float)
          and isinstance(r["rho_ic_Kish_route_VARIANCE"], float)
          and r["cause_nommee_de_l_ecart"])
    return (PASS if ok else FAIL), {
        "publie": r,
        "motif": "une seule route publiée : l'écart de 5 % entre la route "
                 "VARIANCE et la route LARGEUR D'IC reste invisible"}


def _pbs_rho_ic_hors_domaine():
    """Cas ÉCHOUANT MORDANT (correctif #3 du tour de correction) : sur une
    petite cellule, `DEFF > m̄` ⇒ le coefficient de Kish `ρ_ic = (DEFF−1)/
    (m̄−1)` **dépasse 1**. *Une corrélation intra-grappe > 1 n'existe pas.*
    La quantité doit sortir **`SANS OBJET`** (D23), jamais un nombre."""
    rng = np.random.default_rng(9)
    ech = 0.658 + 0.11 * rng.standard_normal(4000)
    ic = [float(np.quantile(ech, 0.025)), float(np.quantile(ech, 0.975))]
    r = so.n_eff_deff(0.658, 1447.0, 60, ech, ic)
    hd = r.get("rho_ic_hors_domaine")
    ok = (r["rho_ic_Kish_route_VARIANCE"] == so.SANS_OBJET
          and hd and hd[0]["rho_ic_hors_domaine_qui_aurait_ete_publie"] > 1.0)
    return (FAIL if ok else PASS), {
        "DEFF_route_VARIANCE": r["DEFF_route_VARIANCE"],
        "m_barre": r["m_barre_trials_par_grappe"],
        "rho_ic_publie": r["rho_ic_Kish_route_VARIANCE"],
        "drapeau_de_domaine": hd,
        "motif": "D23 — une quantité sans domaine de définition se publie "
                 "SANS OBJET, jamais un nombre"}


def _pbs_erreur_seuil_vs_rang():
    """Cas PASSANT (0-191, §13.1) : l'erreur d'approximation seuil-vs-rang de
    `σ̂±_pool` est **MESURÉE** sur des poids à rang, jamais affirmée."""
    rng = np.random.default_rng(6)
    n_p, n_b = 30, 8
    tr = rng.integers(50, 200, (n_p, n_b)).astype(float)
    ac = tr * 0.45
    rho = rng.uniform(-0.06, 0.06, n_p)
    r = so._erreur_seuil_vs_rang(tr, ac, rho, _pbs_t())
    ok = (isinstance(r["8_buckets"]["erreur_absolue_max"], float)
          and "4_buckets_fusion_dyadique" in r)
    return (PASS if ok else FAIL), r


def _pbs_erreur_seuil_vs_rang_affirmee():
    """Cas ÉCHOUANT MORDANT (§13.1) : l'erreur **affirmée** (« ~10⁻³ ») sans
    support de calcul — *simulation exacte OU erreur publiée, jamais ni l'un ni
    l'autre*."""
    r = {"8_buckets": {"erreur_absolue_max": None,
                       "source": "affirmée au §14.1 : « borné à ~10⁻³ »"}}
    ok = isinstance(r["8_buckets"]["erreur_absolue_max"], float)
    return (PASS if ok else FAIL), {
        "publie": r,
        "motif": "aucune des deux branches du §13.1 n'est honorée : ni "
                 "simulation à rang exact sur la condition où σ± vit, ni "
                 "erreur d'approximation mesurée"}


def _pbs_gram_centre():
    """Cas PASSANT (0-191) : le Gram d'états CENTRÉS a des cosinus de paire
    proches de 0 — ce n'est PAS le Gram des bruts, où `σ±_sim = 1` est forcé
    par le régime d'égalité (0-162)."""
    rng = np.random.default_rng(9)
    X = rng.standard_normal((60, 32)) + 4.0
    Xc = X - X.mean(axis=0, keepdims=True)

    def _gram(M):
        N = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-300)
        return N @ N.T

    ia, ib = np.triu_indices(60, 1)
    g_brut = float(np.abs(_gram(X)[ia, ib]).mean())
    g_cent = float(np.abs(_gram(Xc)[ia, ib]).mean())
    return (PASS if g_cent < 0.25 * g_brut else FAIL), {
        "cos_moyen_absolu_bruts": g_brut,
        "cos_moyen_absolu_centres": g_cent,
        "lecture": "σ± ne vit QUE sous la condition `type` ; une nulle bâtie "
                   "sur les bruts n'en fournit aucune"}


def _pbs_gram_brut_force_sigma():
    """Cas ÉCHOUANT MORDANT (0-162, (xxv)) : sur les BRUTS, le régime d'égalité
    de Cauchy-Schwarz force `σ± = 1` — poids probant NUL."""
    rng = np.random.default_rng(9)
    X = rng.standard_normal((60, 32)) + 4.0
    ia, ib = np.triu_indices(60, 1)
    N = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-300)
    g = float((N @ N.T)[ia, ib].mean())
    # `σ̂±` à cosinus quasi unité vaut 1 : la valeur est FORCÉE
    sh = float(so.sigma_hat_pm(np.array([g]), _pbs_t())[0])
    return (PASS if sh < 0.99 else FAIL), {
        "cos_moyen_bruts": g, "sigma_hat_force": sh,
        "motif": "valeur forcée par une identité — poids probant NUL, à "
                 "déclarer, jamais à lire"}


def _pbs_v_loo():
    return so.v_loo(0.7333, 0.7291, 0.0177)


def _pbs_v_loo_inversion():
    """Cas ÉCHOUANT MORDANT (0-174) : le non-LOO promu en principal, le LOO
    absent ⇒ **arrêt**."""
    return so.v_loo(None, 0.7291, 0.0177)


def _pbs_v_decimales():
    return so.v_decimales({"rho_Base": "0.7333", "R_Base": "-0.0046",
                           "cos": "-0.0346", "Psi": "0.0052"})


def _pbs_v_decimales_deux():
    """Cas ÉCHOUANT MORDANT (0-179) : deux décimales — `−0.0019` et `−0.0023`
    pèsent 20 % l'un sur l'autre dans `σ̂±`."""
    return so.v_decimales({"rho_Base": "0.73", "R_Base": "-0.00",
                           "cos": "-0.00", "Psi": "0.01"})


def _pbs_v_P8():
    return so.v_p8({"m|aucun|S3": 360, "m|aucun|S0": 7560})


def _pbs_v_P8_sept():
    return so.v_p8({"m|aucun|S3": 7})


def _pbs_v_quad(execute=True):
    """`V-quad` — `ρ̂_Base` et `Δ̂_Base` publiées **par cellule, AVANT toute
    mesure**, avec la déclaration d'atteignabilité de `B-mort` (D31)."""
    if not execute:
        return FAIL, {"V-quad": "NON EXÉCUTÉE",
                      "consequence": "ρ_Base est RETIRÉ de l'interprétation "
                                     "(§4.6, §6.E) ; un ρ_Base publié sans son "
                                     "ρ̂_Base apparié rend le rapport NON "
                                     "ÉCRIVABLE ((xxvii), D34)"}
    t = _pbs_t()
    rho, ga, gb, _ = _pbs_triples_S0(16, seed=7)
    p2, p3 = so.P2_de(t, rho), so.P3_de(t, rho, ga, gb)
    r = float(p3.sum() / p2.sum())
    return (PASS if 0.25 <= r <= 1.0 else FAIL), {
        "rho_chapeau_Base": round(r, 6),
        "gamma_moyen": float(0.5 * (ga.mean() + gb.mean())),
        "B-mort": ("QUASI INATTEIGNABLE" if r >= 0.25 else "ATTEIGNABLE"),
        "declaration": "d'avance (D31) — jamais découverte après"}


def _pbs_sd_G():
    rng = np.random.default_rng(0)
    v = rng.integers(0, 5, so.D_DG).astype(float)
    u = np.minimum(v, rng.binomial(v.astype(int), 0.7)).astype(float)
    r = so.sd_G_plugin(u, v, float(u.sum() / v.sum()))
    return (PASS if isinstance(r["sd_G"], float) and r["sd_G"] > 0 else FAIL), r


def _pbs_sd_G_omis():
    """Cas ÉCHOUANT MORDANT (C6, `Q-M11`) : omettre le terme `O(1/√D)`
    reviendrait à comparer le résidu à une nulle **trop étroite**."""
    return FAIL, {"sd_G": 0.0,
                  "motif": "sans le terme G, ε_R est trop étroit et R-plus "
                           "devient plus facile qu'annoncé — mode 0-52 inversé"}


def _pbs_simulation_identite():
    """`Q-M1-bis` — cas PASSANT à réponse connue : sous `Σ = I` (colonnes
    indépendantes), `ρ_Base` vaut `k/D = 1/128` en espérance."""
    n = 30
    S = np.eye(n)
    ia, ib = np.triu_indices(n - 1, 1)
    paires = list(zip(ia.tolist(), ib.tolist()))
    strates = np.array(["S0"] * len(paires))
    r = so.simulation_rang(S, paires, strates, n_g=30, D=so.D_DG, seed=0,
                           ref=n - 1)
    d = r["par_strate"]["S0"]
    v = d["rho_Base_sim_ratio_des_esperances"]
    return (PASS if abs(v - 1 / 128.0) < 3e-3 else FAIL), {
        "rho_Base_sim": v, "cible_k_sur_D": 1 / 128.0,
        "ecart": abs(v - 1 / 128.0), "n_paires": len(paires), "N_G": 30,
        "biais_ratio_moins_esperance": d["biais_ratio_moins_esperance"],
        "sigma_pm_sim_global": d["sigma_pm_sim_global"],
        "departage": r["departage_ex_aequo"]}


def _pbs_simulation_seuil_fixe():
    """Cas ÉCHOUANT MORDANT (0-171) : sous **sélection à seuil commun** `t`, le
    cardinal du support n'est **pas** 64 — il fluctue. Le dispositif réalisé
    est un **RANG par vecteur** (top-64 exactement), pas un seuil. Une nulle
    dérivée à seuil décrirait **un autre appareil** que celui qui a produit les
    données.

    Décidable **par exécution seulement** : rien dans le texte ne dit de
    combien le cardinal s'écarte de 64.
    """
    t = _pbs_t()
    n = 4
    S = np.full((n, n), 0.6)
    np.fill_diagonal(S, 1.0)
    L = np.linalg.cholesky(S)
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((so.D_DG, n)) @ L.T
    card = (np.abs(Z) > t).sum(axis=0)
    ok = bool(np.all(card == so.K_TOPK))
    # écart réalisé rang / seuil sur σ±, publié (domaine de validité Q-M10)
    a = np.abs(Z)
    idx = np.argpartition(a, -so.K_TOPK, axis=0)[-so.K_TOPK:, :]
    masque = np.zeros((so.D_DG, n), dtype=bool)
    masque[idx, np.arange(n)[None, :]] = True
    mI = masque[:, 0] & masque[:, 1]
    sig_rang = (float(np.mean(np.sign(Z[mI, 0]) == np.sign(Z[mI, 1])))
                if mI.any() else None)
    seuil = (np.abs(Z[:, 0]) > t) & (np.abs(Z[:, 1]) > t)
    sig_seuil = (float(np.mean(np.sign(Z[seuil, 0]) == np.sign(Z[seuil, 1])))
                 if seuil.any() else None)
    return (PASS if ok else FAIL), {
        "cardinal_du_support_a_seuil_commun": [int(v) for v in card],
        "cardinal_realise_par_rang": so.K_TOPK,
        "sd_t64_rectifiee_14_1": 0.042,
        "sigma_pm_par_rang": sig_rang, "sigma_pm_a_seuil": sig_seuil,
        "motif": "la sélection réalisée est un RANG ; une nulle à seuil "
                 "commun décrit un autre appareil (0-171). La rectification "
                 "§14.1 borne l'effet seuil-vs-rang sur P₂/P₃ à ~1e−3 : elle "
                 "affaiblit l'urgence du défaut sans l'annuler."}


def _pbs_inflation_certifiee():
    """Domaine de validite `Q-M10` (iii) : `|sim - quad| <= 0.005` => cellule
    **certifiee**, exces nul."""
    sim = {"par_strate": {s: {"rho_Base_sim_ratio_des_esperances": 0.6265}
                          for s in so.STRATES}}
    preds = {f"m|aucun|{s}": {"rho_chapeau_Base": 0.6239} for s in so.STRATES}
    r = so.inflation_epsilon_R(sim, preds, "m")
    ok = all(v["certifiee"] is True and v["exces"] == 0.0 for v in r.values())
    return (PASS if ok else FAIL), r


def _pbs_inflation_depassee():
    """Cas ECHOUANT MORDANT : `|sim - quad| > 0.005` => **l'exces s'ajoute a
    `eps_R`** (inflation PUBLIEE) ; *la forme gelee ne change pas*. La porte
    echoue si une cellule non certifiee passait sans inflation."""
    sim = {"par_strate": {s: {"rho_Base_sim_ratio_des_esperances": 0.5917}
                          for s in so.STRATES}}
    preds = {f"m|aucun|{s}": {"rho_chapeau_Base": 0.5858} for s in so.STRATES}
    r = so.inflation_epsilon_R(sim, preds, "m")
    non_certifiees = [s for s, v in r.items() if v["certifiee"] is not True]
    return (FAIL if non_certifiees else PASS), {
        "cellules_non_certifiees": non_certifiees,
        "ecart": r["S0"]["ecart"], "exces_ajoute_a_epsilon_R": r["S0"]["exces"],
        "motif": "au-dela de la tolerance, l'exces s'ajoute a eps_R ; la forme "
                 "gelee ne change pas"}


def _pbs_ordinal_joint():
    """Ordinal `N6` : les deux bras sont reechantillonnes sur le **MEME**
    tirage de tiges. Cas PASSANT : deux bras identiques => ecart nul et IC
    degenere a 0 (le reechantillon commun s'annule exactement)."""
    tiges = [f"t{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    tp = [(tiges[int(i)], tiges[int(j)])
          for i, j in zip(rng.integers(0, 10, 60), rng.integers(0, 10, 60))]
    num = rng.random(60)
    den = np.ones(60)
    r = so.ordinal_joint(num, den, tp, num, den, tp, tiges, b=300, seed=0)
    ok = abs(r["ecart_A_moins_B"]) < 1e-12 and max(abs(x) for x in r["IC_ecart"]) < 1e-12
    return (PASS if ok else FAIL), {
        "ecart": r["ecart_A_moins_B"], "IC": r["IC_ecart"],
        "borne": r["borne_de_la_conjonction"],
        "frac_grappe_manquante": r["fraction_reechantillons_a_grappe_manquante"]}


def _pbs_ordinal_bras_vide():
    """Cas ECHOUANT MORDANT (§14.4) : un bras sans domaine => **l'ordre n'a pas
    de domaine => SANS OBJET, jamais 0** ; et ce SANS OBJET-la (*un terme
    n'existe pas*) ne se confond **jamais** avec celui de la cellule modale
    (*rien ne bouge*)."""
    tiges = [f"t{i}" for i in range(10)]
    r = so.ordinal_joint([], [], [], [1.0], [1.0], [("t0", "t0")], tiges,
                         b=10, seed=0)
    return (FAIL if r.get("statut") == so.SANS_OBJET else PASS), r


def _pbs_poids_bootstrap_equiv():
    """Le poids agrégé **par couple de tiges** doit coïncider avec le poids
    **par paire** lu (`poids_bootstrap`) : une version rapide qui diverge de la
    version lue est un défaut."""
    tiges = [f"t{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    tp = [(tiges[i], tiges[j]) for i in rng.integers(0, 10, 200)
          for j in [int(rng.integers(0, 10))]]
    mult = {t: int(m) for t, m in zip(tiges, rng.integers(0, 4, 10))}
    lu = so.poids_bootstrap(tp, mult)
    cles, inv = so._tiges_combos(tp, tiges)
    rapide = so._poids_combos(inv, np.array([mult[t] for t in tiges]))[cles]
    e = float(np.max(np.abs(lu - rapide)))
    return (PASS if e == 0.0 else FAIL), {"ecart_max": e, "n_paires": len(tp),
                                          "n_couples_de_tiges": len(inv)}


def _pbs_poids_bootstrap_faux():
    """Cas ÉCHOUANT MORDANT : le poids **produit** appliqué aussi aux paires
    INTRA-tige (`m²` au lieu de `m`) — la règle déclarée est `m[t_a]` quand
    `t_a = t_b`."""
    tiges = [f"t{i}" for i in range(4)]
    tp = [("t0", "t0"), ("t1", "t2")]
    mult = {"t0": 3, "t1": 2, "t2": 1, "t3": 0}
    lu = so.poids_bootstrap(tp, mult)
    faux = np.array([mult[a] * mult[b] for a, b in tp], dtype=float)
    return (PASS if np.array_equal(lu, faux) else FAIL), {
        "regle_declaree": lu.tolist(), "regle_produit_partout": faux.tolist(),
        "motif": "sur une paire intra-tige, le produit compte la tige DEUX "
                 "fois et gonfle sa multiplicité"}


def _pbs_partitions_exclusives():
    """Les quatre partitions du §4.5 sont **exhaustives et exclusives**, et
    l'ordre gravé (complémentation EN DERNIER) est respecté : énumération
    mécanique sur une grille de cas."""
    cas_B = [(None, None, 0.0, 0), ([0.55, 0.7], 0.62, 1.0, 100),
             ([0.10, 0.20], 0.15, 1.0, 100), ([0.30, 0.60], 0.45, 1.0, 100)]
    vus_B = [so.classe_B(ic, pt, den, P) for ic, pt, den, P in cas_B]
    cas_D = [(True, None, None), (False, [-0.9, -0.5], 0.1),
             (False, [-0.05, 0.05], 0.1), (False, [0.2, 0.4], 0.1),
             (False, [0.05, 0.4], 0.1)]
    vus_D = [so.classe_Delta(*c) for c in cas_D]
    cas_R = [(True, None, None), (False, [-0.9, -0.5], 0.1),
             (False, [-0.05, 0.05], 0.1), (False, [0.2, 0.4], 0.1),
             (False, [0.05, 0.4], 0.1)]
    vus_R = [so.classe_R(*c) for c in cas_R]
    ok = (vus_B == ["B-vide", "B-haut", "B-mort", "B-ind"]
          and vus_D == ["Δ-sansobjet", "Δ-anti", "Δ-nul", "Δ-priv", "Δ-ind"]
          and vus_R == ["R-vide", "R-moins", "R-nul", "R-plus", "R-ind"])
    return (PASS if ok else FAIL), {
        "classes_B": vus_B, "classes_Delta": vus_D, "classes_R": vus_R,
        "cardinal_formel": 4 * 5 * 5 * 5,
        "regle": "complémentation EN DERNIER (D18)"}


def _pbs_partitions_ordre_faux():
    """Cas ÉCHOUANT MORDANT : `Δ-nul` évalué **avant** `Δ-sansobjet` — un
    vivier vide serait lu « rien ne bouge » au lieu de « un terme n'existe
    pas » (§14.4 : *les deux `SANS OBJET` ne se confondent jamais*)."""
    vrai = so.classe_Delta(True, [-0.01, 0.01], 0.1)
    faux = "Δ-nul"
    return (PASS if vrai == faux else FAIL), {
        "classe_dans_l_ordre_grave": vrai, "classe_si_ordre_inverse": faux,
        "motif": "« l'ordre n'a pas de domaine parce que RIEN NE BOUGE » "
                 "contre « parce qu'UN TERME N'EXISTE PAS » — elles "
                 "n'autorisent pas les mêmes suites (§14.4)"}


def _pbs_cardinaux_enumeres():
    """Cardinaux comptés **par ÉNUMÉRATION**, jamais supposés (§11.3)."""
    dec, cel = _pbs_materiau()
    viv, _ = _pbs_vivier()
    par = {}
    for (a, b), j in viv.items():
        par.setdefault(so.p4.strate(dec[a], dec[b]), set()).add(int(j.size))
    n = len(dec)
    strates = {s: 0 for s in so.STRATES}
    for a in range(n):
        for b in range(a + 1, n):
            strates[so.p4.strate(dec[a], dec[b])] += 1
    attendu = {"S3": 60, "S2": 90, "S1": 360, "S0": 1260}
    ok = (strates == attendu and n == 60 and len(cel) == 6
          and par["S3"] == {1} and par["S2"] == {0} and par["S1"] == {9}
          and par["S0"] <= {24, 27, 30})
    return (PASS if ok else FAIL), {
        "U_unites_decisionnelles": n, "T_cellules_de_capture": len(cel),
        "cellules_decisionnelles": len(so.MODELES) * len(so.STRATES),
        "paires_par_strate_et_par_cellule": strates,
        "vivier_cardinaux_par_strate": {k: sorted(v) for k, v in par.items()},
        "attendu_14_2": so.CARDINAUX_VIVIER_ATTENDUS,
        "unite_de_P": {s: strates[s] * len(cel) for s in so.STRATES}}


def _pbs_cardinaux_supposes():
    """Cas ÉCHOUANT MORDANT (§14.3) : la clause gelée du §4.5 prédit un vivier
    vide **en S3**. L'énumération dit S3 = 1 et **S2 = 0**. La clause n'est
    **pas corrigée** ; la falsification est **consignée**."""
    dec, _ = _pbs_materiau()
    viv, _ = _pbs_vivier()
    s3 = {int(j.size) for (a, b), j in viv.items()
          if so.p4.strate(dec[a], dec[b]) == "S3"}
    s2 = {int(j.size) for (a, b), j in viv.items()
          if so.p4.strate(dec[a], dec[b]) == "S2"}
    faux = (s3 == {0})
    return (PASS if faux else FAIL), {
        "prediction_gelee_4_5": "vivier vide attendu possible en S3",
        "enumere_S3": sorted(s3), "enumere_S2": sorted(s2),
        "constat": "adresse INVERSÉE — S3 n'est jamais vide, le vide certain "
                   "est S2",
        "traitement": "clause NON corrigée, exécutée telle qu'écrite ; "
                      "Δ-sansobjet se déclenche sur la cellule vide QUELLE "
                      "QU'ELLE SOIT ; la falsification est consignée (§14.3)"}


def _pbs_v_diag():
    cel = {"m|aucun|S0": {k: 1 for k in
                          ("rho_Base", "rho_chapeau_Base", "R_Base",
                           "Delta_Base", "Delta_chapeau_Base", "n_vide",
                           "sigma_pm", "sigma_hat_pool")}}
    return so.v_diag_pbs(cel)


def _pbs_v_diag_sans_rho_chapeau():
    """Cas ÉCHOUANT MORDANT (D34, (xxvii)) : `ρ_Base` publié **sans son
    `ρ̂_Base` apparié** ⇒ le rapport est **NON ÉCRIVABLE**."""
    cel = {"m|aucun|S0": {k: 1 for k in
                          ("rho_Base", "R_Base", "Delta_Base",
                           "Delta_chapeau_Base", "n_vide", "sigma_pm",
                           "sigma_hat_pool")}}
    return so.v_diag_pbs(cel)


def _pbs_v_queue():
    return so.v_queue(_pbs_t(), [0.38, 0.40], True)


def _pbs_v_queue_t_en_dur():
    """Cas ÉCHOUANT MORDANT (0-177) : `t = 2.66` **en dur** — la pente
    `(φ/Q)²` varie de 1.5 % entre 2.66 et 2.6601."""
    v, d = so.v_queue(2.66, [0.38, 0.40], True)
    pente_dur = so.pente_plackett(2.66)
    pente_vraie = so.pente_plackett(_pbs_t())
    return v, {"t_en_dur": 2.66, "t_calcule": _pbs_t(),
               "pente_en_dur": pente_dur, "pente_calculee": pente_vraie,
               "variation_relative": abs(pente_dur / pente_vraie - 1.0),
               "motif": d["pool_gele"]}


def _pbs_sigma_mort_atteignable():
    """`Q-M13` — **`Σ-mort` est ATTEIGNABLE, NON MODALE** : le banc doit le
    savoir avant de dépenser une clause sur `Ψ`. Frontière : l'IC contient 0.5
    ssi `ρ_ic ≥ ≈ 0.26` (à `σ± = 0.410`) à `0.38` (à 0.392)."""
    m_barre, n_grappes = 169.0, 60.0
    front = {}
    for sig, rho_ic in ((0.410, 0.26), (0.392, 0.38)):
        deff = 1.0 + (m_barre - 1.0) * rho_ic
        n_eff = 5083.0 / deff
        se = np.sqrt(0.25 / n_eff)
        front[str(sig)] = {"rho_ic_frontiere": rho_ic, "DEFF": deff,
                           "N_effectif": n_eff,
                           "demi_largeur_IC_95": 1.96 * se,
                           "IC_contient_0.5": bool(
                               abs(0.5 - sig) <= 1.96 * se)}
    ok = all(v["IC_contient_0.5"] for v in front.values())
    return (PASS if ok else FAIL), {
        "frontieres": front, "n_grappes": int(n_grappes),
        "statut": "Σ-mort ATTEIGNABLE, NON MODALE — ratifié §14.5",
        "rho_ic": "aucun a priori neuro (interdit (xxiv)) ; V-grappe tranche"}


def _pbs_sigma_mort_binomiale():
    """Cas ÉCHOUANT MORDANT (0-170) : traiter les 5083 trials comme
    indépendants donne « 12 à 15 σ » — **un écart en σ dont le `N` effectif est
    inconnu n'est pas un écart en σ**."""
    se = np.sqrt(0.25 / 5083.0)
    z = abs(0.5 - 0.392) / se
    return (PASS if z < 3.0 else FAIL), {
        "z_sous_independance": z, "N_suppose": 5083,
        "motif": "la dépendance LOO est POSITIVE ⇒ N effectif ≪ 5083, jamais "
                 "établi ; l'ampleur en σ est RECTIFIÉE (P5, D30 alinéa 1)"}


def _pbs_perimetre():
    """`V-perimetre` — 0 GPU, aucun forward, `M` jamais instanciée, `engram/`
    non modifié. Vérifié par LECTURE du source de `support_overlap.py`."""
    src = (ROOT / "eval" / "support_overlap.py").read_text(encoding="utf-8")
    interdits = ["mem.write", "loss.backward", ".cuda(", "device=\"cuda\"",
                 "generate(", "AutoModelForCausalLM"]
    trouves = [x for x in interdits if x in src]
    return (PASS if not trouves else FAIL), {
        "motifs_interdits_trouves": trouves,
        "M_instanciee": "FastWeightMemory est construite pour LIRE `G` "
                        "seulement ; `M` n'est ni lue ni écrite (même chemin "
                        "qu'au cycle précédent)",
        "GPU": 0, "forwards": 0,
        "engram_modifie": False}


def _pbs_perimetre_gpu():
    """Cas ÉCHOUANT MORDANT : un motif GPU dans le source du cycle ⇒ le
    périmètre gravé est violé."""
    faux = 'model.to(device="cuda")'
    return (FAIL if ".cuda" in faux or "cuda" in faux else PASS), {
        "extrait": faux,
        "motif": "le périmètre grave 0 GPU, aucun forward ; V-cache divergent "
                 "⇒ ARRÊT, PAS de re-forward (P8)"}


# -------------------------------------------------------------------------
#  TROIS TROUS DE MORDANT du tour de correction — comblés
# -------------------------------------------------------------------------
#  Le `lab-verifier` a muté trois fonctions du cycle et le banc a laissé
#  `E = 0` : (a) `so.classe_Sigma` forcée à rendre toujours `Σ-epuise` — le
#  classificateur qui attribue les 24 classes n'était gardé par AUCUNE clause
#  mordante ; (b) `so.mu_deux_chemins` forcée à `chemins_concordants=True` ;
#  (c) `so.majorant_de_fuite` forcé à `0.0`. Les trois clauses ci-dessous
#  MORDENT sur ces trois mutations exactes.

def _pbs_classe_Sigma_cinq_branches():
    """Cas PASSANT : les **cinq** classes de la partition `Σ` (§4.5) sont
    atteintes dans l'**ordre gravé**, complémentation EN DERNIER (D18).

    **MORD sur la mutation (a)** : `classe_Sigma` forcée à rendre toujours
    `Σ-epuise` rate quatre des six cas ci-dessous."""
    ic_ok = [[-0.01, 0.01], [-0.02, 0.02]]
    eps = [0.05, 0.05]
    ic_hors = [[0.20, 0.30], [-0.02, 0.02]]
    cas = [
        ("Σ-mort", so.classe_Sigma(True, False, ic_ok, eps, True)),
        ("Σ-vide", so.classe_Sigma(False, True, ic_ok, eps, True)),
        ("Σ-epuise", so.classe_Sigma(False, False, ic_ok, eps, True)),
        # conjoint NON satisfait, mêmes IC : la classe ne peut PAS être
        # `Σ-epuise` — c'est le conjoint qui la retient
        ("Σ-ind", so.classe_Sigma(False, False, ic_ok, eps, False)),
        ("Σ-residu", so.classe_Sigma(False, False, ic_hors, eps, True)),
        ("Σ-ind", so.classe_Sigma(False, False, None, None, True)),
    ]
    ok = all(attendu == obtenu for attendu, obtenu in cas)
    return (PASS if ok else FAIL), {
        "cas": [{"attendu": a, "obtenu": b, "ok": a == b} for a, b in cas],
        "ordre_grave": "Σ-mort → Σ-vide → Σ-epuise → Σ-residu → Σ-ind",
        "conjoint": "le conjoint reçoit la complétion PROMUE (monotonie "
                    "complète) — décision PI du 2026-08-28, D30 alinéa 2"}


def _pbs_classe_Sigma_conjoint_ignore():
    """Cas ÉCHOUANT MORDANT : le profil réel de SmolLM2
    `[0.3371, 0.4227, 0.3932, 0.4123]` est **conforme à deux points** et
    **NON monotone**. Les deux complétions ne rendent pas la même classe : la
    descriptive licencie `Σ-epuise`, la **PROMUE** rend `Σ-ind`.

    **MORD sur la mutation (a)** : si `classe_Sigma` rend toujours `Σ-epuise`,
    les deux classes coïncident et le cas est **VACUÉ PAR SATISFACTION**."""
    profil = [0.3371, 0.4227, 0.3932, 0.4123]
    ic_ok = [[-0.01, 0.01], [-0.02, 0.02], [-0.01, 0.02], [-0.02, 0.01]]
    eps = [0.05, 0.05, 0.05, 0.05]
    c_2pts = so.classe_Sigma(False, False, ic_ok, eps,
                             so.profil_conforme_a_la_loi(profil))
    c_mono = so.classe_Sigma(False, False, ic_ok, eps,
                             so.profil_monotone_complet(profil))
    return (PASS if c_2pts == c_mono else FAIL), {
        "profil": profil,
        "classe_sous_DEUX_POINTS_descriptive": c_2pts,
        "classe_sous_MONOTONIE_COMPLETE_promue": c_mono,
        "conforme_deux_points": so.profil_conforme_a_la_loi(profil),
        "monotone_complet": so.profil_monotone_complet(profil),
        "motif": "monotone ⇒ deux points est une implication STRICTE : la "
                 "complétion à deux points rend la prédiction PLUS FACILE "
                 "(D30 alinéa 2). ARBITRAGE PI RENDU : monotonie complète."}


def _pbs_mu_deux_chemins_divergence_reelle():
    """Cas ÉCHOUANT MORDANT : les deux chemins de `μ_global` divergent
    **RÉELLEMENT** — sommation par paires de numpy contre `math.fsum` sur une
    colonne à annulation catastrophique. Aucun champ n'est injecté : la
    divergence est **produite par la fonction**.

    **MORD sur la mutation (b)** : `mu_deux_chemins` forcée à
    `chemins_concordants=True` rend ce cas `PASS` ⇒ VACUÉ PAR SATISFACTION."""
    H = {"c1": np.array([[1e16, 1.0], [1.0, 1.0]], dtype=np.float64),
         "c2": np.array([[-1e16, 1.0], [1.0, 1.0]], dtype=np.float64)}
    cel = list(H)
    ch = so.mu_deux_chemins(H, cel)
    v, d = so.v_mu(so.citations_par_ligne(), ch,
                   so.majorant_de_fuite(H, cel, _pbs_t()))
    return v, {
        "chemins_concordants": ch["chemins_concordants"],
        "ecart_relatif_en_norme": ch["ecart_relatif_en_norme"],
        "ecart_max_absolu_par_coordonnee":
            ch["ecart_max_absolu_par_coordonnee"],
        "tolerance": ch["tolerance"],
        "motif": "annulation catastrophique sur la coordonnée 0 : la "
                 "sommation par paires de numpy et `math.fsum` ne rendent pas "
                 "le même nombre. La divergence n'est PAS injectée — elle est "
                 "calculée par la fonction elle-même.",
        "verdict_v_mu": v}


def _pbs_majorant_de_fuite_recalcule():
    """Cas PASSANT : le majorant de fuite est **recalculé par force brute**
    sur toutes les paires, indépendamment de la fonction, et comparé à elle.

    **MORD sur la mutation (c)** : `majorant_de_fuite` forcé à `0.0` rate
    l'égalité au recalcul **et** l'exigence de stricte positivité."""
    H = _pbs_mu_synthetique(seed=17, n_cell=12, n_cellules=3, d=16)
    cel = list(H)
    fu = so.majorant_de_fuite(H, cel, _pbs_t())
    Hall = np.concatenate([H[c] for c in cel], axis=0)
    n = Hall.shape[0]
    S = Hall.sum(axis=0)
    mu = S / n
    nmu = float(np.linalg.norm(mu))
    pire_n, pire_g = 0.0, 0.0
    for c in cel:
        Hc = H[c]
        m = Hc.shape[0]
        for i in range(m):
            for j in range(i + 1, m):
                mul = (S - Hc[i] - Hc[j]) / (n - 2)
                pire_n = max(pire_n,
                             float(np.linalg.norm(mul - mu)) / nmu)
                nml = float(np.linalg.norm(mul))
                for x in (Hc[i], Hc[j]):
                    nx = float(np.linalg.norm(x))
                    pire_g = max(pire_g, abs(float(x @ mul) / (nx * nml)
                                             - float(x @ mu) / (nx * nmu)))
    ok = (abs(fu["majorant_ecart_relatif_de_norme"] - pire_n) < 1e-12
          and abs(fu["majorant_ecart_absolu_sur_gamma"] - pire_g) < 1e-12
          and abs(fu["facteur_2_sur_n_moins_2"] - 2.0 / (n - 2)) < 1e-15
          and fu["majorant_de_fuite_sur_rho_Base"] > 0.0)
    return (PASS if ok else FAIL), {
        "publie_norme": fu["majorant_ecart_relatif_de_norme"],
        "recalcul_force_brute_norme": pire_n,
        "publie_gamma": fu["majorant_ecart_absolu_sur_gamma"],
        "recalcul_force_brute_gamma": pire_g,
        "majorant_de_fuite_sur_rho_Base":
            fu["majorant_de_fuite_sur_rho_Base"],
        "n": n, "n_paires_recalculees": 3 * (12 * 11 // 2),
        "exigence": "égalité au recalcul indépendant ET stricte positivité — "
                    "un majorant nul n'est pas un majorant mesuré"}


def _pbs_majorant_de_fuite_force_a_zero():
    """Cas ÉCHOUANT MORDANT : le majorant **forcé à `0.0`**. `V-mu` le laisse
    passer — elle n'exige qu'un flottant fini — donc **seul le recalcul le
    tue**. C'est exactement le trou que la mutation (c) a exhibé.

    **MORD sur la mutation (c)** : si `majorant_de_fuite` rend déjà `0.0`, le
    défaut n'est plus démontrable et le cas devient `PASS` ⇒ VACUÉ."""
    H = _pbs_mu_synthetique(seed=17, n_cell=12, n_cellules=3, d=16)
    cel = list(H)
    fu = so.majorant_de_fuite(H, cel, _pbs_t())
    fu0 = dict(fu)
    fu0["majorant_de_fuite_sur_rho_Base"] = 0.0
    v, _ = so.v_mu(so.citations_par_ligne(),
                   so.mu_deux_chemins(H, cel), fu0)
    defaut = (v == PASS and fu["majorant_de_fuite_sur_rho_Base"] > 0.0)
    return (FAIL if defaut else PASS), {
        "majorant_reel": fu["majorant_de_fuite_sur_rho_Base"],
        "majorant_force": 0.0,
        "verdict_de_V_mu_sur_le_majorant_force": v,
        "motif": "V-mu n'exige qu'un flottant fini : 0.0 la satisfait. Le "
                 "recalcul indépendant est la seule clause qui morde."}


def build_clauses_pbs():
    C = []

    def clause(name, pass_desc, fail_desc, cases_pass, cases_fail,
               structural=None, note=None):
        C.append({"clause": name, "pass_case": pass_desc, "fail_case": fail_desc,
                  "cases_pass": cases_pass, "cases_fail": cases_fail,
                  "structural": structural, "note": note})

    clause("V-cache", "les trois `.npz` de v4 sont lisibles, hash publié",
           "un modèle absent du cache ⇒ ARRÊT, PAS de re-forward (P8)",
           [("cache v4 complet", PASS, lambda: so.v_cache(so.MODELES))],
           [("modèle inexistant", FAIL,
             lambda: so.v_cache(("modele/qui-n-existe-pas",)))],
           note="périmètre gravé : V-cache divergent ⇒ ARRÊT sans re-forward.")

    clause("V-G (v2)", "G du projet instanciée et hashée ; cos fp32/fp64 sous "
           "1e−6", "tolérance nulle ⇒ arrêt de provenance",
           [("tolérance 1e−6", PASS,
             lambda: so.v_g((so.MODELES[0],), tol_precision=1e-6))],
           [("tolérance 0", FAIL,
             lambda: so.v_g((so.MODELES[0],), tol_precision=0.0))])

    clause("Q-M10 — identité 1 : γ_a = γ_b = 0 ⇒ P₃ = P₂·(1/128)",
           "l'identité tient à 1e−14 près", "la cible 1/64 (unité d'INDICES) "
           "au lieu de 1/128 (fraction de k)",
           [("cinq ρ, γ nuls", PASS, _pbs_identite_1)],
           [("cible P₂/64", FAIL, _pbs_identite_1_fausse)],
           note="`t` est calculé au banc en fp64 : t = ndtri(1 − 1/256) ; "
                "2Q(t) = 1/128 EXACTEMENT en binaire.")

    clause("Q-M10 — identité 2 : tous cosinus = 1 ⇒ ρ̂ = 1",
           "le cas dégénéré gelé (|γ| > 1 − 1e−9) rend exactement 1",
           "un clip silencieux de γ à 0.99 déplace ρ̂",
           [("γ = ρ = 1", PASS, _pbs_identite_2)],
           [("γ clippé à 0.99", FAIL, _pbs_identite_2_clip)])

    clause("Q-M10 — identité 3 : ρ = 0, γ = 0 ⇒ P₂ = (1/128)²",
           "forme fermée exacte", "la cible (1/64)²",
           [("ρ = 0", PASS, _pbs_identite_3)],
           [("cible (1/64)²", FAIL, _pbs_identite_3_binomiale)])

    clause("Q-M10 — quadrature 1-D contre-vérifiée",
           "GL 64 et GL 128 sur (t, 8) s'accordent sous 1e−9",
           "une quadrature à 4 nœuds dépasse la tolérance",
           [("GL64 vs GL128 sur régime S3", PASS, _pbs_quadrature_croisee)],
           [("GL4 vs GL128", FAIL, _pbs_quadrature_grossiere)],
           note="scipy n'est pas une dépendance : `quad` est substituée par "
                "Gauss–Legendre (nominal) + Gauss–Kronrod adaptatif "
                "(epsabs 1e−12, epsrel 1e−10). SUBSTITUTION DÉCLARÉE.")

    clause("Q-M10 — F₂ par deux chemins indépendants",
           "Sheppard/Plackett en θ ≡ forme conditionnelle en x sous 1e−10",
           "l'indépendance supposée (Φ(h)Φ(k)) diverge dès r ≠ 0",
           [("six corrélations, quatre coins", PASS, _pbs_F2_deux_chemins)],
           [("produit des marginales", FAIL, _pbs_F2_independance_supposee)])

    clause("Q-M10 — Gram non-PSD ⇒ ARRÊT, jamais de clip",
           "un Gram PSD passe et la quadrature s'exécute",
           "un Gram non-PSD lève ArretQM10 au lieu d'être clippé",
           [("triples de régime S3, det > 0", PASS, _pbs_gram_psd)],
           [("det < −1e−8", FAIL, _pbs_gram_non_psd_arrete)],
           note="« c'est un Gram de vecteurs réels, donc symptôme de bug ».")

    clause("Q-M10 — réduction à UN FACTEUR refusée hors S0",
           "sur S0 (c_ab ≈ 0) la réduction est licite (< 5 %)",
           "sur S3/S2 (c_ab grand et positif) elle est FAUSSE",
           [("c_ab ≈ 0", PASS, _pbs_reduction_un_facteur_S0)],
           [("c_ab ∈ [0.35, 0.70]", FAIL, _pbs_reduction_un_facteur_S3)],
           note="la forme gelée est la trivariée exacte, par paire, avec "
                "(ρ_ab, γ_a, γ_b) tous trois.")

    clause("V-quad — exécutée et publiée AVANT toute mesure",
           "ρ̂_Base par cellule + déclaration d'atteignabilité de B-mort (D31)",
           "non exécutée ⇒ ρ_Base RETIRÉ de l'interprétation",
           [("prédictions calculées", PASS, lambda: _pbs_v_quad(True))],
           [("V-quad non exécutée", FAIL, lambda: _pbs_v_quad(False))],
           note="D34 : un couple (ρ_Base, ρ̂_Base), jamais un nombre.")

    clause("V-T — support vide, énuméré",
           "T, U, cellules et paires inter-cadres comptés par énumération ; "
           "C-σ-mort déclaré INATTEIGNABLE d'avance",
           "antipode déclaré atteignable sans énumération",
           [("énumération du matériau v4", PASS, _pbs_v_T)],
           [("déclaration sans énumération", FAIL, _pbs_v_T_sans_enumeration)],
           note="les paires sont intra-cadre PAR CONSTRUCTION (0-157) : "
                "l'ensemble des paires de cadres t ≠ t' est ∅.")

    clause("V-appar — vivier N-état, lecture RELATIONNELLE (§14.2)",
           "cardinaux énumérés S3=1, S2=0, S1=9, S0 ∈ {24,27,30}",
           "la lecture littérale éteint trois strates sur quatre par vacuité",
           [("lecture relationnelle", PASS, _pbs_v_appar)],
           [("lecture littérale", FAIL, _pbs_v_appar_litterale)],
           note="« un confondant supprimé avec son support n'est pas un "
                "confondant contrôlé — c'est une mesure absente » (§14.2).")

    clause("V-vide — support vide ⇒ SANS OBJET, jamais 0",
           "n_vide publié par cellule et par condition",
           "un dénominateur nul publié en 0",
           [("deux cellules à dénominateur non nul", PASS, _pbs_v_vide)],
           [("dénominateur nul", FAIL, _pbs_v_vide_zero)])

    clause("V-ecriture — écriture littérale de Δ_Base (0-173)",
           "la gravée est calculée et les deux alternatives publiées",
           "un cas où les deux lectures DIVERGENT",
           [("écriture gravée", PASS, _pbs_v_ecriture)],
           [("divergence des deux lectures", FAIL, _pbs_v_ecriture_divergente)],
           note="troisième occurrence du mode 0-155 dans le projet.")

    clause("V-grappe — bootstrap par grappe, BLOQUANTE (0-170, 0-178)",
           "IC, cardinal effectif et méthode de quantile publiés",
           "un bootstrap sans cardinal effectif ni méthode de quantile",
           [("60 grappes, fraction de grappe manquante publiée", PASS,
             _pbs_v_grappe)],
           [("cardinal effectif retiré", FAIL, _pbs_v_grappe_sans_cardinal)],
           note="(xxviii) : aucune phrase qualifiant l'ampleur de σ± n'est "
                "publiable avant PASS de cette porte.")

    clause("V-rang-def — r = max(rang_A, rang_B) gelé, planchers",
           "8 buckets, ≥ 200 trials ET ≥ 8 paires, trois lectures rejetées "
           "par écrit", "un bucket à 199 trials doit être signalé",
           [("profil conforme", PASS, _pbs_v_rang_def)],
           [("bucket à 199 trials", FAIL, _pbs_v_rang_def_199)])

    clause("Fusion dyadique PRÉ-DÉCLARÉE des buckets",
           "8 → 4 → 2 → 1, appliquée AVANT toute lecture",
           "une fusion qui dépendrait du résultat",
           [("buckets sous plancher fusionnés mécaniquement", PASS,
             _pbs_fusion)],
           [("fusion conditionnée au résultat", FAIL,
             _pbs_fusion_apres_lecture)])

    clause("Profil signé de σ± par bucket (§5-7)",
           "l'écart à 0.5 décroît avec le rang (croît avec la magnitude)",
           "un profil plat ⇒ la loi est FAUSSE",
           [("profil décroissant en rang", PASS, _pbs_profil_signe)],
           [("profil plat", FAIL, _pbs_profil_plat)],
           note="0-165 : le ρ de la loi de queue EST le cos des états "
                "centrés ; seule la forme du profil les départage.")

    clause("V-ident — quantités forcées par une identité ((xxv))",
           "les quatre quantités forcées déclarées avec leur dérivation",
           "une quantité forcée non déclarée",
           [("déclarations complètes", PASS, _pbs_v_ident)],
           [("cos des centroïdes retiré", FAIL, _pbs_v_ident_manquante)],
           note="« valeur imposée par l'identité <nommée> ; poids probant "
                "nul ». Ajoutée par ce cycle : l'INVARIANCE D'ÉCHELLE de "
                "topk rend N-plac indépendante de la magnitude.")

    clause("V-S — DEFF_S publié, formule binomiale INTERDITE (0-180)",
           "q₉₅ sur R tirages de S, DEFF_S > 1, méthode de quantile déclarée",
           "DEFF_S = 1 (la binomiale déguisée)",
           [("400 tirages de S", PASS, _pbs_v_S)],
           [("DEFF_S forcé à 1", FAIL, _pbs_v_S_binomiale)],
           note="borne démontrée par lab-math : DEFF_S ≥ 2.8. Poids probant "
                "NUL, déclaré (D32).")

    clause("V-seed — présence, jamais vérité (0-145)",
           "seed = 0 ACCEPTÉ : la porte teste la présence du champ",
           "générateur absent ⇒ la porte mord sur le champ manquant",
           [("spec complète, seed = 0", PASS, _pbs_v_seed)],
           [("générateur retiré", FAIL, _pbs_v_seed_sans_generateur)])

    clause("V-loo — LOO principal, non-LOO descriptif (0-174)",
           "les deux publiés, écart comparé à ε_B",
           "le LOO absent ⇒ arrêt",
           [("les deux valeurs", PASS, _pbs_v_loo)],
           [("LOO manquant", FAIL, _pbs_v_loo_inversion)])

    clause("V-decimales — ≥ 4 décimales (0-179)",
           "cosinus, ρ_Base, R_Base, Ψ à 4 décimales",
           "deux décimales : −0.0019 et −0.0023 deviennent identiques",
           [("quatre décimales", PASS, _pbs_v_decimales)],
           [("deux décimales", FAIL, _pbs_v_decimales_deux)])

    clause("V-P8 — ≥ 8 paires par cellule",
           "deux cellules à P ≥ 8", "une cellule à 7 paires",
           [("P = 360 et 7560", PASS, _pbs_v_P8)],
           [("P = 7", FAIL, _pbs_v_P8_sept)])

    clause("V-queue — t calculé au banc, jamais 2.66 en dur (0-177)",
           "t = ndtri(1 − 1/256) ; σ̂±_pool gelée littéralement",
           "t = 2.66 en dur ⇒ la pente varie de 1.5 %",
           [("t calculé", PASS, _pbs_v_queue)],
           [("t en dur", FAIL, _pbs_v_queue_t_en_dur)])

    clause("V-diag — le couple (ρ_Base, ρ̂_Base) obligatoire (D34)",
           "les huit colonnes publiées par strate, condition et modèle",
           "ρ_Base publié sans son ρ̂_Base ⇒ rapport NON ÉCRIVABLE",
           [("cellule complète", PASS, _pbs_v_diag)],
           [("ρ̂_Base retiré", FAIL, _pbs_v_diag_sans_rho_chapeau)])

    clause("Q-M11 — terme G de ε_R (plug-in)",
           "sd_G calculé sur les (u_i, v_i) observés et ajouté à chaque "
           "réplique", "omettre le terme O(1/√D) rend ε_R trop étroit",
           [("plug-in sur données synthétiques", PASS, _pbs_sd_G)],
           [("terme G omis", FAIL, _pbs_sd_G_omis)])

    clause("Q-M1-bis — simulation EXACTE à rang",
           "sous Σ = I, ρ_Base_sim vaut k/D = 1/128",
           "à seuil commun le cardinal du support n'est pas 64",
           [("Σ = I, N_G = 6", PASS, _pbs_simulation_identite)],
           [("cardinal à seuil commun ≠ 64", FAIL,
             _pbs_simulation_seuil_fixe)],
           note="rectification §14.1 : la fluctuation du seuil réalisé n'est "
                "pas Gumbel (SD 15-20 %) mais la 64ᵉ statistique d'ordre, "
                "sd(t̂₆₄) ≈ 0.042, soit ≈ 1.6 % de t. Le défaut 0-171 et son "
                "correctif sont INCHANGÉS.")

    clause("Certification Q-M10 (iii) — inflation de eps_R",
           "|sim - quad| <= 0.005 => cellule certifiee, exces nul",
           "au-dela, l'exces s'ajoute a eps_R (inflation publiee)",
           [("ecart 0.0026", PASS, _pbs_inflation_certifiee)],
           [("ecart 0.0059", FAIL, _pbs_inflation_depassee)],
           note="la simulation a rang tourne AVANT la synthese : lire eps_R "
                "avant l'inflation serait lire une resolution que la "
                "certification n'accorde pas.")

    clause("Ordinal N6 — bootstrap JOINT, borne par le min (0-63)",
           "les deux bras partagent le meme reechantillon de tiges",
           "un bras sans domaine => SANS OBJET, jamais 0",
           [("deux bras identiques => ecart exactement nul", PASS,
             _pbs_ordinal_joint)],
           [("bras A vide", FAIL, _pbs_ordinal_bras_vide)],
           note="0-63 : tout produit de p-valeurs par modele rend le run "
                "invalide. §14.4 : le bras S3 a R = 1, variance de tirage "
                "NULLE, il entre comme constante.")

    clause("Poids de bootstrap deux-voies (Q-M14 (b))",
           "l'agrégation par couple de tiges ≡ la règle par paire lue",
           "le produit appliqué aussi aux paires INTRA-tige",
           [("200 paires aléatoires", PASS, _pbs_poids_bootstrap_equiv)],
           [("m² sur une paire intra-tige", FAIL, _pbs_poids_bootstrap_faux)],
           note="règle déjà dans le code au cycle précédent (l. 237-243) ; "
                "lab-math la confirme conservatrice.")

    clause("Partitions §4.5 — exhaustives, exclusives, ordre gravé (D18)",
           "les quatre partitions rendent la classe attendue sur chaque cas",
           "Δ-nul évalué avant Δ-sansobjet",
           [("grille énumérée", PASS, _pbs_partitions_exclusives)],
           [("ordre inversé", FAIL, _pbs_partitions_ordre_faux)],
           note="§14.4 : deux SANS OBJET ne se confondent jamais.")

    clause("Cardinaux comptés PAR ÉNUMÉRATION (§11.3)",
           "U = 60, T = 6, paires par strate, vivier par strate — comptés",
           "la clause gelée §4.5 prédit le vide en S3 : FALSIFIÉE",
           [("énumération complète", PASS, _pbs_cardinaux_enumeres)],
           [("prédiction d'adresse du §4.5", FAIL, _pbs_cardinaux_supposes)],
           note="§14.3 : la clause n'est PAS corrigée ; elle est exécutée "
                "telle qu'écrite et la falsification est consignée.")

    clause("Q-M13 — Σ-mort ATTEIGNABLE, NON MODALE",
           "la frontière ρ_ic ≈ 0.26–0.38 est calculée et publiée",
           "traiter 5083 trials comme indépendants donne « 12 à 15 σ »",
           [("frontières calculées", PASS, _pbs_sigma_mort_atteignable)],
           [("indépendance supposée", FAIL, _pbs_sigma_mort_binomiale)],
           note="0-170 : un écart en σ dont le N effectif est inconnu n'est "
                "pas un écart en σ. Rectification autorisée (P5, D30 al. 1).")

    clause("V-mu — formules relues, second chemin, majorant de fuite (§4.6)",
           "les quatre formules ont une occurrence UNIQUE dans le source, les "
           "deux chemins de μ_global concordent, le majorant est publié",
           "une formule sans occurrence unique ⇒ le chiffre serait RECOPIÉ, "
           "pas RELU ⇒ arrêt (§6.D)",
           [("citations + double chemin + majorant", PASS, _pbs_v_mu),
            ("identité μ^LOO − μ = (2/(n−2))(μ − h̄)", PASS,
             _pbs_mu_identite_loo)],
           [("formule absente du source", FAIL, _pbs_v_mu_citation_perdue),
            ("les deux chemins divergent", FAIL,
             _pbs_v_mu_chemins_divergents),
            ("majorant de fuite non publié", FAIL,
             _pbs_v_mu_fuite_non_publiee),
            ("facteur 1/(n−2) au lieu de 2/(n−2)", FAIL,
             _pbs_mu_identite_fausse)],
           note="§6.D : V-mu échec ⇒ run INVALIDE. Porte de PROVENANCE : elle "
                "ne touche aucune mesure. Défaut 0-190 du tour de correction : "
                "la porte était déclarée « partielle » alors que RIEN n'avait "
                "été exécuté.")

    clause("cos_p de σ̂±_pool — les TROIS lectures publiées côte à côte",
           "h (principale déclarée), z et tronqué φ, chacune avec son Ψ et sa "
           "classe", "une seule lecture publiée sous un arbitrage OUVERT",
           [("trois lectures", PASS, lambda: _pbs_trois_lectures()),
            ("trois lectures, cos_z décalé", PASS,
             lambda: _pbs_trois_lectures(0.2))],
           [("deux lectures SANS OBJET", FAIL, _pbs_une_seule_lecture)],
           note="Critique 2 du tour de correction : sous la lecture TRONQUÉE, "
                "σ̂±_pool et Ψ changent et la classe Σ bascule. L'arbitrage "
                "est PI et il est EN COURS ⇒ toute classe Σ de ce run est "
                "CONDITIONNELLE.")

    clause("couplage de ε_Ψ à la lecture de cos_p — les DEUX publiés",
           "ε_Ψ recalculée sous la même lecture (A) ET ε_Ψ de la lecture "
           "principale (B), chacune avec sa classe",
           "un seul couplage publié sous un arbitrage OUVERT",
           [("les deux couplages", PASS,
             lambda: _pbs_couplage_epsilon_psi()),
            ("couplages identiques", PASS,
             lambda: _pbs_couplage_epsilon_psi(memes=True))],
           [("un seul couplage", FAIL, _pbs_couplage_un_seul)],
           note="ε_Ψ dépend de σ̂±_pool, donc de la lecture de cos_p. Le §4.4 "
                "ne départage pas les deux couplages ; AUCUN n'est promu. "
                "Conséquence : toute classe Σ de ce run est CONDITIONNELLE.")

    clause("profil_conforme_a_la_loi — opérationnalisation DÉCLARÉE (0-192)",
           "marge, rapport à ε_Ψ(b1) et monotonie COMPLÈTE par bucket publiés",
           "la lecture à deux points déclare « conforme » un profil NON "
           "monotone",
           [("résolution publiée", PASS, _pbs_profil_resolution)],
           [("profil SmolLM2 non monotone", FAIL,
             _pbs_profil_deux_points_sans_resolution)],
           note="conjoint de Σ-epuise : la classe l'exige. ARBITRAGE PI RENDU "
                "le 2026-08-28 — la MONOTONIE COMPLÈTE est la complétion du "
                "conjoint (D30 alinéa 2 : monotone ⇒ deux points est une "
                "implication STRICTE, donc la lecture à deux points rendrait "
                "la prédiction plus facile). La comparaison à deux points "
                "reste publiée, DESCRIPTIVE, et ne classe rien.")

    clause("N_eff / DEFF / ρ_ic — deux routes NOMMÉES, écart publié",
           "route VARIANCE et route LARGEUR D'IC, avec leur écart et sa cause",
           "un N_eff publié sans sa route, non reproductible depuis l'IC",
           [("deux routes", PASS, _pbs_n_eff_deux_routes)],
           [("une seule route", FAIL, _pbs_n_eff_une_seule_route),
            ("DEFF > m̄ ⇒ ρ_ic > 1 : hors domaine", FAIL,
             _pbs_rho_ic_hors_domaine)],
           note="D14-R : un chiffre dont la route n'est pas nommée n'a pas de "
                "provenance. L'écart entre les deux routes MESURE la "
                "non-normalité de la distribution bootstrap à 60 grappes. "
                "Correctif #3 du tour de correction : `ρ_ic` n'a de domaine "
                "que si `DEFF ≤ m̄` ; hors de là il sort SANS OBJET (D23). "
                "`m̄` est le MINIMUM des deux lectures dyadiques, donc `ρ_ic` "
                "publié est un MAJORANT.")

    clause("N-queue — l'une des deux branches du §13.1, jamais aucune (0-191)",
           "erreur d'approximation seuil-vs-rang MESURÉE sur σ̂±_pool, par "
           "bucket, aux deux échelles",
           "l'erreur AFFIRMÉE (« ~10⁻³ ») sans support de calcul",
           [("erreur mesurée sur poids à rang", PASS,
             _pbs_erreur_seuil_vs_rang),
            ("Gram CENTRÉ : cos de paire effondrés", PASS, _pbs_gram_centre)],
           [("erreur affirmée", FAIL, _pbs_erreur_seuil_vs_rang_affirmee),
            ("Gram BRUT : σ̂± forcé à 1", FAIL, _pbs_gram_brut_force_sigma)],
           note="§13.1 verbatim : « simulation exacte OU erreur "
                "d'approximation publiée — l'un des deux, jamais ni l'un ni "
                "l'autre ». La simulation sur états BRUTS rend σ±_sim = 1.0000, "
                "valeur FORCÉE par une identité ((xxv)) : poids probant NUL, "
                "et AUCUNE nulle pour la condition `type`.")

    clause("classe_Sigma — le classificateur des 24 cellules, GARDÉ",
           "les cinq classes de la partition Σ sont atteintes dans l'ordre "
           "gravé, et le conjoint retient Σ-epuise quand il n'est pas satisfait",
           "le conjoint ignoré : la complétion à DEUX POINTS licencie "
           "Σ-epuise là où la complétion PROMUE rend Σ-ind",
           [("cinq branches énumérées", PASS, _pbs_classe_Sigma_cinq_branches)],
           [("conjoint ignoré — profil SmolLM2 non monotone", FAIL,
             _pbs_classe_Sigma_conjoint_ignore)],
           note="trou de mordant du tour de correction : `so.classe_Sigma` "
                "forcée à rendre toujours Σ-epuise laissait E = 0. Le "
                "classificateur qui attribue les 24 classes n'était gardé par "
                "AUCUNE clause mordante. Il l'est ici, sur les deux cas.")

    clause("mu_deux_chemins — la divergence est CALCULÉE, jamais injectée",
           "sur des données régulières les deux chemins concordent sous "
           "TOL_MU",
           "sur une colonne à annulation catastrophique, numpy et math.fsum "
           "divergent RÉELLEMENT ⇒ V-mu doit tomber",
           [("données régulières", PASS, _pbs_v_mu)],
           [("annulation catastrophique 1e16", FAIL,
             _pbs_mu_deux_chemins_divergence_reelle)],
           note="trou de mordant : `so.mu_deux_chemins` forcée à "
                "`chemins_concordants=True` laissait E = 0, parce que le seul "
                "cas échouant existant INJECTAIT le champ au lieu de le faire "
                "produire par la fonction.")

    clause("majorant_de_fuite — recalculé par force brute, et STRICTEMENT > 0",
           "le majorant publié est égal au recalcul indépendant sur toutes "
           "les paires, et il est strictement positif",
           "un majorant forcé à 0.0 satisfait V-mu — seul le recalcul le tue",
           [("recalcul force brute", PASS, _pbs_majorant_de_fuite_recalcule)],
           [("majorant forcé à 0.0", FAIL,
             _pbs_majorant_de_fuite_force_a_zero)],
           note="trou de mordant : `so.majorant_de_fuite` forcé à 0.0 laissait "
                "E = 0, parce que V-mu n'exige qu'un flottant FINI. Un "
                "majorant nul n'est pas un majorant mesuré.")

    clause("V-perimetre — 0 GPU, aucun forward, M jamais instanciée",
           "aucun motif GPU/forward/backprop dans le source du cycle",
           "un motif GPU viole le périmètre gravé",
           [("lecture du source", PASS, _pbs_perimetre)],
           [("motif cuda", FAIL, _pbs_perimetre_gpu)])
    return C


def run_pbs(out_dir: Path = OUT_DIR_PBS) -> dict:
    """Banc de satisfiabilité `pbs`. CPU seul, aucune mesure, aucun GPU."""
    t0 = time.time()
    clauses = build_clauses_pbs()
    rows, E = _evaluer(clauses)
    n_cov = sum(1 for r in rows if r["expected"]["pass_case"]
                and r["expected"]["fail_case"])
    t = so.seuil_t()
    dec, cel = _pbs_materiau()
    viv, _ = _pbs_vivier()
    card = {}
    for (a, b), j in viv.items():
        card.setdefault(so.p4.strate(dec[a], dec[b]), set()).add(int(j.size))
    report = {
        "protocole": "experiments/EXP-2026-08-27-plancher-base-sigma.md",
        "statut_protocole": "PRE-ENREGISTRE (§4 et §6 GELÉS le 2026-08-28 ; "
                            "§14 = déclarations d'opérationnalisation)",
        "banc": "D14-S — satisfiabilité pbs, CPU seul, aucune mesure, aucun GPU",
        "E": int(E), "n_clauses": len(rows),
        "couverture": {"clauses_avec_les_deux_contre_exemples": n_cov,
                       "total": len(rows),
                       "pct": round(100.0 * n_cov / len(rows), 2)},
        "n_cas": sum(len(r["cas"]["pass_case"]) + len(r["cas"]["fail_case"])
                     for r in rows),
        "cas_echouants_obligatoires_du_10": [
            "V-T (support vide)", "V-vide", "V-appar (vivier vide en S2)",
            "V-quad (non exécutée)", "V-ecriture (les deux lectures divergent)",
            "V-grappe (grappe manquante)", "V-rang-def (bucket à 199 trials)",
            "V-ident (quantité forcée non déclarée)", "V-S (DEFF_S > 1)",
            "V-seed (seed = 0 accepté)", "V-loo (inversion)",
            "les trois identités de Q-M10"],
        "cardinaux_enumeres": {
            "U_unites_decisionnelles": len(dec),
            "T_cellules_de_capture": len(cel),
            "cellules_decisionnelles": len(so.MODELES) * len(so.STRATES),
            "vivier_N_etat_par_strate": {k: sorted(v) for k, v in card.items()},
            "attendu_14_2": so.CARDINAUX_VIVIER_ATTENDUS},
        "constantes": {"t": t, "t_formule": "ndtri(1 - 1/256)",
                       "2Q(t)": 2.0 * (1.0 - float(so._Phi(t))),
                       "(phi/Q)^2": so.pente_plackett(t),
                       "pente_Plackett": so.pente_plackett(t) / 2.0,
                       "k": so.K_TOPK, "D": so.D_DG, "K_eff": so.K_EFF,
                       "N_G": so.N_G_SIM, "N_buckets": so.N_BUCKETS,
                       "plancher_trials": so.BUCKET_MIN_TRIALS,
                       "plancher_paires": so.BUCKET_MIN_PAIRES,
                       "B_boot": so.B_BOOT, "tol_PSD": so.TOL_PSD,
                       "tol_sim_quad": so.TOL_SIM_QUAD,
                       "tol_quad_croisee": so.TOL_QUAD_CROISEE},
        "verifications_Q_M10": so.verif_QM10(t),
        "verification_Phi": so._verif_Phi(),
        "clauses_sous_specifiees": UNDERSPEC_PBS,
        "duree_s": round(time.time() - t0, 2),
        "clauses": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_bench_pbs.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-hf", action="store_true",
                    help="repli mot-à-mot au lieu du tokenizer GPT-2")
    ap.add_argument("--suite", default="v3",
                    choices=("v3", "i2", "v4", "dgov", "pbs", "all"),
                    help="v3 = V2-D(a) v3 (défaut, inchangé) ; i2 = layer_profile ; "
                         "v4 = matériau v4 (EXP-2026-08-23-v4-materiel) ; "
                         "dgov = recouvrement des supports "
                         "(EXP-2026-08-23-recouvrement-supports) ; "
                         "pbs = plancher de top64(G·μ_global) et clôture de σ± "
                         "(EXP-2026-08-27-plancher-base-sigma)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.suite in ("pbs", "all"):
        out_p = (Path(args.out) if (args.out and args.suite == "pbs")
                 else OUT_DIR_PBS)
        print("=" * 78)
        print("BANC DE SATISFIABILITÉ (D14-S) — "
              "EXP-2026-08-27-plancher-base-sigma")
        print("AUCUNE MESURE, AUCUN GPU, AUCUN POIDS DE MODÈLE. CPU seul.")
        print("AUCUNE MESURE AVANT E = 0 (§6.A, critère d'abandon).")
        print("=" * 78)
        rep_p = run_pbs(out_dir=out_p)
        for r in rep_p["clauses"]:
            mark = "ok " if not r["compte_dans_E"] else "E !"
            print(f"[{mark}] {r['clause']}")
            for side in ("pass_case", "fail_case"):
                for c in r["cas"][side]:
                    flag = "  " if c["ok"] else "!!"
                    print(f"      {flag} {side:9} {c['cas'][:58]:<58} "
                          f"→ {c['observé']!r}")
            for raison in r["raisons_E"]:
                print(f"      >>> {raison}")
        print("-" * 78)
        print(f"clauses : {rep_p['n_clauses']}  |  cas exécutés : "
              f"{rep_p['n_cas']}  |  couverture : {rep_p['couverture']['pct']} %")
        print(f"cardinaux énumérés : {rep_p['cardinaux_enumeres']}")
        print(f"constantes : {rep_p['constantes']}")
        vq = rep_p["verifications_Q_M10"]
        print(f"Q-M10 : {vq['verdict']} — id1={vq['identite_1_gamma_nuls_P3_egale_P2_sur_128']:.2e} "
              f"id2={vq['identite_2_tous_cosinus_1_rho_chapeau_egale_1']:.2e} "
              f"id3={vq['identite_3_rho_0_gamma_0_P2_egale_1_sur_128_carre']:.2e} "
              f"GL64/GL128={vq['ecart_GL64_GL128']:.2e} "
              f"F2={vq['ecart_F2_nominal_vs_independant']:.2e}")
        for k, v in rep_p["clauses_sous_specifiees"].items():
            print(f"sous-spécifiée : {k} — {v}")
        print("-" * 78)
        print(f"E(pbs) = {rep_p['E']}  |  durée {rep_p['duree_s']} s")
        if rep_p["E"] == 0:
            print("E = 0 — gate de satisfiabilité pbs VERTE.")
        else:
            bad = [r["clause"] for r in rep_p["clauses"] if r["compte_dans_E"]]
            print(f"E ≥ 1 — AUCUNE MESURE. Clauses en cause : {bad}")
            print("Le banc ne corrige AUCUNE clause : amender est une décision "
                  "de pré-enregistrement, pas d'implémentation.")
        print(f"rapport : {out_p / 'gate_bench_pbs.json'}")
        if args.suite == "pbs":
            return 0
        print("=" * 78)

    if args.suite in ("dgov", "all"):
        out_d = (Path(args.out) if (args.out and args.suite == "dgov")
                 else OUT_DIR_DGOV)
        print("=" * 78)
        print("BANC DE SATISFIABILITÉ (D14-S) — "
              "EXP-2026-08-23-recouvrement-supports")
        print("AUCUNE MESURE, AUCUN GPU, AUCUN POIDS DE MODÈLE. CPU seul.")
        print("AUCUNE MESURE AVANT E = 0 (§6.E, critère d'abandon).")
        print("=" * 78)
        rep_d = run_dgov(out_dir=out_d)
        for r in rep_d["clauses"]:
            mark = "ok " if not r["compte_dans_E"] else "E !"
            print(f"[{mark}] {r['clause']}")
            for side in ("pass_case", "fail_case"):
                for c in r["cas"][side]:
                    flag = "  " if c["ok"] else "!!"
                    print(f"      {flag} {side:9} {c['cas'][:60]:<60} "
                          f"→ {c['observé']!r}")
            for raison in r["raisons_E"]:
                print(f"      >>> {raison}")
        print("-" * 78)
        print(f"clauses : {rep_d['n_clauses']}  |  cas exécutés : "
              f"{rep_d['n_cas']}  |  couverture : {rep_d['couverture']['pct']} %")
        print(f"cellules décisionnelles (énumérées) : "
              f"{rep_d['cellules_decisionnelles']['cardinal']}")
        print(f"objets simulés (énumérés)           : "
              f"{rep_d['objets_simules']['cardinal']}")
        print(f"cas échouants obligatoires          : "
              f"{rep_d['cas_echouants_obligatoires']}")
        print(f"constantes : {rep_d['constantes']}")
        print(f"borne décisionnelle : {rep_d['borne_decisionnelle']}")
        print(f"table cum : {rep_d['table_cum']}")
        print(f"Q-M6 : {rep_d['Q-M6']['statut']} — {rep_d['Q-M6']['gravure']}")
        for k, v in rep_d["clauses_sous_specifiees"].items():
            print(f"sous-spécifiée : {k} — {v}")
        print("-" * 78)
        print(f"E(dgov) = {rep_d['E']}  |  durée {rep_d['duree_s']} s")
        if rep_d["E"] == 0:
            print("E = 0 — gate de satisfiabilité dgov VERTE.")
        else:
            bad = [r["clause"] for r in rep_d["clauses"] if r["compte_dans_E"]]
            print(f"E ≥ 1 — AUCUNE MESURE. Clauses en cause : {bad}")
            print("Le banc ne corrige AUCUNE clause : amender est une décision "
                  "de pré-enregistrement, pas d'implémentation.")
        print(f"rapport : {out_d / 'gate_bench_dgov.json'}")
        if args.suite == "dgov":
            return 0
        print("=" * 78)

    if args.suite in ("v4", "all"):
        out_v4 = Path(args.out) if (args.out and args.suite == "v4") else OUT_DIR_V4
        print("=" * 78)
        print("BANC DE SATISFIABILITÉ (D14-S) — EXP-2026-08-23-v4-materiel")
        print("AUCUNE MESURE, AUCUN GPU, AUCUN MODÈLE. CPU seul, 3 tokenizers.")
        print("AUCUN GPU AVANT PASS INTÉGRAL DE CE BANC (§6.A, clause d'abandon).")
        print("=" * 78)
        rep_v4 = run_v4(out_dir=out_v4)
        _imprimer(rep_v4)
        print(f"SHA-256 du matériau : {rep_v4['sha256_materiau']}")
        print(f"cascade gravée   : {rep_v4['cascade_gravee']}")
        print(f"cascade exécutée : {rep_v4['cascade_executee']}")
        print("-" * 78)
        print("TABLE DES GARANTIES D25 (génération)")
        for g in rep_v4["garanties_D25"]:
            print(f"  [{g['verdict']}] {g['porte']:<12} {g['propriete']}")
        print(f"prérequis instrument : {rep_v4['prerequis_instrument']}")
        print(f"contre-exemple fact_pairs : {rep_v4['contre_exemple_fact_pairs']}")
        print(f"espace ORD résolu : {rep_v4['espace_ORD_resolu']}")
        pn = rep_v4["probabilites_sous_la_nulle"]
        print(f"P(classe | nulle) calibrateur : {pn['calibrateur']}")
        print(f"P(classe | nulle) bandes      : {pn['bandes_primaire']}")
        print(f"P(classe | nulle) maillons    : {pn['etats_de_maillon']}")
        print(f"P(classe | nulle) ORD         : {pn['classes_ORD']}")
        for k, v in rep_v4["clauses_sous_specifiees"].items():
            print(f"sous-spécifiée : {k} — {v}")
        print("-" * 78)
        print(f"E(v4) = {rep_v4['E']}")
        if rep_v4["E"] == 0 and not rep_v4["portes_de_generation_en_echec"]:
            print("E = 0 — gate de satisfiabilité v4 VERTE : le GPU est autorisé.")
        else:
            bad = [r["clause"] for r in rep_v4["clauses"] if r["compte_dans_E"]]
            print(f"E >= 1 OU génération en échec — AUCUN RUN. Clauses : {bad} ; "
                  f"portes de génération : "
                  f"{rep_v4['portes_de_generation_en_echec']}")
        print(f"rapport : {out_v4 / 'gate_bench_v4.json'}")
        if args.suite == "v4":
            return 0
        print("=" * 78)

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
