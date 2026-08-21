# SPDX-License-Identifier: AGPL-3.0-or-later
"""V2-D(a) — kNN-LM nu : borne de l'étage des logits.

Protocole pré-enregistré : experiments/EXP-2026-08-21-knn-borne-logits.md

ARCHITECTURE IMPOSÉE (arbitrage n°4 du protocole) : deux étages strictement séparés.

  `--phase gpu`       une passe forward par condition ; elle LOGGE des bruts et ne
                      calcule AUCUNE métrique de la grille : par requête le vecteur
                      `p_LM` COMPLET (log-probas fp32), l'état final (pré-lm_head),
                      l'état de la couche `layer_index`, la cible ; par datastore les
                      clés (fp32) et les tokens-valeurs.
  `--phase analysis`  tout le reste, SANS GPU : la grille λ × T, la permutation des
                      valeurs (P3), le fait croisé (P4), P5/P5b/P5c/P5d/P5e, P6, P7,
                      P8, le bras G, R1–R4, la frontière complète.

`p_LM` ne dépend ni de λ ni de T : c'est ce qui rend le balayage gratuit.

CONTRAINTES NUMÉRIQUES (portes de tests CPU bloquantes, §4 et §7) :
  * distances en **fp32**, calculées comme ‖q − k‖² (jamais ‖q‖²+‖k‖²−2qᵀk : à
    ‖h‖² ~ 10⁴ l'annulation catastrophique efface un quasi-match) ;
  * softmax kNN **après soustraction de d²_min** (sinon underflow silencieux) ;
  * mélange en **log-espace** (`logaddexp`), jamais en probas brutes.

INTERDITS PORTÉS PAR CE SCRIPT (§2, §5, §10) :
  * les clés kNN ne passent JAMAIS par G/DG (`FastWeightMemory.phi`) — ce serait
    détruire la tolérance à la paraphrase que P1 mesure ;
  * le datastore ne se remplit JAMAIS pendant une mesure de logprob (D7/D8) : il est
    `freeze()`é avant la première requête et toute écriture ensuite lève ;
  * aucun backprop, aucun optimiseur, aucune G apprise (D8/D9) ;
  * `knn_lambda` par défaut = 0.0 ⇒ λ=0 doit être bit-exact vs le E1 courant.

Usage :
  python eval/knn_ceiling.py                      # gpu + analysis
  python eval/knn_ceiling.py --phase gpu
  python eval/knn_ceiling.py --phase analysis     # recalcul intégral, zéro GPU
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
# Toutes fixées par le protocole pré-enregistré (§4, §7, §9). Aucune n'est
# ajustable après mesure.

K_NEIGHBORS = 8                       # §7 : k = 8
LAMBDA_STAR = 1.0 - math.exp(-0.05)   # §3 : λ* = 1 − e^(−0.05) = 0.048771
LAMBDA_GRID = [0.02, LAMBDA_STAR, 0.05, 0.10, 0.25]   # §4, balayage hors-ligne
TEMP_C_GRID = [0.3, 1.0, 3.0]         # §4 : T = c·med_j(d²_j − d²_min), PAR BRAS
ARMS = ["final", "inject"]            # bras P6 : état final vs couche layer_index

V0_LAMBDA = 0.25                      # V0 : porte, 1 secret
V0_R3_MIN = 0.9                       # DESCRIPTIF depuis l'arbitrage du PI (ci-dessous)

# ---------------------------------------------------------------------------
# ARBITRAGE DU PI (2026-08-21, en cours de cycle) — résolution d'une incohérence
# INTERNE du protocole, et NON un assouplissement de seuil. Consigné ici et dans
# `analysis.json` sous `V0.PI_arbitration`, comme la clause V5 du cycle Q-01b.
V0_PI_ARBITRATION = (
    "Le seuil R3 ≥ 0.9 de V0 avait été calibré pour une p_kNN one-hot (grille "
    "T ∈ {1, 10}), régime que la consolidation a EXPLICITEMENT écarté comme "
    "dégénéré en le remplaçant par T = c·med_j(d²_j − d²_min) — sans recalibrer "
    "R3. Les deux clauses de V0 ont donc été écrites sous des hypothèses "
    "incompatibles. Décision du PI : (1) V0 se réduit à sa CLAUSE DE "
    "RÉCUPÉRATION — E1-exact top-1 ET R1 = 1 ; (2) R3 devient DESCRIPTIF, "
    "rapporté avec sa caractérisation complète (R3 par c ; c requis pour "
    "R3 = 0.9 ≈ 0.065–0.120, médiane 0.0917), jamais une porte. Aucune autre "
    "prédiction ni clause du protocole n'est modifiée : elles restent scellées."
)
# ---------------------------------------------------------------------------
V1_LAMBDA = 0.10                      # V1 : porte d'intégrité
V1_TARGET = -math.log(0.90)           # = 0.105361
V1_TOL_FRAC = 0.05                    # écart > 5 % ⇒ « ce n'est pas un mélange »

E3_THRESHOLD = 0.05                   # §4 P5 : seuil dur nats/token
P3_EXACTNESS = 0.01                   # §4 P3 : baisse = log(1−λ) à 1 %
P5D_EDGE_TOL = 1e-6                   # §4 P5d(1) : bord droit dur −log(1−λ)
P5D_EDGE_BAND = 0.01                  # « à 1 % du bord » pour compter la masse
P5B_VAR_GATE = 1e-6                   # porte var(D_t) — le protocole (Math Q1)
                                      # laisse le chiffre à l'interprétation ; le
                                      # Builder fixe ici un PLANCHER de dégénérescence
                                      # numérique (var nulle ⇒ corr = 0/0) et rapporte
                                      # la variance observée pour arbitrage.
P5C_ZONE = (1.0, 3.0)                 # §4 P5c : zone non mordante [1,3] nats
P5C_DECILES = 10

TAU_QUANTILE = 0.95                   # bras G : τ = quantile 95 % des d²_min des
                                      # requêtes-FAIT (règle fixée AVANT toute
                                      # lecture de P1/P2 — voir §4 « G » et Math Q3).
G_LAMBDA = 0.25                       # §4 : le bras G est spécifié à λ=0.25

BOOT_B = 10000                        # §7 : bootstrap par secret, 10 000 tirages
BOOT_SEED = 777
PERM_SEED = 12345                     # P3 : permutation des valeurs, déterministe
DISTRACTOR_TOKENS = 30000             # §7 + décision PI §13.4
CHUNK = 1024                          # fenêtre GPT-2 pour la construction du store
RANK_TOP = 10                         # « top-10 »
P1_SUCCESS_FRAC = 2.0 / 3.0           # §4 P1 : ≥ 2/3 des 3 paraphrases


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


def knn_weights(d2_k, temp):
    """softmax(−(d² − d²_min)/T) sur les k voisins.

    La soustraction de d²_min est OBLIGATOIRE : sans elle, exp(−d²/T) avec
    d² ~ 10⁴ underflow à 0 partout et la distribution devient un one-hot invisible
    (ou un NaN par 0/0). Porte de test CPU.
    """
    d2_k = np.asarray(d2_k, dtype=np.float64)
    if temp <= 0:
        raise ValueError("température kNN nulle ou négative")
    z = -(d2_k - d2_k.min()) / float(temp)
    w = np.exp(z)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("softmax kNN dégénéré (underflow)")
    return w / s


def knn_distribution(d2_k, values_k, temp):
    """Distribution kNN agrégée par token : {token: masse}. `values_k` = les k
    tokens-valeurs des voisins retenus."""
    w = knn_weights(d2_k, temp)
    out: dict[int, float] = {}
    for v, wi in zip(np.asarray(values_k).tolist(), w.tolist()):
        out[int(v)] = out.get(int(v), 0.0) + float(wi)
    return out


def knn_entropy(pk):
    """R4 : entropie (nats) de la distribution kNN agrégée par token."""
    return float(-sum(p * math.log(p) for p in pk.values() if p > 0.0))


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
    """ΔNLL_t = −log[(1−λ) + λ·p_kNN(y)/p_LM(y)] ≤ −log(1−λ).
    Calculée en log-espace (jamais de division de probas brutes)."""
    if lam == 0.0:
        return 0.0
    return -(mix_logprob(logp_lm_target, p_knn_target, lam) - float(logp_lm_target))


def mix_rank(logp_lm_full, pk, lam, target):
    """Rang (1-based) de `target` sous le mélange, EXACT sur tout le vocabulaire.
    Recalculé hors GPU depuis le `p_LM` complet loggé + les k couples (d², token)."""
    p = np.exp(np.asarray(logp_lm_full, dtype=np.float64))
    if lam == 0.0:
        pm = p
    else:
        pm = (1.0 - lam) * p
        if pk:
            idx = np.fromiter(pk.keys(), dtype=np.int64)
            val = np.fromiter(pk.values(), dtype=np.float64)
            pm[idx] += lam * val
    return int((pm > pm[int(target)]).sum()) + 1


def permute_values(values, seed=PERM_SEED):
    """P3 : permutation des VALEURS seules. Les clés — donc toutes les distances —
    sont inchangées par construction (porte de test CPU)."""
    v = np.asarray(values).copy()
    rng = np.random.default_rng(seed)
    return v[rng.permutation(len(v))]


def calibrate_temperature(d2_k_list, c):
    """T = c · med_j(d²_j − d²_min) sur les k voisins, **par bras**.

    Choix Builder documenté (le protocole fixe la formule par requête, pas
    l'agrégation) : la médiane intra-requête est d'abord calculée sur les k
    voisins, puis la MÉDIANE de ces valeurs sur les requêtes du bras donne le
    scalaire T du bras. Sans le « par bras », P6 confondrait profondeur d'indice
    et température.
    """
    per_query = []
    for d2 in d2_k_list:
        a = np.asarray(d2, dtype=np.float64)
        per_query.append(float(np.median(a - a.min())))
    base = float(np.median(per_query)) if per_query else 0.0
    return float(c) * base, base, per_query


# ------------------------------------------------------------------ datastore

class Datastore:
    """Datastore kNN NU : clés = états du cortex gelé, valeurs = tokens suivants.

    Deux invariants portés par la classe (tests CPU (v) et (vi)) :
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
        """Lecture seule : (indices, d², tokens-valeurs) des k plus proches voisins.
        `values` permet d'injecter un vecteur de valeurs PERMUTÉES (P3) sans
        toucher aux clés — les distances sont donc rigoureusement identiques."""
        assert self.frozen, "query() sur un datastore non gelé"
        d2 = squared_distances(q, self.keys)
        idx = topk_neighbors(d2, k)
        vals = self.values if values is None else np.asarray(values)
        return idx, d2[idx], vals[idx], d2


# ------------------------------------------------------------- statistiques

def pearson(x, y):
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
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


def bootstrap_mean(values, b=BOOT_B, seed=BOOT_SEED):
    """§7 : bootstrap PAR SECRET (l'unité est le secret, jamais la graine)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {"mean": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "B": b}
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(b, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)), "B": b}


def mean_sd(xs):
    xs = list(xs)
    if not xs:
        return math.nan, math.nan
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    # §7 : variables fixées. Défauts EngramConfig (gpt2, layer 6, λ=2, cap=0.5,
    # η=0.2, decay=1e-3, thr=4, dg=8192/64, gate keysim, prune 512/0.10, seed 0)
    # + le second point de capture (inerte par défaut, armé ici).
    cfg = EngramConfig(capture_final_state=True)
    print(f"[knn_ceiling / V2-D(a) — passe GPU] {cfg.summary()}")
    print(f"  capture_final_state={cfg.capture_final_state} "
          f"knn_lambda(défaut)={cfg.knn_lambda} knn_k={cfg.knn_k}")

    engine = EngramEngine(cfg)
    device = str(engine.cortex.device)
    print(f"  device : {device}")
    if device == "cpu":
        print("  !! ANOMALIE : repli CPU (CUDA indisponible) — à rapporter")
    model = engine.cortex.model
    d_model = engine.cortex.d_model
    L = cfg.layer_index
    tok = engine.tokenizer

    # ---- porte V-cap : hidden_states[-1] EST bien l'entrée de lm_head ----
    with torch.no_grad():
        probe = tok.encode("The password is swordfish.")
        x = torch.tensor([probe], device=engine.cortex.device)
        o = model(input_ids=x, use_cache=False, output_hidden_states=True)
        relog = model.lm_head(o.hidden_states[-1])
        cap_dev = float((relog - o.logits).abs().max())
    print(f"  porte V-cap : |lm_head(hidden_states[-1]) − logits|_max = {cap_dev:.3e}")
    assert cap_dev < 1e-2, "hidden_states[-1] n'est pas l'entrée de lm_head"

    @torch.no_grad()
    def states_of(ids):
        """Un seul forward (sans cache) → (final[T,d], layerL[T,d]) fp32 CPU.
        `final` = sortie de la normalisation finale = entrée de lm_head."""
        x = torch.tensor([ids], device=engine.cortex.device)
        o = model(input_ids=x, use_cache=False, output_hidden_states=True)
        hs = o.hidden_states
        return (hs[-1][0].float().cpu().numpy().astype(np.float32),
                hs[L + 1][0].float().cpu().numpy().astype(np.float32))

    def build_fact_store(text):
        """Traces égalisées (§5.7) : UNE entrée par token du fait, exactement comme
        M sous force_write=True (clé = état à t−1, valeur = token t)."""
        ids = tok.encode(text)
        fin, l6 = states_of(ids)
        keys_f = fin[:-1]
        keys_l = l6[:-1]
        vals = np.asarray(ids[1:], dtype=np.int64)
        return {"ids": np.asarray(ids, dtype=np.int64), "keys_final": keys_f,
                "keys_inject": keys_l, "values": vals}

    @torch.no_grad()
    def score_request(prompt, continuation, *, read):
        """Mesure D7-conforme : clear_context AVANT, écriture coupée, tokenisation
        dérivée par différence (identique à `logprob_continuation`). Logge par
        position : p_LM COMPLET, état final, état couche L, cible, NLL, entropie."""
        ids_prompt = tok.encode(prompt)
        ids_full = tok.encode(prompt + continuation)
        assert ids_full[:len(ids_prompt)] == ids_prompt, "tokenisation non préfixe-stable"
        ids_cont = ids_full[len(ids_prompt):]
        assert ids_cont, "continuation vide"
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

    secrets = SECRETS[:args.secrets]
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {"config": cfg.summary(), "device": device, "d_model": d_model,
                  "layer_index": L, "secrets": secrets,
                  "questions": [q for q, _ in QUESTIONS],
                  "question_prompts": {q: p for q, p in QUESTIONS},
                  "fact_template": FACT_TEMPLATE, "vcap_dev": cap_dev,
                  "capture_final_state": True}

    # ------------------------------------------------ passe A : store fait-seul, M=0
    def pass_fact(prefix, sec_list, *, load_M):
        writes_log = {}
        for si, secret in enumerate(sec_list):
            fact = FACT_TEMPLATE.format(secret=secret)
            store = build_fact_store(fact)
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
            for qi, (qname, qprompt) in enumerate(QUESTIONS):
                r = score_request(qprompt, f" {secret}", read=load_M)
                for key in ("logp", "final", "inject", "target", "nll_base", "H"):
                    arrays[f"{prefix}_q{qi}_{key}_{si}"] = r[key]
            print(f"  [{prefix}] {secret:<12} store={len(store['values'])} entrées "
                  f"writes={w}")
        return writes_log

    print(f"\n--- V0 (porte, 1 secret) — ÉVALUÉE AVANT TOUT LE RESTE ---")
    t0 = time.time()
    pass_fact("A", secrets[:1], load_M=False)
    np.savez(raw / "passA_partial.npz", **arrays)

    # Porte V0 évaluée INLINE avec les primitives hors-ligne : E1-exact top-1 à
    # λ=0.25, R1 = 1, R3 ≥ 0.9. Échec ⇒ run INVALIDE ⇒ STOP (§4, §6).
    st0 = Datastore("V0")
    st0.add(arrays["A_keys_final_0"], arrays["A_values_0"])
    st0.freeze()
    logp0 = arrays["A_q0_logp_0"][0].astype(np.float64)
    tgt0 = int(arrays["A_q0_target_0"][0])
    key0 = arrays["A_q0_final_0"][0]
    d2_all0 = squared_distances(key0, st0.keys)
    d2k_list = []
    for qi in range(len(QUESTIONS)):
        for p in range(len(arrays[f"A_q{qi}_target_0"])):
            kk = arrays[f"A_q{qi}_final_0"][p]
            d2k_list.append(squared_distances(kk, st0.keys)[
                topk_neighbors(squared_distances(kk, st0.keys), K_NEIGHBORS)])
    T0, _, _ = calibrate_temperature(d2k_list, 1.0)
    idx0 = topk_neighbors(d2_all0, K_NEIGHBORS)
    pk0 = knn_distribution(d2_all0[idx0], st0.values[idx0], T0)
    rank0 = mix_rank(logp0, pk0, V0_LAMBDA, tgt0)
    hit0 = np.nonzero(st0.values == tgt0)[0]
    ci0 = int(hit0[0]) if len(hit0) else -1
    r1_0 = int(np.nonzero(np.argsort(d2_all0, kind="stable") == ci0)[0][0]) + 1
    r3_0 = pk0.get(tgt0, 0.0)
    # PORTE = clause de RÉCUPÉRATION seule (arbitrage PI) : top-1 ET R1 = 1.
    # R3 est logué et rapporté, mais ne décide plus.
    v0_ok = bool(rank0 == 1 and r1_0 == 1)
    print(f"  secret {secrets[0]} | λ={V0_LAMBDA} T={T0:.4f} | store {len(st0)} entrées")
    print(f"  PORTE (récupération) : rang sous mélange = {rank0} (attendu 1) | "
          f"R1 = {r1_0} (attendu 1)")
    print(f"  DESCRIPTIF : R3 = {r3_0:.4f} (référence historique {V0_R3_MIN}, "
          f"déclassée par l'arbitrage PI) | R4 = {knn_entropy(pk0):.4f} | d²_min = "
          f"{float(d2_all0.min()):.4f}")
    print(f"  V0 → {'OK' if v0_ok else 'ÉCHEC'} ({time.time() - t0:.1f}s)")
    meta["V0_gpu_inline"] = {"rank": rank0, "R1": r1_0, "R3_descriptive": r3_0,
                             "R4": knn_entropy(pk0), "d2_min": float(d2_all0.min()),
                             "T": T0, "gate_clause": "top-1 ET R1=1 (arbitrage PI)",
                             "R3_reference_declassified": V0_R3_MIN,
                             "PI_arbitration": V0_PI_ARBITRATION, "ok": v0_ok}
    if not v0_ok:
        (raw / "V0_FAILURE.json").write_text(
            json.dumps(meta["V0_gpu_inline"], default=str), encoding="utf-8")
        print("  V0 ÉCHOUÉ ⇒ RUN INVALIDE (§6). STOP — aucune passe supplémentaire.")
        sys.exit(2)

    print(f"\n--- passe A : store fait-seul, M=0, {len(secrets)} secrets ---")
    t0 = time.time()
    meta["writes_A"] = pass_fact("A", secrets, load_M=False)
    meta["duration_A"] = time.time() - t0
    print(f"  passe A : {meta['duration_A']:.1f}s")

    # -------------------------------------------- passe D : E3 par position, M=0
    # Avec M = 0, l'état du cortex sur NEUTRAL_TEXT ne dépend PAS du secret :
    # une seule passe suffit, les 10 stores sont appliqués hors-ligne.
    print(f"\n--- passe D : NEUTRAL_TEXT par position (M=0) ---")
    t0 = time.time()
    engine.reset_memory()
    ids_n = tok.encode(NEUTRAL_TEXT)
    engine.clear_context()
    w0 = engine.memory.write_count
    n_fin, n_inj, n_nll, n_H, n_tgt = [], [], [], [], []
    import torch as _t
    with _t.no_grad():
        for pos, tid in enumerate(ids_n):
            if engine._last_logits is not None:
                lp = _t.log_softmax(engine._last_logits.float(), dim=-1)
                n_fin.append(engine.cortex.last_h_final.cpu().numpy().astype(np.float32))
                n_inj.append(engine.cortex.last_h_pre.cpu().numpy().astype(np.float32))
                n_nll.append(-float(lp[tid]))
                n_H.append(float(-(lp.exp() * lp).sum()))
                n_tgt.append(int(tid))
            engine._consume(tid, read=False, write=False, force_write=False)
    assert engine.memory.write_count == w0, "D7 : write pendant la passe D"
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

    # ---- porte V1 (intégrité du mélange), ÉVALUÉE AVANT les passes B/C/E ----
    print(f"\n--- V1 (porte d'intégrité) — E3 à λ={V1_LAMBDA} vs −log(0.90) ---")
    v1_vals = []
    for si in range(len(secrets)):
        st = Datastore(f"V1-{si}")
        st.add(arrays[f"A_keys_final_{si}"], arrays[f"A_values_{si}"])
        st.freeze()
        d2m = d2_matrix(arrays["D_final"], st.keys)
        d2k_list = [d2m[t][topk_neighbors(d2m[t], K_NEIGHBORS)]
                    for t in range(0, len(n_tgt))]
        # T du bras « final », calibrée sur les requêtes de rappel (§4)
        acc = []
        for sj in range(len(secrets)):
            stj = Datastore("tmp")
            stj.add(arrays[f"A_keys_final_{sj}"], arrays[f"A_values_{sj}"])
            stj.freeze()
            for qi in range(len(QUESTIONS)):
                for p in range(len(arrays[f"A_q{qi}_target_{sj}"])):
                    dd = squared_distances(arrays[f"A_q{qi}_final_{sj}"][p], stj.keys)
                    acc.append(dd[topk_neighbors(dd, K_NEIGHBORS)])
        T_v1, _, _ = calibrate_temperature(acc, 1.0)
        ds = np.asarray([delta_nll_row(d2m[t], st.values, n_nll[t], n_tgt[t],
                                       V1_LAMBDA, T_v1) for t in range(len(n_tgt))])
        v1_vals.append(float(ds.mean()))
    v1_mean = float(np.mean(v1_vals))
    v1_dev = abs(v1_mean - V1_TARGET) / V1_TARGET
    v1_ok = bool(v1_dev <= V1_TOL_FRAC)
    print(f"  E3(NEUTRAL_TEXT, store fait-seul) = {v1_mean:+.6f} nats/token "
          f"(N={len(v1_vals)} secrets) | cible {V1_TARGET:.6f}")
    print(f"  écart relatif = {v1_dev * 100:.2f} % (tolérance {V1_TOL_FRAC * 100:.0f} %) "
          f"→ {'OK' if v1_ok else 'ÉCHEC : ce n’est pas un mélange'}")
    meta["V1_gpu_inline"] = {"measured": v1_mean, "target": V1_TARGET,
                             "rel_dev": v1_dev, "per_secret": v1_vals, "ok": v1_ok}
    if not v1_ok:
        (raw / "V1_FAILURE.json").write_text(
            json.dumps(meta["V1_gpu_inline"], default=str), encoding="utf-8")
        np.savez(raw / "gpu_raw.npz", **arrays)
        (raw / "gpu_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print("  V1 ÉCHOUÉ ⇒ RUN INVALIDE (§6). STOP — passes B/C/E non exécutées.")
        sys.exit(2)

    # ------------------------------- passe B : M au point courant + kNN (descriptif)
    print(f"\n--- passe B : M chargée (point courant) + kNN ---")
    t0 = time.time()
    meta["writes_B"] = pass_fact("B", secrets, load_M=True)
    meta["duration_B"] = time.time() - t0
    print(f"  passe B : {meta['duration_B']:.1f}s")

    # ------------------------------------------------------------ passe C : E1c
    print(f"\n--- passe C : E1c (M=0 et M chargée) ---")
    t0 = time.time()
    store_c = build_fact_store(E1C_FACT)
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
    print(f"  passe C : {meta['duration_C']:.1f}s (writes M chargée = {meta['writes_C_mload']})")

    # ------------------------------------------- passe E : store distracteur 30 k
    if not args.skip_distractor:
        print(f"\n--- passe E : store distracteur {args.distractor_tokens} tokens ---")
        t0 = time.time()
        root = Path(__file__).resolve().parents[1]
        rfc = root / "data" / "rfc9293.txt"
        if not rfc.exists():
            print(f"  !! ANOMALIE : {rfc} absent — bras distracteur NON exécuté")
            meta["distractor"] = None
        else:
            meta["rfc_sha256"] = sha256_file(rfc)          # §7 : SHA-256 à journaliser
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
                                  "sha256": meta["rfc_sha256"]}
            meta["duration_E"] = time.time() - t0
            print(f"  {len(ids_r)} tokens → {len(arrays['E_values'])} entrées, "
                  f"{meta['duration_E']:.1f}s")
    else:
        meta["distractor"] = None

    # ----------------------------------------------- contrôle §5.4 : λ=0 bit-exact
    # Rejoué EN FIN DE RUN (dérive d'état) : E1 exact, M chargée, comparé au
    # `logprob_continuation` de référence sur les mêmes prompts.
    print(f"\n--- contrôle §5.4 : λ=0 bit-exact vs le E1 courant (rejoué en fin de run) ---")
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
    print(f"  E1 exact rejoué (M chargée, λ=0) : {d_mean:+.3f} ± {d_sd:.3f} (N={len(lam0)})")

    meta["duration_gpu_s"] = time.time() - t_start
    np.savez(raw / "gpu_raw.npz", **arrays)
    (raw / "gpu_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1,
                                                  default=str), encoding="utf-8")
    try:
        (raw / "passA_partial.npz").unlink()
    except OSError:
        pass
    print(f"\n  passe GPU : {meta['duration_gpu_s']:.1f}s → {raw / 'gpu_raw.npz'} "
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


def neighbors_for(store, key, k, values=None):
    idx, d2k, valk, d2_all = store.query(key, k=k, values=values)
    return {"idx": idx, "d2": d2k, "vals": valk, "d2_all": d2_all}


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


def delta_nll_row(d2row, values, nll_base, target, lam, temp, k=K_NEIGHBORS,
                  gate_tau=0.0):
    """ΔNLL d'UNE position, à partir d'une ligne de d² déjà calculée."""
    idx = topk_neighbors(d2row, k)
    d2k = np.asarray(d2row, dtype=np.float64)[idx]
    alpha = 1.0 if (gate_tau <= 0.0 or float(d2k.min()) <= gate_tau) else 0.0
    lam_eff = lam * alpha
    if lam_eff <= 0.0:
        return 0.0
    pk = knn_distribution(d2k, np.asarray(values)[idx], temp)
    return mix_delta_nll(-float(nll_base), pk.get(int(target), 0.0), lam_eff)


def _analysis_phase(args, out_dir):
    raw = out_dir / "raw"
    z = np.load(raw / "gpu_raw.npz")
    meta = json.loads((raw / "gpu_meta.json").read_text(encoding="utf-8"))
    secrets = meta["secrets"]
    qnames = meta["questions"]
    n_s = len(secrets)
    res: dict = {"command": "python eval/knn_ceiling.py --phase analysis",
                 "config": meta["config"], "device_gpu_pass": meta["device"],
                 "lambda_star": LAMBDA_STAR, "k": K_NEIGHBORS,
                 "lambda_grid": LAMBDA_GRID, "temp_c_grid": TEMP_C_GRID,
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
        queries[prefix] = []
        for si in range(n_s):
            per_q = []
            for qi, qname in enumerate(qnames):
                pre = f"{prefix}_q{qi}"
                per_q.append(Query(z[f"{pre}_logp_{si}"], z[f"{pre}_final_{si}"],
                                   z[f"{pre}_inject_{si}"], z[f"{pre}_target_{si}"],
                                   z[f"{pre}_nll_base_{si}"], z[f"{pre}_H_{si}"], qname))
            queries[prefix].append(per_q)

    # entrée CORRECTE du store = celle dont la valeur est le 1er token de « secret »
    correct_idx = []
    for si in range(n_s):
        tgt0 = int(queries["A"][si][0].target[0])
        vals = z[f"A_values_{si}"]
        hit = np.nonzero(vals == tgt0)[0]
        correct_idx.append(int(hit[0]) if len(hit) else -1)
    res["correct_entry_index"] = correct_idx
    assert all(i >= 0 for i in correct_idx), "entrée correcte introuvable dans le store"

    def key_of(q, arm, p=0):
        return (q.final if arm == "final" else q.inject)[p]

    # ------------------------------------------------- T calibrée PAR BRAS (§4/§7)
    temps = {}
    for arm in ARMS:
        d2_list = []
        for si in range(n_s):
            for qi in range(len(qnames)):
                q = queries["A"][si][qi]
                for p in range(len(q.target)):
                    nb = neighbors_for(stores[arm][si], key_of(q, arm, p), K_NEIGHBORS)
                    d2_list.append(nb["d2"])
        for c in TEMP_C_GRID:
            T, base, _ = calibrate_temperature(d2_list, c)
            temps[(arm, c)] = T
        res.setdefault("temperature", {})[arm] = {
            "median_spread_over_queries": base, "n_queries": len(d2_list),
            "T_by_c": {str(c): temps[(arm, c)] for c in TEMP_C_GRID}}
    print("\n================ T calibrée PAR BRAS ================")
    for arm in ARMS:
        r = res["temperature"][arm]
        print(f"  bras {arm:<7} med_q med_j(d²_j − d²_min) = {r['median_spread_over_queries']:.4f} "
              f"| T = " + " ".join(f"c{c}→{temps[(arm, c)]:.4f}" for c in TEMP_C_GRID))

    C_MAIN = 1.0     # c principal ; c=0.3 et c=3 en balayage (§4)

    # ------------------------------------------------------------------ helpers
    def eval_query(store, q, arm, lam, T, p=0, values=None, k=K_NEIGHBORS,
                   gate_tau=0.0):
        """Recalcul hors-ligne complet d'une requête sous (λ, T)."""
        nb = neighbors_for(store, key_of(q, arm, p), k, values=values)
        d2min = float(nb["d2"].min())
        alpha = 1.0 if (gate_tau <= 0.0 or d2min <= gate_tau) else 0.0
        lam_eff = lam * alpha
        pk = knn_distribution(nb["d2"], nb["vals"], T) if lam_eff > 0 else {}
        tgt = int(q.target[p])
        lp_lm = float(q.logp[p][tgt])
        lp_mix = mix_logprob(lp_lm, pk.get(tgt, 0.0), lam_eff)
        rank = mix_rank(q.logp[p], pk, lam_eff, tgt)
        rank_base = mix_rank(q.logp[p], {}, 0.0, tgt)
        return {"nb": nb, "pk": pk, "d2min": d2min, "alpha": alpha,
                "lam_eff": lam_eff, "target": tgt, "logp_lm": lp_lm,
                "logp_mix": lp_mix, "delta_logp": lp_mix - lp_lm,
                "rank": rank, "rank_base": rank_base,
                "p_knn_target": pk.get(tgt, 0.0), "H_knn": knn_entropy(pk) if pk else 0.0}

    def full_logp(store, q, arm, lam, T, values=None, k=K_NEIGHBORS, gate_tau=0.0):
        """Δlogp de la continuation ENTIÈRE (somme sur les positions)."""
        tot_mix = tot_lm = 0.0
        for p in range(len(q.target)):
            e = eval_query(store, q, arm, lam, T, p=p, values=values, k=k,
                           gate_tau=gate_tau)
            tot_mix += e["logp_mix"]
            tot_lm += e["logp_lm"]
        return tot_mix, tot_lm

    def instrumentation(store, q, arm, si, T, p=0):
        """R1–R4 (§4, Neuro B) : rang de l'entrée correcte, d²/cos à la clé stockée,
        masse p_kNN(cible), entropie de p_kNN."""
        key = key_of(q, arm, p)
        d2 = squared_distances(key, store.keys)
        order = np.argsort(d2, kind="stable")
        ci = correct_idx[si]
        r1 = int(np.nonzero(order == ci)[0][0]) + 1
        ck = store.keys[ci].astype(np.float64)
        kq = np.asarray(key, dtype=np.float64)
        cos = float(ck @ kq / (np.linalg.norm(ck) * np.linalg.norm(kq) + 1e-12))
        return {"R1_rank_correct_entry": r1, "R1_in_topk": bool(r1 <= K_NEIGHBORS),
                "R2_d2_to_correct": float(d2[ci]), "R2_cos_to_correct": cos,
                "R2_d2_min": float(d2.min())}

    # =================================================================== V0
    print("\n================ V0 (porte, 1 secret) ================")
    T0 = temps[("final", C_MAIN)]
    e0 = eval_query(stores["final"][0], queries["A"][0][0], "final", V0_LAMBDA, T0)
    i0 = instrumentation(stores["final"][0], queries["A"][0][0], "final", 0, T0)
    # R3 DESCRIPTIF sur les 10 secrets et toute la grille c (caractérisation
    # demandée par l'arbitrage du PI).
    r3_desc = {}
    for c in TEMP_C_GRID:
        Tc = temps[("final", c)]
        r3_desc[str(c)] = [eval_query(stores["final"][si], queries["A"][si][0],
                                      "final", V0_LAMBDA, Tc)["p_knn_target"]
                           for si in range(n_s)]
    c_needed = []
    for si in range(n_s):
        q = queries["A"][si][0]
        nb = neighbors_for(stores["final"][si], key_of(q, "final", 0), K_NEIGHBORS)
        med = float(np.median(np.asarray(nb["d2"], dtype=np.float64) - nb["d2"].min()))
        tgt = int(q.target[0])
        lo, hi = 1e-6, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if knn_distribution(nb["d2"], nb["vals"], mid * med).get(tgt, 0.0) >= V0_R3_MIN:
                lo = mid
            else:
                hi = mid
        c_needed.append(lo)
    v0 = {"secret": secrets[0], "lambda": V0_LAMBDA, "T": T0,
          "gate_clause": "récupération seule : E1-exact top-1 ET R1 = 1 (arbitrage PI)",
          "PI_arbitration": V0_PI_ARBITRATION,
          "rank_mix": e0["rank"], "rank_base": e0["rank_base"],
          "R1": i0["R1_rank_correct_entry"], "R2_d2_min": i0["R2_d2_min"],
          "R3_descriptive": e0["p_knn_target"],
          "R3_reference_declassified": V0_R3_MIN,
          "R4_H_knn": e0["H_knn"], "delta_logp": e0["delta_logp"],
          "R3_by_c_10secrets": {c: {"mean": float(np.mean(v)), "max": float(np.max(v)),
                                    "n_ge_0.9": int(sum(x >= V0_R3_MIN for x in v)),
                                    "per_secret": v}
                                for c, v in r3_desc.items()},
          "c_needed_for_R3_0.9": {"per_secret": c_needed,
                                  "median": float(np.median(c_needed)),
                                  "min": float(np.min(c_needed)),
                                  "max": float(np.max(c_needed))},
          "R1_exact_eq1_count": int(sum(
              1 for si in range(n_s)
              if instrumentation(stores["final"][si], queries["A"][si][0], "final",
                                 si, T0)["R1_rank_correct_entry"] == 1)),
          "ok": bool(e0["rank"] == 1 and i0["R1_rank_correct_entry"] == 1)}
    print(f"  secret {secrets[0]} | λ={V0_LAMBDA} T={T0:.4f}")
    print(f"  PORTE (récupération, arbitrage PI) : E1-exact rang {e0['rank_base']} → "
          f"{e0['rank']} (attendu 1) | R1 = {v0['R1']} (attendu 1) | "
          f"R1=1 sur {v0['R1_exact_eq1_count']}/{n_s} secrets")
    print(f"  DESCRIPTIF : R3 = {e0['p_knn_target']:.4f} | R4 = {e0['H_knn']:.4f} | "
          f"d²_min = {i0['R2_d2_min']:.4f} | Δlogp = {e0['delta_logp']:+.3f}")
    for c, v in v0["R3_by_c_10secrets"].items():
        print(f"      R3 (10 secrets, c={c}) : moy {v['mean']:.4f} max {v['max']:.4f} "
              f"| ≥ {V0_R3_MIN} : {v['n_ge_0.9']}/{n_s}")
    cn = v0["c_needed_for_R3_0.9"]
    print(f"      c requis pour R3 = {V0_R3_MIN} : médiane {cn['median']:.4f} "
          f"[{cn['min']:.4f}, {cn['max']:.4f}]")
    print(f"  V0 → {'OK' if v0['ok'] else 'ÉCHEC ⇒ RUN INVALIDE'}")
    print(f"  ARBITRAGE PI consigné : {V0_PI_ARBITRATION}")
    res["V0"] = v0
    if not v0["ok"]:
        _dump(out_dir, res, csv_rows)
        print("\n  V0 ÉCHOUÉ : run INVALIDE (§6). STOP.")
        sys.exit(2)

    # =================================================================== V1
    print("\n================ V1 (porte d'intégrité du mélange) ================")
    D_nll = z["D_nll_base"].astype(np.float64)
    D_H = z["D_H"].astype(np.float64)
    D_tgt = z["D_target"]
    D_key = {"final": z["D_final"], "inject": z["D_inject"]}

    # Les d² NEUTRAL_TEXT × store ne dépendent NI de λ NI de T : calculés une fois,
    # toute la grille se recalcule ensuite en quelques millisecondes.
    t_d2 = time.time()
    d2_neutral = {si: d2_matrix(D_key["final"], z[f"A_keys_final_{si}"])
                  for si in range(n_s)}
    print(f"  d² NEUTRAL_TEXT × store fait-seul : {len(D_tgt)}×"
          f"{d2_neutral[0].shape[1]} × {n_s} secrets en {time.time() - t_d2:.1f}s")

    def e3_series(d2mat, values, lam, T, k=K_NEIGHBORS, gate_tau=0.0):
        """ΔNLL par position sur NEUTRAL_TEXT. Retourne le vecteur D_t."""
        out = np.empty(len(D_tgt), dtype=np.float64)
        for t in range(len(D_tgt)):
            out[t] = delta_nll_row(d2mat[t], values, D_nll[t], int(D_tgt[t]),
                                   lam, T, k=k, gate_tau=gate_tau)
        return out

    T_final = temps[("final", C_MAIN)]
    v1_per_secret = []
    D_v1 = {}
    for si in range(n_s):
        d = e3_series(d2_neutral[si], z[f"A_values_{si}"], V1_LAMBDA, T_final)
        D_v1[si] = d
        v1_per_secret.append(float(d.mean()))
    v1_mean = float(np.mean(v1_per_secret))
    v1_dev = abs(v1_mean - V1_TARGET) / V1_TARGET
    # diagnostic d'attribution (§5.9) : part des positions dont la cible est un
    # token-valeur du store — ce sont elles qui écartent E3 de −log(1−λ).
    common = []
    for si in range(n_s):
        vs = set(z[f"A_values_{si}"].tolist())
        common.append(float(np.mean([int(t) in vs for t in D_tgt.tolist()])))
    v1 = {"lambda": V1_LAMBDA, "target": V1_TARGET, "measured_mean": v1_mean,
          "per_secret": v1_per_secret, "rel_dev": v1_dev, "tol_frac": V1_TOL_FRAC,
          "ok": bool(v1_dev <= V1_TOL_FRAC),
          "frac_positions_target_in_store_values": common,
          "frac_positions_target_in_store_values_mean": float(np.mean(common))}
    print(f"  E3(NEUTRAL_TEXT, store fait-seul, λ={V1_LAMBDA}) = {v1_mean:+.6f} "
          f"nats/token (N={n_s} secrets)")
    print(f"  cible −log(0.90) = {V1_TARGET:.6f} | écart relatif {v1_dev * 100:.2f} % "
          f"(tolérance {V1_TOL_FRAC * 100:.0f} %)")
    print(f"  diagnostic §5.9 : part des positions dont la cible est un token-valeur "
          f"du store = {v1['frac_positions_target_in_store_values_mean']:.4f}")
    print(f"  V1 → {'OK' if v1['ok'] else 'ÉCHEC ⇒ ce n’est pas un mélange ⇒ RUN INVALIDE'}")
    res["V1"] = v1
    if not v1["ok"]:
        _dump(out_dir, res, csv_rows)
        print("\n  V1 ÉCHOUÉ : run INVALIDE (§6). STOP.")
        sys.exit(2)

    # ============================================ τ du bras G — FIXÉ AVANT P1/P2
    print("\n================ τ du bras G (fixé AVANT toute lecture de P1/P2) ================")
    d2min_fact, d2min_neutral = [], []
    for si in range(n_s):
        for qi in range(len(qnames)):
            q = queries["A"][si][qi]
            nb = neighbors_for(stores["final"][si], key_of(q, "final", 0), K_NEIGHBORS)
            d2min_fact.append(float(nb["d2"].min()))
    for si in range(n_s):
        d2min_neutral.extend(d2_neutral[si].min(axis=1).astype(float).tolist())
    tau = float(np.quantile(np.asarray(d2min_fact), TAU_QUANTILE))
    fpr = float(np.mean(np.asarray(d2min_neutral) <= tau))
    res["tau_gate"] = {"rule": f"quantile {TAU_QUANTILE} des d²_min des requêtes-FAIT",
                       "tau": tau, "n_fact": len(d2min_fact),
                       "n_neutral_sampled": len(d2min_neutral),
                       "d2min_fact": {"median": float(np.median(d2min_fact)),
                                      "min": float(np.min(d2min_fact)),
                                      "max": float(np.max(d2min_fact))},
                       "d2min_neutral": {"median": float(np.median(d2min_neutral)),
                                         "min": float(np.min(d2min_neutral)),
                                         "max": float(np.max(d2min_neutral))},
                       "neutral_pass_rate_at_tau": fpr}
    print(f"  d²_min requêtes-FAIT   : médiane {np.median(d2min_fact):.1f} "
          f"[{np.min(d2min_fact):.1f}, {np.max(d2min_fact):.1f}] (N={len(d2min_fact)})")
    print(f"  d²_min requêtes-NEUTRE : médiane {np.median(d2min_neutral):.1f} "
          f"[{np.min(d2min_neutral):.1f}, {np.max(d2min_neutral):.1f}] "
          f"(N={len(d2min_neutral)})")
    print(f"  τ = quantile {TAU_QUANTILE} des FAIT = {tau:.2f} | "
          f"taux de passage des positions neutres à τ = {fpr:.4f}")

    # =================================================================== P1
    print("\n================ P1 (DÉCISIONNELLE) ================")

    def p1_table(store_list, arm, lam, T, prefix="A", k=K_NEIGHBORS, gate_tau=0.0,
                 values_fn=None, cross=False):
        """Retourne (n_secrets_succès, cas 30 descriptifs, détail par secret)."""
        per_secret, cases = [], []
        for si in range(n_s):
            st = store_list[(si + 1) % n_s] if cross else store_list[si]
            sidx = (si + 1) % n_s if cross else si
            vals = values_fn(sidx) if values_fn else None
            hits, det = 0, []
            for qi in range(1, len(qnames)):     # les 3 PARAPHRASES
                q = queries[prefix][si][qi]
                e = eval_query(st, q, arm, lam, T, p=0, values=vals, k=k,
                               gate_tau=gate_tau)
                ins = instrumentation(st, q, arm, sidx if not cross else sidx, T)
                top = e["rank"] <= RANK_TOP
                hits += int(top)
                rec = {"secret": secrets[si], "question": qnames[qi],
                       "rank_base": e["rank_base"], "rank_mix": e["rank"],
                       "top10": bool(top), "delta_logp": e["delta_logp"],
                       "R1": ins["R1_rank_correct_entry"],
                       "R2_d2_correct": ins["R2_d2_to_correct"],
                       "R2_cos_correct": ins["R2_cos_to_correct"],
                       "R2_d2_min": e["d2min"], "R3": e["p_knn_target"],
                       "R4": e["H_knn"], "alpha": e["alpha"]}
                det.append(rec)
                cases.append(rec)
            per_secret.append({"secret": secrets[si], "hits": hits,
                               "success": hits >= math.ceil(P1_SUCCESS_FRAC * 3),
                               "detail": det})
        n = sum(1 for s in per_secret if s["success"])
        return n, cases, per_secret

    p1 = {}
    for arm in ARMS:
        for c in TEMP_C_GRID:
            T = temps[(arm, c)]
            for lam in LAMBDA_GRID:
                n, cases, per = p1_table(stores[arm], arm, lam, T)
                key = f"{arm}|c={c}|lam={lam:.6f}"
                p1[key] = {"arm": arm, "c": c, "lambda": lam, "T": T,
                           "n_secrets": n, "n_cases_top10": sum(cc["top10"] for cc in cases),
                           "n_cases": len(cases),
                           "sign_test_p": sign_test_exact(n, n_s),
                           "per_secret": [{k: v for k, v in s.items() if k != "detail"}
                                          for s in per],
                           "cases": cases if (arm == "final" and c == C_MAIN) else None}
                csv_rows.append(["P1", key, "n_secrets_top10", n, "", n_s])
    res["P1_grid"] = {k: {kk: vv for kk, vv in v.items() if kk != "cases"}
                      for k, v in p1.items()}
    main_key = f"final|c={C_MAIN}|lam={LAMBDA_STAR:.6f}"
    res["P1_main"] = p1[main_key]
    print(f"  DÉCISIONNELLE — bras final, c={C_MAIN}, λ* = {LAMBDA_STAR:.6f}")
    m = p1[main_key]
    print(f"  n (secrets avec ≥ 2/3 paraphrases en top-10) = {m['n_secrets']}/{n_s}  "
          f"(H vraie ≥ 5 ; H fausse ≤ 1 ; 2–4 = zone grise)")
    print(f"  test des signes exact : p = {m['sign_test_p']:.4g}")
    print(f"  P1-desc (les 30 cas) : {m['n_cases_top10']}/{m['n_cases']} en top-10 "
          f"(descriptif ; H vraie ≥ 15 ; H fausse ≤ 5)")
    print("\n  les 30 cas (bras final, c=1, λ*) :")
    print("    secret       question  rang_base → rang_mix  top10  Δlogp     R1  R2_d²  R2_cos  R3      R4")
    for cc in m["cases"]:
        print(f"    {cc['secret']:<12} {cc['question']:<8} {cc['rank_base']:>8} → "
              f"{cc['rank_mix']:<8} {str(cc['top10']):<6} {cc['delta_logp']:+7.4f}  "
              f"{cc['R1']:>3}  {cc['R2_d2_correct']:7.1f} {cc['R2_cos_correct']:+.4f} "
              f"{cc['R3']:.5f} {cc['R4']:.4f}")
    print("\n  P1 sur la grille λ × T × bras :")
    print("    bras     c     λ         n/10  cas/30  p(signes)")
    for k, v in p1.items():
        print(f"    {v['arm']:<8} {v['c']:<5} {v['lambda']:.6f}  {v['n_secrets']:>4}  "
              f"{v['n_cases_top10']:>6}  {v['sign_test_p']:.4g}")

    # ANTIPODE P1 : succès avec R1 > 1 dans la majorité des succès
    succ = [cc for cc in m["cases"] if cc["top10"]]
    res["P1_antipode"] = {
        "n_success_cases": len(succ),
        "n_success_with_R1_gt_1": sum(1 for cc in succ if cc["R1"] > 1),
        "majority_R1_gt_1": bool(succ and sum(1 for cc in succ if cc["R1"] > 1) > len(succ) / 2)}

    # =================================================================== P2
    print("\n================ P2 (frontière E1c) ================")
    store_c = {}
    for arm in ARMS:
        s = Datastore(f"C-{arm}")
        s.add(z[f"C_keys_{arm}"], z["C_values"])
        store_c[arm] = s.freeze()
    p2 = {}
    for tag in ("m0", "mload"):
        qm = Query(z[f"C_{tag}_w0_logp"], z[f"C_{tag}_w0_final"], z[f"C_{tag}_w0_inject"],
                   z[f"C_{tag}_w0_target"], z[f"C_{tag}_w0_nll_base"],
                   z[f"C_{tag}_w0_H"], "Marseille")
        qp = Query(z[f"C_{tag}_w1_logp"], z[f"C_{tag}_w1_final"], z[f"C_{tag}_w1_inject"],
                   z[f"C_{tag}_w1_target"], z[f"C_{tag}_w1_nll_base"],
                   z[f"C_{tag}_w1_H"], "Paris")
        for arm in ARMS:
            T = temps[(arm, C_MAIN)]
            grid = np.concatenate([np.linspace(0.0, 0.999, 1000), np.asarray(LAMBDA_GRID)])
            grid = np.unique(np.round(grid, 6))
            lam_renv = None
            per_lam = []
            for lam in grid.tolist():
                em = eval_query(store_c[arm], qm, arm, lam, T)
                per_lam.append((lam, em["rank"]))
                if lam_renv is None and em["rank"] == 1:
                    lam_renv = lam
            at_star = eval_query(store_c[arm], qm, arm, LAMBDA_STAR, T)
            at_star_p = eval_query(store_c[arm], qp, arm, LAMBDA_STAR, T)
            entry = {"tag": tag, "arm": arm, "T": T, "lambda_renv": lam_renv,
                     "F": (lam_renv / LAMBDA_STAR) if lam_renv is not None else None,
                     "rank_marseille_base": at_star["rank_base"],
                     "rank_marseille_at_lambda_star": at_star["rank"],
                     "rank_paris_at_lambda_star": at_star_p["rank"],
                     "delta_marseille_at_lambda_star": at_star["delta_logp"],
                     "delta_paris_at_lambda_star": at_star_p["delta_logp"],
                     "R3_marseille_at_lambda_star": at_star["p_knn_target"],
                     "R4_at_lambda_star": at_star["H_knn"],
                     "d2min": at_star["d2min"],
                     "ranks_on_grid": {f"{lam:.6f}": eval_query(store_c[arm], qm, arm,
                                                               lam, T)["rank"]
                                       for lam in LAMBDA_GRID}}
            p2[f"{tag}|{arm}"] = entry
            print(f"  [{tag} / bras {arm}] λ_renv = "
                  f"{('%.6f' % lam_renv) if lam_renv is not None else 'jamais (≤0.999)'} | "
                  f"F = {('%.2f' % entry['F']) if entry['F'] is not None else 'n/a'} "
                  f"(F ≤ 2 si H vraie ; F ≥ 5 si H fausse)")
            print(f"      rang Marseille : base {at_star['rank_base']} → "
                  f"{at_star['rank']} à λ* | rang Paris à λ* : {at_star_p['rank']} | "
                  f"ΔMars {at_star['delta_logp']:+.3f} ΔParis {at_star_p['delta_logp']:+.3f}")
    res["P2"] = p2

    # =================================================================== P3
    print("\n================ P3 (contrôle BLOQUANT : valeurs permutées) ================")
    perm_vals = {si: permute_values(z[f"A_values_{si}"], PERM_SEED + si)
                 for si in range(n_s)}
    # intégrité mécanique : mêmes clés ⇒ distances inchangées
    q0 = queries["A"][0][0]
    nb_a = neighbors_for(stores["final"][0], key_of(q0, "final", 0), K_NEIGHBORS)
    nb_b = neighbors_for(stores["final"][0], key_of(q0, "final", 0), K_NEIGHBORS,
                         values=perm_vals[0])
    dist_identical = bool(np.array_equal(nb_a["d2"], nb_b["d2"]))
    T = temps[("final", C_MAIN)]
    p3 = {"distances_identical": dist_identical, "per_lambda": {}}
    for lam in LAMBDA_GRID:
        n, cases, per = p1_table(stores["final"], "final", lam, T,
                                 values_fn=lambda si: perm_vals[si])
        drops, exactness = [], []
        for si in range(n_s):
            for qi in range(len(qnames)):
                q = queries["A"][si][qi]
                e = eval_query(stores["final"][si], q, "final", lam, T, p=0,
                               values=perm_vals[si])
                drops.append(e["delta_logp"])
                exactness.append(abs(e["delta_logp"] - math.log1p(-lam)))
        expected = math.log1p(-lam)
        p3["per_lambda"][f"{lam:.6f}"] = {
            "n_secrets_top10": n, "n_cases_top10": sum(cc["top10"] for cc in cases),
            "mean_delta_logp": float(np.mean(drops)),
            "expected_log1m_lambda": expected,
            "max_abs_dev": float(np.max(exactness)),
            "rel_dev": float(np.max(exactness) / abs(expected)) if expected else math.nan,
            "exact_within_1pct": bool(np.max(exactness) <= P3_EXACTNESS * abs(expected))}
        e = p3["per_lambda"][f"{lam:.6f}"]
        print(f"  λ={lam:.6f} | top-10 {n}/{n_s} secrets, {e['n_cases_top10']}/30 cas | "
              f"Δlogp moyen {e['mean_delta_logp']:+.6f} vs log(1−λ) = {expected:+.6f} | "
              f"écart max {e['max_abs_dev']:.2e} ({e['rel_dev'] * 100:.3f} %) → "
              f"{'exact à 1 %' if e['exact_within_1pct'] else 'HORS 1 %'}")
    star = p3["per_lambda"][f"{LAMBDA_STAR:.6f}"]
    p3["blocking_ok"] = bool(dist_identical and star["n_secrets_top10"] <= 1
                             and star["exact_within_1pct"])
    print(f"  distances inchangées sous permutation : {dist_identical}")
    print(f"  P3 (à λ*) → {'OK' if p3['blocking_ok'] else 'ÉCHEC ⇒ P1/P2 ININTERPRÉTABLES ⇒ INCONCLUSIF'}")
    res["P3"] = p3

    # =================================================================== P4
    print("\n================ P4 (spécificité : fait croisé, hors-ligne) ================")
    p4 = {}
    for lam in LAMBDA_GRID:
        n, cases, per = p1_table(stores["final"], "final", lam, T, cross=True)
        # top-10 sur la question EXACTE aussi (le cas le plus favorable au store)
        exact_hits = 0
        for si in range(n_s):
            e = eval_query(stores["final"][(si + 1) % n_s], queries["A"][si][0],
                           "final", lam, T)
            exact_hits += int(e["rank"] <= RANK_TOP)
        p4[f"{lam:.6f}"] = {"n_secrets_top10_paraphrases": n,
                            "n_cases_top10": sum(cc["top10"] for cc in cases),
                            "n_exact_top10": exact_hits}
        print(f"  λ={lam:.6f} | store A / prompt B : {n}/{n_s} secrets (paraphrases), "
              f"{sum(cc['top10'] for cc in cases)}/30 cas, exact {exact_hits}/{n_s} "
              f"(critère : ≤ 1/10)")
    res["P4"] = p4

    # =============================================== stores distracteurs (P5/P7)
    dist_store = None
    if "E_keys_final" in z.files:
        dist_store = {}
        for si in range(n_s):
            s = Datastore(f"distractor+fact{si}")
            s.add(z["E_keys_final"], z["E_values"])
            s.add(z[f"A_keys_final_{si}"], z[f"A_values_{si}"])
            dist_store[si] = s.freeze()
        # l'entrée correcte est décalée du nombre d'entrées distractrices
        offset = int(len(z["E_values"]))
        print(f"\n  store distracteur : {offset} entrées + {len(z['A_values_0'])} du fait")
    else:
        offset = 0
        print("\n  !! store distracteur absent : P7 et le bras distracteur de P5 non exécutés")

    # =================================================================== P5
    print("\n================ P5 / P5b / P5c / P5d / P5e (D11, D12) ================")
    H_med = float(np.median(D_H))
    p5 = {"H_median": H_med, "n_positions": int(len(D_tgt)), "conditions": {}}

    def p5_block(label, series_by_secret, lam):
        Dm = np.mean(np.stack(series_by_secret), axis=0)
        edge = -math.log1p(-lam)
        entry = {"lambda": lam, "edge_-log(1-lambda)": edge,
                 "E3_mean_per_secret": [float(s.mean()) for s in series_by_secret],
                 "E3_mean": float(np.mean([s.mean() for s in series_by_secret])),
                 "E3_max_secret": float(np.max([s.mean() for s in series_by_secret])),
                 "above_threshold_0.05": bool(np.mean([s.mean() for s in series_by_secret])
                                              > E3_THRESHOLD)}
        entry["boot_by_secret"] = bootstrap_mean([float(s.mean()) for s in series_by_secret])
        # P5d(1) : bord droit dur
        mx = float(np.max([s.max() for s in series_by_secret]))
        entry["P5d_max_delta"] = mx
        entry["P5d_hard_edge_ok"] = bool(mx <= edge * (1.0 + P5D_EDGE_TOL) + 1e-12)
        # P5d(2) : masse au bord, confiantes vs incertaines
        conf = D_H <= H_med
        at_edge = np.stack(series_by_secret) >= edge * (1.0 - P5D_EDGE_BAND)
        entry["P5d_frac_at_edge_all"] = float(at_edge.mean())
        entry["P5d_frac_at_edge_confident"] = float(at_edge[:, conf].mean())
        entry["P5d_frac_at_edge_uncertain"] = float(at_edge[:, ~conf].mean())
        allv = np.concatenate(series_by_secret)
        lo = float(min(0.0, allv.min()))
        hist, hedges = np.histogram(allv, bins=20, range=(lo, edge))
        entry["P5d_hist"] = hist.tolist()
        entry["P5d_hist_edges"] = [float(x) for x in hedges]
        entry["P5d_min_delta"] = float(allv.min())
        entry["damage_confident"] = float(Dm[conf].mean())
        entry["damage_uncertain"] = float(Dm[~conf].mean())
        # P5b : corr(D, H) par secret, sous porte var(D_t)
        corrs, vars_ = [], []
        for s in series_by_secret:
            vars_.append(float(np.var(s)))
            corrs.append(pearson(s.tolist(), D_H.tolist()))
        entry["P5b_var_D_mean"] = float(np.mean(vars_))
        entry["P5b_gate_ok"] = bool(np.mean(vars_) > P5B_VAR_GATE)
        entry["P5b_corr_D_H_per_secret"] = corrs
        entry["P5b_corr_D_H_mean"], entry["P5b_corr_D_H_sd"] = mean_sd(corrs)
        entry["P5b_spearman_mean"] = float(np.mean(
            [spearman(s.tolist(), D_H.tolist()) for s in series_by_secret]))
        # P5c : appariement par décile de NLL_base + zone [1,3] nats
        dec = np.clip(np.digitize(D_nll, np.quantile(D_nll,
                      np.linspace(0, 1, P5C_DECILES + 1)[1:-1])), 0, P5C_DECILES - 1)
        zone = (D_nll >= P5C_ZONE[0]) & (D_nll <= P5C_ZONE[1])
        corrs_c = []
        for s in series_by_secret:
            resid_D, resid_H = [], []
            for d in range(P5C_DECILES):
                m_ = (dec == d) & zone
                if m_.sum() < 3:
                    continue
                resid_D.extend((s[m_] - s[m_].mean()).tolist())
                resid_H.extend((D_H[m_] - D_H[m_].mean()).tolist())
            corrs_c.append(pearson(resid_D, resid_H) if len(resid_D) > 3 else math.nan)
        ok_c = [c for c in corrs_c if not math.isnan(c)]
        entry["P5c_n_zone"] = int(zone.sum())
        entry["P5c_corr_matched_mean"], entry["P5c_corr_matched_sd"] = mean_sd(ok_c)
        entry["P5c_below_0.15"] = bool(abs(entry["P5c_corr_matched_mean"]) < 0.15)
        entry["P5c_antipode_ge_0.30"] = bool(entry["P5c_corr_matched_mean"] >= 0.30)
        return entry

    if dist_store is not None:
        t_d2 = time.time()
        d2_neutral_dist = d2_matrix(D_key["final"], z["E_keys_final"])
        d2_comb = {si: np.concatenate([d2_neutral_dist, d2_neutral[si]], axis=1)
                   for si in range(n_s)}
        vals_comb = {si: np.concatenate([z["E_values"], z[f"A_values_{si}"]])
                     for si in range(n_s)}
        print(f"  d² NEUTRAL_TEXT × store distracteur ({d2_neutral_dist.shape[1]} "
              f"entrées) : {time.time() - t_d2:.1f}s")

    for lam_label, lam in (("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
        fact_series = [e3_series(d2_neutral[si], z[f"A_values_{si}"], lam, T_final)
                       for si in range(n_s)]
        p5["conditions"][f"fait-seul|{lam_label}"] = p5_block("fait-seul", fact_series, lam)
        if dist_store is not None:
            dist_series = [e3_series(d2_comb[si], vals_comb[si], lam, T_final)
                           for si in range(n_s)]
            p5["conditions"][f"distracteur|{lam_label}"] = p5_block("distracteur",
                                                                   dist_series, lam)
        # P5e : R_out = corr(D_permuté, H) / corr(D_kNN, H)
        perm_series = [e3_series(d2_neutral[si], perm_vals[si], lam, T_final)
                       for si in range(n_s)]
        c_knn = np.mean([pearson(s.tolist(), D_H.tolist()) for s in fact_series])
        c_perm = np.mean([pearson(s.tolist(), D_H.tolist()) for s in perm_series])
        p5["conditions"][f"permute|{lam_label}"] = p5_block("permuté", perm_series, lam)
        p5[f"P5e_R_out|{lam_label}"] = {"corr_perm": float(c_perm),
                                        "corr_knn": float(c_knn),
                                        "R_out": float(c_perm / c_knn) if c_knn else math.nan}

    for name, e in p5["conditions"].items():
        print(f"  [{name}] E3 = {e['E3_mean']:+.6f} (max secret {e['E3_max_secret']:+.6f}) "
              f"| bord {e['edge_-log(1-lambda)']:.6f} | max ΔNLL {e['P5d_max_delta']:.6f} "
              f"→ P5d(1) {'OK' if e['P5d_hard_edge_ok'] else 'VIOLÉ ⇒ RUN INVALIDE'}")
        print(f"      seuil 0.05 : {'DÉPASSÉ' if e['above_threshold_0.05'] else 'sous le seuil'} "
              f"| IC95 par secret [{e['boot_by_secret']['ci_lo']:+.6f}, "
              f"{e['boot_by_secret']['ci_hi']:+.6f}]")
        print(f"      P5b corr(D,H) = {e['P5b_corr_D_H_mean']:+.4f} ± {e['P5b_corr_D_H_sd']:.4f} "
              f"(Spearman {e['P5b_spearman_mean']:+.4f}) | var(D) = {e['P5b_var_D_mean']:.3e} "
              f"→ porte {'OK' if e['P5b_gate_ok'] else 'NON FRANCHIE (0/0)'}")
        print(f"      P5c apparié (décile NLL_base + zone [1,3], N={e['P5c_n_zone']}) = "
              f"{e['P5c_corr_matched_mean']:+.4f} ± {e['P5c_corr_matched_sd']:.4f} "
              f"→ {'|corr| < 0.15' if e['P5c_below_0.15'] else ''}"
              f"{' ANTIPODE ≥ +0.30' if e['P5c_antipode_ge_0.30'] else ''}")
        print(f"      P5d(2) masse au bord : global {e['P5d_frac_at_edge_all']:.4f} | "
              f"confiantes {e['P5d_frac_at_edge_confident']:.4f} | "
              f"incertaines {e['P5d_frac_at_edge_uncertain']:.4f} | "
              f"dommage conf {e['damage_confident']:+.6f} / incert {e['damage_uncertain']:+.6f}")
        csv_rows.append(["P5", name, "E3_mean", f"{e['E3_mean']:.6f}", "", n_s])
        csv_rows.append(["P5b", name, "corr_D_H", f"{e['P5b_corr_D_H_mean']:.6f}",
                         f"{e['P5b_corr_D_H_sd']:.6f}", n_s])
    for k, v in p5.items():
        if k.startswith("P5e_R_out"):
            print(f"  {k} : corr_perm {v['corr_perm']:+.4f} / corr_kNN {v['corr_knn']:+.4f} "
                  f"→ R_out = {v['R_out']:+.4f}")
    # ordinal Neuro : E3(fait-seul) > E3(distracteur) ?
    if dist_store is not None:
        for lam_label in ("lambda_star", "0.25"):
            a = p5["conditions"][f"fait-seul|{lam_label}"]["E3_mean"]
            b = p5["conditions"][f"distracteur|{lam_label}"]["E3_mean"]
            p5[f"ordinal_fait_gt_distracteur|{lam_label}"] = bool(a > b)
            print(f"  ordinal Neuro ({lam_label}) : E3(fait-seul) {a:+.6f} "
                  f"{'>' if a > b else '≤'} E3(distracteur) {b:+.6f} → "
                  f"{'conforme' if a > b else 'INVERSE (normalisation de p_kNN suspecte)'}")
    res["P5"] = p5

    # =================================================================== P6
    print("\n================ P6 (profondeur d'émergence — AUCUNE portée sur X7) ================")
    p6 = {}
    for arm in ARMS:
        T = temps[(arm, C_MAIN)]
        n_par, cases, _ = p1_table(stores[arm], arm, LAMBDA_STAR, T)
        ex_hits, r1s, r2c_ex, r2c_par = 0, [], [], []
        for si in range(n_s):
            e = eval_query(stores[arm][si], queries["A"][si][0], arm, LAMBDA_STAR, T)
            ins = instrumentation(stores[arm][si], queries["A"][si][0], arm, si, T)
            ex_hits += int(e["rank"] <= RANK_TOP)
            r1s.append(ins["R1_rank_correct_entry"])
            r2c_ex.append(ins["R2_cos_to_correct"])
        r2c_par = [cc["R2_cos_correct"] for cc in cases]
        p6[arm] = {"T": T, "exact_top10": ex_hits, "n_secrets_paraphrase": n_par,
                   "n_cases_top10": sum(cc["top10"] for cc in cases),
                   "R1_exact_mean": float(np.mean(r1s)),
                   "R1_exact_eq1": int(sum(1 for r in r1s if r == 1)),
                   "cos_exact_mean": float(np.mean(r2c_ex)),
                   "cos_paraphrase_mean": float(np.mean(r2c_par)),
                   "R1_paraphrase_mean": float(np.mean([cc["R1"] for cc in cases]))}
        e = p6[arm]
        print(f"  bras {arm:<7} T={T:.4f} | exact top-10 {ex_hits}/{n_s} "
              f"(R1=1 : {e['R1_exact_eq1']}/{n_s}) | paraphrases {n_par}/{n_s} secrets, "
              f"{e['n_cases_top10']}/30 cas")
        print(f"           cos(clé stockée, requête) : exact {e['cos_exact_mean']:+.4f} | "
              f"paraphrase {e['cos_paraphrase_mean']:+.4f} | R1 paraphrase moyen "
              f"{e['R1_paraphrase_mean']:.2f}")
    res["P6"] = p6

    # =================================================================== P7
    print("\n================ P7 (sélectivité : store 30 k + le fait) ================")
    if dist_store is not None:
        p7 = {}
        for lam_label, lam in (("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
            # l'entrée correcte est décalée dans le store combiné
            saved = list(correct_idx)
            for si in range(n_s):
                correct_idx[si] = saved[si] + offset
            n, cases, per = p1_table([dist_store[si] for si in range(n_s)], "final",
                                     lam, T_final)
            ex = 0
            for si in range(n_s):
                e = eval_query(dist_store[si], queries["A"][si][0], "final", lam, T_final)
                ex += int(e["rank"] <= RANK_TOP)
            for si in range(n_s):
                correct_idx[si] = saved[si]
            base_n = res["P1_grid"][f"final|c={C_MAIN}|lam={lam:.6f}"]["n_secrets"]
            base_cases = res["P1_grid"][f"final|c={C_MAIN}|lam={lam:.6f}"]["n_cases_top10"]
            p7[lam_label] = {"lambda": lam, "n_secrets": n,
                             "n_cases_top10": sum(cc["top10"] for cc in cases),
                             "exact_top10": ex,
                             "baseline_n_secrets": base_n,
                             "baseline_n_cases": base_cases,
                             "degradation_cases": (1.0 - (sum(cc["top10"] for cc in cases)
                                                          / base_cases))
                                                   if base_cases else math.nan,
                             "R1_paraphrase_mean": float(np.mean([cc["R1"] for cc in cases])),
                             "cases": cases}
            e = p7[lam_label]
            print(f"  λ={lam:.6f} | P1 {n}/{n_s} secrets ({base_n} fait-seul) | "
                  f"{e['n_cases_top10']}/30 cas ({base_cases} fait-seul) | "
                  f"exact {ex}/{n_s} | dégradation cas = "
                  f"{e['degradation_cases'] * 100:.1f} % (critère ≤ 30 %)")
            print(f"      R1 moyen (rang de l'entrée correcte parmi {offset + len(z['A_values_0'])} "
                  f"entrées) = {e['R1_paraphrase_mean']:.1f}")
        res["P7"] = {k: {kk: vv for kk, vv in v.items() if kk != "cases"}
                     for k, v in p7.items()}
    else:
        res["P7"] = None

    # =================================================================== P8
    print("\n================ P8 (forme : rang vs d²_min, poolé) ================")
    pts = []
    for si in range(n_s):
        for qi in range(len(qnames)):
            q = queries["A"][si][qi]
            e = eval_query(stores["final"][si], q, "final", LAMBDA_STAR, T_final)
            pts.append({"secret": secrets[si], "question": qnames[qi],
                        "d2min": e["d2min"], "rank": e["rank"],
                        "rank_base": e["rank_base"], "log_rank": math.log(e["rank"])})
    d2v = [p["d2min"] for p in pts]
    rkv = [p["rank"] for p in pts]
    order = np.argsort(d2v)
    binned = []
    nb_bins = 8
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
    print(f"  N={len(pts)} points | Spearman(d²_min, rang) = {res['P8']['spearman_d2min_rank']:+.4f} "
          f"| Pearson(d²_min, log rang) = {res['P8']['pearson_d2min_logrank']:+.4f}")
    print("    bin  n   d²_min moyen   rang médian   log-rang moyen")
    for i, b in enumerate(binned):
        print(f"    {i:<4} {b['n']:<3} {b['d2min_mean']:12.1f} {b['rank_median']:13.1f} "
              f"{b['log_rank_mean']:16.4f}")
    print(f"  écarts consécutifs de log-rang : "
          + " ".join(f"{d:+.3f}" for d in diffs)
          + f" | monotone croissant : {res['P8']['monotone_increasing_bins']}")

    # =================================================================== bras G
    print("\n================ bras G (kNN-LM GATÉ — non décisionnel) ================")
    n_g, cases_g, _ = p1_table(stores["final"], "final", G_LAMBDA, T_final, gate_tau=tau)
    e3_g = [e3_series(d2_neutral[si], z[f"A_values_{si}"], G_LAMBDA, T_final,
                      gate_tau=tau) for si in range(n_s)]
    e3_g_mean = float(np.mean([s.mean() for s in e3_g]))
    n_g_star, cases_g_star, _ = p1_table(stores["final"], "final", LAMBDA_STAR,
                                         T_final, gate_tau=tau)
    p2_g = {}
    for arm in ("final",):
        qm = Query(z["C_m0_w0_logp"], z["C_m0_w0_final"], z["C_m0_w0_inject"],
                   z["C_m0_w0_target"], z["C_m0_w0_nll_base"], z["C_m0_w0_H"], "Marseille")
        lam_renv = None
        for lam in np.unique(np.round(np.linspace(0.0, 0.999, 1000), 6)).tolist():
            if eval_query(store_c[arm], qm, arm, lam, temps[(arm, C_MAIN)],
                          gate_tau=tau)["rank"] == 1:
                lam_renv = lam
                break
        p2_g[arm] = {"lambda_renv": lam_renv,
                     "F": lam_renv / LAMBDA_STAR if lam_renv is not None else None}
    res["G"] = {"tau": tau, "lambda": G_LAMBDA,
                "E3_mean": e3_g_mean,
                "E3_within_budget": bool(e3_g_mean <= E3_THRESHOLD),
                "E3_per_secret": [float(s.mean()) for s in e3_g],
                "P1_n_secrets_at_G_lambda": n_g,
                "P1_n_cases_at_G_lambda": sum(cc["top10"] for cc in cases_g),
                "P1_n_secrets_at_lambda_star": n_g_star,
                "P1_n_cases_at_lambda_star": sum(cc["top10"] for cc in cases_g_star),
                "P2_gated": p2_g,
                "alpha_pass_rate_fact_queries": float(np.mean(
                    [cc["alpha"] for cc in cases_g]))}
    g = res["G"]
    print(f"  τ = {tau:.2f} (fixé avant P1/P2) | λ = {G_LAMBDA}")
    print(f"  E3 gaté = {e3_g_mean:+.6f} nats/token → "
          f"{'≤ 0.05 (budget tenu)' if g['E3_within_budget'] else '> 0.05 (hors budget)'}")
    print(f"  P1 gaté à λ={G_LAMBDA} : {n_g}/{n_s} secrets, "
          f"{g['P1_n_cases_at_G_lambda']}/30 cas | à λ* : {n_g_star}/{n_s} secrets, "
          f"{g['P1_n_cases_at_lambda_star']}/30 cas")
    print(f"  taux de passage du gate sur les requêtes-fait (α=1) : "
          f"{g['alpha_pass_rate_fact_queries']:.4f}")
    print(f"  P2 gaté : λ_renv = {p2_g['final']['lambda_renv']} | "
          f"F = {p2_g['final']['F']}")

    # ============================================== bras descriptif M + kNN (passe B)
    print("\n================ bras descriptif : M au point courant + kNN (additivité) ================")
    addit = {}
    for lam_label, lam in (("0.0", 0.0), ("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
        rows = []
        for si in range(n_s):
            for qi in range(len(qnames)):
                qa = queries["A"][si][qi]
                qb = queries["B"][si][qi]
                ma, la = full_logp(stores["final"][si], qa, "final", lam, T_final)
                mb, lb = full_logp(stores["final"][si], qb, "final", lam, T_final)
                rows.append({"secret": secrets[si], "question": qnames[qi],
                             "knn_only": ma - la, "M_only": lb - la,
                             "M_plus_knn": mb - la,
                             "additive_gap": (mb - la) - ((ma - la) + (lb - la))})
        for key in ("knn_only", "M_only", "M_plus_knn", "additive_gap"):
            for qname in qnames:
                vals = [r[key] for r in rows if r["question"] == qname]
                m_, s_ = mean_sd(vals)
                addit.setdefault(lam_label, {}).setdefault(qname, {})[key] = \
                    {"mean": m_, "sd": s_, "n": len(vals)}
        exact = addit[lam_label]["exact"]
        print(f"  λ={lam_label:<12} exact : kNN seul {exact['knn_only']['mean']:+.3f} | "
              f"M seule {exact['M_only']['mean']:+.3f} ± {exact['M_only']['sd']:.3f} | "
              f"M+kNN {exact['M_plus_knn']['mean']:+.3f} | "
              f"écart d'additivité {exact['additive_gap']['mean']:+.3f}")
    res["arm_M_plus_knn"] = addit

    # ============================================ Δlogp descriptifs + confondants
    print("\n================ Δlogp descriptifs & confondants (§5.9) ================")
    desc = {}
    for lam_label, lam in (("lambda_star", LAMBDA_STAR), ("0.25", 0.25)):
        for qi, qname in enumerate(qnames):
            vals, ranks, lpbase = [], [], []
            for si in range(n_s):
                q = queries["A"][si][qi]
                mm, ll = full_logp(stores["final"][si], q, "final", lam, T_final)
                vals.append(mm - ll)
                e = eval_query(stores["final"][si], q, "final", lam, T_final)
                ranks.append(e["rank"])
                lpbase.append(float(q.logp[0][int(q.target[0])]))
            m_, s_ = mean_sd(vals)
            desc.setdefault(lam_label, {})[qname] = {
                "delta_logp_mean": m_, "delta_logp_sd": s_,
                "boot": bootstrap_mean(vals),
                "corr_rank_mix_logp_base": spearman(ranks, lpbase),
                "ranks": ranks}
            csv_rows.append(["Dlogp", f"{lam_label}|{qname}", "delta_logp",
                             f"{m_:.6f}", f"{s_:.6f}", n_s])
            print(f"  λ={lam_label:<12} [{qname:<6}] Δlogp = {m_:+.4f} ± {s_:.4f} "
                  f"| corr(rang_mix, logp_base) = "
                  f"{desc[lam_label][qname]['corr_rank_mix_logp_base']:+.4f}")
    res["delta_logp"] = desc

    # tokens communs du fait dans les valeurs
    common_tokens = {}
    for si in range(n_s):
        vals = z[f"A_values_{si}"].tolist()
        common_tokens[secrets[si]] = {
            "n_entries": len(vals), "n_distinct_values": len(set(vals)),
            "frac_neutral_positions_covered": float(
                np.mean([int(t) in set(vals) for t in D_tgt.tolist()]))}
    res["confounders"] = {"store_values": common_tokens,
                          "lambda0_replay_E1": meta.get("lambda0_replay_E1")}

    # ======================================================== frontière complète
    print("\n================ FRONTIÈRE (λ, E3 analytique, E3 mesuré, P1, P2) ================")
    frontier = []
    print("    λ         E3 analytique  E3 mesuré   P1 n/10  P1 cas/30  rang Marseille  F")
    for lam in LAMBDA_GRID:
        e3_meas = float(np.mean([e3_series(d2_neutral[si], z[f"A_values_{si}"],
                                           lam, T_final).mean()
                                 for si in range(n_s)]))
        k = f"final|c={C_MAIN}|lam={lam:.6f}"
        rank_m = p2["m0|final"]["ranks_on_grid"][f"{lam:.6f}"]
        row = {"lambda": lam, "E3_analytic": -math.log1p(-lam), "E3_measured": e3_meas,
               "E3_within_budget": bool(e3_meas <= E3_THRESHOLD),
               "P1_n_secrets": res["P1_grid"][k]["n_secrets"],
               "P1_n_cases": res["P1_grid"][k]["n_cases_top10"],
               "P2_rank_marseille": rank_m,
               "F": p2["m0|final"]["F"]}
        frontier.append(row)
        csv_rows.append(["frontier", f"lam={lam:.6f}", "E3_measured",
                         f"{e3_meas:.6f}", "", n_s])
        print(f"    {lam:.6f}  {row['E3_analytic']:.6f}     {e3_meas:+.6f}  "
              f"{row['P1_n_secrets']:>5}  {row['P1_n_cases']:>8}  {rank_m:>13}  "
              f"{('%.2f' % row['F']) if row['F'] else 'n/a'}")
    res["frontier"] = frontier

    # ================================================ table d'attribution (§4)
    print("\n================ TABLE D'ATTRIBUTION (§4) ================")
    cases_main = res["P1_main"]["cases"] if res["P1_main"].get("cases") else m["cases"]
    r1_par = [cc["R1"] for cc in cases_main]
    r3_par = [cc["R3"] for cc in cases_main]
    cos_par = [cc["R2_cos_correct"] for cc in cases_main]
    cos_ex = [instrumentation(stores["final"][si], queries["A"][si][0], "final", si,
                              T_final)["R2_cos_to_correct"] for si in range(n_s)]
    r1_ex = [instrumentation(stores["final"][si], queries["A"][si][0], "final", si,
                             T_final)["R1_rank_correct_entry"] for si in range(n_s)]
    p1_ok = m["n_secrets"] >= 5
    r1_eq1_frac = float(np.mean([r == 1 for r in r1_par]))
    r3_mean = float(np.mean(r3_par))
    if p1_ok:
        branch = "P1 réussit"
        diag = ("adressage effectif (R1=1 majoritaire, R3 élevée)" if r1_eq1_frac > 0.5
                else "ANTIPODE : succès sans l'entrée correcte ⇒ ININTERPRÉTABLE")
    elif r1_eq1_frac > 0.5 and r3_mean >= 0.5:
        branch = "P1 échoue, R1 = 1, R3 élevée"
        diag = "Branche B — échec côté INJECTION"
    else:
        branch = "P1 échoue, R1 > 1, R3 faible"
        diag = "Branche A — échec côté CLÉ"
    res["attribution"] = {
        "P1_n_secrets": m["n_secrets"], "P1_n_cases": m["n_cases_top10"],
        "R1_paraphrase_mean": float(np.mean(r1_par)),
        "R1_paraphrase_eq1_frac": r1_eq1_frac,
        "R1_exact_mean": float(np.mean(r1_ex)),
        "R1_exact_eq1_frac": float(np.mean([r == 1 for r in r1_ex])),
        "R2_cos_exact_mean": float(np.mean(cos_ex)),
        "R2_cos_paraphrase_mean": float(np.mean(cos_par)),
        "R3_paraphrase_mean": r3_mean,
        "R3_exact_mean": float(np.mean([
            eval_query(stores["final"][si], queries["A"][si][0], "final",
                       LAMBDA_STAR, T_final)["p_knn_target"] for si in range(n_s)])),
        "R4_paraphrase_mean": float(np.mean([cc["R4"] for cc in cases_main])),
        "row": branch, "diagnostic": diag}
    a = res["attribution"]
    print(f"  R1 (rang de l'entrée correcte) : exact {a['R1_exact_mean']:.2f} "
          f"(=1 dans {a['R1_exact_eq1_frac'] * 100:.0f} % des cas) | "
          f"paraphrase {a['R1_paraphrase_mean']:.2f} "
          f"(=1 dans {a['R1_paraphrase_eq1_frac'] * 100:.0f} %)")
    print(f"  R2 cos(clé stockée, requête) : exact {a['R2_cos_exact_mean']:+.4f} | "
          f"paraphrase {a['R2_cos_paraphrase_mean']:+.4f}")
    print(f"  R3 masse p_kNN(cible) à λ* : exact {a['R3_exact_mean']:.5f} | "
          f"paraphrase {a['R3_paraphrase_mean']:.5f}")
    print(f"  R4 entropie de p_kNN (paraphrase) : {a['R4_paraphrase_mean']:.4f} nats")
    print(f"  ⇒ ligne : « {branch} » → {diag}")

    # ---------------------------------------------------- manifeste + sorties
    res["duration_analysis_s"] = time.time() - t_start
    _dump(out_dir, res, csv_rows)
    root = Path(__file__).resolve().parents[1]
    man_paths = [raw / "gpu_raw.npz", raw / "gpu_meta.json",
                 root / "eval" / "knn_ceiling.py", root / "engram" / "config.py",
                 root / "engram" / "cortex.py",
                 root / "experiments" / "EXP-2026-08-21-knn-borne-logits.md"]
    rfc = root / "data" / "rfc9293.txt"
    if rfc.exists():
        man_paths.append(rfc)
    manifest = {"files": {}}
    for p in man_paths:
        if p.exists():
            manifest["files"][str(p).replace("\\", "/")] = {
                "sha256": sha256_file(p), "bytes": p.stat().st_size}
    (out_dir / "manifest_sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  manifeste SHA-256 : {len(manifest['files'])} fichiers")
    print(f"  analyse : {res['duration_analysis_s']:.1f}s → {out_dir}")


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
    ap.add_argument("--secrets", type=int, default=10)
    ap.add_argument("--distractor-tokens", type=int, default=DISTRACTOR_TOKENS)
    ap.add_argument("--skip-distractor", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else \
        root / "experiments" / "results" / "knn-borne-logits"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("gpu", "all"):
        _gpu_phase(args, out_dir)
    if args.phase in ("analysis", "all"):
        _analysis_phase(args, out_dir)


if __name__ == "__main__":
    main()
