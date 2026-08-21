# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration unique du PoC. Tout hyperparamètre vit ici — rien en dur ailleurs.

Les valeurs par défaut sont des points de départ raisonnés, pas des résultats :
elles sont toutes à balayer (voir docs/ARCHITECTURE.md §5 et le journal).
"""

from dataclasses import dataclass, field


@dataclass
class EngramConfig:
    # --- Cortex (le LLM gelé) ---
    model_name: str = "gpt2"  # 124M — cycle d'itération court ; SmolLM2-360M en v1.2
    device: str = "auto"      # "auto" → cuda si dispo, sinon cpu
    layer_index: int = 6      # bloc où l'hippocampe lit/écrit (GPT-2 : 0..11, milieu = 6)

    # --- Hippocampe (lecture) ---
    # λ=2.0, cap=0.5 : régime agressif ROUVERT par le gate keysim (X8, 2026-08-21) —
    # la sélectivité vient du gate (lecture coupée hors domaine), plus du cap serré.
    # E1 +1.353 ± 1.58, E3 −0.014 (GPT-2). L'ancien point sans gate (λ=1, cap=0.25 :
    # E1 +0.740, E3 +0.023) reste la référence d'ablation read_gate="none".
    lam: float = 2.0            # λ : force d'injection de M·φ(h) dans le flux résiduel
    max_read_norm: float = 0.5   # plafond de ‖injection‖ en fraction de ‖h‖

    # --- Hippocampe (écriture) ---
    eta: float = 0.2           # η : pas d'apprentissage de la delta rule
    decay: float = 1e-3        # δ : M ← (1−δ)·M à chaque write
    hebbian_only: bool = False  # ablation D5 : Hebb pur (sans terme correctif −M·k)

    # --- X1 : projection gyrus denté (séparation de patterns) ---
    # φ devient topk(G·h) avec G aléatoire GELÉE (seedée par cfg.seed) : clés creuses
    # en haute dimension → moins d'interférence, M rectangulaire d×dg_dim.
    # Retenue par défaut depuis le verdict X1 du 2026-08-20 (tableau EXTENSIONS.md §4).
    dg_dim: int = 8192   # dimension de projection ; 0 = désactivé (mode X0 dense)
    dg_topk: int = 64    # composantes gardées (64/8192 ≈ 0.8 %, ordre bio ; 128 ≈ pareil)

    # --- I1 : instrumentation — prédicteur d'échec par similarité de clés ---
    # Logge à chaque write la similarité cos max entre la nouvelle clé et les clés
    # déjà écrites (buffer circulaire). Diagnostic pur : ne change aucun mécanisme.
    track_keys: bool = False
    track_keys_max: int = 4096  # taille du buffer de clés (FIFO circulaire)

    # --- X8 : gate de lecture à deux facteurs (voir EXTENSIONS.md §X8) ---
    # g = soft-OR(incertitude du cortex, pertinence du match mémoire). Le cap
    # max_read_norm reste en PLANCHER de sécurité (décision D10). Défaut "keysim"
    # depuis le banc X8 (2026-08-21) : E1 intact, E3 éliminé ; le facteur entropie
    # est disqualifié (anomalie E3 + entropie décalée d'un pas — voir journal).
    read_gate: str = "keysim"        # "none" | "entropy" | "keysim" | "two_factor"
    gate_entropy_mid: float = 2.0    # nats — centre du sigmoïde d'incertitude
    gate_entropy_tau: float = 0.5
    gate_keysim_mid: float = 0.6     # centre du sigmoïde de pertinence — réglé sur GPT-2
                                     # (banc X8) ; à re-balayer PAR MODÈLE (keysim
                                     # d'écriture ≈ 0.14 sur Qwen vs ~0.23 GPT-2)
    gate_keysim_tau: float = 0.05

    # --- V2-D(a) : kNN-LM nu, borne de l'étage des logits ---
    # Protocole pré-enregistré experiments/EXP-2026-08-21-knn-borne-logits-v2.md, §10
    # (antécédent v1 : EXP-2026-08-21-knn-borne-logits.md) et décision PI §13.5 :
    # « champs knn_* dans EngramConfig, défaut = comportement actuel
    # (knn_lambda: float = 0.0) ». INSTRUMENT, pas un mécanisme retenu : la ligne va
    # au tableau des ablations/instruments (§4).
    # Le datastore lui-même vit dans le script d'éval (eval/knn_ceiling.py), JAMAIS
    # dans engram/ — et les clés kNN ne passent JAMAIS par G/DG (§5.8, D9).
    knn_lambda: float = 0.0     # λ du mélange p = (1−λ)·p_LM + λ·p_kNN ; 0.0 = inerte
    knn_k: int = 8              # nombre de voisins (k = 8, §7)
    # SEUL AJOUT DU PROTOCOLE v2 (§10) : `knn_temp_c = 0.0` ≡ **un-hot** ≡ limite
    # c→0⁺ ≡ **uniforme sur l'argmin-set**. T_q = c·med_{j≥2}(d²_j − d²_min) est
    # recalculée PAR REQUÊTE (la médiane-de-médianes par bras du run 1 est abandonnée).
    # Champ purement descriptif ici : aucun code de engram/ ne le lit (le datastore
    # vit dans eval/), donc le défaut ne change aucun comportement courant.
    knn_temp_c: float = 0.0
    knn_key_layer: str = "final"  # "final" (état pré-lm_head) | "inject" (layer_index)
    knn_gate_tau: float = 0.0   # bras G : α = 1[d²_min ≤ τ] ; 0.0 = gate désactivé
                                # (bras G v2 : COURBE sur une grille de τ, aucun τ fixé)

    # --- V2-D(a) : second point de capture (état final, pré-lm_head) ---
    # Inerte par défaut (§10) : le hook n'est même pas posé quand le flag est faux.
    capture_final_state: bool = False

    # --- Gating par surprise ---
    surprise_threshold: float = 4.0  # NLL en nats au-dessus de laquelle on écrit

    # --- Élagage (consolidation pauvre) ---
    prune_every: int = 512     # période en nombre de writes ; 0 = désactivé
    prune_keep: float = 0.10   # fraction des coefficients gardés (par magnitude)

    # --- Divers ---
    seed: int = 0
    notes: str = field(default="", repr=False)  # libre, pour taguer un run dans le journal

    def __post_init__(self) -> None:
        if self.read_gate in ("keysim", "two_factor") and not self.track_keys:
            # Le facteur pertinence lit le buffer de clés I1 : l'activer d'office.
            self.track_keys = True

    def summary(self) -> str:
        """Ligne compacte pour les logs et le journal d'expériences."""
        dg = f"dg={self.dg_dim}/{self.dg_topk}" if self.dg_dim else "dg=off"
        dg += f" gate={self.read_gate}" if self.read_gate != "none" else ""
        # knn_* n'apparaît QUE s'il est armé : à knn_lambda=0.0 (défaut) la ligne de
        # résumé est bit-à-bit celle d'avant V2-D(a).
        knn = ""
        if self.knn_lambda:
            knn = (f" knn=λ{self.knn_lambda}/k{self.knn_k}/c{self.knn_temp_c}"
                   f"/{self.knn_key_layer}"
                   + (f"/τ{self.knn_gate_tau}" if self.knn_gate_tau else ""))
        return (
            f"model={self.model_name} layer={self.layer_index} lam={self.lam} "
            f"eta={self.eta} decay={self.decay} thr={self.surprise_threshold} "
            f"prune={self.prune_every}/{self.prune_keep} hebb_only={self.hebbian_only} {dg}"
            f"{knn}"
        )
