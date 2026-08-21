# SPDX-License-Identifier: AGPL-3.0-or-later
"""L'hippocampe : mémoire associative à fast weights, sans aucun backprop.

Une seule structure d'état : M (d×key_dim, fp32 ; key_dim = d en mode X0, dg_dim
quand la projection gyrus denté est active). Lecture par produit matrice-vecteur,
écriture par delta rule (hebbien corrigé), oubli par decay + élagage top-k.

Invariants à ne pas casser (voir docs/ARCHITECTURE.md, Décisions D1/D2 et §4) :
- M et toute la chaîne d'update restent en float32, même si le cortex est en fp16 :
  les incréments η·(v − M·k)·kᵀ partent en underflow en demi-précision, et la mémoire
  cesse d'apprendre *silencieusement*. Les casts .float() sont volontaires.
- Les clés/valeurs fournies par l'appelant doivent être des états PRÉ-injection
  (c'est l'engine qui garantit ça ; l'hippocampe ne peut pas le vérifier).
"""

from __future__ import annotations

import torch

from .config import EngramConfig

_EPS = 1e-8


class FastWeightMemory:
    """Matrice de fast weights M : lecture λ·M·φ(h), écriture delta rule.

    Sémantique prédictive : on écrit v = h_t sous la clé k = φ(h_{t-1}), donc M
    apprend des *transitions* d'états latents. Au rappel, un état ressemblant à un
    ancien h_{t-1} réinjecte un souvenir de ce qui avait suivi.
    """

    def __init__(self, d_model: int, cfg: EngramConfig, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.d = d_model
        self.device = torch.device(device)

        # X1 — gyrus denté : projection aléatoire GELÉE d → dg_dim (jamais apprise,
        # seedée par cfg.seed pour être identique d'un run à l'autre). Les clés
        # deviennent creuses en haute dimension ; M devient rectangulaire d×dg_dim.
        if cfg.dg_dim:
            gen = torch.Generator().manual_seed(cfg.seed)
            self.G = torch.randn(cfg.dg_dim, d_model, generator=gen).to(self.device)
            self.key_dim = cfg.dg_dim
        else:
            self.G = None
            self.key_dim = d_model

        self.M = torch.zeros(d_model, self.key_dim, dtype=torch.float32, device=self.device)
        self.write_count = 0

        # I1 — instrumentation : buffer circulaire des clés écrites + similarité cos
        # max de chaque nouvelle clé vs les précédentes (les clés sont unitaires,
        # cos = produit scalaire). Diagnostic pur, inactif par défaut.
        if cfg.track_keys:
            self._key_buf = torch.zeros(cfg.track_keys_max, self.key_dim, device=self.device)
            self._key_count = 0
        self.write_similarities: list[float] = []

        # X8 — entropie du cortex au pas courant, fournie par l'engine avant chaque
        # forward (None = pas de logits antérieurs dans ce contexte).
        self.gate_entropy: float | None = None
        # X8.1 — collecteur optionnel des gains g appliqués (diagnostic ; une éval
        # l'active en l'initialisant à []).
        self.gate_log: list[float] | None = None

    # ------------------------------------------------------------------ util

    def phi(self, h: torch.Tensor) -> torch.Tensor:
        """Feature map des clés, normalisée L2 dans les deux modes (borne l'énergie,
        rend η interprétable).

        - X0 (dg_dim=0) : identité normalisée, clé dense de dimension d.
        - X1 : topk(G·h) — projection haute dimension puis sparsification par
          magnitude : deux états proches en dimension d se recouvrent beaucoup moins
          après top-k, c'est la séparation de patterns du gyrus denté.
        """
        h = h.float()
        if self.G is None:
            return h / (h.norm() + _EPS)
        z = self.G @ h
        idx = z.abs().topk(self.cfg.dg_topk).indices
        sparse = torch.zeros_like(z)
        sparse[idx] = z[idx]
        return sparse / (sparse.norm() + _EPS)

    # ------------------------------------------------------------------ read

    def _read_gate(self, k: torch.Tensor) -> float:
        """X8 : gain de lecture g ∈ [0, 1] = soft-OR(incertitude, pertinence).

        - Incertitude : sigmoïde sur l'entropie du cortex au pas courant (« ai-je
          besoin d'aide ? »). Premier token d'un contexte : pas d'info → 1.0.
        - Pertinence : sigmoïde sur le cos max entre la clé de requête et le buffer
          de clés I1 (« ai-je quelque chose de pertinent ? »). Ce facteur peut
          FORCER le passage à basse entropie — c'est la réponse au cas « confiance
          erronée » (fait périmé que le cortex croit encore connaître).
        """
        mode = self.cfg.read_gate
        if mode == "none":
            return 1.0
        if self.gate_entropy is None:
            g_h = 1.0
        else:
            x = (self.gate_entropy - self.cfg.gate_entropy_mid) / self.cfg.gate_entropy_tau
            g_h = float(torch.sigmoid(torch.tensor(x)))
        g_k = 0.0
        n = min(self._key_count, self.cfg.track_keys_max) if self.cfg.track_keys else 0
        if n > 0:
            x = ((self._key_buf[:n] @ k).max().item() - self.cfg.gate_keysim_mid) / self.cfg.gate_keysim_tau
            g_k = float(torch.sigmoid(torch.tensor(x)))
        if mode == "entropy":
            g = g_h
        elif mode == "keysim":
            g = g_k
        else:
            g = 1.0 - (1.0 - g_h) * (1.0 - g_k)  # soft-OR : l'un OU l'autre suffit
        if self.gate_log is not None:
            self.gate_log.append(g)
        return g

    @torch.no_grad()
    def read(self, h: torch.Tensor) -> torch.Tensor:
        """Vecteur à ajouter au flux résiduel : gate X8 éventuel, puis plafond de
        norme (le cap reste en plancher de sécurité — décision D10), dtype de h."""
        k = self.phi(h)
        g = self._read_gate(k)
        r = (self.cfg.lam * g) * (self.M @ k)
        cap = self.cfg.max_read_norm * h.float().norm()
        n = r.norm()
        if n > cap:
            r = r * (cap / (n + _EPS))
        return r.to(h.dtype)

    # ----------------------------------------------------------------- write

    @torch.no_grad()
    def write(self, key_h: torch.Tensor, value_h: torch.Tensor) -> None:
        """Une mise à jour delta rule : M ← M + η·(v − M·k)·kᵀ, puis decay, puis élagage éventuel.

        Le terme correctif −M·k n'écrit que l'erreur de prédiction de la mémoire :
        pas d'accumulation divergente sur les répétitions, et une clé connue peut
        être réécrite avec une nouvelle valeur. `hebbian_only` (ablation) le retire.
        """
        k = self.phi(key_h)
        v = value_h.float()
        if self.cfg.track_keys:
            self._track(k)
        if self.cfg.hebbian_only:
            delta = torch.outer(v, k)
        else:
            delta = torch.outer(v - self.M @ k, k)
        self.M += self.cfg.eta * delta
        self.M *= 1.0 - self.cfg.decay
        self.write_count += 1
        if self.cfg.prune_every and self.write_count % self.cfg.prune_every == 0:
            self.prune()

    @torch.no_grad()
    def _track(self, k: torch.Tensor) -> None:
        """I1 : enregistre la similarité cos max de k vs les clés déjà écrites,
        puis range k dans le buffer circulaire. La toute première clé n'a pas de
        référence : rien n'est enregistré pour elle."""
        n = min(self._key_count, self.cfg.track_keys_max)
        if n > 0:
            self.write_similarities.append((self._key_buf[:n] @ k).max().item())
        self._key_buf[self._key_count % self.cfg.track_keys_max] = k
        self._key_count += 1

    # ------------------------------------------------------- oubli / hygiène

    @torch.no_grad()
    def prune(self) -> None:
        """Ne garde que la fraction `prune_keep` des coefficients de plus grande magnitude."""
        keep = max(1, int(self.cfg.prune_keep * self.M.numel()))
        threshold = torch.topk(self.M.abs().flatten(), keep).values.min()
        self.M[self.M.abs() < threshold] = 0.0

    def reset(self) -> None:
        """M ← 0 : l'« espace latent presque vierge ». Ne touche pas au cache KV."""
        self.M.zero_()
        self.write_count = 0
        if self.cfg.track_keys:
            self._key_count = 0
        self.write_similarities = []

    # ----------------------------------------------------------------- infos

    def stats(self) -> dict:
        return {
            "frobenius": self.M.norm().item(),
            "density": (self.M != 0).float().mean().item(),
            "writes": self.write_count,
            "key_dim": self.key_dim,
        }
