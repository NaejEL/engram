# SPDX-License-Identifier: AGPL-3.0-or-later
"""La boucle en ligne : le cortex avance token par token, l'hippocampe lit à chaque
pas et écrit quand la surprise l'exige.

Chronologie d'un pas (tokens x_0..x_n, états cachés pré-injection h_0..h_n) :

    1. avant de consommer x_t : les logits du pas précédent donnent
       surprise = NLL(x_t) — c'est le signal de gating, calculé AVANT que x_t
       n'influence quoi que ce soit ;
    2. forward de x_t → h_t (stashé par le hook), logits pour x_{t+1} ;
    3. si surprise > seuil (ou write forcé) : M.write(k=h_{t-1}, v=h_t).
       L'association apprise est la transition qui vient de surprendre le modèle.

Règle absolue : on n'écrit JAMAIS sur des tokens auto-générés (generate) — apprendre
de ses propres sorties est la boucle de rétroaction qui fait diverger M (§4 pièges).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import EngramConfig
from .cortex import Cortex
from .hippocampus import FastWeightMemory


@dataclass
class StepRecord:
    token_id: int
    nll: float | None  # None pour le tout premier token d'un contexte (pas de logits avant)
    wrote: bool


class EngramEngine:
    """Assemble cortex + hippocampe et expose les opérations du protocole d'éval :
    stream / generate / logprob_continuation / clear_context / reset_memory."""

    def __init__(self, cfg: EngramConfig | None = None):
        self.cfg = cfg or EngramConfig()
        torch.manual_seed(self.cfg.seed)
        self.cortex = Cortex(self.cfg)
        self.memory = FastWeightMemory(self.cortex.d_model, self.cfg, device=self.cortex.device)
        self.cortex.memory = self.memory
        self.clear_context()

    # ----------------------------------------------------------- état/hygiène

    def clear_context(self) -> None:
        """Vide le cache KV (et l'état de boucle) en GARDANT M.
        C'est l'opération clé des évals : ce qui survit à ça n'est pas du in-context."""
        self._past = None
        self._prev_h: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None
        self._context_len = 0

    def reset_memory(self) -> None:
        """M ← 0. Indépendant du cache KV."""
        self.memory.reset()

    @property
    def tokenizer(self):
        return self.cortex.tokenizer

    # ------------------------------------------------------------- primitives

    def _nll_of(self, token_id: int) -> float | None:
        if self._last_logits is None:
            return None
        return -F.log_softmax(self._last_logits, dim=-1)[token_id].item()

    @torch.no_grad()
    def _consume(self, token_id: int, *, read: bool, write: bool, force_write: bool) -> StepRecord:
        """Avance d'un token en appliquant la chronologie décrite en tête de module."""
        nll = self._nll_of(token_id)
        surprised = force_write or (nll is not None and nll > self.cfg.surprise_threshold)
        prev_h = self.cortex.last_h_pre if self._context_len > 0 else None

        # X8 : fournir l'entropie du pas courant au gate de lecture (gratuit hors gate).
        if self.cfg.read_gate != "none":
            if self._last_logits is None:
                self.memory.gate_entropy = None
            else:
                p = F.softmax(self._last_logits, dim=-1)
                self.memory.gate_entropy = float(-(p * torch.log(p + 1e-12)).sum())

        self.cortex.read_enabled = read
        self._last_logits, self._past = self.cortex.forward_one(token_id, self._past)
        self._context_len += 1

        wrote = False
        if write and surprised and prev_h is not None:
            self.memory.write(prev_h, self.cortex.last_h_pre)
            wrote = True
        return StepRecord(token_id, nll, wrote)

    # -------------------------------------------------------------- opérations

    def stream(
        self,
        text: str,
        *,
        read: bool = True,
        write: bool = True,
        force_write: bool = False,
    ) -> list[StepRecord]:
        """Streame un texte dans le contexte courant, en apprenant en ligne.
        `force_write=True` court-circuite le gating (protocole E1, phase d'injection)."""
        records = []
        for tid in self.tokenizer.encode(text):
            records.append(self._consume(tid, read=read, write=write, force_write=force_write))
        return records

    @torch.no_grad()
    def logprob_continuation(
        self, prompt: str, continuation: str, *, read: bool = True
    ) -> tuple[float, int]:
        """log P(continuation | prompt, contexte courant), et rang du premier token de
        la continuation dans les logits. Écriture désactivée (le rappel ne doit pas
        contaminer M). La tokenisation de la continuation est dérivée par différence
        prompt vs prompt+continuation pour respecter les espaces de début de mot BPE."""
        ids_prompt = self.tokenizer.encode(prompt)
        ids_full = self.tokenizer.encode(prompt + continuation)
        assert ids_full[: len(ids_prompt)] == ids_prompt, (
            "Tokenisation non préfixe-stable pour ce couple prompt/continuation ; "
            "choisir une autre formulation."
        )
        ids_cont = ids_full[len(ids_prompt) :]
        assert ids_cont, "Continuation vide après tokenisation."

        self.stream(prompt, read=read, write=False)
        first_rank = int((self._last_logits > self._last_logits[ids_cont[0]]).sum().item()) + 1
        total = 0.0
        for tid in ids_cont:
            total += -self._nll_of(tid)
            self._consume(tid, read=read, write=False, force_write=False)
        return total, first_rank

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 40, *, read: bool = True) -> str:
        """Génération greedy avec M actif en lecture. Jamais d'écriture ici (règle absolue)."""
        self.stream(prompt, read=read, write=False)
        out_ids = []
        for _ in range(max_new_tokens):
            tid = int(self._last_logits.argmax().item())
            if tid == self.tokenizer.eos_token_id:
                break
            out_ids.append(tid)
            self._consume(tid, read=read, write=False, force_write=False)
        return self.tokenizer.decode(out_ids)
