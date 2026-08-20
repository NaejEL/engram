"""Le néocortex : un LLM gelé, plus le hook d'injection de l'hippocampe.

Rôles :
- charger le modèle HF + tokenizer, geler tous les poids (aucun gradient nulle part) ;
- poser un forward hook sur le bloc `cfg.layer_index` qui, à chaque forward :
    1. stashe l'état caché de la DERNIÈRE position, PRÉ-injection (→ clés/valeurs
       d'écriture, décision D1 : ne jamais écrire les états post-injection) ;
    2. si la lecture est active, ajoute λ·M·φ(h) à cette même position.

La boucle étant token-par-token (décision D4), « dernière position » = la seule
position du forward courant ; le hook resterait néanmoins correct en prefill (il
n'agirait que sur la dernière position, et le stash serait celui du dernier token).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EngramConfig
from .hippocampus import FastWeightMemory


def _resolve_device(cfg: EngramConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _find_blocks(model) -> list:
    """Liste des blocs transformer, pour GPT-2 (`transformer.h`) et les architectures
    Llama-style comme SmolLM2 (`model.layers`)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        f"Architecture non reconnue pour {type(model).__name__} : "
        "ajouter son chemin de blocs dans cortex._find_blocks()."
    )


class Cortex:
    def __init__(self, cfg: EngramConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        # `dtype=` (transformers v5) ; l'ancien nom `torch_dtype` est déprécié.
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.d_model: int = self.model.config.hidden_size

        blocks = _find_blocks(self.model)
        if not 0 <= cfg.layer_index < len(blocks):
            raise ValueError(f"layer_index={cfg.layer_index} hors de [0, {len(blocks)})")
        self.n_layers = len(blocks)

        # Branchés par l'engine ; le hook est inerte tant que memory est None.
        self.memory: FastWeightMemory | None = None
        self.read_enabled: bool = True
        # État caché (dernière position, PRÉ-injection) du dernier forward — la
        # matière première des writes.
        self.last_h_pre: torch.Tensor | None = None

        blocks[cfg.layer_index].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        # Les blocs HF renvoient un tuple dont l'élément 0 est hidden_states.
        hidden = output[0] if isinstance(output, tuple) else output
        h_last = hidden[0, -1, :]
        self.last_h_pre = h_last.detach().float().clone()

        if self.memory is None or not self.read_enabled:
            return output

        injected = hidden.clone()
        injected[0, -1, :] = h_last + self.memory.read(h_last)
        if isinstance(output, tuple):
            return (injected,) + output[1:]
        return injected

    @torch.no_grad()
    def forward_one(self, token_id: int, past_key_values):
        """Un pas de forward (un token, cache KV). Retourne (logits_fp32, past)."""
        input_ids = torch.tensor([[token_id]], device=self.device)
        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values
