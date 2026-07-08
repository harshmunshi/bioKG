"""
BioKG-LoRA: Knowledge Graph Enhanced LLM with LoRA adapters (Stage 3).

Architecture:
    1. Base LLM (Llama-3-8B or gemma-4-e2b-it)  — frozen, loaded in 4-bit
    2. RotatE KG embeddings                       — frozen (from Stage 1)
    3. KG → LM Projection layer                  — trainable (from Stage 2)
    4. LoRA adapters on attention layers          — trainable

Only ~0.5% of parameters are trained (projection + LoRA adapters).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BioKGLoRA(nn.Module):
    """
    Wraps a PEFT-LoRA LLM and injects KG knowledge via embedding augmentation.

    Call hierarchy for a forward pass:
        1. Look up token embeddings from frozen base LLM
        2. For each entity span annotated in the input:
              a. Look up RotatE KG embedding
              b. Project to LM embedding space
              c. Fuse with token embeddings (weighted sum)
        3. Pass fused embeddings through LLM + LoRA adapters
        4. Return logits (standard causal LM output)

    Args:
        base_model_name:  HuggingFace model id
        kg_embedding_path: path to entity_embeddings.pt  (E, 256)
        entity2id_path:    path to entity2id.json
        projection_ckpt:   path to saved projection layer weights
        kg_dim:            RotatE embedding dim (256)
        lm_dim:            LLM hidden dim (4096 for Llama-3-8B)
        kg_weight:         alpha for weighted fusion (0 = no KG, 1 = only KG)
        lora_rank:         LoRA rank
        lora_alpha:        LoRA scaling factor
        lora_dropout:      LoRA dropout
        quantization:      "4bit", "8bit", or None
    """

    def __init__(
        self,
        base_model_name: str = "google/gemma-4-e2b-it",
        kg_embedding_path: Optional[str] = None,
        entity2id_path: Optional[str] = None,
        projection_ckpt: Optional[str] = None,
        kg_dim: int = 256,
        lm_dim: int = 2048,
        kg_weight: float = 0.3,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        quantization: Optional[str] = "4bit",
        target_modules: Optional[List[str]] = None,
        apply_lora: bool = True,
    ):
        super().__init__()
        self.kg_weight = kg_weight
        self.lm_dim = lm_dim

        # ── 1. Load base LLM (4-bit quantised, frozen) ──────────────────────
        self.base_llm = self._load_base_llm(base_model_name, quantization)
        for param in self.base_llm.parameters():
            param.requires_grad = False

        # ── 2. KG entity embeddings (frozen) ────────────────────────────────
        self.entity2id: Dict[str, int] = {}
        self.register_buffer("kg_entity_embeddings", torch.zeros(1, kg_dim))
        if kg_embedding_path and entity2id_path:
            self._load_kg_embeddings(kg_embedding_path, entity2id_path)

        # ── 3. Projection layer (trainable) ─────────────────────────────────
        from .projection import KGProjectionLayer
        self.kg_projection = KGProjectionLayer(kg_dim, 1024, lm_dim)
        if projection_ckpt and Path(projection_ckpt).exists():
            state = torch.load(projection_ckpt, map_location="cpu")
            self.kg_projection.load_state_dict(state)
            logger.info("Loaded projection layer from %s", projection_ckpt)

        # base_llm is already placed on-device via device_map="auto" — match it,
        # since kg_projection is a freshly created submodule that defaults to CPU.
        embedding_device = self.base_llm.get_input_embeddings().weight.device
        self.kg_projection = self.kg_projection.to(embedding_device)

        # ── 4. Apply LoRA adapters ───────────────────────────────────────────
        # target_modules=None lets PEFT pick its own architecture-aware defaults
        # (needed for Gemma-4: its non-text submodules use a custom
        # Gemma4ClippableLinear class that a hardcoded ["q_proj", ...] list would
        # incorrectly match — see https://github.com/huggingface/peft/issues/3129).
        # apply_lora=False (used by load_for_inference) skips this: a trained
        # adapter gets loaded via PeftModel.from_pretrained instead, which builds
        # its own LoRA wrap from adapter_config.json — applying both would nest
        # PEFT wrapping twice and corrupt every loaded weight's key path.
        if apply_lora:
            self.base_llm = self._apply_lora(
                self.base_llm, lora_rank, lora_alpha, lora_dropout, target_modules
            )

    # ── Loading helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_base_llm(model_name: str, quantization: Optional[str]):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info("Loading base LLM: %s (quantization=%s)", model_name, quantization)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not quantization else None,
            trust_remote_code=True,
        )
        return model

    def _load_kg_embeddings(self, emb_path: str, entity2id_path: str) -> None:
        import json
        with open(entity2id_path) as f:
            self.entity2id = json.load(f)
        embs = torch.load(emb_path, map_location="cpu")   # (E, kg_dim)
        self.register_buffer("kg_entity_embeddings", embs)
        logger.info(
            "Loaded KG embeddings: %s entities, dim=%d", len(self.entity2id), embs.shape[1]
        )

    @staticmethod
    def _apply_lora(model, rank: int, alpha: int, dropout: float, target_modules: Optional[List[str]]):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Prepare quantised model for training
        model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    # ── Per-layer embeddings (Gemma-4-specific) ─────────────────────────────────

    @staticmethod
    def _unwrap_to_backbone(model: nn.Module) -> Optional[nn.Module]:
        """
        Walk down through PEFT's wrapping (PeftModel -> LoraModel -> ...) and,
        for multimodal checkpoints, into `.language_model`, to find the actual
        backbone module that exposes `get_per_layer_inputs` (Gemma-4's
        Per-Layer Embeddings). `google/gemma-4-e2b-it` loads as a multimodal
        wrapper (vision/audio/text) even though we only use it for text, so
        the text backbone sits under `.model.language_model`, not directly
        under `.model`. Returns None for architectures without PLE (e.g.
        Llama) or if the expected attribute chain isn't found.
        """
        seen = set()
        obj = model
        for _ in range(8):
            if hasattr(obj, "get_per_layer_inputs"):
                return obj
            nxt = None
            for attr in ("base_model", "model", "language_model"):
                candidate = getattr(obj, attr, None)
                if candidate is not None and candidate is not obj and id(candidate) not in seen:
                    nxt = candidate
                    break
            if nxt is None:
                return None
            seen.add(id(nxt))
            obj = nxt
        return None

    def _get_per_layer_inputs(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Precompute Gemma-4's Per-Layer Embeddings from the real, unmodified
        `input_ids` — before KG fusion overwrites `inputs_embeds` at entity-span
        positions. This must happen with real input_ids: if the model instead
        has to derive them from (KG-augmented) inputs_embeds, it reverse-looks-up
        every position against the full vocab embedding table, which both
        explodes memory (O(batch * seq_len * vocab_size * hidden_size)) and
        produces wrong results wherever we've overwritten the embedding.
        Returns None for architectures without this feature (e.g. Llama).
        """
        backbone = self._unwrap_to_backbone(self.base_llm)
        if backbone is None or not getattr(backbone, "hidden_size_per_layer_input", None):
            return None
        with torch.no_grad():
            return backbone.get_per_layer_inputs(input_ids=input_ids, inputs_embeds=None)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        entity_spans: Optional[List[List[Tuple[int, int, str]]]] = None,
    ) -> "ModelOutput":
        """
        Forward pass with optional KG augmentation.

        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            labels:         (B, L) for loss computation (-100 at ignored positions)
            entity_spans:   list[list[(token_start, token_end, entity_name)]]
                            — outer list is batch dim, inner is entity spans per sample

        Returns:
            CausalLMOutputWithPast (huggingface style, has .loss and .logits)
        """
        # Get base token embeddings
        inputs_embeds = self.base_llm.get_input_embeddings()(input_ids)   # (B, L, d)

        # Precompute from real input_ids before KG fusion touches inputs_embeds
        per_layer_inputs = self._get_per_layer_inputs(input_ids)

        # Augment with KG embeddings where entity spans are provided
        if entity_spans is not None and len(self.entity2id) > 0:
            inputs_embeds = self._augment_with_kg(inputs_embeds, entity_spans)

        forward_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        if per_layer_inputs is not None:
            forward_kwargs["per_layer_inputs"] = per_layer_inputs

        outputs = self.base_llm(**forward_kwargs)
        return outputs

    def _augment_with_kg(
        self,
        token_embs: torch.Tensor,   # (B, L, d)
        entity_spans: List[List[Tuple[int, int, str]]],
    ) -> torch.Tensor:
        """Fuse KG embeddings into token embeddings at entity span positions."""
        device = token_embs.device
        augmented = token_embs.clone()

        for batch_idx, spans in enumerate(entity_spans):
            for start, end, entity_name in spans:
                if entity_name not in self.entity2id:
                    continue
                eid = self.entity2id[entity_name]
                kg_emb = self.kg_entity_embeddings[eid].to(device)         # (kg_dim,)
                kg_proj = self.kg_projection(kg_emb.unsqueeze(0)).squeeze(0)  # (lm_dim,)

                # Weighted addition over the span
                span_mean = augmented[batch_idx, start:end].mean(dim=0)     # (lm_dim,)
                fused = (1 - self.kg_weight) * span_mean + self.kg_weight * kg_proj
                augmented[batch_idx, start:end] = fused.unsqueeze(0).expand(end - start, -1)

        return augmented

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_spans: Optional[List[List[Tuple[int, int, str]]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens with KG-augmented input embeddings.

        repetition_penalty / no_repeat_ngram_size default to values that curb
        degenerate repetition loops — relevant because training data has no
        explicit EOS signal on older checkpoints, so the model doesn't reliably
        know when to stop and can drift into repeating phrases indefinitely.
        """
        inputs_embeds = self.base_llm.get_input_embeddings()(input_ids)

        # Precompute from real input_ids before KG fusion touches inputs_embeds.
        # Only the prompt's first forward pass needs this — once past_key_values
        # exist, subsequent decoding steps use the real id of each newly
        # generated token and compute their own per-layer input normally.
        per_layer_inputs = self._get_per_layer_inputs(input_ids)

        if entity_spans is not None and len(self.entity2id) > 0:
            inputs_embeds = self._augment_with_kg(inputs_embeds, entity_spans)

        generate_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if per_layer_inputs is not None:
            generate_kwargs["per_layer_inputs"] = per_layer_inputs
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        return self.base_llm.generate(**generate_kwargs)

    # ── Parameter groups ──────────────────────────────────────────────────────

    def trainable_parameters(self):
        """Return only trainable parameters (projection + LoRA)."""
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapters(self, output_dir: str) -> None:
        """Save LoRA adapters and projection layer to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.base_llm.save_pretrained(str(out / "lora_adapters"))
        torch.save(self.kg_projection.state_dict(), str(out / "projection.pt"))
        logger.info("Saved adapters to %s", output_dir)

    @classmethod
    def load_for_inference(
        cls,
        adapter_dir: str,
        base_model_name: str,
        kg_embedding_path: str,
        entity2id_path: str,
        projection_ckpt: Optional[str] = None,
        **kwargs,
    ) -> "BioKGLoRA":
        """Load a trained BioKG-LoRA model for inference."""
        from peft import PeftModel

        model = cls(
            base_model_name=base_model_name,
            kg_embedding_path=kg_embedding_path,
            entity2id_path=entity2id_path,
            projection_ckpt=projection_ckpt or str(Path(adapter_dir) / "projection.pt"),
            apply_lora=False,
            **kwargs,
        )
        # Wrap with the trained LoRA adapter (single wrap — __init__ skipped its
        # own via apply_lora=False, so this builds the LoRA structure from
        # adapter_config.json and loads the matching trained weights into it).
        model.base_llm = PeftModel.from_pretrained(
            model.base_llm, str(Path(adapter_dir) / "lora_adapters")
        )
        return model
