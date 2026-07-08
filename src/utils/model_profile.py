"""
Resolves per-model settings (lm_dim, target_modules) from config model_profiles.
"""
from __future__ import annotations
from typing import Dict, List, Any


_FALLBACK_PROFILES: Dict[str, Dict[str, Any]] = {
    "meta-llama/Llama-3-8B": {
        "lm_dim": 4096,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    "google/gemma-4-e2b-it": {
        "lm_dim": 1536,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
}


def resolve_model_profile(cfg: Dict) -> Dict[str, Any]:
    """
    Return the model profile for the active base_model.

    Looks up cfg['model_profiles'][base_model] first; falls back to built-in
    defaults so old configs without a model_profiles section still work.

    Returns a dict with at least:
        lm_dim          (int)
        target_modules  (list[str])
    """
    base_model: str = cfg["lora"]["base_model"]
    profiles: Dict = cfg.get("model_profiles") or {}

    if base_model in profiles:
        profile = dict(profiles[base_model])
    elif base_model in _FALLBACK_PROFILES:
        profile = dict(_FALLBACK_PROFILES[base_model])
    else:
        raise ValueError(
            f"No model profile found for '{base_model}'. "
            f"Add an entry to config.yaml under model_profiles."
        )

    # Allow projection.lm_dim to override when explicitly set (not null)
    explicit_lm_dim = (cfg.get("projection") or {}).get("lm_dim")
    if explicit_lm_dim is not None:
        profile["lm_dim"] = explicit_lm_dim

    # Allow lora.target_modules to override when explicitly set (not null)
    explicit_targets: List[str] | None = cfg["lora"].get("target_modules")
    if explicit_targets is not None:
        profile["target_modules"] = explicit_targets

    return profile
