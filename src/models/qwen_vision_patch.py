import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel


def patched_visual_forward(
    self, hidden_states, grid_thw, output_hidden_states=False, **kwargs
):
    """Patched forward method that supports output_hidden_states"""
    # Call the original method
    original_forward = self._original_visual_forward

    if not output_hidden_states:
        return original_forward(hidden_states, grid_thw, **kwargs)

    # If output_hidden_states=True, we need to collect intermediate states
    all_hidden_states = []

    # Process through patch embedding
    hidden_states = self.patch_embed(hidden_states)

    # Get position embeddings and other setup (copied from original)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    # ... (we'll need to replicate the setup logic)

    # Collect hidden states from each block
    for i, blk in enumerate(self.blocks):
        all_hidden_states.append(hidden_states)
        hidden_states = blk(hidden_states, **kwargs)

    all_hidden_states.append(hidden_states)

    # Final processing
    hidden_states = self.merger(hidden_states)

    return hidden_states, tuple(all_hidden_states)


def patch_qwen_vision_model(model):
    """Monkey patch the vision model to support output_hidden_states"""
    visual = model.model.visual

    # Store original forward method
    visual._original_visual_forward = visual.forward

    # Replace with patched version
    visual.forward = patched_visual_forward.__get__(visual, visual.__class__)

    return model
