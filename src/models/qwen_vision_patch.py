import torch
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel


def patched_visual_forward(
    self, hidden_states, grid_thw, output_hidden_states=False, **kwargs
):
    """Patched forward method that supports output_hidden_states"""
    # Call the original method if not requesting hidden states
    original_forward = self._original_visual_forward
    
    if not output_hidden_states:
        return original_forward(hidden_states, grid_thw, **kwargs)
    
    # Replicate the exact original forward logic with hidden state collection
    all_hidden_states = []
    
    # Patch embedding
    hidden_states = self.patch_embed(hidden_states)
    
    # Rotary position embedding and windowing setup
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # Process through vision blocks, collecting hidden states BEFORE each block
    for layer_num, blk in enumerate(self.blocks):
        # Collect hidden states before processing through the block
        all_hidden_states.append(hidden_states)
        
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    # Collect final hidden states before merger
    all_hidden_states.append(hidden_states)

    # Final merger
    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]
    
    # Collect final processed hidden states
    all_hidden_states.append(hidden_states)
    
    return hidden_states, tuple(all_hidden_states)


def patch_qwen_vision_model(model):
    """Monkey patch the vision model to support output_hidden_states"""
    visual = model.model.visual

    # Store original forward method
    visual._original_visual_forward = visual.forward

    # Replace with patched version
    visual.forward = patched_visual_forward.__get__(visual, visual.__class__)

    return model
