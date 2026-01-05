import torch
import torch.nn as nn

from submodules.ProPE import PropeDotProductAttention


class AtomicAttnPEHelper:
    """A helper for positional embedding on the atomatic multi-head attention mechanism. 
    """
    
    def __init__(self, pe_type="prope", **kwargs):
        # super().__init__()
        if pe_type == "prope":
            self.init_prope(**kwargs)
        elif pe_type == "rope":
            pass
        else:
            raise NotImplementedError(f"Undefined PE type {pe_type}.")
    
    def init_prope(self, **kwargs):
        head_dim     = kwargs["head_dim"]
        patches_x    = kwargs["patches_x"]
        patches_y    = kwargs["patches_y"]
        image_width  = kwargs["image_width"]
        image_height = kwargs["image_height"]
        viewmats_tar = kwargs["viewmats_tar"]
        viewmats_src = kwargs["viewmats_src"]
        Ks_tar       = kwargs["Ks_tar"]
        Ks_src       = kwargs["Ks_src"]
        n_cameras    = kwargs.get("n_cameras", None)
        
        self.prope_attn_tar = PropeDotProductAttention(head_dim=head_dim, 
                                                       patches_x=patches_x, patches_y=patches_y, 
                                                       image_width=image_width, image_height=image_height, n_cameras=n_cameras)
        self.prope_attn_src = PropeDotProductAttention(head_dim=head_dim, 
                                                       patches_x=patches_x, patches_y=patches_y, 
                                                       image_width=image_width, image_height=image_height, n_cameras=n_cameras)
        
        self.prope_attn_tar._precompute_and_cache_apply_fns(viewmats=viewmats_tar, Ks=Ks_tar)
        self.prope_attn_src._precompute_and_cache_apply_fns(viewmats=viewmats_src, Ks=Ks_src)
        
        self.q_embedder = self.prope_attn_tar._apply_to_q
        self.k_embedder = self.prope_attn_src._apply_to_kv
        self.v_embedder = self.prope_attn_src._apply_to_kv
        self.o_embedder = self.prope_attn_tar._apply_to_o
        
    def init_rope(self, **kwargs):
        raise NotImplementedError("Initialization for RoPE is not implemented.")
    
    def pe_q(self, x):
        return self.q_embedder(x)
    
    def pe_k(self, x):
        return self.k_embedder(x)
    
    def pe_v(self, x):
        return self.v_embedder(x)
    
    def pe_o(self, x):
        return self.o_embedder(x)
