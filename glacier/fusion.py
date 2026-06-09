import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Base Fusion Class
# =====================================================================

class FusionMechanismBase(nn.Module):
    """
    Abstract base class for mechanisms that fuse latent representations from two models
    """
    def __init__(self, input_dim_fusion: int, output_dim_fusion: int, fusion_name:str):
        super().__init__()
        self.input_dim_fusion = input_dim_fusion
        self.output_dim_fusion = output_dim_fusion
        self.name = fusion_name

    def forward(self, gnn_embed: torch.Tensor, llm_embed: torch.Tensor, rdkit=None ) -> torch.Tensor:
        """
        Fuses the representations from two models

        Args
            gnn_embed: Tensor from model A, typically (batch_size, input_dim_gnn)
            llm_embed: Tensor from model B, typically (batch_size, input_dim_llm)
            rdkit: rdkit features 
        
        Returns
            A tensor representing the fused representation
        """
        raise NotImplementedError("Subclasses must implement the fusion logic.")

# =====================================================================
# Finsler Geometry-Aware Fusion
# =====================================================================

class FinslerFusion(FusionMechanismBase):  
    """
    Dynamically aligns and fuses student embeddings within a shared Randers space
    utilizing a Finsler geometry-aware attention model.
    """
    def __init__(self, fusion_name:str = "finsler", input_dim_fusion=256, output_dim_fusion=128):
        super().__init__(input_dim_fusion, output_dim_fusion, fusion_name)
        
        self.scale_factor = input_dim_fusion ** -0.5 
        
        #  Dynamic Finsler Parameters
        self.drift_net = nn.Sequential(
            nn.Linear(input_dim_fusion, input_dim_fusion // 2),
            nn.ReLU(),
            nn.Linear(input_dim_fusion // 2, input_dim_fusion)
        )
        
        # Strong Drift Init 
        nn.init.uniform_(self.drift_net[-1].weight, -0.1, 0.1)
        nn.init.zeros_(self.drift_net[-1].bias)
        
        #  Learnable Parameters 
        self.gate_sensitivity = nn.Parameter(torch.tensor(1.0))
        
        # Dynamic Amplitude Generator 
        self.amp_net = nn.Sequential(
            nn.LayerNorm(input_dim_fusion),
            nn.Linear(input_dim_fusion, 1)
        )
        
        nn.init.zeros_(self.amp_net[1].weight)
        nn.init.constant_(self.amp_net[1].bias, 0.55)
        
        #  MLP Projectors 
        hidden_dim = input_dim_fusion
        self.q_proj = nn.Sequential(nn.Linear(input_dim_fusion, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim_fusion))
        self.k_proj = nn.Sequential(nn.Linear(input_dim_fusion, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim_fusion))
        self.v_proj = nn.Sequential(nn.Linear(input_dim_fusion, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim_fusion))

        # Output Layer 
        self.output_linear = nn.Sequential(
            nn.Linear(input_dim_fusion * 3, output_dim_fusion), 
            nn.LayerNorm(output_dim_fusion)
        )

    def forward(self, gnn_embed: torch.Tensor, llm_embed: torch.Tensor, rdkit=None) -> torch.Tensor: # flag true and in finsler to return confidence 
        if gnn_embed.shape[0] != llm_embed.shape[0]:
            raise ValueError("Batch sizes must match!")
        

        # 1: Define Query and Keys 
        Q = self.q_proj(llm_embed).unsqueeze(1) 

        
        teachers = torch.stack([gnn_embed, rdkit], dim=1)
            
        K = self.k_proj(teachers) 
        V = self.v_proj(teachers) 

        # 2: Dynamic Asymmetric Randers Distance 
        diff = K - Q
        euclidean_dist = torch.norm(diff, p=2, dim=-1) 
        
        
        raw_drift = self.drift_net(llm_embed).unsqueeze(1) 
        
        ###
        # Calculate the L2 norm of the raw drift vector 
        drift_norm = torch.norm(raw_drift, p=2, dim=-1, keepdim=True)
        omega_constrained = (raw_drift / (drift_norm + 1e-8)) * (torch.tanh(drift_norm) * 0.99) # ensure convex 
        ###

        drift_term = (diff * omega_constrained).sum(dim=-1)
        
        randers_dist = euclidean_dist + drift_term

        #  3: Finsler Attention 
        attn_logits = -randers_dist * self.scale_factor
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1) 
        correction = (attn_weights * V).sum(dim=1) 

        # 4: Calibrated Geometric Gating 
        min_dist, _ = randers_dist.min(dim=-1, keepdim=True)
        
        sensitivity = F.softplus(self.gate_sensitivity)
        
        # Dynamic Amplitude Calculation
        amp_cap = F.softplus(self.amp_net(llm_embed)) + 1.0
        

        confidence = amp_cap * torch.sigmoid(-min_dist * self.scale_factor * sensitivity)
        
        # Apply Residual
        refined_llm = llm_embed + (confidence * correction)


        # 5: Fused output 
        concat_repr = torch.cat((gnn_embed, refined_llm, rdkit), dim=-1)
        output = self.output_linear(concat_repr)


        return output

# =====================================================================
# Baseline Concatenation Fusion
# =====================================================================

class ConcatenationFusion(FusionMechanismBase):
    """Simple baseline concatenating inputs directly."""
    def __init__(self, input_dim_fusion: int = 256, output_dim_fusion: int = 128):
        super().__init__(input_dim_fusion, output_dim_fusion, "concat")
     

        self.projector = nn.Sequential(
            nn.Linear(input_dim_fusion * 3, output_dim_fusion),
            nn.LayerNorm(output_dim_fusion)
        )

    def forward(self, gnn_embed: torch.Tensor, llm_embed: torch.Tensor, rdkit: torch.Tensor = None) -> torch.Tensor:
        if gnn_embed.shape[0] != llm_embed.shape[0]:
            raise ValueError("Batch sizes must match to concatenate representations!")
        
        fused = torch.cat((gnn_embed, llm_embed, rdkit), dim=-1)

        return self.projector(fused)