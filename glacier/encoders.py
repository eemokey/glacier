import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizerFast

from chemprop import data as chemprop_data, models as chemprop_models
from chemprop import featurizers, nn as chemprop_nn
from chemprop.models import MPNN 

from data.dataloader import BatchMolGraph


# =====================================================================
# Base Encoder Class
# =====================================================================

class BaseEncoder(nn.Module, ABC):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            batch (TrainingBatch): The batch object from your custom dataloader.
        Returns:
            torch.Tensor: A tensor of shape [batch_size, output_dim]
        """
        pass
# =====================================================================
# SMILES Transformer Encoder
# =====================================================================

class TransformerEncoder(nn.Module):
    def __init__(self, output_dim=512, 
                 d_model=128,  
                   nhead=8, 
                 num_layers=2,  
                 max_len=512):
        super().__init__()
        self.d_model = d_model
        

        # get tokenizer path 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(current_dir, "tokenizer.json")
        

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"GLACIER Tokenizer set not found at: {tokenizer_path}. "
                "Ensure tokenizer.json is placed in your directory."
            )


        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path,
            unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")

        # 2. Embeddings 
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, d_model, padding_idx=self.tokenizer.pad_token_id)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model)) # Learnable pos is often better for SMILES
        self.dropout = nn.Dropout(0.1)
        
        # 3. Encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        
        # 4. Dimension match for Fusion
        self.output_map = nn.Linear(d_model, output_dim) if d_model != output_dim else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02) # Truncated normal is more stable than Xavier for LLMs
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, batch):

        if 'input_ids' not in batch:
            inputs = self.tokenizer(batch['smiles'], padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = inputs['input_ids'].to(self.embedding.weight.device)
            attention_mask = inputs['attention_mask'].to(self.embedding.weight.device)
        else:
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)

        # Embedding + Pos
        x = self.embedding(input_ids) 
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Transform (PyTorch expects src_key_padding_mask where True = masked)
        padding_mask = (attention_mask == 0) if attention_mask is not None else None
        z = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # POOLING: Use [CLS] Token (Index 0) 
        mol_embedding = z[:, 0, :] 
        
        return self.output_map(self.final_norm(mol_embedding))
    
    

# =====================================================================
# Graph MPNN Encoder
# =====================================================================

class MPNNEncoder(BaseEncoder):
    """
    Wrapper for MPNN to act as a modular graph encoder.
    """
    def __init__(self, mpnn_model=None, output_dim: int = 512):
        """
        Args:
            mpnn_model: A pre-initialized chemprop.models.MPNN object.
            output_dim: The output dimension of the MPNN.
        """
        super().__init__(output_dim)

        if mpnn_model is None: 
            agg = chemprop_nn.AttentiveAggregation(output_size=300) 
            mp = chemprop_nn.BondMessagePassing()
            output_transform = torch.nn.Identity() #  https://github.com/JacksonBurns/chemeleon/blob/ec0bb30b5c0a84d38c2aa0a3212c2765f5865b66/models/chemprop_mpnn/evaluate.py#L31 
            ffn = chemprop_nn.BinaryClassificationFFN(output_transform=output_transform, input_dim=mp.output_dim ) # Predictor 
            mpnn_model = MPNN(mp, agg, ffn, batch_norm=False, warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)
        self.mpnn = mpnn_model

        # Get chemprop embedding dim 
        self.gnn_unique_embed_dim = self.mpnn.message_passing.output_dim 
        

        self.projection = nn.Sequential(
            nn.Linear(self.gnn_unique_embed_dim, self.gnn_unique_embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.gnn_unique_embed_dim),
            nn.Dropout(0.1), 
            nn.Linear(self.gnn_unique_embed_dim, self.gnn_unique_embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.gnn_unique_embed_dim),
            nn.Linear(self.gnn_unique_embed_dim, output_dim)
        )

    def forward(self, batch) -> torch.Tensor:
        # # 1. Reconstruct BatchMolGraph
        mol_graph = BatchMolGraph.from_tensors(
            V=batch['graph_V'],
            E=batch['graph_E'],
            edge_index=batch['graph_edge_index'],
            rev_edge_index=batch['graph_rev_edge_index'],
            batch=batch['graph_batch'],
            size=batch['graph_size']
        )


        # Pass through Chemprop MPNN
        embeddings = self.mpnn.fingerprint(mol_graph) 

        # Apply LayerNorm
        embeddings = self.projection(embeddings)
        
        return embeddings
    

# =====================================================================
# 4. Tabular MLP Descriptor Encoder
# =====================================================================


class TabularMLPEncoder(nn.Module):
    """
    A simple yet effective MLP encoder for tabular data
    
    It takes Rdkit descriptors
    passes them through hidden layers, and projects them to a shared embedding dimension
    """
    def __init__(self, output_dim: int = 512, input_dim: int = 217,  
                 hidden_dims: list[int] = [128, 64], 
                     dropout: float = 0.1):
        """
        Args:
            input_dim (int): The total size of the input vector (num_numerical + one-hot dims)
            hidden_dims (list[int]): A list of integers defining the size of hidden layers
            output_dim (int): The target shared dimension (e.g., 512 or 768)
            dropout (float): Dropout probability for regularization
        """
        super().__init__()
        
        layers = []
        curr_dim = input_dim

        #  norm (scale values) 
        self.input_norm = nn.LayerNorm(input_dim) 
        
        # Build the hidden layers 
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)) 
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        self.mlp_body = nn.Sequential(*layers)


        self.projection = nn.Sequential(
            nn.Linear(curr_dim, output_dim),  # unique is 217 for rdkit                      
            nn.LayerNorm(output_dim))
        
    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Input_Dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Output_Dim)
        """
        x = batch['rdkit']

        # 0. scale values 
        # Apply Input Batch Norm (Scales data to safe range)
        x_scaled = self.input_norm(x)

        # 1. Pass through the deep dense layers
        features = self.mlp_body(x_scaled)

        
        # 2. Project to the shared dimension
        embeddings = self.projection(features)
    
        
        return embeddings
    


