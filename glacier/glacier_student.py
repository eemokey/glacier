import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Dict, Any, Optional
import json
import os 


from transformers import get_cosine_schedule_with_warmup
from huggingface_hub import PyTorchModelHubMixin


from encoders import TransformerEncoder, MPNNEncoder, TabularMLPEncoder
from fusion import FinslerFusion, ConcatenationFusion
from configuration import GlacierConfig

# =====================================================================
#  Contrastive Multi-Teacher Loss
# =====================================================================

class DynamicMultiTeacherInfoNCELoss(nn.Module):
    """
    Computes weighted InfoNCE loss
    
    """
    def __init__(self, student_dim, num_teachers, temperature=0.2, min_trust=0.1):
        super().__init__()
        self.temperature = temperature
        self.min_trust = min_trust # Prevent the model from ignoring a teacher completely
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none') 
        
        # Internal Trust Head
        self.trust_net = nn.Sequential(
            nn.Linear(student_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_teachers) 
        )
        
        nn.init.constant_(self.trust_net[-1].bias, 4.0)

    def forward(self, student_features, teachers_features_tensor):
        """
        Args:
            student_features: (Batch, Embed_Dim)
            teachers_features_tensor: (Batch, N_Teachers, Embed_Dim)
        """
        # 1. Predict Trust Scores internally
        trust_logits = self.trust_net(student_features.detach())
        
        # Apply Trust Floor
        sigmoid_trust = torch.sigmoid(trust_logits)
        trust_scores = sigmoid_trust * (1.0 - self.min_trust) + self.min_trust
        
        batch_size, num_teachers, _ = teachers_features_tensor.shape
        device = student_features.device
        
        # Normalize Student
        student_norm = F.normalize(student_features, dim=1)
        labels = torch.arange(batch_size).to(device)
        
        total_loss = 0.0
        
        # Iterate over teachers
        for i in range(num_teachers):
            # A. Get Teacher Features
            teacher_features = teachers_features_tensor[:, i, :]
            teacher_norm = F.normalize(teacher_features, dim=1)
            
            # B. Standard InfoNCE Logic
            logits = torch.matmul(student_norm, teacher_norm.T) / self.temperature
            
            # C. Calculate Raw Loss (Per sample!)
            raw_loss = self.cross_entropy(logits, labels)
            
            # D. Apply Dynamic Trust Weighting
            current_trust = trust_scores[:, i] 
            
            # Attenuate the loss: If trust is low, the gradient from raw_loss is killed.
            weighted_loss = raw_loss * current_trust
            
            # Regularizer: Prevents trust from collapsing to zero.
            regularizer = -torch.log(current_trust + 1e-6)
            
            # Combine
            total_loss += (weighted_loss + regularizer).mean()

        return total_loss
    


# =====================================================================
#  Base Student & Teacher Projectors
# =====================================================================
class StudentProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):#2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class TeacherAdapter(nn.Module):
    def __init__(self, input_dim, shared_dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = shared_dim 

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), 
            nn.Linear(hidden_dim, shared_dim),
            nn.LayerNorm(shared_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        """
        Force Orthogonal Initialization.
        This guarantees the matrix is full-rank (preserves all dimensions) 
        at the start of training.
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, teacher_embeddings):
        # Ensure input is detached and float32
        x = teacher_embeddings.detach().float()
        
        return self.net(x)
    
# =====================================================================
#  Main Lightning GLACIER Foundation Model
# =====================================================================

class Glacier(pl.LightningModule, PyTorchModelHubMixin):
    """
    Main multimodal student foundation model. Uses PyTorch Lightning for
    seamless scaling, model checkpointing, and GPU acceleration.
    """
    def __init__(
        self,
        config: Optional[GlacierConfig] = None,
        graph_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        rdkit_encoder: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['graph_encoder', 'text_encoder', 'rdkit_encoder'])
        

        # HF integration 
        if config is None:
            config = GlacierConfig(**kwargs)
        elif isinstance(config, dict):
            # When from_pretrained loads config.json into obj
            combined_kwargs = {**config, **kwargs}
            config = GlacierConfig(**combined_kwargs)
            
        self.config = config
        

        # set encoders if not exist 
        if text_encoder is None:
            text_encoder = TransformerEncoder(output_dim=self.config.fusion_dim)
        if graph_encoder is None:
            graph_encoder = MPNNEncoder(output_dim=self.config.fusion_dim)
        if rdkit_encoder is None:
            rdkit_encoder = TabularMLPEncoder(input_dim=217, output_dim=self.config.fusion_dim)

        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.rdkit_encoder = rdkit_encoder
        

        # Setup standard Finsler geometry attention fusion head
        self.fusion = FinslerFusion(input_dim_fusion=self.config.fusion_dim, output_dim_fusion=self.config.output_dim)
        
        # Projection space for contrastive teacher alignment
        self.student_projector = StudentProjector(input_dim=self.config.output_dim, output_dim=self.config.output_dim)
        self.teacher_adapters = nn.ModuleList([
            TeacherAdapter(
                input_dim=dim, 
                shared_dim=self.config.output_dim,
                hidden_dim=int(self.config.output_dim)
            )
            for dim in self.config.teacher_input_dims
        ])

        self.loss_fn = DynamicMultiTeacherInfoNCELoss(
            student_dim=self.config.output_dim, 
            num_teachers=len(self.config.teacher_input_dims)
        )


    def forward(self, batch) -> torch.Tensor:
        """
        Full forward pass of the fusion model. Runs inputs through encoders and then through the FusionHead.

        Args
            batch: batch of data
        
        Returns
            embeddings
        """
        # Encode modalities
        # Graph Embed 
        graph_emb = self.graph_encoder(batch) # [Batch, D]
        # Text Embed 
        text_emb = self.text_encoder(batch)   # [Batch, D]
        # RDKIT Embed 
        rdkit_embs = self.rdkit_encoder(batch) # [Batch, D]

        # Dropout
        is_stable = False
        # Verify trainer status and check if training is active 
        if self.training and self.trainer is not None:
            # Safe access to callback metrics during active run
            current_loss = self.trainer.callback_metrics.get('train_loss')
            current_loss_val = current_loss.item() if current_loss is not None else 10.0
            is_stable = (self.current_epoch >= 5) or (current_loss_val < 4.0)
        if is_stable:
            dice = torch.rand(1).item()
            p = self.config.modality_dropout
            
            thresh_text = p * 0.10
            thresh_graph = thresh_text + (p * 0.45)
            thresh_rdkit = thresh_graph + (p * 0.45)
            
            if dice < thresh_text:
                text_emb = torch.zeros_like(text_emb)
            elif dice < thresh_graph:
                graph_emb = torch.zeros_like(graph_emb)
            elif dice < thresh_rdkit:
                rdkit_embs = torch.zeros_like(rdkit_embs)

        ## Fuse the embeddings 
        fused_student_embedding = self.fusion(graph_emb, text_emb, rdkit_embs) 
        
        return fused_student_embedding

    def get_teacher_embeddings(self, batch):
        """
        Projects and norms each teacher embedding in list to shared dim 
        
        Returns teachers as [total samples, N teachers, shared dim]
        """

        # 2. Get Teacher Embeddings from the Batch
        raw_teacher_embs = batch['teacher_embs']   # list of teacher embeddings of unique size 

        # 2. Teacher adapter project  
        projected_teachers = []
        for raw_emb, nn_adapter in zip(raw_teacher_embs, self.teacher_adapters):
            proj = nn_adapter(raw_emb)
            projected_teachers.append(proj)
            
        # 3. Stack for the Loss Function
        teacher_stack = torch.stack(projected_teachers, dim=1)

        return teacher_stack


    def training_step(self, batch, batch_idx: int):
        # 1. Get the single, fused student embedding
        student_emb_fused = self(batch) # get_student_embedding forward pass 

        # 2. Apply Projector 
        student_emb_projected = self.student_projector(student_emb_fused)
        
        # 2. Get Teacher Embeddings from the Batch
        teacher_embs = self.get_teacher_embeddings(batch)

        # 3. Compute Multi-Teacher Contrastive Loss
        loss = self.loss_fn(student_emb_projected, teacher_embs)
        
        # 4. Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        # Don't log everys step, save time
        
        return loss

    def validation_step(self, batch, batch_idx: int):
            # 1. Get the single, fused student embedding
            student_emb_fused = self(batch)

            # 2. Apply Projector 
            student_emb_projected = self.student_projector(student_emb_fused)
            
            # 3. Get Teacher Embeddings from the Batch
            teacher_embs = self.get_teacher_embeddings(batch)

            # 4. Compute Multi-Teacher Contrastive Loss
            loss = self.loss_fn(student_emb_projected, teacher_embs)
            
            # 5. Logging (Optimized for Validation)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
            
            return loss



    def configure_optimizers(self):
        """
        Builds optimizer with parameter groups for discriminative learning rates.
        Optimized via Cosine Annealing with Warmup scheduler.
        """
        text_params = list(self.text_encoder.parameters())
        text_param_ids = set(id(p) for p in text_params)
        
        # Collect all remaining network params (Graph, Tabular, Fusion, Projector, Loss)
        other_params = [p for p in self.parameters() if id(p) not in text_param_ids]
        
        optimizer = torch.optim.AdamW([
            {
                "params": other_params, 
                "lr": self.config.graph_lr,
                "weight_decay": self.config.weight_decay
            },
            {
                "params": text_params, 
                "lr": self.config.text_lr, 
                "weight_decay": self.config.weight_decay
            }
        ])

        # estimate total steps for the scheduler
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except Exception:
            total_steps = float('inf')

        if total_steps == float('inf') or total_steps <= 0:
            # Secure fallback using max_epochs and a standard default or datamodule size
            dataset_size = 100000
            batch_size = 1024
            if hasattr(self, 'trainer') and self.trainer is not None and getattr(self.trainer, 'datamodule', None) is not None:
                try:
                    dataset_size = len(self.trainer.datamodule.train_dataloader())
                    batch_size = 1
                except Exception:
                    pass
            total_steps =  self.config.max_epochs * (dataset_size // batch_size)

        # Configure standard 5% warm-up steps
        warmup_steps = int(0.05 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _save_pretrained(self, save_directory: str) -> None:
        """
        Overwritten serialization logic executed by self.save_pretrained().
        Dumps config settings and parameters automatically.
        """
        # Save serialization configuration file
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save underlying parameters
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save companion BPE or custom chemical tokenizer models
        if hasattr(self.text_encoder, "tokenizer"):
            self.text_encoder.tokenizer.save_pretrained(save_directory)
