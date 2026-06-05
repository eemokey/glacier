
from typing import List

class GlacierConfig():
    """
    Configuration class for the GLACIER.
    Inherits from PretrainedConfig to support HF Hub serialization, 
    from_pretrained(), and save_pretrained().
    """
    model_type = "Glacier"
    architectures = ["Glacier"]

    def __init__(
        self,
        fusion_dim: int = 512,
        output_dim: int = 512,
        num_teachers: int = 2,
        graph_lr: float = 1e-3,
        text_lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 250,
        modality_dropout: float = 0.1,
        teacher_input_dims: List[int] = [512, 768],
        rdkit_input_dim: int = 217, 
        **kwargs
    ):
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim
        self.num_teachers = num_teachers
        self.graph_lr = graph_lr
        self.text_lr = text_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.modality_dropout = modality_dropout
        self.teacher_input_dims = teacher_input_dims
        self.rdkit_input_dim = rdkit_input_dim
        
        # Capture any leftover HF metadata (like 'architectures' or 'model_type') 
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        """Used by the ModelHubMixin to serialize the config to JSON."""
        return self.__dict__.copy()