from typing import Iterable, NamedTuple, Sequence, List
import numpy as np
import torch
from torch import Tensor
import logging
from torch.utils.data import DataLoader
from dataclasses import  field, InitVar, dataclass
from typing import Iterable, NamedTuple, Sequence


from chemprop.data.datasets import (
    MolAtomBondDatum, 
    MoleculeDataset, 
    MolAtomBondDataset, 
    ReactionDataset, 
    MulticomponentDataset
)
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import  BatchMolAtomBondGraph
from chemprop import data as chemprop_data
from chemprop import featurizers

from rdkit import Chem
from rdkit.Chem import Descriptors


logger = logging.getLogger(__name__)


####################3

# RDKIT 

import numpy as np
from older.rdkit import Chem
from rdkit.Chem import Descriptors

def compute_rdkit_descriptors(smiles_list, output_file=None, save=False):
    """
    Computes a 2D matrix of 217 RDKit physicochemical descriptors from a list of SMILES.
    """
    descriptor_list = Descriptors.descList
    num_descriptors = len(descriptor_list)  #  217

    def rdkit_descriptors_from_smiles(smiles_str):
        """Generates a flat, index-matched list of numerical float values for a single SMILES string"""
        if not isinstance(smiles_str, str) or not smiles_str:
            return [0.0] * num_descriptors
            
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return [0.0] * num_descriptors

        values = []
        for name, func in descriptor_list:
            try:
                val = func(mol)
                # Catch None, NaN, or Infinite values immediately
                if val is None or np.isnan(val) or np.isinf(val):
                    values.append(0.0)
                else:
                    values.append(float(val))
            except Exception:
                # Fallback to zero if descriptor calculation crashes on a specific molecule
                values.append(0.0)
        return values
    
    # Compute properties
    feature_matrix = []
    print(f"Computing descriptors for {len(smiles_list)} molecules...")

    # Get rdkit descriptors 
    for count, s in enumerate(smiles_list):
        if count % 100 == 0:
            print(f'Working on molecule {count}')
        
        properties_list = rdkit_descriptors_from_smiles(s)
        feature_matrix.append(properties_list)

    # convert list of numerical lists to a array
    feature_matrix = np.array(feature_matrix, dtype=np.float32)

    # check for NaNs/Infs 
    if np.isnan(feature_matrix).any():
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        print('NaNs/Infs found in RDKit matrix, zeroed out.')

    # Apply Log Scaling (for grad safety)
    feature_matrix = np.sign(feature_matrix) * np.log1p(np.abs(feature_matrix))

    # Clip extremes to ensure smooth gradients
    feature_matrix = np.clip(feature_matrix, -10.0, 10.0)

    #  Save locally if requested
    if save and output_file:
        np.savez(output_file, arr=feature_matrix)
        print(f"Saved descriptors to {output_file}")

    return feature_matrix


###################################







def create_MoleculeDatapoint(smis, ys_flat=None):
    """ create molecule datapoint objects from smiles """

    if ys_flat is not None:
        ys = ys_flat.reshape((len(ys_flat), 1))# reshape ys to be dim (len(ys), 1)
        # 2. Create mol objects 
        all_data = [chemprop_data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    else:
        all_data = [chemprop_data.MoleculeDatapoint.from_smi(smi) for smi in smis]  # NOTE slowest line in this function here. can pass in whatever to store in datapoint if needed in the future
    return all_data 


### Custom for MoleculeDataLoader for Chemprop and LLM to track SMILES across batches ### 
class Datum(NamedTuple):
    """a singular training data point
    
    https://github.com/chemprop/chemprop/blob/420b18834b81aac48963cccdef04e500f1dd0160/chemprop/data/datasets.py#L22 
    """

    mg: MolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None
    smiles: np.ndarray | None # Added 
    rdkit: np.ndarray | None # Added 
    teacher_embs: list[np.ndarray] | None # Added: [N_Teachers with unique Dim] 


class SmilesMoleculeDataset:
    """
    Subclass for MoleculeDataset to track SMILES across batches 

    #     Overriding __getitem__ https://github.com/chemprop/chemprop/blob/420b18834b81aac48963cccdef04e500f1dd0160/chemprop/data/datasets.py#L192 
    
    """
    def __init__(self, smiles, teacher_embeddings=None,  labels=None, rdkit=None,
                 featurizer= featurizers.SimpleMoleculeMolGraphFeaturizer(), 
                  is_train=False):
        
        # preprocess smiles
        preprocessed_smiles = create_MoleculeDatapoint(smiles, labels) # convert to MoleculeDatapoint
        self._original_dataset = chemprop_data.MoleculeDataset(preprocessed_smiles, featurizer) # convert to mpnn MoleculeDataset
        
        ####
        print("Pre-loading SMILES into memory... (this might take a moment)")
        # Force convert to a standard list to kill the latency
        self.cached_smiles = list(self._original_dataset.smiles) # added for speed 
        print(f"Loaded {len(self.cached_smiles)} SMILES strings.")
        ###

        # get rdkit descriptors on the fly 
        self._rdkit  = rdkit # np array 
        if rdkit is None:
            self._rdkit = compute_rdkit_descriptors(smiles) 
        
        self._teacher_embeddings = teacher_embeddings # List of teacher embeddings # Structure: [Teacher_1_Array(N_Samples, Dim1), Teacher_2_Array(N_Samples, Dim2)...]

        self.is_train = is_train # to know to randomize smiles 

    def __getitem__(self, idx: int) -> Datum: # Redefine this function to return smiles
        """
        SPED UP version 
        Docstring for __getitem__
        
        :param self: Description
        :param idx: Description
        :type idx: int
        :return: Description
        :rtype: Datum
        """


        d = self._original_dataset.data[idx]
        mg = self._original_dataset.mg_cache[idx]

        smiles = self.cached_smiles[idx] # DONT do this it is slow self._original_dataset.smiles[idx] # Added
        
        # 2. Augment ONLY if training
        if self.is_train:
            smiles = randomize_smiles(smiles)
       
        
        # get rdkit 
        rdkit = None if self._rdkit is None else self._rdkit[idx] # Added


        # Access teacher embeddings if they exist
        # Iterate through the list of teachers to get the idx-th row for each
        teacher_embs = None
        if self._teacher_embeddings is not None:
            # Result: [Array(Dim1), Array(Dim2), ...] for this specific molecule
            teacher_embs = [t_emb[idx] for t_emb in self._teacher_embeddings]

        return Datum(mg, self._original_dataset.V_ds[idx], self._original_dataset.X_d[idx], 
                     self._original_dataset.Y[idx], d.weight, d.lt_mask, d.gt_mask, smiles, rdkit, teacher_embs)

    
    def __len__(self):
        return len(self._original_dataset)
    





##########################################33





# Adapted from chemprop collate.py to use sped up BatchMolGraph



@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence['MolGraph']):
        self.__size = len(mgs)

        # 1. Quick Extraction (List comprehensions are faster than for-loops with append)
        # We assume mg.V, mg.E etc are numpy arrays based on previous code
        Vs = [mg.V for mg in mgs]
        Es = [mg.E for mg in mgs]
        
        # 2. Vectorized Batch Index Creation
        # Get the number of nodes per graph
        v_lengths = [len(v) for v in Vs]
        
        # Create the batch index [0,0,0, 1,1, 2,2,2...] instantly using numpy
        # This replaces the slow `[i] * len` loop
        batch_idx_np = np.repeat(np.arange(self.__size), v_lengths)
        self.batch = torch.as_tensor(batch_idx_np, dtype=torch.long)

        # 3. Vectorized Offset Calculation
        # We need to shift edge indices by the cumulative number of nodes
        # Calculate offsets: [0, n_nodes_g1, n_nodes_g1+g2, ...]
        node_counts = np.array(v_lengths)
        node_offsets = np.cumsum(np.concatenate(([0], node_counts[:-1])))
        
        # Calculate offsets for reverse edges
        edge_counts = np.array([mg.edge_index.shape[1] for mg in mgs])
        edge_offsets = np.cumsum(np.concatenate(([0], edge_counts[:-1])))

        # 4. Apply Offsets & Stack
        # Zip is fast; adding scalar offset to numpy array is highly optimized
        edge_indexes = [mg.edge_index + off for mg, off in zip(mgs, node_offsets)]
        rev_edge_indexes = [mg.rev_edge_index + off for mg, off in zip(mgs, edge_offsets)]

        # 5. Fast Tensor Creation
        # torch.as_tensor with explicit dtype avoids double-copying (double->float)
        self.V = torch.as_tensor(np.concatenate(Vs), dtype=torch.float32)
        self.E = torch.as_tensor(np.concatenate(Es), dtype=torch.float32)
        
        # hstack is needed for 2xN edge indices
        self.edge_index = torch.as_tensor(np.hstack(edge_indexes), dtype=torch.long)
        self.rev_edge_index = torch.as_tensor(np.concatenate(rev_edge_indexes), dtype=torch.long)

    def __len__(self) -> int:
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        return self # Often useful to return self for chaining
    
    @classmethod
    def from_tensors(cls, V, E, edge_index, rev_edge_index, batch, size):
        """Reconstructs the object from raw tensors (used after collation)."""
        # Create an empty instance
        obj = cls.__new__(cls) 
        
        # Manually set the fields
        obj.V = V
        obj.E = E
        obj.edge_index = edge_index
        obj.rev_edge_index = rev_edge_index
        obj.batch = batch
        obj._BatchMolGraph__size = size # Set the private size attribute
        
        return obj








###########################################3
    
### NOTE Functions below adapted from chemprop.data.collate ### 
# source: https://github.com/chemprop/chemprop/blob/420b18834b81aac48963cccdef04e500f1dd0160/chemprop/data/collate.py 

class TrainingBatch(NamedTuple): # NOTE unused now, replaced with dict for speed 
    """
    A NamedTuple for a batch of data.
    
    Includes the batched molecule graph, feature tensors, target tensors,
    and the SMILES strings for the molecules in the batch
    """
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None
    smiles: Sequence[str]  # Added field for SMILES strings
    rdkit: Tensor | None # Added 
    teacher_embs: list[Tensor] | None # Added: # Structure: [Tensor(Batch, Dim1), Tensor(Batch, Dim2), ...]
   


def collate_batch(batch: Iterable[Datum]) -> dict:  # USED 

    # 1. Unzip the batch
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks, smiles, rdkit, teacher_embs = zip(*batch)
    # only mgs is not none for chemprop 


    # 2. Process Teacher Embeddings
    teacher_embs_list = None
    if teacher_embs[0] is not None:
        teacher_embs_list = [
            torch.as_tensor(np.stack(col), dtype=torch.float32) 
            for col in zip(*teacher_embs)
        ]

    # 3. Create BatchMolGraph Tensors MANUALLY for SPEED 
    # We strip the logic from BatchMolGraph.__post_init__ and run it here.
    
    # A. Extraction (Assuming mg.V, mg.E are numpy arrays)
    Vs = [mg.V for mg in mgs]
    Es = [mg.E for mg in mgs]
    batch_size = len(mgs)

    # B. Vectorized Batch Index Creation
    # Get the number of nodes per graph
    v_lengths = [len(v) for v in Vs]
    
    # Create the batch index [0,0,0, 1,1, 2,2,2...]
    batch_idx_np = np.repeat(np.arange(batch_size), v_lengths)
    
    # C. Vectorized Offset Calculation
    # Calculate node offsets: [0, n_nodes_g1, n_nodes_g1+g2, ...]
    node_counts = np.array(v_lengths)
    node_offsets = np.cumsum(np.concatenate(([0], node_counts[:-1])))
    
    # Calculate edge offsets for reverse edges
    edge_counts = np.array([mg.edge_index.shape[1] for mg in mgs])
    edge_offsets = np.cumsum(np.concatenate(([0], edge_counts[:-1])))

    # D. Apply Offsets & Stack
    # Adjust edge indices by the node offset of their specific graph
    edge_indexes = [mg.edge_index + off for mg, off in zip(mgs, node_offsets)]
    rev_edge_indexes = [mg.rev_edge_index + off for mg, off in zip(mgs, edge_offsets)]

    # E. Final Tensor Creation (These are what we return in the dict)
    # Using as_tensor with explicit dtype is the most memory-efficient method
    graph_V = torch.as_tensor(np.concatenate(Vs), dtype=torch.float32)
    graph_E = torch.as_tensor(np.concatenate(Es), dtype=torch.float32)
    graph_edge_index = torch.as_tensor(np.hstack(edge_indexes), dtype=torch.long)
    graph_rev_edge_index = torch.as_tensor(np.concatenate(rev_edge_indexes), dtype=torch.long)
    graph_batch = torch.as_tensor(batch_idx_np, dtype=torch.long)



    return {
        # Graph Tensors (Raw Tensors pin very fast!)
        "graph_V": graph_V,
        "graph_E": graph_E,
        "graph_edge_index": graph_edge_index,
        "graph_rev_edge_index": graph_rev_edge_index,
        "graph_batch": graph_batch,
        "graph_size": batch_size,

        # ... The rest of your data tensors ...
        "V_ds": None if V_ds[0] is None else torch.as_tensor(np.concatenate(V_ds), dtype=torch.float32),
        "X_d": None if x_ds[0] is None else torch.as_tensor(np.stack(x_ds), dtype=torch.float32),
        "Y": None if ys[0] is None else torch.as_tensor(np.stack(ys), dtype=torch.float32),
        "weights": torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1),
        "lt_mask": None if lt_masks[0] is None else torch.as_tensor(np.stack(lt_masks)),
        "gt_mask": None if gt_masks[0] is None else torch.as_tensor(np.stack(gt_masks)),
        "smiles": None if smiles[0] is None else smiles,
        "rdkit": None if rdkit[0] is None else torch.as_tensor(np.vstack(rdkit), dtype=torch.float32), 
        "teacher_embs": None if teacher_embs_list is None else teacher_embs_list
    }
     


class MolAtomBondTrainingBatch(NamedTuple):
    """
    A NamedTuple for a batch of data for atom/bond-level tasks.
    
    Includes attributes for atom and bond predictions, and the SMILES strings
    """
    bmg: BatchMolAtomBondGraph
    V_d: Tensor | None
    E_d: Tensor | None
    X_d: Tensor | None
    Ys: tuple[Tensor | None, Tensor | None, Tensor | None]
    w: tuple[Tensor | None, Tensor | None, Tensor | None]
    lt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    gt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    constraints: tuple[Tensor | None, Tensor | None]
    smiles: Sequence[str]  # Added field for SMILES strings
    rdkit: Tensor | None
    teacher_embs_tensor : list[Tensor] | None # added



def collate_mol_atom_bond_batch(batch: Iterable[MolAtomBondDatum]) -> MolAtomBondTrainingBatch:
    """
    Collates a batch of MolAtomBondDatum objects into a MolAtomBondTrainingBatch
    
    Modified to handle SMILES strings
    """
    mgs, V_ds, E_ds, x_ds, yss, weights, lt_maskss, gt_maskss, constraintss, smiles, rdkit, teacher_embs  = zip(*batch) # Added smiles

    # Handle stacking of teacher embeddings
    if teacher_embs[0] is None:
        teacher_embs_list = None
    else:
        num_teachers = len(teacher_embs[0])
        teacher_embs_list = []
        for i in range(num_teachers):  # Handles jagged array sizes 
            single_teacher_batch = [sample[i] for sample in teacher_embs]
            tensor = torch.from_numpy(np.stack(single_teacher_batch)).float()
            teacher_embs_list.append(tensor)


    weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    weights_tensors = []
    for ys in zip(*yss):
        if ys[0] is None:
            weights_tensors.append(None)
        elif ys[0].ndim == 1:
            weights_tensors.append(weights)
        else:
            repeats = torch.tensor([y.shape[0] for y in ys])
            weights_tensors.append(torch.repeat_interleave(weights, repeats, dim=0))

    if constraintss[0][0] is None and constraintss[0][1] is None:
        constraintss = None
    else:
        constraintss = [
            None if constraints[0] is None else torch.from_numpy(np.array(constraints)).float()
            for constraints in zip(*constraintss)
        ]

    return MolAtomBondTrainingBatch(
        BatchMolAtomBondGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if E_ds[0] is None else torch.from_numpy(np.concatenate(E_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        [None if ys[0] is None else torch.from_numpy(np.vstack(ys)).float() for ys in zip(*yss)],
        weights_tensors,
        [
            None if lt_masks[0] is None else torch.from_numpy(np.vstack(lt_masks))
            for lt_masks in zip(*lt_maskss)
        ],
        [
            None if gt_masks[0] is None else torch.from_numpy(np.vstack(gt_masks))
            for gt_masks in zip(*gt_maskss)
        ],
        constraintss,
        smiles, # Pass the collected SMILES strings
        None if rdkit[0] is None else torch.from_numpy(np.vstack(rdkit)).float(), # is this right? yes gets tensor [batchsize, 211]
        teacher_embs_list
    )


class MulticomponentTrainingBatch(NamedTuple):
    """
    A NamedTuple for a batch of multicomponent data
    
    Modified to include a list of SMILES for each component
    """
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None
    smiles: List[Sequence[str]]  # Added field for a list of SMILES sequences
    rdkit: Tensor | None
    teacher_embs: List[List[Tensor] | None] # List of Lists of Tensors (Components -> Teachers -> Tensor) #List[Tensor | None] # Added: List of teacher tensors per component


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    """
    Collates a batch of multicomponent data
    
    Modified to aggregate the SMILES strings from each component's batch
    """
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
        [tb.smiles for tb in tbs],  # Collect SMILES from each component batch
        tbs[0].rdkit,
        [tb.teacher_embs for tb in tbs] 
    )



def build_dataloader(
    dataset: MoleculeDataset | MolAtomBondDataset | ReactionDataset | MulticomponentDataset,
    batch_size: int = 1024,
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs,
):
    """ Build a dataloader given MolGraphDataset

    Args 
        dataset : MoleculeDataset | ReactionDataset | MulticomponentDataset
            The dataset containing the molecules or reactions to load.
        batch_size : int, default=64
            the batch size to load.
        num_workers : int, default=0
            the number of workers used to build batches.

    Returns
        a torch.utils.data.DataLoader 

    modified from source: https://github.com/chemprop/chemprop/blob/420b18834b81aac48963cccdef04e500f1dd0160/chemprop/data/dataloader.py#L16
    """

    if isinstance(dataset, MulticomponentDataset):
        collate_fn = collate_multicomponent
    elif isinstance(dataset, MolAtomBondDataset):
        collate_fn = collate_mol_atom_bond_batch
    else:
        collate_fn = collate_batch # NOTE used by hybrid model 


    persistent_workers = False
    pin_memory = False
    if num_workers > 0:
        persistent_workers = True
        pin_memory = True 

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,  # Added for speed 
        pin_memory=pin_memory, # Added 
        **kwargs,
    )


def randomize_smiles(smiles: str) -> str:
    """Returns a random valid SMILES string for the same molecule.
    so instead of taking the canonical smiles, you take an equivalent representation """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return smiles # Fallback if invalid
    
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)


#########################################################################################3

