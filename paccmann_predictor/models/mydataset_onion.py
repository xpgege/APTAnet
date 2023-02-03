# -- coding: utf-8 --
# @Time : 12/9/22 3:24 PM
# @Author : XXXX
# @File : mydataset.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os,pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class Canonicalization():
    """Convert any SMILES to RDKit-canonical SMILES.
    Example:
        An example::

            smiles = 'CN2C(=O)N(C)C(=O)C1=C2N=CN1C'
            c = Canonicalization()
            c(smiles)

        Result is: 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'

    """

    def __init__(self, sanitize: bool = True) -> None:
        """Initialize a canonicalizer

        Args:
            sanitize (bool, optional): Whether molecule is sanitized. Defaults to True.
        """
        self.sanitize = sanitize

    def __call__(self, smiles: str) -> str:
        """
        Forward function of canonicalization.

        Args:
            smiles (str): SMILES string for canonicalization.

        Returns:
            str: Canonicalized SMILES string.
        """
        try:
            canon = Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles, sanitize=self.sanitize), canonical=True
            )
            return canon
        except Exception:
            print(f'\nInvalid SMILES {smiles}, no canonicalization done')
            return smiles

class Augment():
    """Augment a SMILES string, according to Bjerrum (2017)."""

    def __init__(
        self,
        kekule_smiles: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        sanitize: bool = True,
        seed: int = -1,
    ) -> None:
        """NOTE:  These parameter need to be passed down to the enumerator."""

        self.kekule_smiles = kekule_smiles
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.sanitize = sanitize
        self.seed = seed
        if self.seed > -1:
            np.random.seed(self.seed)

    def __call__(self, smiles: str) -> str:
        """
        Apply the transform.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            str: randomized SMILES representation.
        """
        molecule = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
        if molecule is None:
            print(f'\nAugmentation skipped for invalid mol: {smiles}')
            return smiles
        if not self.sanitize:
            molecule.UpdatePropertyCache(strict=False)
        atom_indexes = list(range(molecule.GetNumAtoms()))
        if len(atom_indexes) == 0:  # RDkit error handling
            return smiles
        np.random.shuffle(atom_indexes)
        renumbered_molecule = Chem.RenumberAtoms(molecule, atom_indexes)
        if self.kekule_smiles:
            Chem.Kekulize(renumbered_molecule)

        return Chem.MolToSmiles(
            renumbered_molecule,
            canonical=False,
            kekuleSmiles=self.kekule_smiles,
            allBondsExplicit=self.all_bonds_explicit,
            allHsExplicit=self.all_hs_explicit,
        )

# from transformers import T5EncoderModel, T5Tokenizer

class ProteinSmileDataset(Dataset):
    def __init__(self,
                 affinity_filepath,
                 receptor_filepath,
                 Protein_model,
                 Protein_tokenizer,
                 Protein_padding,
                 ligand_filepath,
                 SMILES_model,
                 SMILES_tokenizer,
                 SMILES_padding,
                 SMILES_argument,
                 SMILES_Canonicalization,
                 device
                 ):
        self.affinity = pd.read_csv(affinity_filepath, sep=",", index_col=0)
        self.affinity=self.affinity[~self.affinity["ligand_name"].isin([120,131,134,137])]

        receptor = pd.read_csv(receptor_filepath, sep="\t", header=None, index_col=1)
        receptor[0] = [' '.join(list(x)) for x in receptor[0]]
        self.receptor=receptor[0].to_dict()

        ligand=pd.read_csv(ligand_filepath,index_col=0)
        self.ligand_SMILES = ligand["SMILES"].to_dict()
        self.ligand_AA= ligand["AA"].to_dict()

        #model
        self.Protein_model=Protein_model
        self.Protein_tokenizer=Protein_tokenizer
        self.SMILES_model=SMILES_model
        self.SMILES_tokenizer=SMILES_tokenizer
        #transform
        self.argument=Augment(sanitize=False)
        self.canonicalization=Canonicalization(sanitize=False)
        self.Protein_padding=Protein_padding
        self.SMILES_padding=SMILES_padding
        self.SMILES_argument=SMILES_argument
        self.SMILES_Canonicalization=SMILES_Canonicalization
        self.device = device


    def __len__(self):
        return self.affinity.shape[0]

    def __getitem__(self, index):
        selected_sample = self.affinity.iloc[index]
        affinity_tensor = torch.tensor(
            [selected_sample["label"]],
            dtype=torch.float
        )
        #根据id获取序列
        receptor_index=selected_sample["sequence_id"]
        receptor_seq=self.receptor[receptor_index]

        ligand_index=selected_sample["ligand_name"]
        ligand_SMILES_seq=self.ligand_SMILES[ligand_index]
        ligand_AA_seq=self.ligand_AA[ligand_index]

        if self.argument:
            ligand_SMILES_seq=self.argument(ligand_SMILES_seq)
        if self.canonicalization:
            ligand_SMILES_seq=self.canonicalization(ligand_SMILES_seq)

        #根据序列获取tensor
        #TCR,T5结尾添加special tokens (</s>)
        ids = self.Protein_tokenizer.batch_encode_plus([receptor_seq],
                                                  add_special_tokens=True,
                                                  padding='max_length',
                                                  max_length=self.Protein_padding+1)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        with torch.no_grad():
            receptor_embedding = self.Protein_model(input_ids=input_ids, attention_mask=attention_mask)
        receptor_embedding = receptor_embedding.last_hidden_state
        receptor_embedding=receptor_embedding[0][1:]

        #peptide,bert开头和结尾分别添加 special tokens ([CLS],[SEP])
        ids = self.SMILES_tokenizer.batch_encode_plus([ligand_SMILES_seq],
                                                  add_special_tokens=True,
                                                  padding='max_length',
                                                  max_length=self.SMILES_padding+1)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        with torch.no_grad():
            SMILES_embedding = self.SMILES_model(input_ids=input_ids, attention_mask=attention_mask)
        SMILES_embedding = SMILES_embedding.last_hidden_state
        SMILES_embedding=SMILES_embedding[0][1:]

        return receptor_seq,ligand_AA_seq,ligand_SMILES_seq,receptor_embedding,SMILES_embedding,affinity_tensor





if __name__=='__main__':
    dataset=ProteinSmileDataset(
                 affinity_filepath='/home/xp/ATPnet/data/tcr_split/fold0/train+covid.csv',
                 receptor_filepath='/home/xp/ATPnet/data/tcr_full.csv',
                 Protein_model="Rostlab/prot_bert_bfd",
                 Protein_padding=200,
                 ligand_filepath="/home/xp/ATPnet/data/epitopes_merge.csv",
                 SMILES_model="DeepChem/ChemBERTa-77M-MLM",
                 SMILES_padding=300,
                 SMILES_argument=True,
                 SMILES_Canonicalization=True,
                 device=torch.device('cuda', 1)
                 )
    next(iter(dataset))
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    a,b,c,d,e=next(iter(loader))

