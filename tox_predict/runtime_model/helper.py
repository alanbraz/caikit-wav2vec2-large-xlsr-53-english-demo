import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def convert_to_mgkg(neglogld50s, smiles):
        df_smiles = pd.DataFrame (smiles, columns = ['smiles'])
        PandasTools.AddMoleculeColumnToFrame(df_smiles, smilesCol='smiles')
        for neglogld50, smile in zip(neglogld50s, smiles):
            molwt = Descriptors.MolWt(Chem.MolFromSmiles(smile))
            yield (10**(-1*neglogld50.item()))*1000*molwt


def convert_to_epa(neglogld50s, smiles):
        mgkg = list(convert_to_mgkg(neglogld50s=neglogld50s, smiles=smiles))

        return pd.cut(mgkg, labels=(0,1,2,3), bins=(-np.inf,50,500,5000, np.inf))