from typing import List, Dict, Any, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


# Exact canonical feature ordering for ML training and inference
FEATURE_NAMES: List[str] = [
    "donor_mw",
    "donor_logp",
    "donor_tpsa",
    "donor_hbd",
    "donor_hba",
    "donor_rotatable_bonds",
    "donor_aromatic_rings",
    "donor_fraction_csp3",
    "donor_heavy_atoms",
    "acceptor_mw",
    "acceptor_logp",
    "acceptor_tpsa",
    "acceptor_hbd",
    "acceptor_hba",
    "acceptor_rotatable_bonds",
    "acceptor_aromatic_rings",
    "acceptor_fraction_csp3",
    "acceptor_heavy_atoms",
    "mw_diff",
    "logp_diff",
    "tpsa_total",
    "aromatic_rings_total",
]


def extract_features_from_mols(donor_mol: Chem.Mol, acceptor_mol: Chem.Mol) -> Tuple[np.ndarray, Dict[str, float]]:
    """Extract ordered numeric feature vector from RDKit Mol objects.

    Returns:
        Tuple of (1D numpy array, dictionary of feature name -> float).
    """
    donor_mw = float(Descriptors.MolWt(donor_mol))
    donor_logp = float(Descriptors.MolLogP(donor_mol))
    donor_tpsa = float(Descriptors.TPSA(donor_mol))
    donor_hbd = float(Lipinski.NumHDonors(donor_mol))
    donor_hba = float(Lipinski.NumHAcceptors(donor_mol))
    donor_rotatable_bonds = float(Lipinski.NumRotatableBonds(donor_mol))
    donor_aromatic_rings = float(Lipinski.NumAromaticRings(donor_mol))
    donor_fraction_csp3 = float(Lipinski.FractionCSP3(donor_mol))
    donor_heavy_atoms = float(Descriptors.HeavyAtomCount(donor_mol))

    acceptor_mw = float(Descriptors.MolWt(acceptor_mol))
    acceptor_logp = float(Descriptors.MolLogP(acceptor_mol))
    acceptor_tpsa = float(Descriptors.TPSA(acceptor_mol))
    acceptor_hbd = float(Lipinski.NumHDonors(acceptor_mol))
    acceptor_hba = float(Lipinski.NumHAcceptors(acceptor_mol))
    acceptor_rotatable_bonds = float(Lipinski.NumRotatableBonds(acceptor_mol))
    acceptor_aromatic_rings = float(Lipinski.NumAromaticRings(acceptor_mol))
    acceptor_fraction_csp3 = float(Lipinski.FractionCSP3(acceptor_mol))
    acceptor_heavy_atoms = float(Descriptors.HeavyAtomCount(acceptor_mol))

    mw_diff = abs(donor_mw - acceptor_mw)
    logp_diff = abs(donor_logp - acceptor_logp)
    tpsa_total = donor_tpsa + acceptor_tpsa
    aromatic_rings_total = donor_aromatic_rings + acceptor_aromatic_rings

    feature_dict = {
        "donor_mw": donor_mw,
        "donor_logp": donor_logp,
        "donor_tpsa": donor_tpsa,
        "donor_hbd": donor_hbd,
        "donor_hba": donor_hba,
        "donor_rotatable_bonds": donor_rotatable_bonds,
        "donor_aromatic_rings": donor_aromatic_rings,
        "donor_fraction_csp3": donor_fraction_csp3,
        "donor_heavy_atoms": donor_heavy_atoms,
        "acceptor_mw": acceptor_mw,
        "acceptor_logp": acceptor_logp,
        "acceptor_tpsa": acceptor_tpsa,
        "acceptor_hbd": acceptor_hbd,
        "acceptor_hba": acceptor_hba,
        "acceptor_rotatable_bonds": acceptor_rotatable_bonds,
        "acceptor_aromatic_rings": acceptor_aromatic_rings,
        "acceptor_fraction_csp3": acceptor_fraction_csp3,
        "acceptor_heavy_atoms": acceptor_heavy_atoms,
        "mw_diff": mw_diff,
        "logp_diff": logp_diff,
        "tpsa_total": tpsa_total,
        "aromatic_rings_total": aromatic_rings_total,
    }

    feature_vector = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float64)
    return feature_vector, feature_dict
