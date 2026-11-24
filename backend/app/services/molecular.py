from typing import Any, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def validate_and_canonicalize_smiles(smiles: str) -> Tuple[bool, str, Optional[str]]:
    """Validate a chemical SMILES string and return its canonical representation.

    Args:
        smiles: Raw SMILES string.

    Returns:
        Tuple of (is_valid, canonical_smiles, error_message).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "", "SMILES string cannot be empty or whitespace only"

    clean_smiles = smiles.strip()
    try:
        # MolFromSmiles performs valence checks and sanitization by default
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return False, "", "Invalid chemical structure or unparseable SMILES syntax (check atom symbols, ring closures, and bond valences)"

        canonical = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical, None
    except Exception as e:
        return False, "", f"RDKit parse failure: {str(e)}"


def compute_molecular_descriptors(smiles: str, role: str = "molecule") -> Dict[str, Any]:
    """Calculate scientifically meaningful 2D physicochemical descriptors using RDKit.

    Args:
        smiles: Valid SMILES string.
        role: Functional role description ('donor', 'acceptor', 'molecule').

    Returns:
        Dictionary of computed physicochemical descriptors.

    Raises:
        ValueError: If SMILES is invalid or cannot be parsed.
    """
    is_valid, canonical_smiles, error_msg = validate_and_canonicalize_smiles(smiles)
    if not is_valid:
        raise ValueError(f"Invalid {role} SMILES: {error_msg}")

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"Could not construct RDKit molecule for {role}")

    return {
        "role": role,
        "canonical_smiles": canonical_smiles,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 2),
        "exact_mass": round(float(Descriptors.ExactMolWt(mol)), 4),
        "logp": round(float(Descriptors.MolLogP(mol)), 2),
        "tpsa": round(float(Descriptors.TPSA(mol)), 2),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "aromatic_rings": int(Lipinski.NumAromaticRings(mol)),
        "fraction_csp3": round(float(Lipinski.FractionCSP3(mol)), 3),
        "heavy_atom_count": int(Descriptors.HeavyAtomCount(mol)),
        "ring_count": int(Lipinski.RingCount(mol)),
    }


def analyze_blend(
    donor_smiles: str,
    acceptor_smiles: str,
    donor_name: str = "Donor",
    acceptor_name: str = "Acceptor"
) -> Dict[str, Any]:
    """Calculate descriptors for donor and acceptor separately, preserving their distinct roles,
    and compute blend interaction descriptors.

    Args:
        donor_smiles: SMILES string of the donor material.
        acceptor_smiles: SMILES string of the acceptor material.
        donor_name: Identifier name for donor.
        acceptor_name: Identifier name for acceptor.

    Returns:
        Structured dictionary with distinct donor and acceptor descriptors and composite blend features.
    """
    donor_desc = compute_molecular_descriptors(donor_smiles, role="donor")
    donor_desc["name"] = donor_name

    acceptor_desc = compute_molecular_descriptors(acceptor_smiles, role="acceptor")
    acceptor_desc["name"] = acceptor_name

    # Compute blend-level composite features
    blend_features = {
        "molecular_weight_diff": round(abs(donor_desc["molecular_weight"] - acceptor_desc["molecular_weight"]), 2),
        "logp_diff": round(abs(donor_desc["logp"] - acceptor_desc["logp"]), 2),
        "tpsa_total": round(donor_desc["tpsa"] + acceptor_desc["tpsa"], 2),
        "aromatic_rings_total": donor_desc["aromatic_rings"] + acceptor_desc["aromatic_rings"],
        "heavy_atom_count_total": donor_desc["heavy_atom_count"] + acceptor_desc["heavy_atom_count"],
    }

    return {
        "donor": donor_desc,
        "acceptor": acceptor_desc,
        "blend_features": blend_features,
    }
