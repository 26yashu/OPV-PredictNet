# Processed Datasets

This directory stores tabular datasets generated after:
1. Canonicalizing donor and acceptor SMILES via RDKit.
2. Filtering out unparseable or non-organic entries.
3. Generating 2D physicochemical descriptors and fingerprint vectors.
4. Handling missing values and removing redundant duplicates.
5. Partitioning into train/test sets using scaffold or group splits.

All files in this directory are reproducible using the data preparation scripts in `ml/src/`.
