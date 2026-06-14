X-PIE (Cross-linking-guided Protein Interaction Evaluation, Ensemble modeling, and Elucidation), a fully automated computational framework for the analysis and structural interpretation of intracellular cross-linking mass spectrometry (XL-MS) data.
X-PIE consists of two integrated modules:
1. X-PIE Curation — A computational pipeline for filtering, validating, and annotating pLink XL-MS search results to generate high-confidence protein-protein interaction (PPI) datasets. The workflow filters inter-protein cross-link peptide-spectrum matches (PSMs), resolves ambiguous protein-pair assignments via dataset-level frequency voting, summarizes unique cross-linked residue pairs, and separates STRING-reported PPIs from candidate novel interactions. For previously unreported PPIs, the module queries the RCSB Protein Data Bank (PDB) for homologous complex structures, retaining strict hits only when both interacting proteins exceed a user-defined local sequence identity threshold.
2. X-PIE Modeling — A computational pipeline for building three-dimensional structural models of protein-protein complexes from XL-MS distance restraints. The method automatically clusters cross-linked sites into spatially distinct interaction interfaces and identifies intramolecular flexible regions prior to conformational sampling, enabling automated modeling of multi-interface assemblies and transient encounter complexes. The pipeline performs distance-restraint-driven ensemble sampling and outputs evaluated structural models.

<img width="4960" height="5670" alt="X-PIE2" src="https://github.com/user-attachments/assets/742a3d4a-4837-4beb-ac6e-a7c426c1bd7e" />

<img width="5197" height="5670" alt="X-PIE1" src="https://github.com/user-attachments/assets/1685f7ad-0945-4b7d-bf64-bf63b5ecd18d" />




