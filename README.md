# GGAR-Gradient-Guided-Adaptive-Regularization--deep-learning---classification-
# CDS Filtering Pipeline for Brassica Species
This repository contains a Python script (CDS_Data_Validation.py) to filter coding sequences (CDS) from four Brassica species (B. juncea, B. napus, B. oleracea, B. rapa) downloaded from EnsemblPlants (release-61). The script validates FASTA-formatted CDS against strict biological criteria:

Length divisible by 3

Standard DNA bases only (A/T/C/G)

Valid start (ATG) and stop codons (TAA, TAG, TGA)

No internal stop codons

Correct translation to amino acids

Data Sources
Raw CDS files were retrieved from:

B. juncea: ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/fasta/brassica_juncea/cds/

B. napus: ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/fasta/brassica_napus/cds/

B. oleracea: ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/fasta/brassica_oleracea/cds/

B. rapa: ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/fasta/brassica_rapa/cds/
# Run the script:
python CDS_Data_Validation.py
Input: FASTA files in input_dir (configured in script).

Output: Filtered FASTA files + statistics CSV (filtering_stats.csv).

Dependencies
Python 3.x

Biopython (pip install biopython)

Output
Filtered sequences: Valid CDS per species (.fasta).

Statistics: Detailed filtering metrics (pass/fail counts per criterion).
