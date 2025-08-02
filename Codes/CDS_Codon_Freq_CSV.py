# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 11:39:27 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Compute codon frequencies for all FASTA files and combine into one CSV
with species ordered by sequence count and labeled numerically.
"""

from Bio import SeqIO
from collections import Counter, defaultdict
import pandas as pd
import os

# Configuration
input_dir = r"E:\Bras\CDS_filtering_sequences\Final_Filtered"
output_csv = os.path.join(input_dir, "combined_codon_frequencies_labeled.csv")

# Codon list (fixed order)
codon_order = [
    "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT", "AGA", "AGC", "AGG", "AGT",
    "ATA", "ATC", "ATG", "ATT", "CAA", "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT",
    "CGA", "CGC", "CGG", "CGT", "CTA", "CTC", "CTG", "CTT", "GAA", "GAC", "GAG", "GAT",
    "GCA", "GCC", "GCG", "GCT", "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT",
    "TAA", "TAC", "TAG", "TAT", "TCA", "TCC", "TCG", "TCT", "TGA", "TGC", "TGG", "TGT",
    "TTA", "TTC", "TTG", "TTT"
]

# First pass: count sequences per species
species_counts = defaultdict(int)
for filename in os.listdir(input_dir):
    if filename.endswith((".fa", ".fasta")):
        input_file = os.path.join(input_dir, filename)
        species_name = os.path.splitext(filename)[0]
        count = sum(1 for _ in SeqIO.parse(input_file, "fasta"))
        species_counts[species_name] = count

# Sort species by count (descending)
sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
species_labels = {name: i+1 for i, (name, _) in enumerate(sorted_species)}

# Second pass: collect data in order
all_data = []
for species, _ in sorted_species:
    filename = f"{species}.fasta"  # Adjust extension if needed
    input_file = os.path.join(input_dir, filename)
    
    if not os.path.exists(input_file):
        continue
        
    for record in SeqIO.parse(input_file, "fasta"):
        seq = str(record.seq).upper()
        codon_counts = Counter(seq[i:i+3] for i in range(0, len(seq), 3) if i+3 <= len(seq))
        row = {
            "Species": species,
            "Label": species_labels[species],
            "Sequence_ID": record.id
        }
        row.update({codon: codon_counts.get(codon, 0) for codon in codon_order})
        all_data.append(row)

# Create and save DataFrame
df = pd.DataFrame(all_data)
df.to_csv(output_csv, index=False)

print(f"Saved combined codon frequencies to: {output_csv}")
print("\nSpecies labeling:")
for species, label in species_labels.items():
    print(f"{species}: Label {label} ({species_counts[species]} sequences)")