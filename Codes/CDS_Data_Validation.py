# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 11:13:04 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Comprehensive CDS Filtering Pipeline (Simplified)
Validates coding sequences according to strict biological criteria:
1. Length divisible by 3
2. Contains only standard DNA bases (A, T, C, G)
3. Starts with ATG codon
4. Ends with a valid stop codon (TAA, TAG, TGA)
5. Contains no internal stop codons
6. Valid translation to amino acids
7. Consistent DNA/amino acid length relationship
"""

import os
import csv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
from collections import defaultdict

# Configuration
input_dir = r"E:\Bras\CDS_filtering_sequences"
output_dir = r"E:\Bras\CDS_filtering_sequences\Final_Filtered"
stats_file = os.path.join(output_dir, "filtering_stats.csv")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize statistics tracking
def init_stats():
    return {
        'initial': 0,
        'length_divisible_by_3': 0,
        'valid_bases': 0,
        'starts_with_ATG': 0,
        'ends_with_stop': 0,
        'no_internal_stops': 0,
        'valid_translation': 0,
        'consistent_length': 0,
        'final': 0
    }

# Define biological constants
valid_bases = set("ATCGatcg")
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
stop_codons = set(standard_table.stop_codons)
start_codon = "ATG"

# Process each FASTA file
all_stats = []
for filename in os.listdir(input_dir):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_filtered.fasta")
        
        stats = defaultdict(int)
        stats['filename'] = filename
        kept_records = []
        
        # Initial count
        for record in SeqIO.parse(input_file, "fasta"):
            stats['initial'] += 1
        
        # Filtering pipeline
        for record in SeqIO.parse(input_file, "fasta"):
            seq = str(record.seq).upper()
            passed_all_filters = True
            
            # Filter 1: Length divisible by 3
            if len(seq) % 3 != 0:
                stats['length_divisible_by_3_fail'] = stats.get('length_divisible_by_3_fail', 0) + 1
                passed_all_filters = False
                continue
            stats['length_divisible_by_3'] += 1
            
            # Filter 2: Valid bases only
            if not set(seq).issubset(valid_bases):
                stats['valid_bases_fail'] = stats.get('valid_bases_fail', 0) + 1
                passed_all_filters = False
                continue
            stats['valid_bases'] += 1
            
            # Filter 3: Starts with ATG
            if not seq.startswith(start_codon):
                stats['starts_with_ATG_fail'] = stats.get('starts_with_ATG_fail', 0) + 1
                passed_all_filters = False
                continue
            stats['starts_with_ATG'] += 1
            
            # Filter 4: Ends with stop codon
            if seq[-3:] not in stop_codons:
                stats['ends_with_stop_fail'] = stats.get('ends_with_stop_fail', 0) + 1
                passed_all_filters = False
                continue
            stats['ends_with_stop'] += 1
            
            # Filter 5: No internal stop codons
            codons = [seq[i:i+3] for i in range(0, len(seq)-3, 3)]  # Exclude final stop codon
            internal_stops = sum(1 for codon in codons if codon in stop_codons)
            if internal_stops > 0:
                stats['no_internal_stops_fail'] = stats.get('no_internal_stops_fail', 0) + 1
                passed_all_filters = False
                continue
            stats['no_internal_stops'] += 1
            
            # Filter 6: Valid translation
            try:
                protein = Seq(seq).translate(to_stop=False, table="Standard")
                if "*" in protein[:-1] or len(protein) == 0:
                    stats['valid_translation_fail'] = stats.get('valid_translation_fail', 0) + 1
                    passed_all_filters = False
                    continue
            except Exception as e:
                stats['valid_translation_fail'] = stats.get('valid_translation_fail', 0) + 1
                passed_all_filters = False
                continue
            stats['valid_translation'] += 1
            
            # Filter 7: Consistent length relationship
            expected_aa_length = len(seq) // 3
            if len(protein) != expected_aa_length:
                stats['consistent_length_fail'] = stats.get('consistent_length_fail', 0) + 1
                passed_all_filters = False
                continue
            stats['consistent_length'] += 1
            
            # Passed all filters
            if passed_all_filters:
                kept_records.append(record)
                stats['final'] += 1
        
        # Save filtered sequences for this file
        with open(output_file, "w") as out_fasta:
            SeqIO.write(kept_records, out_fasta, "fasta")
        
        # Add to overall statistics
        all_stats.append(stats)
        
        print(f"\nProcessed: {filename}")
        print(f"Initial sequences: {stats['initial']}")
        print(f"Removed - length not divisible by 3: {stats.get('length_divisible_by_3_fail', 0)}")
        print(f"Removed - non-standard bases: {stats.get('valid_bases_fail', 0)}")
        print(f"Removed - doesn't start with ATG: {stats.get('starts_with_ATG_fail', 0)}")
        print(f"Removed - doesn't end with stop codon: {stats.get('ends_with_stop_fail', 0)}")
        print(f"Removed - has internal stop codons: {stats.get('no_internal_stops_fail', 0)}")
        print(f"Removed - invalid translation: {stats.get('valid_translation_fail', 0)}")
        print(f"Removed - inconsistent lengths: {stats.get('consistent_length_fail', 0)}")
        print(f"Final kept sequences: {stats['final']}")

# Save statistics to CSV
if all_stats:
    with open(stats_file, 'w', newline='') as csvfile:
        fieldnames = [
            'filename', 'initial', 
            'length_divisible_by_3', 'length_divisible_by_3_fail',
            'valid_bases', 'valid_bases_fail',
            'starts_with_ATG', 'starts_with_ATG_fail',
            'ends_with_stop', 'ends_with_stop_fail',
            'no_internal_stops', 'no_internal_stops_fail',
            'valid_translation', 'valid_translation_fail',
            'consistent_length', 'consistent_length_fail',
            'final'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\nStatistics saved to: {stats_file}")

print("\nFiltering complete. Individual filtered files created in:", output_dir)