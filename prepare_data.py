import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import sys
from gtfparse import read_gtf

print("--- Starting Data Preprocessing (Quantitative, Strongest Signal) ---")

# --- Define Paths and Constants ---

links_df_path = 'feature_linkage/feature_linkage.bedpe'
genome_path = 'hg38.pkl' 
output_dir = 'output_samples_regression' # Folder to create for .npy files

# Preprocessing Constants
SIG_THRESH = 1.3  # Corresponds to p-value < 0.05
SEQUENCE_LENGTH = 196_608
OUTPUT_BINS = 896
BIN_WIDTH = 128
OUTPUT_COVERAGE = OUTPUT_BINS * BIN_WIDTH
HALF_COVERAGE = OUTPUT_COVERAGE // 2 # 57,344

# Define the column names from the .bedpe documentation
col_names = [
    'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2',
    'name', 'score', 'strand1', 'strand2', 'significance',
    'distance', 'linkage_type'
]

# Define the chromosome split
split_map = {
    'chr1': 'test',
    'chr8': 'validation',
    'chr9': 'validation',
    'chr10': 'validation'
}

# --- Helper Functions ---

def one_hot_encode(sequence):
    """Converts a DNA sequence string into a (N, 4) numpy array."""
    mapping = {
        'A': np.array([1, 0, 0, 0], dtype=np.float32),
        'C': np.array([0, 1, 0, 0], dtype=np.float32),
        'G': np.array([0, 0, 1, 0], dtype=np.float32),
        'T': np.array([0, 0, 0, 1], dtype=np.float32),
        'N': np.array([0, 0, 0, 0], dtype=np.float32) # 'N's are encoded as all zeros
    }
    encoded_list = [mapping.get(base, mapping['N']) for base in sequence.upper()]
    return np.array(encoded_list)

def calculate_enformer_bin(relative_position):
    """Finds the correct Enformer output bin for a peak."""
    # Check if peak is outside the 114kb central predictable region
    if relative_position < -HALF_COVERAGE or relative_position >= HALF_COVERAGE:
        return -1 # Not in any bin
        
    # Shift position from [-57344, +57344) to [0, 114688)
    shifted_position = relative_position + HALF_COVERAGE
    
    # Calculate bin index
    bin_index = int(np.floor(shifted_position / BIN_WIDTH))
    return bin_index

def load_tss_from_gtf(gtf_path):
    """
    Uses gtfparse to robustly load TSS coordinates for all genes.
    Returns: { 'GENE_NAME': (chrom, tss_coord) }
    """
    print(f"--- Loading GTF using gtfparse: {gtf_path} ---")
    
    try:
        # read_gtf returns a Polars DataFrame in newer versions
        df = read_gtf(gtf_path)
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()
    except Exception as e:
        print(f"Error reading GTF: {e}")
        sys.exit(1)

    # Filter for 'gene' features
    genes_df = df[df["feature"] == "gene"].copy()
    
    if len(genes_df) == 0:
        print("[!] Warning: No 'gene' features found. Trying 'transcript'...")
        genes_df = df[df["feature"] == "transcript"].copy()

    tss_map = {}
    
    # Iterate and build the map
    # We rely on the clean columns your checker script confirmed exist
    for _, row in tqdm(genes_df.iterrows(), total=len(genes_df), desc="Mapping TSS"):
        gene_name = row["gene_name"]
        
        # Skip invalid names
        if pd.isna(gene_name) or gene_name == "":
            continue

        # --- STRAND AWARENESS LOGIC ---
        # If strand is '+', TSS is Start.
        # If strand is '-', TSS is End.
        if row["strand"] == "+":
            tss = row["start"]
        elif row["strand"] == "-":
            tss = row["end"]
        else:
            continue 
            
        tss_map[gene_name] = (row["seqname"], tss)

    print(f"Loaded valid TSS for {len(tss_map)} genes.")
    return tss_map

# --- Main Execution ---

def main():
    # --- Load and Filter Links ---
    print("--- Loading and Filtering Links ---")
    if not os.path.exists(links_df_path):
        print(f"Error: Cannot find linkage file at {links_df_path}", file=sys.stderr)
        print("Please update the 'links_df_path' variable in this script.", file=sys.stderr)
        sys.exit(1)

    links_df = pd.read_csv(links_df_path,
                             sep='\t',
                             header=None,
                             names=col_names)

    valid_types = ['peak-gene', 'gene-peak']
    filtered_links_df = links_df[
        (links_df['linkage_type'].isin(valid_types)) &
        (links_df['significance'] > SIG_THRESH)
    ].copy()

    print(f"Loaded {len(links_df)} links, kept {len(filtered_links_df)} high-confidence (significant) links.")

    # --- Get TSS Info ---
    GTF_PATH = 'gencode.v32.annotation.gtf' 
    tss_map = load_tss_from_gtf(GTF_PATH)

    # --- Normalize Links ---
    print("--- Normalizing Links ---")
    processed_links = []
    missing_genes_count = 0
    
    for _, row in tqdm(filtered_links_df.iterrows(), total=len(filtered_links_df), desc="Normalizing links"):
        link_data = {}

        trimmed_name = row['name'][1:-1]
        name_parts = trimmed_name.split('><')
        if len(name_parts) != 2: continue
        name1, name2 = name_parts[0], name_parts[1]

        link_data['score'] = row['score']

        # 1. Identify which name is the gene vs the peak
        if row['linkage_type'] == 'peak-gene':
            target_gene_name = name2
            link_data['peak_chr'] = row['chrom1']
            link_data['peak_start'] = row['start1']
            link_data['peak_end'] = row['end1']
        else: # gene-peak
            target_gene_name = name1
            link_data['peak_chr'] = row['chrom2']
            link_data['peak_start'] = row['start2']
            link_data['peak_end'] = row['end2']

        # 2. LOOKUP TSS IN GTF
        if target_gene_name in tss_map:
            chrom, tss = tss_map[target_gene_name]
            
            link_data['gene_chr'] = chrom
            link_data['gene_center'] = tss
            link_data['gene_name'] = target_gene_name
            
            processed_links.append(link_data)
        else:
            missing_genes_count += 1
            continue 

    normalized_links_df = pd.DataFrame(processed_links)
    print(f"Normalized {len(normalized_links_df)} links using GTF TSS.")
    print(f"Skipped {missing_genes_count} links where gene was not found in GTF.")

    # --- Load Genome ---
    print("--- Loading Genome hg38.pkl... ---")
    if not os.path.exists(genome_path):
        print(f"Error: Cannot find genome file at {genome_path}", file=sys.stderr)
        print("Please update the 'genome_path' variable in this script.", file=sys.stderr)
        sys.exit(1)

    with open(genome_path, 'rb') as f:
        genome_dict = pickle.load(f)
    print("Genome loaded.")

    # --- Create Output Directories ---
    set_names = ['train', 'validation', 'test']
    for set_name in set_names:
        set_path = os.path.join(output_dir, set_name)
        if not os.path.exists(set_path):
            os.makedirs(set_path)
            
    print(f"--- Saving samples to ./{output_dir}/[train, validation, test] ---")

    # --- Main Processing Loop ---
    links_by_gene = normalized_links_df.groupby('gene_name')
    
    # Initialize counters
    set_counts = {'train': 0, 'validation': 0, 'test': 0}

    for gene_name, group in tqdm(links_by_gene, desc="Processing Genes"):
        gene_row = group.iloc[0]
        chromosome = gene_row['gene_chr']
        gene_center = gene_row['gene_center']
        
        set_name = split_map.get(chromosome, 'train')
        set_path = os.path.join(output_dir, set_name)
        
        seq_start = gene_center - (SEQUENCE_LENGTH // 2)
        seq_end = gene_center + (SEQUENCE_LENGTH // 2)
        
        try:
            sequence_str = genome_dict[chromosome][seq_start:seq_end]
        except (KeyError, IndexError):
            continue
            
        if len(sequence_str) != SEQUENCE_LENGTH:
            continue
            
        X_input = one_hot_encode(sequence_str)
        
        if np.sum(X_input) < (SEQUENCE_LENGTH * 0.9): # Skip if >10% 'N's
            continue
        
        Y_output = np.zeros(OUTPUT_BINS, dtype=np.float32)
        
        for link in group.itertuples():
            peak_center = (link.peak_start + link.peak_end) / 2
            relative_position = peak_center - gene_center
            bin_index = calculate_enformer_bin(relative_position)
            
            if bin_index != -1:
                if abs(link.score) > abs(Y_output[bin_index]):
                    Y_output[bin_index] = link.score
                
        if np.sum(np.abs(Y_output)) > 0:
            safe_gene_name = gene_name.replace('/', '_').replace('.', '_')
            
            np.save(f'{set_path}/X_{safe_gene_name}.npy', X_input)
            np.save(f'{set_path}/Y_{safe_gene_name}.npy', Y_output)
            
            # Increment the correct counter
            set_counts[set_name] += 1

    # Print the final counts
    print("\n--- Data Preprocessing Complete! ---")
    total_genes = sum(set_counts.values())
    print(f"  Total genes processed: {total_genes}")
    print(f"  Training samples:   {set_counts['train']} ({set_counts['train']/total_genes:.1%})")
    print(f"  Validation samples: {set_counts['validation']} ({set_counts['validation']/total_genes:.1%})")
    print(f"  Test samples:       {set_counts['test']} ({set_counts['test']/total_genes:.1%})")

if __name__ == "__main__":
    main()