# "To create our datasets,
#  we extract non-overlapping fragments of lengths 16, 64, and 128 from chain ‘A’ for each protein
#  structure starting at the first residue and calculate the pairwise distance matrices from the alpha-carbon
#  coordinate positions"

import os
import numpy as np
from Bio import PDB
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_structure(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    return structure[0]["A"]  # Return the first model and chain A

def extract_fragments(chain, fragment_length):
    fragments = []
    residues = list(chain.get_residues())
    for i in range(0, len(residues) - fragment_length + 1, fragment_length):
        fragment = residues[i:i+fragment_length]
        if len(fragment) == fragment_length:
            fragments.append(fragment)
    return fragments

def calculate_distance_matrix(fragment):
    coords = []
    for residue in fragment:
        if "CA" in residue:
            coords.append(residue["CA"].coord)
    coords = np.array(coords)
    if len(coords) < 2:
        return np.array([])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix

def process_pdb_file(pdb_file, fragment_length):
    try:
        chain = load_structure(pdb_file)
        fragments = extract_fragments(chain, fragment_length)
        matrices = []
        for fragment in fragments:
            dist_matrix = calculate_distance_matrix(fragment)
            if dist_matrix.shape == (fragment_length, fragment_length):
                matrices.append(dist_matrix)
        return matrices
    except Exception as e:
        return []

def create_datasets(pdb_dir, output_dir, fragment_lengths=[16, 64, 128], num_threads=16, batch_size=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.ent') or f.endswith('.pdb')]
    
    for length in fragment_lengths:
        output_file = os.path.join(output_dir, f'distance_matrices_{length}.npy')
        
        if os.path.exists(output_file):
            print(f"Skipping {length}-residue fragments, output file already exists")
            continue
        
        batch_matrices = []
        total_matrices = 0

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_pdb_file, os.path.join(pdb_dir, f), length): f for f in pdb_files}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {length}-residue fragments"):
                matrices = future.result()
                batch_matrices.extend(matrices)
                
                if len(batch_matrices) >= batch_size:
                    # Save batch to disk
                    batch_array = np.array(batch_matrices)
                    if total_matrices == 0:
                        np.save(output_file, batch_array)
                    else:
                        with open(output_file, 'ab') as f:
                            np.save(f, batch_array)
                    total_matrices += len(batch_matrices)
                    batch_matrices = []
        
        if batch_matrices:
            batch_array = np.array(batch_matrices)
            if total_matrices == 0:
                np.save(output_file, batch_array)
            else:
                with open(output_file, 'ab') as f:
                    np.save(f, batch_array)
            total_matrices += len(batch_matrices)
        
        print(f"Saved {total_matrices} matrices of size {length}x{length}")

create_datasets('pdb_files/train', 'datasets/train', [16])
create_datasets('pdb_files/test', 'datasets/test', [16])