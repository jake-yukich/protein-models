from Bio.PDB import PDBList
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import contextlib

def load_pdb_ids(file_path):
    """
    Load PDB IDs from a text file.

    Args:
        file_path (str): Path to the text file containing PDB IDs.

    Returns:
        list: A list of PDB IDs.
    """
    with open(file_path, 'r') as file:
        pdb_ids = file.read().splitlines()
    return pdb_ids

@contextlib.contextmanager
def filter_stdout(filter_words):
    """
    Context manager to filter specific stdout messages.

    Args:
        filter_words (list): List of words to filter out from stdout.

    Yields:
        None
    """
    class FilteredStream:
        def __init__(self, stream):
            self.stream = stream

        def write(self, message):
            if not any(word in message for word in filter_words):
                self.stream.write(message)

        def flush(self):
            self.stream.flush()

    old_stdout = sys.stdout
    sys.stdout = FilteredStream(sys.stdout)
    try:
        yield
    finally:
        sys.stdout = old_stdout

def download_pdb_file(pdb_id, save_dir):
    """
    Download a single PDB file.

    Args:
        pdb_id (str): The PDB ID of the file to download.
        save_dir (str): The directory to save the downloaded PDB file.

    Returns:
        str: A message indicating the result of the download attempt.

    Raises:
        Exception: If there is an error during the download.
    """
    pdbl = PDBList()  # using Biopython's PDBList class
    file_path = os.path.join(save_dir, f"pdb{pdb_id}.ent")

    if not os.path.exists(file_path):
        try:
            with filter_stdout(["Downloading PDB structure", "Desired structure doesn't exist"]):
                pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format='pdb', overwrite=False)
            return f"Downloaded {pdb_id}"
        except Exception as e:
            return f"Error downloading {pdb_id}: {str(e)}"
    else:
        return f"Skipped {pdb_id}, already exists"

def download_pdb_files(pdb_ids, save_dir='pdb_files', num_threads=16):
    """
    Download PDB files using multiple threads and a progress bar.

    Args:
        pdb_ids (list): List of PDB IDs to download.
        save_dir (str): The directory to save the downloaded PDB files.
        num_threads (int): The number of threads to use for downloading.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_pdb_file, pdb_id, save_dir): pdb_id for pdb_id in pdb_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDB files", unit="file", leave=True):
            results.append(future.result())
    
    for result in results:
        print(result)

# Load PDB IDs from text files
train_pdb_ids = load_pdb_ids('train_ids.txt')
test_pdb_ids = load_pdb_ids('test_ids.txt')

# Download PDB files
download_pdb_files(train_pdb_ids, save_dir='pdb_files/train')
download_pdb_files(test_pdb_ids, save_dir='pdb_files/test')