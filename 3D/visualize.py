import nglview as nv
import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom

def visualize_structure(coordinates: np.ndarray):
    """
    Visualize a 3D structure from coordinates using nglview.

    Args:
        coordinates: (n, 3) array of 3D coordinates
    """
    structure = Structure.Structure("structure")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    structure.add(model)
    model.add(chain)

    for i, (x, y, z) in enumerate(coordinates, start=1):
        # Create a dummy residue and atom
        res_id = (' ', i, ' ')
        residue = Residue.Residue(res_id, "ALA", i)
        atom = Atom.Atom("CA", np.array([x, y, z]), 1.0, 0.0, " ", "CA", i, "C")
        residue.add(atom)
        chain.add(residue)

    view = nv.show_biopython(structure)
    # view.add_representation('cartoon', selection='protein')
    # view.add_representation('ball+stick', selection='all')
    return view 