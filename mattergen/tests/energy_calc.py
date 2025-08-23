import os
import glob
import pandas as pd
from ase.io import read
from mattersim.forcefield import MatterSimCalculator
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import torch
from loguru import logger
import sys

# Check if the output is being redirected to a file
is_redirected = not sys.stdout.isatty()

# Configure loguru
logger.remove()  # Remove the default logger
logger.add(sys.stdout, colorize=not is_redirected, format="<green>{time}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


# Set device for MatterSim
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running MatterSim on {device}")

# Directory containing the extxyz files
base_dir = "/users/eleves-b/2021/auguste.de-lambilly/results/Li-Co-O_f"
extxyz_files = glob.glob(os.path.join(base_dir, "generated_crystals*.extxyz"), recursive=True)

# Initialize MatterSimCalculator
calculator = MatterSimCalculator(device=device, load_path="/Data/auguste.de-lambilly/mattersim_torch/pretrained_models/mattersim-v1.0.0-5M.pth")

# Global variable for the phase diagram
PDIAG = None

def _energy_hull(composition, energy):
    """
    Computes the energy above the hull for a given composition and energy.
    """
    global PDIAG
    dir = "/Data/auguste.de-lambilly/mattergenbis/phase_diagram/"  # Directory for the phase diagram CSV
    if PDIAG is None:
        # Load the CSV file only once
        csv = pd.read_csv(os.path.join(dir, "LiCoO.csv"))
        entries = [
            PDEntry(composition=Composition(csv["Formula"][i]), energy=csv["Energy"][i])
            for i in range(len(csv))
        ]
        PDIAG = PhaseDiagram(entries)

    # Compute the energy on the hull for the given composition
    try:
        hull_energy = PDIAG.get_hull_energy(Composition(composition))
    except Exception as e:
        logger.error(f"Failed to compute hull energy for composition {composition}: {e}")
        return None

    # Compute energy above the hull
    energy_above_hull = energy - hull_energy
    return energy_above_hull

for extxyz_file in extxyz_files:
    logger.info(f"Processing file: {extxyz_file}")
    ext_dir = os.path.dirname(extxyz_file)
    ext_base = os.path.basename(extxyz_file)
    
    # Replace 'generated_crystals' with 'energy' and change extension to .csv
    csv_name = ext_base.replace("generated_crystals", "energy").replace(".extxyz", ".csv")
    output_csv = os.path.join(ext_dir, csv_name)
    
    # Skip if the CSV already exists
    if os.path.exists(output_csv):
        logger.info(f"Skipping {output_csv} (already exists)")
        continue

    # Read all structures from the extxyz file
    atoms_list = read(extxyz_file, index=":")
    results = []

    for idx, atoms in enumerate(atoms_list):
        try:
            # Assign the calculator to the atoms object
            atoms.calc = calculator
            
            # Compute energy and energy per atom
            energy = atoms.get_potential_energy()
            energy_per_atom = energy / len(atoms)
            
            # Compute energy above the hull
            composition = atoms.get_chemical_formula(empirical=True)
            energy_above_hull = _energy_hull(composition, energy)
            
            # Append results
            results.append({
                "structure_idx": idx,
                "energy (eV)": energy,
                "energy_per_atom (eV/atom)": energy_per_atom,
                "energy_above_hull (eV)": energy_above_hull
            })
        except Exception as e:
            logger.error(f"Error processing structure {idx} in file {extxyz_file}: {e}")

    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved energies to {output_csv}")