from pymatgen.core import Structure
import numpy as np
import numpy.linalg as linalg
from scipy.special import sph_harm
import os
import sys
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Element
from pymatgen.core.periodic_table import Element
import pyscal.core as pc
import json
from pymatgen.core import Composition
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


parity = True  # Include parity or not for the plot
degree_l = 40

api_key = "M244rOwcXhVorQLQwwH6s2GXVO88BCIJ"

# List of 3d transition metals based on their atomic numbers
transition_metals_3d = [Element(sym).symbol for sym in [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"
]]

def read_xyz_file(file_path):
    """
    Read atomic coordinates from an XYZ file.

    Parameters:
    file_path (str): The path to the XYZ file to be read.

    Returns:
    np.ndarray: A NumPy array of atomic coordinates with shape (n, 3), where n is the number of atoms in the molecule. Each row corresponds to the x, y, and z coordinates of an atom.
    """

    with open(file_path, 'r') as xyz_file:
        # Skip the first two lines (metadata)
        lines = xyz_file.readlines()[2:]

    # Extract atomic symbols (not used here but might be useful for other purposes)
    atomic_symbols = np.array([line.split()[0] for line in lines])

    # Extract and store atomic coordinates
    atomic_coordinates = np.array([line.split()[1:4]
                                  for line in lines], dtype=float)

    return atomic_coordinates, atomic_symbols


from pymatgen.core import Structure
import numpy as np

def extract_cluster(
    source,
    mode='symbol',          # 'symbol' or 'index'
    atom=None,              # atomic symbol or index depending on mode
    cluster_radius=3
):
    """
    Extracts a cluster of atoms from a structure or CIF file based on atomic symbol or index.
    The central atom will be first in the returned arrays.

    Parameters:
    - source (str or Structure): Path to CIF file or a pymatgen Structure object.
    - mode (str): 'symbol' or 'index'. Determines how to identify the central atom.
    - atom (str or int): Atomic symbol (e.g. 'Fe') or index, depending on mode.
    - cluster_radius (float): Radius in angstroms to include neighboring atoms.

    Returns:
    - atomic_coords (np.ndarray): Centered coordinates of atoms (n, 3).
    - atomic_symbols (np.ndarray): Atomic symbols of the atoms, central atom first.
    - atomic_numbers (np.ndarray): Atomic numbers (Z), central atom first.
    """
    # Load structure from file or take directly
    if isinstance(source, str):
        structure = Structure.from_file(source)
    elif isinstance(source, Structure):
        structure = source
    else:
        raise TypeError("source must be a file path or pymatgen Structure object")

    # Identify central atom index
    if mode == 'symbol':
        if not isinstance(atom, str):
            raise ValueError("For mode='symbol', 'atom' must be a string.")
        for i, site in enumerate(structure):
            if site.specie.symbol == atom:
                central_index = i
                break
        else:
            raise ValueError(f"No atom with symbol '{atom}' found.")
    elif mode == 'index':
        if not isinstance(atom, int):
            raise ValueError("For mode='index', 'atom' must be an integer.")
        if not (0 <= atom < len(structure)):
            raise IndexError(f"Index {atom} is out of bounds.")
        central_index = atom
    else:
        raise ValueError("mode must be either 'symbol' or 'index'")

    # Get central atom and neighbors
    central_coords = structure[central_index].coords
    sites = structure.get_sites_in_sphere(central_coords, cluster_radius)

    cluster_sites = [structure[central_index]] + [
        site[0] for site in sites if not np.allclose(site[0].coords, central_coords)
    ]

    atomic_coords = np.array([site.coords for site in cluster_sites])
    atomic_symbols = np.array([site.specie.symbol for site in cluster_sites])
    atomic_numbers = np.array([site.specie.Z for site in cluster_sites])

    # Center the cluster
    atomic_coords = atomic_coords - atomic_coords[0]

    return atomic_coords, atomic_symbols, atomic_numbers


def translate_coords(coords):
    """
    Translates a list of 3D coordinates so that the first entry is at (0, 0, 0),
    and applies the same translation to all other coordinates.

    Args:
        coords (list of lists or numpy array): A list of coordinates, where each coordinate is a list or array of [x, y, z].

    Returns:
        translated_coords (numpy array): The translated coordinates with the first entry centered at (0, 0, 0).
    """
    # Convert coords to numpy array if not already
    coords = np.array(coords)

    # Take the first entry as the translation vector
    translation_vector = coords[0]

    # Print message for checking cluster
    print("Checking cluster...")

    # Check if the first entry is already at (0, 0, 0)
    if np.allclose(translation_vector, [0, 0, 0]):
        print("Cluster already centered.")
        return coords
    else:
        print("Centering cluster...")

    # Apply translation: subtract the translation vector from all coords
    translated_coords = coords - translation_vector

    return translated_coords


def detect_3d_transition_metal(structure):
    """
    Detect the first occurrence of a transition metal in the given structure.
    
    Args:
        structure (Structure): A pymatgen Structure object representing the crystal.
    
    Returns:
        str: The symbol of the transition metal, or None if no transition metal is found.
    """
    for site in structure:
        if site.specie.symbol in transition_metals_3d:
            return site.specie.symbol
    return None

def cartesian_to_spherical(coords):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    coords (np.ndarray): A NumPy array of shape (n, 3) containing n points in
                         Cartesian coordinates, where each row represents [x, y, z].

    Returns:
    np.ndarray: A NumPy array of shape (n, 3) containing n points in spherical
                coordinates, where each row represents [r, theta, phi].
                Theta and phi are in radians. Theta is polar angle phi is the azimuthal angle.
    """

    # Extract x, y, and z coordinates from the input array
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Compute the radial distance for each point
    r = np.sqrt(x**2 + y**2 + z**2)

    # Calculate theta(polar anlge), defaulting to 0 where r is 0 (which also covers x=y=0)
    theta = np.where(r > 0.0, np.arctan2(y, x), 0)

    # Calculate phi(azimuthal angle), handling division by zero by setting theta to 0 where z is 0
    phi = np.where(r > 0.0, np.arccos(z/r), 0)

    # Stack the computed spherical coordinates into a single array
    spherical_coords = np.vstack((r, theta, phi)).T

    return spherical_coords

def calculate_real_sph_harm(order_m, degree_l, theta, phi):
    """
    Computes the real-valued spherical harmonic function Y_l^m based on its complex form.

    This transformation is used to convert complex spherical harmonics into real-valued
    versions, which are common in physical applications like atomic orbitals and bond-order analysis.

    Parameters:
    - order_m (int): Order of the spherical harmonic (can be negative, zero, or positive).
    - degree_l (int): Degree of the spherical harmonic (degree_l >= 0, and |order_m| <= degree_l).
    - theta (float or np.ndarray): Azimuthal angle (in radians), typically in [0, 2π].
    - phi (float or np.ndarray): Polar angle (in radians), typically in [0, π].

    Returns:
    - Ylm_real (float or np.ndarray): The real-valued spherical harmonic evaluated at the given angles.
    """

    # Compute the complex spherical harmonic Y_l^m(θ, φ)
    Ylm = sph_harm(order_m, degree_l, theta, phi)

    # Compute its complex conjugate
    Ylm_conjugate = np.conj(Ylm)

    # Apply real-valued transformation rules
    if order_m > 0:
        # Cosine component for positive m
        Ylm_real = ((-1)**order_m / np.sqrt(2.0)) * (Ylm + Ylm_conjugate)
    elif order_m == 0:
        # Y_l^0 is already real
        Ylm_real = Ylm
    else:
        # Sine component for negative m (imaginary part extracted)
        Ylm_real = ((-1)**order_m / (1j * np.sqrt(2.0))) * (Ylm - Ylm_conjugate)

    return Ylm_real



def calculate_lbop_r(spherical_coords, atomic_numbers, degree_l, order_m, parity=True, norm_exp = 6):
    """
    Calculate the local bond order paramater for a set of spherical coordinates.

    This function computes the local bond order paramater, which is a measure used
    in the analysis of local atomic environments. It involves
    summing up spherical harmonics for a set of points described by spherical
    coordinates, relative to a central atom assumed to be at the origin.

    Parameters:
    - spherical_coords (np.ndarray): An array of spherical coordinates for the neighbors,
      where each row represents a point with [r, theta, phi] format.
    - degree_l (int): The degree 'l' of the spherical harmonic, a non-negative integer.
    - order_m (int): The order 'm' of the spherical harmonic, where m is an integer
      such that -l <= m <= l.

    Returns:
    - float: The local bond order paramater calculated for the given spherical coordinates
      and spherical harmonic parameters. Weighted by a factor of 1/r^4
    """
    # Get number of nearest neighbors to central atom at (0,0,0)
    
    n_neighbors = spherical_coords.shape[0]

    # Extract r from the spherical coordinates
    r = spherical_coords[:, 0]

    # Extract theta(asimuthal angle) and phi(polar angle) from the spherical coordinates
    theta = spherical_coords[:, 1]
    phi = spherical_coords[:, 2]

    # Compute the spherical harmonics for each [r, theta, phi] pair and sum them up
    # Note: calculate_real_sph_harm expects phi first, then theta

    # Compute considering parity
    if parity == True:
        # Sum over all neighbors
        Ylm_sum = np.sum(sph_harm(order_m, degree_l, theta, phi)*(1/r**norm_exp)*1/atomic_numbers)

    # Compute without considering parity
    else:
        # Sum over all neighbors
        Ylm_sum = np.sum(
            np.abs(calculate_real_sph_harm(order_m, degree_l, theta, phi))*(1/r**norm_exp)*1/atomic_numbers)

    # Calculate the local bond order paramater
    local_bond_order_paramater = 1 / n_neighbors * Ylm_sum

    return local_bond_order_paramater


def calculate_steinhart(spherical_coords, atomic_numbers, degree_l, norm_exp):
    """
    Calculate the Steinhardt parameter (ql) for a given degree l using atomic information
    provided in spherical coordinates. This function computes ql by summing the squares of
    the local bond order parameters (q_lm) for each order m, from -l to l, and then normalizing
    the sum according to the specified degree l.

    Parameters:
    - spherical_coords (array-like): The spherical coordinates of atoms. This should be an array
      where each element represents the spherical coordinates (r, theta, phi) of each atom.
    - atomic_numbers (array-like): An array of atomic numbers corresponding to each atom represented
      in spherical_coords. This is used to differentiate between different types of atoms when calculating q_lm.
    - degree_l (int): The degree l which specifies the level of angular resolution in the calculation
      of the bond order parameters.

    Returns:
    - float: The calculated Steinhardt parameter ql for the provided degree l.
    """
    q_lm_squared_sum = 0  # Initialize the sum of q_lm values
    order_m = -degree_l  # Start with the lowest order m

    # Iterate over all m values from -l to l, inclusive
    while order_m <= degree_l:
        # Calculate the SP for each m and add it to the sum
        q_lm_squared_sum += np.abs((calculate_lbop_r(spherical_coords, atomic_numbers,
                                                     degree_l, order_m, parity, norm_exp)))**2
        order_m += 1  # Move to the next order m

    # Calculate the overall SP for degree l using the accumulated sum of q_lm values
    ql = np.sqrt((4 * np.pi) / (2 * degree_l + 1)
                 * q_lm_squared_sum)

    # Return the ql for given degree l
    return ql


def calculate_steinhart_sum(spherical_coords, atomic_numbers, degree_l, norm_exp):
    """
    Calculates the sum of Steinhardt parameters (q_l) up to a given degree (l) for a cluster of atoms.

    This function calculates the Steinhardt parameters (q_l) for each degree up to the specified degree_l
    using spherical coordinates and atomic numbers provided as input. The Steinhardt parameters are a measure
    of the local structural order around an atom in a cluster and are used to characterize the local symmetry.

    Parameters:
    - spherical_coords (np.ndarray): Array of spherical coordinates with shape (n, 3) for each atom in the cluster.
    - atomic_numbers (np.ndarray or list): Array or list of atomic numbers corresponding to each atom.
    - degree_l (int): The maximum degree (l) for which the Steinhardt parameters (q_l) will be calculated.
                      This function will calculate q_l for all degrees from 0 up to and including degree_l.

    Returns:
    - float: The sum of the calculated Steinhardt parameters (q_l) for each degree from 0 up to degree_l.
    """

    # Sum over ql for each degree
    q_l_sum = 0

    while degree_l >= 0:
        q_lm_squarred_sum = 0  # Initialize the sum of q_lm values
        order_m = -degree_l  # Start with the lowest order m

        # Iterate over all m values from -l to l, inclusive
        while order_m <= degree_l:
            # Calculate the SP for each m and add it to the sum
            q_lm_squarred_sum += np.abs(calculate_lbop_r(spherical_coords, atomic_numbers,
                                                         degree_l, order_m, parity, norm_exp))**2
            order_m += 1  # Move to the next order m

        # Calculate the overall SP for degree l using the accumulated sum of q_lm values
        q_l = np.sqrt((4 * np.pi) / (2 * degree_l + 1) * q_lm_squarred_sum)

        q_l_sum += q_l  # Add q_l for the current degree to the sum

        degree_l -= 1  # Decrease degree_l for the next iteration

    # Return the sum of q_l for the given degree
    return q_l_sum


def compute_steinhart_vector(spherical_coords, atomic_numbers, degree_l, cluster_name="Cluster", norm_exp = 6):
    """
    Compute the Steinhardt parameters for all degrees from 0 up to the specified degree_l
    based on atomic coordinates and types, assuming the coordinates are given in spherical form
    and atomic numbers are provided.

    Parameters:
    - spherical_coords (np.ndarray): Array of atomic spherical coordinates with shape (n, 3), excluding the central atom.
    - atomic_numbers (list or np.ndarray): List or array of atomic numbers corresponding to the atoms, excluding the central atom.
    - degree_l (int): The highest degree (l) of Steinhardt parameters to compute.
    - cluster_name (str): Name of the cluster being processed. 
    - norm_exp (int): 1/r^n norm for lbop

    Returns:
    - list: A list of Steinhardt parameters ql for each degree from 0 to degree_l.
    - str: The name of the cluster (same as input).

    Note:
    - This function assumes that `calculate_steinhart` is available and used to compute the individual
      ql values based on spherical coordinates and atomic numbers.
    """

    ql_list = []
    for i in range(degree_l + 1):  # Compute ql for each degree from 0 to degree_l
        ql = calculate_steinhart(spherical_coords, atomic_numbers, i, norm_exp)
        ql_list.append(ql)

    return ql_list, cluster_name


def site_index_by_symbol(structure, symbol):
    """
    Get the site index of the first occurrence of a specified atom by its chemical symbol.

    Parameters:
        structure (pymatgen.Structure): The atomic structure.
        symbol (str): The chemical symbol of the atom to search for (e.g., "Na").

    Returns:
        int: The index of the first site with the specified symbol, or -1 if not found.
    """
    for site_index, site in enumerate(structure):
        if symbol in site.species_string:
            return site_index  # Return the first matching index
    return -1  # Return -1 if no matching site is found


def get_neighbor_indices_crystalnn(structure, site_index):
    """
    Get the neighbor indices for a specific site using CrystalNN.

    Parameters:
        structure (pymatgen.Structure): The atomic structure.
        site_index (int): Index of the target site.

    Returns:
        list: A list of site indices of the neighbors.
    """
    crystal_nn = CrystalNN()
    neighbors = crystal_nn.get_nn_info(structure, site_index)
    neighbor_indices = [neighbor["site_index"] for neighbor in neighbors]
    return neighbor_indices


def extract_filename(file_path):
    """
    Extracts the filename without extension from a given file path.

    Parameters:
    - file_path (str): The complete file path from which the filename is to be extracted.

    Returns:
    - str: The filename without its extension.

    Example:
    If file_path is 'clusters/octohedral.xyz', the function returns 'octohedral'.
    """
    # Use os.path.basename to get the filename with extension from the file path
    file_name_with_ext = os.path.basename(file_path)

    # Use os.path.splitext to remove the file extension and get the filename
    file_name, _ = os.path.splitext(file_name_with_ext)

    return file_name


def order_data(data):
    """
    Organizes each sublist in the provided data such that the tuples are ordered by the second element in descending order.

    Parameters:
    - data (list of lists of tuples): The data to be organized. Each sublist contains tuples of the form (name, value).

    Returns:
    - list of lists of tuples: The organized data with tuples sorted by the second element in descending order within each sublist.
    """
    # Sort each sublist based on the second element of the tuples (the value), in descending order
    sorted_data = [sorted(sublist, key=lambda x: x[1], reverse=True)
                   for sublist in data]

    return sorted_data


def flatten_data(data):
    """
    Flattens a list of lists of tuples into a single list of tuples.

    Parameters:
    - data (list of lists of tuples): The data to be flattened.

    Returns:
    - list of tuples: The flattened data.
    """
    # Use a list comprehension to flatten the list of lists
    flattened_data = [item for sublist in data for item in sublist]
    return flattened_data


def get_oxidation_state(atom, formula=None, possible_species=None):
    """
    Returns the oxidation state of a given atom, using either a chemical formula
    or a list of possible species strings (e.g., from Materials Project).

    You must provide either:
    - `formula` (str): Chemical formula like "Fe2O3", OR
    - `possible_species` (list): List of strings like ['Fe2+', 'O2-'].

    Parameters:
    - atom (str): The atomic symbol of interest (e.g., 'Fe').
    - formula (str, optional): A chemical formula to guess oxidation states from.
    - possible_species (list of str, optional): Species strings with oxidation states.

    Returns:
    - float or str or None: Oxidation state of the atom, or a message if not found or failed.
    """
    if possible_species:
        for species in possible_species:
            if species.startswith(atom):
                suffix = species[len(atom):]

                if suffix.endswith('+'):
                    return 1.0 if suffix[:-1] == '' else float(suffix[:-1])
                elif suffix.endswith('-'):
                    return -1.0 if suffix[:-1] == '' else -float(suffix[:-1])
                elif suffix.startswith('+'):
                    return 1.0 if suffix[1:] == '' else float(suffix[1:])
                elif suffix.startswith('-'):
                    return -1.0 if suffix[1:] == '' else float(suffix)
                else:
                    try:
                        return float(suffix)
                    except ValueError:
                        continue
        return None  # Not found in species list

    elif formula:
        try:
            comp = Composition(formula)
            guesses = comp.oxi_state_guesses()
            if guesses:
                return guesses[0].get(atom, None)
            else:
                return "Oxidation states could not be determined."
        except Exception as e:
            return f"An error occurred: {e}"

    else:
        raise ValueError("You must provide either 'formula' or 'possible_species'.")



def get_cluster_properties(mp_id, api_key=api_key):
    """
    Retrieves key material properties from the Materials Project for a given MP-ID.

    This includes band gap, density, space group number and symbol, and chemical formula.

    Args:
        mp_id (str): The Materials Project ID of the material.
        api_key (str): Your Materials Project API key.

    Returns:
        dict: A dictionary containing:
            - band gap (float or str)
            - density (float or str)
            - space group number (int or str)
            - space group symbol (str)
            - formula (str): Pretty chemical formula (e.g., "Cr2O3")
    """

    print(f"Using MP ID: {mp_id}")
    print("MP ID repr:", repr(mp_id))

    with MPRester(api_key) as mpr:
        # Search for the material using its MP-ID
        materials = mpr.materials.summary.search(material_ids=[mp_id])

        if not materials:
            raise ValueError(f"No materials found for the MP ID '{mp_id}'. "
                             "Check that it is a valid Materials Project ID.")

        material = materials[0]
        properties = {}

        # Extract properties
        properties['band_gap'] = getattr(material, 'band_gap', "Band gap not available")
        properties['density'] = getattr(material, 'density', "Density not available")
        properties['formula'] = getattr(material, 'formula_pretty', "Formula not available")

        # Get space group data
        if hasattr(material, 'symmetry') and material.symmetry:
            sym = material.symmetry.dict()
            properties['space_group_number'] = sym.get('number', 'Not available')
            properties['space_group_symbol'] = sym.get('symbol', 'Not available')
        else:
            properties['space_group_number'] = "Not available"
            properties['space_group_symbol'] = "Not available"

        return properties


def get_clusters_from_json(file_path): 

    """
    Parses a JSON file containing atomic cluster data for multiple materials and returns
    a structured dictionary mapping MP-IDs to processed cluster information.

    The JSON file is expected to contain keys in the format "mp-XXXXX_Element", where
    "mp-XXXXX" is the Materials Project ID and "Element" is the central atom symbol.
    Each entry contains a dictionary of charges indexed by stringified integers and a list
    of atomic coordinates in the format [x, y, z, charge_index, r].

    Args:
        file_path (str): Path to the input JSON file.

    Returns:
        dict: A dictionary where each key is an MP-ID and each value is another dictionary
              containing:
                - "central_atom" (str): Element symbol of the central atom.
                - "charges" (dict): Mapping of charge indices to charge values.
                - "coords" (list of dict): List of dictionaries, each with:
                    - "x" (float): x-coordinate
                    - "y" (float): y-coordinate
                    - "z" (float): z-coordinate
                    - "charge" (float): Actual charge value (mapped via charge index)
                    - "r" (float): Distance or radius associated with the atom.
    """

    # Load your JSON file
    with open(file_path) as f:
        raw_data = json.load(f)

    # Prepare new dictionary
    processed_data = {}

    # Loop through each material entry
    for full_key, entry in raw_data.items():
        # Split mp-id and central atom
        if "_" in full_key:
            mp_id, central_atom = full_key.split("_", 1)
        else:
            mp_id, central_atom = full_key, "Unknown"

        # Extract charges: assume string keys like "0", "1", ...
        charges = {int(k): v for k, v in entry.items() if k.isdigit()}

        # Extract and process coordinates
        coords = []
        for atom in entry["Atoms Coordinates"]:
            x, y, z, charge_index = map(float, atom)
            charge_val = charges[int(charge_index)]
            coords.append({
                "x": x,
                "y": y,
                "z": z,
                "charge": charge_val,
                "atomic_number": charge_index
            })

        # Store in final structure
        processed_data[mp_id] = {
            "central_atom": central_atom,
            "charges": charges,
            "coords": coords
        }

    return processed_data  


def compute_electronegativity_stats(atomic_numbers):
    """
    Computes the average and standard deviation of electronegativity for a cluster given atomic numbers.

    Parameters:
        atomic_numbers (list or np.ndarray): List or array of atomic numbers (e.g., [24, 52, 24, ...])

    Returns:
        tuple: (average electronegativity, standard deviation of electronegativity)
    """
    electronegativities = []
    for num in atomic_numbers:
        element = Element.from_Z(num)
        electronegativity = element.X  # Pauling electronegativity
        if electronegativity is None:
            raise ValueError(f"Electronegativity not defined for atomic number: {num} ({element.symbol})")
        electronegativities.append(electronegativity)
    
    avg_electronegativity = np.mean(electronegativities)
    std_electronegativity = np.std(electronegativities)
    return avg_electronegativity, std_electronegativity




def read_mp_id_txtfile(file_path):
    """
    Reads a text file containing compound names and their corresponding 
    Materials Project MP-IDs in a specific format, and returns a dictionary.

    The expected file format is:
    # Comment line or header (optional)
    CompoundFormula: MP-ID

    Example:
    NiO: mp-19009
    Fe2O3: mp-19770
    V2O5: mp-25279

    The function skips comment lines that start with a '#' and empty lines.

    Args:
        file_path (str): The path to the text file containing the data.

    Returns:
        dict: A dictionary where the keys are compound formulas (str) 
              and the values are their corresponding Materials Project IDs (str).

    Example:
        >>> read_mp_id_file('compounds.txt')
        {'NiO': 'mp-19009', 'Fe2O3': 'mp-19770', 'V2O5': 'mp-25279'}
    """
    compound_mp_id = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comment or empty lines
            if line.startswith('#') or not line.strip():
                continue
            # Split line by the colon
            parts = line.split(':')
            if len(parts) == 2:
                formula = parts[0].strip()
                mp_id = parts[1].strip()
                compound_mp_id[formula] = mp_id

    return compound_mp_id


def quadrupole_moment_normalized(positions, charges, qm_exponent):
    """
    Calculate the non traceless form of the quadrupole moment tensor for a system of point charges, normalized by the atomic number.

    Args:
    - positions: Nx3 array, where N is the number of particles, and each row is the (x, y, z) coordinates of a particle.
    - charges: 1D array of length N, where each element is the charge of the corresponding particle.
    - qm_exponent: integer, Defines the exponent on the dist for normalization

    Returns:
    - Q: 3x3 numpy array representing the normalized quadrupole moment tensor.
    """
    Q = np.zeros(
        # Initialize the quadrupole moment tensor as a 3x3 zero matrix.
        (3, 3))

    # Loop through each position, charge, and atomic number
    for pos, charge in zip(positions, charges):
        r_x, r_y, r_z = pos
        dist = np.sqrt(r_x**2+r_y**2+r_z**2)

        # Update the Q matrix using the normalized formula.
        normalization_factor = charge / dist**qm_exponent

        Q[0, 0] += normalization_factor * (r_x * r_x)
        Q[0, 1] += normalization_factor * (r_x * r_y)
        Q[0, 2] += normalization_factor * (r_x * r_z)

        Q[1, 0] += normalization_factor * (r_y * r_x)
        Q[1, 1] += normalization_factor * (r_y * r_y)
        Q[1, 2] += normalization_factor * (r_y * r_z)

        Q[2, 0] += normalization_factor * (r_z * r_x)
        Q[2, 1] += normalization_factor * (r_z * r_y)
        Q[2, 2] += normalization_factor * (r_z * r_z)

    return Q

def diagonalize_quadrupole_matrix(Q):
    """
    Diagonalizes a 3x3 quadrupole matrix and returns its eigenvalues.

    Args:
        Q (np.ndarray): 3x3 quadrupole matrix.

    Returns:
        np.ndarray: Eigenvalues of the matrix (in no particular order).
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    return eigenvalues, eigenvectors


def quadrupole_anisotropy_matrix_from_eigenvalues(eigenvalues):
    """
    Forms the 3x3 quadrupole anisotropy matrix from three eigenvalues.
    
    Args:
        eigenvalues (array-like): A list or array of three eigenvalues.
        
    Returns:
        np.ndarray: The 3x3 quadrupole anisotropy matrix.
    """
    # Ensure eigenvalues is a numpy array
    eig = np.array(eigenvalues)
    # Construct the matrix with |eig_i - eig_j|, and 0 on the diagonal
    matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = abs(eig[i] - eig[j])
    return matrix


def quadrupole_anisotropy_matrix(qxx, qyy, qzz):
    """
    Calculate the quadrupole anisotropy matrix.

    Parameters:
        qxx (float): Quadrupole component along xx.
        qyy (float): Quadrupole component along yy.
        qzz (float): Quadrupole component along zz.

    Returns:
        np.ndarray: 3x3 quadrupole anisotropy matrix.
    """
    # Normalization factor
    normalization = 1
    #(qxx + qyy + qzz) / 3.0

    if normalization == 0:
        raise ValueError("Normalization factor is zero; qxx, qyy, and qzz cannot all be zero.")

    # Initialize the anisotropy matrix with absolute differences normalized
    q_anisotropy_matrix = np.array([
    [0, (np.abs(qxx - qyy)) / normalization, (np.abs(qxx - qzz)) / normalization],
    [(np.abs(qyy - qxx)) / normalization, 0, (np.abs(qyy - qzz)) / normalization],
    [(np.abs(qzz - qxx)) / normalization, (np.abs(qzz - qyy)) / normalization, 0]
])


    return q_anisotropy_matrix


def q_anisotropy_matrix_sum(q_anisotropy_matrix):
    """
    Compute the sum of select off-diagonal elements of an anisotropy matrix.

    This function calculates the sum of the elements located at indices (0, 1), (0, 2),
    and (1, 2) of the input matrix. It assumes that the provided matrix is a 2D array-like
    object with at least 3 rows and 3 columns.

    Parameters:
        q_anisotropy_matrix (array-like): A two-dimensional array or matrix representing
            anisotropy values. The matrix must be indexable with two indices and have dimensions
            that allow access to the elements at (0, 1), (0, 2), and (1, 2).

    Returns:
        float: The sum of the matrix elements at positions (0, 1), (0, 2), and (1, 2).
    """

    sum = q_anisotropy_matrix[0,1] + q_anisotropy_matrix[0,2] + q_anisotropy_matrix[1,2] 

    return sum



def dipole_moment_normalized(positions, charges, dm_exponent):
    """
    Compute the normalized dipole moment vector for a system of charges.

    Parameters:
    positions : list of tuples
        A list of 3D position vectors (x, y, z) for the charges.
    charges : list of floats
        A list of charges corresponding to the position vectors.
    dm_exponent: integer
        Defines the exponent on the dist for normalization

    Returns:
    numpy.ndarray
        A 3D vector representing the normalized dipole moment.

    Notes:
    ------
    - Positions with a distance of zero are skipped to avoid division by zero.
    - The normalization factor is calculated as charge / (distance^5).
    """

    # Initialize the dipole moment vector as a 3D vector
    P = np.zeros(3)

    for pos, charge in zip(positions, charges):
        r_x, r_y, r_z = pos
        dist = np.sqrt(r_x**2 + r_y**2 + r_z**2)
        
        # Avoid division by zero in normalization
        if dist == 0:
            continue

        # Normalize the position vector
        normalization_factor = charge / dist**dm_exponent
        P[0] += normalization_factor * r_x
        P[1] += normalization_factor * r_y
        P[2] += normalization_factor * r_z

    return P


def dipole_anisotropy_matrix(dipole_vector):
    """
    Create a matrix where the components are the difference of the components of the vector

    Parameters:
    dipole_vector (array-like): A 3D vector representing the dipole moment.

    Returns:
    numpy.ndarray: A 3x3 matrix as described.
    """
    # Convert the input vector to a NumPy array
    dipole_vector = np.array(dipole_vector)
    
    # Compute the mean of the components
    #mean_value = np.mean(dipole_vector)
    
    # Check for zero mean to avoid division by zero
    #if mean_value == 0:
        #raise ValueError("The mean of the dipole vector components is zero, normalization not possible.")
    
    # Declare the normalization variable
    normalization = 1

    # Initialize the matrix using absolute differences normalized by the mean
    dipole_matrix = np.array([
        [0, np.abs(dipole_vector[0] - dipole_vector[1]) / normalization, np.abs(dipole_vector[0] - dipole_vector[2]) / normalization],
        [np.abs(dipole_vector[1] - dipole_vector[0]) / normalization, 0, np.abs(dipole_vector[1] - dipole_vector[2]) / normalization],
        [np.abs(dipole_vector[2] - dipole_vector[0]) / normalization, np.abs(dipole_vector[2] - dipole_vector[1]) / normalization, 0]
    ])
    
    return dipole_matrix


def d_anisotropy_matrix_sum(dipole_anisotropy_matrix):
    """
    Compute the sum of select off-diagonal elements of a dipole anisotropy matrix.

    This function calculates the sum of the elements located at indices (0, 1), (0, 2),
    and (1, 2) of the input matrix. It assumes that the provided matrix is a 2D array-like
    object with at least 3 rows and 3 columns.

    Parameters:
        dipole_anisotropy_matrix (array-like): A two-dimensional array or matrix representing
            anisotropy values. The matrix must be indexable with two indices and have dimensions
            that allow access to the elements at (0, 1), (0, 2), and (1, 2).

    Returns:
        float: The sum of the matrix elements at positions (0, 1), (0, 2), and (1, 2).
    """

    # Compute the sum of the specific off-diagonal elements
    sum = dipole_anisotropy_matrix[0, 1] + dipole_anisotropy_matrix[0, 2] + dipole_anisotropy_matrix[1, 2]

    return sum

 

def get_charges(atomic_symbols, oxidation_states):
    """
    Given a list of atomic symbols and a dictionary mapping
    symbols to oxidation states, return a list of charges for
    each atomic symbol in atom_list.
    
    :param atom_list: List of atomic symbols (e.g. ["Nb","Se","Cr"]).
    :param oxidation_states: Dictionary mapping atomic symbols to charges 
                             (e.g. {"Nb": 2.5, "Cr": 3.0, "Se": -2.0}).
    :return: List of charges corresponding to the symbols in atom_list.
    :raises ValueError: if a symbol in atom_list is not found in oxidation_states.
    """
    charge_array = []
    
    for atom in atomic_symbols:
        if atom not in oxidation_states:
            raise ValueError(f"Unknown atomic symbol '{atom}' in the oxidation states dictionary.")
        charge_array.append(oxidation_states[atom])
    
    return charge_array


def get_unique_output_folder(base_folder):
    """Generate a unique folder name by adding a numeric suffix if the folder exists."""
    folder = base_folder
    counter = 1
    while os.path.exists(folder):
        folder = f"{base_folder}_{counter}"
        counter += 1
    os.makedirs(folder)
    return folder


class DualWriter:
    def __init__(self, log_file):
        self.terminal = sys.stdout  # Keep the original terminal stdout
        self.log_file = log_file    # File to write logs to

    def write(self, message):
        self.terminal.write(message)    # Write to terminal
        self.log_file.write(message)    # Write to log file

    def flush(self):
        # This flushes the output for both the terminal and the log file
        self.terminal.flush()
        self.log_file.flush()


def write_factor_dictionary_to_file(factor_dict, filename):
    
    """
    Writes the factor dictionary to a JSON file.

    Args:
    - factor_dict: Dictionary containing data (including numpy arrays).
    - filename: The name of the file to write the dictionary to.
    """
    print(f"Started writing dictionary to {filename}")
    # Convert the dictionary to a JSON-serializable format
    serializable_dict = convert_to_json_serializable(factor_dict)

    with open(filename, "w") as fp:
        json.dump(serializable_dict, fp, indent=4)  # Use JSON to serialize the dictionary
        fp.flush()  # Ensure the buffer is flushed
        os.fsync(fp.fileno())  # Ensure file is written to disk
    print(f"Done writing dict to {filename}")


def convert_to_json_serializable(data):
    """
    Recursively convert data to JSON-serializable format.
    Converts numpy arrays to lists, and handles other non-serializable data types as needed.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(element) for element in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_json_serializable(element) for element in data)
    else:
        return data    


class MaterialData:
    def __init__(self, json_data, options):
        """
        Initializes a MaterialData object for a given JSON structure and configuration options.
        """
        self.data = json_data
        self.options = options
        self.material_id = json_data.get("mp-id", "unknown")
        self.row = []
        self.columns = []

    def extract_metadata(self):
        """Extract basic identifying information from the material dictionary."""
        d = self.data
        self.row.extend([
            self.material_id,
            d.get("formula", "unknown"),
            d.get("cif_name", "unknown"),
            d.get("central_atom", "unknown"),
            d.get("space_group_number", np.nan),
        ])
        self.columns.extend([
            "material", "chemical_formula", "cif_name",
            "central_atom", "space_group_number"
        ])

    def extract_sam(self):
        """Extract the average spectral anisotropy matrix (SAM), flattened to a 1D array."""
        sam = self.data.get("avg_spectra_anisotropy_matrix")
        return np.array(sam).flatten() if sam else [None] * 9

    def extract_material_properties(self):
        """Extract general material properties such as bond length, density, electronegativity, and MP descriptors."""
        d = self.data
        props = [
            d.get("average_bond_length", 0),
            d.get("bond_length_std", 0),
            d.get("average_bond_angle", 0),
            d.get("bond_angle_std", 0),
            d.get("number_of_unique_ligands", 0),
            d.get("average_electronegativity", 0),
            d.get("electronegativity_std", 0),
            d.get("band_gap", 0.0),
            d.get("density", 0.0),
            str(d.get("oxidation_states", {})),
            d.get("predicted_formation_energy", 0.0),
            d.get("energy_above_hull", 0.0),
            d.get("total_magnetization", 0.0),
            d.get("mag_sites_ratio", 0.0)
        ]
        headers = [
            "average_bond_length", "bond_length_std", "average_bond_angle", "bond_angle_std",
            "number_of_unique_ligands", "average_electronegativity", "std_electronegativity",
            "band_gap", "density", "oxidation_states",
            "predicted_formation_energy", "energy_above_hull", "total_magnetization", "mag_sites_ratio"
        ]
        self.row.extend(props)
        self.columns.extend(headers)


    def extract_anisotropy_matrix(self, kind):
        """
        Extracts anisotropy matrix (dipole or quadrupole) and their scalar sums.
        Handles both standard and 1/r^n normalized variants based on options.
        
        Parameters:
            kind (str): Either 'dipole' or 'quadrupole'.
        """
        prefix = "dipole" if kind == "dipole" else "quadrupole"
        matrix_prefix = "dam" if kind == "dipole" else "qam"
        sum_label = "dams" if kind == "dipole" else "qams"
        orders = self.options.get("normalization_orders")

        if orders:
            # Handle normalized anisotropy matrices for each 1/r^n
            for n in orders:
                aniso_key = f"normalized_{prefix}_anisotropy_matrix_1/r^{n}"
                sum_key = f"normalized_{prefix}_anisotropy_matrix_sum_1/r^{n}"

                aniso = np.array(self.data.get(aniso_key, [[None]*3]*3)).flatten()
                total = self.data.get(sum_key)

                self.row.extend(aniso)
                self.row.append(total)

                self.columns.extend([f"{matrix_prefix}_1/^{n}_{i}" for i in range(9)])
                self.columns.append(f"{sum_label}_1/r^{n}")
        else:
            # Handle the standard (non-normalized) anisotropy matrix
            aniso_key = f"normalized_{prefix}_anisotropy_matrix"
            sum_key = f"normalized_{prefix}_anisotropy_matrix_sum"

            aniso = np.array(self.data.get(aniso_key, [[None]*3]*3)).flatten()
            total = self.data.get(sum_key)

            self.row.extend(aniso)
            self.row.append(total)

            self.columns.extend([f"{matrix_prefix}_{i}" for i in range(9)])
            self.columns.append(sum_label)



    def extract_steinhart_parameters(self):
        """Extract Steinhart vector and its sum."""
        d = self.data
        vector = d.get("steinhart_vector", [[0]])[0]
        total = d.get("steinhart_parameter_sum", 0.0)

        self.row.extend(vector)
        self.row.append(total)

        self.columns.extend([f"steinhart_vector_{i}" for i in range(len(vector))])
        self.columns.append("steinhart_vector_sum")

    def build_row(self):
        """Build the full row of features for one material."""
        self.extract_metadata()
        if self.options.get("mat_props"):
            self.extract_material_properties()
        if self.options.get("dipole"):
            self.extract_anisotropy_matrix("dipole")
        if self.options.get("quadrupole"):
            self.extract_anisotropy_matrix("quadrupole")
        if self.options.get("steinhart"):
            self.extract_steinhart_parameters()
        return self.row, self.columns


def generate_factor_sam_df(factor_dict_dir_path, **options):
    """
    Constructs two dataframes:
    1. factor_df: containing scalar and tensor properties extracted from each factor dictionary.
    2. sam_df: containing spectral anisotropy matrix (SAM) values.

    Parameters:
    factor_dict_dir_path (str): Directory containing JSON factor dictionaries.
    options (dict): Flags to include/exclude dipole, quadrupole, steinhart, etc.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: (factor_df, sam_df)
    """
    records = []
    columns = None
    sam_records = []
    sam_index = []

    for path in Path(factor_dict_dir_path).glob("*.json"):
        with open(path, "r") as file:
            material_dict = json.load(file)

        # Skip entries if oxidation states were not properly computed (optional)
        if options.get("ox"):
            if material_dict.get("oxidation_states") == "Oxidation states could not be determined.":
                continue

        material = MaterialData(material_dict, options)
        row, row_columns = material.build_row()
        records.append(row)

        if columns is None:
            columns = row_columns

        if options.get("spectra"):
            sam_records.append(material.extract_sam())
            sam_index.append(material.material_id)

    factor_df = pd.DataFrame(records, columns=columns).set_index("material")

    sam_columns = [f"sam_m{i}{j}" for i in range(3) for j in range(3)]
    sam_df = pd.DataFrame(sam_records, columns=sam_columns, index=sam_index)
    sam_df.index.name = "material"

    return factor_df, sam_df


def load_sam_matrix(file_path, center_atom=None):
    """
    Loads a spectral anisotropy matrix (SAM) from a CSV or JSON file, 
    formats the data, and optionally filters by a specified central atom.

    Parameters:
        file_path (str or Path): Path to the .csv or .json file.
        center_atom (str, optional): If provided, filters rows where the central atom matches.

    Returns:
        pd.DataFrame: A formatted DataFrame with 3x3 matrix columns and optional absorber annotation.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".csv":
        sam_df = pd.read_csv(file_path)
        sam_df.set_index(sam_df.columns[0], inplace=True)

        if center_atom:
            sam_df = sam_df[sam_df.index.str.contains(center_atom)]

        # Extract MP-ID and absorber atom from index
        new_index = []
        absorber_atoms = []
        for entry in sam_df.index:
            parts = entry.split('_')
            new_index.append(parts[0])         # e.g., "mp-866094"
            absorber_atoms.append(parts[1])    # e.g., "Zn"

        sam_df.index = new_index
        sam_df.index.name = "Material"
        sam_df["absorber"] = absorber_atoms

    elif ext == ".json":
        with open(file_path, 'r') as f:
            data = json.load(f)

        records = []
        for mpid, mat in data.items():
            flat = [val for row in mat for val in row]
            if len(flat) != 9:
                print(f"Warning: {mpid} has an unexpected matrix shape.")
                continue
            records.append({'Material': mpid, **{f'm{i}{j}': flat[i*3 + j] for i in range(3) for j in range(3)}})

        sam_df = pd.DataFrame(records).set_index("Material")

    else:
        raise ValueError("Unsupported file type. Only .csv and .json are accepted.")

    return sam_df


def extract_all_sites_with_charges(data):
    determined_charges = {str(k): v for k, v in data.get("Determined Charges", {}).items()}
    site_atoms = {}
    for site_label, site_data in data.items():
        if isinstance(site_data, dict) and "Atoms Coordinates" in site_data:
            atoms_list = []
            for atom_entry in site_data["Atoms Coordinates"]:
                x, y, z, atomic_number = atom_entry
                atomic_number_str = str(int(float(atomic_number)))
                charge = float(determined_charges[atomic_number_str]) if atomic_number_str in determined_charges else None
                atoms_list.append([float(x), float(y), float(z), int(atomic_number_str), charge])
            site_atoms[site_label] = atoms_list
    return site_atoms

def load_mpid_dict(subfolder_path):
    """Given the path to a single mp-id subfolder, returns {mpid: {site: [atoms]}}"""
    mpid = os.path.basename(subfolder_path)
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError("No .json file found in " + subfolder_path)
    json_path = os.path.join(subfolder_path, json_files[0])
    with open(json_path, "r") as f:
        data = json.load(f)
    return {mpid: extract_all_sites_with_charges(data)}, data


def print_factor_dict(factor_dict_path):
    """
    Reads a JSON factor dictionary from a given path and prints its contents.

    Parameters:
        factor_dict_path (str or Path): Path to the JSON file.

    Returns:
        dict: The loaded dictionary.
    """
    factor_dict_path = Path(factor_dict_path)

    if not factor_dict_path.exists():
        print(f"Error: File '{factor_dict_path}' not found.")
        return None

    # Load the JSON file
    with open(factor_dict_path, 'r') as file:
        factor_dict = json.load(file)

    # Print the keys and values
    print(f"\nContents of {factor_dict_path.name}:")
    print("-" * 50)
    
    for key, val in factor_dict.items():
        print(f"{key}: {val}")

    return factor_dict



def compute_off_diagonal_sum(anisotropy_spectra_matrix, normalize=False):
    """
    Compute the sum of off-diagonal entries for each row in the given anisotropy spectra matrix.

    Parameters:
    anisotropy_spectra_matrix (pd.DataFrame): A pandas DataFrame containing anisotropy matrix data.
    normalize (bool): If True, normalize the off-diagonal sums by the maximum sum across all rows.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column for off-diagonal sums (normalized if specified).
    """
    off_diagonal_cols = ["sam_m01", "sam_m02", "sam_m12"]

    # Compute the row-wise sum of off-diagonal elements
    off_diag_sum = anisotropy_spectra_matrix[off_diagonal_cols].sum(axis=1)

    if normalize:
        max_sum = off_diag_sum.max()
        # Avoid divide-by-zero
        if max_sum != 0:
            off_diag_sum = off_diag_sum / max_sum

    # Assign the result to a new column
    anisotropy_spectra_matrix['sams'] = off_diag_sum

    return anisotropy_spectra_matrix



def compute_normed_spacegroup_number(factor_df):
    """
    Computes the normalized space group number by dividing 
    each space group number by 230.

    Parameters:
    factor_df (pd.DataFrame): A pandas DataFrame containing a column 
                              "Space Group Number" with integer values.

    Returns:
    pd.DataFrame: The input DataFrame with an added column 
                  "Normed Spacegroup Number" containing the 
                  normalized values.
    """

    if "space_group_number" not in factor_df.columns:
        raise KeyError("The input DataFrame must contain a 'space_group_number' column.")

    normalization = 1 / 230
    factor_df["normed_space_group_number"] = normalization * factor_df["space_group_number"]

    return factor_df


def filter_and_match(factor_df, sam_df):
    """
    Filters two DataFrames to keep only rows with matching MP-IDs and matching
    central atom (in factor_df) and absorber (in sam_df).

    Parameters:
        factor_df (pd.DataFrame): The factor DataFrame indexed by MP-ID.
        sam_df (pd.DataFrame): The spectral anisotropy matrix (sam) DataFrame indexed by MP-ID.

    Returns:
        tuple: (filtered_factor_df, filtered_sam_df, dropped_factor_df)
               where dropped_factor_df contains entries removed from factor_df.
    """
    original_factor_index = factor_df.index

    # Step 1: Keep only rows with matching MP-IDs
    common_mp_ids = factor_df.index.intersection(sam_df.index)
    filtered_factor_df = factor_df.loc[common_mp_ids]
    filtered_sam_df = sam_df.loc[common_mp_ids]

    # Step 2: Keep only rows where central_atom matches absorber
    match = filtered_factor_df["Central Atom"] == filtered_sam_df["Absorber"]
    final_factor_df = filtered_factor_df[match]
    final_sam_df = filtered_sam_df[match]

    # Step 3: Track what was dropped
    dropped_mp_ids = original_factor_index.difference(final_factor_df.index)
    dropped_factor_df = factor_df.loc[dropped_mp_ids]

    return final_factor_df, final_sam_df, dropped_factor_df


def remove_nan_entries(factor_df, sam_df):
    """
    Removes rows with NaN values from the factor DataFrame and filters
    the sam DataFrame accordingly.

    Parameters:
        factor_df (pd.DataFrame): The factor DataFrame indexed by MP-ID.
        sam_df (pd.DataFrame): The spectral anisotropy matrix (sam) DataFrame indexed by MP-ID.

    Returns:
        tuple: (cleaned_factor_df, cleaned_sam_df, dropped_factor_df)
               where dropped_factor_df contains the factor rows removed.
    """
    
    original_factor_index = factor_df.index

    # Drop rows with NaN values from the factor DataFrame
    cleaned_factor_df = factor_df.dropna()

    # Extract MP-IDs of the valid rows
    valid_mp_ids = cleaned_factor_df.index

    # Filter the sam DataFrame to keep only rows matching the valid MP-IDs
    cleaned_sam_df = sam_df.loc[sam_df.index.intersection(valid_mp_ids)]

    # Any factor_df rows not in 'valid_mp_ids' were dropped
    dropped_mp_ids = original_factor_index.difference(valid_mp_ids)
    dropped_factor_df = factor_df.loc[dropped_mp_ids]

    return cleaned_factor_df, cleaned_sam_df, dropped_factor_df


def align_dataframes_by_index(factor_df, sam_df):
    """
    Align two DataFrames by their indices, ensuring they have the same set of
    rows in the same order.

    Args:
        factor_df (pd.DataFrame): First DataFrame (indexed by MP-ID).
        sam_df (pd.DataFrame): Second DataFrame the spectral anisotropy matrix (sam) DataFrame indexed by MP-ID.

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            The aligned factor DataFrame, the aligned sam DataFrame,
            and a factor DataFrame of dropped entries.
    """

    original_factor_index = factor_df.index

    # Find the common indices
    common_indices = factor_df.index.intersection(sam_df.index)

    if common_indices.empty:
        print("Warning: No common indices. Returning empty DataFrames.")


    # Filter and sort both DataFrames by the common indices
    factor_df_aligned = factor_df.loc[common_indices].sort_index()
    sam_df_aligned = sam_df.loc[common_indices].sort_index()

    # Identify dropped factor entries
    dropped_mp_ids = original_factor_index.difference(common_indices)
    dropped_factor_df = factor_df.loc[dropped_mp_ids]

    return factor_df_aligned, sam_df_aligned, dropped_factor_df


def drop_duplicate_indices(df):
    """
    Removes all rows with duplicate indices from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        tuple: (cleaned_df, dropped_df)
    """
    duplicated = df.index.duplicated(keep=False)
    cleaned_df = df[~duplicated]
    dropped_df = df[duplicated]
    return cleaned_df, dropped_df


def align_dataframes(factor_df, sam_df, compare_absorbers = False):
    """
    Cleans and aligns the factor DataFrame and sam spectra DataFrame by:
      1. Dropping all rows with duplicate indices.
      2. Filtering only rows with matching MP-IDs.
      3. Removing rows with NaN values in the factor DataFrame
         and filtering sam accordingly.
      4. Ensuring both DataFrames have the same indices in the same order.

    Parameters:
        factor_df (pd.DataFrame): Factor dictionary DataFrame indexed by MP-ID.
        sam_df (pd.DataFrame): sam spectra
                                                 DataFrame indexed by MP-ID.

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            - Final aligned factor_df
            - Final aligned sam_df
            - factor_df of all dropped entries at any step
    """
    all_dropped_factor_df = pd.DataFrame()

    # 1) Remove duplicate indices
    factor_df, dropped_duplicates1 = drop_duplicate_indices(factor_df)
    sam_df, _ = drop_duplicate_indices(sam_df)
    all_dropped_factor_df = pd.concat([all_dropped_factor_df, dropped_duplicates1])
    print(f"Dropped duplicate indices:\n{dropped_duplicates1}\n")

    if compare_absorbers:
        # 2) Filter for matching MP-IDs and absorbers
        factor_df, sam_df, dropped1 = filter_and_match(factor_df, sam_df)
        all_dropped_factor_df = pd.concat([all_dropped_factor_df, dropped1])
        print(f"Dropped during matching:\n{dropped1}\n")

    # 3) Remove NaN entries in factor_df
    factor_df, sam_df, dropped2 = remove_nan_entries(factor_df, sam_df)
    all_dropped_factor_df = pd.concat([all_dropped_factor_df, dropped2])
    print(f"Dropped due to NaN entries:\n{dropped2}\n")

    
    # 4) Final alignment by index
    factor_df, sam_df, dropped3 = align_dataframes_by_index(factor_df, sam_df)
    all_dropped_factor_df = pd.concat([all_dropped_factor_df, dropped3])
    print(f"Dropped during final alignment:\n{dropped3}\n")

    # Remove duplicate entries in the drop report
    all_dropped_factor_df = all_dropped_factor_df[~all_dropped_factor_df.index.duplicated(keep='first')]

    return factor_df, sam_df, all_dropped_factor_df


def plot_anisotropy_bars(
    qam_matrix, spectra_matrix, qams, smas, sams_pred, title="Anisotropy Matrix Elements and Sums"
):
    qam_flat = np.array(qam_matrix).flatten()
    spectra_flat = np.array(spectra_matrix).flatten()
    indices = [f"m{i}{j}" for i in range(3) for j in range(3)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Matrix elements (no text values)
    x = np.arange(9)
    width = 0.35
    axs[0].bar(x - width/2, qam_flat, width, label='Quadrupole')
    axs[0].bar(x + width/2, spectra_flat, width, label='Spectra')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(indices)
    axs[0].set_ylabel("Matrix value")
    axs[0].set_title("Matrix Elements")
    axs[0].legend(loc='upper left', bbox_to_anchor=(0,1))  # Move legend to upper left

    # Plot 2: Sums (with text values)
    labels = ["Quadrupole Sum", "Spectra Sum", "Predicted Spectra Sum"]
    sums = [qams, smas, sams_pred]
    bar_colors = ["#348ABD", "#E24A33", "#3CB371"]  # blue, orange, green
    x2 = np.arange(3)
    axs[1].bar(x2, sums, color=bar_colors)
    axs[1].set_xticks(x2)
    axs[1].set_xticklabels(labels, rotation=15)
    axs[1].set_ylabel("Sum value")
    axs[1].set_title("Matrix Sums")
    for i, v in enumerate(sums):
        axs[1].text(x2[i], v, f"{v:.5f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle(title, fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def get_anisotropy_matrices_and_sums(factor_df, spectra_df, mp_id):
    """
    Returns qam_matrix, spectra_matrix, qams, smas for the given mp_id.
    """
    qam_matrix = factor_df.loc[mp_id, [f"Aniso QM 1/r^7 {i}" for i in range(9)]].values.reshape(3,3)
    qams = factor_df.loc[mp_id, "Aniso Sum QM 1/r^7"]

    spectra_matrix = spectra_df.loc[mp_id, [f"m{i}{j}" for i in range(3) for j in range(3)]].values.reshape(3,3)
    smas = spectra_df.loc[mp_id, "Anisotropy Matrix Sum"]

    return qam_matrix, spectra_matrix, qams, smas


def plot_anisotropy_bars_all(
    factor_df, spectra_df, mp_ids, y_pred, title_prefix="Anisotropy Matrix Elements and Sums for "
):
    """
    Plots anisotropy bars for a list of mp-ids.
    
    Args:
        factor_df: DataFrame with factor data
        spectra_df: DataFrame with spectra data
        mp_ids: list of mp-id strings
        y_pred: dict or Series mapping mp-id to predicted spectra sum
        title_prefix: str to prepend to each title
    """
    for mp_id in mp_ids:
        try:
            qam_matrix, spectra_matrix, qams, sams = get_anisotropy_matrices_and_sums(factor_df, spectra_df, mp_id)
            sams_pred = y_pred[mp_id]
            plot_anisotropy_bars(
                qam_matrix, spectra_matrix, qams, sams, sams_pred,
                title=f"{title_prefix}{mp_id}"
            )
        except Exception as e:
            print(f"Could not plot {mp_id}: {e}")

def analyze_outliers_over_domain(
    y_true,
    y_pred,
    ids=None,
    threshold=None,
    domain=None,
    n_top=10
):
    """
    Finds outliers within a domain, returns a dataframe of just those outliers,
    and plots all points (blue) with only outliers (in-domain & outlier) as red.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if ids is None:
        ids = np.arange(len(y_true))

    abs_error = np.abs(y_true - y_pred)
    pred_ratio = y_pred / y_true
    df = pd.DataFrame({
        'id': ids,
        'y_true': y_true,
        'y_pred': y_pred,
        'abs_error': abs_error,
        'pred_ratio': pred_ratio
    })

    # --- Restrict to domain for finding outliers ---
    if domain is not None:
        domain_min, domain_max = domain
        mask_domain = (df['y_true'] >= domain_min) & (df['y_true'] <= domain_max)
        df_domain = df[mask_domain]
    else:
        df_domain = df.copy()

    # --- Find outliers in domain ---
    if threshold is not None:
        mask_outlier = df_domain['abs_error'] > threshold
    else:
        # Take the top n_top by abs_error in domain
        mask_outlier = np.zeros(len(df_domain), dtype=bool)
        if len(df_domain) > 0:
            mask_outlier[np.argsort(-df_domain['abs_error'].values)[:n_top]] = True

    df_outliers = df_domain[mask_outlier].copy()
    outlier_ids = set(df_outliers['id'])

    # --- Plot (full) ---
    plt.figure(figsize=(10, 7))
    # All points
    plt.scatter(df['y_true'], df['y_pred'], color='blue', label='All Data')
    # Outliers (only those found in domain)
    if len(df_outliers) > 0:
        mask_plot = df['id'].isin(outlier_ids)
        plt.scatter(df.loc[mask_plot, 'y_true'], df.loc[mask_plot, 'y_pred'],
                    color='red', label='Outliers in Domain')
    lims = [min(df['y_true'].min(), df['y_pred'].min()),
            max(df['y_true'].max(), df['y_pred'].max())]
    plt.plot(lims, lims, 'r--', lw=2, alpha=0.5)
    plt.xlabel("Actual Output (y)")
    plt.ylabel("Predicted Output ($\\hat{{y}}$)")
    plt.title(f"Predicted vs Actual Output\n(Only Domain Outliers in Red)\nThreshold: {threshold}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot (zoomed domain) ---
    if domain is not None:
        plt.figure(figsize=(10, 7))
        # All domain points
        plt.scatter(df_domain['y_true'], df_domain['y_pred'], color='blue', label='Domain Data')
        # Domain outliers
        if len(df_outliers) > 0:
            plt.scatter(df_outliers['y_true'], df_outliers['y_pred'],
                        color='red', label='Outliers in Domain')
        lims = [min(df_domain['y_true'].min(), df_domain['y_pred'].min()),
                max(df_domain['y_true'].max(), df_domain['y_pred'].max())]
        plt.plot(lims, lims, 'r--', lw=2, alpha=0.5)
        plt.xlabel("Actual Output (y)")
        plt.ylabel("Predicted Output ($\\hat{{y}}$)")
        plt.title(f"Predicted vs Actual Output (Zoom: {domain}) (Threshold: {threshold})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_outliers.reset_index(drop=True)

def plot_distribution(df, column_name, bins=30):
    """
    Plot the distribution of a specific column in a DataFrame using a histogram.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to plot.
    bins : int, optional (default=30)
        Number of bins to use in the histogram.

    Returns:
    -------
    None
        Displays the histogram plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name].dropna(), bins=bins)
    plt.title(f'Distribution of {column_name}', fontsize=16)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
