from datetime import datetime 

load_up_start = datetime.now()

import json
import pprint
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers  # Import the helpers module from the parent directory

import numpy as np
from pymatgen.core import Structure

import importlib
importlib.reload(helpers) #Reload helpers if necessary

#Define Global Variables

DEGREE_L = 10 #Defined for steinhart parameters

#Specify the range of exponents for the distance norm in the ST QM and DM
NORM_EXP_LOW = 7
NORM_EXP_HIGH = 8

CHARGES = "non uniform" #Set to "uniform" to make all charges equal to 1

load_up_end = datetime.now()

total_load_time = load_up_end - load_up_start #Gets the time needed to load all the dependencies 

def factor_dictionary(mp_id, central_atom, mpid_dict):
    """
    Generate a dictionary of physical and chemical descriptors for all local clusters
    (sites) around a specified central atom in a material, and return their averages.

    For each atomic environment ("site") found in the cluster extracted from mpid_dict,
    this function computes a broad set of local descriptors: electronegativity statistics,
    Steinhardt parameters, dipole/quadrupole moments and their anisotropies, for a range
    of normalization exponents. The function then averages each descriptor across all sites.

    Parameters
    ----------
    mp_id : str
        Materials Project ID for the material (e.g., "mp-1044222").

    central_atom : str
        Chemical symbol of the central atom (e.g., "Ti", "Fe") for which local clusters are defined.

    mpid_dict : dict
        Dictionary structured as {mpid_centralatom: {site_label: [atom_data, ...], ...}}, where
        each atom_data is [x, y, z, atomic_number, charge].

    Returns
    -------
    factor_dict : dict or None
        Dictionary mapping descriptor names to their average values across all sites.
        Returns None if any required calculation fails.

    Notes
    -----
    - Each site (Wyckoff position) is processed independently; the final output only contains
      the average value for each descriptor over all sites.
    - The following descriptors are computed per-site (when possible):
        - Average and standard deviation of electronegativity
        - Steinhardt parameters (vector and sum) for various normalization exponents
        - Dipole moment, dipole anisotropy, and anisotropy sum (for various exponents)
        - Quadrupole moment, quadrupole anisotropy, and anisotropy sum (for various exponents)
    - Structural/chemical metadata (MP-ID, central atom, material properties) are also included.
    - The function expects that mpid_dict has been preprocessed into the correct format.
    - If oxidation states, material properties, or any descriptor cannot be computed, returns None.
    - Prints progress and descriptor calculation details for each site to stdout (and log if redirected).

    Example
    -------
    >>> factor_dict = factor_dictionary("mp-1044222", "Ti", mpid_dict)
    >>> print(factor_dict["average electronegativity"])
    2.14
    """

    factor_dict = {}

    print("Adding mp-id and cif name to factor dictionary")

    factor_dict['mp-id'] = mp_id
    factor_dict['central_atom'] = central_atom


    # Retrieve material properties and add them to the dictionary
    print(f"\nRetrieving material properties (Band gap, density, chemical formula, space group number, and space group symbol) for MP-ID {mp_id}")
    try:
        print(mp_id)
        properties = helpers.get_cluster_properties(mp_id)
        factor_dict.update(properties)
        print("Material properties retrieved successfully.")
    except Exception as e:
        print(f"Error retrieving material properties: {e}")
        return None
    
    # Initialize a pretty printer for better output formatting
    PP = pprint.PrettyPrinter(indent=2)

    print(f"Initializing the factors")

    # Extract coord data from the mpid_dict
    sites_dict = mpid_dict[mp_id+"_"+central_atom]

    factor_dicts = []  # To store factor_dict for each site

    for index, (site_label, atom_list) in enumerate(sites_dict.items()):
        print(f"\nProcessing site: {site_label}")


        # Build the list of dicts for each atom, to match your target format
        atoms = [
            {"x": atom[0], "y": atom[1], "z": atom[2], "atomic_number": atom[3], "charge": atom[4]}
            for atom in atom_list
        ]
        
        # Create the arrays just like your example
        coords = np.array([(atom["x"], atom["y"], atom["z"]) for atom in atoms])
        coords = np.atleast_2d(coords)

        charges = np.array([atom["charge"] for atom in atoms])
        charges = np.atleast_1d(charges)

        atomic_numbers = np.array([atom["atomic_number"] for atom in atoms])
        atomic_numbers = np.atleast_1d(atomic_numbers)

        # Get rid of atom at the origin
        coords = coords[1:]
        charges = charges[1:]
        atomic_numbers = atomic_numbers[1:]

        # Temporary dictionary to hold factors for this site
        factor_dict_temp = {}

        print(f"[{site_label}] Computing average and standard deviation of electronegativity for cluster")
        try:
            avg_en, std_en = helpers.compute_electronegativity_stats(atomic_numbers)
        except Exception as e:
            print(f"[{site_label}] Error computing average and std of electronegativity: {e}")
            return None

        factor_dict_temp["average_electronegativity"] = avg_en
        factor_dict_temp["electronegativity_std"]= std_en

        print(f"[{site_label}] Converting Cartesian coordinates to spherical coordinates to calculate steinhart parameters")
        spherical_coords = helpers.cartesian_to_spherical(coords)
        print(f"[{site_label}] Spherical coordinates:\n{spherical_coords}")

        for st_exp in range(NORM_EXP_LOW, NORM_EXP_HIGH):

            print(f"[{site_label}] Computing Steinhart vector (l={DEGREE_L}) for the cluster with normalization 1/r^{st_exp}")
            try:
                steinhart_vector = helpers.compute_steinhart_vector(spherical_coords, atomic_numbers, DEGREE_L, norm_exp=st_exp)
                print(f"[{site_label}] {steinhart_vector}")
                factor_dict_temp[f"steinhart_vector_1/r^{st_exp}"] = steinhart_vector
                print(f"[{site_label}] Steinhart vector 1/r^{st_exp} normalization computed successfully.")
            except Exception as e:
                print(f"[{site_label}] Error computing Steinhart vector: {e}")
                return None

            print(f"[{site_label}] Computing Steinhart parameter sum normalization 1/r^{st_exp}")
            try:
                steinhart_param_sum = helpers.calculate_steinhart_sum(spherical_coords, atomic_numbers, DEGREE_L, st_exp)
                factor_dict_temp[f"steinhart_parameter_sum 1/r^{st_exp}"] = steinhart_param_sum
                print(f"[{site_label}] Steinhart parameter sum norm 1/r^{st_exp}: {steinhart_param_sum}")
            except Exception as e:
                print(f"[{site_label}] Error computing Steinhart parameter sum: {e}")
                return None

        if CHARGES == "uniform":
            print(f"[{site_label}] ----------------- Using uniform charges -----------------")
            charges = np.ones(len(coords))

        print(f"[{site_label}] Charges: {charges}")

        for exp in range(NORM_EXP_LOW, NORM_EXP_HIGH):

            print(f"[{site_label}] Computing normalized dipole moment with 1/r^{exp}")
            try:
                dipole_moment_normalized = helpers.dipole_moment_normalized(coords, charges, exp)
                factor_dict_temp[f"dipole_moment_normalized_1/r^{exp}"] = dipole_moment_normalized
                print(f"[{site_label}] Normalized dipole moment 1/r^{exp} calculated successfully.")
            except Exception as e:
                print(f"[{site_label}] Error computing normalized dipole moment 1/r^{exp}: {e}")
                return None
            
            print(f"[{site_label}] Computing normalized dipole anisotropy matrix 1/r^{exp}")
            try:
                normalized_dipole_anisotropy_matrix = helpers.dipole_anisotropy_matrix(dipole_moment_normalized)
                factor_dict_temp[f"normalized_dipole_anisotropy_matrix_1/r^{exp}"] = normalized_dipole_anisotropy_matrix
                print(f"[{site_label}] Normalized dipole anisotropy matrix 1/r^{exp} calculated successfully.")
            except Exception as e:
                print(f"[{site_label}] Error computing normalized dipole anisotropy matrix 1/r^{exp}: {e}")
                return None

            print(f"[{site_label}] Computing normalized dipole anisotropy matrix sum 1/r^{exp}")
            try:
                normalized_dipole_anisotropy_matrix_sum = helpers.d_anisotropy_matrix_sum(normalized_dipole_anisotropy_matrix)
                factor_dict_temp[f"normalized_dipole_anisotropy_matrix_sum_1/r^{exp}"] = normalized_dipole_anisotropy_matrix_sum
                print(f"[{site_label}] Computing normalized dipole anisotropy matrix sum 1/r^{exp} calculated successfully.")
            except Exception as e:
                print(f"[{site_label}] Error computing normalized dipole anisotropy matrix sum 1/r^{exp}: {e}")
                return None

            print(f"[{site_label}] Computing normalized quadrupole moment with 1/r^{exp}")
            try:
                quad_moment_normalized = helpers.quadrupole_moment_normalized(coords, charges, exp)
                factor_dict_temp[f"quadrupole_moment_normalized_1/r^{exp}"] = quad_moment_normalized
                print(f"[{site_label}] Normalized quadrupole moment 1/r^{exp} calculated successfully.")
            except Exception as e:
                print(f"[{site_label}] Error computing normalized quadrupole moment 1/r^{exp}: {e}")
                return None
            
            print(f"[{site_label}] Computing normalized quadrupole anisotropy matrix 1/r^{exp}")
            try:
                eigenvalues, eigenvectors = helpers.diagonalize_quadrupole_matrix(quad_moment_normalized)
                print(f"[{site_label}] Eigenvalues: {eigenvalues}")
                normalized_quadrupole_anisotropy_matrix = helpers.quadrupole_anisotropy_matrix_from_eigenvalues(eigenvalues)
                factor_dict_temp[f"normalized_quadrupole_anisotropy_matrix_1/r^{exp}"] = normalized_quadrupole_anisotropy_matrix
                print(f"[{site_label}] Normalized quadrupole anisotropy matrix 1/r^{exp} calculated successfully.")
            except Exception as e:
                print(f"[{site_label}] Error computing normalized quadrupole anisotropy matrix 1/r^{exp}: {e}")
                return None
            
            print(f"[{site_label}] Computing normalized quadrupole anisotropy matrix sum 1/r^{exp}")
            try:
                normalized_quadrupole_anisotropy_matrix_sum = helpers.q_anisotropy_matrix_sum(normalized_quadrupole_anisotropy_matrix)
                factor_dict_temp[f"normalized_quadrupole_anisotropy_matrix_sum_1/r^{exp}"] = normalized_quadrupole_anisotropy_matrix_sum
                print(f"[{site_label}] Computing normalized quadrupole anisotropy matrix sum 1/r^{exp} calculated successfully.")
            except Exception as e:
                print(f"[{site_label}] Error computing normalized quadrupole anisotropy matrix sum 1/r^{exp}: {e}")
                return None
        
        # ... inside your loop:
        print(f"\n[{site_label}] Factor dict:")
        PP.pprint(factor_dict_temp)

        factor_dicts.append(factor_dict_temp)

    # Average the factors across all sites
    if not factor_dicts:
        print("No factor dictionaries were generated. Exiting.")
        return None 
    
    print("Averaging the factors across all sites")

    # Find all keys present (across all dicts)
    all_keys = set().union(*(d.keys() for d in factor_dicts))

    for key in all_keys:
        vals = [d[key] for d in factor_dicts if key in d]
        if not vals:
            continue
        try:
            arr = np.array(vals)
            avg_val = np.mean(arr, axis=0)
            # Convert 0d array to scalar float for pretty output
            if np.isscalar(avg_val) or getattr(avg_val, "shape", ()) == ():
                avg_val = float(avg_val)
            factor_dict[key] = avg_val
        except Exception:
            continue  # Skips keys with non-numeric values

    print("Factor dictionary computation completed successfully.")    

    sys.stdout.flush()  # Flush to ensure all output is printed
    return factor_dict


def process_clusters(
    root_dir,
    output_folder,
    log_message=None,
):
    """
    Processes all cluster data folders in a specified root directory,
    computes averaged physical/chemical descriptors for each cluster,
    and writes the results and logs to an output folder.

    For each mp-id/central-atom subfolder in `root_dir`, this function:
    - Loads preprocessed atomic environment data (via helpers.load_mpid_dict).
    - Computes a comprehensive factor dictionary using `factor_dictionary`.
    - Writes the averaged factor dictionary to a JSON file.
    - Logs progress, failures, and summary statistics.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing subfolders (one per material/cluster, named as "mpid_centralatom").

    output_folder : str
        Path to the folder where the output factor dictionaries and logs will be written.
        A unique subfolder will be created automatically if the target already exists.

    log_message : str, optional
        An optional message to include at the start of the process log for context or description.

    Returns
    -------
    None

    Notes
    -----
    - Each subfolder in `root_dir` should contain a JSON file with the atomic environments needed by `factor_dictionary`.
    - Results for each cluster are written as a separate JSON file in the output folder.
    - Logs include per-cluster progress, error messages for failed runs, and overall summary statistics.
    - If any computation fails for a subfolder, the error is logged and processing continues with the next subfolder.
    """
    # Ensure a unique output folder exists
    output_folder = helpers.get_unique_output_folder(output_folder)

    # Create a subfolder for logs inside the output folder
    logs_folder = os.path.join(output_folder, "logs")
    os.makedirs(logs_folder, exist_ok=True)  # Create the logs folder if it doesn't exist

    # Get time
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set up the log file
    log_path = os.path.join(logs_folder, "process_log.txt")
    
    with open(log_path, "w") as log_file:
        # Add the custom log message if provided
        if log_message:
            log_file.write(f"{log_message}\n\n")
        # Write the start of the log
        log_file.write(f"Beginning Factor Dictionary Calculations at {time_str}\n")
        log_file.write("=" * 50 + "\n")

    # Redirect stdout to both log file and terminal
    with open(log_path, "a") as log_file:
        sys.stdout = helpers.DualWriter(log_file)

        try:
            print("Starting to Compute Factor Dictionary\n")

            # Progress metrics
            successful_runs = 0
            failed_runs = 0
            index = 1
            failed_index = []

            for subfolder in os.listdir(root_dir):
                subfolder_path = os.path.join(root_dir, subfolder)
                if os.path.isdir(subfolder_path):
                    try:

                        mpid_site_dict, mpid_json_data = helpers.load_mpid_dict(subfolder_path)  # defined earlier

                        # Get the mp_id and central atom from the mpid_dict
                        mpid_centralatom = list(mpid_site_dict.keys())[0]
                        mp_id, central_atom = mpid_centralatom.rsplit("_", 1)

                        print(f"Processing subfolder: {subfolder_path} with mp_id: {mp_id} and central atom: {central_atom}")

                        factor_dict = factor_dictionary(mp_id, central_atom, mpid_site_dict)
                        
                        if factor_dict is not None:

                            # Check if "Avg Spectral Anisotropy Matrix" exists in mpid_dict, then get it and add to factor_dict
                            if "Avg Spectral Anisotropy Matrix" in mpid_json_data:
                                avg_spectral_aniso_matrix = mpid_json_data["Avg Spectral Anisotropy Matrix"]
                                factor_dict["avg_spectra_anisotropy_matrix"] = avg_spectral_aniso_matrix
                                print("Avg Spectral Anisotropy Matrix:", avg_spectral_aniso_matrix)
                            else:
                                print("Avg Spectral Anisotropy Matrix not found in this mpid_dict.")

                            # Write out the factor dictionary
                            output_filename = f"{mp_id}_{central_atom}_factor_dict.json"
                            output_path = os.path.join(output_folder, output_filename)
                            helpers.write_factor_dictionary_to_file(factor_dict, output_path)
                            successful_runs += 1   
                        else: 
                            raise ValueError(f"factor_dict is None for {mp_id}")  # <-- Only count as fail if exception is raised

                        index += 1

                    except Exception as e:
                        print(f"Skipping {subfolder_path}: {e}")
                        failed_runs += 1
                        failed_index.append(("None", "Index:", index, mp_id if 'mp_id' in locals() else 'unknown'))


            # Summary of runs
            print(f"\nSummary:")
            print(f"Success: {successful_runs}")
            print(f"Failures: {failed_runs}")

            if failed_index:
                print(f"Failure {failed_index}")


            # Log the total computation time
            end_time = datetime.now()
            total_time = end_time - datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') + total_load_time

            # Format the total time nicely
            hours, remainder = divmod(total_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"Approximate total computation time: {hours}h {minutes}m {seconds:.2f}s")

        finally:
            sys.stdout = sys.__stdout__  # Restore original stdout



example_log_message = """Date: 5/31/2025 Cluster Data: Example"""
test_for_big_dataset_log_message = """Date: 6/7/2025 Cluster Data: test_for_big_dataset"""

#process_clusters("structure_data/Example", "transformed_data/ExampleFD/factor_df_5_31_2025", log_message=example_log_message)

process_clusters("structure_data/test_for_big_dataset", "transformed_data/test_for_big_dataset/factor_df_6_7_2025", log_message=test_for_big_dataset_log_message)