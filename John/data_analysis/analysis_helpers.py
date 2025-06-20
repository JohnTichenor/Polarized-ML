import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    df = pd.DataFrame({
        'id': ids,
        'y_true': y_true,
        'y_pred': y_pred,
        'abs_error': abs_error
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
    plt.title("Predicted vs Actual Output\n(Only Domain Outliers in Red)")
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
        plt.title(f"Predicted vs Actual Output (Zoom: {domain})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_outliers.reset_index(drop=True)
