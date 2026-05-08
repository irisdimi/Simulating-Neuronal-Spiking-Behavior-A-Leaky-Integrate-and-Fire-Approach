import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")


def load_and_analyze(data_folder, session="ec013.528", electrode=1, max_neurons=20):
    """Combined loading and analysis function."""

    # File paths
    files = [f"{session}.res.{electrode}", f"{session}.clu.{electrode}"]

    if not all(os.path.exists(os.path.join(data_folder, f)) for f in files):
        files = [f"{session}.res", f"{session}.clu"]

    res_file, clu_file = [os.path.join(data_folder, f) for f in files]

    try:
        # Load data efficiently
        spike_times = np.loadtxt(res_file, dtype=int) / 20000 * 1000
        cluster_ids = np.loadtxt(clu_file, dtype=int, skiprows=1)

        # Filter and create DataFrame in one step
        mask = cluster_ids > 1
        data = pd.DataFrame({
            "spike_time": spike_times[mask],
            "neuron_id": cluster_ids[mask]
        })

        # Analyze neurons using vectorized operations
        duration_sec = spike_times.max() / 1000
        stats = []

        for neuron_id in data["neuron_id"].unique()[:max_neurons]:
            spikes = data[data["neuron_id"] == neuron_id]["spike_time"].values

            if len(spikes) > 1:
                isi = np.diff(spikes)
                stats.append([
                    neuron_id,
                    len(spikes) / duration_sec,
                    np.std(isi) / np.mean(isi),
                    len(spikes)
                ])

        stats_df = pd.DataFrame(
            stats,
            columns=["neuron_id", "firing_rate", "cv_isi", "n_spikes"]
        )

        print(f"Loaded {len(data):,} spikes from {len(data['neuron_id'].unique())} neurons")
        print(f"Duration: {duration_sec / 60:.1f} minutes")

        print("First 5 rows of spike data:")
        print(data.head().to_string(index=False))

        print("First 5 neurons analyzed:")
        print(stats_df.head().to_string(index=False, float_format="%.3f"))

        return data, stats_df, duration_sec

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


def create_plots(data, stats):
    """Optimized plotting function."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("CRCNS Neural Data Analysis", fontsize=14, fontweight="bold")

    # Get data subsets once
    time_limit = min(300000, data["spike_time"].max())
    plot_data = data[data["spike_time"] <= time_limit]
    top_neurons = plot_data["neuron_id"].unique()[:10]

    # 1. Spike raster
    for i, neuron_id in enumerate(top_neurons):
        spikes = plot_data[plot_data["neuron_id"] == neuron_id]["spike_time"].values

        if len(spikes) > 0:
            axes[0, 0].scatter(spikes / 1000, [i] * len(spikes), s=2, alpha=0.7)

    axes[0, 0].set(
        xlabel="Time (seconds)",
        ylabel="Neuron ID",
        title="Spike Raster (5 min)"
    )

    # 2. Firing rate distribution
    sns.histplot(stats["firing_rate"], bins=20, kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(
        stats["firing_rate"].mean(),
        color="red",
        linestyle="--"
    )
    axes[0, 1].set(
        xlabel="Firing Rate (Hz)",
        title="Firing Rate Distribution"
    )

    # 3. ISI distribution
    all_isis = np.concatenate([
        np.diff(data[data["neuron_id"] == nid]["spike_time"].values)
        for nid in top_neurons
        if len(data[data["neuron_id"] == nid]) > 1
    ])

    sns.histplot(all_isis[all_isis < 100], bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].set(
        xlabel="Inter-Spike Interval (ms)",
        title="ISI Distribution"
    )

    # 4. Rate vs regularity
    sns.scatterplot(data=stats, x="firing_rate", y="cv_isi", ax=axes[1, 1])
    axes[1, 1].set(
        xlabel="Firing Rate (Hz)",
        ylabel="CV of ISI",
        title="Rate vs Regularity"
    )

    plt.tight_layout()
    return fig


def save_results(fig, stats, filename="crcns_analysis"):
    """Save results with error handling."""

    saved_files = []

    # Save plot
    try:
        fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
        saved_files.append(f"{filename}.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # Save data
    try:
        stats.to_csv(f"{filename}_stats.csv", index=False)
        saved_files.append(f"{filename}_stats.csv")
    except Exception as e:
        print(f"Could not save CSV: {e}")

    if saved_files:
        print(f"Saved: {', '.join(saved_files)}")

    return saved_files


def print_summary(stats):
    """Print formatted summary statistics."""

    print("Summary Statistics:")
    print(
        f"Mean firing rate: "
        f"{stats['firing_rate'].mean():.2f} ± {stats['firing_rate'].std():.2f} Hz"
    )
    print(
        f"Rate range: "
        f"{stats['firing_rate'].min():.2f} - {stats['firing_rate'].max():.2f} Hz"
    )
    print(f"Mean CV ISI: {stats['cv_isi'].mean():.3f}")
    print(f"Active neurons (>0.1 Hz): {len(stats[stats['firing_rate'] > 0.1])}")


def main():
    """Streamlined main function."""

    print("CRCNS Data Analyzer")
    print("=" * 30)

    # Get input
    data_path = input("Enter CRCNS data folder path: ").strip() or "./"

    # Load and analyze
    data, stats, duration = load_and_analyze(data_path)

    if data is None:
        return

    # Generate results
    print("Generating analysis...")
    fig = create_plots(data, stats)
    print_summary(stats)
    save_results(fig, stats)

    plt.show()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
