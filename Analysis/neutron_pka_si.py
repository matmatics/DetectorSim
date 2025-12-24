from __future__ import annotations

import argparse
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class HistogramBin:
    energy_low_mev: float
    energy_high_mev: float
    count: int


@dataclass
class SimulationResult:
    n_neutrons: int
    neutron_energy_mev: float
    recoil_max_energy_mev: float
    average_recoil_energy_mev: float
    average_damage_energy_mev: float
    total_damage_energy_mev: float
    niel_per_neutron_mev: float
    niel_per_neutron_kev: float
    average_displacements: float
    histogram: List[HistogramBin]


def recoil_energy_max(neutron_energy_mev: float, target_mass_ratio: float) -> float:
    """Return the maximum recoil energy (in MeV) for elastic n-target scattering."""
    return 4 * target_mass_ratio / (1 + target_mass_ratio) ** 2 * neutron_energy_mev


def kinchin_pease_damage(recoil_energy_mev: float, displacement_energy_mev: float) -> tuple[float, float]:
    """Compute Kinchin-Pease damage energy and displacement count for a recoil."""
    if recoil_energy_mev < displacement_energy_mev:
        return 0.0, 0.0
    if recoil_energy_mev < 2 * displacement_energy_mev:
        return displacement_energy_mev, 1.0
    displacements = recoil_energy_mev / (2.0 * displacement_energy_mev)
    return recoil_energy_mev / 2.0, displacements


def uniform_histogram(samples: Iterable[float], bins: int, energy_max: float) -> List[HistogramBin]:
    bin_width = energy_max / bins
    counts = [0 for _ in range(bins)]
    for value in samples:
        index = int(value / bin_width)
        if index == bins:
            index -= 1
        counts[index] += 1
    histogram: List[HistogramBin] = []
    for i, count in enumerate(counts):
        low = i * bin_width
        high = low + bin_width
        histogram.append(HistogramBin(low, high, count))
    return histogram


def run_simulation(
    n_neutrons: int = 100_000,
    neutron_energy_mev: float = 1.0,
    silicon_atomic_mass: float = 28.0855,
    displacement_energy_ev: float = 25.0,
    n_bins: int = 50,
    seed: int = 42,
) -> SimulationResult:
    """
    Simulate elastic scattering of mono-energetic neutrons in silicon.

    We assume isotropic scattering in the center-of-mass frame so that the
    recoil spectrum is uniform between 0 and the kinematic maximum.
    Damage is estimated using the Kinchin-Pease model.
    """

    random.seed(seed)
    mass_ratio = silicon_atomic_mass  # relative to neutron mass units
    recoil_max = recoil_energy_max(neutron_energy_mev, mass_ratio)

    recoil_energies: List[float] = [random.uniform(0.0, recoil_max) for _ in range(n_neutrons)]

    displacement_energy_mev = displacement_energy_ev * 1.0e-6

    total_damage = 0.0
    total_displacements = 0.0
    for recoil in recoil_energies:
        damage_energy, displacements = kinchin_pease_damage(recoil, displacement_energy_mev)
        total_damage += damage_energy
        total_displacements += displacements

    histogram = uniform_histogram(recoil_energies, n_bins, recoil_max)

    avg_recoil = sum(recoil_energies) / n_neutrons
    avg_damage_energy = total_damage / n_neutrons
    avg_displacements = total_displacements / n_neutrons
    niel_per_neutron = avg_damage_energy

    return SimulationResult(
        n_neutrons=n_neutrons,
        neutron_energy_mev=neutron_energy_mev,
        recoil_max_energy_mev=recoil_max,
        average_recoil_energy_mev=avg_recoil,
        average_damage_energy_mev=avg_damage_energy,
        total_damage_energy_mev=total_damage,
        niel_per_neutron_mev=niel_per_neutron,
        niel_per_neutron_kev=niel_per_neutron * 1.0e3,
        average_displacements=avg_displacements,
        histogram=histogram,
    )


def write_histogram_csv(histogram: Iterable[HistogramBin], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("energy_low_mev,energy_high_mev,count\n")
        for bin_entry in histogram:
            fh.write(
                f"{bin_entry.energy_low_mev:.8f},{bin_entry.energy_high_mev:.8f},{bin_entry.count}\n"
            )


def estimate_neutron_count(histogram: Iterable[HistogramBin]) -> int:
    return sum(entry.count for entry in histogram)


def write_summary(result: SimulationResult, output_path: Path) -> None:
    neutron_total = estimate_neutron_count(result.histogram)
    summary_lines = [
        "Simulation summary:",
        f"  Neutrons requested: {result.n_neutrons}",
        f"  Neutrons recorded in histogram: {neutron_total}",
        f"  Incident neutron energy: {result.neutron_energy_mev:.3f} MeV",
        f"  Maximum silicon recoil energy: {result.recoil_max_energy_mev:.6f} MeV",
        f"  Average silicon recoil energy: {result.average_recoil_energy_mev:.6f} MeV",
        f"  Average damage energy per event: {result.average_damage_energy_mev:.6e} MeV",
        f"  Average displacement count per event: {result.average_displacements:.6f}",
        f"  Total damage energy (all events): {result.total_damage_energy_mev:.6f} MeV",
        f"  NIEL per incident neutron: {result.niel_per_neutron_mev:.6e} MeV",
        f"  NIEL per incident neutron (keV): {result.niel_per_neutron_kev:.6e} keV",
    ]
    output_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def plot_histogram(
    histogram: Iterable[HistogramBin],
    output_path: Path,
    *,
    neutron_energy_mev: float,
) -> None:
    plt = _get_matplotlib_pyplot()
    bins = list(histogram)
    if not bins:
        raise ValueError("Histogram is empty; nothing to plot")

    bin_widths = [entry.energy_high_mev - entry.energy_low_mev for entry in bins]
    left_edges = [entry.energy_low_mev for entry in bins]
    counts = [entry.count for entry in bins]

    plt.figure(figsize=(8, 4.5))
    plt.bar(left_edges, counts, width=bin_widths, align="edge", edgecolor="black")
    plt.xlabel("Silicon recoil energy [MeV]")
    plt.ylabel("Counts per bin")
    plt.title(f"PKA energy spectrum for {neutron_energy_mev:.2f} MeV neutrons")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _get_matplotlib_pyplot():
    if not hasattr(_get_matplotlib_pyplot, "_cached"):
        if importlib.util.find_spec("matplotlib") is None:
            raise ModuleNotFoundError(
                "matplotlib is required for plotting. Install it or run with --skip-plot."
            )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _get_matplotlib_pyplot._cached = plt
    return _get_matplotlib_pyplot._cached


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate primary knock-on atom (PKA) recoil energies in silicon and "
            "produce a spectrum, summary statistics, and a NIEL estimate."
        )
    )
    parser.add_argument("--neutrons", type=int, default=100_000, help="Number of incident neutrons to simulate")
    parser.add_argument("--energy", type=float, default=1.0, help="Incident neutron energy in MeV")
    parser.add_argument(
        "--displacement-energy",
        type=float,
        default=25.0,
        help="Silicon displacement energy in eV (used in Kinchin-Pease damage model)",
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for the recoil energy histogram")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where output files will be written",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generation of the histogram plot if matplotlib is unavailable",
    )
    args = parser.parse_args()

    result = run_simulation(
        n_neutrons=args.neutrons,
        neutron_energy_mev=args.energy,
        displacement_energy_ev=args.displacement_energy,
        n_bins=args.bins,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "si_pka_energy_spectrum.csv"
    summary_path = args.output_dir / "si_pka_summary.txt"
    plot_path = args.output_dir / "si_pka_energy_spectrum.png"

    write_histogram_csv(result.histogram, csv_path)
    write_summary(result, summary_path)

    plot_message: str
    if not args.skip_plot:
        try:
            plot_histogram(result.histogram, plot_path, neutron_energy_mev=result.neutron_energy_mev)
            plot_message = f"  Plot: {plot_path}"
        except ModuleNotFoundError as exc:
            plot_message = f"  Plot skipped ({exc})"
    else:
        plot_message = "  Plot skipped (--skip-plot)"

    neutron_count = estimate_neutron_count(result.histogram)
    print(
        "\n".join(
            [
                "Simulation complete.",
                f"  Neutrons simulated: {neutron_count}",
                f"  Histogram CSV: {csv_path}",
                f"  Summary: {summary_path}",
                plot_message,
            ]
        )
    )


if __name__ == "__main__":
    main()
