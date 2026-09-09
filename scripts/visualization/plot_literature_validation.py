"""Plot the literature spin-wave reproduction. Static output is PNG only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from altermagnetism_LLG.scripts.core.project_paths import asset_path


ROOT = Path(__file__).resolve().parents[2]


def analytic_spinwave_frequency(kx: np.ndarray, ky: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Supplementary Eq. (S.9), evaluated without importing the solver."""
    j1_mev, j2_mev, j_tilde_mev = 11.1, 1.88, 0.8
    a0_m, mu_s = 0.448e-9, 9.2740100783e-24
    gamma, mev_to_j = 1.76085963023e11, 1.602176634e-22
    eta, epsilon = j2_mev/j1_mev, j_tilde_mev/j1_mev
    ax, ay = a0_m*kx, a0_m*ky
    a_k = np.cos(ax/2)*np.cos(ay/2)
    b_k = 1 + eta*(np.sin(ax/2)**2 + np.sin(ay/2)**2)
    c_k = epsilon*np.sin(ax)*np.sin(ay)
    base = np.sqrt(np.maximum(b_k*b_k - a_k*a_k, 0.0))
    omega0 = 4*gamma*(j1_mev*mev_to_j)/mu_s
    scale = 2*np.pi*1e12
    return omega0*(base-c_k)/scale, omega0*(base+c_k)/scale


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=ROOT / "data" / "literature" / "literature_validation" /
                   "spinwave_trajectory.npz")
    p.add_argument("--output", type=Path,
                   default=ROOT / "assets" / "literature" / "literature_validation" /
                   "spinwave_reproduction.png")
    args = p.parse_args()
    data = np.load(args.input, allow_pickle=False)
    qa = np.linspace(0, np.pi, 400)
    kval = qa / 0.448e-9
    zeros = np.zeros_like(kval)
    diag = analytic_spinwave_frequency(kval, kval)
    node = analytic_spinwave_frequency(kval, zeros)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), constrained_layout=True)
    ax = axes[0]
    ax.plot(qa, diag[0], color="#2671b8", label=r"$[110],\ \omega_-$")
    ax.plot(qa, diag[1], color="#d64b3c", label=r"$[110],\ \omega_+$")
    ax.plot(qa, node[0], color="#333333", linestyle="--",
            label=r"$[100]$ (degenerate)")
    ax.set(xlabel=r"Reduced wave vector $k a_0$", ylabel="Frequency (THz)",
           title="Published d-wave magnon splitting")
    ax.legend(frameon=False, fontsize=9)

    angle = np.linspace(0, 2*np.pi, 721)
    q0 = 0.5*np.pi
    angular = analytic_spinwave_frequency(
        q0*np.cos(angle)/0.448e-9,
        q0*np.sin(angle)/0.448e-9,
    )
    splitting = angular[1] - angular[0]
    ax = axes[1]
    ax.plot(np.degrees(angle), splitting/np.max(np.abs(splitting)),
            color="#6f4c9b", linewidth=1.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set(xlabel="Crystal angle (degree)",
           ylabel=r"Normalized $f_+-f_-$",
           title="d-wave angular fingerprint")

    freq = data["fft_frequency_hz"] / 1e12
    power = data["fft_power"]
    power = power / max(power.max(), np.finfo(float).tiny)
    expected = data["expected_frequency_hz"] / 1e12
    measured = data["measured_frequency_hz"] / 1e12
    ax = axes[2]
    keep = np.abs(freq) < 1.5 * expected.max()
    ax.plot(freq[keep], power[keep], color="#5b4b8a", linewidth=1.1)
    signed_expected = np.array([-expected[0], expected[1]])
    signed_measured = np.array([-measured[0], measured[1]])
    for i, value in enumerate(signed_expected):
        ax.axvline(value, color=("#2671b8", "#d64b3c")[i], linestyle="--",
                   label=(r"Eq. (S.9) $f_-$", r"Eq. (S.9) $f_+$")[i])
        order = np.argsort(freq)
        ax.plot(signed_measured[i], np.interp(signed_measured[i], freq[order], power[order]), "o",
                color=("#2671b8", "#d64b3c")[i], markersize=5)
    ax.set(xlabel="Frequency (THz)", ylabel="Normalized spectral power",
           title="Signed circular spectrum from the LLG trajectory")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset_path(args.output), dpi=220, facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
