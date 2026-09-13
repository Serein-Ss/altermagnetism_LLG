"""Place the numerical reproduction beside the cited open-access figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    generated = Image.open(
        ROOT / "assets" / "literature_validation" / "spinwave_reproduction.png"
    )
    reference = Image.open(
        ROOT / "assets" / "literature_validation" / "reference" /
        "gomonay_2024_fig2.png"
    )
    output = (ROOT / "assets" / "literature_validation" /
              "spinwave_paper_side_by_side.png")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    axes[0].imshow(generated)
    axes[0].set_title("a  This work: literature Hamiltonian + LLG trajectory",
                      loc="left", fontsize=12, fontweight="bold")
    axes[1].imshow(reference)
    axes[1].set_title("b  Published Fig. 2 (CC BY 4.0)", loc="left",
                      fontsize=12, fontweight="bold")
    for ax in axes:
        ax.axis("off")
    fig.text(
        0.02, 0.015,
        "Comparison target: diagonal-direction magnon splitting, nodal-direction "
        "degeneracy, and opposite circular branches. LLG frequency errors: 0.87% / 0.67%.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.92, bottom=0.08, wspace=0.025)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
