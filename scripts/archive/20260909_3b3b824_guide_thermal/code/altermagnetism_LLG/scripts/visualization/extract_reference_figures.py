"""Crop cited figures from the rendered open-access article pages."""
from pathlib import Path
import subprocess
import tempfile
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "literature_validation" / "reference"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data" / "literature_validation" / "Gomonay_2024_main.pdf"
    with tempfile.TemporaryDirectory(prefix="gomonay_fig2_") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run([
            "pdftoppm", "-f", "3", "-l", "3", "-r", "180", "-png",
            str(source), str(prefix),
        ], check=True)
        page3 = Image.open(Path(tmp) / "page-3.png")
        # Main article p.3, Fig. 2 visual only; caption is stored separately.
        figure2 = page3.crop((545, 88, 1415, 700))
        figure2.save(OUT_DIR / "gomonay_2024_fig2.png")
    print((OUT_DIR / "gomonay_2024_fig2.png").resolve())


if __name__ == "__main__":
    main()
