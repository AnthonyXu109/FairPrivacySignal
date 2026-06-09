from pathlib import Path

from fairprivacysignal.architecture_diagram import plot_architecture_diagram


def test_architecture_diagram_renders(tmp_path: Path) -> None:
    out_path = tmp_path / "architecture_diagram.png"

    plot_architecture_diagram(out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
