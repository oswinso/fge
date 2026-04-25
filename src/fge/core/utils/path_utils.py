import pathlib


def get_root_dir() -> pathlib.Path:
    path = pathlib.Path(__file__).parent.parent.parent.parent.parent
    assert (path / "src").exists()
    return path

def get_scripts_dir() -> pathlib.Path:
    return get_root_dir() / "scripts"


def get_runs_dir():
    return get_root_dir() / "runs"


def get_benchmark_dir():
    return get_root_dir() / "benchmarks"


def paper_mat_dir():
    return get_root_dir() / "paper_material"


def paper_plot_dir():
    return paper_mat_dir() / "paper_plots"
