"""Shared helpers for experiment-result inspection and plotting scripts."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"

BENCHMARK_CATEGORIES = {
    "C-DTLZs": ["C1DTLZ1", "C1DTLZ3", "C2DTLZ2", "C3DTLZ4"],
    "DC-DTLZs": [
        "DC1DTLZ1",
        "DC1DTLZ3",
        "DC2DTLZ1",
        "DC2DTLZ3",
        "DC3DTLZ1",
        "DC3DTLZ3",
    ],
    "DAS-CMOP": [f"DASCMOP{i}" for i in range(1, 10)],
    "LIR-CMOP": [f"LIRCMOP{i}" for i in range(1, 15)],
}

MAIN_ALGORITHMS = ["DSOCOL", "APSEA", "CMOCSO", "IMCMOEAD"]
ABLATION_ALGORITHMS = ["DSOCOL", "DSOCOL1", "DSOCOL3", "DSOCOL4"]
ALGORITHM_LABELS = {
    "DSOCOL": "DSOCOL (Ours)",
    "APSEA": "APSEA",
    "CMOCSO": "CMOCSO",
    "IMCMOEAD": "IM-C-MOEA/D",
    "DSOCOL1": "w/o NGSS",
    "DSOCOL3": "w/o COL",
    "DSOCOL4": "w/o Trend",
}

PROBLEMS_INFO = [
    {"name": "C1DTLZ1", "category": "C-DTLZs", "objs": 3},
    {"name": "DC1DTLZ1", "category": "DC-DTLZs", "objs": 3},
    {"name": "DASCMOP1", "category": "DAS-CMOP", "objs": 2},
    {"name": "DASCMOP7", "category": "DAS-CMOP", "objs": 3},
    {"name": "LIRCMOP1", "category": "LIR-CMOP", "objs": 2},
    {"name": "LIRCMOP13", "category": "LIR-CMOP", "objs": 3},
]


def resolve_path(value: str | Path | None, default: Path) -> Path:
    """Resolve a CLI path while allowing both absolute and project-relative paths."""
    if value is None:
        return default
    return Path(value).expanduser()


def classify_problem(problem_name: str) -> str:
    """Return the Benchmark category corresponding to a problem name."""
    name = problem_name.upper()
    if name.startswith("DC"):
        return "DC-DTLZs"
    if name.startswith("C") and "DTLZ" in name:
        return "C-DTLZs"
    if name.startswith("DAS"):
        return "DAS-CMOP"
    if name.startswith("LIR"):
        return "LIR-CMOP"
    return "Other-Benchmarks"


def ensure_category_structure(results_dir: Path) -> None:
    """Create category folders and move legacy problem folders into them."""
    if not results_dir.exists():
        return

    for algorithm_dir in (
        item
        for item in results_dir.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ):
        for category in BENCHMARK_CATEGORIES:
            (algorithm_dir / category).mkdir(exist_ok=True)

        for item in list(algorithm_dir.iterdir()):
            if (
                item.is_dir()
                and item.name not in BENCHMARK_CATEGORIES
                and not item.name.startswith(".")
            ):
                category = classify_problem(item.name)
                target = algorithm_dir / category / item.name
                if not target.exists():
                    item.rename(target)
                    print(
                        f"[整理] 移动 {algorithm_dir.name}/{item.name} "
                        f"-> {algorithm_dir.name}/{category}/{item.name}"
                    )


def discover_algorithms(results_dir: Path) -> list[str]:
    """Discover top-level algorithm directories that contain NPZ result files."""
    if not results_dir.exists():
        return []
    algorithms = []
    for item in sorted(results_dir.iterdir()):
        if item.is_dir() and not item.name.startswith(".") and any(item.glob("**/*.npz")):
            algorithms.append(item.name)
    return algorithms


def infer_stat_algorithms(results_dir: Path) -> list[str]:
    """Choose the historical four-algorithm profile for a statistics run."""
    found = set(discover_algorithms(results_dir))
    if set(ABLATION_ALGORITHMS).issubset(found):
        return ABLATION_ALGORITHMS.copy()
    main = [algorithm for algorithm in MAIN_ALGORITHMS if algorithm in found]
    return main or sorted(found)


def label_for_algorithm(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm, algorithm)
