"""
Taillard Benchmark Instance Parser & Downloader

Parses Taillard flow shop instances from the standard format used in:
    E. Taillard (1993). "Benchmarks for Basic Scheduling Problems"
    European Journal of Operational Research, 64(2):278-285.

Instance source: https://github.com/chneau/go-taillard
    Original: http://mistic.heig-vd.ch/taillard/

Format:
    Line 1: Header text (ignored)
    Line 2: n_jobs  n_machines  seed  upper_bound  lower_bound
    Line 3: "processing times :" (ignored)
    Lines 4+: m rows of n processing times each (machine × jobs)

The 120 instances span 12 classes:
    20×5, 20×10, 20×20, 50×5, 50×10, 50×20,
    100×5, 100×10, 100×20, 200×10, 200×20, 500×20
    (10 instances per class)
"""

from __future__ import annotations
import os
import urllib.request
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# Base URL for raw Taillard instances
TAILLARD_BASE_URL = (
    "https://raw.githubusercontent.com/chneau/go-taillard/"
    "master/pfsp/instances"
)

# Local cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "taillard"


@dataclass
class TaillardInstanceInfo:
    """Metadata for a Taillard benchmark instance."""
    name: str           # e.g. "tai20_5_0"
    n_jobs: int
    n_machines: int
    seed: int
    upper_bound: int    # Best known solution (from the file header)
    lower_bound: int
    instance_index: int  # 0-9 within its class


# All 120 Taillard instances organized by (n_jobs, n_machines)
TAILLARD_CLASSES: list[tuple[int, int]] = [
    (20, 5), (20, 10), (20, 20),
    (50, 5), (50, 10), (50, 20),
    (100, 5), (100, 10), (100, 20),
    (200, 10), (200, 20),
    (500, 20),
]

# Best known upper bounds for the first 40 instances (tai20_5 through tai50_10)
# These are from the instance file headers in the go-taillard repository.
# Instances where UB == LB are proven optimal.
BEST_KNOWN_UPPER_BOUNDS: dict[str, int] = {
    # 20×5 (ta001–ta010)
    "tai20_5_0": 1278, "tai20_5_1": 1359, "tai20_5_2": 1081,
    "tai20_5_3": 1293, "tai20_5_4": 1236, "tai20_5_5": 1195,
    "tai20_5_6": 1239, "tai20_5_7": 1206, "tai20_5_8": 1230,
    "tai20_5_9": 1108,
    # 20×10 (ta011–ta020)
    "tai20_10_0": 1582, "tai20_10_1": 1659, "tai20_10_2": 1496,
    "tai20_10_3": 1378, "tai20_10_4": 1419, "tai20_10_5": 1397,
    "tai20_10_6": 1484, "tai20_10_7": 1538, "tai20_10_8": 1593,
    "tai20_10_9": 1591,
    # 20×20 (ta021–ta030)
    "tai20_20_0": 2297, "tai20_20_1": 2100, "tai20_20_2": 2326,
    "tai20_20_3": 2223, "tai20_20_4": 2291, "tai20_20_5": 2226,
    "tai20_20_6": 2273, "tai20_20_7": 2200, "tai20_20_8": 2237,
    "tai20_20_9": 2178,
    # 50×5 (ta031–ta040)
    "tai50_5_0": 2724, "tai50_5_1": 2834, "tai50_5_2": 2621,
    "tai50_5_3": 2751, "tai50_5_4": 2863, "tai50_5_5": 2829,
    "tai50_5_6": 2725, "tai50_5_7": 2683, "tai50_5_8": 2552,
    "tai50_5_9": 2782,
    # 50×10 (ta041–ta050)
    "tai50_10_0": 3025, "tai50_10_1": 2892, "tai50_10_2": 2864,
    "tai50_10_3": 3064, "tai50_10_4": 2986, "tai50_10_5": 3006,
    "tai50_10_6": 3107, "tai50_10_7": 3039, "tai50_10_8": 2902,
    "tai50_10_9": 3091,
}


def get_instance_name(n_jobs: int, n_machines: int, index: int) -> str:
    """Generate Taillard instance name, e.g. 'tai20_5_0'."""
    return f"tai{n_jobs}_{n_machines}_{index}"


def get_instance_url(name: str) -> str:
    """Get download URL for a Taillard instance."""
    return f"{TAILLARD_BASE_URL}/{name}.fsp"


def download_instance(name: str, cache: bool = True) -> str:
    """
    Download a Taillard instance and return the file content as a string.

    Args:
        name: Instance name, e.g. "tai20_5_0"
        cache: If True, cache downloaded files locally.

    Returns:
        File content as string.
    """
    if cache:
        cache_path = CACHE_DIR / f"{name}.fsp"
        if cache_path.exists():
            return cache_path.read_text()

    url = get_instance_url(name)
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        raise ConnectionError(
            f"Failed to download Taillard instance '{name}' from {url}: {e}"
        )

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{name}.fsp").write_text(content)

    return content


def parse_taillard_string(content: str) -> tuple[np.ndarray, TaillardInstanceInfo]:
    """
    Parse a Taillard instance from its string content.

    Returns:
        Tuple of (processing_times array of shape (m, n), instance info).
    """
    lines = [line.strip() for line in content.strip().split('\n')
             if line.strip()]

    # Line 0: header text (skip)
    # Line 1: n  m  seed  upper_bound  lower_bound
    header_values = lines[1].split()
    n_jobs = int(header_values[0])
    n_machines = int(header_values[1])
    seed = int(header_values[2])
    upper_bound = int(header_values[3])
    lower_bound = int(header_values[4])

    # Line 2: "processing times :" (skip)
    # Lines 3+: m rows of processing times
    processing_times = np.zeros((n_machines, n_jobs), dtype=int)

    data_start = 3  # After header, values line, and "processing times" line
    for i in range(n_machines):
        values = list(map(int, lines[data_start + i].split()))
        processing_times[i] = values

    info = TaillardInstanceInfo(
        name="",  # Will be filled by caller
        n_jobs=n_jobs,
        n_machines=n_machines,
        seed=seed,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        instance_index=0,
    )

    return processing_times, info


def load_taillard_instance(
    name: str,
    cache: bool = True,
) -> tuple[np.ndarray, TaillardInstanceInfo]:
    """
    Download and parse a Taillard instance by name.

    Args:
        name: Instance name, e.g. "tai20_5_0"
        cache: Cache downloaded files locally.

    Returns:
        Tuple of (processing_times array, instance info).

    Example:
        >>> p, info = load_taillard_instance("tai20_5_0")
        >>> print(p.shape)  # (5, 20)
        >>> print(info.upper_bound)  # 1278
    """
    content = download_instance(name, cache=cache)
    processing_times, info = parse_taillard_string(content)

    # Parse index from name
    parts = name.split('_')
    info.name = name
    info.instance_index = int(parts[-1])

    return processing_times, info


def load_taillard_class(
    n_jobs: int,
    n_machines: int,
    cache: bool = True,
) -> list[tuple[np.ndarray, TaillardInstanceInfo]]:
    """
    Load all 10 instances of a given Taillard class.

    Args:
        n_jobs: Number of jobs (20, 50, 100, 200, or 500).
        n_machines: Number of machines.
        cache: Cache downloaded files locally.

    Returns:
        List of 10 (processing_times, info) tuples.
    """
    instances = []
    for idx in range(10):
        name = get_instance_name(n_jobs, n_machines, idx)
        instances.append(load_taillard_instance(name, cache=cache))
    return instances


def parse_taillard_file(filepath: str) -> tuple[np.ndarray, TaillardInstanceInfo]:
    """
    Parse a Taillard instance from a local file.

    Args:
        filepath: Path to a .fsp file.

    Returns:
        Tuple of (processing_times array, instance info).
    """
    content = Path(filepath).read_text()
    processing_times, info = parse_taillard_string(content)
    info.name = Path(filepath).stem
    return processing_times, info


if __name__ == "__main__":
    # Demo: download and inspect tai20_5_0
    print("Downloading tai20_5_0...")
    p, info = load_taillard_instance("tai20_5_0")
    print(f"Name:        {info.name}")
    print(f"Size:        {info.n_jobs} jobs × {info.n_machines} machines")
    print(f"Upper bound: {info.upper_bound}")
    print(f"Lower bound: {info.lower_bound}")
    print(f"Shape:       {p.shape}")
    print(f"First row:   {p[0]}")
