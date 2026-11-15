"""Download raw datasets into dataset/training/raw/. Safe to re-run: existing files are skipped.

Usage:
    python -m src.download leakdb_net1 --n 100
    python -m src.download leakdb_hanoi --n 100
    python -m src.download ltown
    python -m src.download networks
"""
import argparse
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "dataset" / "training" / "raw"

LEAKDB_URL = "https://filedn.com/lumBFq2P9S74PNoLPWtzxG4/EPyT-Flow/LeakDB-Original/{net}_CMH/Scenario-{i}.zip"
ZENODO_URL = "https://zenodo.org/records/4017659/files/{name}?download=1"
LTOWN_FILES = [
    "L-TOWN.inp",
    "README.txt",
    "dataset_configuration.yaml",
    "2018_Fixed_Leakages_Report.txt",
    "2018_Leakages.csv",
    "2019_Leakages.csv",
    "2018_SCADA_Pressures.csv",
    "2018_SCADA_Flows.csv",
    "2018_SCADA_Levels.csv",
    "2018_SCADA_Demands.csv",
    "2019_SCADA_Pressures.csv",
    "2019_SCADA_Flows.csv",
    "2019_SCADA_Levels.csv",
    "2019_SCADA_Demands.csv",
]
NETWORK_URLS = {
    "Net3.inp": "https://raw.githubusercontent.com/USEPA/WNTR/main/examples/networks/Net3.inp",
    "Net1.inp": "https://raw.githubusercontent.com/USEPA/WNTR/main/examples/networks/Net1.inp",
}


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waterboard-data-pipeline"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    return dest


def download_leakdb(net: str, n: int, threads: int = 8) -> None:
    out = RAW_DIR / f"leakdb_{net.lower()}"

    def one(i: int) -> str:
        folder = out / f"Scenario-{i}"
        if (folder / "Labels.csv").exists():
            return f"Scenario-{i}: present"
        z = fetch(LEAKDB_URL.format(net=net, i=i), out / f"Scenario-{i}.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        z.unlink()  # the extracted folder is the raw copy; the zip only doubles disk use
        return f"Scenario-{i}: downloaded"

    with ThreadPoolExecutor(threads) as pool:
        for msg in pool.map(one, range(1, n + 1)):
            print(msg, flush=True)


def download_ltown() -> None:
    for name in LTOWN_FILES:
        fetch(ZENODO_URL.format(name=name), RAW_DIR / "ltown" / name)
        print(name, flush=True)


def download_networks() -> None:
    for name, url in NETWORK_URLS.items():
        fetch(url, RAW_DIR / "networks" / name)
        print(name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["leakdb_net1", "leakdb_hanoi", "ltown", "networks"])
    ap.add_argument("--n", type=int, default=100, help="number of LeakDB scenarios (1..n)")
    ap.add_argument("--threads", type=int, default=8, help="parallel LeakDB downloads")
    args = ap.parse_args()
    if args.dataset == "leakdb_net1":
        download_leakdb("Net1", args.n, args.threads)
    elif args.dataset == "leakdb_hanoi":
        download_leakdb("Hanoi", args.n, args.threads)
    elif args.dataset == "ltown":
        download_ltown()
    else:
        download_networks()
