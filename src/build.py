"""Convert raw data to the unified format.

Usage:
    python -m src.build leakdb_net1 [--limit N]
    python -m src.build leakdb_hanoi [--limit N]
    python -m src.build ltown
    python -m src.build sim --inp dataset/training/raw/networks/Net3.inp --n 20
    python -m src.build topology --names leakdb_net1 sim_net3   (refresh nodes/links in network.json)
"""
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["leakdb_net1", "leakdb_hanoi", "ltown", "sim", "topology"])
    ap.add_argument("--limit", type=int, default=None, help="LeakDB: only the first N scenarios")
    ap.add_argument("--inp", help="sim: path to the .inp file")
    ap.add_argument("--n", type=int, default=20, help="sim: number of scenarios")
    ap.add_argument("--name", default=None, help="sim: dataset name (default sim_<inp stem>)")
    ap.add_argument("--days", type=float, default=7, help="sim: scenario length in days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", nargs="+", default=[], help="topology: processed datasets to refresh")
    args = ap.parse_args()

    if args.dataset == "topology":
        from .schema import refresh_topology
        for name in args.names:
            print("refreshed", refresh_topology(name))
    elif args.dataset.startswith("leakdb_"):
        from .loaders import leakdb
        leakdb.build(args.dataset.split("_")[1], args.limit)
    elif args.dataset == "ltown":
        from .loaders import ltown
        ltown.build()
    else:
        from .loaders import wntr_sim
        wntr_sim.simulate(args.inp, args.n, dataset=args.name, duration_days=args.days, seed=args.seed)
