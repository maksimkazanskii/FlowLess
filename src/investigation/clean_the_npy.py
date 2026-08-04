#!/usr/bin/env python3

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=Path,
    default=Path("data/results/investigation/mnist"),
)

args = parser.parse_args()

count = 0

for npy_file in args.root.rglob("*.npy"):
    npy_file.unlink()
    print(f"Removed {npy_file}")
    count += 1

print(f"\nRemoved {count} .npy files.")