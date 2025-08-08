#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

PATTERN = re.compile(
    r"Epoch:\s*(\d+),\s*step:\s*(\d+),\s*loss:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),\s*avg loss:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

def parse_log(path):
    iters, epochs, steps, losses, avg_losses = [], [], [], [], []
    i = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            epoch, step, loss, avg_loss = m.groups()
            epochs.append(int(epoch))
            steps.append(int(step))
            losses.append(float(loss))
            avg_losses.append(float(avg_loss))
            iters.append(i)
            i += 1
    if not losses:
        raise ValueError("No matching lines found. Check the log format or regex.")
    return iters, epochs, steps, losses, avg_losses

def main():
    ap = argparse.ArgumentParser(description="Plot loss and avg loss from training log.")
    ap.add_argument("logfile", type=Path, help="Path to the training .txt log")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output image file (e.g., plot.png). If omitted, shows the plot.")
    args = ap.parse_args()

    iters, epochs, steps, losses, avg_losses = parse_log(args.logfile)

    plt.figure()
    plt.semilogy(iters, losses, label="loss")
    plt.semilogy(iters, avg_losses, label="avg loss")
    plt.xlabel("Iteration (log order)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
