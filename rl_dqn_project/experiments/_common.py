"""Shared CLI + table helpers for the experiment scripts."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets `python experiments/xyz.py` import agents/, envs/, common/
    sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"


def use_quick_dirs() -> None:
    """--quick output goes to results/quick/ so it never overwrites the real figures/tables."""
    global FIG_DIR, TABLE_DIR
    FIG_DIR = ROOT / "results" / "quick" / "figures"
    TABLE_DIR = ROOT / "results" / "quick" / "tables"


def seed_list(args) -> list[int]:
    return list(range(args.seed_start, args.seed_start + (2 if args.quick else args.seeds)))


def figpath(name: str) -> Path:
    return FIG_DIR / name


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds, default 5")
    p.add_argument("--seed-start", type=int, default=0,
                   help="first seed (runs seed-start .. seed-start+seeds-1); lets several machines split the seeds")
    p.add_argument("--workers", type=int, default=None, help="parallel processes (default: cores - 1)")
    p.add_argument("--force", action="store_true", help="recompute runs even if cached")
    p.add_argument("--plot-only", action="store_true", help="only rebuild figures/tables from cached runs")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: tiny budgets, 2 seeds, written to a separate *_quick experiment")
    return p.parse_args()


def fmt(cell, digits=1) -> str:
    """(mean, ci_half, n) -> 'm ± h' ."""
    m, h, n = cell
    if n == 0 or m != m:  # NaN
        return "--"
    return f"{m:.{digits}f} ± {h:.{digits}f}"


def write_table(name: str, header: list[str], rows: list[list[str]], caption: str) -> None:
    """Write CSV (for humans/Excel) and a LaTeX booktabs table (\\input by the report)."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    def tex(s: str) -> str:
        for a, b in [("_", r"\_"), ("%", r"\%"), ("±", r"$\pm$"), ("ε", r"$\varepsilon$"), ("τ", r"$\tau$"),
                     ("≤", r"$\le$"), ("≥", r"$\ge$"), ("−", "-"), ("†", r"$^\dagger$"), ("₀", r"$_0$"),
                     ("|x|", r"$|x|$")]:
            s = s.replace(a, b)
        return s

    wide = len(header) > 5  # wide tables span both columns of the two-column report
    env, width = ("table*", r"\textwidth") if wide else ("table", r"\columnwidth")
    lines = [rf"\begin{{{env}}}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
             rf"\resizebox{{{width}}}{{!}}{{%",
             r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", rf"\end{{{env}}}"]
    (TABLE_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote tables/{name}.csv / .tex")
    # also echo to the terminal
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header, *rows]:
        print("   " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
