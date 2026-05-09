"""
extract_results.py — parse Vitis HLS synthesis report for the NTT kernel.

Usage (run from kernel/ directory after vitis_hls -f run_hls.tcl):
    python extract_results.py

Outputs:
    data/csynth_loop_info.csv      — pipeline II, depth, latency per loop
    data/csynth_resource_usage.csv — LUT/FF/BRAM/DSP per module + totals

Requires: pandas  (pip install pandas)
"""

import os
import sys
import pandas as pd
from csynthparse import CsynthParser

SOL_PATH = os.path.join(os.path.dirname(__file__), 'ntt_project', 'solution1')
DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')

def main():
    if not os.path.isdir(SOL_PATH):
        print(f"ERROR: solution directory not found: {SOL_PATH}")
        print("Run 'vitis_hls -f run_hls.tcl' first to generate the synthesis report.")
        sys.exit(1)

    parser = CsynthParser(sol_path=SOL_PATH)

    # ── Pipeline / latency ──────────────────────────────────────────────────
    parser.get_loop_pipeline_info()
    print("=" * 70)
    print("Loop Pipeline Info")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(parser.loop_df.to_string())

    # ── Resource utilisation ────────────────────────────────────────────────
    parser.get_resources()
    print("\n" + "=" * 70)
    print("Resource Utilisation")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(parser.res_df.to_string())

    # ── Butterfly loop highlight ────────────────────────────────────────────
    bfly_rows = parser.loop_df[parser.loop_df.index.str.contains('butterfly|inner|bfly|pipeline', case=False, na=False)]
    if not bfly_rows.empty:
        print("\n── Butterfly / inner loop ──")
        print(bfly_rows.to_string())

    # ── Save CSVs ───────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    loop_csv = os.path.join(DATA_DIR, 'csynth_loop_info.csv')
    res_csv  = os.path.join(DATA_DIR, 'csynth_resource_usage.csv')
    parser.loop_df.to_csv(loop_csv, index=True)
    parser.res_df.to_csv(res_csv,   index=True)
    print(f"\nSaved: {loop_csv}")
    print(f"Saved: {res_csv}")

    # ── Quick README table ──────────────────────────────────────────────────
    if 'Total' in parser.res_df.index and 'Available' in parser.res_df.index:
        total = parser.res_df.loc['Total']
        avail = parser.res_df.loc['Available']
        print("\n── Copy-paste into README.md ──")
        print("| Resource | Used | Available | Utilization |")
        print("|----------|------|-----------|-------------|")
        for res in total.index:
            u = total[res]
            a = avail[res]
            try:
                pct = f"{int(u)/int(a)*100:.1f}%"
            except (ZeroDivisionError, TypeError, ValueError):
                pct = "N/A"
            print(f"| {res:<8} | {u:<4} | {a:<9} | {pct:<11} |")

if __name__ == "__main__":
    main()
