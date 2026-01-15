#!/usr/bin/env bash
set -e

BENCH="../Benchmark/l40s1e-6PDLPresults.csv"
LP="lp_results.csv"
OUT="lp_comparison.csv"

echo "instance,lp_obj,bench_obj,abs_diff,rel_diff" > $OUT

awk -F',' '
NR==FNR {
    if (NR > 1) {
        bench[$1] = $3
    }
    next
}
NR > 1 {
    inst = $1
    lp = $2
    b = bench[inst]
    abs = (lp > b) ? lp - b : b - lp
    rel = (b != 0) ? abs / ( (b<0)?-b:b ) : 0
    printf "%s,%.15g,%.15g,%.15g,%.15g\n", inst, lp, b, abs, rel
}
' "$BENCH" "$LP" >> "$OUT"

echo "Comparison written to $OUT"

