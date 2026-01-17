BENCH="../Benchmark/l40s1e-6PDLPresults.csv"
LP="lp_results.csv"
OUT="lp_comparison.csv"

echo "instance,lp_obj,bench_obj,abs_diff,rel_diff" > "$OUT"

awk -F',' -v out="$OUT" '
NR==FNR {
    if (NR > 1) {
        bench[$1] = $3   # instance -> objective
    }
    next
}
NR > 1 {
    inst = $1
    lp = $2

    # normalize instance name
    gsub("instance_", "relaxed_", inst)

    if (!(inst in bench)) {
        printf "%s,%.15g,NA,NA,NA\n", inst, lp >> out
        next
    }

    b = bench[inst]
    abs = (lp > b) ? lp - b : b - lp
    rel = (b != 0) ? abs / ((b < 0) ? -b : b) : 0

    printf "%s,%.15g,%.15g,%.15g,%.15g\n", inst, lp, b, abs, rel >> out
}
' "$BENCH" "$LP"

echo "Comparison written to $OUT"

