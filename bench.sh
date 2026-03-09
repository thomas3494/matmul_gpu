#!/bin/sh

if [ "$#" -ne 1 ]; then
    printf 'Usage: %s OUTDIR\n' "$0" >&2
    printf '\tOUTDIR: directory to store result\n' >&2
    exit 1
fi

outdir="$1"

mkdir -p "$outdir"

bench()
{
    name="$1"
    n="$2"
    make bin/"$name"

    echo "gflops,stddev" > "${outdir}/${name}_${n}.csv"

    ./bin/"$name" "$n" "$n" "$n" >> "${outdir}/${name}_${n}.csv"
}

bench cublas    16384
bench L2        16384
bench simple    16384
bench one-block 16384
bench L2-2D     16384

bench cublas    8192
bench L2        8192
bench simple    8192
bench one-block 8192
bench L2-2D     8192
