#!/bin/sh

if [ "$#" -ne 2 ]; then
    printf 'Usage: %s ITER OUTDIR\n' "$0" >&2
    printf '\tITER:   number of times to repeat the experiment\n' >&2
    printf '\tOUTDIR: directory to store result\n' >&2
    exit 1
fi

iter="$1"
outdir="$2"

mkdir -p "$outdir"

bench()
{
    name="$1"
    n="$2"
    make bin/"$name"

    echo "gflops,stddev" > "${outdir}/${name}_${n}.csv"

    # Warmup
    i=1
    while [ $i -le 3 ]
    do
        ./bin/"$name" "$n" "$n" "$n"
        i=$(( i + 1 ))
    done

    {
        i=1
        while [ $i -le "$iter" ]
        do
            ./bin/"$name" "$n" "$n" "$n"
            i=$(( i + 1 ))
        done
    } | awk '{
               b = a + ($1 - a) / NR;
               q += ($1 - a) * ($1 - b);
               a = b;
             } END {
               printf "%f,%f", a, sqrt(q / (NR - 1));
             }' >> "${outdir}/${name}_${n}.csv"
}

#bench cublas    16384
#bench L2        16384
#bench simple    16384
bench one-block 16384

#bench cublas 8192
#bench L2     8192
#bench simple 8192
bench one-block 8192
