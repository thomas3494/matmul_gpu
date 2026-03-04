#!/bin/sh

make bin/"$1"

ncu --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section Occupancy --section SpeedOfLight --section WarpStateStats --section SourceCounters --import-source yes -f -o ncu_profile ./bin/"$1" 8192 8192 8192
