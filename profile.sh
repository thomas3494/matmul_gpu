#!/bin/sh

make bin/"$1"_profile

ncu --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section Occupancy --section SpeedOfLight --section WarpStateStats --section SourceCounters --import-source yes -f -o ncu_profile ./bin/"$1"_profile 16384 16384 16384
