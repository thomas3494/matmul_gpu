#!/bin/sh

make bin/"$1"
ncu --metrics dram__bytes_read.sum bin/"$1" 8192 8192 8192
