#!/bin/sh

make bin/"$1"_profile
ncu --metrics lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_read_lookup_hit.sum,dram__bytes_read.sum,dram__bytes_write.sum bin/"$1"_profile 16384 16384 16384
