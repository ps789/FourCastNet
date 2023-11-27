#!/bin/bash
next_index=0
for ((year=2007; year <= 2017; year++));
do
	let index=$next_index
	let next_index=$next_index+365
	if [ $((year % 4)) -eq 0 ]; then
		let next_index=$next_index+1
	fi
	echo $next_index
	ncks -d time,$(($index)),$(($next_index-1)) ../ERA5_data/data_constant_mask.nc -O ../ERA5_data/data_constant_mask_$year.nc
done

