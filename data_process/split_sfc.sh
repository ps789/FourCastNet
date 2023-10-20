#!/bin/bash

for ((year = 2007; year <= 2017; year++));
do
	ncks -d time,$((365*$(($year-2007)))),$((364+365*$(($year - 2007)))) ../ERA5_data/data_sfc.nc -O ../ERA5_data/data_sfc_$year.nc
done

