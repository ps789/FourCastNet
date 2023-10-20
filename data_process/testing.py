import h5py
from mpi4py import MPI
import numpy as np
import time
from netCDF4 import Dataset as DS
import os

fsrc = DS("../ERA5_data/data_pl_2017.nc", 'r', format="NETCDF4").variables
print(fsrc["v"])
