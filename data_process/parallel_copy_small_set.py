#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation


# Instructions: 
# Set Nimgtot correctly

import h5py
from mpi4py import MPI
import numpy as np
import time
from netCDF4 import Dataset as DS
import os

def writetofile(src, dest, channel_idx, varslist, src_idx=0, frmt='nc'):
    if not os.path.isfile(src):
        print("not a path")
    if os.path.isfile(src):
        batch = 5
        rank = MPI.COMM_WORLD.rank
        Nproc = MPI.COMM_WORLD.size
        Nimgtot = 365#src_shape[0]

        Nimg = Nimgtot//Nproc
        base = rank*Nimg
        end = (rank+1)*Nimg if rank<Nproc - 1 else Nimgtot
        idx = base

        for variable_name in varslist:

            if frmt == 'nc':
                fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
            elif frmt == 'h5':
                fsrc = h5py.File(src, 'r')[varslist[0]]
            fdest = h5py.File(dest, 'a', driver='mpio', comm=MPI.COMM_WORLD)

            start = time.time()
            while idx<end:
                if end - idx < batch:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:end,src_idx]
                    else:
                        ims = fsrc[idx:end]
                    print(ims.shape)
                    fdest['fields'][idx:end, channel_idx, :, :] = ims
                    print(channel_idx)
                    break
                else:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:idx+batch,src_idx]
                    else:
                        ims = fsrc[idx:idx+batch]
                    #ims = fsrc[idx:idx+batch]
                    fdest['fields'][idx:idx+batch, channel_idx, :, :] = ims
                    idx+=batch
                    ttot = time.time() - start
                    eta = (end - base)/((idx - base)/ttot)
                    hrs = eta//3600
                    mins = (eta - 3600*hrs)//60
                    secs = (eta - 3600*hrs - 60*mins)

            ttot = time.time() - start
            hrs = ttot//3600
            mins = (ttot - 3600*hrs)//60
            secs = (ttot - 3600*hrs - 60*mins)
            channel_idx += 1 
dir_dict = {}
for year in np.arange(2007, 2016):
    dir_dict[year] = 'train'

for year in np.arange(2016, 2017):
    dir_dict[year] = 'test'

for year in np.arange(2017, 2018):
    dir_dict[year] = 'out_of_sample'


print(dir_dict)

years = np.arange(2007, 2018)

for year in years:

    dest = f'../ERA5_data_formatted/{dir_dict[year]}/{year}.h5'
    with h5py.File(dest, 'w') as f:
        f.create_dataset('fields', shape = (365, 20, 73, 144), dtype='f')
    src = f'../ERA5_data/data_sfc_{year}.nc'
    #u10 v10 t2m
    writetofile(src, dest, 0, ['u10'])
    writetofile(src, dest, 1, ['v10'])
    writetofile(src, dest, 2, ['t2m'])

    #sp mslp
    writetofile(src, dest, 3, ['sp'])
    writetofile(src, dest, 4, ['msl'])

    #t850
    src = f'../ERA5_data/data_pl_{year}.nc'
    writetofile(src, dest, 5, ['t'], 30)

    #uvz1000
    writetofile(src, dest, 6, ['u'], 36)
    writetofile(src, dest, 7, ['v'], 36)
    writetofile(src, dest, 8, ['z'], 36)

    #uvz850
    writetofile(src, dest, 9, ['u'], 30)
    writetofile(src, dest, 10, ['v'], 30)
    writetofile(src, dest, 11, ['z'], 30)

    #uvz 500
    writetofile(src, dest, 12, ['u'], 21)
    writetofile(src, dest, 13, ['v'], 21)
    writetofile(src, dest, 14, ['z'], 21)

    #t500
    writetofile(src, dest, 15, ['t'], 21)

    #z50
    writetofile(src, dest, 16, ['z'], 8)

    #r500 
    writetofile(src, dest, 17, ['r'], 21)

    #r850
    writetofile(src, dest, 18, ['r'], 30)
    
    src = f'../ERA5_data/data_sfc_{year}.nc'
    #tcwv
    writetofile(src, dest, 19, ['tcwv'])

    #sst
    #src = '/project/projectdirs/dasrepo/ERA5/oct_2021_19_31_sfc.nc'
    #writetofile(src, dest, 20, ['sst'])


