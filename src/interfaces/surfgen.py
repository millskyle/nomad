# file: surfgen.py
# 
# Routines for running surfgen.x surface evaluation
#
import sys
import os
import shutil
import pathlib
import numpy as np
import scr.fmsio.fileio as fileio
from ctypes import *


# number of atoms
n_atoms = 0
# number of states
n_states = 0
# surfgen library
libsurf = None
# cache of derivative value

class Surface:
    """Object containing potential energy surface data."""
    def __init__(self, tag, n_states, t_dim):
        # necessary for array allocation
        self.tag      = tag
        self.n_states = n_states
        self.t_dim    = t_dim

        # these are the standard quantities ALL interface_data objects return
        self.data_keys = []
        self.geom      = np.zeros(t_dim)
        self.potential = np.zeros(n_states)
        self.deriv     = np.zeros((t_dim, n_states, n_states))
        # for time being, this is just the hessian -- may be desirable to
        # have derivatives of derivative couplings later
        self.deriv2    = np.zeros((t_dim, t_dim, n_states))
        self.coupling  = np.zeros((t_dim, n_states, n_states))

    def copy(self):
        """Creates a copy of a Surface object."""
        new_info = Surface(self.tag, self.n_states, self.t_dim)

        # required potential data
        new_info.data_keys = copy.copy(self.data_keys)
        new_info.geom      = copy.deepcopy(self.geom)
        new_info.potential = copy.deepcopy(self.potential)
        new_info.deriv     = copy.deepcopy(self.deriv)
        new_info.deriv2    = copy.deepcopy(self.deriv2)
        new_info.coupling  = copy.deepcopy(self.coupling)

        return new_info

#---------------------------------------------------------------------
#
# Functions called from interface object
#
#---------------------------------------------------------------------
#
# init_interface: intialize surfgen and set up for evaluation
#
def init_interface():
    global libsurf
    err = 0
    
    # Check that $SURFGEN is set and load library, then check for input files.
    sgen_path = os.environ['SURFGEN']
    if not os.path.isfile(sgen_path+'/lib/libsurfgen.so'):
        print("Surfgen library not found in: "+sgen_path+'/lib')
        sys.exit()
    libsurf = cdll.LoadLibrary(sgen_path+'/lib/libsurfgen.so')

    err = check_surfgen_input('./input')
    if err != 0:
        print("Missing surfgen input files at: ./input")
        sys.exit()
        
    initialize_surfgen_potential()
        
#
# evaluate_trajectory: evaluate all reaqusted electronic structure
# information for a single trajectory
#
def evaluate_trajectory(traj, t=None):
    global n_atoms, n_states

    na = c_longlong(n_atoms)
    ns = c_longlong(n_states)

    na3 = 3 * na
    ns2 = ns * ns

    # convert to c_types for interfacing with surfgen shared
    # library.
    cgeom = traj.x()
    energy = [0.0] * ns
    agrads = [0.0] * (ns2 * na3)
    hmat = [0.0] * ns2
    dgrads = [0.0] * (ns2 * na3)

    cgeom = (c_double * na3)(*cgeom)
    energy= (c_double * nstates)(*energy)
    agrads= (c_double * (ns2 * na3)) (*cgrads)
    hmat  = (c_double * ns2) (*hmat)
    dgrads= (c_double * (ns2 * na3)) (*dgrads)

    lib.evaluatesurfgen77_(byref(na), byref(ns), cgeom,
                           energy, agrads, hmat, dgrads)

    surf_info = Surface(traj.label, n_states, 3*n_atoms)
    surf_info.geom      = traj.x()
    surf_info.potential = energy
    surf_info.deriv     = set_phase(traj, cgrads)
    surf_info.coupling  = surf_info.deriv - np.diag(np.diag(surf_info.deriv))

    return surf_info

#
# evaluate_centroid: evaluate all requested electronic structure 
# information at a centroid
#
def evalutate_centroid(cent, t=None):

    return evaluate_trajectory(cent, t=None) 


#---------------------------------------------------------------------
#
# "Private" functions
#
#--------------------------------------------------------------------
#
# initialize_surfgen_potential: call initpotential_ to initialize
# surfgen surface evalutation.
#
def initialize_surfgen_potential():
    global libsurf, n_atoms, n_states
    print("\n --- INITIALIZING SURFGEN SURFACE --- \n")
    libsurf.initpotential_()
    n_atoms = c_longlong.in_dll(libsurf,'__progdata_MOD_natoms').value
    n_states= c_longlong.in_dll(libsurf,'__hddata_MOD_nstates').value

#
# check_surfgen_input: check for all files necessary for successful
# surfgen surface evaluation.
#
def check_surfgen_input(path):
    # Input:
    #  path = path to directoy executing surfgen
    #
    # The following files are necessary for surfgen.x to run:
    #  hd.data, surfgen.in, coord.in, refgeom, irrep.in,
    #  error.log
    files = [path+'/hd.data', path+'/surfgen.in', path+'/coord.in',\
             path+'/refgeom', path+'/irrep.in', path+'/error.log']
    for i in range(len(files)):
        err = check_file_exists(files[i])
        if err != 0:
            print("File not found: "+files[i])

    return err

#
# check_file_exists: check if file exists
#
def check_file_exists(fname):
    err = 0
    if not os.path.isfile(fname):
        err = 1
    return err


#
# determine the phase of the computed coupling that yields smallest
# change from previous coupling
#
def set_phase(traj, new_coup):
    global n_atoms, n_states

    label = traj.label
    if type(traj) is trajectory.Trajectory:
        state = traj.state
    else:
        state = min(traj.pstates)

    # pull data to make consistent
    if traj.pes_data is not None:
        old_coup = traj.pes_data.deriv
    else:
        old_coup = np.zeros((3*n_atoms, n_states, n_states))

    for i in range(n_states):
        for j in range(i):
            # if the previous coupling is vanishing, phase of new coupling is arbitrary
            if np.linalg.norm(old_coup[:,i,j]) > glbl.fpzero:
                # check the difference between the vectors assuming phases of +1/-1
                norm_pos = np.linalg.norm( new_coup[:,i,j] - old_coup[:,i,j])
                norm_neg = np.linalg.norm(-new_coup[:,i,j] - old_coup[:,i,j])

                if norm_pos > norm_neg:
                    new_coup[:,i,j] *= -1.
                    new_coup[:,j,i] *= -1.

    return new_coup

