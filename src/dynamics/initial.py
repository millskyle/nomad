"""
Routines for initializing dynamics calculations.
"""
import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
import src.basis.bundle as bundle
import src.dynamics.surface as surface


#--------------------------------------------------------------------------
#
# Externally visible routines
#
#--------------------------------------------------------------------------
def init_bundle(master):
    """Initializes the trajectories."""
    pes = __import__('src.interfaces.'+glbl.fms['interface'],
                     fromlist=['NA'])
    distrib = __import__('src.dynamics.'+glbl.fms['init_sampling'],
                         fromlist=['NA'])

    # initialize the trajectory and bundle output files
    fileio.init_fms_output()

    # initialize the interface we'll be using the determine the
    # the PES. There are some details here that trajectories
    # will want to know about
    pes.init_interface()

    # initialize the surface module -- caller of the pes interface
    surface.init_surface(glbl.fms['interface'])

    # now load the initial trajectories into the bundle
    if glbl.fms['restart']:
        init_restart(master)
    else:
        # sample the requested phase distribution, some methods may also
        # set the electronic state of the trajectory. If so, don't call
        # set_initial_state separately
        state_set = distrib.sample_distribution(master)
        # set the initial state of the trajectories in bundle. This may
        # require evaluation of electronic structure
        if not state_set:
            set_initial_state(master)

    # add virtual basis functions, if desired (i.e. virtual basis = true)
    if glbl.fms['virtual_basis']:
        virtual_basis(master)

    # update all pes info for all trajectories and centroids (where necessary)
    surface.update_pes(master)
    # compute the hamiltonian matrix...
    master.update_matrices()
    # so that we may appropriately renormalize to unity
    master.renormalize()

    # this is the bundle at time t=0.  Save in order to compute auto
    # correlation function
    glbl.bundle0 = master.copy()

    # write to the log files
    master.update_logs()

    fileio.print_fms_logfile('t_step', [master.time, glbl.fms['default_time_step'],
                                        master.nalive])

    return master.time


#---------------------------------------------------------------------------
#
# Private routines
#
#----------------------------------------------------------------------------
def init_restart(master):
    """Initializes a restart."""
    if glbl.fms['restart_time'] == -1.:
        fname = fileio.home_path+'/last_step.dat'
    else:
        fname = fileio.home_path+'/checkpoint.dat'

    master.read_bundle(fname, glbl.fms['restart_time'])


def set_initial_state(master):
    """Sets the initial state of the trajectories in the bundle."""
    if glbl.fms['init_state'] != -1:
        for i in range(master.n_traj()):
            master.traj[i].state = glbl.fms['init_state']
    elif glbl.fms['init_brightest']:
        # set all states to the ground state
        for i in range(master.n_traj()):
            master.traj[i].state = 0

        # compute transition dipoles
        surface.update_pes(master)

        # set the initial state to the one with largest t. dip.
        for i in range(master.n_traj()):
            tdip = (np.linalg.norm(master.traj[i].dipole(j))
                    for j in range(1, glbl.fms['n_states']+1))
            master.traj[i].state = np.argmax(tdip)+1
    else:
        raise ValueError('Ambiguous initial state assignment.')


def virtual_basis(master):
    """Add virtual basis funcions.

    If additional virtual basis functions requested, for each trajectory
    in bundle, add aditional basis functions on other states with zero
    amplitude.
    """
    for i in range(master.n_traj()):
        for j in range(master.nstates):
            if j == master.traj[i].state:
                continue

            new_traj = master.traj[i].copy()
            new_traj.amplitude = 0j
            new_traj.state = j
            master.add_trajectory(new_traj)
