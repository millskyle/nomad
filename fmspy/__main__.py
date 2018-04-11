#!/usr/bin/env python
"""
Main module used to initiate and run FMSpy.
"""
import os
import sys
import random
import numpy as np
import mpi4py.MPI as MPI
import fmspy.fmsio.glbl as glbl
import fmspy.fmsio.fileio as fileio
import fmspy.basis.bundle as bundle
import fmspy.dynamics.timings as timings
import fmspy.dynamics.initialize as initialize
import fmspy.dynamics.step as step


def init():
    """Initializes the FMSpy inputs.

    This must be separate from main so that an error which occurs
    before the input file is created will be written to stdout.
    """
    # initialize MPI communicator
    if glbl.mpi['parallel']:
        glbl.mpi['comm']  = MPI.COMM_WORLD
        glbl.mpi['rank']  = glbl.mpi['comm'].Get_rank()
        glbl.mpi['nproc'] = glbl.mpi['comm'].Get_size()
    else:
        glbl.mpi['rank']  = 0
        glbl.mpi['nproc'] = 1

    # start the global timer
    timings.start('global')

    # read in options/variables pertaining to the running
    # of the dynamics, pass the starting time of the simluation
    # a nd the end time
    fileio.read_input_file()

    # initialize random number generator
    random.seed(glbl.sampling['seed'])

    # initialize the trajectory and bundle output files
    fileio.init_fms_output()


def main():
    """Runs the main FMSpy routine."""
    # Create the collection of trajectories
    master = bundle.Bundle(glbl.propagate['n_states'])

    # set the initial conditions for trajectories
    initialize.init_bundle(master)

    while master.time < glbl.propagate['simulation_time']:
        # set the time step --> top level time step should always
        # be default time step. fms_step_bundle will decide if/how
        # dt should be shortened for numerics
        time_step = step.fms_time_step(master)

        # take an fms dynamics step
        master = step.fms_step_bundle(master, time_step)

        # if no more live trajectories, simulation is complete
        if master.nalive == 0:
            break

        # determine whether it is necessary to update the output logs
        if fileio.update_logs(master) and glbl.mpi['rank'] == 0:
            # update the fms output files, as well as checkpoint, if necessary
            master.update_logs()

    # clean up, stop the global timer and write logs
    fileio.cleanup_end()


def main_cli():
    # parse command line arguments
    if '-mpi' in sys.argv:
        glbl.mpi['parallel'] = True

    # initialize
    init()
    # if an error occurs, cleanup and report the error
    sys.excepthook = fileio.cleanup_exc
    # run the main routine
    main()


if __name__ == '__main__':
    main_cli()
