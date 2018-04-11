"""
The Trajectory object and its associated functions.
"""
import sys
import copy
import numpy as np
import fmspy.dynamics.timings as timings
import fmspy.fmsio.glbl as glbl


class Trajectory:
    """Class constructor for the Trajectory object."""
    def __init__(self, nstates, dim, width=None, mass=None,
                 label=0, parent=0):
        # total number of states
        self.nstates = int(nstates)
        # dimensionality of the trajectory
        self.dim     = int(dim)
        # widths of gaussians for each dimension
        if width is None:
            self.width = np.zeros(dim)
        else:
            self.width = np.asarray(width)
        # masses associated with each dimension
        if mass is None:
            self.mass = np.zeros(dim)
        else:
            self.mass = np.array(mass)
        # unique identifier for trajectory
        self.label        = label
        # trajectory that spawned this one:
        self.parent     = parent

        # current position of the trajectory
        self.pos        = np.zeros(self.dim)
        # current momentum of the trajectory
        self.mom        = np.zeros(self.dim)
        # state trajectory exists on
        self.state      = 0
        # whether the trajectory is alive (i.e. contributes to the wavefunction)
        self.alive      = True
        # whether the trajectory is active (i.e. is being propagated)
        self.active     = True
        # amplitude of trajectory
        self.amplitude  = 0j
        # phase of the trajectory
        self.gamma      = 0.
        # time from which the death watch begini as
        self.deadtime   = -1.
        # time of last spawn
        self.last_spawn = np.zeros(self.nstates)
        # time trajectory last left coupling region
        self.exit_time  = np.zeros(self.nstates)
        # data structure to hold the pes data from the interface
        self.pes_data  = None

    @timings.timed
    def copy(self):
        """Copys a Trajectory object with new references."""
        new_traj = Trajectory(self.nstates, self.dim, self.width, self.mass,
                              self.label, self.parent)
        new_traj.state      = copy.copy(self.state)
        new_traj.alive      = copy.copy(self.alive)
        new_traj.amplitude  = copy.copy(self.amplitude)
        new_traj.gamma      = copy.copy(self.gamma)
        new_traj.deadtime   = copy.copy(self.deadtime)
        new_traj.pos        = copy.deepcopy(self.pos)
        new_traj.mom        = copy.deepcopy(self.mom)
        new_traj.last_spawn = copy.deepcopy(self.last_spawn)
        new_traj.exit_time  = copy.deepcopy(self.exit_time)
        if self.pes_data is not None:
            new_traj.pes_data = self.pes_data.copy()
        return new_traj

    #-------------------------------------------------------------------
    #
    # Trajectory status functions
    #
    #--------------------------------------------------------------------
    def dead(self, time, uncoupled_thresh):
        """Returns true if the trajectory is dead."""
        return self.deadtime != -1 and (time - self.deadtime) > uncoupled_thresh

    #----------------------------------------------------------------------
    #
    # Functions for setting basic pes information from trajectory
    #
    #----------------------------------------------------------------------
    def update_x(self, pos):
        """Updates the position of the trajectory.
        """
        self.pos = np.array(pos)

    def update_p(self, mom):
        """Updates the momentum of the trajectory.
        """
        self.mom = np.array(mom)

    def update_phase(self, phase):
        """Updates the nuclear phase."""
        self.gamma = 0.5 * np.dot(self.x(), self.p())
#        self.gamma = phase
#        if abs(self.gamma) > 2*np.pi:
#            self.gamma = self.gamma % 2*np.pi

    def update_amplitude(self, amplitude):
        """Updates the amplitude of the trajectory."""
        self.amplitude = amplitude

    def update_pes_info(self, pes_info):
        """Updates information about the potential energy surface."""
        self.pes_data = pes_info.copy()

    #-----------------------------------------------------------------------
    #
    # Functions for retrieving basic pes information from trajectory
    #
    #-----------------------------------------------------------------------
    def x(self):
        """Returns the position of the trajectory as an array."""
        return self.pos

    def p(self):
        """Returns the momentum of the trajectory as an array."""
        return self.mom

    def phase(self):
        """Returns the phase of the trajectory."""
        return self.gamma

    def masses(self):
        """Returns a vector containing masses associated with each dimension"""
        return self.mass

    def widths(self):
        """Returns a vector containing the widths of the basis functions
        along each degree of freedom."""
        return self.width

    #--------------------------------------------------------------------
    #
    # Functions to update information about the potential energy surface
    #
    #--------------------------------------------------------------------
    def energy(self, state):
        """Returns the potential energies.

        Add the energy shift right here. If not current, recompute them.
        """
        if np.linalg.norm(self.pes_data.geom - self.x()) > 10.*glbl.constants['fpzero']:
            print('WARNING: trajectory.energy() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes_data.geom))
        return self.pes_data.potential[state] + glbl.propagate['pot_shift']

    def derivative(self, state_i, state_j):
        """Returns the derivative with ket state = rstate.

        Bra state assumed to be the current state.
        """
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes_data.geom))
        return self.pes_data.deriv[:, state_i, state_j]

    def hessian(self, state_i):
        """Returns the hessian of the potential on state state_i
        """
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes_data.geom))
        return self.pes_data.deriv2[:, :, state_i]

    def coupling(self, state_i, state_j):
        """Returns the coupling between surfaces state_i and state_j
        """
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: trajectory.derivative() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes_data.geom))
        return self.pes_data.coupling[:, state_i, state_j]

    def scalar_coup(self, state_i, state_j):
        """Returns the scalar coupling for Hamiltonian
           block (self.state,c_state)."""
        if 'scalar_coup' not in self.pes_data.data_keys:
            return 0.
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: trajectory.scalar_coup() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes_data.geom))
        return self.pes_data.scalar_coup[state_i, state_j]

    def nact(self, state_i, state_j):
        """Returns the derivative coupling between adiabatic states
           block (self.state,c_state)."""
        if 'nac' not in self.pes_data.data_keys:
            return 0.
        if np.linalg.norm(self.pes_data.geom - self.x()) > glbl.constants['fpzero']:
            print('WARNING: trajectory.nact() called, ' +
                  'but pes_geom != trajectory.x(). ID=' + str(self.label)+
                  '\ntraj.x()='+str(self.x())+"\npes_geom="+str(self.pes_data.geom))
        return self.pes_data.nac[:,state_i, state_j]


    #------------------------------------------------------------------------
    #
    # Computed quantities from the trajectory
    #
    #------------------------------------------------------------------------
    def potential(self):
        """Returns classical potential energy of the trajectory."""
        return self.energy(self.state)

    def kinetic(self):
        """Returns classical kinetic energy of the trajectory."""
        return sum( self.p() * self.p() * glbl.pes.kecoeff )

    def classical(self):
        """Returns the classical energy of the trajectory."""
        return self.potential() + self.kinetic()

    def velocity(self):
        """Returns the velocity of the trajectory."""
        return self.p() * (2. * glbl.pes.kecoeff)

    def force(self):
        """Returns the gradient of the trajectory state."""
        return -self.derivative(self.state, self.state)

    def phase_dot(self):
        """Returns time derivatives of the phase."""
        # d[gamma]/dt = T - V - alpha/(2M)
        if not glbl.propagate['phase_prop']:
            return 0.
        else:
            return (self.kinetic() - self.potential() -
                    np.dot(self.widths(), glbl.pes.kecoeff) )

    def coupling_norm(self, j_state):
        """Returns the norm of the coupling vector."""
        return np.linalg.norm(self.coupling(self.state, j_state))

    def coup_dot_vel(self, j_state):
        """Returns the coupling dotted with the velocity."""
        if glbl.variables['surface_rep'] == 'diabatic':
            return 0.
        else:
            return np.dot(self.velocity(), self.coupling(self.state, j_state))

    def eff_coup(self, j_state):
        """Returns the effective coupling."""
        if glbl.variables['surface_rep'] == 'diabatic':
            return self.coupling_norm(j_state)
        else:
            # F.p/m
            coup = self.coup_dot_vel(j_state)
            # G
            if glbl.interface['coupling_order'] > 1:
                coup += self.scalar_coup(self.state, j_state)
            return coup

    def same_state(self, j_state):
        """Determines if a given state is the same as the trajectory state."""
        return self.state == j_state

    #--------------------------------------------------------------------------
    #
    # routines to write/read trajectory from a file stream
    #
    #--------------------------------------------------------------------------
    def write_trajectory(self, chkpt):
        """Writes trajectory information to a file stream."""
        np.set_printoptions(precision=8, linewidth=80, suppress=False)
        chkpt.write('     {:5s}            alive\n'.format(str(self.alive)))
        chkpt.write('{:10d}            nstates\n'.format(self.nstates))
        chkpt.write('{:10d}            traj ID\n'.format(self.label))
        chkpt.write('{:10d}            state\n'.format(self.state))
        chkpt.write('{:10d}            parent ID\n'.format(self.parent))
        chkpt.write('{:10.2f}            dead time\n'.format(self.deadtime))
        chkpt.write('{:16.12f}      phase\n'.format(self.gamma))
        chkpt.write('{:16.12f}         amplitude\n'.format(self.amplitude))
        chkpt.write('# potential energy -- nstates\n')
        energies = np.array([self.energy(i) for i in range(self.nstates)])
        energies.tofile(chkpt, ' ', '%14.10f')
        chkpt.write('\n# exit coupling region\n')
        self.exit_time.tofile(chkpt, ' ', '%8.2f')
        chkpt.write('\n# last spawn\n')
        self.last_spawn.tofile(chkpt, ' ', '%8.2f')
        chkpt.write('\n# position\n')
        self.x().tofile(chkpt, ' ', '%12.8f')
        chkpt.write('\n# momentum\n')
        self.p().tofile(chkpt, ' ', '%12.8f')

        # Writes out gradient
        chkpt.write('\n# gradient state = {:4d}\n'.format(self.state))
        self.derivative(self.state,self.state).tofile(chkpt, ' ', '%16.10e')

        # write out the coupling
        for i in range(self.nstates):
            if i != self.state:
                chkpt.write('# coupling state = {:4d}\n'.format(i))
                self.derivative(self.state,i).tofile(chkpt, ' ', '%16.10e')
                chkpt.write('\n')


    def read_trajectory(self, chkpt):
        """Reads the trajectory information from a file.

        This assumes the trajectory invoking this function has been
        initially correctly and can hold all the information.
        """
        self.alive     = bool(chkpt.readline().split()[0])
        self.nstates   = int(chkpt.readline().split()[0])
        self.label     = int(chkpt.readline().split()[0])
        self.state     = int(chkpt.readline().split()[0])
        self.parent    = int(chkpt.readline().split()[0])
        self.deadtime  = float(chkpt.readline().split()[0])
        self.gamma     = float(chkpt.readline().split()[0])
        self.amplitude = complex(chkpt.readline().split()[0])

        # create Surface object, if doesn't already exist
        if self.pes_data is None:
            self.pes_data = glbl.pes.Surface(self.label, self.nstates,
                                             self.dim)

        # potential energy -- nstates
        chkpt.readline()
        self.pes_data.potential = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        # exit coupling region
        chkpt.readline()
        self.exit_time = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        # last spawn
        chkpt.readline()
        self.last_spawn = np.fromstring(chkpt.readline(), sep=' ', dtype=float)
        # position
        chkpt.readline()
        self.update_x(np.fromstring(chkpt.readline(), sep=' ', dtype=float))
        # momentum
        chkpt.readline()
        self.update_p(np.fromstring(chkpt.readline(), sep=' ', dtype=float))

        # read gradients
        chkpt.readline()
        self.pes_data.deriv[:,self.state,self.state] = np.fromstring(chkpt.readline(),
                                                                     sep=' ', dtype=float)

        # read couplings
        for i in range(self.nstates):
            if i != self.state:
                chkpt.readline()
                self.pes_data.deriv[:,self.state,i] = np.fromstring(chkpt.readline(),
                                                                    sep=' ', dtype=float)
                self.pes_data.deriv[:,i,self.state] = -self.pes_data.deriv[:,self.state,i]

