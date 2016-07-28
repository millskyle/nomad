"""
Compute dirac delta integrals over trajectories traveling on adiabataic
potentials
""" 

import sys
import numpy as np
import src.fmsio.glbl as glbl
import src.interfaces.vcham.hampar as ham

# Let propagator know if we need data at centroids to propagate
require_centroids = False

# Determines the basis set
basis = 'dirac_delta'

#
# potential coupling matrix element between two trajectories
#
def v_integral(traj1, traj2=None, centroid=None, S_ij=None):
    """if we are passed a single trajectoy"""
    if traj2 is None:
        # Adiabatic energy
        return traj1.energy(traj1.state)
    #
    # off-diagonal matrix element, between trajectories on the same
    # state
    elif traj1.state == traj2.state:
        # Adiabatic energy
        return traj1.energy(traj1.state) * traj2.h_overlap(traj1)
    #
    # off-diagonal matrix elements between trajectories on different 
    # elecronic states
    elif traj1.state != traj2.state:
        # Derivative coupling
        fij = traj2.derivative(traj1.state)
        v = np.dot(fij, traj2.deldx_m(traj1))
        return v
    
    else:
        print('ERROR in v_integral -- argument disagreement')
        return 0.
        
#
# kinetic energy integral over trajectories
#
def ke_integral(traj1, traj2, S_ij=None):
    """Returns kinetic energy integral over trajectories."""
    ke = complex(0.,0.)
    if traj1.state == traj2.state:
        if glbl.fms['interface'] == 'vibronic':
            for i in range(traj1.n_particle):
                ke -= (traj2.particles[i].deld2x(traj1.particles[i])
                       * ham.freq[i]/2.0)
        else:
            for i in range(traj1.n_particle):
                ke -= (traj2.particles[i].deld2x(traj1.particles[i]) /
                       (2.*traj2.particles[i].mass))
        return ke * traj2.h_overlap(traj1)
    else:
        return ke
    
#
# return the matrix element <Psi_1 | d/dt | Psi_2> 
#
def sdot_integral(traj1, traj2, S_ij=None):
    sdot =  np.dot( traj2.velocity(), traj1.deldx(traj2) ) + \
            np.dot( traj2.force()   , traj1.deldp(traj2) ) + \
            complex(0.,1.) * traj2.phase_dot() * traj2.h_overlap(traj1)
    return sdot
    
