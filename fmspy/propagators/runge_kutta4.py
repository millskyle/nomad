"""
Routines for propagation with the 4th order Runge-Kutta algorithm.

4th order Runge-Kutta:
  x(t+dt) = x(t) + dt*[(1/6)kx1 + (1/3)kx2 + (1/3)kx3 + (1/6)kx4]
  p(t+dt) = p(t) + dt*[(1/6)kp1 + (1/3)kp2 + (1/3)kp3 + (1/6)kp4]

  ky1 = f[t, y(t)] = dy(t)/dt
  ky2 = f[t + (1/2)dt, y(t) + (1/2)ky1*dt]
  ky3 = f[t + (1/2)dt, y(t) + (1/2)ky2*dt]
  ky4 = f[t + dt, y(t) + ky3*dt]
"""
import numpy as np
import fmspy.fmsio.glbl as glbl
import fmspy.dynamics.timings as timings
import fmspy.dynamics.surface as surface


rk_ordr = 4
coeff = np.array([0.5, 0.5, 1.])
wgt = np.array([1./6., 1./3., 1./3., 1./6.])
propphase = glbl.propagate['phase_prop']


@timings.timed
def propagate_bundle(master, dt):
    """Propagates the Bundle object with RK4."""
    ncrd = master.traj[0].dim
    ntraj = master.n_traj()
    kx = np.zeros((ntraj, rk_ordr, ncrd))
    kp = np.zeros((ntraj, rk_ordr, ncrd))
    kg = np.zeros((ntraj, rk_ordr, ncrd))

    for rk in range(rk_ordr):
        tmpbundle = master.copy()
        for i in range(ntraj):
            if tmpbundle.traj[i].active:
                propagate_rk(tmpbundle.traj[i], dt, rk, kx[i], kp[i], kg[i])

        # update the PES to evaluate new gradients
        if rk < rk_ordr - 1:
            surface.update_pes(tmpbundle, update_centroids=False)

    # update to the final position
    for i in range(ntraj):
        if master.traj[i].active:
            master.traj[i].update_x(master.traj[i].x() +
                                    np.sum(wgt[:,np.newaxis]*kx[i], axis=0))
            master.traj[i].update_p(master.traj[i].p() +
                                    np.sum(wgt[:,np.newaxis]*kp[i], axis=0))
            if propphase:
                master.traj[i].update_phase(master.traj[i].phase() +
                                            np.sum(wgt[:,np.newaxis]*kg[i], axis=0))
    surface.update_pes(master)


@timings.timed
def propagate_trajectory(traj, dt):
    """Propagates a single trajectory with RK4."""
    ncrd = traj.dim
    kx = np.zeros((rk_ordr, ncrd))
    kp = np.zeros((rk_ordr, ncrd))
    kg = np.zeros((rk_ordr, ncrd))

    for rk in range(rk_ordr):
        tmptraj = traj.copy()
        propagate_rk(tmptraj, dt, rk, kx, kp, kg)

        # update the PES to evaluate new gradients
        if rk < rk_ordr - 1:
            surface.update_pes_traj(tmptraj)

    # update to the final position
    traj.update_x(traj.x() + np.sum(wgt[:,np.newaxis]*kx, axis=0))
    traj.update_p(traj.p() + np.sum(wgt[:,np.newaxis]*kp, axis=0))
    if propphase:
        traj.update_phase(traj.phase() + np.sum(wgt[:,np.newaxis]*kg, axis=0))
    surface.update_pes_traj(traj)


def propagate_rk(traj, dt, rk, kxi, kpi, kgi):
    """Gets k values and updates the position and momentum by
    a single rk step."""
    # calculate the k values at this point
    kxi[rk] = dt * traj.velocity()
    kpi[rk] = dt * traj.force()
    if propphase:
        kgi[rk] = dt * traj.phase_dot()

    # update the position using the last k value, except for k4
    if rk < rk_ordr - 1:
        traj.update_x(traj.x() + coeff[rk]*kxi[rk])
        traj.update_p(traj.p() + coeff[rk]*kpi[rk])
        if propphase:
            traj.update_phase(traj.phase() + coeff[rk]*kgi[rk])
