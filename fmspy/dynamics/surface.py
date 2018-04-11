"""
Routines for handling the potential energy surface.

All calls to update the pes are localized here.  This facilitates parallel
execution of potential evaluations which is essential for ab initio PES.
"""
from functools import partial
import numpy as np
import fmspy.fmsio.glbl as glbl
import fmspy.basis.trajectory as trajectory
import fmspy.basis.centroid as centroid

pes_cache  = dict()

def update_pes(master, update_centroids=None):
    """Updates the potential energy surface."""
    global pes_cache
    success = True

    # this conditional checks to see if we actually need centroids,
    # even if propagator requests them
    if update_centroids is None or not glbl.integrals.require_centroids:
        update_centroids = glbl.integrals.require_centroids

    if glbl.mpi['parallel']:
        # update electronic structure
        exec_list = []
        n_total = 0 # this ensures traj.0 is on proc 0, etc.
        for i in range(master.n_traj()):
            if master.traj[i].active and not cached(master.traj[i].label,
                                                    master.traj[i].x()):
                n_total += 1
                if n_total % glbl.mpi['nproc'] == glbl.mpi['rank']:
                    exec_list.append(master.traj[i])

        if update_centroids:
            # update the geometries
            master.update_centroids()
            # now update electronic structure in a controled way to allow for
            # parallelization
            for i in range(master.n_traj()):
                for j in range(i):
                    if master.centroid_required(master.traj[i],master.traj[j]) and not \
                                           cached(master.cent[i][j].label,
                                                  master.cent[i][j].x()):
                        n_total += 1
                        if n_total % glbl.mpi['nproc'] == glbl.mpi['rank']:
                            exec_list.append(master.cent[i][j])

        local_results = []
        for i in range(len(exec_list)):
            if type(exec_list[i]) is trajectory.Trajectory:
                pes_calc = glbl.pes.evaluate_trajectory(exec_list[i], master.time)
            elif type(exec_list[i]) is centroid.Centroid:
                pes_calc = glbl.pes.evaluate_centroid(exec_list[i], master.time)
            else:
                raise TypeError('type='+str(type(exec_list[i]))+
                                'not recognized')
            local_results.append(pes_calc)

        global_results = glbl.mpi['comm'].allgather(local_results)

        # update the cache
        for i in range(glbl.mpi['nproc']):
            for j in range(len(global_results[i])):
                pes_cache[global_results[i][j].tag] = global_results[i][j]

        # update the bundle:
        # live trajectories
        for i in range(master.n_traj()):
            if master.traj[i].alive:
                master.traj[i].update_pes_info(pes_cache[master.traj[i].label])

        # and centroids
        if update_centroids:
            for i in range(master.n_traj()):
                for j in range(i):
                    if master.cent[i][j].label in pes_cache:
                        master.cent[i][j].update_pes_info(pes_cache[master.cent[i][j].label])
                        master.cent[j][i] = master.cent[i][j]

    # if parallel overhead not worth the time and effort (eg. pes known in closed form),
    # simply run over trajectories in serial (in theory, this too could be cythonized,
    # but unlikely to ever be bottleneck)
    else:
        # iterate over trajectories..
        for i in range(master.n_traj()):
            if master.traj[i].active:
                master.traj[i].update_pes_info(glbl.pes.evaluate_trajectory(
                                               master.traj[i], master.time))

        # ...and centroids if need be
        if update_centroids:
            # update the geometries
            master.update_centroids()
            for i in range(master.n_traj()):
                for j in range(i):
                # if centroid not initialized, skip it
                    if master.cent[i][j] is not None:
                        master.cent[i][j].update_pes_info(
                                          glbl.pes.evaluate_centroid(
                                          master.cent[i][j], master.time))
                        master.cent[j][i] = master.cent[i][j]

    return success


def update_pes_traj(traj):
    """Updates a single trajectory

    Used during spawning.
    """
    results = None

    if glbl.mpi['rank'] == 0:
        results = glbl.pes.evaluate_trajectory(traj)

    if glbl.mpi['parallel']:
        results = glbl.mpi['comm'].bcast(results, root=0)
        glbl.mpi['comm'].barrier()

    traj.update_pes_info(results)


def cached(label, geom):
    """Returns True if the surface in the cache corresponds to the current
    trajectory (don't recompute the surface)."""
    global pes_cache

    if label not in pes_cache:
        return False

    dg = np.linalg.norm(geom - pes_cache[label].geom)
    if dg <= glbl.constants['fpzero']:
        return True

    return False
