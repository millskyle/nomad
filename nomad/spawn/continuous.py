"""
Routines for the continuous spawning algorithm.

Schematic:

  start, ti
    |           if Min(parent.overlap(traj(st',ti))) < Omin
parent(r,p,st,ti) ------------------------------------> child(r,p',st',ti)

1. if the minimum overlap between the "parent" trajectory on state st with
   trajectories on state st' drops below a user-defined threshold, spawn a
   new function on st' at the same position, with a scaled momentum to
   enforce constraint that classical energies be equal.
"""
import numpy as np
import nomad.fmsio.glbl as glbl
import nomad.fmsio.fileio as fileio
import nomad.dynamics.timings as timings
import nomad.basis.trajectory as trajectory
import nomad.spawn.utilities as utilities

@timings.timed
def spawn(master, dt):
    """Spawns a basis function if the minimum overlap drops below a given
    threshold."""
    basis_grown  = False
    current_time = master.time
    for i in range(master.n_traj()):
        if not master.traj[i].alive:
            continue

        parent = master.traj[i]
        for st in range(master.nstates):
            # don't check overlap with basis functions on same state
            if st == parent.state:
                continue

            s_array = [abs(glbl.integrals.traj_overlap(parent,
                                                  master.traj[j],
                                                  nuc_only=True))
                       if master.traj[j].state == st
                       and master.traj[j].alive else 0.
                       for j in range(master.n_traj())]
            if max(s_array, key=abs) < glbl.spawning['continuous_min_overlap']:
                child           = parent.copy()
                child.amplitude = 0j
                child.state     = st
                child.parent    = parent.label

                success = utilities.adjust_child(parent, child,
                                    parent.nact(parent.state, child.state))
                sij = glbl.integrals.traj_overlap(parent, child, nuc_only=True)

                # try to set up the child
                if not success:
                    pass
#                    fileio.print_fms_logfile('spawn_bad_step',
#                                             ['cannot adjust kinetic energy of child'])
                elif abs(sij) < glbl.spawning['spawn_olap_thresh']:
                    pass
#                    fileio.print_fms_logfile('spawn_bad_step',
#                                             ['child-parent overlap too small'])
                else:
                    child_created = True
                    spawn_time = current_time
                    parent.last_spawn[child.state] = spawn_time
                    child.last_spawn[parent.state] = spawn_time

                    bundle_overlap = utilities.overlap_with_bundle(child, master)
                    if not bundle_overlap:
                        basis_grown = True
                        master.add_trajectory(child)
                        fileio.print_fms_logfile('spawn_success',
                                                 [current_time, parent.label, st])
                        utilities.write_spawn_log(current_time, current_time,
                                                  current_time, parent,
                                                  master.traj[-1])
                    else:
                        err_msg = ('Traj ' + str(parent.label) + ' from state ' +
                                   str(parent.state) + ' to state ' + str(st) +
                                   ': ' + 'overlap with bundle too large,' +
                                   ' s_max=' + str(glbl.propagate['sij_thresh']))
                        fileio.print_fms_logfile('spawn_bad_step', [err_msg])
    return basis_grown

def in_coupled_regime(bundle):
    """Checks if we are in spawning regime. Since we are always spawning,
       this function always returns False for continuous spawning -- no 
       need to change timestep to spawn."""
 
    return False 


