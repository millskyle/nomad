"""
Form the non-Hermitian Hamiltonian matrix used in the collocation
method in the basis of the FMS trajectories.

This will necessarily involve a set of additional matrices. For ab initio
propagation, this is never the rate determining step. For numerical
propagation, however, it is THE bottleneck. Thus, this function is
compiled in Cython in order to make use of openMP parallelization.

As a matter of course, this function also builds:
   - the S matrix
   - the time derivative of the S matrix (sdot)
   - the effective Hamiltonian (i.e. i * S^-1 [ S * H - Sdot])
     --> this is the matrix employed to solve for the time
         dependent amplitudes

"""
import sys
import numpy as np
from scipy import linalg
import src.dynamics.timings as timings
import src.fmsio.glbl as glbl


def c_ind(i, j):
    """Returns the index in the cent array of the centroid between
    trajectories i and j."""
    if i == j:
        return -1
    else:
        a = max(i, j)
        b = min(i, j)
        return int(a*(a-1)/2 + b)


def ij_ind(index):
    """Gets the (i,j) index of an upper triangular matrix from the
    sequential matrix index 'index'."""
    i = 0
    while i*(i+1)/2 - 1 < index:
        i += 1
    return int(index-i*(i-1)/2), int(i-1)


@timings.timed
def pseudo_inverse(mat, dim):
    """ Modified version of the scipy pinv function. Altered such that
    the the cutoff for singular values can be set to a hard
    value. Note that by default the scipy cutoff of 1e-15*sigma_max is
    taken."""

    invmat = np.zeros((dim, dim), dtype=complex)
    mat = np.conjugate(mat)

    # SVD of the overlap matrix
    u, s, vt = linalg.svd(mat, full_matrices=True)

    # Condition number
    if s[dim-1] < 1e-90:
        cond = 1e+90
    else:
        cond = s[0]/s[dim-1]

    # Moore-Penrose pseudo-inverse
    if glbl.fms['sinv_thrsh'] == -1.0:
        cutoff = glbl.fms['sinv_thrsh'] * np.maximum.reduce(s)
    else:
        cutoff = glbl.fms['sinv_thrsh']
    for i in range(dim):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    invmat = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis],
                                                  np.transpose(u)))

    return invmat, cond


@timings.timed
def build_hamiltonian(intlib, traj_list, traj_alive, cent_list=None):
    """Builds the Hamiltonian matrix from a list of trajectories."""
    integrals = __import__('src.integrals.' + intlib, fromlist=['a'])

    n_alive = len(traj_alive)
    n_elem  = int(n_alive * (n_alive + 1) / 2)

    T        = np.zeros((n_alive, n_alive), dtype=complex)
    V        = np.zeros((n_alive, n_alive), dtype=complex)
    H        = np.zeros((n_alive, n_alive), dtype=complex)
    S        = np.zeros((n_alive, n_alive), dtype=complex)
    Strue    = np.zeros((n_alive, n_alive), dtype=complex)
    S_orthog = np.zeros((n_alive, n_alive), dtype=complex)
    Sinv     = np.zeros((n_alive, n_alive), dtype=complex)
    Sdot     = np.zeros((n_alive, n_alive), dtype=complex)
    Heff     = np.zeros((n_alive, n_alive), dtype=complex)

    for ij in range(n_elem):

        i, j = ij_ind(ij)
        ii = traj_alive[i]
        jj = traj_alive[j]

        # overlap matrix (excluding electronic component)
        S[i,j] = traj_list[ii].h_overlap(traj_list[jj])
        if i != j:
            S[j,i] = traj_list[jj].h_overlap(traj_list[ii])

        # True overlap matrix
        Strue[i,j] = traj_list[ii].overlap(traj_list[jj])
        Strue[j,i] = Strue[i,j].conjugate()

        # overlap matrix (including electronic component)
        if traj_list[ii].state == traj_list[jj].state:
            S_orthog[i,j] = S[i,j]
            S_orthog[j,i] = S[j,i]

            # time derivative of the overlap matrix
            Sdot[i,j] = integrals.sdot_integral(traj_list[ii], traj_list[jj],
                                                S_ij=S[i,j])
            if i != j:
                Sdot[j,i] = integrals.sdot_integral(traj_list[jj], traj_list[ii],
                                                    S_ij=S[j,i])

            # kinetic energy matrix
            T[i,j] = integrals.ke_integral(traj_list[ii], traj_list[jj],
                                           S_ij=S[i,j])
            if i != j:
                T[j,i] = integrals.ke_integral(traj_list[jj], traj_list[ii],
                                               S_ij=S[j,i])

        else:
            S_orthog[i,j] = 0.
            S_orthog[j,i] = 0.

        # potential energy matrix
        if integrals.require_centroids:
            if i == j:
                V[i,j] = integrals.v_integral(traj_list[ii], traj_list[jj],
                                              traj_list[ii], S_ij=S[i,j])
            else:
                V[i,j] = integrals.v_integral(traj_list[ii], traj_list[jj],
                                              cent_list[c_ind(ii,jj)], S_ij=S[i,j])
                V[j,i] = integrals.v_integral(traj_list[jj], traj_list[ii],
                                              cent_list[c_ind(jj,ii)], S_ij=S[j,i])
        else:
            V[i,j] = integrals.v_integral(traj_list[ii], traj_list[jj],
                                          S_ij=S[i,j])
            if i != j:
                V[j,i] = integrals.v_integral(traj_list[jj], traj_list[ii],
                                              S_ij=S[j,i])

        # Hamiltonian matrix in non-orthongonal basis
        H[i,j] = T[i,j] + V[i,j]
        H[j,i] = T[j,i] + V[j,i]

    # compute the S^-1, needed to compute Heff
    Sinv, cond = pseudo_inverse(S_orthog, n_alive)

    Heff = np.dot( Sinv, H - 1j * Sdot )
    return T, V, Strue, Sdot, Heff
