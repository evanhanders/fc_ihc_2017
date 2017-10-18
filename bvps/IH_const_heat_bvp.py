"""
Dedalus script for the Lane-Emden equation.

This is a 1D script and should be ran serially.  It should converge within
roughly a dozen iterations, and should take under a minute to run.

In astrophysics, the Lane–Emden equation is a dimensionless form of Poisson's
equation for the gravitational potential of a Newtonian self-gravitating,
spherically symmetric, polytropic fluid [1].

It is usually written as:
    dr(dr(f)) + (2/r)*dr(f) + f**n = 0
    f(r=0) = 1
    dr(f)(r=0) = 0
where n is the polytropic index, and the equation is solved over the interval
r=[0,R], where R is the n-dependent first zero of f(r).

Following [2], we rescale the equation by defining r=R*x:
    dx(dx(f)) + (2/x)*dx(f) + (R**2)*(f**n) = 0
    f(x=0) = 1
    dx(f)(x=0) = 0
    f(x=1) = 0
This is a nonlinear eigenvalue problem over the interval x=[0,1], with the
additional boundary condition fixing the eigenvalue R.

References:
    [1]: http://en.wikipedia.org/wiki/Lane–Emden_equation
    [2]: J. P. Boyd, "Chebyshev spectral methods and the Lane-Emden problem,"
         Numerical Mathematics Theory (2011).

"""

import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from stratified_dynamics.const_heat import FC_ConstHeating_2d_kappa_mu
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

import h5py

with h5py.File('bvps/profile_info.h5', 'r') as f:
    T1_file = f['T1'].value[0,:]
    ln_rho1_file = f['ln_rho1'].value[0,:]
    w_file = f['w'].value[0,:]


Ra = 460
Pr = 1
epsilon = 1e-4
n_rho = 3
r = 2

nz = 128

def solve_BVP(T1_in, ln_rho1_in, w_in, Ra=460, Pr=1, epsilon=1e-4, n_rho=3, r=2, nz=128):

    atmosphere = FC_ConstHeating_2d_kappa_mu(nz=nz, constant_kappa=True, constant_mu=True, 
                                             epsilon=epsilon, gamma=5./3, n_rho_cz=n_rho,
                                             r=r, dimensions=1, comm=MPI.COMM_SELF)
    atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'w', 'T1_z', 'w_z'])#, 'M1'])
    atmosphere.problem.substitutions['u'] = '0'
    atmosphere.problem.substitutions['ln_rho1'] = '0'
    atmosphere.problem.substitutions['v'] = '0'
    atmosphere.problem.substitutions['u_z'] = '0'
    atmosphere.problem.substitutions['v_z'] = '0'
    atmosphere.problem.substitutions['dx(A)'] = '0'
    atmosphere._set_diffusivities(Rayleigh=Ra, Prandtl=Pr)
    atmosphere._set_parameters()
    atmosphere._set_subs()

    ln_rho_ev = atmosphere._new_ncc()
    del_ln_rho_ev = atmosphere._new_ncc()
    rho_ev = atmosphere._new_ncc()
    ln_rho_ev.set_scales(len(ln_rho1_in)/nz, keep_data=True)
    ln_rho_ev['g'] = ln_rho1_in
    ln_rho_ev.differentiate('z', out=del_ln_rho_ev)
    ln_rho_ev.set_scales(1, keep_data=True)
    rho_ev['g'] = np.exp(ln_rho_ev['g'])

    #Set NCCs from file
    atmosphere.problem.substitutions['NL_continuity'] = '0'
    atmosphere.problem.substitutions['NL_w_momentum'] = '0'
    atmosphere.problem.substitutions['NL_energy'] = '0'
    atmosphere.problem.substitutions['dx2_T1'] = '0'
    atmosphere.problem.substitutions['dx2_w'] = '0'
    atmosphere.problem.substitutions['dx_u_z'] = '0'
    atmosphere.problem.substitutions['dx_u'] = '0'
    atmosphere.problem.substitutions['dx_u'] = '0'

    atmosphere.problem.parameters['rho_ev'] = rho_ev
    atmosphere.problem.parameters['ln_rho_ev'] = ln_rho_ev
    atmosphere.problem.parameters['del_lnrho_ev'] = del_ln_rho_ev
    #atmosphere.problem.substitutions['PE_flux'] = 'phi*rho_ev*rho0*w'
    atmosphere.problem.substitutions['PE_flux'] = '0' 

    atmosphere.problem.add_equation("dz(w) - w_z = 0")
    atmosphere.problem.add_equation("dz(T1) - T1_z = 0")

    atmosphere.problem.add_equation(("dz(-κ*T1_z + rho_ev*rho0*w*T0*(Cv + 1) + PE_flux) = dz(-(rho_ev*rho0*w**3)/2 - rho_ev*rho0*w*T1*(Cv+1) + μ * w * (4/3)*w_z)"))

    logger.debug("Setting z-momentum equation")
    atmosphere.problem.add_equation((" T1_z    + T1*(del_ln_rho0+del_lnrho_ev) "+\
                                    "-(μ/(rho0*rho_ev))*((4/3) * dz(w_z) + del_ln_μ * (4/3) * w_z) = -T0*del_lnrho_ev"))

    atmosphere.problem.add_bc('left(T1_z) = 0') 
    atmosphere.problem.add_bc('left(w) = 0') 
    atmosphere.problem.add_bc("right(T1) = 0")
    atmosphere.problem.add_bc('right(w) = 0') 

    solver = atmosphere.problem.build_solver()

    #Set initial conditions from file
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    w = solver.state['w']
    w_z = solver.state['w_z']

    T1.set_scales(len(ln_rho1_in)/nz, keep_data=True)
    w.set_scales(len(ln_rho1_in)/nz, keep_data=True)
    T1['g'] = T1_in
    T1.differentiate('z', out=T1_z)
    w['g'] = w_in
    w.differentiate('z', out=w_z)

    # Iterations
    tolerance=1e-3*epsilon
    pert = solver.perturbations.data
    pert.fill(epsilon)
    count = 0
    while np.sum(np.abs(pert)) > tolerance:
        count += 1
        
        solver.newton_iteration()
        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))

    
    T1.set_scales(1, keep_data=True)
    w.set_scales(1, keep_data=True)
    T1_z.set_scales(1, keep_data=True)
    w_z.set_scales(1, keep_data=True)
    return T1['g'], w['g'], T1_z['g'], w_z['g']

#    fig = plt.figure()
#    ax = fig.add_subplot(3,1,1)
#    atmosphere.T0.set_scales(1, keep_data=True)
#    T1.set_scales(1, keep_data=True)
#    T = atmosphere.T0['g'] + T1['g']
#
#    plt.plot(atmosphere.z, T1['g'])
#
#
#    f = atmosphere._new_ncc()
#    f.set_scales(len(ln_rho1_in)/nz, keep_data=True)
#    f['g'] = T1_in
#    f.set_scales(1, keep_data=True)
#    plt.plot(atmosphere.z, f['g'])
#    diff_w = np.abs((T1['g'] - f['g'])/T1['g'])
#    bx = fig.add_subplot(3,1,2)
#    w.set_scales(1, keep_data=True)
#    plt.plot(atmosphere.z, w['g'])
#    diff_w = np.abs((w['g'] - f['g'])/w['g'])
#
#
#    f.set_scales(len(ln_rho1_in)/nz, keep_data=True)
#    f['g'] = w_in
#    f.set_scales(1, keep_data=True)
#    plt.plot(atmosphere.z, f['g'])
#
#    cx = fig.add_subplot(3,1,3)
#    plt.plot(atmosphere.z, diff_w)
#    plt.show()

if __name__ == '__main__':
    solve_BVP(T1_file, ln_rho1_file, w1_file)
