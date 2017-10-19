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


Ra = 460
Pr = 1
epsilon = 1e-4
n_rho = 3
r = 2

nz = 128

def solve_BVP(field_dict, Ra=460, Pr=1, epsilon=1e-4, n_rho=3, r=2, nz=128):
    
    T1_in = field_dict['T1']
    ln_rho1_in = field_dict['ln_rho1']
    w_in    = field_dict['w']

    atmosphere = FC_ConstHeating_2d_kappa_mu(nz=nz, constant_kappa=True, constant_mu=True, 
                                             epsilon=epsilon, gamma=5./3, n_rho_cz=n_rho,
                                             r=r, dimensions=1, comm=MPI.COMM_SELF)
    atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z', 'rho1', 'int_rho1'])
    atmosphere.problem.substitutions['u'] = '0'
    atmosphere.problem.substitutions['w'] = '0'
    atmosphere.problem.substitutions['ln_rho1'] = '0'
    atmosphere.problem.substitutions['v'] = '0'
    atmosphere.problem.substitutions['u_z'] = '0'
    atmosphere.problem.substitutions['w_z'] = '0'
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


    T1_ev = atmosphere._new_ncc()
    T1_z_ev = atmosphere._new_ncc()
    T1_zz_ev = atmosphere._new_ncc()
    T1_ev['g'] = T1_in
    T1_ev.differentiate('z', out=T1_z_ev)
    T1_z_ev.differentiate('z', out=T1_zz_ev)

    w_ev = atmosphere._new_ncc()
    w_ev['g'] = w_in
    atmosphere.problem.parameters['w_ev'] = w_ev

    udotgradw = atmosphere._new_ncc()
    udotgradw['g'] = T1_in = field_dict['UdotGrad(w, w_z)']
    atmosphere.problem.parameters['udotgradw'] = udotgradw



    #Add rho, T from evolution as NCCs
    atmosphere.problem.parameters['rho_ev'] = rho_ev
    atmosphere.problem.parameters['ln_rho_ev'] = ln_rho_ev
    atmosphere.problem.parameters['del_lnrho_ev'] = del_ln_rho_ev

    atmosphere.problem.parameters['T1_ev']  = T1_ev
    atmosphere.problem.parameters['T1_z_ev']  = T1_z_ev
    atmosphere.problem.parameters['T1_zz_ev']  = T1_zz_ev


    L_visc = atmosphere._new_ncc()
    R_visc = atmosphere._new_ncc()
    L_visc['g'] = field_dict['L_visc_w']
    R_visc['g'] = field_dict['R_visc_w']
    atmosphere.problem.parameters['R_visc_w_ev']  = R_visc
    atmosphere.problem.parameters['L_visc_w_ev']  = L_visc

    atmosphere.problem.substitutions['rho0_tot'] = '(rho_ev + rho0)'
    atmosphere.problem.substitutions['T0_tot'] = '(T0 + T1_ev)'
    atmosphere.problem.substitutions['T0_z_tot'] = '(T0_z + T1_z_ev)'
    atmosphere.problem.substitutions['T0_zz_tot'] = '(T0_zz + T1_zz_ev)'
    atmosphere.problem.substitutions['visc_w_ev'] = '(L_visc_w_ev + R_visc_w_ev)'

    vel_rms = atmosphere._new_ncc()
    vel_rms['g'] = field_dict['vel_rms']
    atmosphere.problem.parameters['vel_rms_ev'] = vel_rms

    visc_flux = atmosphere._new_ncc()
    visc_flux['g'] = field_dict['viscous_flux_z']
    atmosphere.problem.parameters['VF_ev'] = visc_flux

    atmosphere.problem.substitutions['PE_flux'] = '0' 

    #Thermal subs
    atmosphere.problem.substitutions['L_enth_flux'] = '(rho0_tot * w_ev * (Cv + 1) * T1 + rho1 * w_ev * (Cv + 1) * T0_tot )'
    atmosphere.problem.substitutions['R_enth_flux'] = '(rho1 * w_ev * (Cv + 1))'
    atmosphere.problem.substitutions['L_KE_flux']   = '(rho1 * w_ev * (vel_rms_ev)**2)/2'
    atmosphere.problem.substitutions['R_KE_flux']   = '(rho0_tot * w_ev * (vel_rms_ev)**2)/2'

    #Momentum eqn substitutions
    atmosphere.problem.substitutions['L_rho_gradT'] = '(rho0_tot * T1_z + rho1 * T0_z_tot)'
    atmosphere.problem.substitutions['R_rho_gradT'] = '(rho0_tot * T0_z_tot + rho1 * T1_z)'
    atmosphere.problem.substitutions['L_T_gradrho'] = '(T0_tot * dz(rho1) + T1 * dz(rho0_tot))'
    atmosphere.problem.substitutions['R_T_gradrho'] = '(T0_tot * dz(rho0_tot) + T1 * dz(rho1))'

    logger.debug('setting base equations')
    atmosphere.problem.add_equation("dz(T1) - T1_z = 0")
    atmosphere.problem.add_equation("dz(int_rho1) - rho1 = 0")

    logger.debug('setting energy equation')
    atmosphere.problem.add_equation(("-κ * dz(T1_z) + dz(L_enth_flux  + L_KE_flux)  = κ * (T0_zz_tot + IH) - dz(R_enth_flux + R_KE_flux + VF_ev) "))

    logger.debug("Setting z-momentum equation")
    atmosphere.problem.add_equation((" rho1 * (udotgradw - visc_w_ev + g) + L_rho_gradT + L_T_gradrho = -rho0_tot * (udotgradw - visc_w_ev + g) - R_rho_gradT - R_T_gradrho"))

    atmosphere.problem.add_bc('left(T1_z) = 0') 
    atmosphere.problem.add_bc("right(T1) = 0")
    atmosphere.problem.add_bc('left(int_rho1) = 0') 
    atmosphere.problem.add_bc('right(int_rho1) = 0') 

    solver = atmosphere.problem.build_solver()


    # Iterations
    tolerance=1e-3*epsilon
    pert = solver.perturbations.data
    pert.fill(epsilon)
    count = 0
    while np.sum(np.abs(pert)) > tolerance:
        count += 1
        
        solver.newton_iteration()
        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))

    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    rho1 = solver.state['rho1']

    T1.set_scales(1, keep_data=True)
    T1_z.set_scales(1, keep_data=True)
    T1_ev.set_scales(1, keep_data=True)
    T1_z_ev.set_scales(1, keep_data=True)
    rho1.set_scales(1, keep_data=True)
    rho_ev.set_scales(1, keep_data=True)
    atmosphere.rho0.set_scales(1, keep_data=True)
    return_dict = dict()
    return_dict['T1'] = T1['g'] + T1_ev['g']
    return_dict['T1_z'] = T1_z['g'] + T1_z_ev['g']
    return_dict['ln_rho1'] = np.log(1 + (rho1['g'] + rho_ev['g'])/atmosphere.rho0['g'])
    return return_dict
