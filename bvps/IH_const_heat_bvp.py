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
import matplotlib
matplotlib.use('Agg')
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

FIELDS = {  'T1_IVP': 'T1',
            'T1_z_IVP': 'T1_z',
            'ln_rho1_IVP': 'ln_rho1',
            'w_IVP': 'w', 
            'w_z_IVP': 'w_z', 
            'VF': 'viscous_flux_z', 
            'VH': 'R_visc_heat',
            'L_visc': 'L_visc_w', 
            'R_visc': 'R_visc_w', 
            'UdotGrad_w': 'UdotGrad(w, w_z)', 
            'vel_rms_IVP': 'vel_rms', 
            'PE_flux': 'PE_flux_z',
            'Div_u_IVP': 'Div_u'  
        }
VARS   = {  'T1_IVP': 'T1', 
            'T1_z_IVP': 'T1_z', 
            'ln_rho1_IVP': 'ln_rho1'
        }

class IH_BVP_solver:

    def __init__(self, nz, flow, comm, solver, bvp_time):
        """
        
        nz   - the vertical resolution of the IVP
        flow - a dedalus.extras.flow_tools.GlobalFlowProperty for the IVP solver
        comm - An MPI comm object for the IVP solver
        solver - The IVP solver
        bvp_time - How often to perform a BVP, in sim time units
        """
        self.flow = flow
        self.solver = solver
        self.nz   = nz
        self.bvp_time = bvp_time
        self.profiles_dict = dict()
        self.solver_states = dict()
        for st in VARS.keys():
            self.solver_states[st] = self.solver.state[VARS[st]]
        for fd in FIELDS.keys():
            self.flow.add_property('plane_avg({})'.format(FIELDS[fd]), name='{}_avg'.format(fd))
            self.profiles_dict[fd] = np.zeros(nz)
        self.avg_time_elapsed = 0.
        self.avg_time_start   = 0.
        self.avg_started      = False
       
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.n_per_proc = nz/self.size

    def get_full_profile(self, prof_name):
        local = np.zeros(self.nz)
        glob  = np.zeros(self.nz)
        local[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.flow.properties['{}_avg'.format(prof_name)]['g'][0,:]
        self.comm.Allreduce(local, glob, op=MPI.SUM)
        return glob

    def update_avgs(self, dt, min_Re = 1):
        """
        dt - The size of the current timestep taken.
        """
        if self.flow.grid_average('Re') > min_Re:
            self.avg_time_elapsed += dt
            for fd in FIELDS.keys():
                self.profiles_dict[fd] += dt*self.get_full_profile(fd)
            if not self.avg_started:
                self.avg_started=True
                self.avg_time_start = self.solver.sim_time

    def check_if_solve(self):
        return self.avg_started*((self.solver.sim_time - self.avg_time_start) >= self.bvp_time)
    
    def solve_BVP(self, Ra=460, Pr=1, epsilon=1e-4, n_rho=3, r=2, nz=256, use_therm=True):
        for k in FIELDS.keys():
            self.profiles_dict[k] /= self.avg_time_elapsed
        self.avg_time_elapsed=0.
        self.avg_time_start = self.solver.sim_time

        atmosphere = FC_ConstHeating_2d_kappa_mu(nz=nz, constant_kappa=True, constant_mu=True, 
                                                 epsilon=epsilon, gamma=5./3, n_rho_cz=n_rho,
                                                 r=r, dimensions=1, comm=MPI.COMM_SELF)
        atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z', 'rho1'])#, 'int_rho1'])
        old_vars = ['u', 'w', 'ln_rho1', 'v', 'u_z', 'w_z', 'v_z', 'dx(A)']
        for sub in old_vars:
            atmosphere.problem.substitutions[sub] = '0'
        atmosphere._set_diffusivities(Rayleigh=Ra, Prandtl=Pr)
        atmosphere._set_parameters()
        atmosphere._set_subs()

        for k in FIELDS.keys():
            f = atmosphere._new_ncc()
            f.set_scales(self.nz / nz, keep_data=True)
            f['g'] = self.profiles_dict[k]
            atmosphere.problem.parameters[k] = f
        
        atmosphere.problem.substitutions['T1_zz_IVP'] = 'dz(T1_z_IVP)'
        atmosphere.problem.substitutions['rho1_IVP']  = 'rho0*(exp(ln_rho1_IVP) - 1)'
        atmosphere.problem.substitutions['rho1_z_IVP'] = 'dz(rho1_IVP)'
        atmosphere.problem.substitutions['rho0_z'] = 'dz(rho0)'

        if use_therm:
            atmosphere.problem.substitutions['rho0_tot'] = '(rho1_IVP + rho0)'
            atmosphere.problem.substitutions['T0_tot'] = '(T0 + T1_IVP)'
            atmosphere.problem.substitutions['T0_z_tot'] = '(T0_z + T1_z_IVP)'
            atmosphere.problem.substitutions['T0_zz_tot'] = '(T0_zz + T1_zz_IVP)'
        else:
            atmosphere.problem.substitutions['rho0_tot'] = '(rho0)'
            atmosphere.problem.substitutions['T0_tot'] = '(T0)'
            atmosphere.problem.substitutions['T0_z_tot'] = '(T0_z)'
            atmosphere.problem.substitutions['T0_zz_tot'] = '(T0_zz)'


        #Thermal subs
        atmosphere.problem.substitutions['L_T_advec'] = 'Cv * w_IVP * ( rho1 * T0_z_tot + rho0_tot * T1_z )'
        atmosphere.problem.substitutions['R_T_advec'] = 'Cv * w_IVP * ( rho0_tot * T0_z_tot + rho1 * T1_z )'
        atmosphere.problem.substitutions['L_PdV']     = 'Cv * (gamma - 1) * Div_u_IVP * (rho1 * T0_tot + rho0_tot * T1)'
        atmosphere.problem.substitutions['R_PdV']     = 'Cv * (gamma - 1) * Div_u_IVP * (rho0_tot * T0_tot + rho1 * T1)'
        atmosphere.problem.substitutions['L_kappa']   = '(-κ * dz(T1_z))'
        if use_therm:
            atmosphere.problem.substitutions['R_kappa'] = '(-κ * dz(T1_z_IVP))'
        else:
            atmosphere.problem.substitutions['R_kappa'] = '0'
        atmosphere.problem.substitutions['L_VH'] = 'rho1 * Cv * VH'
        atmosphere.problem.substitutions['R_VH'] = 'rho0_tot * Cv * VH'
        


        #Momentum eqn substitutions
        atmosphere.problem.substitutions['L_rho_gradT'] = '(rho0 * T1_z + rho1 * T0_z)'
        atmosphere.problem.substitutions['R_rho_gradT'] = '(rho1 * T1_z)'
        atmosphere.problem.substitutions['L_T_gradrho'] = '(T0 * dz(rho1) + T1 * dz(rho0))'
        atmosphere.problem.substitutions['R_T_gradrho'] = '(T1 * dz(rho1))'

        atmosphere.problem.substitutions['L_HSB'] = '(L_rho_gradT + L_T_gradrho + rho1*g)'
        atmosphere.problem.substitutions['R_NL_HSB'] = '(rho1*T1_z + T1*dz(rho1))'
        if use_therm:
            atmosphere.problem.substitutions['R_IVP_HSB'] = '(rho0*T1_z_IVP + T0 * dz(rho1_IVP) + rho1_IVP*T0_z + T1_IVP * dz(rho0) +'\
                                                            ' rho1_IVP * T1_z_IVP + T1_IVP * dz(rho1_IVP) + rho1_IVP*g )'
            atmosphere.problem.substitutions['L_IVP_HSB'] = '( rho1_IVP * T1_z + T1_IVP * dz(rho1) + rho1 * T1_z_IVP + T1 * dz(rho1_IVP)  )'
        else:
            atmosphere.problem.substitutions['R_IVP_HSB'] = '0'
            atmosphere.problem.substitutions['L_IVP_HSB'] = '0'
        atmosphere.problem.substitutions['visc_w'] = '(L_visc + R_visc)'

        logger.debug('setting base equations')
        atmosphere.problem.add_equation("dz(T1) - T1_z = 0")
        #atmosphere.problem.add_equation("dz(int_rho1) - rho1 = 0")

        logger.debug('setting energy equation')
        atmosphere.problem.add_equation(("L_T_advec + L_PdV  + L_kappa - L_VH = "
                                         "-R_T_advec - R_PdV - R_kappa + R_VH ") )

        logger.debug("Setting z-momentum equation")
        atmosphere.problem.add_equation((" rho1 * UdotGrad_w + L_HSB + L_IVP_HSB - rho1*visc_w = "
                                         " - R_IVP_HSB - R_NL_HSB + rho0_tot*(visc_w - UdotGrad_w)"))

        atmosphere.problem.add_bc('left(T1_z) = 0') 
        atmosphere.problem.add_bc("right(T1) = 0")
        atmosphere.problem.add_bc('left(rho1) - right(rho1) = 0') 
#        atmosphere.problem.add_bc('right(int_rho1) = 0') 

        solver = atmosphere.problem.build_solver()


        # Iterations
        tolerance=1e-3*epsilon
        pert = solver.perturbations.data
        pert.fill(1+tolerance)
        count = 0
        while np.sum(np.abs(pert)) > tolerance:
            count += 1
            
            solver.newton_iteration()
            logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))

        T1 = solver.state['T1']
        T1_z = solver.state['T1_z']
        rho1 = solver.state['rho1']

        fields = [('T1', T1), ('T1_z', T1_z), ('rho1', rho1)]
        for nm, f in fields:
            f.set_scales(1, keep_data=True)
            plt.plot(atmosphere.z, f['g'])
            plt.savefig('{}_plot.png'.format(nm))
            plt.close()

        return_dict = dict()
        for v in VARS.keys():
            return_dict[v] = np.zeros(self.nz)

        if use_therm:
            f = atmosphere._new_ncc()
            T1.set_scales(1, keep_data=True)
            f['g'] = T1['g']
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_IVP'] = f['g'] + self.profiles_dict['T1_IVP']

            f.set_scales(1, keep_data=False)
            T1_z.set_scales(1, keep_data=True)
            f['g'] = T1_z['g']
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_z_IVP'] = f['g'] + self.profiles_dict['T1_z_IVP']

            f.set_scales(1, keep_data=False)
            rho1.set_scales(1, keep_data=True)
            f['g'] = rho1['g']
            f.set_scales(self.nz/nz, keep_data=True)
            atmosphere.rho0.set_scales(self.nz/nz, keep_data=True)
            f['g'] += atmosphere.rho0['g']*np.exp(self.profiles_dict['ln_rho1_IVP'])
            f.set_scales(self.nz/nz, keep_data=True)
            atmosphere.rho0.set_scales(self.nz/nz, keep_data=True)
            return_dict['ln_rho1_IVP'] = np.log(f['g']) - np.log(atmosphere.rho0['g'])
        else:
            f = atmosphere._new_ncc()
            T1.set_scales(1, keep_data=True)
            f['g'] = T1['g']
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_IVP'] = f['g']

            f.set_scales(1, keep_data=False)
            T1_z.set_scales(1, keep_data=True)
            f['g'] = T1_z['g']
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_z_IVP'] = f['g']

            f.set_scales(1, keep_data=False)
            rho1.set_scales(1, keep_data=True)
            atmosphere.rho0.set_scales(1, keep_data=True)
            f['g'] = atmosphere.rho0['g'] + rho1['g']
            f.set_scales(self.nz/nz, keep_data=True)
            atmosphere.rho0.set_scales(self.nz/nz, keep_data=True)
            return_dict['ln_rho1_IVP'] = np.log(f['g']) - np.log(atmosphere.rho0['g'])

        for k in return_dict.keys():
            f = atmosphere._new_ncc()
            f['g'] = atmosphere.z
            f.set_scales(self.nz/nz, keep_data=True)
            plt.plot(f['g'], return_dict[k] - self.profiles_dict[k])
            plt.savefig('{}_plot.png'.format(k))
            plt.close()



        for v in VARS.keys():
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] += (return_dict[v] - self.profiles_dict[v])
