from collections import OrderedDict
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

class IH_BVP_solver:
    
    FIELDS = OrderedDict([  ('T1_IVP', 'T1'),
                            ('T1_z_IVP', 'T1_z'),
                            ('ln_rho1_IVP', 'ln_rho1'),
                            ('w_IVP', 'w'), 
                            ('EF_IVP', 'h_flux_z'), 
                            ('VF_IVP', 'viscous_flux_z'), 
                            ('KEF_IVP', 'KE_flux_z'),
                            ('L_visc', 'L_visc_w'), 
                            ('R_visc', 'R_visc_w'), 
                            ('UdotGrad_w', 'UdotGrad(w, w_z)'), 
                            ('rho_UdotGrad_w', '(rho_full * UdotGrad(w, w_z))'), 
                            ('KEF_norho', '(w*(vel_rms**2)/2)'), 
                            ('rho_w_IVP', '(rho_full * w)'),
                            ('T_w_IVP', '(T_full * w)'),
                            ('rho_gradT_IVP', '(rho_full*dz(T_full))'),
                            ('rho_full_IVP', '(rho_full)'),
                            ('T_gradrho_IVP', '(T_full * dz(rho_full))'),
                            ('dz_rho_full_IVP', 'dz(rho_full)')
                        ])
    VARS   = OrderedDict([  ('T1_IVP', 'T1'),
                            ('T1_z_IVP', 'T1_z'), 
                            ('ln_rho1_IVP', 'ln_rho1')
                        ])



    def __init__(self, nz, flow, comm, solver, bvp_time, num_bvps, bvp_equil_time):
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
        self.num_bvps = num_bvps
        self.completed_bvps = 0
        self.profiles_dict = dict()
        self.solver_states = dict()
        for st in IH_BVP_solver.VARS.keys():
            self.solver_states[st] = self.solver.state[IH_BVP_solver.VARS[st]]
        for fd in IH_BVP_solver.FIELDS.keys():
            self.flow.add_property('plane_avg({})'.format(IH_BVP_solver.FIELDS[fd]), name='{}_avg'.format(fd))
            self.profiles_dict[fd] = np.zeros(nz)
        self.avg_time_elapsed = 0.
        self.avg_time_start   = 0.
        self.bvp_equil_time   = bvp_equil_time
        self.avg_started      = False
       
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.n_per_proc = self.nz/self.size

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
        if self.completed_bvps >= self.num_bvps:
            return
        if self.flow.grid_average('Re') > min_Re:
            if (self.solver.sim_time - self.avg_time_start) < self.bvp_equil_time:
                return
            self.avg_time_elapsed += dt
            for fd in IH_BVP_solver.FIELDS.keys():
                self.profiles_dict[fd] += dt*self.get_full_profile(fd)
            if not self.avg_started:
                self.avg_started=True
                self.avg_time_start = self.solver.sim_time

    def check_if_solve(self):
        return self.avg_started*((self.solver.sim_time - self.avg_time_start - self.bvp_equil_time) >= self.bvp_time)*(self.completed_bvps < self.num_bvps)

    def _get_evolved_ln_rho(self, atmosphere, nz_atmo, nz_IVP):
        #Calculate from continuity
        grad_ln_rho = atmosphere._new_ncc()
        grad_ln_rho.set_scales(nz_IVP/nz_atmo, keep_data=False)
        grad_ln_rho['g'] = -(1/self.profiles_dict['w_IVP'])\
                       *(self.profiles_dict['Div_u_IVP'] + self.profiles_dict['u_dx_ln_rho_IVP'])
        
        #Seperate out fluctuating part
        grad_ln_rho1 = atmosphere._new_ncc()
        grad_ln_rho.set_scales(1, keep_data=True)
        atmosphere.del_ln_rho0.set_scales(1, keep_data=True)
        grad_ln_rho1['g'] = grad_ln_rho['g'] - atmosphere.del_ln_rho0['g']

        #Integrate fluctuating part
        ln_rho1 = atmosphere._new_ncc()
        grad_ln_rho1.antidifferentiate('z', ('right', 0), out=ln_rho1)

        return grad_ln_rho1, ln_rho1

    
    def solve_BVP(self, Ra=460, Pr=1, epsilon=1e-4, n_rho=3, r=2, nz=256, use_therm=True):
        for k in IH_BVP_solver.FIELDS.keys():
            self.profiles_dict[k] /= self.avg_time_elapsed
        self.avg_time_elapsed=0.
        self.avg_time_start = self.solver.sim_time
        self.completed_bvps += 1

        if self.rank == 0:
            atmosphere = FC_ConstHeating_2d_kappa_mu(nz=nz, constant_kappa=True, constant_mu=True, 
                                                     epsilon=epsilon, gamma=5./3, n_rho_cz=n_rho,
                                                     r=r, dimensions=1, comm=MPI.COMM_SELF)
            atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z', 'rho1', 'M1'])
            old_vars = ['u', 'w', 'ln_rho1', 'v', 'u_z', 'w_z', 'v_z', 'dx(A)']
            for sub in old_vars:
                atmosphere.problem.substitutions[sub] = '0'
            atmosphere._set_diffusivities(Rayleigh=Ra, Prandtl=Pr)
            atmosphere._set_parameters()
            atmosphere._set_subs()

            for k in IH_BVP_solver.FIELDS.keys():
                f = atmosphere._new_ncc()
                f.set_scales(self.nz / nz, keep_data=True)
                f['g'] = self.profiles_dict[k]
                atmosphere.problem.parameters[k] = f
            

            atmosphere.problem.substitutions['rho0_tot'] = 'rho_full_IVP'
            atmosphere.problem.substitutions['rho0_z_tot'] = 'dz_rho_full_IVP'
            atmosphere.problem.substitutions['T0_tot'] = '(T0 + T1_IVP)'
            atmosphere.problem.substitutions['T0_z_tot'] = '(T0_z + T1_z_IVP)'
            atmosphere.problem.substitutions['T1_zz_IVP'] = 'dz(T1_z_IVP)'
            atmosphere.problem.substitutions['T0_zz_tot'] = '(T0_zz + T1_zz_IVP)'
            
            # Enthalpy flux = w * (Cv + 1) * T
            atmosphere.problem.substitutions['enth_flux_L'] = '((Cv+1)*( rho_w_IVP*T1 + T_w_IVP*rho1  ) )'
            atmosphere.problem.substitutions['enth_flux_R'] = '( EF_IVP + rho1*T1*w_IVP*(Cv+1) )'
            # KE flux = w * (vel_rms)^2 / 2
            atmosphere.problem.substitutions['KE_flux_L']   = '( rho1 * KEF_norho )'
            atmosphere.problem.substitutions['KE_flux_R']   = '( KEF_IVP )'
            # Viscous flux -- double check
            atmosphere.problem.substitutions['visc_flux_R'] = '(VF_IVP)'
            # Conductive flux
            atmosphere.problem.substitutions['kappa_flux_L'] = '(-(κ) * T1_z)' 
            atmosphere.problem.substitutions['kappa_flux_R'] = '(-(κ) * T0_z_tot)' 

            atmosphere.problem.substitutions['flux_L'] = '(kappa_flux_L + enth_flux_L + KE_flux_L)'
            atmosphere.problem.substitutions['flux_R'] = '(kappa_flux_R + enth_flux_R + KE_flux_R + visc_flux_R)'


            #Momentum eqn substitutions
            atmosphere.problem.substitutions['L_rho_gradT'] = '(rho0_tot * T1_z + rho1 * T0_z_tot)'
            atmosphere.problem.substitutions['R_rho_gradT'] = '(rho1 * T1_z + rho_gradT_IVP)'
            atmosphere.problem.substitutions['L_T_gradrho'] = '(T0_tot * dz(rho1) + T1 * rho0_z_tot)'
            atmosphere.problem.substitutions['R_T_gradrho'] = '(T1 * dz(rho1) + T_gradrho_IVP )'

            atmosphere.problem.substitutions['L_HSB'] = '(L_rho_gradT + L_T_gradrho + rho1*g)'
            atmosphere.problem.substitutions['R_HSB'] = '(R_rho_gradT + R_T_gradrho + rho0_tot*g)'
            atmosphere.problem.substitutions['visc_w'] = '(L_visc + R_visc)'


            logger.debug('setting equations')
            atmosphere.problem.add_equation("dz(T1) - T1_z = 0")
            atmosphere.problem.add_equation("dz(M1) - rho1 = 0")
            atmosphere.problem.add_equation(("dz(flux_L) = -dz(flux_R) + κ*IH"))

            logger.debug("Setting z-momentum equation")
            atmosphere.problem.add_equation((" rho1 * UdotGrad_w + L_HSB - rho1*visc_w = "
                                             " - R_HSB + rho0_tot*visc_w - rho_UdotGrad_w"))

            atmosphere.problem.add_bc("left(T1_z) = 0") 
            atmosphere.problem.add_bc("right(T1) = 0")
            atmosphere.problem.add_bc("left(M1) = 0")
            atmosphere.problem.add_bc("right(M1) = 0")


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
            fields = [('T1', T1), ('T1_z', T1_z), ('lho1', rho1)]
            for nm, f in fields:
                f.set_scales(1, keep_data=True)
                plt.plot(atmosphere.z, f['g'])
                plt.savefig('{}_plot.png'.format(nm))
                plt.close()

        return_dict = dict()
        for v in IH_BVP_solver.VARS.keys():
            return_dict[v] = np.zeros(self.nz)

        if self.rank == 0:
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


            f.set_scales(self.nz/nz, keep_data=False)
            rho1.set_scales(self.nz/nz, keep_data=True)
            atmosphere.rho0.set_scales(self.nz/nz, keep_data=True)
            f['g'] = atmosphere.rho0['g']*(np.exp(self.profiles_dict['ln_rho1_IVP'])) + rho1['g']
            f['g'] = np.log(f['g']) - np.log(atmosphere.rho0['g'])
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['ln_rho1_IVP'] = f['g']
    
            for k in return_dict.keys():
                f = atmosphere._new_ncc()
                f['g'] = atmosphere.z
                f.set_scales(self.nz/nz, keep_data=True)
                plt.plot(f['g'], return_dict[k] - self.profiles_dict[k])
                plt.savefig('{}_plot.png'.format(k))
                plt.close()
        
        self.comm.Barrier()
        if self.size > 1:
            for v in IH_BVP_solver.VARS.keys():
                glob = np.zeros(self.nz)
                self.comm.Allreduce(return_dict[v], glob, op=MPI.SUM)
                return_dict[v] = glob

        for v in IH_BVP_solver.VARS.keys():
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] += (return_dict[v] - self.profiles_dict[v])[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)]

        for fd in IH_BVP_solver.FIELDS.keys():
            self.profiles_dict[fd] = np.zeros(self.nz)
