from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dedalus import public as de

from stratified_dynamics.const_heat import FC_ConstHeating_2d_kappa_mu


class IH_BVP_solver:
    """
    A class for solving a BVP in the middle of a running IVP.

    This class solves equations of mass conservation, energy conservation,
    and hydrostatic equilibrium for an internally heated atmosphere which has
    fixed T boundary conditions at the top and fixed Tz boundary conditions at the bottom.

    Objects of this class are paired with a dedalus solver which is timestepping forward
    through an IVP.  This class calculates horizontal and time averages of important flow
    fields from that IVP, then uses those as NCCs in a BVP to get a more evolved thermal state

    CLASS VARIABLES
    ---------------
        FIELDS - An OrderedDict of strings of which the time- and horizontally- averaged 
                 profiles are tracked (and fed into the BVP)
        VARS   - An OrderedDict of variables which will be updated by the BVP

    Object Attributes:
    ------------------
        avg_started         - If True, time averages for FIELDS has begun
        avg_time_elapsed    - Amount of IVP simulation time over which averages have been taken so far
        avg_time_start      - Simulation time at which average began
        bvp_equil_time      - Amount of sim time to wait for velocities to converge before starting averages
                                at the beginning of IVP or after a BVP is solved
        bvp_time            - Length of sim time to average over before doing bvp
        comm                - COMM_WORLD for IVP
        completed_bvps      - # of BVPs that have been completed during this run
        flow                - A dedalus flow_tools.GlobalFlowProperty object for the IVP solver which is tracking
                                the Reynolds number, and will track FIELDS variables
        n_per_proc          - Number of z-points per core (for parallelization)
        num_bvps            - Total number of BVPs to complete
        nz                  - z-resolution of the IVP grid
        profiles_dict       - a dictionary containing the time/horizontal average of FIELDS
        rank                - comm rank
        size                - comm size
        solver              - The corresponding dedalus IVP solver object
        solver_states       - The states of VARS in solver

    """
    
    FIELDS = OrderedDict([  ('T1_IVP',              'T1'),
                            ('T1_z_IVP',            'T1_z'),
                            ('ln_rho1_IVP',         'ln_rho1'),
                            ('w_IVP',               'w'), 
                            ('EF_IVP',              'h_flux_z'), 
                            ('VF_IVP',              'viscous_flux_z'), 
                            ('KEF_IVP',             'KE_flux_z'),
                            ('visc_IVP',            '(rho_full*(R_visc_w + L_visc_w))'), 
                            ('UdotGrad_w',          'UdotGrad(w, w_z)'), 
                            ('rho_UdotGrad_w',      '(rho_full * UdotGrad(w, w_z))'), 
                            ('KEF_norho',           '(w*(vel_rms**2)/2)'), 
                            ('rho_w_IVP',           '(rho_full * w)'),
                            ('T_w_IVP',             '(T_full * w)'),
                            ('rho_gradT_IVP',       '(rho_full*dz(T_full))'),
                            ('rho_full_IVP',        '(rho_full)'),
                            ('T_gradrho_IVP',       '(T_full * dz(rho_full))'),
                            ('dz_rho_full_IVP',     'dz(rho_full)')
                        ])
    VARS   = OrderedDict([  ('T1_IVP',              'T1'),
                            ('T1_z_IVP',            'T1_z'), 
                            ('ln_rho1_IVP',         'ln_rho1')
                        ])



    def __init__(self, nz, flow, comm, solver, bvp_time, num_bvps, bvp_equil_time):
        """
        Initializes the object; grabs solver states and makes room for profile averages
        
        Arguments:
        nz              - the vertical resolution of the IVP
        flow            - a dedalus.extras.flow_tools.GlobalFlowProperty for the IVP solver
        comm            - An MPI comm object for the IVP solver
        solver          - The IVP solver
        bvp_time        - How often to perform a BVP, in sim time units
        num_bvps        - Maximum number of BVPs to solve
        bvp_equil_time  - Sim time to wait after a bvp before beginning averages for the next one
        """
        #Get info about IVP
        self.flow       = flow
        self.solver     = solver
        self.nz         = nz

        #Specify how BVPs work
        self.bvp_time           = bvp_time
        self.num_bvps           = num_bvps
        self.completed_bvps     = 0
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = 0.
        self.bvp_equil_time     = bvp_equil_time
        self.avg_started        = False

        #Get info about MPI distribution
        self.comm           = comm
        self.rank           = comm.rank
        self.size           = comm.size
        self.n_per_proc     = self.nz/self.size

        # Set up tracking dictionaries for flow fields
        self.profiles_dict = dict()
        self.solver_states = dict()
        for st in IH_BVP_solver.VARS.keys():
            self.solver_states[st] = self.solver.state[IH_BVP_solver.VARS[st]]
        for fd in IH_BVP_solver.FIELDS.keys():
            self.flow.add_property('plane_avg({})'.format(IH_BVP_solver.FIELDS[fd]), name='{}_avg'.format(fd))
            self.profiles_dict[fd] = np.zeros(nz)
       

    def get_full_profile(self, prof_name):
        """
        Given a profile name, which is a key to the class FIELDS dictionary, communicate the
        full vertical profile across all processes, then return the full profile as a function
        of depth.

        Arguments:
            prof_name       - A string, which is a key to the class FIELDS dictionary
        """
        local = np.zeros(self.nz)
        glob  = np.zeros(self.nz)
        local[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.flow.properties['{}_avg'.format(prof_name)]['g'][0,:]
        self.comm.Allreduce(local, glob, op=MPI.SUM)
        return glob

    def update_avgs(self, dt, min_Re = 1):
        """
        If proper conditions are met, this function adds the time-weighted vertical profile
        of all profiles in FIELDS to the appropriate arrays which are tracking classes. The
        size of the timestep is also recorded.

        The averages taken by this class are time-weighted averages of horizontal averages, such
        that sum(dt * Profile) / sum(dt) = <time averaged profile used for BVP>

        Arguments:
            dt          - The size of the current timestep taken.
            min_Re      - Only count this timestep toward the average if vol_avg(Re) is greater than this.
        """
        #Don't average if all BVPs are done
        if self.completed_bvps >= self.num_bvps:
            return

        if self.flow.grid_average('Re') > min_Re:
            # Don't count point if a BVP has been completed very recently
            if (self.solver.sim_time - self.avg_time_start) < self.bvp_equil_time:
                return

            #Update sums for averages
            self.avg_time_elapsed += dt
            for fd in IH_BVP_solver.FIELDS.keys():
                self.profiles_dict[fd] += dt*self.get_full_profile(fd)
            if not self.avg_started:
                self.avg_started=True
                self.avg_time_start = self.solver.sim_time

    def check_if_solve(self):
        """ Returns a boolean.  If True, it's time to solve a BVP """
        return self.avg_started*(self.avg_time_elapsed >= self.bvp_time)*(self.completed_bvps < self.num_bvps)

    def solve_BVP(self, Ra=460, Pr=1, epsilon=1e-4, n_rho=3, r=2, nz=256):
        """
        Solves a BVP in a 2D FC_ConstHeating atmosphere under the kappa/mu formulation of the equations.
        Run parameters are specified, and should be similar to those of the bvp.

        The BVP calculates updated rho / temperature fields, then updates the solver states which are
        tracked in self.solver_states.  This automatically updates the IVP's fields.

        """
        # Turn profiles from time-weighted sums into appropriate averages.
        for k in IH_BVP_solver.FIELDS.keys():
            self.profiles_dict[k] /= self.avg_time_elapsed
        # Restart counters for next BVP
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = self.solver.sim_time
        self.completed_bvps     += 1

        # No need to waste processor power on multiple bvps, only do it on one
        if self.rank == 0:
            atmosphere = FC_ConstHeating_2d_kappa_mu(nz=nz, constant_kappa=True, constant_mu=True, 
                                                     epsilon=epsilon, gamma=5./3, n_rho_cz=n_rho,
                                                     r=r, dimensions=1, comm=MPI.COMM_SELF)
            #Variables are T, dz(T), rho, integrated mass
            atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z', 'rho1', 'M1'])

            #Zero out old varables to make atmospheric substitutions happy.
            old_vars = ['u', 'w', 'ln_rho1', 'v', 'u_z', 'w_z', 'v_z', 'dx(A)']
            for sub in old_vars:
                atmosphere.problem.substitutions[sub] = '0'

            atmosphere._set_diffusivities(Rayleigh=Ra, Prandtl=Pr)
            atmosphere._set_parameters()
            atmosphere._set_subs()

            #Add time and horizontally averaged profiles from IVP to the problem as parameters
            for k in IH_BVP_solver.FIELDS.keys():
                f = atmosphere._new_ncc()
                f.set_scales(self.nz / nz, keep_data=True) #If nz(bvp) =/= nz(ivp), this allows interaction between them
                f['g'] = self.profiles_dict[k]
                atmosphere.problem.parameters[k] = f
            

            # Full background thermodynamics from IVP
            atmosphere.problem.substitutions['rho0_tot'] = 'rho_full_IVP'
            atmosphere.problem.substitutions['rho0_z_tot'] = 'dz_rho_full_IVP'
            atmosphere.problem.substitutions['T0_tot'] = '(T0 + T1_IVP)'
            atmosphere.problem.substitutions['T0_z_tot'] = '(T0_z + T1_z_IVP)'
            atmosphere.problem.substitutions['T0_zz_tot'] = '(T0_zz + dz(T1_z_IVP))'
            
            #### Energy equation substitutions
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


            #### Modified hydrostatic balance substitutions
            # rho * grad(T)
            atmosphere.problem.substitutions['L_rho_gradT'] = '(rho0_tot * T1_z + rho1 * T0_z_tot)'
            atmosphere.problem.substitutions['R_rho_gradT'] = '(rho1 * T1_z + rho_gradT_IVP)'
            # T * grad(rho)
            atmosphere.problem.substitutions['L_T_gradrho'] = '(T0_tot * dz(rho1) + T1 * rho0_z_tot)'
            atmosphere.problem.substitutions['R_T_gradrho'] = '(T1 * dz(rho1) + T_gradrho_IVP )'

            # Full linear/nonlinear Hydrostatic balance
            atmosphere.problem.substitutions['L_HSB'] = '(L_rho_gradT + L_T_gradrho + rho1*g)'
            atmosphere.problem.substitutions['R_HSB'] = '(R_rho_gradT + R_T_gradrho + rho0_tot*g)'

            logger.debug('setting T1_z eqn')
            atmosphere.problem.add_equation("dz(T1) - T1_z = 0")

            logger.debug('setting M1 eqn')
            atmosphere.problem.add_equation("dz(M1) - rho1 = 0")

            logger.debug('Setting energy equation')
            atmosphere.problem.add_equation(("dz(flux_L) = -dz(flux_R) + κ*IH"))

            logger.debug("Setting modified hydrostatic equilibrium")
            atmosphere.problem.add_equation((" rho1 * UdotGrad_w + L_HSB = "
                                             " - R_HSB + visc_IVP - rho_UdotGrad_w"))

            # Use thermal BCs from IVP
            atmosphere.problem.add_bc("left(T1_z) = 0") 
            atmosphere.problem.add_bc("right(T1) = 0")

            # Conserve mass
            atmosphere.problem.add_bc("left(M1) = 0")
            atmosphere.problem.add_bc("right(M1) = 0")


            # Solve the BVP
            solver = atmosphere.problem.build_solver()

            tolerance=1e-3*epsilon
            pert = solver.perturbations.data
            pert.fill(1+tolerance)
            while np.sum(np.abs(pert)) > tolerance:
                solver.newton_iteration()
                logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))

            T1 = solver.state['T1']
            T1_z = solver.state['T1_z']
            rho1 = solver.state['rho1']

        # Create space for the returned profiles on all processes.
        return_dict = dict()
        for v in IH_BVP_solver.VARS.keys():
            return_dict[v] = np.zeros(self.nz)

        if self.rank == 0:
            #Appropriately adjust T1 in IVP
            f = atmosphere._new_ncc()
            T1.set_scales(1, keep_data=True)
            f['g'] = T1['g']
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_IVP'] = f['g'] + self.profiles_dict['T1_IVP']

            #Appropriately adjust T1_z in IVP
            f.set_scales(1, keep_data=False)
            T1_z.set_scales(1, keep_data=True)
            f['g'] = T1_z['g']
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_z_IVP'] = f['g'] + self.profiles_dict['T1_z_IVP']

            #Appropriately adjust ln_rho1 in IVP
            f.set_scales(self.nz/nz, keep_data=False)
            rho1.set_scales(self.nz/nz, keep_data=True)
            atmosphere.rho0.set_scales(self.nz/nz, keep_data=True)
            f['g'] = atmosphere.rho0['g']*(np.exp(self.profiles_dict['ln_rho1_IVP'])) + rho1['g']
            f['g'] = np.log(f['g']) - np.log(atmosphere.rho0['g'])
            f.set_scales(self.nz/nz, keep_data=True)
            return_dict['ln_rho1_IVP'] = f['g']
    
#            Plot out evolved profiles from BVP (debugging)
#            for k in return_dict.keys():
#                f = atmosphere._new_ncc()
#                f['g'] = atmosphere.z
#                f.set_scales(self.nz/nz, keep_data=True)
#                plt.plot(f['g'], return_dict[k] - self.profiles_dict[k])
#                plt.savefig('{}_plot.png'.format(k))
#                plt.close()
#        
        self.comm.Barrier()
        # Communicate output profiles from proc 0 to all others.
        if self.size > 1:
            for v in IH_BVP_solver.VARS.keys():
                glob = np.zeros(self.nz)
                self.comm.Allreduce(return_dict[v], glob, op=MPI.SUM)
                return_dict[v] = glob

        # Actually update IVP states
        for v in IH_BVP_solver.VARS.keys():
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] += (return_dict[v] - self.profiles_dict[v])[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)]

        # Reset profile arrays for getting the next bvp average
        for fd in IH_BVP_solver.FIELDS.keys():
            self.profiles_dict[fd] = np.zeros(self.nz)
