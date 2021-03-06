import numpy as np
import scipy.special as scp
import os
from mpi4py import MPI

from collections import OrderedDict

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tools import analysis
except:
    from sys import path
    path.insert(0, './tools')
    from ..tools import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de
from dedalus.core.field import Field

class Atmosphere:
    def __init__(self, verbose=False, fig_dir='./', dimensions=2, **kwargs):
        self._set_domain(**kwargs)
        
        self.make_plots = verbose
        self.fig_dir = fig_dir + '/'
        self.dimensions = dimensions
        
        if self.fig_dir[-1] != '/':
            self.fig_dir += '/'
        if self.domain.dist.comm_cart.rank == 0 and not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
            
    def evaluate_at_point(self, f, z=0):
        return f.interpolate(z=z)

    def value_at_boundary(self, field):
        orig_scale = field.meta[:]['scale']
        try:
            field_top    = self.evaluate_at_point(field, z=self.Lz)['g'][0][0]
            if not np.isfinite(field_top):
                logger.info("Likely interpolation error at top boundary; setting field=1")
                logger.info("orig_scale: {}".format(orig_scale))
                field_top = 1
            field_bottom = self.evaluate_at_point(field, z=0)['g'][0][0]
            field.set_scales(orig_scale, keep_data=True)
        except:
            logger.debug("field at top shape {}".format(field['g'].shape))
            field_top = None
            field_bottom = None
        
        return field_bottom, field_top
    
    def _set_atmosphere(self):
        self.necessary_quantities = OrderedDict()

        self.phi = self._new_ncc()
        self.necessary_quantities['phi'] = self.phi

        self.del_ln_rho0 = self._new_ncc()
        self.ln_rho0 = self._new_ncc()
        self.rho0 = self._new_ncc()
        self.necessary_quantities['del_ln_rho0'] = self.del_ln_rho0
        self.necessary_quantities['ln_rho0'] = self.ln_rho0
        self.necessary_quantities['rho0'] = self.rho0

        self.del_s0 = self._new_ncc()
        self.necessary_quantities['del_s0'] = self.del_s0
        
        self.T0_zz = self._new_ncc()
        self.T0_z = self._new_ncc()
        self.T0 = self._new_ncc()
        self.necessary_quantities['T0_zz'] = self.T0_zz
        self.necessary_quantities['T0_z'] = self.T0_z
        self.necessary_quantities['T0'] = self.T0

        self.del_P0 = self._new_ncc()
        self.P0 = self._new_ncc()
        self.necessary_quantities['del_P0'] = self.del_P0
        self.necessary_quantities['P0'] = self.P0

        self.nu = self._new_ncc()
        self.chi = self._new_ncc()
        self.nu_l = self._new_ncc()
        self.chi_l = self._new_ncc()
        self.del_chi_l = self._new_ncc()
        self.del_nu_l = self._new_ncc()
        self.necessary_quantities['nu_l'] = self.nu_l
        self.necessary_quantities['chi_l'] = self.chi_l
        self.necessary_quantities['del_chi_l'] = self.del_chi_l
        self.necessary_quantities['del_nu_l'] = self.del_nu_l
        self.nu_r = self._new_ncc()
        self.chi_r = self._new_ncc()
        self.del_chi_r = self._new_ncc()
        self.del_nu_r = self._new_ncc()
        self.necessary_quantities['nu_r'] = self.nu_r
        self.necessary_quantities['chi_r'] = self.chi_r
        self.necessary_quantities['del_chi_r'] = self.del_chi_r
        self.necessary_quantities['del_nu_r'] = self.del_nu_r

        self.scale = self._new_ncc()
        self.scale_continuity = self._new_ncc()
        self.scale_energy = self._new_ncc()
        self.scale_momentum = self._new_ncc()
        self.necessary_quantities['scale'] = self.scale
        self.necessary_quantities['scale_continuity'] = self.scale_continuity
        self.necessary_quantities['scale_energy'] = self.scale_energy
        self.necessary_quantities['scale_momentum'] = self.scale_momentum

        #Add IH flux, IH, and Lsm1 (first moment of entropy gradient) location
        self.IH_flux = self._new_ncc()
        self.IH = self._new_ncc()
        self.cz_mask = self._new_ncc()
        self.cz_mask['g'] = 1
        self.necessary_quantities['IH_flux'] = self.IH_flux
        self.necessary_quantities['IH']      = self.IH
        self.necessary_quantities['cz_mask'] = self.cz_mask
        self.Lsm1 = 0
        self.d_conv = 0

    def copy_atmosphere(self, atmosphere):
        '''
        Copies values from a target atmosphere into the current atmosphere.
        '''
        self.necessary_quantities = atmosphere.necessary_quantities
            
    def plot_atmosphere(self):

        for key in self.necessary_quantities:
            logger.debug("plotting atmospheric quantity {}".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(2,1,1)
            quantity = self.necessary_quantities[key]
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], quantity['g'][0,:])
            if np.min(quantity['g'][0,:]) != np.max(quantity['g'][0,:]):
                ax.set_ylim(np.min(quantity['g'][0,:])-0.05*np.abs(np.min(quantity['g'][0,:])),
                        np.max(quantity['g'][0,:])+0.05*np.abs(np.max(quantity['g'][0,:])))
            ax.set_xlabel('z')
            ax.set_ylabel(key)
            
            ax = fig_q.add_subplot(2,1,2)
            power_spectrum = np.abs(quantity['c'][0,:]*np.conj(quantity['c'][0,:]))
            ax.plot(np.arange(len(quantity['c'][0,:])), power_spectrum)
            ax.axhline(y=1e-20, color='black', linestyle='dashed') # ncc_cutoff = 1e-10
            ax.set_xlabel('z')
            ax.set_ylabel("Tn power spectrum: {}".format(key))
            ax.set_yscale("log", nonposy='clip')
            ax.set_xscale("log", nonposx='clip')

            fig_q.savefig("atmosphere_{}_p{}.png".format(key, self.domain.distributor.rank), dpi=300)
            plt.close(fig_q)

        for key in self.necessary_quantities:
            if key not in ['P0', 'rho0']:
                continue
            logger.debug("plotting atmosphereic quantity ln({})".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(1,1,1)
            quantity = self.necessary_quantities[key]
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], np.log(quantity['g'][0,:]))
            if np.min(quantity['g'][0,:]) != np.max(quantity['g'][0,:]):
                ax.set_ylim(np.min(np.log(quantity['g'][0,:]))-0.05*np.abs(np.min(np.log(quantity['g'][0,:]))),
                        np.max(np.log(quantity['g'][0,:]))+0.05*np.abs(np.max(np.log(quantity['g'][0,:]))))
            ax.set_xlabel('z')
            ax.set_ylabel('ln_'+key)
            fig_q.savefig(self.fig_dir+"atmosphere_ln_{}_p{}.png".format(key, self.domain.distributor.rank), dpi=300, bbox_inches='tight')
            plt.close(fig_q)
      
        fig_atm = plt.figure()
        axT = fig_atm.add_subplot(2,2,1)
        axT.plot(self.z[0,:], self.T0['g'][0,:])
        axT.set_ylabel('T0')
        axP = fig_atm.add_subplot(2,2,2)
        axP.semilogy(self.z[0,:], self.P0['g'][0,:]) 
        axP.set_ylabel('P0')
        axR = fig_atm.add_subplot(2,2,3)
        axR.semilogy(self.z[0,:], self.rho0['g'][0,:])
        axR.set_ylabel(r'$\rho0$')
        axS = fig_atm.add_subplot(2,2,4)
        analysis.semilogy_posneg(axS, self.z[0,:], self.del_s0['g'][0,:], color_neg='red')
        
        axS.set_ylabel(r'$\nabla s0$')
        fig_atm.savefig("atmosphere_quantities_p{}.png".format(self.domain.distributor.rank), dpi=300)

        fig_atm = plt.figure()
        axS = fig_atm.add_subplot(2,2,1)
        axdelS = fig_atm.add_subplot(2,2,2)
        axlnP = fig_atm.add_subplot(2,2,3)
        axdellnP = fig_atm.add_subplot(2,2,4)

        Cv_inv = self.gamma-1
        axS.plot(self.z[0,:], 1/Cv_inv*np.log(self.T0['g'][0,:]) - 1/Cv_inv*(self.gamma-1)*np.log(self.rho0['g'][0,:]), label='s0', linewidth=2)
        axS.plot(self.z[0,:], (1+(self.gamma-1)/self.gamma*self.g)*np.log(self.T0['g'][0,:]), label='s based on lnT', linewidth=2)
        axS.plot(self.z[0,:], np.log(self.T0['g'][0,:]) - (self.gamma-1)/self.gamma*np.log(self.P0['g'][0,:]), label='s based on lnT and lnP', linewidth=2)
        
        axdelS.plot(self.z[0,:], self.del_s0['g'][0,:], label=r'$\nabla s0$', linewidth=2)
        axdelS.plot(self.z[0,:], self.T0_z['g'][0,:]/self.T0['g'][0,:] + self.g*(self.gamma-1)/self.gamma*1/self.T0['g'][0,:],
                    label=r'$\nabla s0$ from T0', linewidth=2, linestyle='dashed',color='red')
         
        axlnP.plot(self.z[0,:], np.log(self.P0['g'][0,:]), label='ln(P)', linewidth=2)
        axlnP.plot(self.z[0,:], self.ln_P0['g'][0,:], label='lnP', linestyle='dashed', linewidth=2)
        axlnP.plot(self.z[0,:], -self.g*np.log(self.T0['g'][0,:])*(self.T0_z['g'][0,:]), label='-g*lnT', linewidth=2, linestyle='dotted')
        
        axdellnP.plot(self.z[0,:], self.del_ln_P0['g'][0,:], label='dellnP', linewidth=2)
        axdellnP.plot(self.z[0,:], -self.g/self.T0['g'][0,:], label='-g/T', linestyle='dashed', linewidth=2, color='red')
        
        #axS.legend()
        axS.set_ylabel(r'$s0$')
        fig_atm.savefig("atmosphere_s0_p{}.png".format(self.domain.distributor.rank), dpi=300)

    def plot_scaled_atmosphere(self):

        for key in self.necessary_quantities:
            logger.debug("plotting atmospheric quantity {}".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(2,1,1)
            quantity = self.necessary_quantities[key]
            quantity['g'] *= self.scale['g']
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], quantity['g'][0,:])
            ax.set_xlabel('z')
            ax.set_ylabel(key+'*scale')

            ax = fig_q.add_subplot(2,1,2)
            ax.plot(np.arange(len(quantity['c'][0,:])), np.abs(quantity['c'][0,:]*np.conj(quantity['c'][0,:])))
            ax.set_xlabel('z')
            ax.set_ylabel("Tn power spectrum: {}*scale".format(key))
            ax.set_yscale("log", nonposy='clip')
            ax.set_xscale("log", nonposx='clip')
            
            fig_q.savefig("atmosphere_{}scale_p{}.png".format(key, self.domain.distributor.rank), dpi=300)
            plt.close(fig_q)
                        
    def check_that_atmosphere_is_set(self):
        for key in self.necessary_quantities:
            quantity = self.necessary_quantities[key]['g']
            quantity_set = quantity.any()
            if not quantity_set:
                logger.info("WARNING: atmosphere {} is all zeros on process 0".format(key))
                
    def test_hydrostatic_balance(self, P_z=None, P=None, T=None, rho=None, make_plots=False):

        if rho is None:
            logger.error("HS balance test requires rho (currently)")
            raise
        
        if P_z is None:
            if P is None:
                if T is None:
                    logger.error("HS balance test requires P_z, P or T")
                    raise
                else:
                    T_scales = T.meta[:]['scale']
                    rho_scales = rho.meta[:]['scale']
                    if rho_scales != 1:
                        rho.set_scales(1, keep_data=True)
                    if T_scales != 1:
                        T.set_scales(1, keep_data=True)
                    P = self._new_field()
                    T.set_scales(self.domain.dealias, keep_data=True)
                    rho.set_scales(self.domain.dealias, keep_data=True)
                    P.set_scales(self.domain.dealias, keep_data=False)
                    P['g'] = T['g']*rho['g']
                    T.set_scales(T_scales, keep_data=True)
                    rho.set_scales(rho_scales, keep_data=True)

            P_z = self._new_field()
            P.differentiate('z', out=P_z)
            P_z.set_scales(1, keep_data=True)

        rho_scales = rho.meta[:]['scale']
        rho.set_scales(1, keep_data=True)
        # error in hydrostatic balance diagnostic
        HS_balance = P_z['g']+self.g*rho['g']
        relative_error = HS_balance/P_z['g']
        rho.set_scales(rho_scales, keep_data=True)
        
        HS_average = self._new_field()
        HS_average['g'] = HS_balance
        if self.dimensions > 1:
            HS_average.integrate('x')
            HS_average['g'] /= self.Lx
        HS_average.set_scales(1, keep_data=True)

        relative_error_avg = self._new_field()
        relative_error_avg['g'] = relative_error
        if self.dimensions > 1:
            relative_error_avg.integrate('x')
            relative_error_avg['g'] /= self.Lx
        relative_error_avg.set_scales(1, keep_data=True)

        if self.make_plots or make_plots:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            if self.dimensions > 1:
                ax1.plot(self.z[0,:], P_z['g'][0,:])
                ax1.plot(self.z[0,:], -self.g*rho['g'][0,:])
            else:
                ax1.plot(self.z[:], P_z['g'][:])
                ax1.plot(self.z[:], -self.g*rho['g'][:])
            ax1.set_ylabel(r'$\nabla P$ and $\rho g$')
            ax1.set_xlabel('z')

            ax2 = fig.add_subplot(2,1,2)
            if self.dimensions > 1:
                ax2.semilogy(self.z[0,:], np.abs(relative_error[0,:]))
                ax2.semilogy(self.z[0,:], np.abs(relative_error_avg['g'][0,:]))
            else:
                ax2.semilogy(self.z[:], np.abs(relative_error[:]))
                ax2.semilogy(self.z[:], np.abs(relative_error_avg['g'][:]))
            ax2.set_ylabel(r'$|\nabla P + \rho g |/|\nabla P|$')
            ax2.set_xlabel('z')
            fig.savefig(self.fig_dir+"atmosphere_HS_balance_p{}.png".format(self.domain.distributor.rank), dpi=300)

        max_rel_err = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error)), op=MPI.MAX)
        max_rel_err_avg = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error_avg['g'])), op=MPI.MAX)
        logger.info('max error in HS balance: point={} avg={}'.format(max_rel_err, max_rel_err_avg))

    def check_atmosphere(self, make_plots=False, **kwargs):
        if self.make_plots or make_plots:
            try:
                self.plot_atmosphere()
            except:
                logger.info("Problems in plot_atmosphere: atm full of NaNs?")
        self.test_hydrostatic_balance(make_plots=make_plots, **kwargs)
        self.check_that_atmosphere_is_set()
        

class ConstHeating(Atmosphere):
    '''
    A reference atmosphere in which to study some interesting systems in stratified
    convection.  These atmospheres are in thermal equilibrium with a constant
    internal heating term and constant kappa.  This sets the profile of temperature.
    Density is backed out from hydrostatic equilibrium.  There are three characteristic
    parameters of these atmospheres:
    
    (1) n_rho_cz -- the number of density scale heights spanned by the portion of
                      the atmosphere carrying a superadiabatic amount of flux.
    (2) r        -- the depth of the radiative zone, in units of the convective
                      zone depth, below the convective zone.
    (3) epsilon  -- As in polytropes, this sets the value of gravity and determines
                      the mach number of the flows.  For small eps, Ma propto eps^{1/2}

    Class parameters:
        atmosphere_name     - A string containing the name of the atmosphere
        aspect_ratio        - The atmospheric aspect ratio, Lx/Lz
        n_rho_cz            - As defined above
        r                   - As defined above
        epsilon             - As defined above
        d_conv              - The vertical extent of the convective zone
        z_cross             - The z-coordinate at the boundary between RZ/CZ
        H                   - The magnitude of internal heating
        Lx, Ly, Lz          - Physical length of atmosphere in corresponding dimension
        gamma, Cp, Cv       - Thermodynamic quantities of ideal gas
        m_ad                - The adiabatic polytropic index, 1/(gamma-1)
        g                   - The value of gravitational acceleration
        Rayleigh            - The Rayleigh number at the first moment of TgradS
        Prandtl             - The Prandtl number at same location as Ra
    '''
    def __init__(self,
                 nx=256,
                 ny=256,
                 nz=128,
                 aspect_ratio=4,
                 n_rho_cz = 4, #Initially use the number of density scale heights of an adiabatic polytrope
                 epsilon=1e-4, gamma=5/3, r=1,
                 constant_kappa=True, constant_mu=True,
                 **kwargs):
        
        self.atmosphere_name = 'Internally Heated Atmosphere -- Constant Heating'
        self.aspect_ratio    = aspect_ratio
        self.n_rho_cz        = n_rho_cz
        self.r               = r
        self._set_atmosphere_parameters(gamma=gamma, epsilon=epsilon)
        self.Lz = self._calculate_Lz_cz(n_rho_cz, self.m_ad)
        self.d_conv  = self.Lz / (1 + self.r)
        self.z_cross = self.Lz - self.d_conv
        self.H = self.epsilon / (self.d_conv * self.Cp)
        self.Lx = self.Ly = self.Lz*aspect_ratio
            
        super(ConstHeating, self).__init__(nx=nx, ny=ny, nz=nz, Lx=self.Lx, Ly=self.Ly, Lz=self.Lz, **kwargs)
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
      
        #TODO: Improve language around "constant kappa/mu" and "dynamic diffusivities" stuff
        self.constant_kappa = constant_kappa
        self.constant_mu    = constant_mu
        if self.constant_kappa == False and self.constant_mu == False:
            self.constant_diffusivities = True
        else:
            self.constant_diffusivities = False

        
        from scipy.special import erf

        self._set_atmosphere()
        self.cz_mask['g'] = 0.5 * (erf(100*(self.z - self.z_cross)) + 1)
        self.d_conv  = self.Lz / (1 + self.r)
        self._set_timescales()

    def depth_root_find(self, L):
        H = self.epsilon / (L * self.Cp)
        xi = np.sqrt(2*H + 1)
        return self.n_rho_cz + np.log(1 - (H/2)*L**2 + L) \
               - (self.g/xi)*np.log((xi**2 - 1 + H*L*(xi + 1))/(xi**2 - 1 - H*L*(xi - 1)))

    def _calculate_Lz_cz(self, n_rho_cz, m_ad):
        '''
        Calculate the Lz of the CZ based on the number of specified density scale heights
        using root-finding, then extend the atmosphere below that an appropriate amount
        based on the value of f to make room for the RZ.
        '''
        from scipy.optimize import brentq
        Lz_guess = (np.exp(n_rho_cz/np.abs((m_ad-self.epsilon)))-1)*(self.g/self.Cp)
        r = brentq(self.depth_root_find, Lz_guess/2, Lz_guess*2, maxiter=int(1e3), full_output=True)
        return r[0] * (1 + self.r)
    
    def _set_atmosphere_parameters(self, gamma, epsilon):
        # polytropic atmosphere characteristics
        self.gamma = gamma
        self.Cv = 1/(self.gamma-1)
        self.Cp = self.gamma*self.Cv
        self.epsilon = epsilon

        self.m_ad = 1/(self.gamma-1)
        self.g = self.m_ad + 1 - self.epsilon

        logger.info("IH Atmosphere parameters:")
        logger.info("   m_ad = {:g}, epsilon = {:g}, gravity = {:g}, gamma = {:g}".format(self.m_ad, self.epsilon, self.g, self.gamma))
    
    def _set_atmosphere(self):
        super(ConstHeating, self)._set_atmosphere()

        self.T0_zz['g'] = -self.H
        self.T0_z['g'] = - 1 + self.H * (self.Lz - self.z)
        self.T0_z.antidifferentiate('z', ('right', 1), out=self.T0)
#        self.T0['g'] = (1 - self.H * self.Lz)*(self.Lz - self.z) + (self.H/2)*(self.Lz**2 - self.z**2) + 1

        self.T0_z.set_scales(1, keep_data=True)
        self.T0.set_scales(1, keep_data=True)
        self.del_ln_rho0['g'] = - (self.g + self.T0_z['g']) / self.T0['g']
        self.del_ln_rho0.antidifferentiate('z', ('right', 0), out=self.ln_rho0)
        self.ln_rho0.set_scales(1, keep_data=True)
        self.rho0['g'] = np.exp(self.ln_rho0['g'])

        self.T0_z.set_scales(1, keep_data=True)
        self.T0.set_scales(1, keep_data=True)
        self.del_ln_rho0.set_scales(1, keep_data=True)
        #Note: this is grad S, not grad S / cP
        self.del_s0['g'] = (1/(self.gamma-1)) * self.T0_z['g']/self.T0['g'] - self.del_ln_rho0['g']
        s0 = self._new_ncc()
        self.del_s0.antidifferentiate('z', ('right', 0), out=s0)
        self.delta_s = np.mean(s0.interpolate(z=self.Lz)['g'][0]) - np.mean(s0.interpolate(z=self.z_cross)['g'][0])
 
        self.T0.set_scales(1, keep_data=True)
        self.rho0.set_scales(1, keep_data=True)
        self.P0['g'] = self.rho0['g']*self.T0['g']
        self.P0.differentiate('z', out=self.del_P0)
        self.del_P0.set_scales(1, keep_data=True)
        self.P0.set_scales(1, keep_data=True)
        
        if self.constant_diffusivities:
            self.rho0.set_scales(1, keep_data=True)
            self.T0.set_scales(1, keep_data=True)
            #scale = self.rho0['g']
            scale = self.T0['g']
            scale = 1
            self.scale['g']            = scale
            self.scale_continuity['g'] = scale
            self.scale_momentum['g']   = scale
            self.scale_energy['g']     = scale
        else:
            # consider whether to scale nccs involving chi differently (e.g., energy equation)
            self.rho0.set_scales(1, keep_data=True)
            self.T0.set_scales(1, keep_data=True)
            #scale = self.rho0['g']
            scale = self.T0['g']
            self.scale['g']            = scale
            self.scale_continuity['g'] = scale
            self.scale_momentum['g']   = scale
            self.scale_energy['g']     = scale

        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.Lz + 1 - self.z)

        rho0_max, rho0_min = self.value_at_boundary(self.rho0)
        if rho0_max is not None:
            try:
		# For "strange" resolutions (e.g., 96x192), sometimes this crashes.  Need to investigate. (9/12/2017)
                # TODO: make this work for our case
                rho0_ratio = rho0_max/rho0_min
                logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))
                logger.info("   density scale heights = {:g} (measured)".format(np.log(rho0_ratio)))
            except:
                if self.domain.distributor.comm_cart.rank == 0:
                    logger.error("Something went wrong with reporting density range")
           
        #TODO: Fix this, make it general and calculate it from fields.
        #       OR use the analytical expression of grad ln rho to do this.
        H_rho_top = - 1 / ( self.g - 1 )
        H_rho_bottom = - (1 - (self.H/2)*self.Lz**2 + self.Lz) / (self.g - 1 + self.H * self.Lz)
        H_rho_bot_CZ = - (1 - (self.H/2)*self.Lz**2 + self.Lz - (self.H/2)*self.z_cross**2 - (1 - self.H*self.Lz)*self.z_cross) / \
                         ( self.g - 1 + self.H*(self.d_conv) )
        logger.info("   H_rho = {:g} (top)  {:g} (bottom)   {:g} (bot CZ)".format(H_rho_top,H_rho_bottom, H_rho_bot_CZ))
        if self.delta_x != None:
            logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)    {:g} (bot CZ)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x,
                                                                          H_rho_bot_CZ/self.delta_x))
        
    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
            
        # min of global quantity
        atmosphere.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']/self.Cp))), op=MPI.MIN)
        atmosphere.freefall_time = np.sqrt(self.d_conv/self.g)
        atmosphere.buoyancy_time = np.sqrt(np.abs(self.d_conv / (self.g * self.delta_s / self.Cp)))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(atmosphere.min_BV_time,
                                                                                               atmosphere.freefall_time,
                                                                                               atmosphere.buoyancy_time))
    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1, split_diffusivities=False):
       
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl

        # set nu and chi at top based on Rayleigh number

        #The first moment of the superadiabatic temperature gradient is where the rayleigh number should be set.  
        TgradS        = self._new_ncc()
        TgradS_int    = self._new_ncc()
        TgradS_L      = self._new_ncc()
        TgradS_L_int  = self._new_ncc()
        self.del_s0.set_scales(1, keep_data=True)
        TgradS['g'] = self.T0['g']*self.del_s0['g']/self.Cp
        TgradS.set_scales(1, keep_data=True)
        TgradS_L['g'] = self.z*TgradS['g']
        TgradS.antidifferentiate('z', ('right', 0), out=TgradS_int)
        TgradS_L.antidifferentiate('z', ('right', 0), out=TgradS_L_int)

        self.Lsm1 = np.mean(TgradS_L_int.interpolate(z=self.z_cross)['g'])/np.mean(TgradS_int.interpolate(z=self.z_cross)['g'])

        nu_moment    = np.sqrt(Prandtl * (self.d_conv**4 * np.abs(np.mean(self.del_s0.interpolate(z=self.Lsm1)['g'])/self.Cp) * self.g) / Rayleigh)

        self.nu_top  = nu_top = nu_moment * np.mean(self.rho0.interpolate(z=self.Lsm1)['g'])
        self.chi_top = chi_top = nu_top/Prandtl

        #self.nu_top = nu_top = np.sqrt(Prandtl * (self.d_conv**3 * np.abs(self.delta_s/self.Cp) * self.g) / Rayleigh)
        #self.nu_top = nu_top = np.sqrt(Prandtl * (self.d_conv**4 * self.epsilon * self.g) / Rayleigh)

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = nu_top
            chi = chi_top

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))
        else:
            if self.constant_kappa:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    chi_l = chi_top/(self.rho0['g'])
                    chi_r = 0
                else:
                    if self.poly_m < 1:
                        chi_l = np.exp(self.n_rho_cz)*chi_top/(self.z0 - self.z)
                    else:
                        chi_l = chi_top/(self.z0 - self.z)
                    chi_r = chi_top/(self.rho0['g']) - chi_l
                logger.info('using constant kappa')
            else:
                chi_l = chi_top
                chi_r = 0
                logger.info('using constant chi')
            if self.constant_mu:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    nu_l  = nu_top/(self.rho0['g'])
                    nu_r = 0
                else:
                    if self.poly_m < 1:
                        nu_l  = np.exp(self.n_rho_cz)*nu_top/(self.z0 - self.z)
                    else:
                        nu_l  = nu_top/(self.z0 - self.z)
                    nu_r  = nu_top/(self.rho0['g']) - nu_l
                logger.info('using constant mu')
            else:
                nu_l  = nu_top
                nu_r = 0
                logger.info('using constant nu')

      
            logger.info("   nu_top = {:g}, chi_top = {:g}".format(nu_top, chi_top))

        self.T0_z.set_scales(1, keep_data=True)
        self.IH_flux['g'] = - self.T0_z['g']

        #TODO: look at everything below this
                    
        #Allows for atmosphere reuse
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.set_scales(1, keep_data=True)
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
        self.nu_l['g'] = nu_l
        self.chi_l['g'] = chi_l
        self.nu_r['g'] = nu_r
        self.chi_r['g'] = chi_r
        self.nu['g'] = nu_l + nu_r
        self.chi['g'] = chi_l + chi_r

        self.chi_l.differentiate('z', out=self.del_chi_l)
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.differentiate('z', out=self.del_nu_l)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.differentiate('z', out=self.del_chi_r)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.differentiate('z', out=self.del_nu_r)
        self.nu_r.set_scales(1, keep_data=True)

        # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
        self.thermal_time = self.Lz**2/(self.chi.interpolate(z=0)['g'][0])
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/(self.nu.interpolate(z=0)['g'][0])
        self.top_viscous_time = 1/nu_top

        if self.dimensions == 2:
            self.thermal_time = self.thermal_time[0]
            self.viscous_time = self.viscous_time[0]
        if self.dimensions > 2:
            #Need to communicate across processes if mesh is weird in 3D
            therm = np.zeros(1, dtype=np.float64)
            visc  = np.zeros(1, dtype=np.float64)
            therm_rcv, visc_rcv = np.zeros_like(therm), np.zeros_like(visc)
            therm[0] = np.mean(self.thermal_time)
            visc[0]  = np.mean(self.viscous_time)
            if np.isnan(therm): therm[0] = 0
            if np.isnan(visc):  visc[0]  = 0
            self.domain.dist.comm_cart.Allreduce(therm, therm_rcv, op=MPI.MAX)
            self.thermal_time = therm_rcv[0]
            self.domain.dist.comm_cart.Allreduce(visc, visc_rcv, op=MPI.MAX)
            self.viscous_time = visc_rcv[0]

        logger.info("thermal_time = {}, top_thermal_time = {}".format(self.thermal_time,
                                                                      self.top_thermal_time))
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)

    def save_atmosphere_file(self, data_dir):
        #This creates an output file that contains all of the useful atmospheric info at the beginning of the run
        out_dir = data_dir + '/atmosphere/'
        out_file = out_dir + 'atmosphere.h5'
        if self.domain.dist.rank == 0:
            if not os.path.exists('{:s}'.format(out_dir)):
                os.mkdir('{:s}'.format(out_dir))
            f = h5py.File('{:s}'.format(out_file), 'w')
        indxs = [0]*self.dimensions
        indxs[-1] = range(self.nz)
        key_set = list(self.problem.parameters.keys())
        extended_keys = ['chi','nu','del_chi','del_nu']
        key_set.extend(extended_keys)
        logger.debug("Outputing atmosphere parameters for {}".format(key_set))
        for key in key_set:
            # Figure out what type of data we're dealing with
            if 'scale' in key:
                continue
            if key in extended_keys:
                field_key = True
            elif type(self.problem.parameters[key]) == Field:
                field_key = True
                self.problem.parameters[key].set_scales(1, keep_data=True)
            else:
                field_key = False

            # Get the proper data
            if field_key:
                try:
                    if key in extended_keys:
                        self.problem.parameters[key+'_l'].require_layout(self.domain.dist.layouts[1])
                        self.problem.parameters[key+'_r'].require_layout(self.domain.dist.layouts[1])
                        array = self.problem.parameters[key+'_l'].data[indxs] +\
                                self.problem.parameters[key+'_r'].data[indxs]
                    else:
                        self.problem.parameters[key].require_layout(self.domain.dist.layouts[1])
                        array = self.problem.parameters[key].data[indxs]
                except:
                    if self.domain.dist.rank == 0:
                        logger.error("key error on atmosphere output {}".format(key))
                    array = 0
                if self.domain.dist.rank == 0:
                    f[key] = array
            elif self.domain.dist.rank == 0:
                f[key] = self.problem.parameters[key]
        
        z_value = self.domain.bases[-1].grid(1)
        if self.domain.dist.rank == 0:
            f['z'] = z_value

        if self.domain.dist.rank == 0:
            f['dimensions']     = self.dimensions
            if self.dimensions > 1:
                f['nx']             = self.nx
            if self.dimensions > 2:
                f['ny']             = self.ny
            f['nz']             = self.nz
            f['r']             = self.r
            f['z_cross']       = self.z_cross
            f['H']             = self.H
            f['m_ad']           = self.m_ad
            f['epsilon']        = self.epsilon
            f['n_rho_cz']       = self.n_rho_cz
            f['rayleigh']       = self.Rayleigh
            f['prandtl']        = self.Prandtl
            f['aspect_ratio']   = self.aspect_ratio
            f['atmosphere_name']= self.atmosphere_name
            f['t_buoy']         = self.buoyancy_time
            f['t_therm']        = self.thermal_time
            f.close()


