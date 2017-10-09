"""
Dedalus script for finding the onset of compressible convection in a 
constantly internally heated atmosphere

Usage:
    FC_onset_curve.py [options] 

Options:
    --rayleigh_start=<Rayleigh>         Rayleigh number start [default: 5e-1]
    --rayleigh_stop=<Rayleigh>          Rayleigh number stop [default: 5e1]
    --rayleigh_steps=<steps>            Integer number of steps between start 
                                         and stop Ra   [default: 20]
    --kx_start=<kx_start>               kx to start at [default: 0.1]
    --kx_stop=<kx_stop>                 kx to stop at  [default: 1]
    --kx_steps=<kx_steps>               Num steps in kx space [default: 20]

    --ky_start=<ky_start>               kx to start at [default: 0.1]
    --ky_stop=<ky_stop>                 kx to stop at  [default: 1]
    --ky_steps=<ky_steps>               Num steps in kx space [default: 20]

    --nz=<nz>                           z (chebyshev) resolution [default: 48]

    --bcs=<bcs>                         Boundary conditions ('fixed', 'mixed', 
                                            or 'flux') [default: mixed]
    --Prandtl=<Pr>                      Prandtl number [default: 1]
    --Taylor=<Ta>                       If not None, solve for rotating convection

    #Const Heating options
    --n_rho_cz=<n_rho_cz>                     nrho of convecting region [default: 3]
    --epsilon=<epsilon>                 epsilon of const heating atmo  [default: 0.5]
    --r=<r>                             The r-parameter of the const heating atmo [default: 1]
    --gamma=<gamma>                     Gamma of atmosphere [default: 5/3]
    --constant_chi                      If true, use const chi
    --constant_nu                       If true, use const nu


    --3D                                If flagged, use 3D eqns and search kx & ky
    --2.5D                              If flagged, use 3D eqns with ky = 0

    --load                              If flagged, attempt to load data from output file
    --exact                             If flagged, after doing the course search + interpolation,
                                            iteratively solve for the exact critical using
                                            optimization routines
    --out_dir=<out_dir>                 Base output dir [default: ./]

    --sparse                            If flagged, use sprase solve
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
from onset_solver import OnsetSolver
import numpy as np

args = docopt(__doc__)

atmo_type = 0
file_name = 'FC_ConstHeat_onsets'



##########################################
#Set up ra / kx / ky spans
ra_log, kx_log, ky_log = False, False, False

ra_start = float(args['--rayleigh_start'])
ra_stop = float(args['--rayleigh_stop'])
ra_steps = float(args['--rayleigh_steps'])
if np.abs(ra_stop/ra_start) > 10:
    ra_log = True

kx_start = float(args['--kx_start'])
kx_stop = float(args['--kx_stop'])
kx_steps = float(args['--kx_steps'])
if np.abs(kx_stop/kx_start) > 10:
    kx_log = True

ky_start = float(args['--ky_start'])
ky_stop = float(args['--ky_stop'])
ky_steps = float(args['--ky_steps'])
if np.abs(ky_stop/ky_start) > 10:
    ky_log = True

##########################################
#Set up defaults for the atmosphere
const_kap = True
if args['--constant_chi']:
    const_kap = False
const_mu = True
if args['--constant_nu']:
    const_mu = False

nz = int(args['--nz'])
n_rho = float(args['--n_rho_cz'])
epsilon = float(args['--epsilon'])
r = float(args['--r'])
file_name += '_eps{}'.format(args['--epsilon'])
file_name += '_nrho{}'.format(args['--n_rho_cz'])
try:
    gamma = float(args['--gamma'])
except:
    from fractions import Fraction
    gamma = float(Fraction(args['--gamma']))


atmo_kwargs = {'n_rho_cz':       n_rho,
               'epsilon':        epsilon,
               'constant_kappa': const_kap,
               'constant_mu':    const_mu,
               'nz':             nz,
               'r':              r,
               'gamma':          gamma}

##############################################
#Setup default arguments for equation building
taylor = args['--Taylor']
eqn_kwargs = dict()
if taylor != None:
    file_name += '_Ta{}'.format(taylor)
    taylor = float(taylor)
    eqn_kwargs['Taylor'] = taylor
eqn_args = [float(args['--Prandtl'])]

############################################
#Set up BCs
bcs = args['--bcs']
fixed_flux = False
fixed_T    = False
mixed_T    = False
if bcs == 'mixed':
    mixed_T = True
    file_name += '_mixedBC'
elif bcs == 'fixed':
    fixed_T = True
    file_name += '_fixedTBC'
else:
    fixed_flux = True
    file_name += '_fluxBC'

bc_kwargs = {'fixed_temperature': fixed_T,
               'mixed_flux_temperature': mixed_T,
               'fixed_flux': fixed_flux,
               'stress_free': True}

#####################################################
#Initialize onset solver
if args['--3D']:
    file_name += '_3D'
    ky_steps = (ky_start, ky_stop, ky_steps, ky_log)
    threeD = True
elif args['--2.5D']:
    file_name += '_2.5D'
    ky_steps = None
    threeD = True
else:
    file_name += '_2D'
    ky_steps = None
    threeD = False

solver = OnsetSolver(
            eqn_set=0, atmosphere=atmo_type, threeD=threeD,
            ra_steps=(ra_start, ra_stop, ra_steps, ra_log),
            kx_steps=(kx_start, kx_stop, kx_steps, kx_log),
            ky_steps=ky_steps,
            atmo_kwargs=atmo_kwargs,
            eqn_args=eqn_args,
            eqn_kwargs=eqn_kwargs,
            bc_kwargs=bc_kwargs,
            sparse=args['--sparse'])

#############################################
#Crit find!
out_dir = args['--out_dir']
load    = args['--load']
exact   = args['--exact']
solver.find_crits(out_dir=out_dir, out_file='{:s}'.format(file_name), load=load, exact=exact)
