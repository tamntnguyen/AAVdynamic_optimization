
import os
import numpy as np
from pyomo.environ import *
from pyomo.dae import *
from idaes import bin_directory
from idaes.core.util.model_statistics import degrees_of_freedom as dof
import hdf5storage

from params import Params, get_param_values
from states import States, initialize_states
from eqns import Equations
from config import find_bounds
from formulation import fix_vars_cons, set_objective, check_divide_by_zero, \
    show_fixed, scale_model, check_wrong_initialization
from util import saver, runsolver, plot, cprint, which_infeasible, compare_with_sim
from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))


def main():
    # Operation specification
    m = ConcreteModel()
    m.system = 'sflask'  # sflask / bioreactor
    m.mode = 'estimation'  # estimation / simulation / optimization(open-loop)
    m.objective = 'mse'
    m.disc_scheme = 'finite_difference'  # collocation / finite_difference
    m.bound_source = 'manual'
    m.scaling = True
    m.save_model = False
    m.save_result = True
    
    # Kinetics to include
    m.growth = True
    m.meta = True  # if False, Amm and Lac will be approximated by polynomials
    m.viralprod = True

    # Parameter estimation setting
    m.est_growth = False
    m.est_meta = False
    m.est_viralprod = True

    # System dependent setting
    m.level_control = True  # if the system is bioreactor

    # Set domains
    if m.system == 'bioreactor':
        dT = 6  # discretization interval [h]
        Ti = 24  # inoculation time [h]
        Tf = 24*14  # termination time [h]
        Ts = 24  # sampling and discrete bolus injection interval [h]
        Vs = 6  # sampling volume [mL]
        m.Ti = Ti
        m.Tf = Tf
        m.Ts = np.arange(Ti, Tf, Ts)
        m.Vs = Vs
        m.dt = 1  # assitive points for discrete inputs [h]
        Ts1 = m.Ts - m.dt
        T = np.arange(0, Tf+1, dT).tolist() + Ts1.tolist()
        T = np.array(T)
        T = T[T<=Tf]
        I = [1]  # simulation data to use for the initialization

    elif m.system == 'sflask':
        m.dt = 0.1  # assitive points for discrete inputs [h]
        T = np.arange(0, 6, 1).tolist() \
            + np.arange(6, 24, 2).tolist() \
            + np.arange(24, 30, 1).tolist() \
            + np.arange(30, 48, 2).tolist() \
            + np.arange(48, 59, 1).tolist() \
            + np.arange(59, 72, 2).tolist() \
            + np.arange(72, 78, 1).tolist() \
            + np.arange(78, 100, 6).tolist() \
            + np.arange(100, 106, 1).tolist() \
            + np.arange(106, 109, 2).tolist() \
            + np.arange(109, 121, 6).tolist() \
                + list([24-m.dt, 48-m.dt, 72-m.dt, 53-m.dt, 100-m.dt, 120])
        I = [1,2,3,4]
    
    m.t = ContinuousSet(initialize=T)
    m.i = Set(initialize=I)  # indices for available experimental data
    
    # Define parameters, variables, and equations
    Params(m)
    States(m)
    Equations(m)
    
    # Discretization
    TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=len(m.t)-1)
    
    # Initialization and scaling
    initialize_states(m)
    scale_model(m)

    # Formulate problem
    fix_vars_cons(m)
    set_objective(m)
    
    # Checker
    cprint('DOF of the model is: ' + str(dof(m)))
    # check_divide_by_zero(m)
    # check_wrong_initialization(m)
    # show_fixed(m)
    # compare_with_sim(m)
    # plot(m)

    # Run solver
    m, results = runsolver(m, solver='mumps')

    # Check parameters are nearly to bounds
    if m.mode == 'estimation':
        which_infeasible(m, target='param')
    
    # Save results
    saver(m, time_stamp=False)
    

if __name__ == '__main__':
    main()