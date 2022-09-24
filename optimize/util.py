import os
import sys
import dill
from datetime import datetime
import numpy as np
import hdf5storage
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from pyomo.environ import *

from config import get_root_var_name, namespaces
from params import save_params
from states import find_sim_value_transformed, save_X0
from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))


def to_array(obj):
    if not (isinstance(obj, Param) or isinstance(obj, Var) or isinstance(obj, Expression)):
        return

    dim = obj.dim()
    if dim == 0:  # scalar case
        array = np.array(value(obj))
    else:
        size = [0]*dim
        indexer = []
        indices = obj.index_set()
        for i, idx in enumerate(indices.subsets()):
            indexer.append(idx.ordered_data())
            size[i] = len(idx.ordered_data())
        
        # Generate array
        array = np.empty(size)
        data = obj.extract_values()
        coord = np.empty(dim, dtype=np.int64)
        for indices, val in data.items():
            if isinstance(indices, float) or isinstance(indices, int):
                array[indexer[0].index(indices)] = value(val)
            elif isinstance(indices, tuple):
                for i, idx in enumerate(indices):
                    coord[i] = indexer[i].index(idx)
                array[tuple(coord)] = value(val)
            else:
                raise('#TODO')
    return array


def get_name(obj):
    name = obj.name
    for char in ['[', ']', '.']:
        name = name.replace(char, '_')
    name = name.replace('__', '_')
    return name


def saver(m, time_stamp=False):
    _, _, datadir = set_path(sys.path[0])
    time_index = datetime.now().strftime('%y%m%d_%H%M%S')

    if m.save_model == True:
        if time_stamp:
            filename = os.path.join(datadir, m.system, 'model_' + m.mode + '_' + time_index +'.json')
        else:
            filename = os.path.join(datadir, m.system, 'model_' + m.mode + '.json')
        with open(filename, 'wb') as f:
            dill.dump(m, f)
            
    if m.save_result == True:
        if m.mode == 'estimation':
            save_params(m)
            if m.est_growth:
                save_X0(m)

        X_list = namespaces(m.system, target='X')
        dX_list = namespaces(m.system, target='dX')
        R_list = namespaces(m.system, target='R')
        k_growth_list, k_meta_list, k_viral_list = namespaces(m.system, target='k')
        U_list = namespaces(m.system, target='U')

        d = {}
        T = len(m.t)
        I = len(m.i)
        X = np.empty(shape=[T, I, len(X_list)])
        R = np.empty(shape=[T, I, len(R_list)])
        dX = np.empty(shape=[T, I, len(dX_list)])
        k_growth = np.empty(shape=len(k_growth_list))
        k_meta = np.empty(shape=len(k_meta_list))
        k_viral = np.empty(shape=len(k_viral_list))
        U = np.empty(shape=[T, I, len(U_list)])
        
        for obj in m.component_objects():
            name = get_name(obj)
            val = to_array(obj)
            if name in X_list:
                idx = X_list.index(name)
                X[:,:,idx] = val
            elif name in dX_list:
                idx = dX_list.index(name)
                dX[:,:,idx] = val
            elif name in R_list:
                idx = R_list.index(name)
                R[:,:,idx] = val
            elif name in k_growth_list:
                idx = k_growth_list.index(name)
                k_growth[idx] = val
            elif name in k_meta_list:
                idx = k_meta_list.index(name)
                k_meta[idx] = val
            elif name in k_viral_list:
                idx = k_viral_list.index(name)
                k_viral[idx] = val
            elif name in U_list:
                idx = U_list.index(name)
                U[:,:,idx] = val
        d['X'] = X
        d['dX'] = dX
        d['R'] = R
        d['k_growth'] = k_growth
        d['k_meta'] = k_meta
        d['k_viral'] = k_viral
        d['U'] = U
        d['t'] = np.double(m.t.ordered_data())
        
        if time_stamp:
            filedir = os.path.join(datadir, m.system, 'result_' + m.mode + '_' + time_index +'.mat')
        else:
            filedir = os.path.join(datadir, m.system, 'result_' + m.mode + '.mat')
        hdf5storage.savemat(filedir, d, oned_as='column')
    return True


def runsolver(m, solver=None, max_iter=int(1e4), printout=True, \
    debug=False, nlp_solver='ipopt', mip_solver='gurobi', 
    sub_solver='ipopt', mip_solver_tee=False, nlp_solver_tee=False,
    check_scaling=False, approx_hessian=False, specify_user_scaling=False):
    
    # Solver options
    options = {}
    options['max_iter'] = max_iter
    if specify_user_scaling:
        options['nlp_scaling_method'] = 'user-scaling'
    if solver in ['MA27', 'MA57', 'MA97', 'mumps']:
        options['linear_solver'] = solver
    if debug:
        options['halt_on_ampl_error'] = 'yes'
    if approx_hessian:
        options['hessian_approximation'] = 'limited-memory'
        
    # Setting up solver
    if solver == None:
        raise('Solver must be specified')
    elif solver in ['ipopt', 'MA27', 'MA57', 'MA97', 'mumps']:
        m.solver = SolverFactory('ipopt')
        m.solver.options = options
    elif solver == 'multistart':
        m.solver = SolverFactory('multistart')
        m.solver.solver_args = options
    elif solver == 'shot':
        m.solver = SolverFactory('shot')
        m.solver.options['Subsolver.Ipopt.MaxIterations'] = max_iter
        if printout:
            m.solver.options['Output.Console.LogLevel'] = 1
        if mip_solver == 'cplex':
            m.solver.options['Dual.MIP.Solver'] = 0
        elif mip_solver == 'gurobi':
            m.solver.options['Dual.MIP.Solver'] = 1
        elif mip_solver == 'cbc':
            m.solver.options['Dual.MIP.Solver'] = 2
        else:
            raise('Unexpected behavior')
    elif solver == 'baron':
        sys.path.append(r'C:\GAMS\32\apifiles\Python\api_38')
        sys.path.append(r'C:\GAMS\32\apifiles\GAMS')
        m.solver = SolverFactory('gams', solver_io='direct', solver=solver)
    elif solver == 'scip':
        m.solver = SolverFactory('scip', executable=r'C:\Program Files\SCIPOptSuite 8.0.0\bin\scip.exe')
    else:
        m.solver = SolverFactory(solver)
    
    
    # Scaling and run
    if m.scaling:
        cprint('Model scaling is performed...')
        m_original = m.clone()
        m_scaled = scaler(m_original, check_scaling=check_scaling)
        
        cprint('Running the solver...')
        if solver == 'multistart':
            results = m_scaled.solver.solve(m_scaled, solver=sub_solver)
        elif solver == 'mindtpy':
            results = m_scaled.solver.solve(m_scaled, \
                mip_solver=mip_solver, nlp_solver=nlp_solver, tee=printout, 
                mip_solver_tee=mip_solver_tee, nlp_solver_tee=nlp_solver_tee,
                num_solution_iteration=10)
        elif solver == 'baron':
            results = m_scaled.solver.solve(m_scaled, tee=printout, solver=solver)
        else:
            results = m_scaled.solver.solve(m_scaled, tee=printout)
            
        cprint('Recovering back to the unscaled model')
        m = scaler(m_original, recover=True, m_scaled=m_scaled)
    else:
        cprint('Running the solver...')
        if solver == 'multistart':
            results = m.solver.solve(m, solver=sub_solver)
        elif solver == 'mindtpy':
            results = m.solver.solve(m, \
                mip_solver=mip_solver, nlp_solver=nlp_solver, tee=printout, 
                mip_solver_tee=mip_solver_tee, nlp_solver_tee=nlp_solver_tee,
                num_solution_iteration=10)
        elif solver == 'baron':
            results = m.solver.solve(m, tee=printout, solver=m.solver)
        else:
            results = m.solver.solve(m, tee=printout)
        
    return m, results
 

def scaler(m_original, recover=None, m_scaled=None, check_scaling=False):
    if recover == True:
        TransformationFactory('core.scale_model').propagate_solution(m_scaled, m_original)
        return m_original
    else:
        m_scaled = TransformationFactory('core.scale_model').create_using(m_original)
        return m_scaled
        

def bound_touch_detector(obj, idx, threshold):
    if isinstance(obj, Constraint):
        val = value(obj[idx])
        lb = value(obj[idx].lower) - 1e-3
        ub = value(obj[idx].upper) + 1e-3
    elif isinstance(obj, Var):
        val = value(obj[idx])
        lb = value(obj[idx].lb)
        ub = value(obj[idx].ub)
    else:
        raise('Unexpected behavior')

    bounds = False
    if (lb != None) and (ub != None):
        scale = ub - lb
        pos = (val - lb)/scale
        if pos < threshold:
            bounds = 'lower'
        elif pos > 1 - threshold:
            bounds = 'upper'
    elif (lb != None) and (ub == None):
        pos = val - lb
        if pos/abs(lb) < threshold:
            bounds = 'lower'
    elif (lb == None) and (ub != None):
        pos = ub - val
        if pos/abs(ub) > 1 - threshold:
            bounds = 'upper'
    else:
        raise('Unexpected behavior')

    if bounds:
        if isinstance(obj, Constraint):
            print('Constraint near ' + bounds + ' bound: ',
                  obj[idx].name, ' = ', value(obj[idx]))
        elif isinstance(obj, Var):
            name, vtype = get_root_var_name(obj)
            if vtype == 'log':
                print('Variable near ' + bounds + ' bound: ',
                    name, ' = ', exp(value(obj[idx])))
            else:
                print('Variable near ' + bounds + ' bound: ',
                    obj[idx].name, ' = ', value(obj[idx]))
    return bounds


def which_infeasible(m, threshold=1e-2, target='all'):
    touch_bounds = 0
    if target in ['all', 'var']:
        for var in m.component_objects(Var):
            if 'log_k_d' in var.name:
                a = 1
            for idx in var.keys():
                if not var[idx].fixed:
                    bounds = bound_touch_detector(var, idx, threshold)
                    if bounds:
                        touch_bounds = 1
    if target in ['all', 'con']:
        for con in m.component_objects(Constraint):
            for idx in con.keys():
                if con[idx].active:
                    bounds = bound_touch_detector(con, idx, threshold)    
                    if bounds:
                        touch_bounds = 1
    if target == 'param':
        k_growth_list, k_meta_list, k_viral_list = namespaces(m.system, 'k')
        params = k_growth_list + k_meta_list + k_viral_list
        for var in m.component_objects(Var):
            name, vtype = get_root_var_name(var)
            if name in params:
                bounds = bound_touch_detector(var, None, threshold)
                if bounds:
                    touch_bounds = 1
    return touch_bounds


def check_var_error(m, var, idx, verbose=False, apply_scaling=False):
    name, vtype = get_root_var_name(var)
    val_sim = find_sim_value_transformed(m, name, idx, vtype)
        
    detected = False
    val_calc = value(var[idx])

    if abs(val_sim) > m.eps:
        rerror = abs(val_calc - val_sim)/val_sim
        if rerror > 0.3:
            detected = True
    else:
        if abs(val_calc) > m.eps:
            detected = True
    if detected or verbose:
        if apply_scaling:
            scaling = m.scaling_factor[var]
            print('%s | real: %1.2g | cal: %1.2g'  % (var[idx].name, val_sim*scaling, val_calc*scaling))
        else:
            print('%s | real: %1.2g | cal: %1.2g'  % (var[idx].name, val_sim, val_calc))


def check_con_error(m, con, idx, verbose=False, apply_scaling=False):
    val = value(con[idx])
    detected = False
    if abs(val) > 1e-3:
        detected = True
    if detected or verbose:
        if apply_scaling:
            scaling = m.scaling_factor[con]
            print('%s | cal: %1.2g'  % (con[idx].name, val*scaling))
        else:
            print('%s | cal: %1.2g'  % (con[idx].name, val))


def compare_with_sim(m, target='all', verbose=False, apply_scaling=False):  # compare with simulation value
    X_list = namespaces(m.system, target='X')

    if target in ['all', 'var']: 
        for var in m.component_objects(Var):
            if not var.name in X_list:
                continue
            for idx in var.keys():
                check_var_error(m, var, idx, verbose, apply_scaling)
        for expr in m.component_objects(Expression):
            if not expr.name in X_list:
                continue
            for idx in expr.keys():
                check_var_error(m, expr, idx, verbose, apply_scaling)
    if target in ['all', 'con']:
        for con in m.component_objects(Constraint):
            for idx in con.keys():
                check_con_error(m, con, idx, verbose, apply_scaling)
    return


def cprint(string, color='green'):
    print(colored(string, color))
    

def plot(var):
    r = var.root_block()
    t = list(r.t.ordered_data())
    colors = pl.cm.winter(np.linspace(0,1,len(r.i)))
    if var == r:  # display all variables
        X_list = namespaces(r.system, 'X')
        m = 4
        n = 11
        X_list = np.reshape(X_list, (m, n))
        fig1, axs1 = plt.subplots(m, n)
        for i in range(m):
            for j in range(n):
                name = X_list[i,j]
                try:
                    var = r.find_component(name)
                    val = to_array(var)
                    for k in r.i:
                        y = val[:,k-1]
                        axs1[i,j].plot(t, y, color=colors[k-1])
                except:
                    try:
                        var = r.find_component('log_' + name)
                        val = to_array(var)
                        for k in r.i:
                            y = exp(val[:,k-1])
                            axs1[i,j].plot(t, y, color=colors[k-1])
                    except:
                        print('model has no ' + name)
                axs1[i,j].set_title(name)

        R_list = namespaces(r.system, 'R')
        m = 3
        n = 5
        R_list = np.reshape(R_list, (m, n))
        fig2, axs2 = plt.subplots(m, n)
        for i in range(m):
            for j in range(n):
                name = R_list[i,j]
                try:
                    var = r.find_component(name)
                    val = to_array(var)
                    for k in r.i:
                        y = val[:,k-1]
                        axs2[i,j].plot(t, y, color=colors[k-1])
                except:
                    try:
                        var = r.find_component('log_' + name)
                        val = to_array(var)
                        for k in r.i:
                            y = exp(val[:,k-1])
                            axs2[i,j].plot(t, y, color=colors[k-1])
                    except:
                        print('model has no ' + name)
                axs2[i,j].set_title(name)
        plt.show()
    else:
        name = get_name(var)
        val = to_array(var)

        for i in r.i:
            y = val[:,i-1]
            plt.plot(t, y, color=colors[i-1])
        plt.show()


if __name__ == '__main__':
    wdir, rdir, ddir = set_path(os.path.abspath(__file__))
    a = 1
    
    