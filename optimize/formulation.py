import os
import numpy as np
import hdf5storage
from pprint import pprint
from pyomo.environ import *
from pyomo.core.base.expression import IndexedExpression
from pyomo.core.base.var import IndexedVar
from pyomo.dae.diffvar import DerivativeVar
from pyomo.core.base.var import ScalarVar
from pyomo.core.base.expression import ScalarExpression
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
from idaes.core.util.model_statistics import degrees_of_freedom as dof

from config import get_root_var_name, load_data, get_scaling_val, namespaces, is_before_perfusion
from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))



def fCap_cum_yield(m):
    n = 1
    cumsum = 0
    for t in m.t:
        if t <= m.t[-1]:
            try:
                dt = m.t[n+1] - m.t[n]
            except:
                dt = m.t[n] - m.t[n-1]
            for i in m.i:
                cumsum += (m.fCap_media[t,i]*(m.Fh[t,i] + m.Fb[t,i]) \
                    + m.fCap_cell[t,i]*m.Fb[t,i])*dt
            n += 1
    return cumsum


def fCap_cum_purity(m):  # TODO: consider NaN case
    n = 1
    cumsum1 = 0
    cumsum2 = 0
    for t in m.t:
        if t <= m.t[-1]:
            try:
                dt = m.t[n+1] - m.t[n]
            except:
                dt = m.t[n] - m.t[n-1]
            for i in m.i:
                cumsum1 += m.fCap_media[t,i]*(m.Fh[t,i] + m.Fb[t,i])*dt
                cumsum2 += m.eCap_media[t,i]*(m.Fh[t,i] + m.Fb[t,i])*dt
            n += 1
    return cumsum1/(cumsum1 + cumsum2)


def MSE(m):
    X_exp, X_index = load_data(based_on='exp', system=m.system)

    idx_Exp = X_index['Exp']
    idx_time = X_index['time']
    mse = 0
    n = 0
    for X_name in X_index:
        if X_name in ['Xv', 'Xd', 'pPack_cell', 'pVec_cell', 'pHelp_cell', 
                      'fCap_cell', 'fCap_media', 'eCap_cell', 'eCap_media', 
                      'Rep_cell', 'rDNA_cell', 
                      'Glc', 'Lac', 'Gln', 'Amm', 'Glu']:
            j = X_index[X_name]
            var = m.find_component(X_name)
            if not hasattr(var, 'name'):
                print('formulation::MSE: variable ', X_name, ' is not defined in the model. Skip.')
                continue
            for i in range(0, X_exp.shape[0]):
                t = X_exp[i, idx_time]
                Exp = X_exp[i, idx_Exp]
                if not Exp in m.i:
                    continue
                val = X_exp[i, j]
                if np.isnan(val) or Exp == 0:
                    continue
                if val == 0:
                    mse += (var[t, Exp] - val)**2
                else:
                    mse += ((var[t, Exp] - val)/val)**2
                n += 1
    return mse/n


def L1(m):
    l1 = 0
    for t in m.t:
        for i in m.i:
            l1 += sqrt(m.Vb[t,i]**2)
    return l1


def set_objective(m, target=None):
    if m.mode == 'simulation':
        m.obj = Objective(expr=0, sense=minimize)

    elif m.mode == 'optimization':
        if m.objective == 'cum_yield':
            m.obj = Objective(expr=log(fCap_cum_yield(m)), sense=maximize)

        elif m.objective == 'cum_purity':
            m.obj = Objective(expr=log(fCap_cum_purity(m)), sense=maximize)

        elif m.objective == 'pareto':
            m.obj = Objective(expr=w1*log(fCap_cum_yield(m)) + w2*log(fCap_cum_purity(m)), sense=maximize)

    elif m.mode == 'estimation':
        m.obj = Objective(expr=MSE(m), sense=minimize)
        
    else:
        raise('Unexpected behavior!')


def fix_vars_cons(m, target=None):
    X_list = namespaces(m.system, target='X')
    X_viral_list = namespaces(m.system, target='X_viral')
    dX_list = namespaces(m.system, target='dX')
    R_list = namespaces(m.system, target='R')
    U_list = namespaces(m.system, target='U')
    eqn_viral_list = namespaces(m.system, target='eqn_viral_list')
    
    # Mode dependent
    if m.mode == 'simulation':
        for var in m.component_objects(Var):
            if var.name in U_list:
                var.fix()
    
    # System dependent
    system = m.system
    if system == 'sflask':
        for var in m.component_objects(Var):
            name, _ = get_root_var_name(var)

            if name in X_list:  # initial condition
                for idx in var.keys():
                    if idx[0] == 0:
                        var[idx].fix()
            
            if var.name in U_list:  # no manipulation
                for idx in var.keys():
                    var[idx].fix()
                    val = var[idx].value
                    
                    if m.meta and 'replace' in var.name and val != 0:
                        vname = var.name.split('_')[0]
                        v = m.find_component(vname)
                        v[idx].fix(val)
                        if 'Glc' in var.name:
                            m.eqn3[idx].deactivate()
                        elif 'Gln' in var.name:
                            m.eqn6[idx].deactivate()
                        else:
                            raise('Unexpected behavior')
        # no AlaGln
        if m.meta:
            m.AlaGln.fix(0)
            m.eqn5.deactivate()
            for idx in m.d_AlaGln.keys():
                m.d_AlaGln[idx].fix(0)
            m.d_AlaGln_disc_eq.deactivate()

    elif system == 'bioreactor':
        for var in m.component_objects(Var):
            name, _ = get_root_var_name(var)
            if name in X_list + dX_list + R_list:  # initial condition
                for idx in var.keys():
                    if is_before_perfusion(m, idx):
                        var[idx].fix()
            if name in U_list:
                for idx in var.keys():
                    if is_before_perfusion(m, idx):
                        var[idx].fix(0)
                    if name == 'Vb' and not idx[0] in m.Ts:
                        var[idx].fix(0)

        if m.mode == 'optimization' and m.level_control:
            m.eqn_lc = Constraint(m.t, m.i, rule=lambda m, t, i: m.V[t,i] == 300)                
            m.scaling_factor[m.eqn_lc] = 1/300
        
        if m.mode == 'optimization':
            m.eqn_Vb_max = Constraint(m.i, rule=eqn_Vb_max)
            m.scaling_factor[m.eqn_Vb_max] = 1/100

        for con in m.component_objects(Constraint):
            if con.name == 'eqn_Vb_max':
                continue
            for idx in con.keys():
                if is_before_perfusion(m, idx):
                    con[idx].deactivate()
    return


def eqn_Vb_max(m, i):
    expr = 0
    for t in m.t:
        if t < m.Ti:
            continue
        expr += m.Vb[t,i]
    return expr <= 20


def maintain_action(m, U_list):
    m.eqn_maintain_action = ConstraintList()
    for var in m.component_objects(Var):
        if not (var.name in U_list):
            continue
        for idx in var.keys():
            t = idx[0]
            i = idx[1]
            if is_before_perfusion(m, idx):
                continue
            shift = t % m.Td
            if shift == 0:
                continue
            tt = m.t.prev(t, shift)
            m.eqn_maintain_action.add(var[t,i] == var[tt,i])
    

def scale_model(m):
    if not m.scaling:
        return
    else:
        m.scaling_factor = Suffix(direction=Suffix.EXPORT)
    
    # Variables
    for Objects in [Var, DerivativeVar, Expression]:
        for obj in m.component_objects(Objects):
            m.scaling_factor[obj] = 1/get_scaling_val(obj)

    # Discretization constraints
    w = 1/1e6
    for con in m.component_objects(Constraint):
        if not 'disc_eq' in con.name:
            continue
        vname = con.name.split('_disc_eq')[0]
        var = m.find_component(vname)
        val = max([var[idx].upper for idx in var.keys()])
        if val == 0:
            print('Divide by zero detected for ', con.name, '! Scaling factor is set to 1.')
            m.scaling_factor[con] = w
        else:
            m.scaling_factor[con] = w/val
    
    # DAE constraints
    w = 1
    m.scaling_factor[m.eqn1] = w/get_scaling_val(m.d_Xv)
    m.scaling_factor[m.eqn2] = w/get_scaling_val(m.d_Xd)
    if m.meta:
        m.scaling_factor[m.eqn3] = w/get_scaling_val(m.d_Glc)
        m.scaling_factor[m.eqn4] = w/get_scaling_val(m.d_Lac)
        m.scaling_factor[m.eqn5] = w/get_scaling_val(m.d_AlaGln)
        m.scaling_factor[m.eqn6] = w/get_scaling_val(m.d_Gln)
        m.scaling_factor[m.eqn7] = w/get_scaling_val(m.d_Amm)
        m.scaling_factor[m.eqn8] = w/get_scaling_val(m.d_Glu)
    m.scaling_factor[m.eqn9] = w/get_scaling_val(m.d_pPack_media)
    m.scaling_factor[m.eqn10] = w/get_scaling_val(m.d_pPack_vcell)
    m.scaling_factor[m.eqn11] = w/get_scaling_val(m.d_pPack_cell)
    m.scaling_factor[m.eqn12] = w/get_scaling_val(m.d_pVec_media)
    m.scaling_factor[m.eqn13] = w/get_scaling_val(m.d_pVec_vcell)
    m.scaling_factor[m.eqn14] = w/get_scaling_val(m.d_pVec_cell)
    m.scaling_factor[m.eqn15] = w/get_scaling_val(m.d_pHelp_media)
    m.scaling_factor[m.eqn16] = w/get_scaling_val(m.d_pHelp_vcell)
    m.scaling_factor[m.eqn17] = w/get_scaling_val(m.d_pHelp_cell)

    if m.viralprod:
        m.scaling_factor[m.eqn18] = w/get_scaling_val(m.d_pPack_endo)
        m.scaling_factor[m.eqn19] = w/get_scaling_val(m.d_pPack_cyto)
        m.scaling_factor[m.eqn20] = w/get_scaling_val(m.d_pPack_nuc)
        m.scaling_factor[m.eqn21] = w/get_scaling_val(m.d_pVec_endo)
        m.scaling_factor[m.eqn22] = w/get_scaling_val(m.d_pVec_cyto)
        m.scaling_factor[m.eqn23] = w/get_scaling_val(m.d_pVec_nuc)
        m.scaling_factor[m.eqn24] = w/get_scaling_val(m.d_pHelp_endo)
        m.scaling_factor[m.eqn25] = w/get_scaling_val(m.d_pHelp_cyto)
        m.scaling_factor[m.eqn26] = w/get_scaling_val(m.d_pHelp_nuc)
        m.scaling_factor[m.eqn27] = w/get_scaling_val(m.d_Rep)
        m.scaling_factor[m.eqn28] = w/get_scaling_val(m.d_Rep_cell)
        m.scaling_factor[m.eqn29] = w/get_scaling_val(m.d_VP)
        m.scaling_factor[m.eqn30] = w/get_scaling_val(m.d_vDNA)
        m.scaling_factor[m.eqn31] = w/get_scaling_val(m.d_rDNA_cell)
        m.scaling_factor[m.eqn32] = w/get_scaling_val(m.d_eCap_nuc)
        m.scaling_factor[m.eqn33] = w/get_scaling_val(m.d_eCap_cyto)
        m.scaling_factor[m.eqn34] = w/get_scaling_val(m.d_eCap_cell)
        m.scaling_factor[m.eqn35] = w/get_scaling_val(m.d_eCap_media)
        m.scaling_factor[m.eqn36] = w/get_scaling_val(m.d_fCap_nuc)
        m.scaling_factor[m.eqn37] = w/get_scaling_val(m.d_fCap_cyto)
        m.scaling_factor[m.eqn38] = w/get_scaling_val(m.d_fCap_cell)
        m.scaling_factor[m.eqn39] = w/get_scaling_val(m.d_fCap_media)
        m.scaling_factor[m.eqn40] = w/get_scaling_val(m.d_Cap_nuc)
        m.scaling_factor[m.eqn41] = w/get_scaling_val(m.d_Cap_cyto)        
        m.scaling_factor[m.eqn42] = w/get_scaling_val(m.d_Cap_cell)
        m.scaling_factor[m.eqn43] = w/get_scaling_val(m.d_Cap_media)
    if m.system == 'bioreactor':
        m.scaling_factor[m.eqn44] = w/get_scaling_val(m.d_V)
    m.scaling_factor[m.eqn45] = w/get_scaling_val(m.P_vcell_S)
    m.scaling_factor[m.eqn46] = w/get_scaling_val(m.P_media)
    m.scaling_factor[m.eqn47] = w/get_scaling_val(m.mu)
    m.scaling_factor[m.eqn48] = w/get_scaling_val(m.omega)
    if m.meta:
        m.scaling_factor[m.eqn49] = w/get_scaling_val(m.q_Glc)
        m.scaling_factor[m.eqn50] = w/get_scaling_val(m.q_Lac)
        m.scaling_factor[m.eqn51] = w/get_scaling_val(m.q_Gln)
        m.scaling_factor[m.eqn52] = w/get_scaling_val(m.q_Amm)
        m.scaling_factor[m.eqn53] = w/get_scaling_val(m.q_Glu)
    m.scaling_factor[m.eqn54] =w/get_scaling_val(m.r_pPack_uptk)
    m.scaling_factor[m.eqn55] = w/get_scaling_val(m.r_pVec_uptk)
    m.scaling_factor[m.eqn56] = w/get_scaling_val(m.r_pHelp_uptk)

    if m.viralprod:
        m.scaling_factor[m.eqn57] = w/get_scaling_val(m.r_Rep)
        m.scaling_factor[m.eqn58] = w/get_scaling_val(m.r_VP)
        m.scaling_factor[m.eqn59] = w/get_scaling_val(m.r_vDNA)
        m.scaling_factor[m.eqn60] = w/get_scaling_val(m.r_Pack)


def check_divide_by_zero(m):
    for con in m.component_objects(Constraint):
        for idx in con.keys():
            try:
                value(con[idx])
            except:
                con[idx].pprint()
                raise('Error occurred!')


def check_wrong_initialization(m):
    for var in m.component_objects(Var):
        for idx in var.keys():
            if var[idx] == -np.inf or var[idx] == np.inf:
                raise('Error occurred!')


def show_fixed(m):
    print('Fixed variables are:')
    for var in m.component_objects(Var):
        if type(var) in [IndexedVar, DerivativeVar]:
            for idx in var.keys():
                if var[idx].fixed:
                    print(var.name, idx)
        else:  # scalar
            if var.fixed:
                print(var.name, idx)
    return