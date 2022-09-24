import os
import numpy as np
import hdf5storage
from pyomo.environ import *
from pyomo.dae import *
from pyomo.environ import NonNegativeReals as NNReals
from config import find_bounds, get_root_var_name, find_sim_value, \
    get_scaling_val, namespaces
from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))

def States(m):
    if m.bound_source == 'manual':
        X_min = find_bounds(m, based_on='manual', arg='X_min')
        X_max = find_bounds(m, based_on='manual', arg='X_max')
    elif m.bound_source == 'sim':
        X_min = find_bounds(m, based_on='sim', arg='X_min')
        X_max = find_bounds(m, based_on='sim', arg='X_max')

    # Cell growth
    m.Xv = Var(m.t, m.i, bounds=(X_min[0], X_max[0]), initialize=0)
    m.Xd = Var(m.t, m.i, bounds=(X_min[1], X_max[1]), initialize=0)
    
    if m.meta:
        # Metabolites
        m.Glc = Var(m.t, m.i, bounds=(X_min[2], X_max[2]), initialize=0)
        m.Lac = Var(m.t, m.i, bounds=(X_min[3], X_max[3]), initialize=0)
        m.AlaGln = Var(m.t, m.i, bounds=(X_min[4], X_max[4]), initialize=0)
        m.Gln = Var(m.t, m.i, bounds=(X_min[5], X_max[5]), initialize=0)
        m.Amm = Var(m.t, m.i, bounds=(X_min[6], X_max[6]), initialize=0)
        m.Glu = Var(m.t, m.i, bounds=(X_min[7], X_max[7]), initialize=0)

    # Transfection reagents
    m.pPack_media = Var(m.t, m.i, bounds=(X_min[8], X_max[8]), initialize=0)
    m.pPack_vcell = Var(m.t, m.i, bounds=(X_min[9], X_max[9]), initialize=0)
    m.pPack_cell = Var(m.t, m.i, bounds=(X_min[10], X_max[10]), initialize=0)
    m.pVec_media = Var(m.t, m.i, bounds=(X_min[11], X_max[11]), initialize=0)
    m.pVec_vcell = Var(m.t, m.i, bounds=(X_min[12], X_max[12]), initialize=0)
    m.pVec_cell = Var(m.t, m.i, bounds=(X_min[13], X_max[13]), initialize=0)
    m.pHelp_media = Var(m.t, m.i, bounds=(X_min[14], X_max[14]), initialize=0)
    m.pHelp_vcell = Var(m.t, m.i, bounds=(X_min[15], X_max[15]), initialize=0)
    m.pHelp_cell = Var(m.t, m.i, bounds=(X_min[16], X_max[16]), initialize=0)

    if m.viralprod:
        # Trafficking
        m.pPack_endo = Var(m.t, m.i, bounds=(X_min[17], X_max[17]), initialize=0)
        m.pPack_cyto = Var(m.t, m.i, bounds=(X_min[18], X_max[18]), initialize=0)
        m.pPack_nuc = Var(m.t, m.i, bounds=(X_min[19], X_max[19]), initialize=0)
        m.pVec_endo = Var(m.t, m.i, bounds=(X_min[20], X_max[20]), initialize=0)
        m.pVec_cyto = Var(m.t, m.i, bounds=(X_min[21], X_max[21]), initialize=0)
        m.pVec_nuc = Var(m.t, m.i, bounds=(X_min[22], X_max[22]), initialize=0)
        m.pHelp_endo = Var(m.t, m.i, bounds=(X_min[23], X_max[23]), initialize=0)
        m.pHelp_cyto = Var(m.t, m.i, bounds=(X_min[24], X_max[24]), initialize=0)
        m.pHelp_nuc = Var(m.t, m.i, bounds=(X_min[25], X_max[25]), initialize=0)
        
        # Viral production
        m.Rep = Var(m.t, m.i, bounds=(X_min[26], X_max[26]), initialize=0)
        m.Rep_cell = Var(m.t, m.i, bounds=(X_min[27], X_max[27]), initialize=0)
        m.VP = Var(m.t, m.i, bounds=(X_min[28], X_max[28]), initialize=0)
        m.vDNA = Var(m.t, m.i, bounds=(X_min[29], X_max[29]), initialize=0)
        m.rDNA_cell = Var(m.t, m.i, bounds=(X_min[30], X_max[30]), initialize=0)
        m.eCap_nuc = Var(m.t, m.i, bounds=(X_min[31], X_max[31]), initialize=0)
        m.eCap_cyto = Var(m.t, m.i, bounds=(X_min[32], X_max[32]), initialize=0)
        m.eCap_cell = Var(m.t, m.i, bounds=(X_min[33], X_max[33]), initialize=0)
        m.eCap_media = Var(m.t, m.i, bounds=(X_min[34], X_max[34]), initialize=0)
        m.fCap_nuc = Var(m.t, m.i, bounds=(X_min[35], X_max[35]), initialize=0)
        m.fCap_cyto = Var(m.t, m.i, bounds=(X_min[36], X_max[36]), initialize=0)
        m.fCap_cell = Var(m.t, m.i, bounds=(X_min[37], X_max[37]), initialize=0)
        m.fCap_media = Var(m.t, m.i, bounds=(X_min[38], X_max[38]), initialize=0)
        m.Cap_nuc = Var(m.t, m.i, bounds=(X_min[39], X_max[39]), initialize=0)
        m.Cap_cyto = Var(m.t, m.i, bounds=(X_min[40], X_max[40]), initialize=0)
        m.Cap_cell = Var(m.t, m.i, bounds=(X_min[41], X_max[41]), initialize=0)
        m.Cap_media = Var(m.t, m.i, bounds=(X_min[42], X_max[42]), initialize=0)
    
    # Media
    if m.system == 'perfusion':
        m.V = Var(m.t, m.i, bounds=(200, 350), initialize=300)
    
    if m.bound_source == 'manual':
        log_R_min = find_bounds(m, based_on='manual', arg='log_R_min')
        log_R_max = find_bounds(m, based_on='manual', arg='log_R_max')
    elif m.bound_source == 'sim':
        log_R_min = find_bounds(m, based_on='sim', arg='log_R_min')
        log_R_max = find_bounds(m, based_on='sim', arg='log_R_max')

    # Algebraic equations (rates)
    m.log_P_vcell_S = Var(m.t, m.i, bounds=(log_R_min[0], log_R_max[0]), initialize=0)
    m.log_P_media = Var(m.t, m.i, bounds=(log_R_min[1], log_R_max[1]), initialize=0)
    m.log_mu = Var(m.t, m.i, bounds=(log_R_min[2], log_R_max[2]), initialize=0)
    m.log_omega = Var(m.t, m.i, bounds=(log_R_min[3], log_R_max[3]), initialize=0)
    if m.meta:
        m.log_q_Glc = Var(m.t, m.i, bounds=(log_R_min[4], log_R_max[4]), initialize=0)
        m.log_q_Lac = Var(m.t, m.i, bounds=(log_R_min[5], log_R_max[5]), initialize=0)
        m.log_q_Gln = Var(m.t, m.i, bounds=(log_R_min[6], log_R_max[6]), initialize=0)
        m.log_q_Amm = Var(m.t, m.i, bounds=(log_R_min[7], log_R_max[7]), initialize=0)
        m.log_q_Glu = Var(m.t, m.i, bounds=(log_R_min[8], log_R_max[8]), initialize=0)
    m.log_r_pPack_uptk = Var(m.t, m.i, bounds=(log_R_min[9], log_R_max[9]), initialize=0)
    m.log_r_pVec_uptk = Var(m.t, m.i, bounds=(log_R_min[10], log_R_max[10]), initialize=0)
    m.log_r_pHelp_uptk = Var(m.t, m.i, bounds=(log_R_min[11], log_R_max[11]), initialize=0)
    if m.viralprod:
        m.log_r_Rep = Var(m.t, m.i, bounds=(log_R_min[12], log_R_max[12]), initialize=0)
        m.log_r_VP = Var(m.t, m.i, bounds=(log_R_min[13], log_R_max[13]), initialize=0)
        m.log_r_vDNA = Var(m.t, m.i, bounds=(log_R_min[14], log_R_max[14]), initialize=0)
        m.log_r_Pack = Var(m.t, m.i, bounds=(log_R_min[15], log_R_max[15]), initialize=0)
    
    # Transformation
    m.P_vcell_S = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_P_vcell_S[t,i]) - m.alpha)
    m.P_media = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_P_media[t,i]) - m.alpha)
    m.mu = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_mu[t,i]) - m.alpha)
    m.omega = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_omega[t,i]) - m.alpha)
    if m.meta:
        m.q_Glc = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_q_Glc[t,i]) - m.alpha)
        m.q_Lac = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_q_Lac[t,i]) - m.alpha)
        m.q_Gln = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_q_Gln[t,i]) - m.alpha)
        m.q_Amm = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_q_Amm[t,i]) - m.alpha)
        m.q_Glu = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_q_Glu[t,i]) - m.alpha)
    m.r_pPack_uptk = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_r_pPack_uptk[t,i]) - m.alpha)
    m.r_pVec_uptk = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_r_pVec_uptk[t,i]) - m.alpha)
    m.r_pHelp_uptk = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_r_pHelp_uptk[t,i]) - m.alpha)
    if m.viralprod:
        m.r_Rep = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_r_Rep[t,i]) - m.alpha)
        m.r_VP = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_r_VP[t,i]) - m.alpha)
        m.r_vDNA = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_r_vDNA[t,i]) - m.alpha)
        m.r_Pack = Expression(m.t, m.i, rule=lambda m, t, i: exp(m.log_r_Pack[t,i]) - m.alpha)

    # Manipulation variables
    if m.bound_source == 'manual':
        U_min = find_bounds(m, based_on='manual', arg='U_min')
        U_max = find_bounds(m, based_on='manual', arg='U_max')
    elif m.bound_source == 'sim':
        U_min = find_bounds(m, based_on='sim', arg='U_min')
        U_max = find_bounds(m, based_on='sim', arg='U_max')

    if m.system == 'bioreactor':
        m.Vb = Var(m.t, m.i, bounds=(0, 40), initialize=0)
        m.Fb = Var(m.t, m.i, bounds=(0, 50), initialize=0)
        m.Fh = Var(m.t, m.i, bounds=(0, 50), initialize=0)
        m.Fi = Var(m.t, m.i, bounds=(0, 50), initialize=0)
    elif m.system == 'sflask':
        m.Glc_replace = Var(m.t, m.i, bounds=(U_min[0], U_max[0]), initialize=0)
        m.Gln_replace = Var(m.t, m.i, bounds=(U_min[1], U_max[1]), initialize=0)
        m.pPack_add = Var(m.t, m.i, bounds=(U_min[2], U_max[2]), initialize=0)
        m.pVec_add = Var(m.t, m.i, bounds=(U_min[3], U_max[3]), initialize=0)
        m.pHelp_add = Var(m.t, m.i, bounds=(U_min[4], U_max[4]), initialize=0)

    # Derivatives (give bounds for scaling and convergence)
    if m.bound_source == 'manual':
        d_absX_max = find_bounds(m, based_on='manual', arg='d_absX_max')
    elif m.bound_source == 'sim':
        d_absX_max = find_bounds(m, based_on='sim', arg='d_absX_max')

    m.d_Xv = DerivativeVar(m.Xv, wrt=m.t, bounds=(-d_absX_max[0], d_absX_max[0]), initialize=0)
    m.d_Xd = DerivativeVar(m.Xd, wrt=m.t, bounds=(-d_absX_max[1], d_absX_max[1]), initialize=0)
    if m.meta:
        m.d_Glc = DerivativeVar(m.Glc, wrt=m.t, bounds=(-d_absX_max[2], d_absX_max[2]), initialize=0)
        m.d_Lac = DerivativeVar(m.Lac, wrt=m.t, bounds=(-d_absX_max[3], d_absX_max[3]), initialize=0)
        m.d_AlaGln = DerivativeVar(m.AlaGln, wrt=m.t, bounds=(-d_absX_max[4], d_absX_max[4]), initialize=0)
        m.d_Gln = DerivativeVar(m.Gln, wrt=m.t, bounds=(-d_absX_max[5], d_absX_max[5]), initialize=0)
        m.d_Amm = DerivativeVar(m.Amm, wrt=m.t, bounds=(-d_absX_max[6], d_absX_max[6]), initialize=0)
        m.d_Glu = DerivativeVar(m.Glu, wrt=m.t, bounds=(-d_absX_max[7], d_absX_max[7]), initialize=0)
    m.d_pPack_media = DerivativeVar(m.pPack_media, wrt=m.t, bounds=(-d_absX_max[8], d_absX_max[8]), initialize=0)
    m.d_pPack_vcell = DerivativeVar(m.pPack_vcell, wrt=m.t, bounds=(-d_absX_max[9], d_absX_max[9]), initialize=0)
    m.d_pPack_cell = DerivativeVar(m.pPack_cell, wrt=m.t, bounds=(-d_absX_max[10], d_absX_max[10]), initialize=0)
    m.d_pVec_media = DerivativeVar(m.pVec_media, wrt=m.t, bounds=(-d_absX_max[11], d_absX_max[11]), initialize=0)
    m.d_pVec_vcell = DerivativeVar(m.pVec_vcell, wrt=m.t, bounds=(-d_absX_max[12], d_absX_max[12]), initialize=0)
    m.d_pVec_cell = DerivativeVar(m.pVec_cell, wrt=m.t, bounds=(-d_absX_max[13], d_absX_max[13]), initialize=0)
    m.d_pHelp_media = DerivativeVar(m.pHelp_media, wrt=m.t, bounds=(-d_absX_max[14], d_absX_max[14]), initialize=0)
    m.d_pHelp_vcell = DerivativeVar(m.pHelp_vcell, wrt=m.t, bounds=(-d_absX_max[15], d_absX_max[15]), initialize=0)
    m.d_pHelp_cell = DerivativeVar(m.pHelp_cell, wrt=m.t, bounds=(-d_absX_max[16], d_absX_max[16]), initialize=0)
    if m.viralprod:
        m.d_pPack_endo = DerivativeVar(m.pPack_endo, wrt=m.t, bounds=(-d_absX_max[17], d_absX_max[17]), initialize=0)
        m.d_pPack_cyto = DerivativeVar(m.pPack_cyto, wrt=m.t, bounds=(-d_absX_max[18], d_absX_max[18]), initialize=0)
        m.d_pPack_nuc = DerivativeVar(m.pPack_nuc, wrt=m.t, bounds=(-d_absX_max[19], d_absX_max[19]), initialize=0)
        m.d_pVec_endo = DerivativeVar(m.pVec_endo, wrt=m.t, bounds=(-d_absX_max[20], d_absX_max[20]), initialize=0)
        m.d_pVec_cyto = DerivativeVar(m.pVec_cyto, wrt=m.t, bounds=(-d_absX_max[21], d_absX_max[21]), initialize=0)
        m.d_pVec_nuc = DerivativeVar(m.pVec_nuc, wrt=m.t, bounds=(-d_absX_max[22], d_absX_max[22]), initialize=0)
        m.d_pHelp_endo = DerivativeVar(m.pHelp_endo, wrt=m.t, bounds=(-d_absX_max[23], d_absX_max[23]), initialize=0)
        m.d_pHelp_cyto = DerivativeVar(m.pHelp_cyto, wrt=m.t, bounds=(-d_absX_max[24], d_absX_max[24]), initialize=0)
        m.d_pHelp_nuc = DerivativeVar(m.pHelp_nuc, wrt=m.t, bounds=(-d_absX_max[25], d_absX_max[25]), initialize=0)
        m.d_Rep = DerivativeVar(m.Rep, wrt=m.t, bounds=(-d_absX_max[26], d_absX_max[26]), initialize=0)
        m.d_Rep_cell = DerivativeVar(m.Rep_cell, wrt=m.t, bounds=(-d_absX_max[27], d_absX_max[27]), initialize=0)
        m.d_VP = DerivativeVar(m.VP, wrt=m.t, bounds=(-d_absX_max[28], d_absX_max[28]), initialize=0)
        m.d_vDNA = DerivativeVar(m.vDNA, wrt=m.t, bounds=(-d_absX_max[29], d_absX_max[29]), initialize=0)
        m.d_rDNA_cell = DerivativeVar(m.rDNA_cell, wrt=m.t, bounds=(-d_absX_max[30], d_absX_max[30]), initialize=0)
        m.d_eCap_nuc = DerivativeVar(m.eCap_nuc, wrt=m.t, bounds=(-d_absX_max[31], d_absX_max[31]), initialize=0)
        m.d_eCap_cyto = DerivativeVar(m.eCap_cyto, wrt=m.t, bounds=(-d_absX_max[32], d_absX_max[32]), initialize=0)
        m.d_eCap_cell = DerivativeVar(m.eCap_cell, wrt=m.t, bounds=(-d_absX_max[33], d_absX_max[33]), initialize=0)
        m.d_eCap_media = DerivativeVar(m.eCap_media, wrt=m.t, bounds=(-d_absX_max[34], d_absX_max[34]), initialize=0)
        m.d_fCap_nuc = DerivativeVar(m.fCap_nuc, wrt=m.t, bounds=(-d_absX_max[35], d_absX_max[35]), initialize=0)
        m.d_fCap_cyto = DerivativeVar(m.fCap_cyto, wrt=m.t, bounds=(-d_absX_max[36], d_absX_max[36]), initialize=0)
        m.d_fCap_cell = DerivativeVar(m.fCap_cell, wrt=m.t, bounds=(-d_absX_max[37], d_absX_max[37]), initialize=0)
        m.d_fCap_media = DerivativeVar(m.fCap_media, wrt=m.t, bounds=(-d_absX_max[38], d_absX_max[38]), initialize=0)
        m.d_Cap_nuc = DerivativeVar(m.Cap_nuc, wrt=m.t, bounds=(-d_absX_max[39], d_absX_max[39]), initialize=0)
        m.d_Cap_cyto = DerivativeVar(m.Cap_cyto, wrt=m.t, bounds=(-d_absX_max[40], d_absX_max[40]), initialize=0)
        m.d_Cap_cell = DerivativeVar(m.Cap_cell, wrt=m.t, bounds=(-d_absX_max[41], d_absX_max[41]), initialize=0)
        m.d_Cap_media = DerivativeVar(m.Cap_media, wrt=m.t, bounds=(-d_absX_max[42], d_absX_max[42]), initialize=0)

    if m.system == 'bioreactor':
        m.d_V = DerivativeVar(m.V, wrt=m.t, bounds=(-1e3, 1e3), initialize=0)


def find_sim_value_transformed(m, name, idx, var_type):
    if var_type == 'raw':
        val = find_sim_value(m, name, idx)
    elif var_type == 'log':
        val = log(find_sim_value(m, name, idx) + m.alpha)
    elif var_type == 'd_log':
        val1 = find_sim_value(m, name, idx)
        val2 = find_sim_value(m, name.split('d_')[1], idx)
        val = val1/(val2 + m.alpha)
    return val


def initialize_states(m):
    names = [obj.name for obj in list(m.component_objects())]
    if not any(['disc_eq' in name for name in names]):
        raise('Initialization should be placed after discretization!')
    
    X_list = namespaces(m.system, target='X')
    dX_list = namespaces(m.system, target='dX')
    R_list = namespaces(m.system, target='R')
    U_list = namespaces(m.system, target='U')
    for VarType in [Var, DerivativeVar]:
        for var in m.component_objects(VarType):
            # Checker
            name, var_type = get_root_var_name(var)
            if not name:
                print(var.name, ' is not state related variable. Skip initialization.')
                continue
            if not name in X_list + R_list + dX_list + U_list:
                continue

            for idx in var.keys():
                val = find_sim_value_transformed(m, name, idx, var_type)                
                var[idx].set_value(val)
    return


def save_X0(m):
    d = {}
    d['Xv'] = np.array(value(m.Xv[0,:]))
    d['Xd'] = np.array(value(m.Xd[0,:]))
    filedir = os.path.join(datadir, 'parameter', 'X0_opt.mat')
    hdf5storage.savemat(filedir, d, oned_as='column')
    return