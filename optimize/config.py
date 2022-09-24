import os
import numpy as np
import hdf5storage
from pyomo.core.base.expression import IndexedExpression
from pyomo.core.base.var import IndexedVar
from pyomo.dae.diffvar import DerivativeVar
from pyomo.core.base.var import ScalarVar
from pyomo.core.base.expression import ScalarExpression

from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))


def flatten(data):
    if data.shape == ():
        return [data.item().flatten()]
    else:
        for i, dat in enumerate(data):
            data[i] = dat.flatten()
        return data


def load_data(based_on, system):
    if based_on == 'sim':
        if not hasattr(load_data, 'X_sim'):
            filedir = os.path.join(datadir, system, 'data_sim.mat')
            print('load_data:: load simulation data from: ' + filedir)
            datapack = hdf5storage.loadmat(filedir)

            t_sim = datapack['t'].squeeze()
            t_sim = flatten(t_sim)

            tq = []  # depreciated 

            X_sim = datapack['X'].squeeze()
            R_sim = datapack['R'].squeeze()
            dX_sim = datapack['dX'].squeeze()
            U_sim = datapack['U'].squeeze()
            if system == 'perfusion':
                U_min = datapack['U_min'].squeeze()
                U_max = datapack['U_max'].squeeze()
            else:
                U_min = []
                U_max = []
            
            if X_sim.shape == ():
                X_sim = X_sim.flatten()
                R_sim = R_sim.flatten()
                dX_sim = dX_sim.flatten()
                U_sim = U_sim.flatten()
            
            load_data.tq = tq
            load_data.t_sim = t_sim
            load_data.X_sim = X_sim
            load_data.R_sim = R_sim
            load_data.dX_sim = dX_sim
            load_data.U_sim = U_sim
            load_data.U_min = U_min
            load_data.U_max = U_max
            return tq, t_sim, X_sim, R_sim, dX_sim, U_sim, U_min, U_max
        else:
            return load_data.tq, load_data.t_sim, load_data.X_sim, \
                load_data.R_sim, load_data.dX_sim, load_data.U_sim, \
                    load_data.U_min, load_data.U_max

    
    elif based_on == 'exp':
        if not hasattr(load_data, 'X_exp'):
            filedir = os.path.join(datadir, system, 'data.mat')
            datapack = hdf5storage.loadmat(filedir)
            header = datapack['header'].squeeze()
            X_exp = datapack['data_array'].squeeze()
            index = dict()
            for i, head in enumerate(header):
                index[head[0]] = i
            load_data.X_exp = X_exp
            load_data.index = index
            return X_exp, index
        else:
            return load_data.X_exp, load_data.index
    else:
        raise('Unexpected behavior!')


def get_scaling_val(obj):
    m = obj.root_block()
    if type(obj) in [IndexedVar, DerivativeVar]:  # upper X_bounds are already defined
        val = max([obj[idx].upper for idx in obj.keys()])

    elif type(obj) == IndexedExpression:  # need to find source term
        if 'd_' in obj.name:
            raw_var = obj.name.split('d_')[1]
            obj1 = m.find_component('d_log_' + raw_var)
            val1 = max([obj1[idx].upper for idx in obj1.keys()])
            obj2 = m.find_component('log_' + raw_var)
            val2 = max([np.exp(obj2[idx].upper) for idx in obj2.keys()])
            val = (val2 + m.alpha)*val1
        else:
            obj1 = m.find_component('log_' + obj.name)
            val = np.exp(max([obj1[idx].upper for idx in obj1.keys()])) - m.alpha
    
    elif type(obj) == ScalarVar:
        val = obj.upper

    elif type(obj) == ScalarExpression:
        obj1 = m.find_component('log_' + obj.name)
        val = np.exp(obj1.upper)

    if val == 0:
        print('Divide by zero detected for ', obj.name, '! Scaling factor is set to 1.')
        return 1
    else:
        return val


def is_before_perfusion(m, idx, obj=None):
    t = idx[0]
    if obj == None and t < m.Ti:
        return True
    else:
        return False



def find_best_case(m):
    if hasattr(find_best_case, 'i'):
        return find_best_case.i
    else:
        filedir = os.path.join(datadir, m.system, 'max_idx.mat')
        datapack = hdf5storage.loadmat(filedir)
        idx = datapack['idx'].squeeze()
        if m.objective == 'yield':
            find_best_case.i = idx[0]
        elif m.objective == 'purity':
            find_best_case.i = idx[1]
        else:
            raise('#TODO')
        return find_best_case.i


def find_sim_value(m, name, idx):
    X_list = namespaces(m.system, target='X')
    dX_list = namespaces(m.system, target='dX')
    R_list = namespaces(m.system, target='R')
    U_list = namespaces(m.system, target='U')
    tq, t_sim, X_sim, R_sim, dX_sim, U_sim, U_min, U_max = load_data(based_on='sim', system=m.system)

    t = idx[0]
    if m.system == 'perfusion' and m.mode == 'optimization':
        i = 1
    else:
        i = idx[1]

    if name in X_list:
        v = X_list.index(name)
        val = np.interp(t, t_sim[i-1], X_sim[i-1][:,v])
    elif name in R_list:
        v = R_list.index(name)
        val = np.interp(t, t_sim[i-1], R_sim[i-1][:,v])
    elif name in dX_list:
        v = dX_list.index(name)
        val = np.interp(t, t_sim[i-1], dX_sim[i-1][:,v])
    elif name in U_list:
        v = U_list.index(name)
        if m.system == 'sflask':
            try:
                tt = t_sim[i-1].tolist().index(t)
                val = U_sim[i-1][tt,v]
            except:
                val = 0.0
        else:
            val = np.interp(t, t_sim[i-1], U_sim[i-1][:,v])
    return val



def namespaces(system, target=None):
    if not target:
        raise('Unexpected behavior')
    
    # Initialize 
    if hasattr(namespaces, 'X_list'):
        X_list = namespaces.X_list
        X_viral_list = namespaces.X_viral_list
        dX_list = namespaces.dX_list
        R_list = namespaces.R_list
        k_growth_list = namespaces.k_growth_list
        k_meta_list = namespaces.k_meta_list
        k_viral_list = namespaces.k_viral_list
        U_list = namespaces.U_list
        eqn_viral_list = namespaces.eqn_viral_list
    else:
        X_list = ['Xv', 'Xd', 'Glc', 'Lac', 'AlaGln', 'Gln', 'Amm', 'Glu', 
                'pPack_media', 'pPack_vcell', 'pPack_cell', 
                'pVec_media', 'pVec_vcell', 'pVec_cell', 
                'pHelp_media', 'pHelp_vcell', 'pHelp_cell', 
                'pPack_endo', 'pPack_cyto', 'pPack_nuc', 
                'pVec_endo', 'pVec_cyto', 'pVec_nuc', 
                'pHelp_endo', 'pHelp_cyto', 'pHelp_nuc', 
                'Rep', 'Rep_cell', 'VP', 'vDNA', 'rDNA_cell', 
                'eCap_nuc', 'eCap_cyto', 'eCap_cell', 'eCap_media', 
                'fCap_nuc', 'fCap_cyto', 'fCap_cell', 'fCap_media', 
                'Cap_nuc', 'Cap_cyto', 'Cap_cell', 'Cap_media', 'V']
        X_viral_list = ['pPack_endo', 'pPack_cyto', 'pPack_nuc', 
                        'pVec_endo', 'pVec_cyto', 'pVec_nuc', 
                        'pHelp_endo', 'pHelp_cyto', 'pHelp_nuc', 
                        'Rep', 'Rep_cell', 'VP', 'vDNA', 'rDNA_cell', 
                        'eCap_nuc', 'eCap_cyto', 'eCap_cell', 'eCap_media', 
                        'fCap_nuc', 'fCap_cyto', 'fCap_cell', 'fCap_media', 
                        'Cap_nuc', 'Cap_cyto', 'Cap_cell', 'Cap_media']
        dX_list = ['d_' + X for X in X_list]
        R_list = ['P_vcell_S', 'P_media', 'mu', 'omega', 
                  'q_Glc', 'q_Lac', 'q_Gln', 'q_Amm', 'q_Glu',
                  'r_pPack_uptk', 'r_pVec_uptk', 'r_pHelp_uptk',
                  'r_Rep', 'r_VP', 'r_vDNA', 'r_Pack']
        k_growth_list = ['mu_max', 'K_PlsmdIhbt_cell', 'K_PlsmdIhbt_media', 
                         'K_muLac', 'k_d',  'k_PlsmdCyto_cell', 'K_PlsmdCyto_cell', 
                         'k_AmmCyto', 'K_AmmCyto', 'k_PlsmdDeg', 
                         'k_PlsmdUptk', 'K_PlsmdUptk_Xv', 'K_PlsmdUptk_mu', 'k_GlnDeg', 'k_GlnGen']
        k_meta_list = ['Y_Glc', 'm_Glc', 'K_GlcLac', 'Y_Lac', 'm_Lac', 
                       'k_Lac', 'K_Lacmu', 'Y_Gln', 'm_Gln', 'K_GlnGlc', 
                       'Y_Amm', 'm_Amm', 'K_AmmXv', 'Y_Glu', 'm_Glu', 
                       'k_Glu', 'K_Glumu', 'k_GlnDeg', 'k_GlnGen']
        k_viral_list = [ 'k_escape', 'k_nucEntry', 'k_Rep', 'k_expel', 
                         'k_RepDeg', 'K_Amm', 'K_Rep_pHelp', 'k_VP', 
                         'k_assembly', 'k_VPdeg', 'k_CapDeg', 'k_sec2cyto', 
                         'k_sec2media', 'K_vDNA_pHelp', 'k_vDNA', 
                         'K_vDNA_Rep', 'K_Pack_Rep', 'k_Pack', 
                         'k_vDNAdeg', 'K_vDNA_pVec', 'K_Pack_eCap', 'K_VP_Rep']

        if system == 'bioreactor':
            U_list = ['Vp', 'Fb', 'Fh', 'Fi']
        elif system == 'sflask':
            U_list = ['Glc_replace', 'Gln_replace', 'pPack_add', 'pVec_add', 'pHelp_add']

        eqn_viral_list =  ['eqn' + str(x) for x in np.arange(1, 18)]

        namespaces.X_list = X_list
        namespaces.X_viral_list = X_viral_list
        namespaces.dX_list = dX_list
        namespaces.R_list = R_list
        namespaces.k_growth_list = k_growth_list
        namespaces.k_meta_list = k_meta_list
        namespaces.k_viral_list = k_viral_list
        namespaces.U_list = U_list
        namespaces.eqn_viral_list = eqn_viral_list
    
    if target == 'X':
        return X_list
    elif target == 'X_viral':
        return X_viral_list
    elif target == 'dX':
        return dX_list
    elif target == 'R':
        return R_list
    elif target == 'k':
        return k_growth_list, k_meta_list, k_viral_list
    elif target == 'U':
        return U_list
    elif target == 'eqn_viral_list':
        return eqn_viral_list


def find_bounds(m, based_on=None, arg=None, unique=False, multiplier=None):
    if not (based_on and arg):
        raise('find_bounds:: Need to specify based_on and arg!')
    
    if based_on == 'manual':
        X_bounds = [[0, 3e7],  # Xv
                    [0, 3e7],  # Xd
                    [0, 6],  # Glc
                    [0, 6],  # Lac
                    [0, 6],  # AlaGln
                    [0, 6],  # Gln
                    [0, 6],  # Amm
                    [0, 6],  # Glu
                    [0, 1e11],  # pPack_media
                    [0, 3e10],
                    [0, 3e10],
                    [0, 1e11],  # pVec_media
                    [0, 3e10],
                    [0, 3e10],
                    [0, 1e11],  # pHelp_media
                    [0, 3e10],
                    [0, 3e10],
                    [0, 3e3],  # pPack_endo
                    [0, 6e4],
                    [0, 3e3],
                    [0, 3e3],  # pVec_endo
                    [0, 6e4],
                    [0, 3e3],
                    [0, 3e3],  # pHelp_endo
                    [0, 6e4],
                    [0, 3e3],
                    [0, 1e7],  # Rep
                    [0, 3e13],  # Rep_cell
                    [0, 1e6],  # VP
                    [0, 3e4],  # vDNA
                    [0, 3e11],  # rDNA_cell
                    [0, 4e4],  # eCap_nuc
                    [0, 1e6],
                    [0, 3e12],
                    [0, 2e12],
                    [0, 1e4],  # fCap_nuc
                    [0, 1e5],
                    [0, 1e11],
                    [0, 1e11],
                    [0, 4e4],  # Cap_nuc
                    [0, 1e6],
                    [0, 4e12],
                    [0, 2e12],
                    [0, 500]]  # V
        R_bounds = [[0, 1e5],  # P_vcell_S
                    [0, 1e12],  # P_media
                    [0, 1],  # mu
                    [0, 0.1],  # omega
                    [-1e-7, 1e-5],  # q_Glc
                    [-1e-7, 1e-5],  # q_Lac
                    [-1e-7, 1e-5],  # q_Gln
                    [-1e-7, 1e-5],  # q_Amm
                    [-1e-7, 1e-5],  # q_Glu
                    [0, 1e10],  # r_pPack_uptk
                    [0, 1e10],  # r_pVec_uptk
                    [0, 1e10],  # r_pHelp_uptk
                    [0, 3e6],  # r_Rep
                    [0, 1e7],  # r_VP
                    [0, 2e4],  # r_vDNA
                    [0, 1e4]]  # r_Pack
        
        X_bounds = np.array(X_bounds)
        R_bounds = np.array(R_bounds)
        d_absX_bounds = X_bounds[:,1] * 1e2  #! when there is any finely discretized point, this value needs to be increased
        d_absX_bounds = np.array(d_absX_bounds)
        
        k_growth_bounds  = [[1e-3, 1],  # mu_max
                            [1e2, 1e5],  # K_PlsmdIhbt_cell
                            [1e9, 1e13],  # K_PlsmdIhbt_media
                            [1e-2, 1e2],  # K_muLac
                            [1e-5, 1],  # k_d
                            [1e-1, 1],  # k_PlsmdCyto_cell
                            [1e4, 1e6],  # K_PlsmdCyto_cell
                            [1e-5, 1e-2],  # k_AmmCyto
                            [1, 1e2],  # K_AmmCyto
                            [1e-3, 0.1],  # k_PlsmdDeg
                            [1e-3, 0.1],  # k_PlsmdUptk
                            [1e4, 1e8],  # K_PlsmdUptk_Xv
                            [1e-4, 0.1]]  # K_PlsmdUptk_mu
        k_meta_bounds = [[5e-7, 1e-5],  # Y_Glc
                         [1e-8, 1e-7],  # m_Glc
                         [1e-1, 1],  # K_GlcLac
                         [1e-8, 1e-5],  # Y_Lac
                         [1e-8, 1e-7],  # m_Lac
                         [1e-8, 1e-7],  # k_Lac
                         [1e-3, 1],  # K_Lacmu
                         [1e-7, 1e-5],  # Y_Gln
                         [1e-8, 1e-7],  # m_Gln
                         [1e-9, 1e-6],  # K_GlnGlc
                         [1e-7, 1e-3],  # Y_Amm
                         [1e-8, 1e-3],  # m_Amm
                         [1e3, 1e6],  # K_AmmXv
                         [1e-7, 1e-5],  # Y_Glu
                         [1e-8, 1e-5],  # m_Glu
                         [1e-9, 1e-7],  # k_Glu
                         [1e-3, 1e-2]]  # K_Glumu
        k_viral_bounds = [[1e-1, 1],  # k_escape
                          [1e-3, 1e-2],  # k_nucEntry
                          [1e3, 1e4],  # k_Rep
                          [1e-2, 1e-1],  # k_expel
                          [1e-2, 1],  # k_RepDeg
                          [1e-1, 1],  # K_Amm
                          [10, 1e3],  # K_Rep_pHelp
                          [3e3, 3e5],  # k_VP
                          [1e-2, 1],  # k_assembly
                          [1e-3, 1e-2],  # k_VPdeg
                          [1e-2, 0.1],  # k_CapDeg
                          [1, 10],  # k_sec2cyto
                          [1e-3, 1e-1],  # k_sec2media
                          [1, 10],  # K_vDNA_pHelp
                          [1e3, 1e4],  # k_vDNA
                          [1e3, 1e5],  # K_vDNA_Rep
                          [1e5, 1e7],  # K_Pack_Rep
                          [1e-2, 50],  # k_Pack
                          [1e-1, 1],  # k_vDNAdeg
                          [10, 1e2],  # K_vDNA_pVec
                          [0.1, 1e2],  # K_Pack_eCap
                          [1e5, 1e7]]  # K_VP_Rep
        k_growth_bounds = np.array(k_growth_bounds)
        k_meta_bounds = np.array(k_meta_bounds)
        k_viral_bounds = np.array(k_viral_bounds)

        if m.system == 'sflask':
            U_bounds = [[0, 6],
                        [0, 6],
                        [0, 7.7e10],
                        [0, 7.7e10], 
                        [0, 7.7e10]]
        elif m.system == 'bioreactor':
            U_bounds = [[0, 40],  # volume of plasmid cocktail [mL]
                        [0, 50],  # 50 mL/h = 4 vvd
                        [0, 50], 
                        [0, 50]]
        U_bounds = np.array(U_bounds)

        if arg == 'X_min':
            return X_bounds[:,0]
        elif arg == 'X_max':
            return X_bounds[:,1]
        elif arg == 'log_X_min':
            return np.log(X_bounds[:,0] + m.alpha)
        elif arg == 'log_X_max':
            return np.log(X_bounds[:,1] + m.alpha)

        elif arg == 'd_absX_max':
            return d_absX_bounds
        elif arg == 'd_log_absX_max':
            return np.log(d_absX_bounds + m.alpha)

        elif arg == 'R_min':
            return R_bounds[:,0]
        elif arg == 'R_max':
            return R_bounds[:,1]

        elif arg == 'log_R_min':
            return np.log(R_bounds[:,0] + m.alpha)
        elif arg == 'log_R_max':
            return np.log(R_bounds[:,1] + m.alpha)

        elif arg == 'k_growth_min':
            return k_growth_bounds[:,0]
        elif arg == 'k_growth_max':
            return k_growth_bounds[:,1]
        elif arg == 'log_k_growth_min':
            return np.log(k_growth_bounds[:,0])
        elif arg == 'log_k_growth_max':
            return np.log(k_growth_bounds[:,1])

        elif arg == 'k_meta_min':
            return k_meta_bounds[:,0]
        elif arg == 'k_meta_max':
            return k_meta_bounds[:,1]
        elif arg == 'log_k_meta_min':
            return np.log(k_meta_bounds[:,0])
        elif arg == 'log_k_meta_max':
            return np.log(k_meta_bounds[:,1])

        elif arg == 'k_viral_min':
            return k_viral_bounds[:,0]
        elif arg == 'k_viral_max':
            return k_viral_bounds[:,1]
        elif arg == 'log_k_viral_min':
            return np.log(k_viral_bounds[:,0])
        elif arg == 'log_k_viral_max':
            return np.log(k_viral_bounds[:,1])

        elif arg == 'U_min':
            return U_bounds[:,0]
        elif arg == 'U_max':
            return U_bounds[:,1]

    elif based_on == 'sim':
        if not multiplier:  #! These multipliers also affect in scaling!
            if 'absX_max' in arg:
                multiplier = 1e3
            elif 'min' in arg:
                multiplier = 1e-2
            elif 'max' in arg:
                multiplier = 1e2
            else:
                raise('Unexpected behavior')    

        _, _, X, R, dX, U, U_min, U_max = load_data(based_on=based_on, system=m.system)
        
        if arg == 'X_min':
            return np.min(np.stack([np.min(multiplier*x, axis=0) for x in X], axis=0), axis=0)
        elif arg == 'X_max':
            return np.max(np.stack([np.max(multiplier*x, axis=0) for x in X], axis=0), axis=0)
        elif arg == 'log_X_min':
            return np.min(np.stack([np.min(np.log(multiplier*x + m.alpha), axis=0) for x in X], axis=0), axis=0)
        elif arg == 'log_X_max':
            return np.max(np.stack([np.max(np.log(multiplier*x + m.alpha), axis=0) for x in X], axis=0), axis=0)

        elif arg == 'R_min':
            return np.min(np.stack([np.min(multiplier*r, axis=0) for r in R], axis=0), axis=0)
        elif arg == 'R_max':
            return np.max(np.stack([np.max(multiplier*r, axis=0) for r in R], axis=0), axis=0)
        elif arg == 'log_R_min':
            return np.min(np.stack([np.min(np.log(multiplier*r + m.alpha), axis=0) for r in R], axis=0), axis=0)
        elif arg == 'log_R_max':
            return np.max(np.stack([np.max(np.log(multiplier*r + m.alpha), axis=0) for r in R], axis=0), axis=0)

        elif arg == 'U_min':
            if m.system == 'sflask':
                return np.min(np.stack([np.min(multiplier*u, axis=0) for u in U], axis=0), axis=0)
            elif m.system == 'bioreactor':
                return multiplier*U_min
        elif arg == 'U_max':
            if m.system == 'sflask':
                return np.max(np.stack([np.max(multiplier*u, axis=0) for u in U], axis=0), axis=0)
            elif m.system == 'bioreactor':
                return multiplier*U_max

        elif arg == 'd_absX_max':
            return np.max(np.stack([np.max(multiplier*abs(dx), axis=0) for dx in dX], axis=0), axis=0)
        elif arg == 'd_log_absX_max':
            return np.max(np.stack([np.max(multiplier*abs(dx/(x + m.alpha)), axis=0) for (dx, x) in zip(dX,X)], axis=0), axis=0)
            
    raise('find_bounds:: No return!')


def get_root_var_name(var):
    m = var.root_block()
    X_list = namespaces(m.system, target='X')
    dX_list = namespaces(m.system, target='dX')
    R_list = namespaces(m.system, target='R')
    k_growth_list, k_meta_list, k_viral_list = namespaces(m.system, target='k')
    U_list = namespaces(m.system, target='U')
    V_list = X_list + dX_list + R_list + k_growth_list + k_meta_list + k_viral_list + U_list

    name = None
    vtype = None
    if var.name in V_list:
        name = var.name
        vtype = 'raw'
    elif 'd_log_' in var.name and var.name.split('d_log_')[1] in V_list:
        name = 'd_' + var.name.split('d_log_')[1]
        vtype = 'd_log'
    elif 'log_' in var.name and var.name.split('log_')[1] in V_list:
        name = var.name.split('log_')[1]
        vtype = 'log'
    return name, vtype
