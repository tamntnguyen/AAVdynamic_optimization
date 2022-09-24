import os
import numpy as np
import hdf5storage
from pyomo.environ import *
from pyomo.dae import *

from config import get_scaling_val, namespaces, find_bounds
from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))


def save_params(m):
    d = {}
    d['k_growth'] = np.array(get_param_values(m, 'k_growth'))
    d['k_meta'] = np.array(get_param_values(m, 'k_meta'))
    d['k_viral'] = np.array(get_param_values(m, 'k_viral'))
    filedir = os.path.join(datadir, 'parameter', 'k_opt.mat')
    hdf5storage.savemat(filedir, d, oned_as='column')
    return


def get_param_values(m, ptype):
    k_growth_list, k_meta_list, k_viral_list = namespaces(m.system, 'k')
    
    if ptype == 'k_growth':
        param = np.array(k_growth_list).tolist()  # make instances to avoid pointers
    elif ptype == 'k_meta':
        param = np.array(k_meta_list).tolist()
    elif ptype == 'k_viral':
        param = np.array(k_viral_list).tolist()

    for obj in m.component_objects():
        name = obj.name
        if name in param:
            idx = param.index(name)
            param[idx] = value(obj)
    for idx, val in enumerate(param):
        if isinstance(val, str):
            param[idx] = np.NaN
    return param


def eval_param_sensitivity(m):
    theta = {'p1':m.p1(), 'p2':m.p2()}  # nominal values
    sigma_p = np.array([[2, 0], [0, 1]])  # covariance matrix
    theta_names = ['p1', 'p2']  # names of uncertain parameters
    
    results = propagate_uncertainty(m, theta, sigma_p, theta_names)
    return



def Params(m):
    # Numerical settings
    m.eps = 1e-23
    m.alpha = 1  # shifter for log transformation
    
    # Load parameters
    filedir = os.path.join(datadir, 'parameter', 'k.mat')
    data = hdf5storage.loadmat(filedir)
    k_growth = np.squeeze(data['k_growth'])
    k_meta = np.squeeze(data['k_meta'])
    k_viral = np.squeeze(data['k_viral'])

    # Cell growth
    if m.mode == 'estimation' and m.est_growth:
        k_growth_min = find_bounds(m, based_on='manual', arg='k_growth_min')
        k_growth_max = find_bounds(m, based_on='manual', arg='k_growth_max')
        log_k_growth_min = find_bounds(m, based_on='manual', arg='log_k_growth_min')
        log_k_growth_max = find_bounds(m, based_on='manual', arg='log_k_growth_max')
        log_k_growth = np.log(k_growth)

        m.log_mu_max = Var(bounds=(log_k_growth_min[0], log_k_growth_max[0]), initialize=log_k_growth[0])
        m.K_PlsmdIhbt_cell = Var(bounds=(k_growth_min[1], k_growth_max[1]), initialize=k_growth[1])
        m.K_PlsmdIhbt_media = Var(bounds=(k_growth_min[2], k_growth_max[2]), initialize=k_growth[2])
        m.K_muLac = Var(bounds=(k_growth_min[3], k_growth_max[3]), initialize=k_growth[3])
        m.log_k_d = Var(bounds=(log_k_growth_min[4], log_k_growth_max[4]), initialize=log_k_growth[4])
        m.log_k_PlsmdCyto_cell = Var(bounds=(log_k_growth_min[5], log_k_growth_max[5]), initialize=log_k_growth[5])
        m.K_PlsmdCyto_cell = Var(bounds=(k_growth_min[6], k_growth_max[6]), initialize=k_growth[6])
        m.log_k_AmmCyto = Var(bounds=(log_k_growth_min[7], log_k_growth_max[7]), initialize=log_k_growth[7])
        m.K_AmmCyto = Var(bounds=(k_growth_min[8], k_growth_max[8]), initialize=k_growth[8])
        m.log_k_PlsmdDeg = Var(bounds=(log_k_growth_min[9], log_k_growth_max[9]), initialize=log_k_growth[9])
        m.log_k_PlsmdUptk = Var(bounds=(log_k_growth_min[10], log_k_growth_max[10]), initialize=log_k_growth[10])
        m.K_PlsmdUptk_Xv = Var(bounds=(k_growth_min[11], k_growth_max[11]), initialize=k_growth[11])
        m.K_PlsmdUptk_mu = Var(bounds=(k_growth_min[12], k_growth_max[12]), initialize=k_growth[12])

        # Transformation
        m.mu_max = Expression(expr=exp(m.log_mu_max))
        m.k_d = Expression(expr=exp(m.log_k_d))
        m.k_PlsmdCyto_cell = Expression(expr=exp(m.log_k_PlsmdCyto_cell))
        m.k_AmmCyto = Expression(expr=exp(m.log_k_AmmCyto))
        m.k_PlsmdDeg = Expression(expr=exp(m.log_k_PlsmdDeg))
        m.k_PlsmdUptk = Expression(expr=exp(m.log_k_PlsmdUptk))
    else:
        m.mu_max = Param(default=k_growth[0], mutable=True)
        m.K_PlsmdIhbt_cell = Param(default=k_growth[1], mutable=True)
        m.K_PlsmdIhbt_media = Param(default=k_growth[2], mutable=True)
        m.K_muLac = Param(default=k_growth[3], mutable=True)
        m.k_d = Param(default=k_growth[4], mutable=True)
        m.k_PlsmdCyto_cell = Param(default=k_growth[5], mutable=True)
        m.K_PlsmdCyto_cell = Param(default=k_growth[6], mutable=True)
        m.k_AmmCyto = Param(default=k_growth[7], mutable=True)
        m.K_AmmCyto = Param(default=k_growth[8], mutable=True)
        m.k_PlsmdDeg = Param(default=k_growth[9], mutable=True)
        m.k_PlsmdUptk = Param(default=k_growth[10], mutable=True)
        m.K_PlsmdUptk_Xv = Param(default=k_growth[11], mutable=True)
        m.K_PlsmdUptk_mu = Param(default=k_growth[12], mutable=True)

    # Metabolism
    if m.mode == 'estimation' and m.est_meta:
        k_meta_min = find_bounds(m, based_on='manual', arg='k_meta_min')
        k_meta_max = find_bounds(m, based_on='manual', arg='k_meta_max')
        log_k_meta_min = find_bounds(m, based_on='manual', arg='log_k_meta_min')
        log_k_meta_max = find_bounds(m, based_on='manual', arg='log_k_meta_max')
        log_k_meta = np.log(k_meta)
        
        m.log_Y_Glc = Var(bounds=(log_k_meta_min[0], log_k_meta_max[0]), initialize=log_k_meta[0])
        m.log_m_Glc = Var(bounds=(log_k_meta_min[1], log_k_meta_max[1]), initialize=log_k_meta[1])
        m.K_GlcLac = Var(bounds=(k_meta_min[2], k_meta_max[2]), initialize=k_meta[2])
        m.log_Y_Lac = Var(bounds=(log_k_meta_min[3], log_k_meta_max[3]), initialize=log_k_meta[3])
        m.log_m_Lac = Var(bounds=(log_k_meta_min[4], log_k_meta_max[4]), initialize=log_k_meta[4])
        m.log_k_Lac = Var(bounds=(log_k_meta_min[5], log_k_meta_max[5]), initialize=log_k_meta[5])
        m.K_Lacmu = Var(bounds=(k_meta_min[6], k_meta_max[6]), initialize=k_meta[6])
        m.log_Y_Gln = Var(bounds=(log_k_meta_min[7], log_k_meta_max[7]), initialize=log_k_meta[7])
        m.log_m_Gln = Var(bounds=(log_k_meta_min[8], log_k_meta_max[8]), initialize=log_k_meta[8])
        m.K_GlnGlc = Var(bounds=(k_meta_min[9], k_meta_max[9]), initialize=k_meta[9])
        m.log_Y_Amm = Var(bounds=(log_k_meta_min[10], log_k_meta_max[10]), initialize=log_k_meta[10])
        m.log_m_Amm = Var(bounds=(log_k_meta_min[11], log_k_meta_max[11]), initialize=log_k_meta[11])
        m.K_AmmXv = Var(bounds=(k_meta_min[12], k_meta_max[12]), initialize=k_meta[12])
        m.log_Y_Glu = Var(bounds=(log_k_meta_min[13], log_k_meta_max[13]), initialize=log_k_meta[13])
        m.log_m_Glu = Var(bounds=(log_k_meta_min[14], log_k_meta_max[14]), initialize=log_k_meta[14])
        m.log_k_Glu = Var(bounds=(log_k_meta_min[15], log_k_meta_max[15]), initialize=log_k_meta[15])
        m.K_Glumu = Var(bounds=(k_meta_min[16], k_meta_max[16]), initialize=k_meta[16])

        # Transformation
        m.Y_Glc = Expression(expr=exp(m.log_Y_Glc))
        m.m_Glc = Expression(expr=exp(m.log_m_Glc))
        m.Y_Lac = Expression(expr=exp(m.log_Y_Lac))
        m.m_Lac = Expression(expr=exp(m.log_m_Lac))
        m.k_Lac = Expression(expr=exp(m.log_k_Lac))
        m.Y_Gln = Expression(expr=exp(m.log_Y_Gln))
        m.m_Gln = Expression(expr=exp(m.log_m_Gln))
        m.Y_Amm = Expression(expr=exp(m.log_Y_Amm))
        m.m_Amm = Expression(expr=exp(m.log_m_Amm))
        m.Y_Glu = Expression(expr=exp(m.log_Y_Glu))
        m.m_Glu = Expression(expr=exp(m.log_m_Glu))
        m.k_Glu = Expression(expr=exp(m.log_k_Glu))
    else:
        m.Y_Glc = Param(default=k_meta[0], mutable=True)
        m.m_Glc = Param(default=k_meta[1], mutable=True)
        m.K_GlcLac = Param(default=k_meta[2], mutable=True)
        m.Y_Lac = Param(default=k_meta[3], mutable=True)
        m.m_Lac = Param(default=k_meta[4], mutable=True)
        m.k_Lac = Param(default=k_meta[5], mutable=True)
        m.K_Lacmu = Param(default=k_meta[6], mutable=True)
        m.Y_Gln = Param(default=k_meta[7], mutable=True)
        m.m_Gln = Param(default=k_meta[8], mutable=True)
        m.K_GlnGlc = Param(default=k_meta[9], mutable=True)
        m.Y_Amm = Param(default=k_meta[10], mutable=True)
        m.m_Amm = Param(default=k_meta[11], mutable=True)
        m.K_AmmXv = Param(default=k_meta[12], mutable=True)
        m.Y_Glu = Param(default=k_meta[13], mutable=True)
        m.m_Glu = Param(default=k_meta[14], mutable=True)
        m.k_Glu = Param(default=k_meta[15], mutable=True)
        m.K_Glumu = Param(default=k_meta[16], mutable=True)

    m.k_GlnDeg = Param(default=k_meta[17], mutable=True)  # Spontaneous Gln degradation
    m.k_GlnGen = Param(default=k_meta[18], mutable=True)  # Gln generation from AlaGln (GlutaMAX)

    # Viral production
    if m.mode == 'estimation' and m.est_viralprod:
        k_viral_min = find_bounds(m, based_on='manual', arg='k_viral_min')
        k_viral_max = find_bounds(m, based_on='manual', arg='k_viral_max')
        log_k_viral_min = find_bounds(m, based_on='manual', arg='log_k_viral_min')
        log_k_viral_max = find_bounds(m, based_on='manual', arg='log_k_viral_max')
        log_k_viral = np.log(k_viral)

        m.log_k_escape = Var(bounds=(log_k_viral_min[0], log_k_viral_max[0]), initialize=log_k_viral[0])
        m.log_k_nucEntry = Var(bounds=(log_k_viral_min[1], log_k_viral_max[1]), initialize=log_k_viral[1])
        m.log_k_Rep = Var(bounds=(log_k_viral_min[2], log_k_viral_max[2]), initialize=log_k_viral[2])
        m.log_k_expel = Var(bounds=(log_k_viral_min[3], log_k_viral_max[3]), initialize=log_k_viral[3])
        m.log_k_RepDeg = Var(bounds=(log_k_viral_min[4], log_k_viral_max[4]), initialize=log_k_viral[4])
        m.K_Amm = Var(bounds=(k_viral_min[5], k_viral_max[5]), initialize=k_viral[5])
        m.log_K_Rep_pHelp = Var(bounds=(log_k_viral_min[6], log_k_viral_max[6]), initialize=log_k_viral[6])
        m.log_k_VP = Var(bounds=(log_k_viral_min[7], log_k_viral_max[7]), initialize=log_k_viral[7])
        m.k_assembly = Var(bounds=(k_viral_min[8], k_viral_max[8]), initialize=k_viral[8])
        m.log_k_VPdeg = Var(bounds=(log_k_viral_min[9], log_k_viral_max[9]), initialize=log_k_viral[9])
        m.k_CapDeg = Var(bounds=(k_viral_min[10], k_viral_max[10]), initialize=k_viral[10])
        m.k_sec2cyto = Var(bounds=(k_viral_min[11], k_viral_max[11]), initialize=k_viral[11])
        m.log_k_sec2media = Var(bounds=(log_k_viral_min[12], log_k_viral_max[12]), initialize=log_k_viral[12])
        m.K_vDNA_pHelp = Var(bounds=(k_viral_min[13], k_viral_max[13]), initialize=k_viral[13])
        m.log_k_vDNA = Var(bounds=(log_k_viral_min[14], log_k_viral_max[14]), initialize=log_k_viral[14])
        m.log_K_vDNA_Rep = Var(bounds=(log_k_viral_min[15], log_k_viral_max[15]), initialize=log_k_viral[15])
        m.log_K_Pack_Rep = Var(bounds=(log_k_viral_min[16], log_k_viral_max[16]), initialize=log_k_viral[16])
        m.k_Pack = Var(bounds=(k_viral_min[17], k_viral_max[17]), initialize=k_viral[17])
        m.k_vDNAdeg = Var(bounds=(k_viral_min[18], k_viral_max[18]), initialize=k_viral[18])
        m.K_vDNA_pVec = Var(bounds=(k_viral_min[19], k_viral_max[19]), initialize=k_viral[19])
        m.K_Pack_eCap = Var(bounds=(k_viral_min[20], k_viral_max[20]), initialize=k_viral[20])
        m.K_VP_Rep = Var(bounds=(k_viral_min[21], k_viral_max[21]), initialize=k_viral[21])
        
        # Transformation
        m.k_escape = Expression(expr=exp(m.log_k_escape))
        m.k_nucEntry = Expression(expr=exp(m.log_k_nucEntry))
        m.k_Rep = Expression(expr=exp(m.log_k_Rep))
        m.k_expel = Expression(expr=exp(m.log_k_expel))
        m.k_RepDeg = Expression(expr=exp(m.log_k_RepDeg))
        m.K_Rep_pHelp = Expression(expr=exp(m.log_K_Rep_pHelp))
        m.k_VP = Expression(expr=exp(m.log_k_VP))
        m.k_VPdeg = Expression(expr=exp(m.log_k_VPdeg))
        m.k_sec2media = Expression(expr=exp(m.log_k_sec2media))
        m.k_vDNA = Expression(expr=exp(m.log_k_vDNA))
        m.K_vDNA_Rep = Expression(expr=exp(m.log_K_vDNA_Rep))
        m.K_Pack_Rep = Expression(expr=exp(m.log_K_Pack_Rep))
    else:
        m.k_escape = Param(default=k_viral[0], mutable=True)
        m.k_nucEntry = Param(default=k_viral[1], mutable=True)
        m.k_Rep = Param(default=k_viral[2], mutable=True)
        m.k_expel = Param(default=k_viral[3], mutable=True)
        m.k_RepDeg = Param(default=k_viral[4], mutable=True)
        m.K_Amm = Param(default=k_viral[5], mutable=True)
        m.K_Rep_pHelp = Param(default=k_viral[6], mutable=True)
        m.k_VP = Param(default=k_viral[7], mutable=True)
        m.k_assembly = Param(default=k_viral[8], mutable=True)
        m.k_VPdeg = Param(default=k_viral[9], mutable=True)
        m.k_CapDeg = Param(default=k_viral[10], mutable=True)
        m.k_sec2cyto = Param(default=k_viral[11], mutable=True)
        m.k_sec2media = Param(default=k_viral[12], mutable=True)
        m.K_vDNA_pHelp = Param(default=k_viral[13], mutable=True)
        m.k_vDNA = Param(default=k_viral[14], mutable=True)
        m.K_vDNA_Rep = Param(default=k_viral[15], mutable=True)
        m.K_Pack_Rep = Param(default=k_viral[16], mutable=True)
        m.k_Pack = Param(default=k_viral[17], mutable=True)
        m.k_vDNAdeg = Param(default=k_viral[18], mutable=True)
        m.K_vDNA_pVec = Param(default=k_viral[19], mutable=True)
        m.K_Pack_eCap = Param(default=k_viral[20], mutable=True)
        m.K_VP_Rep = Param(default=k_viral[21], mutable=True)
        
    # System dependent parameters
    if m.system == 'bioreactor':
        filedir = os.path.join(datadir, 'parameter', 'concentration.mat')
        data = hdf5storage.loadmat(filedir)
        C = np.squeeze(data['C'])
        
        m.C_pPack = Param(default=C[0], mutable=True)
        m.C_pVec = Param(default=C[1], mutable=True)
        m.C_pHelp = Param(default=C[2], mutable=True)
        m.C_Glc = Param(default=C[3], mutable=True)
        m.C_AlaGln = Param(default=C[4], mutable=True)

        m.V_ratio = 0.8881

        filedir = os.path.join(datadir, 'parameter', 'misc.mat')
        data = hdf5storage.loadmat(filedir)
        m.Fevap = np.squeeze(data['Fevap'])

    # Case when property approximation is needed
    if m.system == 'sflask' and not m.meta:
        filedir = os.path.join(datadir, 'parameter', 'p.mat')
        data = hdf5storage.loadmat(filedir)
        p_amm = np.squeeze(data['p_amm'])
        p_lac = np.squeeze(data['p_lac'])

        def Amm_rule(m, t, i):
            if i == 5:  # no transfection case
                i = 4
            return (p_amm[i-1, 0]*t**3 + p_amm[i-1, 1]*t**2 + p_amm[i-1, 2]*t + p_amm[i-1, 3])
        m.Amm_approx = Param(m.t, m.i, rule=Amm_rule, default=Amm_rule, mutable=True)

        def Lac_rule(m, t, i):
            if i == 5:  # no transfection case
                i = 4
            return (p_lac[i-1, 0]*t**3 + p_lac[i-1, 1]*t**2 + p_lac[i-1, 2]*t + p_lac[i-1, 3])
        m.Lac_approx = Param(m.t, m.i, rule=Lac_rule, default=Lac_rule, mutable=True)