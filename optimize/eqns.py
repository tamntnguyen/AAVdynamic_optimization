import os
from pyomo.environ import *
import hdf5storage
import numpy as np

from config import get_scaling_val
from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))


# Cell growth
def eqn1(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Xv[t,i] == (m.mu[t,i] - m.omega[t,i])*m.Xv[t,i] \
                - m.Xv[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_Xv[t,i] == (m.mu[t,i] - m.omega[t,i])*m.Xv[t,i] \
                - m.Xv[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_Xv[t,i] == (m.mu[t,i] - m.omega[t,i])*m.Xv[t,i]
    

def eqn2(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Xd[t,i] == m.omega[t,i]*m.Xv[t,i] \
                - m.Xd[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_Xd[t,i] == m.omega[t,i]*m.Xv[t,i] \
                - m.Xd[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_Xd[t,i] == m.omega[t,i]*m.Xv[t,i]
    
    
# Metabolism
def eqn3(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Glc[t,i] == - m.Xv[t,i]*m.q_Glc[t,i] \
                - m.Glc[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i] \
                    + m.C_Glc*(m.Fi[t,i] + m.Vb[t,i]*m.V_ratio/m.dt)/m.V[t,i]
        else:
            return m.d_Glc[t,i] == - m.Xv[t,i]*m.q_Glc[t,i] \
                - m.Glc[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i] \
                    + m.C_Glc*m.Fi[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_Glc[t,i] == - m.Xv[t,i]*m.q_Glc[t,i]
    
def eqn4(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Lac[t,i] == m.Xv[t,i]*m.q_Lac[t,i] \
                - m.Lac[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_Lac[t,i] == m.Xv[t,i]*m.q_Lac[t,i] \
                - m.Lac[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_Lac[t,i] == m.Xv[t,i]*m.q_Lac[t,i]

def eqn5(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_AlaGln[t,i] == - m.k_GlnGen*m.AlaGln[t,i]*m.Xv[t,i] \
                - m.AlaGln[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i] \
                    + m.C_AlaGln*(m.Fi[t,i] + m.Vb[t,i]*m.V_ratio/m.dt)/m.V[t,i]
        else:
            return m.d_AlaGln[t,i] == - m.k_GlnGen*m.AlaGln[t,i]*m.Xv[t,i] \
                - m.AlaGln[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i] \
                    + m.C_AlaGln*m.Fi[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_AlaGln[t,i] == - m.k_GlnGen*m.AlaGln[t,i]*m.Xv[t,i]

def eqn6(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Gln[t,i] == - m.Xv[t,i]*m.q_Gln[t,i] \
                - m.Gln[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i] \
                    - m.k_Glu_deg*m.Gln[t,i] + m.k_GlnGen*m.AlaGln[t,i]*m.Xv[t,i]
        else:
            return m.d_Gln[t,i] == - m.Xv[t,i]*m.q_Gln[t,i] \
                - m.Gln[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i] \
                    - m.k_Glu_deg*m.Gln[t,i] + m.k_GlnGen*m.AlaGln[t,i]*m.Xv[t,i]
    elif m.system == 'sflask':
        return m.d_Gln[t,i] == - m.Xv[t,i]*m.q_Gln[t,i] - m.k_GlnDeg*m.Gln[t,i] + m.k_GlnGen*m.AlaGln[t,i]*m.Xv[t,i]
    
def eqn7(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Amm[t,i] == m.Xv[t,i]*m.q_Amm[t,i] \
                - m.Amm[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i] \
                    + m.k_Glu_deg*m.Gln[t,i]
        else:
            return m.d_Amm[t,i] == m.Xv[t,i]*m.q_Amm[t,i] \
                - m.Amm[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i] \
                    + m.k_Glu_deg*m.Gln[t,i]
    elif m.system == 'sflask':
        return m.d_Amm[t,i] == m.Xv[t,i]*m.q_Amm[t,i] + m.k_GlnDeg*m.Gln[t,i]
    
def eqn8(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Glu[t,i] == m.Xv[t,i]*m.q_Glu[t,i] \
                - m.Glu[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_Glu[t,i] == m.Xv[t,i]*m.q_Glu[t,i] \
                - m.Glu[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_Glu[t,i] == m.Xv[t,i]*m.q_Glu[t,i]     

# Transfection reagents
def eqn9(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pPack_media[t,i] == - m.r_pPack_uptk[t,i] \
                - m.pPack_media[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i] \
                    + m.C_pPack*m.Vb[t,i]/m.V[t,i]/m.dt
        else:
            return m.d_pPack_media[t,i] == - m.r_pPack_uptk[t,i] \
                - m.pPack_media[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pPack_media[t,i] == - m.r_pPack_uptk[t,i] + m.pPack_add[t,i]/m.dt
    
def eqn10(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pPack_vcell[t,i] == m.r_pPack_uptk[t,i] \
                - (m.k_PlsmdDeg + m.omega[t,i])*m.pPack_vcell[t,i] \
                    - m.pPack_vcell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_pPack_vcell[t,i] == m.r_pPack_uptk[t,i] \
                - (m.k_PlsmdDeg + m.omega[t,i])*m.pPack_vcell[t,i] \
                    - m.pPack_vcell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pPack_vcell[t,i] == m.r_pPack_uptk[t,i] \
            - (m.k_PlsmdDeg + m.omega[t,i])*m.pPack_vcell[t,i]
    
def eqn11(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pPack_cell[t,i] == m.r_pPack_uptk[t,i] \
                - m.k_PlsmdDeg*m.pPack_cell[t,i] \
                - m.pPack_cell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_pPack_cell[t,i] == m.r_pPack_uptk[t,i] \
                - m.k_PlsmdDeg*m.pPack_cell[t,i] \
                - m.pPack_cell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pPack_cell[t,i] == m.r_pPack_uptk[t,i] \
            - m.k_PlsmdDeg*m.pPack_cell[t,i]
    

def eqn12(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pVec_media[t,i] == - m.r_pVec_uptk[t,i] \
                - m.pVec_media[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i] \
                    + m.C_pVec*m.V_pVec[t,i]/m.V[t,i]/m.dt
        else:
            return m.d_pVec_media[t,i] == - m.r_pVec_uptk[t,i] \
                - m.pVec_media[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pVec_media[t,i] == - m.r_pVec_uptk[t,i] + m.pVec_add[t,i]/m.dt
    
def eqn13(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pVec_vcell[t,i] == m.r_pVec_uptk[t,i] \
                - (m.k_PlsmdDeg + m.omega[t,i])*m.pVec_vcell[t,i] \
                    - m.pVec_vcell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_pVec_vcell[t,i] == m.r_pVec_uptk[t,i] \
                - (m.k_PlsmdDeg + m.omega[t,i])*m.pVec_vcell[t,i] \
                    - m.pVec_vcell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pVec_vcell[t,i] == m.r_pVec_uptk[t,i] \
            - (m.k_PlsmdDeg + m.omega[t,i])*m.pVec_vcell[t,i]
    
def eqn14(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pVec_cell[t,i] == m.r_pVec_uptk[t,i] \
                - m.k_PlsmdDeg*m.pVec_cell[t,i] \
                - m.pVec_cell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_pVec_cell[t,i] == m.r_pVec_uptk[t,i] \
                - m.k_PlsmdDeg*m.pVec_cell[t,i] \
                - m.pVec_cell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pVec_cell[t,i] == m.r_pVec_uptk[t,i] \
            - m.k_PlsmdDeg*m.pVec_cell[t,i]
    

def eqn15(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pHelp_media[t,i] == - m.r_pHelp_uptk[t,i] \
                - m.pHelp_media[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i] \
                    + m.C_pHelp*m.Vb[t,i]/m.V[t,i]/m.dt
        else:
            return m.d_pHelp_media[t,i] == - m.r_pHelp_uptk[t,i] \
                - m.pHelp_media[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pHelp_media[t,i] == - m.r_pHelp_uptk[t,i] + m.pHelp_add[t,i]/m.dt

    
def eqn16(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pHelp_vcell[t,i] == m.r_pHelp_uptk[t,i] \
                - (m.k_PlsmdDeg + m.omega[t,i])*m.pHelp_vcell[t,i] \
                    - m.pHelp_vcell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_pHelp_vcell[t,i] == m.r_pHelp_uptk[t,i] \
                - (m.k_PlsmdDeg + m.omega[t,i])*m.pHelp_vcell[t,i] \
                    - m.pHelp_vcell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pHelp_vcell[t,i] == m.r_pHelp_uptk[t,i] \
            - (m.k_PlsmdDeg + m.omega[t,i])*m.pHelp_vcell[t,i]
    
def eqn17(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_pHelp_cell[t,i] == m.r_pHelp_uptk[t,i] \
                - m.k_PlsmdDeg*m.pHelp_cell[t,i] \
                - m.pHelp_cell[t,i]*(m.Fb[t,i] +m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_pHelp_cell[t,i] == m.r_pHelp_uptk[t,i] \
                - m.k_PlsmdDeg*m.pHelp_cell[t,i] \
                - m.pHelp_cell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_pHelp_cell[t,i] == m.r_pHelp_uptk[t,i] \
            - m.k_PlsmdDeg*m.pHelp_cell[t,i]
    

# Trafficking
def eqn18(m, t, i):
    return m.d_pPack_endo[t,i] == m.r_pPack_uptk[t,i]/m.Xv[t,i] \
        - (m.k_escape + m.k_PlsmdDeg + m.mu[t,i])*m.pPack_endo[t,i]

def eqn19(m, t, i):
    return m.d_pPack_cyto[t,i] == m.k_escape*m.pPack_endo[t,i] \
        + m.k_expel*m.pPack_nuc[t,i] - (m.k_nucEntry + m.k_PlsmdDeg + m.mu[t,i]) * m.pPack_cyto[t,i]

def eqn20(m, t, i):
    return m.d_pPack_nuc[t,i] == m.k_nucEntry*m.pPack_cyto[t,i] \
        - (m.k_PlsmdDeg + m.mu[t,i] + m.k_expel)*m.pPack_nuc[t,i]


def eqn21(m, t, i):
    return m.d_pVec_endo[t,i] == m.r_pVec_uptk[t,i]/m.Xv[t,i] \
        - (m.k_escape + m.k_PlsmdDeg + m.mu[t,i])*m.pVec_endo[t,i]

def eqn22(m, t, i):
    return m.d_pVec_cyto[t,i] == m.k_escape*m.pVec_endo[t,i] \
        + m.k_expel*m.pVec_nuc[t,i] \
            - (m.k_nucEntry + m.k_PlsmdDeg + m.mu[t,i]) * m.pVec_cyto[t,i]

def eqn23(m, t, i):
    return m.d_pVec_nuc[t,i] == m.k_nucEntry*m.pVec_cyto[t,i] \
        - (m.k_PlsmdDeg + m.mu[t,i] + m.k_expel)*m.pVec_nuc[t,i]


def eqn24(m, t, i):
    return m.d_pHelp_endo[t,i] == m.r_pHelp_uptk[t,i]/m.Xv[t,i] \
        - (m.k_escape + m.k_PlsmdDeg + m.mu[t,i])*m.pHelp_endo[t,i]

def eqn25(m, t, i):
    return m.d_pHelp_cyto[t,i] == m.k_escape*m.pHelp_endo[t,i] \
        + m.k_expel*m.pHelp_nuc[t,i] - (m.k_nucEntry + \
            m.k_PlsmdDeg + m.mu[t,i]) * m.pHelp_cyto[t,i]

def eqn26(m, t, i):
    return m.d_pHelp_nuc[t,i] == m.k_nucEntry*m.pHelp_cyto[t,i] \
        - (m.k_PlsmdDeg + m.mu[t,i] + m.k_expel)*m.pHelp_nuc[t,i]
        

# Viral production
def eqn27(m, t, i):
    return m.d_Rep[t,i] == m.r_Rep[t,i] - (m.k_RepDeg + m.mu[t,i]) * m.Rep[t,i]

def eqn28(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_Rep_cell[t,i] == m.r_Rep[t,i]*m.Xv[t,i] - m.k_RepDeg * m.Rep_cell[t,i] \
                - m.Rep_cell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_Rep_cell[t,i] == m.r_Rep[t,i]*m.Xv[t,i] - m.k_RepDeg * m.Rep_cell[t,i] \
                - m.Rep_cell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_Rep_cell[t,i] == m.r_Rep[t,i]*m.Xv[t,i] - m.k_RepDeg * m.Rep_cell[t,i]
    
def eqn29(m, t, i):
    return m.d_VP[t,i] == m.r_VP[t,i] - 60*m.k_assembly*m.VP[t,i] \
        - (m.k_VPdeg + m.mu[t,i])*m.VP[t,i]

def eqn30(m, t, i):
    return m.d_vDNA[t,i] == m.r_vDNA[t,i] - m.r_Pack[t,i] - m.k_vDNAdeg*m.vDNA[t,i]

def eqn31(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_rDNA_cell[t,i] == m.r_vDNA[t,i]*m.Xv[t,i] \
                - m.k_vDNAdeg*(m.rDNA_cell[t,i] - m.fCap_cell[t,i]) \
                    - (m.k_sec2media + m.k_CapDeg)*m.fCap_cyto[t,i]*m.Xv[t,i] \
                        - m.rDNA_cell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_rDNA_cell[t,i] == m.r_vDNA[t,i]*m.Xv[t,i] \
                - m.k_vDNAdeg*(m.rDNA_cell[t,i] - m.fCap_cell[t,i]) \
                    - (m.k_sec2media + m.k_CapDeg)*m.fCap_cyto[t,i]*m.Xv[t,i] \
                        - m.rDNA_cell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_rDNA_cell[t,i] == m.r_vDNA[t,i]*m.Xv[t,i] \
            - m.k_vDNAdeg*(m.rDNA_cell[t,i] - m.fCap_cell[t,i]) \
                - (m.k_sec2media + m.k_CapDeg)*m.fCap_cyto[t,i]*m.Xv[t,i]
    
def eqn32(m, t, i):
    return m.d_eCap_nuc[t,i] == m.k_assembly*m.VP[t,i] \
        - (m.k_sec2cyto + m.mu[t,i])*m.eCap_nuc[t,i] - m.r_Pack[t,i]

def eqn33(m, t, i):
    return m.d_eCap_cyto[t,i] == m.k_sec2cyto*m.eCap_nuc[t,i] \
        - (m.k_sec2media + m.k_CapDeg + m.mu[t,i])*m.eCap_cyto[t,i]

def eqn34(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_eCap_cell[t,i] == m.k_assembly*m.VP[t,i]*m.Xv[t,i] \
                - m.r_Pack[t,i]*m.Xv[t,i] \
                    - (m.k_sec2media + m.k_CapDeg)*m.eCap_cyto[t,i]*m.Xv[t,i] \
                        - m.eCap_cell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_eCap_cell[t,i] == m.k_assembly*m.VP[t,i]*m.Xv[t,i] \
                - m.r_Pack[t,i]*m.Xv[t,i] \
                    - (m.k_sec2media + m.k_CapDeg)*m.eCap_cyto[t,i]*m.Xv[t,i] \
                        - m.eCap_cell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_eCap_cell[t,i] == m.k_assembly*m.VP[t,i]*m.Xv[t,i] \
            - m.r_Pack[t,i]*m.Xv[t,i] - (m.k_sec2media + m.k_CapDeg)*m.eCap_cyto[t,i]*m.Xv[t,i]
    
def eqn35(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_eCap_media[t,i] == m.k_sec2media*m.eCap_cyto[t,i]*m.Xv[t,i] \
                - m.eCap_media[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_eCap_media[t,i] == m.k_sec2media*m.eCap_cyto[t,i]*m.Xv[t,i] \
                - m.eCap_media[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_eCap_media[t,i] == m.k_sec2media*m.eCap_cyto[t,i]*m.Xv[t,i]
    
def eqn36(m, t, i):
    return m.d_fCap_nuc[t,i] == m.r_Pack[t,i] - (m.k_sec2cyto + m.mu[t,i])*m.fCap_nuc[t,i]

def eqn37(m, t, i):
    return m.d_fCap_cyto[t,i] == m.k_sec2cyto*m.fCap_nuc[t,i] \
        - (m.k_sec2media + m.k_CapDeg + m.mu[t,i])*m.fCap_cyto[t,i]

def eqn38(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_fCap_cell[t,i] == m.r_Pack[t,i]*m.Xv[t,i] \
                - (m.k_sec2media + m.k_CapDeg)*m.fCap_cyto[t,i]*m.Xv[t,i] \
                    - m.fCap_cell[t,i]*(m.Fb[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_fCap_cell[t,i] == m.r_Pack[t,i]*m.Xv[t,i] \
                - (m.k_sec2media + m.k_CapDeg)*m.fCap_cyto[t,i]*m.Xv[t,i] \
                    - m.fCap_cell[t,i]*m.Fb[t,i]/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_fCap_cell[t,i] == m.r_Pack[t,i]*m.Xv[t,i] \
            - (m.k_sec2media + m.k_CapDeg)*m.fCap_cyto[t,i]*m.Xv[t,i]
    
def eqn39(m, t, i):
    if m.system == 'bioreactor':
        if t in m.Ts:
            return m.d_fCap_media[t,i] == m.k_sec2media*m.fCap_cyto[t,i]*m.Xv[t,i] \
                - m.fCap_media[t,i]*(m.Fb[t,i] + m.Fh[t,i] + m.Vs[t,i]/m.dt)/m.V[t,i]
        else:
            return m.d_fCap_media[t,i] == m.k_sec2media*m.fCap_cyto[t,i]*m.Xv[t,i] \
                - m.fCap_media[t,i]*(m.Fb[t,i] + m.Fh[t,i])/m.V[t,i]
    elif m.system == 'sflask':
        return m.d_fCap_media[t,i] == m.k_sec2media*m.fCap_cyto[t,i]*m.Xv[t,i]
    
def eqn40(m, t, i):
    return m.d_Cap_nuc[t,i] == m.d_eCap_nuc[t,i] + m.d_fCap_nuc[t,i]

def eqn41(m, t, i):
    return m.d_Cap_cyto[t,i] == m.d_eCap_cyto[t,i] + m.d_fCap_cyto[t,i]
    
def eqn42(m, t, i):
    return m.d_Cap_cell[t,i] == m.d_eCap_cell[t,i] + m.d_fCap_cell[t,i]

def eqn43(m, t, i):
    return m.d_Cap_media[t,i] == m.d_eCap_media[t,i] + m.d_fCap_media[t,i]


# Media volume change
def eqn44(m, t, i):
    if t in m.Ts:
        return m.d_V[t,i] == m.Fi[t,i] - m.Fb[t,i] - m.Fh[t,i] - m.Vs[t,i]/m.dt
    else:
        return m.d_V[t,i] == m.Fi[t,i] - m.Fb[t,i] - m.Fh[t,i]


# Algebraic equations
def eqn45(m, t, i):  
    return m.P_vcell_S[t,i] == (m.pPack_vcell[t,i] + m.pVec_vcell[t,i] + m.pHelp_vcell[t,i])/m.Xv[t,i]

def eqn46(m, t, i):  
    return m.P_media[t,i] == (m.pPack_media[t,i] + m.pVec_media[t,i] + m.pHelp_media[t,i])

def eqn47(m, t, i):
    if m.meta:
        return m.mu[t,i] == m.mu_max * m.K_PlsmdIhbt_cell/(m.K_PlsmdIhbt_cell + m.P_vcell_S[t,i]) \
            * m.K_PlsmdIhbt_media/(m.K_PlsmdIhbt_media + m.P_media[t,i]) * m.K_muLac/(m.Lac[t,i] + m.K_muLac)
    else:
        return m.mu[t,i] == m.mu_max * m.K_PlsmdIhbt_cell/(m.K_PlsmdIhbt_cell + m.P_vcell_S[t,i]) \
            * m.K_PlsmdIhbt_media/(m.K_PlsmdIhbt_media + m.P_media[t,i])* m.K_muLac/(m.Lac_approx[t,i] + m.K_muLac)

def eqn48(m, t, i):
    if m.meta:
        return m.omega[t,i] == m.k_d + m.k_PlsmdCyto_cell*m.P_vcell_S[t,i]/(m.P_vcell_S[t,i] + m.K_PlsmdCyto_cell) \
            + m.k_AmmCyto*m.Amm[t,i]/(m.Amm[t,i] + m.K_AmmCyto)
    else:
        return m.omega[t,i] == m.k_d + m.k_PlsmdCyto_cell*m.P_vcell_S[t,i]/(m.P_vcell_S[t,i] + m.K_PlsmdCyto_cell) \
            + m.k_AmmCyto*m.Amm_approx[t,i]/(m.Amm_approx[t,i] + m.K_AmmCyto)


def eqn49(m, t, i):
    return m.q_Glc[t,i] == (m.mu[t,i]*m.Y_Glc + m.m_Glc) * m.K_GlcLac/(m.K_GlcLac + m.Lac[t,i])

def eqn50(m, t, i):
    return m.q_Lac[t,i] == (m.mu[t,i]*m.Y_Lac + m.m_Lac) - m.k_Lac*m.Lac[t,i]*m.mu[t,i]/(m.mu[t,i] + m.K_Lacmu)

def eqn51(m, t, i):
    return m.q_Gln[t,i] == (m.mu[t,i]*m.Y_Gln + m.m_Gln) * m.K_GlnGlc/(m.q_Glc[t,i] + m.K_GlnGlc)
                        
def eqn52(m, t, i):
    return m.q_Amm[t,i] == (m.mu[t,i]*m.Y_Amm + m.m_Amm) * m.K_AmmXv/(m.K_AmmXv + m.Xv[t,i])

def eqn53(m, t, i):
    return m.q_Glu[t,i] == (m.mu[t,i]*m.Y_Glu + m.m_Glu) - m.k_Glu*m.Glu[t,i]*m.mu[t,i]/(m.mu[t,i] + m.K_Glumu)


def eqn54(m, t, i):
    return m.r_pPack_uptk[t,i] == m.k_PlsmdUptk * m.pPack_media[t,i] \
        * m.Xv[t,i]/(m.Xv[t,i] + m.K_PlsmdUptk_Xv)\
        * m.mu[t,i]/(m.mu[t,i] + m.K_PlsmdUptk_mu)

def eqn55(m, t, i):  
    return m.r_pVec_uptk[t,i] == m.k_PlsmdUptk * m.pVec_media[t,i] \
        * m.Xv[t,i]/(m.Xv[t,i] + m.K_PlsmdUptk_Xv)\
        * m.mu[t,i]/(m.mu[t,i] + m.K_PlsmdUptk_mu)

def eqn56(m, t, i):  
    return m.r_pHelp_uptk[t,i] == m.k_PlsmdUptk * m.pHelp_media[t,i] \
        * m.Xv[t,i]/(m.Xv[t,i] + m.K_PlsmdUptk_Xv)\
        * m.mu[t,i]/(m.mu[t,i] + m.K_PlsmdUptk_mu)


def eqn57(m, t, i):
    if m.meta:
        return m.r_Rep[t,i] == m.k_Rep*m.pPack_nuc[t,i]/(m.K_Amm**2 + m.Amm[t,i]**2) \
                * m.pHelp_nuc[t,i]/(m.pHelp_nuc[t,i] + m.K_Rep_pHelp)
    else:
        return m.r_Rep[t,i] == m.k_Rep*m.pPack_nuc[t,i]/(m.K_Amm**2 + m.Amm_approx[t,i]**2) \
                * m.pHelp_nuc[t,i]/(m.pHelp_nuc[t,i] + m.K_Rep_pHelp)

def eqn58(m, t, i):
    if m.meta:
        return m.r_VP[t,i] == m.k_VP*m.pPack_nuc[t,i]/(m.K_Amm**2 + m.Amm[t,i]**2) \
                * m.Rep[t,i]/(m.Rep[t,i] + m.K_VP_Rep)
    else:
        return m.r_VP[t,i] == m.k_VP*m.pPack_nuc[t,i]/(m.K_Amm**2 + m.Amm_approx[t,i]**2) \
                * m.Rep[t,i]/(m.Rep[t,i] + m.K_VP_Rep)
"""
def eqn57(m, t, i):
    return m.r_Rep[t,i] == m.k_Rep*m.pPack_nuc[t,i]*m.mu[t,i] \
            * m.pHelp_nuc[t,i]/(m.pHelp_nuc[t,i] + m.K_Rep_pHelp)

def eqn58(m, t, i):
    return m.r_VP[t,i] == m.k_VP*m.pPack_nuc[t,i]*m.mu[t,i] \
            * (m.Rep[t,i] + m.eps)**0.5
"""

def eqn59(m, t, i):
    return m.r_vDNA[t,i] == m.k_vDNA*m.pVec_nuc[t,i]/(m.K_vDNA_pVec + m.pVec_nuc[t,i]) \
        * m.pHelp_nuc[t,i] / (m.K_vDNA_pHelp + m.pHelp_nuc[t,i]) \
            * m.Rep[t,i] / (m.K_vDNA_Rep + m.Rep[t,i])

def eqn60(m, t, i):
    return m.r_Pack[t,i] == m.k_Pack*m.vDNA[t,i]*m.Rep[t,i] \
        / (m.K_Pack_Rep + m.Rep[t,i]) * m.eCap_nuc[t,i]/(m.K_Pack_eCap + m.eCap_nuc[t,i])


def Equations(m):
    m.eqn1 = Constraint(m.t, m.i, rule=eqn1)
    m.eqn2 = Constraint(m.t, m.i, rule=eqn2)
    if m.meta:
        # Metabolism
        m.eqn3 = Constraint(m.t, m.i, rule=eqn3)
        m.eqn4 = Constraint(m.t, m.i, rule=eqn4)
        m.eqn5 = Constraint(m.t, m.i, rule=eqn5)
        m.eqn6 = Constraint(m.t, m.i, rule=eqn6)
        m.eqn7 = Constraint(m.t, m.i, rule=eqn7)
        m.eqn8 = Constraint(m.t, m.i, rule=eqn8)
    
    # Transfection reagents
    m.eqn9 = Constraint(m.t, m.i, rule=eqn9)
    m.eqn10 = Constraint(m.t, m.i, rule=eqn10)
    m.eqn11 = Constraint(m.t, m.i, rule=eqn11)
    m.eqn12 = Constraint(m.t, m.i, rule=eqn12)
    m.eqn13 = Constraint(m.t, m.i, rule=eqn13)
    m.eqn14 = Constraint(m.t, m.i, rule=eqn14)
    m.eqn15 = Constraint(m.t, m.i, rule=eqn15)
    m.eqn16 = Constraint(m.t, m.i, rule=eqn16)
    m.eqn17 = Constraint(m.t, m.i, rule=eqn17)

    if m.viralprod:
        m.eqn18 = Constraint(m.t, m.i, rule=eqn18)
        m.eqn19 = Constraint(m.t, m.i, rule=eqn19)
        m.eqn20 = Constraint(m.t, m.i, rule=eqn20)
        m.eqn21 = Constraint(m.t, m.i, rule=eqn21)
        m.eqn22 = Constraint(m.t, m.i, rule=eqn22)
        m.eqn23 = Constraint(m.t, m.i, rule=eqn23)
        m.eqn24 = Constraint(m.t, m.i, rule=eqn24)
        m.eqn25 = Constraint(m.t, m.i, rule=eqn25)    
        m.eqn26 = Constraint(m.t, m.i, rule=eqn26)

        # Viral production
        m.eqn27 = Constraint(m.t, m.i, rule=eqn27)
        m.eqn28 = Constraint(m.t, m.i, rule=eqn28)
        m.eqn29 = Constraint(m.t, m.i, rule=eqn29)
        m.eqn30 = Constraint(m.t, m.i, rule=eqn30)
        m.eqn31 = Constraint(m.t, m.i, rule=eqn31)
        m.eqn32 = Constraint(m.t, m.i, rule=eqn32)
        m.eqn33 = Constraint(m.t, m.i, rule=eqn33)
        m.eqn34 = Constraint(m.t, m.i, rule=eqn34)
        m.eqn35 = Constraint(m.t, m.i, rule=eqn35)
        m.eqn36 = Constraint(m.t, m.i, rule=eqn36)
        m.eqn37 = Constraint(m.t, m.i, rule=eqn37)
        m.eqn38 = Constraint(m.t, m.i, rule=eqn38)
        m.eqn39 = Constraint(m.t, m.i, rule=eqn39)
        m.eqn40 = Constraint(m.t, m.i, rule=eqn40)
        m.eqn41 = Constraint(m.t, m.i, rule=eqn41)
        m.eqn42 = Constraint(m.t, m.i, rule=eqn42)
        m.eqn43 = Constraint(m.t, m.i, rule=eqn43)
    if m.system == 'bioreactor':
        # Media volume change
        m.eqn44 = Constraint(m.t, m.i, rule=eqn44)

    # Algebraic equations
    m.eqn45 = Constraint(m.t, m.i, rule=eqn45)
    m.eqn46 = Constraint(m.t, m.i, rule=eqn46)
    m.eqn47 = Constraint(m.t, m.i, rule=eqn47)
    m.eqn48 = Constraint(m.t, m.i, rule=eqn48)
    if m.meta:
        m.eqn49 = Constraint(m.t, m.i, rule=eqn49)
        m.eqn50 = Constraint(m.t, m.i, rule=eqn50)
        m.eqn51 = Constraint(m.t, m.i, rule=eqn51)
        m.eqn52 = Constraint(m.t, m.i, rule=eqn52)
        m.eqn53 = Constraint(m.t, m.i, rule=eqn53)
    m.eqn54 = Constraint(m.t, m.i, rule=eqn54)
    m.eqn55 = Constraint(m.t, m.i, rule=eqn55)
    m.eqn56 = Constraint(m.t, m.i, rule=eqn56)
    if m.viralprod:
        m.eqn57 = Constraint(m.t, m.i, rule=eqn57)
        m.eqn58 = Constraint(m.t, m.i, rule=eqn58)
        m.eqn59 = Constraint(m.t, m.i, rule=eqn59)
        m.eqn60 = Constraint(m.t, m.i, rule=eqn60)
