from pyomo.environ import *
import os
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm

def initialize_DERload(model, DER_at_bus=None, DER_names=None, DER_phases=None, DER_pi=None, DER_P_per_phase=None, DERP_dict=None, alpha=None, MutDER=None, InfeasibleFlag=False, InfeasibleDERs=None):
    
    model.DERAtBus = Set(model.Buses, initialize=DER_at_bus)
    model.DER = Set(initialize = DER_names)
    model.DERPi =  Param(model.DER, initialize=DER_pi, within=Any)
    model.DERP_Phases = Param(model.Phases, model.DER, initialize=DERP_dict, within=Any)
    model.DERPhases = Param(model.DER, initialize=DER_phases, within=Any)
    model.DERP  = Param(model.DER, initialize=DER_P_per_phase, within=Any)

    model.DERAlpha = Var(model.DER, bounds=[0,1], within=Reals)

    if alpha is not None:
        model_ders = set(model.DER)
        missing_alpha = model_ders - set(alpha)

        if missing_alpha:
            raise ValueError(f"Missing alpha values for DERs: {missing_alpha}")

        infeasible_ders = set(InfeasibleDERs or  ())
        unkown_ders = infeasible_ders - model_ders

        if unkown_ders:
            raise ValueError(f"Unknown DERs in InfeasibleDERs: {unkown_ders}")

        for d in model.DER:
            alpha_ref = min(1.0, max(0.0, float(alpha[d])))

            model.DERAlpha[d].set_value(alpha_ref)

            if d in infeasible_ders and alpha_ref > 1e-8:
                model.DERAlpha[d].unfix()
                model.DERAlpha[d].setlb(0.0)
                # model.DERAlpha[d].setlb(float(alpha_ref/2.0))
                model.DERAlpha[d].setub(alpha_ref)
            else:
                model.DERAlpha[d].fix(alpha_ref)
    if MutDER is not None:
        for d in MutDER:
            model.DERAlpha[d].unfix()

def initialize_DER_reactive_recourse_variable(model):
    model.DERQ = Var(model.DER, within=Reals, initialize=0.0, doc="Per-phase")

