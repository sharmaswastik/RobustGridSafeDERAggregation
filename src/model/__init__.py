import logging
import pandas as pd
import numpy as np
import re
from IDSO import DER_at_bus, DER_bids_at_bus, DER_offers_at_bus, DER_names, DER_bids_names, DER_offers_names, DER_phases, DER_bids_phases, DER_offers_phases, DER_pi, DER_pi_bids, DER_pi_offers, DER_P_per_phase, DER_P_per_phase_bids, DER_P_per_phase_offers, DERP_dict, DERP_offers_dict, DERP_bids_dict, MaxP

from .model import (create_model, initialize_buses,
                initialize_time_periods, initialize_model, thermal_limits, Suffix
                    )
from .network import (initialize_network, derive_network, calculate_network_parameters \
    , create_incidence_matrix)

from .generators import (initialize_generators, maximum_minimum_activepower_output_generators, \
    maximum_minimum_reactivepower_output_generators) 

from .spot_load import (initialize_spotload)
from .DER_load import (initialize_DER_reactive_recourse_variable, initialize_DERload)

from .constraints import (activepower_balance_constraint, reactivepower_balance_constraint, \
    voltage_constraint, objective_function, constraint_net_activepower, constraint_net_reactivepower, constraint_net_reactivepower_adjustable, der_reactive_capability_constraints, constraint_slack_bus, thermal_limit_constraint, substation_power_limit_constraint, mutual_case_constraint, constraint_active_scheduled_interchange, constraint_reactive_scheduled_interchange, feasibility_objective_function,
    der_reactive_deviation_constraints,
    der_reactive_deviation_objective_function,
    network_violation_objective_function,
    network_violation_constraints
    )

from .global_params import update_func

from .capacitor import (initialize_cap, maximum_output_capacitors, switching_cap_func)

from .utils import (cap_values_per_phase)

from solver import solve_model, PSSTResults

logger = logging.getLogger(__file__)

def build_model(case=None,
                substation_df=None,
                generator_df=None,
                Pload_df=None,
                Qload_df=None,
                branch_df=None,
                bus_df=None,
                config=None,
                SpotLoads=None,
                CapData=None,
                thermal_df=None,
                Inp_thermal=None,
                alpha=None,
                results_cap=None,
                OnlyBids=False,
                OnlyOffers=False,
                BothBidsOffers=False,
                NaiveCase=False,
                FinalCase=False,
                MutualCase=False,
                MutDER=None,
                OnlyBids_MC=False,
                MCData=None,
                SourceActivePower=None,
                SourceReactivePower=None,
                InfeasibleFlag=False,
                AdjustableDERQ=False,
                MinimizeDERQDeviation=False,
                FixedGeneration=False,
                PowerFlowMode=False,
                removed_bids=None,
                removed_offers=None,
                MinimizeNetworkViolations=False,
                InfeasibleDERs=None,
                CollectDuals=True,
                ):
    if InfeasibleDERs is not None:
        if alpha is None:
            raise ValueError("alpha must be provided if InfeasibleDERs is not None")
        if MutualCase or PowerFlowMode or NaiveCase:
            raise ValueError(
                "Infeasible DERs is not compatible with MutualCase, PowerFlowMode, or NaiveCase.")
        if (
            MinimizeNetworkViolations or MinimizeDERQDeviation
        ):
            raise ValueError(
                "Infeasible DERs is not compatible with MinimizeNetworkViolations or MinimizeDERQDeviation."
            )
        AdjustableDERQ = True
        
    if MinimizeNetworkViolations:
        if PowerFlowMode:
            raise ValueError(
                "MinimizeNetworkViolations and PowerFlowMode "
                "must not be enabled together."
            )

        if MinimizeDERQDeviation:
            raise ValueError(
                "MinimizeNetworkViolations and MinimizeDERQDeviation "
                "define different objectives."
            )

        AdjustableDERQ = True
    
    if config is None:
        config = dict()
    
    generator_df = generator_df 
    Pload_df = Pload_df 
    Qload_df = Qload_df 
    branch_df = branch_df 
    bus_df = bus_df 

    config = config

    branch_df.index = np.arange(1,len(branch_df)+1)
    bus_df.index = np.arange(1,len(bus_df)+1)
    # Pload_df.index = np.arange(1,len(Pload_df)+1)
    # Qload_df.index = np.arange(1,len(Qload_df)+1)
    generator_df.index = np.arange(1,len(generator_df)+1)
    substation_df.index = np.arange(1,len(substation_df)+1)

    
    branch_df.index = branch_df.index.astype(object)
    generator_df.index = generator_df.set_index('Gen_i').index.astype(object)
    bus_df.index = bus_df.set_index('bus_i').index.astype(object)
    Pload_df.index = Pload_df.index.astype(object)
    Qload_df.index = Qload_df.index.astype(object)
    branch_df = branch_df.astype(object)
    generator_df = generator_df.astype(object)
    substation_df = substation_df.astype(object)
    bus_df = bus_df.astype(object)
    Qload_df = Qload_df.astype(object)
    CapData.index = CapData.set_index('Cap_i').index.astype(object)
    CapData = CapData.astype(object)

    c_size=int((config[1].shape[0])/3)

    model = create_model(collect_duals=CollectDuals)

    time_periods = list(Pload_df[Pload_df['Phase'] == 'A'].copy().index)
    initialize_buses(model, bus_names=bus_df.index)
    initialize_time_periods(model, time_periods=time_periods)

    Z_base = (substation_df['V_Base'].unique().item()**2/(substation_df['S_Base'].unique().item()/3000))
    I_base = (substation_df['S_Base'].unique().item())/(substation_df['V_Base'].unique().item()*3)
    HeadBus = substation_df['HeadBus'].values[0]

    initialize_network(model, transmission_lines=list(branch_df.index), leng=branch_df['Length'].to_dict(), bus_from=branch_df['F_BUS'].to_dict(), bus_to=branch_df['T_BUS'].to_dict(), config=branch_df['Config'].to_dict(), Z_base=Z_base, ThermalCap=branch_df['ThermalCap'].to_dict(), branch_phases=branch_df['Phases'].to_dict(), HeadBus=[HeadBus])

    lines_to = {b: list() for b in bus_df.index.unique()}
    lines_from = {b: list() for b in bus_df.index.unique()}
    
    for i, l in branch_df.iterrows():
        lines_from[l['F_BUS']].append(i)
        lines_to[l['T_BUS']].append(i)

    derive_network(model, lines_from=lines_from, lines_to=lines_to)
    A_tilde, A_0, A = create_incidence_matrix(model)
    inv_A = np.linalg.inv(A)
    D_r, D_x = calculate_network_parameters(model, config=config, c_size=c_size)
    A_dash, R_d, X_d = update_func(A_0, A, D_r, D_x) #I've incorporated that V will account for P_L
    generator_at_bus = {b: list() for b in generator_df['GEN_BUS'].unique()}

    for i, g in generator_df.iterrows():
        generator_at_bus[g['GEN_BUS']].append(i)
   
    initialize_generators(model,
                        generator_names=generator_df.index,
                        generator_at_bus=generator_at_bus)
    
    maximum_minimum_activepower_output_generators(model,
                                        minimum_power_output=generator_df['PMIN'].to_dict(),
                                        maximum_power_output=generator_df['PMAX'].to_dict())

    maximum_minimum_reactivepower_output_generators(model,
                                        minimum_power_output=generator_df['QMIN'].to_dict(),
                                        maximum_power_output=generator_df['QMAX'].to_dict())
    
    
    initialize_spotload(model, SpotLoads = SpotLoads)

    removed_bids = set(removed_bids or [])
    removed_offers = set(removed_offers or [])

    unknown_bids = removed_bids - set(DER_bids_names)
    unknown_offers = removed_offers - set(DER_offers_names)

    if unknown_bids:
        raise ValueError(f"Unknown bid DERs: {sorted(unknown_bids)}")

    if unknown_offers:
        raise ValueError(f"Unknown offer DERs: {sorted(unknown_offers)}")

    if OnlyBids or OnlyBids_MC:
        source_names = DER_bids_names
        source_at_bus = DER_bids_at_bus
        source_phases = DER_bids_phases
        source_pi = DER_pi_bids
        source_p = DER_P_per_phase_bids
        source_p_by_phase = DERP_bids_dict
        removed_ders = removed_bids

    elif OnlyOffers:
        source_names = DER_offers_names
        source_at_bus = DER_offers_at_bus
        source_phases = DER_offers_phases
        source_pi = DER_pi_offers
        source_p = DER_P_per_phase_offers
        source_p_by_phase = DERP_offers_dict
        removed_ders = removed_offers

    else:
        # BothBidsOffers, NaiveCase, FinalCase, MutualCase
        source_names = DER_names
        source_at_bus = DER_at_bus
        source_phases = DER_phases
        source_pi = DER_pi
        source_p = DER_P_per_phase
        source_p_by_phase = DERP_dict
        removed_ders = removed_bids | removed_offers

    active_names = [ d for d in source_names if d not in removed_ders ]
    active_set = set(active_names)
    active_at_bus = {
        bus: [d for d in ders if d in active_set]
        for bus, ders in source_at_bus.items()
        if any(d in active_set for d in ders)
    }
    active_phases = {
        d : source_phases[d]
        for d in active_names
    }
    active_pi = {
        d: source_pi[d]
        for d in active_names
    }
    active_p = {
        d: source_p[d]
        for d in active_names
    }
    active_p_by_phase = {
        (phase, d): value
        for (phase, d), value in source_p_by_phase.items()
        if d in active_set
    }
    active_alpha = (
        None
        if alpha is None
        else {d: alpha[d] for d in active_names}
    )
    active_mut_der = (
        None
        if MutDER is None
        else [d for d in MutDER if d in active_set]
    )

    initialize_DERload(
        model,
        DER_at_bus=active_at_bus,
        DER_names=active_names,
        DER_phases=active_phases,
        DER_pi=active_pi,
        DER_P_per_phase=active_p,
        DERP_dict=active_p_by_phase,
        alpha=active_alpha,
        MutDER=active_mut_der if MutualCase else None,
        InfeasibleFlag=InfeasibleFlag,
        InfeasibleDERs=InfeasibleDERs,
    )

    if AdjustableDERQ:
        initialize_DER_reactive_recourse_variable(model)
    
    if FixedGeneration is not None:
        for key, dispatch in FixedGeneration["P"].items():
            model.ActivePowerGenerated[key].fix(dispatch)
        for key, dispatch in FixedGeneration["Q"].items():
            model.ReactivePowerGenerated[key].fix(dispatch)

    cap_at_bus = {b: list() for b in CapData['Node'].unique()}
    
    for i, c in CapData.iterrows():
        cap_at_bus[c['Node']].append(i)
    
    maximum_output = cap_values_per_phase(CapData)
    
    initialize_cap(model, cap_names=CapData.index,
                            cap_at_bus=cap_at_bus)

    maximum_output_capacitors(model,
                            maximum_output = maximum_output,
                            Switching = CapData['Switching'].to_dict())

    switching_cap_func(model, 
                    Switching = CapData['Switching'].to_dict(),
                    SwitchingCost=CapData['Cost'].to_dict())

    if results_cap == None:

        if NaiveCase:
            initialize_model(model, V_max=bus_df['Vmax'].unique().item(), V_min=bus_df['Vmin'].unique().item(), V_base = (substation_df['V_Base'].unique().item()), S_base = (substation_df['S_Base'].unique().item()), I_base = I_base, Sub_S_Max= (substation_df['Ssub_Max'].unique().item()), bus_phases = bus_df['Phases'].to_dict(), BigM=case[0], pf=case[1], MaxPOffer=MaxP, PowerFlowMode=PowerFlowMode, MinimizeNetworkViolations=MinimizeNetworkViolations)
        else:
            initialize_model(model, V_max=bus_df['Vmax'].unique().item(), V_min=bus_df['Vmin'].unique().item(), V_base = (substation_df['V_Base'].unique().item()), S_base = (substation_df['S_Base'].unique().item()), I_base = I_base, Sub_S_Max= (substation_df['Ssub_Max'].unique().item()), bus_phases = bus_df['Phases'].to_dict(), BigM=case[0], pf=case[1], MaxPOffer=MaxP, PowerFlowMode=PowerFlowMode, MinimizeNetworkViolations=MinimizeNetworkViolations)

    if Inp_thermal == '1' and not NaiveCase:
        thermal_df.index = thermal_df.set_index('c').index.astype(object)
        thermal_limits(model, alpha_c=thermal_df['alpha_c'].to_dict(), beta_c=thermal_df['beta_c'].to_dict(), delta_c=thermal_df['delta_c'].to_dict())
        if not PowerFlowMode and not MinimizeNetworkViolations:
            thermal_limit_constraint(model)
    
    if NaiveCase:
        constraint_slack_bus(model, V_Slack = bus_df['V_0'].iloc[int(HeadBus[-1][-1])])
        voltage_constraint(model, A_dash, R_d, X_d)
        activepower_balance_constraint(model, A.T, A_0.T)
        reactivepower_balance_constraint(model, A.T, A_0.T)
        constraint_net_activepower(model)
        constraint_net_reactivepower(model)
        objective_function(model)
    
    else:
        #active_power_source_bus_constraint(model)
        #reactive_power_source_bus_constraint(model)
        constraint_slack_bus(model, V_Slack = bus_df['V_0'].iloc[int(HeadBus[-1][-1])])
        voltage_constraint(model, A_dash, R_d, X_d)
        activepower_balance_constraint(model, A.T, A_0.T)
        reactivepower_balance_constraint(model, A.T, A_0.T)
        constraint_net_activepower(model)

        if AdjustableDERQ:
            constraint_net_reactivepower_adjustable(model)
            der_reactive_capability_constraints(model)

            if MinimizeDERQDeviation:
                der_reactive_deviation_constraints(model)
        else:
            constraint_net_reactivepower(model)
        
        if MutualCase:
            mutual_case_constraint(model)
        
        if OnlyBids_MC:
            constraint_active_scheduled_interchange(model, SourceActivePower)
            constraint_reactive_scheduled_interchange(model, SourceReactivePower)

        if (hasattr(model, "Edges") and not PowerFlowMode and not MinimizeNetworkViolations):
            substation_power_limit_constraint(model)

        if InfeasibleDERs is not None and Inp_thermal == '1':
            required_constraints = [
                "ThermalLimitConstraint",
                "SubstationPowerLimitConstraint",
            ]
            missing_constraints = [
                name for name in required_constraints
                if not hasattr(model, name)
            ]
            if missing_constraints:
                raise RuntimeError(
                    "InfeasibleDERs model is missing constraints: "
                    f"{missing_constraints}"
                )

        if MinimizeNetworkViolations:
            network_violation_constraints(model)
            network_violation_objective_function(model)

        elif InfeasibleDERs is not None:
            objective_function(model)

        elif AdjustableDERQ:
            if MinimizeDERQDeviation:
                der_reactive_deviation_objective_function(model)
            else:
                feasibility_objective_function(model)

        elif PowerFlowMode:
            feasibility_objective_function(model)

        else:
            # substation_power_limit_constraint(model)
            objective_function(model)
        # if OnlyOffers:
        #     # constraint_net_activepower_onlyoffers(model)
        #     # constraint_net_reactivepower_onlyoffers(model)
        #     constraint_net_activepower(model)
        #     constraint_net_reactivepower(model)
        # else:
        #     constraint_net_activepower(model)
        #     if AdjustableDERQ:
        #         constraint_net_reactivepower_adjustable(model)
        #         der_reactive_capability_constraints(model)
        #     else:
        #         constraint_net_reactivepower(model)
        
        # if MutualCase:
        #     mutual_case_constraint(model)
        
        # if OnlyBids_MC:
        #     constraint_active_scheduled_interchange(model, SourceActivePower)
        #     constraint_reactive_scheduled_interchange(model, SourceReactivePower)

        # substation_power_limit_constraint(model)
        # if AdjustableDERQ:
        #     der_reactive_capability_constraints(model)
        # else:
        #     objective_function(model)
    return PSSTModel(model)

class PSSTModel(object):

    def __init__(self, model, is_solved=False):
        self._model = model
        self._is_solved = is_solved
        self._status = None
        self._results = None

    def __repr__(self):

        repr_string = 'status={}'.format(self._status)

        string = '<{}.{}({})>'.format(
                    self.__class__.__module__,
                    self.__class__.__name__,
                    repr_string,)


        return string

    def solve(self, solver='glpk', verbose=False, keepfiles=True, **kwargs):
        TC = solve_model(self._model, solver=solver, verbose=verbose, keepfiles=keepfiles, **kwargs)
        self._results = PSSTResults(self._model)
        return TC

    def sort_buses(self):
        self._model.Buses = sorted(self._model.Buses, key=lambda bus: int(re.search(r'\d+', bus).group()))

    @property
    def results(self):
        return self._results
