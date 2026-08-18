from pyomo.environ import *
import numpy as np

def create_model():
    model = ConcreteModel()
    model.dual = Suffix(direction = Suffix.IMPORT)
    return model

def initialize_buses(model,
                    bus_names=None            ):

    model.Buses = Set(ordered=True, initialize=bus_names)
    model.Dim_r = RangeSet(1, 3*(len(model.Buses)-1))
    model.Dim_c = RangeSet(1, 3*(len(model.Buses)-1))
    model.Dim_a = RangeSet(1, 3)
  
def initialize_time_periods(model,
                    time_periods=None,
                    time_period_length=1.0
                    ):

    if time_periods is None:
        number_of_time_periods = None
    else:
        number_of_time_periods = len(time_periods)

    model.TimePeriods = RangeSet(1, number_of_time_periods)
    model.NumTimePeriods = Param(initialize=number_of_time_periods)
    model.TimePeriodLength = Param(initialize=time_period_length)
def initialize_model(model,
                    V_max,
                    V_min,
                    V_base,
                    S_base,
                    I_base,
                    Sub_S_Max,
                    bus_phases,
                    time_period_length=1.0,
                    results_cap=None,
                    BigM=1000,
                    pf = 0.9,
                    MaxPOffer=1000,
                    PowerFlowMode=False,
                    MinimizeNetworkViolations=False):   
    def activepower_bounds_rule(m, p, g, t):
        return(0, m.MaximumActivePowerOutput[g]/(m.S_Base/3))
    
    def reactivepower_bounds_rule(m, p, g, t):
        return(m.MinimumReactivePowerOutput[g]/(m.S_Base/3), m.MaximumReactivePowerOutput[g]/(m.S_Base/3))
    
    model.BusPhase = Param(model.Buses, initialize=bus_phases, within=Any)         
    # Amount of power flowing in each line each time period
    model.P_L = Var(model.Phases, model.TransmissionLines, model.TimePeriods, initialize = 0, within=Reals)
    model.Q_L = Var(model.Phases, model.TransmissionLines, model.TimePeriods, initialize = 0, within=Reals)
    # Amount of net power injected in a bus by a generator at each time period
    model.P = Var(model.Phases, model.Buses, model.TimePeriods, initialize = 0, within=Reals)
    model.Q = Var(model.Phases, model.Buses, model.TimePeriods, initialize = 0, within=Reals)
    # Base Values to convert V, P and Q to p.u. values
    model.V_Base = Param(initialize=V_base)
    model.S_Base = Param(initialize=S_base)
    model.I_Base = Param(initialize=I_base)
    
    model.V_min  = Param(initialize=V_min)
    model.V_max  = Param(initialize=V_max)
    model.SubSMax = Param(initialize=Sub_S_Max)
    model.ActivePowerGenerated = Var(model.Phases, model.Generators, model.TimePeriods, within = NonNegativeReals, bounds = activepower_bounds_rule)
    model.ReactivePowerGenerated = Var(model.Phases, model.Generators, model.TimePeriods, within = Reals, bounds = reactivepower_bounds_rule)

    if not (PowerFlowMode or MinimizeNetworkViolations):
        model.V = Var(model.Phases, model.Buses, model.TimePeriods, within = NonNegativeReals, bounds=(V_min**2, V_max**2), initialize=1)
    else:
        model.V = Var(model.Phases, model.Buses, model.TimePeriods, within = Reals, initialize=1)
        
    model.ActivePowerAtSourceBus = Var(model.Phases, model.HeadBus, model.TimePeriods, within=Reals)
    model.ReactivePowerAtSourceBus = Var(model.Phases, model.HeadBus, model.TimePeriods, within=Reals)
    model.M = Param(initialize=BigM)
    model.PF = Param(initialize=pf)
    model.MaxPOffer = Param(initialize=MaxPOffer)
    # model.ReactivePowerPerCapacitor = Var(model.Phases, model.Capacitors, model.TimePeriods, within = NonNegativeReals, bounds = cappower_bounds_rule)
    
    # if results_cap == None:
    #     model.CapacitorState = Var(model.Phases, model.Capacitors, model.TimePeriods, within=NonNegativeIntegers, bounds=max_switch_rule)
    # else:
    #     model.CapacitorState = Var(model.Phases, model.Capacitors, model.TimePeriods, within=NonNegativeIntegers, bounds=max_switch_rule)
    #     for c in model.Capacitors:
    #         for t in model.TimePeriods:
    #             for p in model.Phases:
    #                 model.CapacitorState[p,c,t].fixed = True
    #                 model.CapacitorState[p,c,t] = results_cap[p,c,t]
    # model.CapacitorState = Var(model.Phases, model.Capacitors, model.TimePeriods, within=Reals, bounds=[0,1])
   
def thermal_limits(model, alpha_c, beta_c, delta_c):
    model.Edges = RangeSet(1,12)
    model.Alpha_c = Param(model.Edges, initialize=alpha_c)
    model.Beta_c = Param(model.Edges, initialize=beta_c)
    model.Delta_c = Param(model.Edges, initialize=delta_c)
    

    
