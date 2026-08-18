from pyomo.environ import *
import click

def initialize_generators(model,
                        generator_names=None,
                        generator_at_bus=None):

    model.Generators = Set(initialize=generator_names)
    model.GeneratorsAtBus = Set(model.Buses, initialize=generator_at_bus)
    
def maximum_minimum_activepower_output_generators(model, minimum_power_output=None, maximum_power_output=None):

    model.MinimumActivePowerOutput = Param(model.Generators, initialize=minimum_power_output, within=NonNegativeReals, default=0.0)
    model.MaximumActivePowerOutput = Param(model.Generators, initialize=maximum_power_output, within=NonNegativeReals, default=0.0)

def maximum_minimum_reactivepower_output_generators(model, minimum_power_output=None, maximum_power_output=None):

    model.MinimumReactivePowerOutput = Param(model.Generators, initialize=minimum_power_output, within=Reals, default=0.0)
    model.MaximumReactivePowerOutput = Param(model.Generators, initialize=maximum_power_output, within=Reals, default=0.0)
