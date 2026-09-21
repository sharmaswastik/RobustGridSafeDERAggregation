import ast
import natsort
import os
import sys
import click
import pandas as pd
import random
import numpy as np
from pathlib import Path
from glob import glob
import datetime
import pytz
from tqdm import tqdm
import re
import threading
import pickle
import time
import pyomo.environ as pyo
import seaborn as sns
from model.constraints import model_objective_function
from model import build_model
from model.utils import (create_matrix_dict_pandas, correct_names, data_indexing, phases_from_config, descending_sort_dict, get_duals_in_numpy_vector, custom_sort_key)
from IDSO import count_bids, count_offers, combined_data, bubble_sizes, MaxP, args, DER_names, DER_pi, DER_P_per_phase, DER_P
from robust import alpha_at_vertex, wpm_find_vertices, build_robust_model
from figure_style import *

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
figures_dir = parent_dir / "Figures"
output_dir = parent_dir / "OutputFiles"
csv_dir = output_dir /"CSVFiles"
np.seterr(all='raise')

SOLVER = os.getenv('PSST_SOLVER')
LMP = args.LMP 
M = args.M
pf = args.pf
C_IDSO = args.C0

def OPF(solver, data=None, OnlyBids=False, OnlyOffers=False, BothBidsOffers=False, NaiveCase=False, FinalCase=False, alpha=None, LMP=LMP, MutualCase=False, MutDER = None, MCData=None, OnlyBids_MC=False, SourceActivePower=None, SourceReactivePower=None, InfeasibleFlag=False, AdjustableDERQ=False, MinimizeDERQDeviation=False, FixedGeneration=None, PowerFlowMode=False, removed_bids=None, removed_offers=None, MinimizeNetworkViolations=False, InfeasibleDERs=None, BuildOnly=False, CollectDuals=True):

	global model, SolverOutcomes
	result = None
	Status = None
	SolverOutcomes = (None, "not_solved")

	if OnlyBids:
		outf = 'OnlyBids'
	elif InfeasibleDERs is not None:
		outf = 'InfeasibleDERs'
	elif OnlyOffers:
		outf = 'OnlyOffers'
	elif BothBidsOffers:
		outf = 'BothBidsOffers'
	elif NaiveCase:
		outf = 'NaiveCase'
	elif FinalCase:
		outf = 'FinalCase'
	elif MutualCase:
		outf = 'MutualCase'
	elif OnlyBids_MC:
		outf = 'OnlyBids_MC'
	elif PowerFlowMode:
		outf = 'PowerFlowMode'
	elif MinimizeNetworkViolations:
		outf = 'MinimizeNetworkViolations'
	
	t = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d-%H-%M-%S-%f")
	output = os.path.join(os.getcwd(), "../OutputFiles/T-DOPFFiles/Results_{}_{}.dat").format(t,outf)

	datafile_csv = "OPF_data/"+args.TestCase

	branch_df = pd.read_csv(datafile_csv+'/branch.csv')
	generator_df = pd.read_csv(datafile_csv+'/gen.csv')
	bus_df = pd.read_csv(datafile_csv+'/bus.csv')
	capacitor_df = pd.read_csv(datafile_csv+'/CapData.csv')
	substation_df = pd.read_csv(datafile_csv+'/Substation.csv')
	SpotLoadsP = data_indexing(datafile_csv+'/SpotLoadsP.csv', 'Node', len(bus_df), 'Bus')
	SpotLoadsQ = data_indexing(datafile_csv+'/SpotLoadsQ.csv', 'Node', len(bus_df), 'Bus')

	global fixedloadsum
	fixedloadsum = SpotLoadsP.sum().sum()

	if MCData is not None:
		for i in MCData:
			for k, v in i.items():
				for x in v[1]:
					if k[1] in SpotLoadsP.index:
						print(k[1], v[0]*v[-1]/len(v[1]))
						SpotLoadsP.loc[k[1], x] -= v[0]*v[-1]/len(v[1])
						SpotLoadsQ.loc[k[1], x] -= v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1]))
					else:
						if x == 'A':
							print(k[1],v[0]*v[-1]/len(v[1]))
							SpotLoadsP.loc[k[1]] =  [-v[0]*v[-1]/len(v[1]), 0, 0]
							SpotLoadsQ.loc[k[1]] =  [-v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1])), 0, 0]
						if x == 'B':
							print(k[1],v[0]*v[-1]/len(v[1]))
							SpotLoadsP.loc[k[1]] =  [0, -v[0]*v[-1]/len(v[1]), 0]
							SpotLoadsQ.loc[k[1]] =  [0, -v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1])), 0]
						if x == 'C':
							print(k[1],v[0]*v[-1]/len(v[1]))
							SpotLoadsP.loc[k[1]] =  [0, 0, -v[0]*v[-1]/len(v[1])]
							SpotLoadsQ.loc[k[1]] =  [0, 0, -v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1]))]
			
			SpotLoadsP.sort_index(inplace=True)
			SpotLoadsQ.sort_index(inplace=True)


	if os.path.isfile(datafile_csv+"/thermal_constraint_data.xlsx"): 
		thermal_const_df = pd.read_excel(datafile_csv+"/thermal_constraint_data.xlsx")
		print("\n\nThermal constraints related data found.")
		Inp_thermal = '1'

	else:
		thermal_const_df = None
		Inp_thermal = '0'
		print("Thermal constraint related data file not found, the problem will ignore thermal constraints.")

	config_dict, config_numpy = create_matrix_dict_pandas(datafile_csv+"/config.csv")

	branch_df = correct_names(branch_df, 'F_BUS', 'Bus')
	branch_df = correct_names(branch_df, 'T_BUS', 'Bus')
	branch_df = phases_from_config(branch_df, config_numpy, need_phases=True)
	bus_df = correct_names(bus_df, 'bus_i', 'Bus')
	bus_df = phases_from_config(bus_df, need_phases=False)
	generator_df = correct_names(generator_df, 'Gen_i', 'GenCo')
	generator_df = correct_names(generator_df, 'GEN_BUS', 'Bus')
	capacitor_df = correct_names(capacitor_df, 'Cap_i', 'Cap')
	capacitor_df = correct_names(capacitor_df, 'Node', 'Bus')
	substation_df = correct_names(substation_df, 'HeadBus', 'Bus')

	cwd = os.getcwd()
	path = r"OPF_data\load" 
	path = os.path.join(cwd, path)
	Pload_df = []
	Qload_df = []

	csv_files = glob(os.path.join(path, "*.csv"))
	
	for f in csv_files:
		df = pd.read_csv(f)
		p_df = os.path.basename(f)
		p = p_df.split('-')[-1].split('.')[0]
		p = p.capitalize()
		df['Phase'] = p
		if p_df[0] == 'P':
			Pload_df.append(df)
		else:
			Qload_df.append(df)
	Pload_df = pd.concat(Pload_df, ignore_index=False)
	Qload_df = pd.concat(Qload_df, ignore_index=False)

	HeadBus = substation_df['HeadBus'].values[0]

	def build_case_model(alpha_values=None):
		"""Build the fully specified model for this OPF case."""
		model_kwargs = {
			"case": (M, pf),
			"substation_df": substation_df,
			"branch_df": branch_df,
			"generator_df": generator_df,
			"bus_df": bus_df,
			"Pload_df": Pload_df,
			"Qload_df": Qload_df,
			"config": (config_dict, config_numpy),
			"SpotLoads": (SpotLoadsP, SpotLoadsQ),
			"CapData": capacitor_df,
			"thermal_df": thermal_const_df,
			"Inp_thermal": Inp_thermal,
			"OnlyBids": OnlyBids,
			"OnlyOffers": OnlyOffers,
			"BothBidsOffers": BothBidsOffers,
			"NaiveCase": NaiveCase,
			"FinalCase": FinalCase,
			"MutualCase": MutualCase,
			"OnlyBids_MC": OnlyBids_MC,
			"SourceActivePower": SourceActivePower,
			"SourceReactivePower": SourceReactivePower,
			"InfeasibleFlag": InfeasibleFlag,
			"AdjustableDERQ": AdjustableDERQ,
			"MinimizeDERQDeviation": MinimizeDERQDeviation,
			"FixedGeneration": FixedGeneration,
			"PowerFlowMode": PowerFlowMode,
			"removed_bids": removed_bids,
			"removed_offers": removed_offers,
			"MinimizeNetworkViolations": MinimizeNetworkViolations,
			"InfeasibleDERs": InfeasibleDERs,
			"CollectDuals": CollectDuals,
		}

		if alpha_values is not None:
			model_kwargs["alpha"] = alpha_values
		if MutualCase:
			model_kwargs["MutDER"] = MutDER
		if OnlyBids_MC and MCData is not None:
			model_kwargs["MCData"] = MCData

		return build_model(**model_kwargs)

	if BuildOnly:
		return build_case_model(alpha)
	
	def display_loading_bar():
		while not done:
			print("Working on problem...", end="\r")
			time.sleep(0.5)
	
	done = False
	loading_thread = threading.Thread(target=display_loading_bar)
	loading_thread.start()

	try:
		if NaiveCase and alpha is not None:
			print("Doing Naive Case without using Bins")
		elif FinalCase and alpha is not None:
			print("Doing WPM Case for checking")
		elif MutualCase and alpha is not None:
			print("Doing Mutually Contingent Case")
		elif OnlyBids_MC and MCData is not None:
			print("Doing Only Bids Mutually Contingent Case")

		model = build_case_model(alpha)
		SolverOutcomes = model.solve(solver=solver)
		Status = str(SolverOutcomes[1])

		if Status == 'optimal' and NaiveCase and alpha is None:
			first_instance = model._model
			results_alpha = {}

			for bus in first_instance.Buses:
				if bus in first_instance.DERAtBus:
					for d in first_instance.DERAtBus[bus].data():
						if first_instance.DERP[d] >= 0:
							if first_instance.DERPi[d] <= LMP:
								results_alpha[d] = first_instance.DERAlpha[d].value
							else:
								results_alpha[d] = 0.0
						else:
							if first_instance.DERPi[d] >= LMP:
								results_alpha[d] = first_instance.DERAlpha[d].value
							else:
								results_alpha[d] = 0.0

			model = build_case_model(results_alpha)
			SolverOutcomes = model.solve(solver=solver)
			Status = str(SolverOutcomes[1])

	finally:
		done=True
		loading_thread.join()
		datetime_india = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

		if (Status == 'optimal'):
			model.sort_buses()
			instance = model._model
			result = model.results

			with open(output.strip("'"), 'w') as f:
				if PowerFlowMode:
					f.write("THE POWER FLOW WAS RUN AT : ")
					f.write(datetime_india.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
					f.write("\n\nSOLUTION_STATUS\n")
					f.write("optimal \t")
					f.write("\nEND_SOLUTION_STATUS\n\n")
					f.write("VOLTAGE_VIOLATIONS\n\n")
					for t in instance.TimePeriods:
						for bus in instance.Buses:
							for p in instance.Phases:
								if p not in instance.BusPhase[bus]:
									continue
								V = np.sqrt(instance.V[p, bus, t].value)
								if instance.V[p, bus, t].value < 0 :
									V = np.nan
									violation = "Negative Voltage"
								else:
									if V < instance.V_min.value:
										violation = "Undervoltage"
									elif V > instance.V_max.value:
										violation = "Overvoltage"
									else:
										violation = None
								if violation is not None:
									f.write('Phase: {} Bus: {} Interval: {} : {} p.u. ({})\n'.format(str(p), str(bus), str(t), str(round(V, 5)), violation))
					f.write("\nEND_VOLTAGE_VIOLATIONS\n\n")

					f.write("LINE_POWER_FLOWS_Violations\n\n")
					for l in sorted(instance.TransmissionLines):
						for t in instance.TimePeriods:
							for p in instance.Phases:
								P = instance.P_L[p, l, t].value
								Q = instance.Q_L[p, l, t].value

								residuals = {}
								for e in instance.Edges:
									residuals[e] = (
										instance.Alpha_c[e] * P + instance.Beta_c[e] * Q + instance.Delta_c[e]*(instance.ThermalCap[l]/instance.I_Base.value)
									)
								limiting_edge = max(residuals, key=residuals.get)
								maximum_residual = residuals[limiting_edge]
								if maximum_residual > 1e-6:
									f.write('Phase: {} Line Connecting: {} to {} Interval: {} : {} kW (Thermal limit exceeded on edge {})\n'.format(str(p), str(instance.BusFrom[l]), str(instance.BusTo[l]), str(t), str(round(P * instance.S_Base.value/3, 5)), limiting_edge))
					f.write("\nEND_LINE_POWER_FLOWS_Violations\n\n")

					f.write("SUBSTATION_POWER_LIMIT_VIOLATIONS\n\n")
					for t in instance.TimePeriods:
						for p in instance.Phases:
							P = instance.ActivePowerAtSourceBus[p,HeadBus, t].value
							Q = instance.ReactivePowerAtSourceBus[p,HeadBus, t].value

							residuals = {}
							for e in instance.Edges:
								residuals[e] = (
									instance.Alpha_c[e] * P + instance.Beta_c[e] * Q + instance.Delta_c[e]*(instance.SubSMax.value/instance.I_Base.value)
								)
							limiting_edge = max(residuals, key=residuals.get)
							maximum_residual = residuals[limiting_edge]
							if maximum_residual > 1e-6:
								f.write('Phase: {} Substation Bus: {} Interval: {} : {} kW (Substation power limit exceeded on edge {})\n'.format(str(p), str(HeadBus), str(t), str(round(P * instance.S_Base.value/3, 5)), limiting_edge))
					f.write("\nEND_SUBSTATION_POWER_LIMIT_VIOLATIONS\n\n")
				else:
					f.write("THE OPF WAS RUN AT : ") 
					f.write(datetime_india.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
					f.write("\n\nSOLUTION_STATUS\n")
					f.write("optimal \t")
					f.write("\nEND_SOLUTION_STATUS\n\n")

					f.write("VOLTAGE MAGNITUDES\n\n")
					for bus in instance.Buses:
						for t in instance.TimePeriods:
							for p in instance.Phases: 
								if p in instance.BusPhase[bus]:
									f.write('Phase: {} Bus: {} Interval: {} : {} p.u.\n'.format(str(p), str(bus), str(t), str(round(np.sqrt(instance.V[p, bus, t].value), 5))))
								else:
									f.write('Phase: {} Bus: {} Interval: {} : {} \n'.format(str(p), str(bus), str(t), 'NaN', 5))
					f.write("\nEND VOLTAGE MAGNITUDES\n\n")

					f.write("ACTIVE_POWER_AT_SOURCE_BUS\n\n")
					for bus in instance.Buses:
						for t in instance.TimePeriods:
							for p in instance.Phases:
								if bus == HeadBus:
									if instance.ActivePowerAtSourceBus[p,bus, t].value != None:
										f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(round(instance.ActivePowerAtSourceBus[p, bus, t].value * instance.S_Base.value/3, 5))))
									else:
										f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(0)))
					f.write("\nEND_ACTIVE_POWER_AT_SOURCE_BUS\n\n")
					
					f.write("REACTIVE_POWER_AT_SOURCE_BUS\n\n")
					for bus in instance.Buses:
						for t in instance.TimePeriods:
							for p in instance.Phases:
								if bus == HeadBus:
									if instance.ReactivePowerAtSourceBus[p,bus, t].value != None:
										f.write('Phase: {} Bus: {} Interval: {} : {} kVAr\n'.format(str(p), str(bus), str(t), str(round(instance.ReactivePowerAtSourceBus[p, bus, t].value * instance.S_Base.value/3, 5))))
									else:
										f.write('Phase: {} Bus: {} Interval: {} : {} kVAr\n'.format(str(p), str(bus), str(t), str(0)))
					f.write("\nEND_REACTIVE_POWER_AT_SOURCE_BUS\n\n")

					f.write("Active Power at Each Node [Excludes Source Bus Injection]\n\n")
					for bus in instance.Buses:
						for t in instance.TimePeriods:
							for p in instance.Phases:
								f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(round(instance.P[p, bus, t].value * instance.S_Base.value/3, 5))))
					f.write("\nEND_ACTIVE_POWER_AT_EACH_NODE\n\n")

					f.write("Reactive Power at Each Node [Excludes Source Bus Injection]\n\n")
					for bus in instance.Buses:
						for t in instance.TimePeriods:
							for p in instance.Phases:
								f.write('Phase: {} Bus: {} Interval: {} : {} kVAr\n'.format(str(p), str(bus), str(t), str(round(instance.Q[p, bus, t].value * instance.S_Base.value/3, 5))))
					f.write("\nEND_REACTIVE_POWER_AT_EACH_NODE\n\n")
				
					f.write("LINE_ACTIVE_POWER_FLOWS\n\n")
					for l in sorted(instance.TransmissionLines):
						for t in instance.TimePeriods:
							for p in instance.Phases:
								f.write('Phase: {} Line Connecting: {} to {} Interval: {} : {} kW\n'.format(str(p), str(instance.BusFrom[l]), str(instance.BusTo[l]), str(t), str(round(instance.P_L[p, l, t].value  * instance.S_Base.value/3, 5))))
					f.write("\nEND_LINE_ACTIVE_POWER_FLOWS\n\n")

					f.write("LINE_REACTIVE_POWER_FLOWS\n\n")
					for l in sorted(instance.TransmissionLines):
						for t in instance.TimePeriods:
							for p in instance.Phases:
								f.write('Phase: {} Line Connecting: {} to {} Interval: {} : {} kVAr\n'.format(str(p), str(instance.BusFrom[l]), str(instance.BusTo[l]), str(t), str(round(instance.Q_L[p, l, t].value * instance.S_Base.value/3, 5))))
					f.write("\nEND_LINE_REACTIVE_POWER_FLOWS\n\n")


					f.write("DER_States\n\n")
					for d in instance.DER.data():
						phases = list(instance.DERPhases[d])
						alpha = float(instance.DERAlpha[d].value or 0.0)
						for t in instance.TimePeriods:
							p_total = (
								alpha * float(instance.DERP[d] * len(phases))
							)
							if hasattr(instance, "DERQ"):
								q_total = float(instance.DERQ[d].value or 0.0)*( instance.S_Base.value / 3) * len(phases)
							else:
								q_total =(
									p_total * 
									np.sqrt((1/(instance.PF.value)**2)-1)
								)

							apparent_power = np.hypot(p_total, q_total)

							if apparent_power <= 1e-8:
								pf_value = 0.0
								pf_type = "OFF"
							else:

								pf_value = abs(p_total / apparent_power)
								if abs(q_total) <= 1e-8:
									pf_type = "Unity"
								elif abs(p_total) <= 1e-8:	
									pf_type = "Reactive"
								elif p_total * q_total >0:
									pf_type = "Lagging"
								else:
									pf_type = "Leading"

							f.write(
								"\t{}: Interval: {} Phases: {} Alpha: {} Pi: {} "
								"P: {} kW Q: {} kVAr PF: {} {}\n".format(
									d,
									t,
									phases,
									round(alpha, 3),
									round(float(instance.DERPi[d]), 3),
									round(p_total, 3),
									round(q_total, 3),
									round(pf_value, 2),
									pf_type,
								)
							)
					f.write("\nEND_DER_States\n\n")


		elif (Status == 'infeasible'):
			with open(output.strip("'"), 'w') as f:
				f.write("THE OPF WAS RUN AT : ") 
				f.write(datetime_india.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
				f.write("\nSOLUTION_STATUS\n")
				f.write("infeasible \t")
				f.write("\nEND_SOLUTION_STATUS\n")

	return result, SolverOutcomes

if __name__ == "__main__":
	PreAlphaValues = {}
	SourceActivePower = {}
	SourceReactivePower = {}

	#####Bin I###################

	results, SolverOutcome = OPF(SOLVER, OnlyBids=True)
	instance = model._model

	for d in instance.DER:
		PreAlphaValues[d] = float(instance.DERAlpha[d].value)

	
# ##########Bin II ####################

	results, SolverOutcome = OPF(SOLVER, OnlyOffers=True)
	instance = model._model	
	FinalAlphaWPM = {
			d: 0.0 for d in DER_names
		}
	for d in instance.DER:
		PreAlphaValues[d] = float(instance.DERAlpha[d].value)

	for d in DER_names:
		if (PreAlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] < 0):
			if DER_pi[d] >= LMP + C_IDSO:
				FinalAlphaWPM[d] = PreAlphaValues[d]

	for d in DER_names:
		if (PreAlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] > 0):
			if DER_pi[d] <= LMP - C_IDSO:
				FinalAlphaWPM[d] = PreAlphaValues[d]

	prealpha_values_df = pd.DataFrame.from_dict(
			PreAlphaValues,
			orient="index",
			columns=["Alpha"],
		)
	prealpha_values_df.index.name = "DER"
	prealpha_values_df = prealpha_values_df.sort_index(key=natsort.natsort_keygen())
	prealpha_values_df.to_csv(
			output_dir / f"{args.TestCase}DERPreAlphaValues.csv"
		)

	qualified_bidvalue = sum(
			PreAlphaValues[d] * DER_P[d]
			for d in DER_names if PreAlphaValues.get(d, 0.0) > 1e-8 and DER_P_per_phase[d] < 0
		)

	qualified_bidcount = sum(
			1 for d in DER_names if PreAlphaValues.get(d, 0.0) > 1e-8 and DER_P_per_phase[d] < 0
	)

	qualified_offercount = sum(
			1 for d in DER_names if PreAlphaValues.get(d, 0.0) > 1e-8 and DER_P_per_phase[d] > 0
	)
	
	qualified_offervalue = sum(
		PreAlphaValues[d] * DER_P[d]
		for d in DER_names if PreAlphaValues.get(d, 0.0) > 1e-8 and DER_P_per_phase[d] > 0
	)

	print(f"\nTotal qualified bid value: {qualified_bidvalue:.2f} kW")
	print(f"\nTotal qualified bid count: {qualified_bidcount}")
	print(f"\nTotal qualified offer value: {qualified_offervalue:.2f} kW")
	print(f"\nTotal qualified offer count: {qualified_offercount}")
###########Robust Optimization with all clearing scenarios enforced ####################
	START_TIME = time.perf_counter()
	template_wrapper = OPF(
		SOLVER,
		BothBidsOffers=True,
		AdjustableDERQ=True,
		BuildOnly=True,
		CollectDuals=False,
	)
	template = template_wrapper._model

	qualified_bids = sorted(
		(
			d for d in template.DER
			if pyo.value(template.DERP[d]) < 0
		),
		key=lambda d: pyo.value(template.DERPi[d]),
		reverse=True,
	)
	qualified_offers = sorted(
		(
			d for d in template.DER
			if pyo.value(template.DERP[d]) > 0
		),
		key=lambda d: pyo.value(template.DERPi[d]),
	)

	IDSO_prices = {
		d: pyo.value(template.DERPi[d]) - C_IDSO
		for d in qualified_bids
	}
	IDSO_prices.update({
		d: pyo.value(template.DERPi[d]) + C_IDSO
		for d in qualified_offers
	})

	vertices = wpm_find_vertices(
		qualified_bids,
		qualified_offers,
		IDSO_prices,
	)

	robust_model = build_robust_model(
		template,
		vertices,
		model_objective_function,
	)

	print(
		f"\nOptimizing {len(robust_model.DER)} DER alphas "
		f"jointly across {len(vertices)} clearing scenarios."
	)

	robust_solver = pyo.SolverFactory(SOLVER)
	robust_solver.available(exception_flag=True)

	solve_result = robust_solver.solve(
		robust_model,
		tee=False,
		load_solutions=False,
	)

	termination = solve_result.solver.termination_condition
	if termination != pyo.TerminationCondition.optimal:
		raise RuntimeError(
			"Joint robust optimization did not reach optimality: "
			f"{termination}"
		)

	robust_model.solutions.load_from(solve_result)

	AlphaValues = {
		d: float(pyo.value(robust_model.DERAlpha[d]))
		for d in robust_model.DER
	}

	robust_results = [
		{
			"Vertex": s,
			"Mode": "JointRobustSolve",
			"Status": str(termination),
		}
		for s in robust_model.Scenarios
	]

	print("\nJoint robust optimization solved with all clearing scenarios enforced.")
	print(
		"Robust objective:",
		pyo.value(robust_model.GenerationObjective),
	)
	END_TIME = time.perf_counter()
	execution_time = END_TIME - START_TIME
	print(f"Robust Optimization time: {execution_time:.2f} seconds")

	FinalAlpha = AlphaValues.copy()

	FinalAlphaBids = {
		d: AlphaValues[d]
		for d in qualified_bids
		if AlphaValues[d] > 1e-8
	}
	FinalAlphaOffers = {
		d: AlphaValues[d]
		for d in qualified_offers
		if AlphaValues[d] > 1e-8
	}

	print("\nCERTIFIED BID ALPHAS")
	print(FinalAlphaBids)
	print("\nCERTIFIED OFFER ALPHAS")
	print(FinalAlphaOffers)

	alpha_values_df = pd.DataFrame.from_dict(
		AlphaValues,
		orient="index",
		columns=["Alpha"],
	)
	alpha_values_df.index.name = "DER"
	alpha_values_df = alpha_values_df.sort_index(key=natsort.natsort_keygen())
	alpha_values_df.to_csv(
		output_dir / f"{args.TestCase}DERRobustAlphaValues.csv"
	)

	pd.DataFrame(robust_results).to_csv(
		output_dir / "RobustVertexResults.csv",
		index=False,
	)

	pd.DataFrame([
		{
			"Vertex": s,
			"DER": d,
			"Activation": pyo.value(
				robust_model.Activation[s, d]
			),
		}
		for s in robust_model.Scenarios
		for d in robust_model.DER
	]).to_csv(
		output_dir / "RobustActivations.csv",
		index=False,
	)

	clearing_u = {d: 0.0 for d in robust_model.DER}

	for d in qualified_bids:
		clearing_u[d] = float(IDSO_prices[d] >= LMP)

	for d in qualified_offers:
		clearing_u[d] = float(IDSO_prices[d] <= LMP)

	qualified_bidvalue = sum(
		AlphaValues[d] * DER_P[d]
		for d in qualified_bids
	)

	qualified_offervalue = sum(
		AlphaValues[d] * DER_P[d]
		for d in qualified_offers
	)

	qualified_bidcount = sum(
		1 for d in qualified_bids
		if AlphaValues[d] > 1e-8
	)
	qualified_offercount = sum(
		1 for d in qualified_offers
		if AlphaValues[d] > 1e-8
	)

	print(f"\nTotal qualified bid value (ROBUST): {qualified_bidvalue:.2f} kW")
	print(f"\nTotal qualified bid count (ROBUST): {qualified_bidcount}")
	print(f"\nTotal qualified offer value (ROBUST): {qualified_offervalue:.2f} kW")
	print(f"\nTotal qualified offer count (ROBUST): {qualified_offercount}")

	# FinalAlphaWPM = alpha_at_vertex(
	# 	all_ders=list(robust_model.DER),
	# 	AlphaValues=AlphaValues,
	# 	u=clearing_u,
	# )

##################Getting WPM Results Before Moving Further ######################

	ClearedBidsOffers = [d for d in FinalAlphaWPM.keys() if FinalAlphaWPM[d] > 1e-8]
	print(f"\n\nCleared DERs at {LMP} ¢/kWh are {len(ClearedBidsOffers)}")
	results, SolverOutcome = OPF(SOLVER, FinalCase=True, LMP=LMP, alpha=FinalAlphaWPM)#, removed_bids=removed_bids, removed_offers=removed_offers)
	LMPs = [6, 10, 14]
	if str(SolverOutcome[1]) != 'optimal':
		print("\n\nFinal WPM case infeasible; Checking Violations.")
		if args.TestCase == "IEEE13TestCase":
			fig, ax = plt.subplots(nrows=1, ncols=len(LMPs), sharey=True, figsize=FIGSIZE_13TestCase, gridspec_kw={'wspace': 0.04})
			for i, l in enumerate(LMPs):
				FinalAlpha = {
						d: 0.0 for d in DER_names
					}
				
				for d in FinalAlpha.keys():
					if (PreAlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] < 0 and DER_pi[d] >= l + C_IDSO):
						FinalAlpha[d] = PreAlphaValues[d]
				
				for d in FinalAlpha.keys():
					if (PreAlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] > 0 and DER_pi[d] <= l - C_IDSO):
						FinalAlpha[d] = PreAlphaValues[d]
				
				results, SolverOutcome = OPF(
					SOLVER,
					FinalCase=True,
					LMP=l,
					alpha=FinalAlpha,
					PowerFlowMode = True)

				instance = model._model
				labels = list(instance.Buses)
				Voltage_A = {}
				Voltage_B = {}
				Voltage_C = {}
				Thermal_margin = {}
				LinePowerP = {}
				LinePowerQ = {}
			
				for b in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases: 
							if p in instance.BusPhase[b]:
								if p == 'A':
									Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
								if p=='B':
									Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
								if p=='C':
									Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
		
				df = pd.DataFrame({
						'Phase A': Voltage_A,
						'Phase B': Voltage_B,
						'Phase C': Voltage_C
					})
					
				df = df.transpose()
				df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
				df2 = df.transpose()
				df2.to_csv(csv_dir/f'{args.TestCase}FinalVoltageProfile{l}.csv', index=True)

				lower_limit = 0.95
				upper_limit = 1.05
		
				x_values = np.arange(len(df.columns))
				n = len(df.columns)
		
				markers = {
					'Phase A': 'o',
					'Phase B': 'x',
					'Phase C': 's'
				}

				for phase in df.index:
					ax[i].plot(
						x_values,
						df.loc[phase].to_numpy(dtype=float),
						marker=markers.get(phase, 'o'),
						markersize=3.2,
						linewidth=1.05,
						label=phase
					)
				ax[i].axhline(
					y=lower_limit,
					color='purple',
					linestyle='--',
					linewidth=1.2,
					label='Voltage limits'
				)
				ax[i].axhline(
					y=upper_limit,
					color='purple',
					linestyle='--',
					linewidth=1.2
				)
				
				# ax[i].set_xlabel(
				# 	'Buses',
				# 	fontsize=9,
				# 	fontweight='normal'
				# )
				ax[i].set_xticks(x_values)
				
				custom_labels = [
					str(labels[j])[3:] if (j-1) % 2 == 0 else "" for j in range(len(labels))
				]
		
				ax[i].set_xticklabels(custom_labels, fontsize=8)
				ax[i].tick_params(axis='both', labelsize=8)
		
				ax[i].set_xlim(-0.25, n-0.75)
		
				all_values = df.to_numpy(dtype=float)
				data_min = []
				data_max = []
				data_min.append(np.nanmin(all_values))
				data_max.append(np.nanmax(all_values))
		
				ax[i].grid(alpha=0.15, linewidth=0.7)
				style_phase_plot(ax[i])
				ax[i].text(
					0.04,
					0.04,
					f"LMP = {l} ¢/kWh",
					transform=ax[i].transAxes,
					ha="left",
					va="bottom",
					fontsize=6,
					bbox={
						"facecolor": "white",
						"edgecolor": "none",
						"alpha": 0.75,
						"pad": 0.3,
					},
				)

			# ax[0].set_ylim(
			# 					min(0.94, min(data_min) - 0.005),
			# 					max(1.06, max(data_max) + 0.005)
			# 				)

			y_min = min(a.get_ylim()[0] for a in ax)
			y_max = max(a.get_ylim()[1] for a in ax)

			plt.ylim(y_min, y_max)
			ax[0].set_xlabel(
							'Bus\n(a)',
							fontsize=9,
							fontweight='normal'
						)
			ax[1].set_xlabel(
							'Bus\n(b)',
							fontsize=9,
							fontweight='normal'
						)
			ax[2].set_xlabel(
							'Bus\n(c)',
							fontsize=9,
							fontweight='normal'
						)
			ax[0].set_ylabel(
							'Voltage Magnitude (p.u.)',
							fontsize=9,
							fontweight='normal'
						)
			handles, labels = [], []
	
			h, l = ax[0].get_legend_handles_labels()
			handles.extend(h)
			labels.extend(l)

			fig.legend(
				handles, labels,
				loc='upper center',
				ncol=4,
				fontsize=8,
				frameon=False
			)

			fig.savefig(
				figures_dir / "PDFs" / f"VoltageWPMCheckViolations{args.TestCase}.pdf",
				dpi=600, pad_inches=0.05, bbox_inches='tight'
			)
			fig.savefig(
				figures_dir / "PNGs" / f"VoltageWPMCheckViolations{args.TestCase}.png",
				dpi=600, pad_inches=0.05, bbox_inches='tight'
			)
			
		else:
			FinalAlpha = {
					d: 0.0 for d in DER_names
				}
			
			for d in FinalAlpha.keys():
				if (PreAlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] < 0 and DER_pi[d] >= LMP + C_IDSO):
					FinalAlpha[d] = PreAlphaValues[d]
			
			for d in FinalAlpha.keys():
				if (PreAlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] > 0 and DER_pi[d] <= LMP - C_IDSO):
					FinalAlpha[d] = PreAlphaValues[d]

			fig, ax = plt.subplots(figsize=FIGSIZE_123TestCase)

			if args.TestCase == "IEEE123StressTestCase":
				results, SolverOutcome = OPF(
					SOLVER,
					FinalCase=True,
					LMP=LMP,
					alpha=FinalAlpha,
					MinimizeNetworkViolations=True,)
			else:
				results, SolverOutcome = OPF(
					SOLVER,
					FinalCase=True,
					LMP=LMP,
					alpha=FinalAlpha,
					PowerFlowMode = True)

			instance = model._model
			labels = list(instance.Buses)
			Voltage_A = {}
			Voltage_B = {}
			Voltage_C = {}
			Thermal_margin = {}
			LinePowerP = {}
			LinePowerQ = {}
		
			for b in instance.Buses:
				for t in instance.TimePeriods:
					for p in instance.Phases: 
						if p in instance.BusPhase[b]:
							if p == 'A':
								Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
							if p=='B':
								Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
							if p=='C':
								Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

			df = pd.DataFrame({
					'Phase A': Voltage_A,
					'Phase B': Voltage_B,
					'Phase C': Voltage_C
				})
				
			df = df.transpose()
			df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
			df2 = df.transpose()
			df2.to_csv(csv_dir/f'{args.TestCase}FinalVoltageProfileViolations.csv', index=True)

			lower_limit = 0.95
			upper_limit = 1.05

			x_values = np.arange(len(df.columns))
			n = len(df.columns)

			markers = {
				'Phase A': 'o',
				'Phase B': 'x',
				'Phase C': 's'
			}

			for phase in df.index:
				ax.plot(
					x_values,
					df.loc[phase].to_numpy(dtype=float),
					marker=markers.get(phase, 'o'),
					markersize=3.2,
					linewidth=1.05,
					label=phase
				)

			ax.axhline(
				y=lower_limit,
				color='purple',
				linestyle='--',
				linewidth=1.2,
				label='Voltage limits'
			)

			ax.axhline(
				y=upper_limit,
				color='purple',
				linestyle='--',
				linewidth=1.2
			)

			ax.set_ylabel(
				'Voltage Magnitude (p.u.)',
				fontsize=9,
				fontweight='normal'
			)

			ax.set_xlabel(
				'Bus',
				fontsize=9,
				fontweight='normal'
			)

			ax.set_xticks(x_values)

			custom_labels = [
				str(labels[i])[3:] if (i-1) % 5 == 0 else ""  for i in range(len(labels))
			]

			ax.set_xticklabels(custom_labels, fontsize=8)
			ax.tick_params(axis='both', labelsize=8)

			ax.set_xlim(-1, n)

			all_values = df.to_numpy(dtype=float)
			data_min = np.nanmin(all_values)
			data_max = np.nanmax(all_values)

			ax.set_ylim(
				min(0.94, data_min - 0.005),
				max(1.06, data_max + 0.005)
			)

			ax.grid(alpha=0.15, linewidth=0.7)

			def add_violation_inset(indices, limit, violation_type):

				if len(indices) == 0:
					return

				x1 = max(0, int(np.min(indices)) - 2)
				x2 = min(n - 1, int(np.max(indices)) + 2)

				violation_centre = np.mean(indices)

				horizontal_location = (
					'right' if violation_centre < (n - 1) / 2 else 'left'
				)

				if violation_type == 'lower':
					inset_location = f'center'
				else:
					inset_location = f'lower {horizontal_location}'

				axins = inset_axes(
					ax,
					width="40%",
					height="40%",
					loc="center",
					bbox_to_anchor=(0.4, 0.50, 0.50, 0.50),
					bbox_transform=ax.transAxes,
					borderpad=0
				)

				for phase in df.index:
					axins.plot(
						x_values,
						df.loc[phase].to_numpy(dtype=float),
						marker=markers.get(phase, 'o'),
						markersize=2.7,
						linewidth=0.90,
						label=phase
					)

				axins.axhline(
					y=limit,
					color='purple',
					linestyle='--',
					linewidth=1.0
				)

				axins.set_xlim(x1, x2)


				local_values = df.iloc[:, x1:x2 + 1].to_numpy(dtype=float)

				if violation_type == 'lower':
					violating_values = local_values[local_values < limit]
					extreme_value = np.min(violating_values)

					distance = max(limit - extreme_value, 0.001)
					padding = max(0.0015, 0.18 * distance)

					axins.set_ylim(
						extreme_value - padding,
						limit + padding
					)

				else:
					violating_values = local_values[local_values > limit]
					extreme_value = np.max(violating_values)

					distance = max(extreme_value - limit, 0.001)
					padding = max(0.0015, 0.18 * distance)

					axins.set_ylim(
						limit - padding,
						extreme_value + padding
					)

				axins.set_xticks(np.arange(x1, x2 + 1))

				inset_labels = [
					str(labels[i])[3:] if (i - x1) % 2 == 0 else ""
					for i in range(x1, x2 + 1)
				]

				axins.set_xticklabels(
					inset_labels,
					fontsize=7,
					rotation=0
				)

				axins.tick_params(
					axis='y',
					labelsize=7,
					direction='in'
				)

				axins.tick_params(
					axis='x',
					direction='in'
				)

				# axins.set_title(
				# 	# title,
				# 	fontsize=8,
				# 	fontweight='normal',
				# 	fontstyle='italic',
				# 	pad=1.5
				# )

				axins.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
				axins.yaxis.set_major_locator(mticker.MaxNLocator(4))
				style_phase_plot(axins, inset=True, markevery=1)
				axins.patch.set_facecolor('white')
				axins.patch.set_alpha(1.0)

				if horizontal_location == 'right':
					connector_1, connector_2 = 2, 3
				else:
					connector_1, connector_2 = 1, 3

				mark_inset(
					ax,
					axins,
					loc1=connector_1,
					loc2=connector_2,
					fc="none",
					ec="0.55",
					linewidth=0.55
				)

			lower_violation_indices = np.where(
				np.any(all_values < lower_limit, axis=0)
			)[0]

			upper_violation_indices = np.where(
				np.any(all_values > upper_limit, axis=0)
			)[0]
			if args.TestCase == "IEEE123TestCase":
				add_violation_inset(
					lower_violation_indices,
					lower_limit,
					'lower'
				)

			# add_violation_inset(
			# 	upper_violation_indices,
			# 	upper_limit,
			# 	'upper'
			# )

			style_phase_plot(ax)

			ax.legend(
				loc='upper center',
				bbox_to_anchor=(0.5, 1.13),
				ncol=4,
				fontsize=8,
				frameon=False
			)

			fig.savefig(
				figures_dir /"PDFs" / f"VoltageWPMCheckViolations{args.TestCase}.pdf",
				bbox_inches='tight',
				pad_inches=0,
				dpi=600
			)
			fig.savefig(
				figures_dir /"PNGs" / f"VoltageWPMCheckViolations{args.TestCase}.png",
				bbox_inches='tight',
				pad_inches=0,
				dpi=600
			)
	
		print("Final WPM case violations saved, now checking if the final WPM case is feasible with adjustable DER reactive power.")
		if args.TestCase == "IEEE13TestCase":
			LMPs = [10, 14]
			fig, ax = plt.subplots(nrows=1, ncols=len(LMPs), sharey=True, figsize=FIGSIZE_13TestCaseCompact, gridspec_kw={'wspace': 0.04})
			for i, l in enumerate(LMPs):
				FinalAlpha = {
						d: 0.0 for d in DER_names
					}
				
				for d in FinalAlpha.keys():
					if (AlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] < 0 and DER_pi[d] >= l + C_IDSO):
						FinalAlpha[d] = AlphaValues[d]
				
				for d in FinalAlpha.keys():
					if (AlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] > 0 and DER_pi[d] <= l - C_IDSO):
						FinalAlpha[d] = AlphaValues[d]
					
				results, SolverOutcome = OPF(
					SOLVER,
					FinalCase=True,
					LMP=l,
					alpha=FinalAlpha,
					AdjustableDERQ=True,
					MinimizeDERQDeviation=True,)
					#removed_bids=removed_bids, removed_offers=removed_offers)

				instance = model._model
				labels = list(instance.Buses)
				Voltage_A = {}
				Voltage_B = {}
				Voltage_C = {}
				Thermal_margin = {}
				LinePowerP = {}
				LinePowerQ = {}
				powerfactor_setpoints = {}
				
				for d in instance.DER.data():
					if FinalAlpha.get(d, 0.0) <= 1e-8:
						continue
					phases = list(instance.DERPhases[d])
					for t in instance.TimePeriods:
						p_total = (
							FinalAlpha[d] * float(instance.DERP[d] * len(phases))
						)
						q_total = float(instance.DERQ[d].value or 0.0)*( instance.S_Base.value / 3) * len(phases)
						
						apparent_power = np.hypot(p_total, q_total)
		
						if apparent_power <= 1e-8:
							continue
						
						pf_value = abs(p_total/apparent_power)
						if abs(q_total) < 1e-8:
							signed_pf = pf_value
						elif abs(p_total) < 1e-8:
							signed_pf = pf_value
						elif p_total * q_total > 0:
							signed_pf = pf_value
						else:
							signed_pf = -pf_value
						powerfactor_setpoints[d] = signed_pf

				df_pf = pd.DataFrame.from_dict(
						powerfactor_setpoints,
						orient='index',
						columns=['Power Factor Setpoints']
					)

				df_pf.to_csv(csv_dir/f'{args.TestCase}FinalPowerFactorSetpoints{l}DERQ.csv', index=True)
			
				for b in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases: 
							if p in instance.BusPhase[b]:
								if p == 'A':
									Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
								if p=='B':
									Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
								if p=='C':
									Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
		
				df = pd.DataFrame({
						'Phase A': Voltage_A,
						'Phase B': Voltage_B,
						'Phase C': Voltage_C
					})
					
				df = df.transpose()
				df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
				df2 = df.transpose()
				df2.to_csv(csv_dir/f'{args.TestCase}FinalVoltageProfile{l}DERQ.csv', index=True)

				lower_limit = 0.95
				upper_limit = 1.05
		
				x_values = np.arange(len(df.columns))
				n = len(df.columns)
		
				markers = {
					'Phase A': 'o',
					'Phase B': 'x',
					'Phase C': 's'
				}

				for phase in df.index:
					ax[i].plot(
						x_values,
						df.loc[phase].to_numpy(dtype=float),
						marker=markers.get(phase, 'o'),
						markersize=3.2,
						linewidth=1.05,
						label=phase
					)
				ax[i].axhline(
					y=lower_limit,
					color='purple',
					linestyle='--',
					linewidth=1.2,
					label='Voltage limits'
				)
				ax[i].axhline(
					y=upper_limit,
					color='purple',
					linestyle='--',
					linewidth=1.2
				)
				
				# ax[i].set_xlabel(
				# 	'Buses',
				# 	fontsize=9,
				# 	fontweight='normal'
				# )
				ax[i].set_xticks(x_values)
				
				custom_labels = [
					str(labels[j])[3:] if (j-1) % 2 == 0 else "" for j in range(len(labels))
				]
		
				ax[i].set_xticklabels(custom_labels, fontsize=8)
				ax[i].tick_params(axis='both', labelsize=8)
		
				ax[i].set_xlim(-0.25, n-0.75)
		
				all_values = df.to_numpy(dtype=float)
				data_min = []
				data_max = []
				data_min.append(np.nanmin(all_values))
				data_max.append(np.nanmax(all_values))
		
				ax[i].grid(alpha=0.15, linewidth=0.7)
				style_phase_plot(ax[i])
				ax[i].text(
					0.04,
					0.04,
					f"LMP = {l} ¢/kWh",
					transform=ax[i].transAxes,
					ha="left",
					va="bottom",
					fontsize=6,
					bbox={
						"facecolor": "white",
						"edgecolor": "none",
						"alpha": 0.75,
						"pad": 0.3,
					},
				)

			y_min = min(a.get_ylim()[0] for a in ax)
			y_max = max(a.get_ylim()[1] for a in ax)

			
			ax[0].set_ylim(y_min, y_max)

			ax[0].set_xlabel(
							'Bus\n(a)',
							fontsize=9,
							fontweight='normal'
						)
			ax[1].set_xlabel(
							'Bus\n(b)',
							fontsize=9,
							fontweight='normal'
						)
			ax[0].set_ylabel(
							'Voltage Magnitude (p.u.)',
							fontsize=9,
							fontweight='normal'
						)
			handles, labels = [], []

			h, l = ax[0].get_legend_handles_labels()
			handles.extend(h)
			labels.extend(l)

			fig.legend(
				handles, labels,
				loc='upper center',
				ncol=4,
				fontsize=8,
				frameon=False
			)

			fig.savefig(
				figures_dir / "PDFs" / f"VoltageWPMCheckViolations{args.TestCase}DERQActivated.pdf",
				dpi=600, pad_inches=0.1, bbox_inches='tight')
			fig.savefig(
				figures_dir / "PNGs" / f"VoltageWPMCheckViolations{args.TestCase}DERQActivated.png",
				dpi=600, pad_inches=0.1, bbox_inches='tight')

		else:
			FinalAlpha = {
					d: 0.0 for d in DER_names
				}
			
			for d in FinalAlpha.keys():
				if (AlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] < 0 and DER_pi[d] >= LMP + C_IDSO):
					FinalAlpha[d] = AlphaValues[d]
			
			for d in FinalAlpha.keys():
				if (AlphaValues.get(d, 0.0) >1e-8 and DER_P_per_phase[d] > 0 and DER_pi[d] <= LMP - C_IDSO):
					FinalAlpha[d] = AlphaValues[d]

			results, SolverOutcome = OPF(
				SOLVER,
				FinalCase=True,
				LMP=LMP,
				alpha=FinalAlpha,
				AdjustableDERQ=True,
				MinimizeDERQDeviation=True,)
				#removed_bids=removed_bids, removed_offers=removed_offers)

			instance = model._model
			labels = list(instance.Buses)
			Voltage_A = {}
			Voltage_B = {}
			Voltage_C = {}
			Thermal_margin = {}
			LinePowerP = {}
			LinePowerQ = {}

			for b in instance.Buses:
				for t in instance.TimePeriods:
					for p in instance.Phases: 
						if p in instance.BusPhase[b]:
							if p == 'A':
								Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
							if p=='B':
								Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
							if p=='C':
								Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

			for p in instance.Phases:
				for t in instance.TimePeriods:
					SourceActivePower[p] = instance.ActivePowerAtSourceBus[p,instance.HeadBus.at(1),t].value
					SourceReactivePower[p] = instance.ReactivePowerAtSourceBus[p,instance.HeadBus.at(1),t].value

			for t in instance.TimePeriods:	
				for l in sorted(instance.TransmissionLines):
					I= []
					sum_power_P = 0
					sum_power_Q = 0
					for p in instance.Phases:
						I.append(np.sqrt(instance.P_L[p,l,t].value**2 + instance.Q_L[p,l,t].value**2)/instance.V[p, instance.BusFrom[l], t].value)
						sum_power_P += instance.P_L[p,l,t].value*instance.S_Base.value/3
						sum_power_Q += instance.Q_L[p,l,t].value*instance.S_Base.value/3
					Thermal_margin[(instance.BusFrom[l], instance.BusTo[l])] = (1 - max(I)/(instance.ThermalCap[l]/instance.I_Base))*100
					LinePowerP[(instance.BusFrom[l], instance.BusTo[l])] = sum_power_P
					LinePowerQ[(instance.BusFrom[l], instance.BusTo[l])] = sum_power_Q
					# 	sum_power_P += instance.P_L[p,l,t].value
					# 	sum_power_Q += instance.Q_L[p,l,t].value
					# LinePowerMargin[(instance.BusFrom[l], instance.BusTo[l])] = round(np.sqrt(sum_power_P**2 + sum_power_Q**2),4)
						# if p == 'A':
						# 	LineActivePower_A[(instance.BusFrom[l], instance.BusTo[l])] = instance.P_L[p,l,t].value
						# 	LineReactivePower_A[(instance.BusFrom[l], instance.BusTo[l])] = instance.Q_L[p,l,t].value
						# if p == 'B':
						# 	LineActivePower_B[(instance.BusFrom[l], instance.BusTo[l])] = instance.P_L[p,l,t].value
						# 	LineReactivePower_B[(instance.BusFrom[l], instance.BusTo[l])] = instance.Q_L[p,l,t].value
						# if p == 'C':
						# 	LineActivePower_C[(instance.BusFrom[l], instance.BusTo[l])] = instance.P_L[p,l,t].value
						# 	LineReactivePower_C[(instance.BusFrom[l], instance.BusTo[l])] = instance.Q_L[p,l,t].value
			
			df_ThermalMargin = pd.DataFrame({
			'Margin (%)': Thermal_margin
			})
			df_ThermalMargin = df_ThermalMargin.sort_values(by='Margin (%)', ascending=True)
			df_ThermalMargin.to_csv(csv_dir/f'{args.TestCase}FinalLineThermalMargin.csv', index=True)
			
			df_LinePower = pd.DataFrame({
			'Active Power (kW)': LinePowerP,
			'Reactive Power (kVAr)': LinePowerQ
			})
			df_LinePower.to_csv(csv_dir/f'{args.TestCase}FinalLinePowerFlow.csv', index=True)

			df = pd.DataFrame({
				'Phase A': Voltage_A,
				'Phase B': Voltage_B,
				'Phase C': Voltage_C
			})
			
			df = df.transpose()
			df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
			df2 = df.transpose()
			df2.to_csv(csv_dir/f'{args.TestCase}FinalVoltageProfile.csv', index=True)

			fig, ax = plt.subplots(figsize=FIGSIZE_123TestCase)
			lower_limit = 0.95
			upper_limit = 1.05

			x_values = np.arange(len(df.columns))
			n = len(df.columns)

			markers = {
				'Phase A': 'o',
				'Phase B': 'x',
				'Phase C': 's'
			}

			for phase in df.index:
				ax.plot(
					x_values,
					df.loc[phase].to_numpy(dtype=float),
					marker=markers.get(phase, 'o'),
					markersize=3.2,
					linewidth=1.05,
					label=phase
				)

			ax.axhline(
				y=lower_limit,
				color='purple',
				linestyle='--',
				linewidth=1.2,
				label='Voltage limits'
			)

			ax.axhline(
				y=upper_limit,
				color='purple',
				linestyle='--',
				linewidth=1.2
			)

			ax.set_ylabel(
				'Voltage Magnitude (p.u.)',
				fontsize=9,
				fontweight='normal'
			)

			ax.set_xlabel(
				'Bus',
				fontsize=9,
				fontweight='normal'
			)

			ax.set_xticks(x_values)

			custom_labels = [
				str(labels[i])[3:] if (i-1) % 5 == 0 else ""  for i in range(len(labels))
			]

			ax.set_xticklabels(custom_labels, fontsize=8)
			ax.tick_params(axis='both', labelsize=8)

			ax.set_xlim(-1, n)

			all_values = df.to_numpy(dtype=float)
			data_min = np.nanmin(all_values)
			data_max = np.nanmax(all_values)

			ax.set_ylim(
				min(0.94, data_min - 0.005),
				max(1.06, data_max + 0.005)
			)

			ax.grid(alpha=0.15, linewidth=0.7)

			style_phase_plot(ax)
			
			ax.legend(
				loc='upper center',
				bbox_to_anchor=(0.5, 1.13),
				ncol=4,
				fontsize=8,
				frameon=False
			)
			
			fig.savefig(
				figures_dir/"PDFs" / f"VoltageWPMCheckViolations{args.TestCase}DERQ.pdf",
				bbox_inches='tight',
				pad_inches=0,
				dpi=600
			)
			fig.savefig(
				figures_dir/"PNGs" / f"VoltageWPMCheckViolations{args.TestCase}DERQ.png",
				bbox_inches='tight',
				pad_inches=0,
				dpi=600
			)
