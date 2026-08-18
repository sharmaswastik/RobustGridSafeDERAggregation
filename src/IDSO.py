import pickle
import pandas as pd
from pathlib import Path
import random
import numpy as np
from scipy.stats import truncnorm
import itertools
import re
import matplotlib.pyplot as plt
import argparse
from model.utils import correct_names, format_phases
plt.rc('font', family='serif', serif=['Times New Roman'])
plt.rc('legend', fontsize=12)

import plotly.express as px


def load_der_scenario(scenario_path, bus_data):
    
    scenario_df = pd.read_csv(scenario_path)
    scenario_df = correct_names(scenario_df, "Bus Location", "Bus")
    scenario_df["Phase Connection"] = scenario_df["Phase Connection"].apply(format_phases)

    required_columns = {
        "DER Name",
        "Bus Location",
        "Phase Connection",
        "Quantity (kW)",
    }

    scenario_df["DER Name"] = scenario_df["DER Name"].astype(str).str.strip()
    valid_buses = set(bus_data["Bus_Name"])
    bus_phases = {
        row["Bus_Name"]: set(str(row["Phases"]).upper())
        for _, row in bus_data.iterrows()
    }
    bus_order = bus_data["Bus_Name"].tolist()
    der_by_bus = {}
    quantities = []

    for row_number, row in scenario_df.iterrows():
        der_name = row["DER Name"]
        bus_name = row["Bus Location"]
        phases = row["Phase Connection"]
        quantity = float(row["Quantity"])
        price = float(row["Price"])
    
        der_type = "Bid" if quantity < 0 else "Offer"

        if "Type" in scenario_df.columns and pd.notna(row["Type"]):
            supplied_type = str(row["Type"]).strip().lower()
            if supplied_type not in {"bid", "offer"}:
                raise ValueError(
                    f"Row {row_number + 2}: Type must be Bid or Offer."
                )
            if supplied_type != der_type.lower():
                raise ValueError(
                    f"Row {row_number + 2}: {der_name} has Type={row['Type']} "
                    f"but Quantity (kW)={quantity}; bids must be negative and offers positive."
                )

        der_by_bus.setdefault(bus_name, {})[f'{der_name}_{phases}'] = [
            np.array([price], dtype=float),
            np.array([quantity], dtype=float),
        ]
        quantities.append(quantity)

    der_nodes = [bus_name for bus_name in bus_order if bus_name in der_by_bus]
    n_ders = {
        bus_name: len(der_by_bus.get(bus_name, {}))
        for bus_name in bus_order
    }
    return der_by_bus, der_nodes, n_ders


argparser = argparse.ArgumentParser(description="Simulation Parameters")
argparser.add_argument('--TestCase', type=str, default='IEEE123TestCase', help='Test case folder, e.g. 4BusTestCase, IEEE13TestCase, or IEEE123TestCase')
argparser.add_argument('--seed', type=int, default=5, help='Random seed for reproducibility')
argparser.add_argument('--LMP', type=float, default=10, help='LMP value in cents per kWh')
argparser.add_argument('--M', type=float, default=1000, help='Big M value')
argparser.add_argument('--pf', type=float, default=1.0, help='Power factor')
argparser.add_argument('--C0', type=float, default=2.5, help='IDSO marginal cost in cents per kWh')
args = argparser.parse_args()

np.random.seed(args.seed) #seed for paper = 85
random.seed(args.seed) #seed for paper = 85

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
figures_dir = parent_dir / "Figures"
output_dir = parent_dir / "OutputFiles"
csv_dir = output_dir /"CSVFiles"


file_csv = "OPF_data/"+args.TestCase+"/bus.csv"

bus_data = pd.read_csv(file_csv)   
bus_data = bus_data[~bus_data['bus_i'].isin([0])]
bus_data['Bus_Name'] = bus_data['bus_i'].apply(lambda x: f'Bus{x}')


def get_possible_phases(phases_str):

    phases = list(phases_str)
    possible_combinations = []
    for i in range(1, len(phases) + 1):
        combinations = list(itertools.combinations(phases, i))
        for combo in combinations:
            possible_combinations.append(''.join(combo))
    return possible_combinations


bus_data['Possible_DER_Phases'] = bus_data['Phases'].apply(get_possible_phases)

der_scenario_path = Path(file_csv).with_name("DERScenario.csv")
use_der_scenario = der_scenario_path.is_file()

if use_der_scenario:
    total_DERs = 0
elif "4BusTestCase" in file_csv:
    total_DERs = 4
else:
    total_DERs = 450

bus_names = bus_data['Bus_Name'].tolist()

N_DERs = {bus_name: 0 for bus_name in bus_names}

if use_der_scenario:
    pass
elif "4BusTestCase" in file_csv:
    N_DERs["Bus1"] = 4
    N_DERs["Bus2"] = 1
    N_DERs["Bus3"] = 1
else:
    for _ in range(total_DERs):
        bus_name = random.choice(bus_names)
        N_DERs[bus_name] += 1

DER_Nodes = [bus_name for bus_name, n_ders in N_DERs.items() if n_ders > 0]

DER = {}

if use_der_scenario:
    DER, DER_Nodes, N_DERs = load_der_scenario(
        der_scenario_path,
        bus_data,
    )
    print(
        f"Loaded {sum(N_DERs.values())} deterministic DERs from "
        f"{der_scenario_path}."
    )

elif "4BusTestCase" in file_csv:

    der_plan = [
        {"pi": 20, "P":-501},
        {"pi": 8.5, "P":250},
        {"pi": 15, "P":-250},
        {"pi": 10, "P":500},
        {"pi": 10, "P":-500},
        {"pi": 5, "P":500}
    ]
    k = 1
    idx = 0
    for i in DER_Nodes:
        pi_p = {}
        bus_row = bus_data[bus_data['Bus_Name'] == i].iloc[0]
        possible_phases = bus_row['Possible_DER_Phases']

        for j in range(N_DERs[i]):
            pi = np.array(der_plan[idx]["pi"])
            P = np.array(der_plan[idx]["P"])
            c = "A"
            pi_p[f'DER_{k}_{c}'] = [pi, P]
            k += 1
            idx += 1
        DER[i] = pi_p

else:
    a_trunc = 5
    b_trunc = 45
    loc = 20
    scale = 10
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale

    a_pi = 1
    b_pi = 25
    mu = 15
    sigma = 5
    a_t, b_t = (a_pi-mu)/sigma, (b_pi-mu)/sigma

    k = 1
    for i in DER_Nodes:
        pi_p = {}
        bus_row = bus_data[bus_data['Bus_Name'] == i].iloc[0]
        possible_phases = bus_row['Possible_DER_Phases']

        for j in range(N_DERs[i]):
            # is_bid = random.choice([True, False])
            
            pi_array = truncnorm.rvs(a=a_t, b=b_t, loc=mu, scale=sigma, size=1, random_state=None)
            pi = pi_array  # Extract scalar from array
            
            is_bid = random.choices([True, False], weights=[0.5, 0.5], k=1)[0]
            if is_bid:
                    P = -truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=1, random_state=None)

            else:
                    P = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=1, random_state=None)

            c = random.choice(possible_phases)
            pi_p[f'DER_{k}_{c}'] = [pi, P]
            k += 1
        DER[i] = pi_p

DER_names = []
DER_phases = {}
DER_pi = {}
DER_P = {}
DER_bids_names = []
DER_bids_phases = {}
DER_pi_bids = {}
DER_P_bids = {}
DER_offers_names = []
DER_offers_phases = {}
DER_pi_offers = {}
DER_P_offers = {}
DER_bids_at_bus = {}
DER_offers_at_bus = {}
DER_at_bus = {}
for i in DER_Nodes:
    DER_list = []
    DER_offers_list = []
    DER_bids_list = []

    for key, value in DER[i].items():
        name, phase = key.rsplit('_', 1)
        DER_names.append(name)
        DER_phases[name] = list(phase)
        pi, P = value
        if args.TestCase == "IEEE123StressTestCase":
            pi = pi.item()
        else:
            pi = round(pi.item(), 0)
        P = round(P.item(), 0)  
        DER_pi[name] = pi
        DER_P[name] = P
        
        if P >= 0:
            DER_pi_offers[name] = pi
            DER_P_offers[name] = P
            DER_offers_names.append(name)
            DER_offers_phases[name] = list(phase)
            DER_offers_list.append(name)
        else:
            DER_pi_bids[name] = pi
            DER_P_bids[name] = P
            DER_bids_names.append(name)
            DER_bids_phases[name] = list(phase)
            DER_bids_list.append(name)

        DER_list.append(name)
    DER_at_bus[i] = DER_list
    if DER_bids_list:
        DER_bids_at_bus[i] = DER_bids_list
    if DER_offers_list:
        DER_offers_at_bus[i] = DER_offers_list

count_bids = sum(1 for value in DER_P.values() if value <= 0)
count_offers = sum(1 for value in DER_P.values() if value > 0)
MaxP = max(DER_P_offers.values(), key=abs)
DER_num_phases = {der: len(phases) for der, phases in DER_phases.items()}
DER_P_per_phase = {}
DER_P_per_phase_bids = {}
DER_P_per_phase_offers = {}

for der in DER_names:
    num_phases = DER_num_phases[der]
    P_per_phase = DER_P[der] / num_phases
    DER_P_per_phase[der] = P_per_phase
    if DER_P[der] <= 0:
        DER_P_per_phase_bids[der] = P_per_phase
    else:
        DER_P_per_phase_offers[der] = P_per_phase

DERP_dict = {}
DERP_offers_dict = {}
DERP_bids_dict = {}

Phases = {'A', 'B', 'C'}

for der in DER_names:
    for phase in Phases:
        if phase in DER_phases[der]:
            DERP_dict[(phase, der)] = DER_P_per_phase[der]
            if DER_P[der] <= 0:
                DERP_bids_dict[(phase, der)] = DER_P_per_phase_bids[der]
            else:
                DERP_offers_dict[(phase, der)] = DER_P_per_phase_offers[der]
        else:
            DERP_dict[(phase, der)] = 0.0
            if DER_P[der] <= 0:
                DERP_bids_dict[(phase, der)] = 0.0
            else:
                DERP_offers_dict[(phase, der)] = 0.0

bid_names = []
bid_quantities = []
bid_prices = []

for der in DER_pi_bids:
    name = der[:3] + ' ' + der[4:]
    quantity = DER_P_bids[der]
    price = DER_pi_bids[der]
    bid_names.append(name)
    bid_quantities.append(quantity)
    bid_prices.append(price)


offer_names = []
offer_quantities = []
offer_prices = []

data_list = []


for bus_name, DER_list in DER_at_bus.items():
    for der_name in DER_list:
        bus_location = bus_name
        if der_name in DER_pi_bids:
            bid_price = DER_pi_bids[der_name]
            offer_price = ''  
            quantity = DER_P_bids[der_name]
        elif der_name in DER_pi_offers:
            bid_price = ''  
            offer_price = DER_pi_offers[der_name]
            quantity = DER_P_offers[der_name]
        else:
            bid_price = ''
            offer_price = ''
            quantity = 0
        data_list.append({
            'DER Name': der_name,
            'Bus Location': bus_location,
            'Phase Connection': DER_phases[der_name],
            'Bid Price (¢/kWh)': bid_price,
            'Offer Price (¢/kWh)': offer_price,
            'Quantity (kW)': quantity
        })


df = pd.DataFrame(data_list)

df.to_csv(csv_dir/'DERBidData.csv', index=False)


for der in DER_pi_offers:
    name = der[:3] + ' ' + der[4:]
    quantity = abs(DER_P_offers[der])
    price = abs(DER_pi_offers[der])  # Convert to positive for plotting
    offer_names.append(name)
    offer_quantities.append(quantity)
    offer_prices.append(price)


bid_data = list(zip(bid_prices, bid_quantities, bid_names))
bid_data.sort(key=lambda x: x[0], reverse=True)  # Highest price at the top

sorted_bid_prices, sorted_bid_quantities, sorted_bid_names = zip(*bid_data)


offer_data = list(zip(offer_prices, offer_quantities, offer_names))
offer_data.sort(key=lambda x: x[0]) 

sorted_offer_prices, sorted_offer_quantities, sorted_offer_names = zip(*offer_data)


DER_bus_mapping = {}

for bus_name in DER:
    bus_number = int(bus_name.replace('Bus', ''))
    for key in DER[bus_name]:
        # Extract the DER name without the phase
        name_phase = key.rsplit('_', 1)
        der_name = name_phase[0]
        DER_bus_mapping[der_name] = bus_number
bid_names = list(DER_pi_bids.keys())
bid_quantities = [abs(DER_P_bids[name]) for name in bid_names]
bid_prices = [DER_pi_bids[name] for name in bid_names]
bid_bus_numbers = [DER_bus_mapping.get(name, None) for name in bid_names]

bid_data = pd.DataFrame({
    'DER': bid_names,
    'Quantity': bid_quantities,
    'Price': bid_prices,
    'Type': 'Bid',
    'Bus_Number': bid_bus_numbers
})


offer_names = list(DER_pi_offers.keys())
offer_quantities = [DER_P_offers[name] for name in offer_names]
offer_prices = [abs(DER_pi_offers[name]) for name in offer_names]  # Convert to positive values
offer_bus_numbers = [DER_bus_mapping.get(name, None) for name in offer_names]

offer_data = pd.DataFrame({
    'DER': offer_names,
    'Quantity': offer_quantities,
    'Price': offer_prices,
    'Type': 'Offer',
    'Bus_Number': offer_bus_numbers
})


combined_data = pd.concat([bid_data, offer_data], ignore_index=True)
combined_data['Bus_Number'] = combined_data['Bus_Number'].astype(str)
unique_bus_numbers = sorted(combined_data['Bus_Number'].unique(), key=int)

bus_number_to_pos = {bus_number: pos for pos, bus_number in enumerate(unique_bus_numbers)}
combined_data['Bus_Pos'] = combined_data['Bus_Number'].map(bus_number_to_pos)

num_buses = len(unique_bus_numbers)
fig_width = max(12, num_buses * 0.2)
fig = plt.figure(figsize=(fig_width, 5))


max_quantity = combined_data['Quantity'].max()
bubble_sizes = (combined_data['Quantity'] / max_quantity) * 750  

bid_subset = combined_data[combined_data['Type'] == 'Bid']
plt.scatter(
    bid_subset['Bus_Pos'],
    bid_subset['Price'],
    s=bubble_sizes[bid_subset.index],
    c='green',
    alpha=0.5,
    label='Bids'
)

offer_subset = combined_data[combined_data['Type'] == 'Offer']
plt.scatter(
    offer_subset['Bus_Pos'],
    offer_subset['Price'],
    s=bubble_sizes[offer_subset.index],
    c='red',
    alpha=0.5,
    label='Offers'
)

plt.xticks(
    ticks=range(num_buses),
    labels=unique_bus_numbers,
    rotation=90
)
# plt.axhline(y=8, color='black', linestyle='--', label='Observed LMP')
plt.margins(x=0.01)

plt.xlabel('Bus Number', fontsize=18, fontweight='bold')
plt.ylabel('Price (¢/kWh)',  fontsize=18, fontweight='bold')
plt.tick_params(axis='both', labelsize=14)
plt.legend()
plt.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(figures_dir / "BidOffers.pdf", bbox_inches='tight')

with open(figures_dir/"BidOfferPlot.pkl", "wb") as f:
    pickle.dump(fig, f)
