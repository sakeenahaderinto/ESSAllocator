import streamlit as st
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from math import pi
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ESS Allocation Optimizer",
    page_icon="🔋",
    layout="wide"
)

# ============================================================
# SCENARIO DATA
# ============================================================
DEMAND_SCENARIOS = {
    'Typical Weekday': {
        't1': 0.684511335492475, 't2': 0.644122690036197, 't3': 0.61306915602972,
        't4': 0.599733282530006, 't5': 0.588874071251667, 't6': 0.5980186702229,
        't7': 0.5980186702229, 't8': 0.651743189178891, 't9': 0.706039245570585,
        't10': 0.787007048961707, 't11': 0.839016955610593, 't12': 0.852733854067441,
        't13': 0.870642027052772, 't14': 0.834254143646409, 't15': 0.816536483139646,
        't16': 0.819394170318156, 't17': 0.874071251666984, 't18': 1.0,
        't19': 0.983615926843208, 't20': 0.936368832158506, 't21': 0.887597637645266,
        't22': 0.809297008954087, 't23': 0.74585635359116, 't24': 0.733473042484283
    },
    'Summer Peak': {
        't1': 0.620, 't2': 0.580, 't3': 0.560, 't4': 0.550, 't5': 0.555, 't6': 0.570,
        't7': 0.620, 't8': 0.680, 't9': 0.740, 't10': 0.810, 't11': 0.870, 't12': 0.920,
        't13': 0.970, 't14': 1.0, 't15': 0.995, 't16': 0.980, 't17': 0.940, 't18': 0.910,
        't19': 0.870, 't20': 0.820, 't21': 0.760, 't22': 0.710, 't23': 0.670, 't24': 0.640
    },
    'Winter Peak': {
        't1': 0.780, 't2': 0.750, 't3': 0.730, 't4': 0.720, 't5': 0.740, 't6': 0.820,
        't7': 0.910, 't8': 0.950, 't9': 0.920, 't10': 0.880, 't11': 0.870, 't12': 0.860,
        't13': 0.850, 't14': 0.840, 't15': 0.860, 't16': 0.910, 't17': 0.970, 't18': 1.0,
        't19': 0.990, 't20': 0.970, 't21': 0.930, 't22': 0.880, 't23': 0.830, 't24': 0.800
    },
    'Weekend Low': {
        't1': 0.550, 't2': 0.520, 't3': 0.500, 't4': 0.490, 't5': 0.495, 't6': 0.510,
        't7': 0.530, 't8': 0.560, 't9': 0.610, 't10': 0.670, 't11': 0.720, 't12': 0.760,
        't13': 0.780, 't14': 0.790, 't15': 0.800, 't16': 0.810, 't17': 0.830, 't18': 0.870,
        't19': 0.910, 't20': 1.0, 't21': 0.940, 't22': 0.850, 't23': 0.740, 't24': 0.650
    }
}

WIND_SCENARIOS = {
    'Typical Variable': {
        't1': 0.0786666666666667, 't2': 0.0866666666666667, 't3': 0.117333333333333,
        't4': 0.258666666666667, 't5': 0.361333333333333, 't6': 0.566666666666667,
        't7': 0.650666666666667, 't8': 0.566666666666667, 't9': 0.484, 't10': 0.548,
        't11': 0.757333333333333, 't12': 0.710666666666667, 't13': 0.870666666666667,
        't14': 0.932, 't15': 0.966666666666667, 't16': 1.0, 't17': 0.869333333333333,
        't18': 0.665333333333333, 't19': 0.656, 't20': 0.561333333333333,
        't21': 0.565333333333333, 't22': 0.556, 't23': 0.724, 't24': 0.84
    },
    'Calm Day': {
        't1': 0.20, 't2': 0.18, 't3': 0.17, 't4': 0.16, 't5': 0.18, 't6': 0.22,
        't7': 0.25, 't8': 0.28, 't9': 0.31, 't10': 0.34, 't11': 0.36, 't12': 0.35,
        't13': 0.34, 't14': 0.33, 't15': 0.31, 't16': 0.29, 't17': 0.27, 't18': 0.25,
        't19': 0.23, 't20': 0.22, 't21': 0.21, 't22': 0.20, 't23': 0.19, 't24': 0.20
    },
    'Windy Day': {
        't1': 0.82, 't2': 0.85, 't3': 0.88, 't4': 0.90, 't5': 0.87, 't6': 0.83,
        't7': 0.79, 't8': 0.76, 't9': 0.78, 't10': 0.82, 't11': 0.85, 't12': 0.88,
        't13': 0.91, 't14': 0.94, 't15': 0.96, 't16': 1.0, 't17': 0.97, 't18': 0.93,
        't19': 0.89, 't20': 0.86, 't21': 0.83, 't22': 0.80, 't23': 0.78, 't24': 0.81
    },
    'Night Wind': {
        't1': 0.85, 't2': 0.88, 't3': 0.90, 't4': 0.87, 't5': 0.82, 't6': 0.75,
        't7': 0.65, 't8': 0.55, 't9': 0.45, 't10': 0.38, 't11': 0.32, 't12': 0.28,
        't13': 0.26, 't14': 0.25, 't15': 0.27, 't16': 0.30, 't17': 0.35, 't18': 0.42,
        't19': 0.52, 't20': 0.63, 't21': 0.72, 't22': 0.78, 't23': 0.82, 't24': 0.84
    }
}

# ============================================================
# CONSTANTS & NETWORK DATA
# ============================================================
Sbase = 100      # Power Base (100 MVA)
VOLL = 10000     # Value of lost load (£/MWh)
VOLW = 50        # Value of wind curtailment (£/MWh)
eta_c = 0.95     # Charging efficiency
eta_d = 0.9      # Discharging efficiency
NBUSMAX = 5      # Max ESS units per bus

# Time periods
t_hours = [f't{i}' for i in range(1, 25)]

# Generator data
GD = {
    ('g1', 'Pmax'): 400, ('g1', 'Pmin'): 100, ('g1', 'b'): 5.47, ('g1', 'RU'): 47, ('g1', 'RD'): 47,
    ('g2', 'Pmax'): 400, ('g2', 'Pmin'): 100, ('g2', 'b'): 5.47, ('g2', 'RU'): 47, ('g2', 'RD'): 47,
    ('g3', 'Pmax'): 152, ('g3', 'Pmin'): 30.4, ('g3', 'b'): 13.32, ('g3', 'RU'): 14, ('g3', 'RD'): 14,
    ('g4', 'Pmax'): 152, ('g4', 'Pmin'): 30.4, ('g4', 'b'): 13.32, ('g4', 'RU'): 14, ('g4', 'RD'): 14,
    ('g5', 'Pmax'): 155, ('g5', 'Pmin'): 54.25, ('g5', 'b'): 16, ('g5', 'RU'): 21, ('g5', 'RD'): 21,
    ('g6', 'Pmax'): 155, ('g6', 'Pmin'): 54.25, ('g6', 'b'): 10.52, ('g6', 'RU'): 21, ('g6', 'RD'): 21,
    ('g7', 'Pmax'): 310, ('g7', 'Pmin'): 108.5, ('g7', 'b'): 10.52, ('g7', 'RU'): 21, ('g7', 'RD'): 21,
    ('g8', 'Pmax'): 350, ('g8', 'Pmin'): 140, ('g8', 'b'): 10.89, ('g8', 'RU'): 28, ('g8', 'RD'): 28,
    ('g9', 'Pmax'): 350, ('g9', 'Pmin'): 75, ('g9', 'b'): 20.7, ('g9', 'RU'): 49, ('g9', 'RD'): 49,
    ('g10', 'Pmax'): 591, ('g10', 'Pmin'): 206.85, ('g10', 'b'): 20.93, ('g10', 'RU'): 21, ('g10', 'RD'): 21,
    ('g11', 'Pmax'): 60, ('g11', 'Pmin'): 12, ('g11', 'b'): 26.11, ('g11', 'RU'): 7, ('g11', 'RD'): 7,
    ('g12', 'Pmax'): 300, ('g12', 'Pmin'): 0, ('g12', 'b'): 0, ('g12', 'RU'): 35, ('g12', 'RD'): 35,
}

# Generator-to-bus mapping
GB_data = {
    (18, 'g1'): 1, (21, 'g2'): 1, (1, 'g3'): 1, (2, 'g4'): 1,
    (15, 'g5'): 1, (16, 'g6'): 1, (23, 'g7'): 1, (23, 'g8'): 1,
    (7, 'g9'): 1, (13, 'g10'): 1, (15, 'g11'): 1, (22, 'g12'): 1,
}

# Bus demand data (MW)
BusData_ = {
    1: {'pd': 108}, 2: {'pd': 97}, 3: {'pd': 180}, 4: {'pd': 74}, 
    5: {'pd': 71}, 6: {'pd': 136}, 7: {'pd': 125}, 8: {'pd': 171},
    9: {'pd': 175}, 10: {'pd': 195}, 13: {'pd': 265}, 14: {'pd': 194}, 
    15: {'pd': 317}, 16: {'pd': 100}, 18: {'pd': 333}, 19: {'pd': 181}, 
    20: {'pd': 128}
}

# Wind capacity per bus (MW)
Wcap_data_base = {8: 200, 19: 200, 21: 200}

# Transmission line data
branch_data = {
    (1, 2): {'x': 0.0139, 'Limit': 175}, (1, 3): {'x': 0.2112, 'Limit': 175},
    (1, 5): {'x': 0.0845, 'Limit': 175}, (2, 4): {'x': 0.1267, 'Limit': 175},
    (2, 6): {'x': 0.1920, 'Limit': 175}, (3, 9): {'x': 0.1190, 'Limit': 175},
    (3, 24): {'x': 0.0839, 'Limit': 400}, (4, 9): {'x': 0.1037, 'Limit': 175},
    (5, 10): {'x': 0.0883, 'Limit': 175}, (6, 10): {'x': 0.0605, 'Limit': 175},
    (7, 8): {'x': 0.0614, 'Limit': 175}, (8, 9): {'x': 0.1651, 'Limit': 175},
    (8, 10): {'x': 0.1651, 'Limit': 175}, (9, 11): {'x': 0.0839, 'Limit': 400},
    (9, 12): {'x': 0.0839, 'Limit': 400}, (10, 11): {'x': 0.0839, 'Limit': 400},
    (10, 12): {'x': 0.0839, 'Limit': 400}, (11, 13): {'x': 0.0476, 'Limit': 500},
    (11, 14): {'x': 0.0418, 'Limit': 500}, (12, 13): {'x': 0.0476, 'Limit': 500},
    (12, 23): {'x': 0.0966, 'Limit': 500}, (13, 23): {'x': 0.0865, 'Limit': 500},
    (14, 16): {'x': 0.0389, 'Limit': 500}, (15, 16): {'x': 0.0173, 'Limit': 500},
    (15, 21): {'x': 0.0245, 'Limit': 1000}, (15, 24): {'x': 0.0519, 'Limit': 500},
    (16, 17): {'x': 0.0259, 'Limit': 500}, (16, 19): {'x': 0.0231, 'Limit': 500},
    (17, 18): {'x': 0.0144, 'Limit': 500}, (17, 22): {'x': 0.1053, 'Limit': 500},
    (18, 21): {'x': 0.0130, 'Limit': 1000}, (19, 20): {'x': 0.0198, 'Limit': 1000},
    (20, 23): {'x': 0.0108, 'Limit': 1000}, (21, 22): {'x': 0.0678, 'Limit': 500}
}

# ============================================================
# OPTIMIZATION FUNCTIONS
# ============================================================
@st.cache_data(show_spinner=False)
def build_and_solve_model(demand_scenario_name, wind_scenario_name, ess_budget, ess_capacity):
    """
    Build and solve the Pyomo optimization model.
    Returns results dict or error message.
    """
    
    # Build WD dictionary from selected scenarios
    WD = {}
    for t in t_hours:
        WD[t] = {
            'd': DEMAND_SCENARIOS[demand_scenario_name][t],
            'w': WIND_SCENARIOS[wind_scenario_name][t]
        }
    
    # Prepare data structures
    
    BusData = {i: {'pd': 0} for i in range(1, 25)}
    BusData.update(BusData_)
    
    Wcap_data = {i: 0 for i in range(1, 25)}
    Wcap_data.update(Wcap_data_base)
    
    SOCMax = {i: ess_capacity for i in range(1, 25)}
    SOC0 = {i: 0.2 * SOCMax[i] / Sbase for i in range(1, 25)}
    
    GB = {(bus, gen): 0 for bus in range(1, 25) for gen in [f'g{i}' for i in range(1, 13)]}
    GB.update(GB_data)
    
    # Build branch dictionary with both directions
    branch = {}
    for (bus, node), data in branch_data.items():
        branch[(bus, node)] = data.copy()
        branch[(bus, node)]['bij'] = 1 / data['x']
        branch[(node, bus)] = data.copy()
        branch[(node, bus)]['bij'] = 1 / data['x']
    
    # Build Pyomo model
    model = ConcreteModel()
    model.name = "ESS Allocation Optimizer"
    
    # Sets
    model.t = Set(initialize=t_hours)
    model.bus = RangeSet(1, 24)
    model.node = RangeSet(1, 24)
    model.Gen = Set(initialize=[f'g{i}' for i in range(1, 13)])
    model.slack = Set(initialize=[13])
    
    # Variables
    model.delta = Var(model.bus, model.t, bounds=(-pi/2, pi/2))
    for t in model.t:
        model.delta[13, t].fix(0)  # Slack bus
    
    model.lsh = Var(model.bus, model.t, bounds=lambda m, bus, t: (0, WD[t]['d'] * BusData[bus]['pd'] / Sbase))
    model.Pg = Var(model.Gen, model.t, bounds=lambda m, gen, t: (GD[gen,'Pmin']/Sbase, GD[gen,'Pmax']/Sbase))
    
    def Limit(model, bus, node, t):
        if (bus, node) in branch:
            lim = branch[bus, node]['Limit'] / Sbase
            return (-lim, lim)
        return (0.0, 0.0)
    model.Pij = Var(model.bus, model.node, model.t, bounds=Limit)
    
    model.Pw = Var(model.bus, model.t, bounds=lambda m, bus, t: (0, WD[t]['w'] * Wcap_data[bus] / Sbase))
    model.pwc = Var(model.bus, model.t, bounds=lambda m, bus, t: (0, WD[t]['w'] * Wcap_data[bus] / Sbase))
    
    model.NESS = Var(model.bus, bounds=(0, NBUSMAX))
    model.SOC = Var(model.bus, model.t)
    model.Pc = Var(model.bus, model.t)
    model.Pd = Var(model.bus, model.t)
    
    # Constraints
    model.const_maxsoc = Constraint(model.bus, model.t, 
        rule=lambda m, bus, t: m.SOC[bus,t] <= m.NESS[bus]*SOCMax[bus]/Sbase)
    model.const_minsoc = Constraint(model.bus, model.t,
        rule=lambda m, bus, t: m.SOC[bus,t] >= 0)
    
    model.const_maxpc = Constraint(model.bus, model.t,
        rule=lambda m, bus, t: m.Pc[bus,t] <= 0.2*m.NESS[bus]*SOCMax[bus]/Sbase)
    model.const_minpc = Constraint(model.bus, model.t,
        rule=lambda m, bus, t: m.Pc[bus,t] >= 0)
    
    model.const_maxpd = Constraint(model.bus, model.t,
        rule=lambda m, bus, t: m.Pd[bus,t] <= 0.2*m.NESS[bus]*SOCMax[bus]/Sbase)
    model.const_minpd = Constraint(model.bus, model.t,
        rule=lambda m, bus, t: m.Pd[bus,t] >= 0)
    
    def ramp_up_rule(model, gen, t):
        if t == 't24':
            return Constraint.Skip
        return model.Pg[gen, t_hours[t_hours.index(t) + 1]] - model.Pg[gen, t] <= GD[gen,'RU'] / Sbase
    model.const_rampup = Constraint(model.Gen, model.t, rule=ramp_up_rule)
    
    def ramp_down_rule(model, gen, t):
        if t == 't1':
            return Constraint.Skip
        return model.Pg[gen, t_hours[t_hours.index(t) - 1]] - model.Pg[gen, t] <= GD[gen,'RD'] / Sbase
    model.const_rampdown = Constraint(model.Gen, model.t, rule=ramp_down_rule)
    
    def power_flow_rule(model, bus, node, t):
        if (bus, node) in branch:
            return model.Pij[bus, node, t] == branch[bus, node]['bij'] * (model.delta[bus, t] - model.delta[node, t])
        return Constraint.Skip
    model.const_powerflow = Constraint(model.bus, model.node, model.t, rule=power_flow_rule)
    
    def power_balance_rule(model, bus, t):
        item = sum(model.Pg[gen, t] for gen in model.Gen if GB[bus, gen] == 1) - WD[t]['d']*BusData[bus]['pd']/Sbase
        if BusData[bus]['pd'] > 0:
            item += model.lsh[bus, t]
        if Wcap_data[bus] > 0:
            item += model.Pw[bus, t]
        if SOCMax[bus] > 0:
            item += model.Pd[bus, t] - model.Pc[bus, t]
        item += sum(model.Pij[bus, node, t] for node in model.node if (node, bus) in branch)
        return item == 0
    model.const_balance = Constraint(model.bus, model.t, rule=power_balance_rule)
    
    def wind_curtailment_rule(model, bus, t):
        if Wcap_data[bus] > 0:
            return model.pwc[bus, t] == WD[t]['w'] * Wcap_data[bus] / Sbase - model.Pw[bus, t]
        return Constraint.Skip
    model.const_windcurt = Constraint(model.bus, model.t, rule=wind_curtailment_rule)
    
    model.const_ess_budget = Constraint(rule=lambda m: sum(m.NESS[bus] for bus in m.bus) <= ess_budget)
    model.const_ess_final = Constraint(model.bus, rule=lambda m, bus: m.SOC[bus,'t24'] == m.NESS[bus]*SOC0[bus])
    
    def soc_evolution_rule(model, bus, t):
        if SOCMax[bus] > 0:
            if t == 't1':
                return model.SOC[bus, t] == model.NESS[bus]*SOC0[bus] + model.Pc[bus,t]*eta_c - model.Pd[bus, t]/eta_d
            else:
                prev_t = t_hours[t_hours.index(t)-1]
                return model.SOC[bus,t] == model.SOC[bus,prev_t] + model.Pc[bus,t]*eta_c - model.Pd[bus, t]/eta_d
        return Constraint.Skip
    model.const_soc = Constraint(model.bus, model.t, rule=soc_evolution_rule)
    
    # Objective
    def objective_rule(model):
        gen_cost = sum(model.Pg[gen, t] * GD[gen,'b'] * Sbase for t in model.t for gen in model.Gen)
        voll_cost = sum(VOLL * model.lsh[bus,t] * Sbase for t in model.t for bus in model.bus)
        curt_cost = sum(VOLW * model.pwc[bus,t] * Sbase for t in model.t for bus in model.bus)
        return gen_cost + voll_cost + curt_cost
    model.obj = Objective(rule=objective_rule, sense=minimize)
    
    # Solve
    try:
        solver = SolverFactory('glpk')
        results = solver.solve(model, tee=False)
        
        # Check solution status
        if results.solver.status != SolverStatus.ok or results.solver.termination_condition != TerminationCondition.optimal:
            return {'status': 'error', 'message': 'Solver did not find optimal solution'}
        
        # Extract results
        ess_allocation = {}
        for bus in model.bus:
            val = model.NESS[bus].value
            if val and val > 0.01:
                ess_allocation[bus] = round(val)
        
        # Calculate costs
        total_cost = model.obj()
        gen_cost = sum(model.Pg[gen, t].value * GD[gen,'b'] * Sbase for t in model.t for gen in model.Gen)
        voll_cost = sum(VOLL * model.lsh[bus,t].value * Sbase for t in model.t for bus in model.bus)
        curt_cost = sum(VOLW * model.pwc[bus,t].value * Sbase for t in model.t for bus in model.bus)
        load_shed = sum(model.lsh[bus, t].value * Sbase for bus in model.bus for t in model.t)
        wind_curt = sum(model.pwc[bus, t].value * Sbase for bus in model.bus for t in model.t)
        
        # Extract time series data
        gen_dispatch = []
        for t in model.t:
            for gen in model.Gen:
                gen_dispatch.append({
                    'time': t,
                    'hour': int(t[1:]),
                    'generator': gen,
                    'power_MW': model.Pg[gen, t].value * Sbase
                })
        
        soc_data = []
        for bus in ess_allocation.keys():
            for t in model.t:
                soc_data.append({
                    'bus': bus,
                    'time': t,
                    'hour': int(t[1:]),
                    'SOC_MWh': model.SOC[bus, t].value * Sbase,
                    'charge_MW': model.Pc[bus, t].value * Sbase,
                    'discharge_MW': model.Pd[bus, t].value * Sbase
                })
        
        wind_data = []
        for bus in [8, 19, 21]:  # Wind buses
            for t in model.t:
                wind_data.append({
                    'bus': bus,
                    'time': t,
                    'hour': int(t[1:]),
                    'output_MW': model.Pw[bus, t].value * Sbase,
                    'curtailed_MW': model.pwc[bus, t].value * Sbase,
                    'available_MW': WD[t]['w'] * Wcap_data[bus]
                })
        
        return {
            'status': 'success',
            'total_cost': total_cost,
            'gen_cost': gen_cost,
            'voll_cost': voll_cost,
            'curt_cost': curt_cost,
            'load_shed': load_shed,
            'wind_curt': wind_curt,
            'ess_allocation': ess_allocation,
            'df_gen': pd.DataFrame(gen_dispatch),
            'df_soc': pd.DataFrame(soc_data),
            'df_wind': pd.DataFrame(wind_data)
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# ============================================================
# APP HEADER
# ============================================================
st.title("Energy Storage System Allocation Optimizer")
st.markdown("""
Optimize battery placement in a 24-bus power network to minimize operating costs 
while maximizing renewable energy integration.
""")

# ============================================================
# SIDEBAR: USER INPUTS
# ============================================================
st.sidebar.header("⚙️ Optimization Settings")

st.sidebar.subheader("Scenario Selection")
demand_scenario = st.sidebar.selectbox(
    "Demand Pattern",
    options=list(DEMAND_SCENARIOS.keys()),
    help="Select the electricity demand profile for the day"
)

wind_scenario = st.sidebar.selectbox(
    "Wind Pattern",
    options=list(WIND_SCENARIOS.keys()),
    help="Select the wind availability profile for the day"
)

st.sidebar.subheader("Storage Parameters")
ess_budget = st.sidebar.slider(
    "Total ESS Units Available",
    min_value=5,
    max_value=25,
    value=15,
    step=1,
    help="Maximum number of storage units to allocate across the network"
)

ess_capacity = st.sidebar.slider(
    "ESS Unit Capacity (MWh)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
    help="Energy capacity of each storage unit"
)

# ============================================================
# SCENARIO PREVIEW
# ============================================================
st.header("Selected Scenarios")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demand Profile")
    # Create demand preview chart
    hours = list(range(1, 25))
    demand_values = [DEMAND_SCENARIOS[demand_scenario][f't{i}'] for i in hours]
    
    fig_demand = go.Figure()
    fig_demand.add_trace(go.Scatter(
        x=hours,
        y=demand_values,
        mode='lines+markers',
        name='Demand',
        line=dict(color='#FF6B6B', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    fig_demand.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Demand (per unit)",
        height=300,
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig_demand, width='stretch')

with col2:
    st.subheader("Wind Profile")
    wind_values = [WIND_SCENARIOS[wind_scenario][f't{i}'] for i in hours]
    
    fig_wind = go.Figure()
    fig_wind.add_trace(go.Scatter(
        x=hours,
        y=wind_values,
        mode='lines+markers',
        name='Wind',
        line=dict(color='#4ECDC4', width=3),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.2)'
    ))
    fig_wind.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Wind Availability (per unit)",
        height=300,
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig_wind, width='stretch')

# Show scenario description
st.info(f"""
**Current Configuration:** {demand_scenario} + {wind_scenario}  
**ESS Budget:** {ess_budget} units x {ess_capacity} MWh = {ess_budget * ess_capacity} MWh total capacity
""")

# ============================================================
# RUN OPTIMIZATION
# ============================================================
st.divider()

if st.button("Run Optimization", type="primary", width='stretch'):
    with st.spinner("⚡ Solving optimization model... This may take 10-30 seconds"):
        results = build_and_solve_model(
            demand_scenario,
            wind_scenario,
            ess_budget,
            ess_capacity
        )
    
    if results['status'] == 'error':
        st.error(f"❌ Optimization failed: {results['message']}")
    else:
        st.success("✅ Optimal solution found!")
        
        # ============================================================
        # RESULTS: KEY METRICS
        # ============================================================
        st.header("Optimization Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cost",
                f"£{results['total_cost']:,.0f}",
                help="Total operating cost for 24-hour period"
            )
        
        with col2:
            total_units = sum(results['ess_allocation'].values())
            st.metric(
                "ESS Units Used",
                f"{total_units:.0f} / {ess_budget}",
                help="Number of storage units allocated"
            )
        
        with col3:
            st.metric(
                "Load Shed",
                f"{results['load_shed']:.1f} MWh",
                delta="0 is ideal" if results['load_shed'] == 0 else None,
                delta_color="inverse",
                help="Unmet demand (lower is better)"
            )
        
        with col4:
            st.metric(
                "Wind Curtailed",
                f"{results['wind_curt']:.1f} MWh",
                help="Renewable energy wasted"
            )
        
        # ============================================================
        # RESULTS: ESS ALLOCATION
        # ============================================================
        st.subheader("Energy Storage Allocation")
        
        if results['ess_allocation']:
            # Create bar chart
            ess_df = pd.DataFrame([
                {'Bus': bus, 'Units': units, 'Capacity_MWh': units * ess_capacity}
                for bus, units in results['ess_allocation'].items()
            ]).sort_values('Units', ascending=True)
            
            fig_ess = go.Figure()
            fig_ess.add_trace(go.Bar(
                y=[f"Bus {bus}" for bus in ess_df['Bus']],
                x=ess_df['Units'],
                orientation='h',
                marker=dict(color='#4ECDC4'),
                text=ess_df['Units'],
                textposition='outside',
                hovertemplate='<b>Bus %{y}</b><br>Units: %{x}<br>Capacity: %{customdata} MWh<extra></extra>',
                customdata=ess_df['Capacity_MWh']
            ))
            fig_ess.update_layout(
                xaxis_title="Number of ESS Units",
                yaxis_title="",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_ess, width='stretch')
            
            # Show context table
            st.markdown("**Why these locations?**")
            context_data = []
            for bus in results['ess_allocation'].keys():
                has_gen = any(GB_data.get((bus, f'g{i}'), 0) == 1 for i in range(1, 13))
                gen_list = [f'g{i}' for i in range(1, 13) if GB_data.get((bus, f'g{i}'), 0) == 1]
                context_data.append({
                    'Bus': bus,
                    'ESS Units': results['ess_allocation'][bus],
                    'Local Demand (MW)': BusData_.get(bus, 0),
                    'Wind Capacity (MW)': Wcap_data_base.get(bus, 0),
                    'Generator': ', '.join(gen_list) if gen_list else 'None'
                })
            
            st.dataframe(pd.DataFrame(context_data), width='stretch', hide_index=True)
        else:
            st.warning("No ESS units were allocated. Try adjusting parameters.")
        
        # ============================================================
        # RESULTS: COST BREAKDOWN
        # ============================================================
        st.subheader("Cost Breakdown")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pie chart
            cost_data = pd.DataFrame({
                'Category': ['Generation', 'Load Shedding Penalty', 'Wind Curtailment Penalty'],
                'Cost': [results['gen_cost'], results['voll_cost'], results['curt_cost']]
            })
            
            fig_costs = px.pie(
                cost_data,
                values='Cost',
                names='Category',
                hole=0.4,
                color_discrete_sequence=['#FF6B6B', '#FFE66D', '#4ECDC4']
            )
            fig_costs.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_costs, width='stretch')
        
        with col2:
            st.markdown("**Cost Details:**")
            st.write(f"🔥 Generation: £{results['gen_cost']:,.0f}")
            st.write(f"⚠️ Load Shedding: £{results['voll_cost']:,.0f}")
            st.write(f"🌬️ Curtailment: £{results['curt_cost']:,.0f}")
            st.write(f"**Total: £{results['total_cost']:,.0f}**")
