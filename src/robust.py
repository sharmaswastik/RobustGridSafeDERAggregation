from pyomo.environ import *

def wpm_find_vertices(qualified_bids, qualified_offers, prices):

    qualified_bids = list(qualified_bids)
    qualified_offers = list(qualified_offers)

    all_ders = (qualified_bids+ qualified_offers)

    if not all_ders:
        return [{}]

    distinct_prices = sorted(set(float(prices[der]) for der in all_ders))
    # print(distinct_prices)
    candidate_vertices = [float("-inf")]

    for i, price in enumerate(distinct_prices):
        candidate_vertices.append(price)

        if i +1 < len(distinct_prices):
            next_price = distinct_prices[i+1]
            candidate_vertices.append((price + next_price)/2)

    candidate_vertices.append(float("inf"))

    vertices = []
    seen_patterns = set()

    for clearing_price in candidate_vertices:
        u = {}

        for d in qualified_bids:
            u[d] = float(prices[d] >= clearing_price)
        for d in qualified_offers:
            u[d] = float(prices[d] <= clearing_price)

        pattern = tuple(u[d] for d in all_ders)
        if pattern in seen_patterns:
            continue

        seen_patterns.add(pattern)
        vertices.append(u)

    return vertices

def vertex_transition(previous_u, current_u, base_alpha, all_ders, tol=1e-8,):

    eligible_ders = {d for d in all_ders if base_alpha.get(d, 0.0) > tol}
    
    entered = {d for d in eligible_ders if current_u.get(d, 0.0) > previous_u.get(d, 0.0)}

    exited = {d for d in eligible_ders if current_u.get(d, 0.0) < previous_u.get(d, 0.0)}

    return entered, exited


def alpha_at_vertex(all_ders, AlphaValues, u):
    return { der: AlphaValues.get(der, 0.0) * u.get(der, 0.0) for der in all_ders}

def build_robust_model(template, vertices, objective_rule):
    if not vertices:
        raise ValueError("No vertices provided to build the robust model.")

    required = {
        "DERQ",
        "ThermalLimitConstraint",
        "SubstationPowerLimitConstraint",
    }

    for name in required:
        if not hasattr(template, name):
            raise AttributeError(f"OPF is missing required component: {name}")

    m = ConcreteModel()
    m.DER = Set(initialize=template.DER, ordered=True)
    m.Scenarios = Set(initialize=range(len(vertices)), ordered=True)

    m.DERAlpha = Var(m.DER, bounds=(0.0,1.0))

    m.DERP = Param(m.DER, initialize={d: value(template.DERP[d]) for d in m.DER})
    m.DERPi = Param(m.DER, initialize={d: value(template.DERPi[d]) for d in m.DER})
    m.DERPhases = Param(m.DER, within=Any, initialize={d: list(template.DERPhases[d]) for d in m.DER})

    m.M = Param(initialize=value(template.M))
    m.Activation = Param(m.Scenarios, m.DER, within=UnitInterval, initialize={(s,d): float(vertices[s].get(d, 0.0)) for s in m.Scenarios for d in m.DER})

    m.Scenario = Block(m.Scenarios)

    for s in m.Scenarios:
        block = m.Scenario[s]
        block.transfer_attributes_from(template.clone())

        for obj in block.component_objects(Objective, active=True):
            obj.deactivate()

        # Discard inherited OPF dual collection if a caller supplies it.
        if hasattr(block, "dual"):
            block.del_component("dual")

        for d in m.DER:
            block.DERAlpha[d].unfix()
            block.DERAlpha[d].setlb(0.0)
            block.DERAlpha[d].setub(1.0)

        block.ActivationLink = Constraint(m.DER, rule=lambda b,d, s=s:(b.DERAlpha[d] == m.Activation[s,d]*m.DERAlpha[d]))

    m.GenerationObjective = Objective(rule=objective_rule, sense=minimize)

    return m


