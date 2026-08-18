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


