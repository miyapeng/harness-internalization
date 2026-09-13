from ..core.types import Cost, Journal

def total_cost(costs):
    return sum(costs, Cost())
