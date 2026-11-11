import math
import statistics
from collections import Counter


def describe(records):
    return {'revenue_total': math.fsum(row['revenue'] for row in records), 'cost_total': math.fsum(row['cost'] for row in records), 'net_total': math.fsum(row['revenue'] - row['cost'] for row in records)}
