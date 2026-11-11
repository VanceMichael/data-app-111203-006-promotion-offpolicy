FIELDS = {'observation_id': 'str', 'session': 'str', 'action': 'str', 'revenue': 'float', 'cost': 'float', 'propensity': 'float', 'week': 'int', 'segment': 'str', 'candidate': 'str'}
REQUIRED = ['observation_id', 'session', 'action', 'revenue', 'cost', 'propensity', 'week', 'segment', 'candidate']
ENTITY = 'session'
GROUP = 'action'
MEASURE = 'revenue'
PRODUCT = 'promotion-offpolicy'
UNIT = '元'
