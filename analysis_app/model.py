"""逐折交叉拟合的收益模型。

岭回归以 segment 与 action 的哑变量为特征预测净收益（revenue-cost）。
每一折的编码水平、惩罚系数选择与系数拟合只使用该折训练期
（验证周之前的全部周）的数据，不允许先看全量数据再报告交叉拟合。
"""
import math
import statistics

from .io import fingerprint
from .schema import PRODUCT

MODEL_KIND = 'offpolicy_reward_model'
MODEL_VERSION = 1
DEFAULT_PENALTIES = (0.01, 0.1, 1.0, 10.0)


def order_rows(records):
    """训练与评估统一按 (week, observation_id) 排序，输入行序不影响结果。"""
    return sorted(records, key=lambda row: (row['week'], row['observation_id']))


def net(row):
    return row['revenue'] - row['cost']


def fold_plan(records, min_train_weeks):
    """按 week 向前推进：每折验证一周，训练期为此之前的全部周。"""
    weeks = sorted({row['week'] for row in records})
    return [{'validation_week': weeks[cut], 'train_weeks': weeks[:cut]}
            for cut in range(min_train_weeks, len(weeks))]


def encode_levels(rows):
    """编码水平只来自传入的训练行；排序保证与行序无关。"""
    return (sorted({row['segment'] for row in rows}),
            sorted({row['action'] for row in rows}))


def feature_names(segment_levels, action_levels):
    return (['intercept']
            + ['segment:' + level for level in segment_levels[1:]]
            + ['action:' + level for level in action_levels[1:]])


def encode(segment, action, segment_levels, action_levels):
    """排序后的首个水平是基线；训练期未出现的水平回落到全零哑变量。"""
    return ([1.0]
            + [1.0 if segment == level else 0.0 for level in segment_levels[1:]]
            + [1.0 if action == level else 0.0 for level in action_levels[1:]])


def _solve(matrix, vector):
    """部分主元高斯消元，输入确定则输出确定。"""
    size = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            aug[col][col] = 1e-12
        for r in range(col + 1, size):
            factor = aug[r][col] / aug[col][col]
            if factor == 0:
                continue
            for k in range(col, size + 1):
                aug[r][k] -= factor * aug[col][k]
    out = [0.0] * size
    for i in range(size - 1, -1, -1):
        residual = aug[i][size] - math.fsum(aug[i][j] * out[j] for j in range(i + 1, size))
        out[i] = residual / aug[i][i]
    return out


def fit_ridge(rows, segment_levels, action_levels, penalty):
    """岭回归，截距不惩罚，其余系数统一惩罚。"""
    names = feature_names(segment_levels, action_levels)
    size = len(names)
    xtx = [[0.0] * size for _ in range(size)]
    xty = [0.0] * size
    for row in rows:
        vec = encode(row['segment'], row['action'], segment_levels, action_levels)
        target = net(row)
        for i in range(size):
            if vec[i] == 0.0:
                continue
            xty[i] += vec[i] * target
            for j in range(size):
                if vec[j] != 0.0:
                    xtx[i][j] += vec[i] * vec[j]
    for diag in range(1, size):
        xtx[diag][diag] += penalty
    coef = _solve(xtx, xty)
    return dict(zip(names, coef))


def predict(model, segment, action):
    coef = model['coefficients']
    return math.fsum([coef.get('intercept', 0.0),
                      coef.get('segment:' + segment, 0.0),
                      coef.get('action:' + action, 0.0)])


def select_penalty(rows, penalties):
    """只用训练期内部的前向切分选择惩罚系数。

    单一候选直接返回；训练期不足两周无法内部验证时取网格中位数并标记。
    其余情况在训练周内做前向链式验证，按平均 MSE 最小（并列取更小惩罚）。
    """
    if len(penalties) == 1:
        return penalties[0], 'fixed', []
    weeks = sorted({row['week'] for row in rows})
    if len(weeks) < 2:
        return penalties[len(penalties) // 2], 'fallback_single_train_week', []
    totals = {penalty: 0.0 for penalty in penalties}
    for cut in range(1, len(weeks)):
        inner_train = [row for row in rows if row['week'] < weeks[cut]]
        inner_val = [row for row in rows if row['week'] == weeks[cut]]
        seg_levels, act_levels = encode_levels(inner_train)
        for penalty in penalties:
            coef = fit_ridge(inner_train, seg_levels, act_levels, penalty)
            model = {'coefficients': coef}
            mse = statistics.fmean(
                (predict(model, row['segment'], row['action']) - net(row)) ** 2
                for row in inner_val)
            totals[penalty] += mse
    scores = [{'penalty': penalty, 'mse': totals[penalty] / (len(weeks) - 1)}
              for penalty in penalties]
    best = min(penalties, key=lambda penalty: totals[penalty])
    return best, 'inner_forward', scores


def train_models(records, min_train_weeks=1, penalties=DEFAULT_PENALTIES):
    """按折计划逐折训练，返回可序列化、可校验的模型文件内容。"""
    ordered = order_rows(records)
    grid = sorted(penalties)
    folds = []
    for spec in fold_plan(ordered, min_train_weeks):
        train_weeks = set(spec['train_weeks'])
        train_rows = [row for row in ordered if row['week'] in train_weeks]
        seg_levels, act_levels = encode_levels(train_rows)
        penalty, method, scores = select_penalty(train_rows, grid)
        coef = fit_ridge(train_rows, seg_levels, act_levels, penalty)
        folds.append({'validation_week': spec['validation_week'],
                      'train_weeks': spec['train_weeks'],
                      'train_rows': len(train_rows),
                      'train_fingerprint': fingerprint(train_rows),
                      'model': {'penalty': penalty, 'selection': method, 'scores': scores,
                                'segment_levels': seg_levels, 'action_levels': act_levels,
                                'coefficients': coef}})
    return {'product': PRODUCT, 'kind': MODEL_KIND, 'schema_version': MODEL_VERSION,
            'config': {'min_train_weeks': min_train_weeks, 'penalties': grid},
            'rows': len(ordered), 'input_fingerprint': fingerprint(ordered),
            'folds': folds}


def verify_models(records, blob):
    """加载模型前校验：折结构与每折训练数据指纹必须与当前输入一致。"""
    if not isinstance(blob, dict) or blob.get('kind') != MODEL_KIND:
        raise ValueError('模型文件类型不符')
    if blob.get('schema_version') != MODEL_VERSION:
        raise ValueError('模型文件版本不符')
    config = blob.get('config') or {}
    ordered = order_rows(records)
    plan = fold_plan(ordered, config.get('min_train_weeks', 1))
    folds = blob.get('folds') or []
    if len(folds) != len(plan):
        raise ValueError('模型文件折数与当前数据不一致')
    for fold, spec in zip(folds, plan):
        if (fold.get('validation_week') != spec['validation_week']
                or fold.get('train_weeks') != spec['train_weeks']):
            raise ValueError('模型文件折结构与当前数据不一致')
        train_weeks = set(spec['train_weeks'])
        train_rows = [row for row in ordered if row['week'] in train_weeks]
        if fold.get('train_fingerprint') != fingerprint(train_rows):
            raise ValueError('模型文件训练指纹与当前数据不一致')
    return blob
