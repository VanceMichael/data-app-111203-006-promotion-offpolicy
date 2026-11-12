"""离线策略评估：支持域检查、IPW/SNIPS/DR 估计、裁剪披露与区间。

估计目标是候选策略在可评估会话上的单次会话净收益（revenue-cost）期望。
每一行先接受按时间前向切分的支持域检查，被排除的行按原因披露，
不进入任何估计分母；三个估计量共用同一可评估集合，便于直接比较。
"""
import math
import statistics

from .io import fingerprint
from .model import net, order_rows, predict
from .schema import PRODUCT, UNIT

DECISION_KIND = 'offpolicy_decision'
DECISION_VERSION = 1
REASONS = ('invalid_propensity', 'zero_propensity', 'insufficient_history', 'no_action_coverage')
REASON_LABELS = {
    'invalid_propensity': '不合法概率（propensity 不在 [0,1] 内）',
    'zero_propensity': '零概率（propensity 为 0，实际动作不可能被记录）',
    'insufficient_history': '历史窗口不足（该行周只进入训练期，无对应验证折）',
    'no_action_coverage': '候选动作无历史覆盖（该客群训练期内从未实际投放此动作）',
}
ESTIMATORS = ('ipw', 'snips', 'dr')


def _classify(ordered, folds):
    """逐行给出排除原因，优先级：不合法概率 > 零概率 > 历史不足 > 无覆盖。"""
    weekly = {}
    for row in ordered:
        weekly.setdefault(row['week'], {}).setdefault(row['segment'], set()).add(row['action'])
    coverage = []
    for fold in folds:
        cover = {}
        for week in fold['train_weeks']:
            for segment, actions in weekly.get(week, {}).items():
                cover.setdefault(segment, set()).update(actions)
        coverage.append(cover)
    validation_weeks = {fold['validation_week']: idx for idx, fold in enumerate(folds)}
    status = {}
    for row in ordered:
        propensity = row['propensity']
        if propensity < 0 or propensity > 1:
            status[row['observation_id']] = ('invalid_propensity', None)
        elif propensity == 0:
            status[row['observation_id']] = ('zero_propensity', None)
        elif row['week'] not in validation_weeks:
            status[row['observation_id']] = ('insufficient_history', None)
        else:
            idx = validation_weeks[row['week']]
            if row['candidate'] not in coverage[idx].get(row['segment'], ()):
                status[row['observation_id']] = ('no_action_coverage', idx)
            else:
                status[row['observation_id']] = (None, idx)
    return status


def _none_block():
    return {'value': None, 'se': None, 'ci': None,
            'gain': None, 'gain_se': None, 'gain_ci': None}


def _se(deviations):
    size = len(deviations)
    if size < 2:
        return None
    mean = math.fsum(deviations) / size
    var = math.fsum((item - mean) ** 2 for item in deviations) / (size - 1)
    return math.sqrt(var / size)


def _interval(center, se, z):
    if center is None or se is None:
        return None
    return [center - z * se, center + z * se]


def _estimator_block(center, phi, targets, baseline, z):
    se = _se(phi)
    gain = None if center is None else center - baseline
    gain_se = _se([item - target for item, target in zip(phi, targets)])
    return {'value': center, 'se': se, 'ci': _interval(center, se, z),
            'gain': gain, 'gain_se': gain_se, 'gain_ci': _interval(gain, gain_se, z)}


def _compute(entries, z):
    """在同一可评估集合上计算三个估计量；entries 的 weight 已按阈值裁剪。"""
    if not entries:
        return {'n': 0, 'matched': 0, 'ess': 0.0, 'baseline': None,
                'ipw': _none_block(), 'snips': _none_block(), 'dr': _none_block()}
    size = len(entries)
    targets = [entry['net'] for entry in entries]
    weights = [entry['weight'] for entry in entries]
    baseline = math.fsum(targets) / size
    sum_w = math.fsum(weights)
    sum_w2 = math.fsum(weight * weight for weight in weights)
    ess = (sum_w * sum_w / sum_w2) if sum_w2 > 0 else 0.0
    ipw_phi = [weight * target for weight, target in zip(weights, targets)]
    ipw = _estimator_block(math.fsum(ipw_phi) / size, ipw_phi, targets, baseline, z)
    if sum_w > 0:
        sn_value = math.fsum(weight * target for weight, target in zip(weights, targets)) / sum_w
        sn_phi = [weight * (target - sn_value) * size / sum_w
                  for weight, target in zip(weights, targets)]
        snips = _estimator_block(sn_value, sn_phi, targets, baseline, z)
    else:
        snips = _none_block()
    dr_phi = [entry['pred_candidate'] + entry['weight'] * (entry['net'] - entry['pred_actual'])
              for entry in entries]
    dr = _estimator_block(math.fsum(dr_phi) / size, dr_phi, targets, baseline, z)
    matched = sum(1 for entry in entries if entry['weight_raw'] > 0)
    return {'n': size, 'matched': matched, 'ess': ess, 'baseline': baseline,
            'ipw': ipw, 'snips': snips, 'dr': dr}


def evaluate(records, models, clip=None, confidence=0.95):
    """对候选策略做离线评估，返回完整决策文件内容（含被排除部分）。

    models 是 train_models 的产物（或经 verify_models 校验的加载结果）；
    clip 为权重裁剪阈值，None 表示不裁剪。
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError('置信水平需在 (0,1) 内')
    if clip is not None and clip <= 0:
        raise ValueError('裁剪阈值需为正数，或传 None 表示不裁剪')
    ordered = order_rows(records)
    folds = models['folds']
    status = _classify(ordered, folds)
    z = statistics.NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    excluded = {reason: 0 for reason in REASONS}
    entries = []
    fold_entries = [[] for _ in folds]
    fold_excluded = [{reason: 0 for reason in REASONS} for _ in folds]
    fold_row_counts = [0] * len(folds)
    week_to_fold = {fold['validation_week']: idx for idx, fold in enumerate(folds)}
    segment_rows = {}
    segment_excluded = {}
    for row in ordered:
        reason, idx = status[row['observation_id']]
        segment_rows[row['segment']] = segment_rows.get(row['segment'], 0) + 1
        segment_excluded.setdefault(row['segment'], {key: 0 for key in REASONS})
        week_idx = week_to_fold.get(row['week'])
        if week_idx is not None:
            fold_row_counts[week_idx] += 1
        if reason is not None:
            excluded[reason] += 1
            segment_excluded[row['segment']][reason] += 1
            if week_idx is not None:
                fold_excluded[week_idx][reason] += 1
            continue
        weight_raw = (1.0 / row['propensity']) if row['action'] == row['candidate'] else 0.0
        weight = min(weight_raw, clip) if clip is not None else weight_raw
        model = folds[idx]['model']
        entry = {'net': net(row), 'weight': weight, 'weight_raw': weight_raw,
                 'pred_candidate': predict(model, row['segment'], row['candidate']),
                 'pred_actual': predict(model, row['segment'], row['action']),
                 'segment': row['segment']}
        entries.append(entry)
        fold_entries[idx].append(entry)
    total = len(ordered)
    size = len(entries)
    overall = _compute(entries, z)
    weights_raw = [entry['weight_raw'] for entry in entries]
    sum_raw = math.fsum(weights_raw)
    sum_w = math.fsum(entry['weight'] for entry in entries)
    rows_clipped = sum(1 for entry in entries if entry['weight'] < entry['weight_raw'])
    raw_w2 = math.fsum(weight * weight for weight in weights_raw)
    ess_unclipped = (sum_raw * sum_raw / raw_w2) if raw_w2 > 0 else 0.0
    weights_info = {'clip': clip, 'rows_clipped': rows_clipped,
                    'clipped_weight_share': (1.0 - sum_w / sum_raw) if sum_raw > 0 else 0.0,
                    'max_weight': max(weights_raw) if weights_raw else None,
                    'ess': overall['ess'], 'ess_unclipped': ess_unclipped,
                    'ess_share': (overall['ess'] / size) if size else None}
    segments = {}
    for segment in sorted(segment_rows):
        seg_entries = [entry for entry in entries if entry['segment'] == segment]
        seg = _compute(seg_entries, z)
        segments[segment] = {
            'rows': segment_rows[segment], 'evaluable': seg['n'],
            'coverage': seg['n'] / segment_rows[segment],
            'excluded': segment_excluded[segment],
            'matched': seg['matched'], 'ess': seg['ess'], 'baseline': seg['baseline'],
            'estimates': {name: seg[name]['value'] for name in ESTIMATORS},
            'gains': {name: seg[name]['gain'] for name in ESTIMATORS}}
    fold_section = []
    for idx, fold in enumerate(folds):
        fold_section.append({'validation_week': fold['validation_week'],
                             'train_weeks': fold['train_weeks'],
                             'train_rows': fold['train_rows'],
                             'rows': fold_row_counts[idx],
                             'evaluable': len(fold_entries[idx]),
                             'excluded': fold_excluded[idx],
                             'matched': sum(1 for entry in fold_entries[idx]
                                            if entry['weight_raw'] > 0),
                             'model': fold['model']})
    warnings = []
    if not folds:
        warnings.append('没有可用验证折：所有行仅落在训练窗口（历史窗口不足）')
    if total and size == 0:
        warnings.append('没有可评估行：全部估计为 null，请检查 excluded 分布')
    if size and sum_w == 0:
        warnings.append('自归一化估计不可用：权重和为零')
    if size == 1:
        warnings.append('可评估行不足两行：区间不可用')
    if rows_clipped:
        warnings.append('存在被裁剪的权重：详见 evaluation.weights')
    if any(excluded.values()):
        warnings.append('存在被排除的行：详见 evaluation.excluded 与 segments')
    return {'product': PRODUCT, 'kind': DECISION_KIND, 'schema_version': DECISION_VERSION,
            'unit': UNIT,
            'estimand': '候选策略在可评估会话上的单次会话净收益（revenue-cost）期望；'
                        '被排除行按原因披露，不进入任何估计分母',
            'config': {'clip': clip, 'confidence': confidence, 'z': z,
                       'min_train_weeks': models['config']['min_train_weeks'],
                       'penalties': models['config']['penalties']},
            'reason_labels': REASON_LABELS,
            'input_fingerprint': fingerprint(ordered),
            'rows': total,
            'evaluation': {'rows_total': total, 'rows_evaluable': size,
                           'coverage': (size / total) if total else None,
                           'matched_rows': overall['matched'],
                           'excluded': excluded,
                           'weights': weights_info,
                           'baseline': overall['baseline'],
                           'estimators': {name: overall[name] for name in ESTIMATORS}},
            'segments': segments,
            'folds': fold_section,
            'warnings': warnings}
