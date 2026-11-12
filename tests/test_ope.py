import copy
import csv
import json
import random
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis_app.io import load_records
from analysis_app.model import DEFAULT_PENALTIES, train_models
from analysis_app.ope import evaluate
from analysis_app.schema import FIELDS

# 夹具数据（已知随机策略 P(A0)=0.2, P(A1)=0.3, P(A2)=0.5）上的独立核对结果，
# 由不经过包实现的脚本按公式直接从 CSV 重算得到。
FIXTURE_IPW = 35.949665247215016
FIXTURE_SNIPS = 38.26899848897082
FIXTURE_BASELINE = 43.22172682309995
FIXTURE_ESS = 63.88919667590027
FIXTURE_EVALUABLE = 264
FIXTURE_MATCHED = 72


def make_row(obs, week, segment='G', action='X', candidate='X',
             propensity=0.5, revenue=10.0, cost=0.0):
    return {'observation_id': obs, 'session': 's-' + obs, 'action': action,
            'revenue': float(revenue), 'cost': float(cost),
            'propensity': float(propensity), 'week': week,
            'segment': segment, 'candidate': candidate}


def warning_texts(decision):
    return '；'.join(decision['warnings'])


def hand_rows():
    """两周小样本：week 0 提供覆盖，week 1 用于手算核对。"""
    return [
        make_row('w0-x', 0, action='X', candidate='X', propensity=0.5, revenue=5.0),
        make_row('w0-y', 0, action='Y', candidate='Y', propensity=0.5, revenue=1.0),
        make_row('r1', 1, action='X', candidate='X', propensity=0.5, revenue=12.0, cost=2.0),
        make_row('r2', 1, action='Y', candidate='X', propensity=0.5, revenue=6.0),
        make_row('r3', 1, action='X', candidate='X', propensity=0.25, revenue=8.0),
        make_row('r4', 1, action='Y', candidate='Y', propensity=0.5, revenue=4.0, cost=1.0),
    ]


class HandComputedTest(unittest.TestCase):
    """两个加权估计量与 DR 在手算可复核的小样本上的精确核对。"""

    def test_ipw_snips_exact_values(self):
        decision = evaluate(hand_rows(), train_models(hand_rows(), 1, [1.0]), clip=None)
        ev = decision['evaluation']
        self.assertEqual(ev['rows_evaluable'], 4)
        self.assertEqual(ev['excluded']['insufficient_history'], 2)
        # IPW = (2*10 + 0*6 + 4*8 + 2*3) / 4
        self.assertAlmostEqual(ev['estimators']['ipw']['value'], 14.5, places=12)
        # SNIPS = 58 / (2 + 0 + 4 + 2)
        self.assertAlmostEqual(ev['estimators']['snips']['value'], 7.25, places=12)
        self.assertAlmostEqual(ev['baseline'], 6.75, places=12)
        self.assertAlmostEqual(ev['estimators']['ipw']['gain'], 7.75, places=12)
        self.assertAlmostEqual(ev['estimators']['snips']['gain'], 0.5, places=12)
        # ESS = 8^2 / (4 + 0 + 16 + 4)
        self.assertAlmostEqual(ev['weights']['ess'], 64.0 / 24.0, places=12)
        self.assertEqual(ev['weights']['max_weight'], 4.0)
        self.assertEqual(ev['weights']['rows_clipped'], 0)

    def test_dr_exact_with_known_ridge_solution(self):
        # 训练行只有 X(净 5) 与 Y(净 1)，惩罚 1.0、截距不惩罚的岭回归
        # 解析解为 intercept=11/3、action:Y=-4/3，逐行推出 DR = 67/6。
        decision = evaluate(hand_rows(), train_models(hand_rows(), 1, [1.0]), clip=None)
        dr = decision['evaluation']['estimators']['dr']
        self.assertAlmostEqual(dr['value'], 67.0 / 6.0, places=12)
        self.assertAlmostEqual(dr['gain'], 67.0 / 6.0 - 6.75, places=12)

    def test_clip_changes_weights_and_is_disclosed(self):
        decision = evaluate(hand_rows(), train_models(hand_rows(), 1, [1.0]), clip=3.0)
        weights = decision['evaluation']['weights']
        self.assertEqual(weights['rows_clipped'], 1)
        self.assertAlmostEqual(weights['clipped_weight_share'], 0.125, places=12)
        # 裁剪后权重 2,0,3,2：IPW = (20 + 0 + 24 + 6) / 4
        self.assertAlmostEqual(decision['evaluation']['estimators']['ipw']['value'], 12.5, places=12)
        self.assertAlmostEqual(decision['evaluation']['estimators']['snips']['value'],
                               50.0 / 7.0, places=12)
        self.assertAlmostEqual(weights['ess'], 49.0 / 17.0, places=12)
        # 裁剪压缩权重差异，ESS 应上升
        self.assertGreater(weights['ess'], weights['ess_unclipped'])
        self.assertIn('存在被裁剪的权重', warning_texts(decision))

    def test_self_replicating_policy_identity(self):
        # 候选策略等于记录策略且 propensity 恒为 0.5：SNIPS 恰为样本均值，IPW 恰为 2 倍均值。
        rows = [make_row(f'train-{i}', 0, action='X', propensity=0.5, revenue=float(i))
                for i in range(4)]
        rows += [make_row(f'eval-{i}', 1, action='X', propensity=0.5, revenue=float(10 + i))
                 for i in range(4)]
        decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=None)
        mean = statistics.fmean([10.0, 11.0, 12.0, 13.0])
        self.assertAlmostEqual(decision['evaluation']['estimators']['snips']['value'], mean, places=12)
        self.assertAlmostEqual(decision['evaluation']['estimators']['ipw']['value'], 2 * mean, places=12)


class KnownRandomPolicyTest(unittest.TestCase):
    """已知随机策略的合成实验：两个加权估计与 DR 的蒙特卡洛无偏性核对。"""

    def test_estimators_center_on_true_value(self):
        rng = random.Random(20261112)
        props = [0.2, 0.3, 0.5]
        means = [10.0, 20.0, 30.0]
        reps, per_week = 300, 200
        collected = {name: [] for name in ('ipw', 'snips', 'dr')}
        for rep in range(reps):
            rows = []
            for i in range(per_week * 3):
                action = rng.choices([0, 1, 2], weights=props)[0]
                net = means[action] + rng.gauss(0.0, 1.0)
                rows.append(make_row(f'{rep}-{i}', i % 3, action=f'A{action}',
                                     candidate='A2', propensity=props[action],
                                     revenue=net + 1.0, cost=1.0))
            decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=None)
            self.assertEqual(decision['evaluation']['excluded']['insufficient_history'], per_week)
            for name in collected:
                collected[name].append(decision['evaluation']['estimators'][name]['value'])
        # 候选策略恒取 A2，真实价值 30；固定随机种子下结果确定。
        self.assertAlmostEqual(statistics.fmean(collected['ipw']), 30.0, delta=0.4)
        self.assertAlmostEqual(statistics.fmean(collected['snips']), 30.0, delta=0.4)
        self.assertAlmostEqual(statistics.fmean(collected['dr']), 30.0, delta=0.25)


class PropensityEdgeTest(unittest.TestCase):
    """极小概率触发裁剪披露；零概率与不合法概率分别给出原因。"""

    def test_tiny_zero_and_invalid_propensity(self):
        rows = [
            make_row('w0-x', 0, action='X', propensity=0.5, revenue=5.0),
            make_row('w0-y', 0, action='Y', propensity=0.5, revenue=1.0),
            make_row('ok', 1, action='X', candidate='X', propensity=0.5, revenue=10.0),
            make_row('tiny', 1, action='X', candidate='X', propensity=1e-12, revenue=4.0),
            make_row('zero', 1, action='X', candidate='X', propensity=0.0, revenue=1.0),
            make_row('neg', 1, action='X', candidate='X', propensity=-0.1, revenue=1.0),
            make_row('big', 1, action='X', candidate='X', propensity=1.5, revenue=1.0),
        ]
        decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=5.0)
        ev = decision['evaluation']
        self.assertEqual(ev['excluded']['zero_propensity'], 1)
        self.assertEqual(ev['excluded']['invalid_propensity'], 2)
        self.assertEqual(ev['rows_evaluable'], 2)
        weights = ev['weights']
        self.assertEqual(weights['rows_clipped'], 1)
        self.assertEqual(weights['max_weight'], 1e12)
        self.assertGreater(weights['clipped_weight_share'], 0.99)
        # 裁剪后权重 2 与 5：IPW = (2*10 + 5*4) / 2
        self.assertAlmostEqual(ev['estimators']['ipw']['value'], 20.0, places=9)
        self.assertAlmostEqual(weights['ess'], 49.0 / 29.0, places=9)

    def test_propensity_one_is_valid(self):
        rows = [make_row('w0', 0, action='X', propensity=1.0, revenue=5.0),
                make_row('e1', 1, action='X', candidate='X', propensity=1.0, revenue=8.0)]
        decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=None)
        self.assertEqual(decision['evaluation']['rows_evaluable'], 1)
        self.assertEqual(decision['evaluation']['estimators']['ipw']['value'], 8.0)
        self.assertIsNone(decision['evaluation']['estimators']['ipw']['ci'])
        self.assertIn('可评估行不足两行', warning_texts(decision))


class CoverageTest(unittest.TestCase):
    """候选动作无历史覆盖与全无匹配动作时的披露行为。"""

    def test_covered_but_never_matched(self):
        rows = [
            make_row('w0-x', 0, action='X', propensity=0.5, revenue=5.0),
            make_row('w0-y', 0, action='Y', propensity=0.5, revenue=1.0),
            make_row('e1', 1, action='X', candidate='Y', propensity=0.5, revenue=6.0),
            make_row('e2', 1, action='Y', candidate='X', propensity=0.5, revenue=4.0),
        ]
        decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=None)
        ev = decision['evaluation']
        self.assertEqual(ev['rows_evaluable'], 2)
        self.assertEqual(ev['matched_rows'], 0)
        self.assertEqual(ev['weights']['ess'], 0.0)
        self.assertEqual(ev['estimators']['ipw']['value'], 0.0)
        self.assertIsNone(ev['estimators']['snips']['value'])
        self.assertIsNotNone(ev['estimators']['dr']['value'])
        self.assertIn('自归一化估计不可用', warning_texts(decision))

    def test_candidate_without_any_history(self):
        rows = [make_row('w0', 0, action='X', propensity=1.0, revenue=5.0)]
        rows += [make_row(f'e{i}', 1, action='X', candidate='Z', propensity=1.0,
                          revenue=float(i)) for i in range(3)]
        decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=None)
        ev = decision['evaluation']
        self.assertEqual(ev['rows_evaluable'], 0)
        self.assertEqual(ev['excluded']['no_action_coverage'], 3)
        for name in ('ipw', 'snips', 'dr'):
            self.assertIsNone(ev['estimators'][name]['value'])
        self.assertEqual(decision['segments']['G']['coverage'], 0.0)
        self.assertEqual(decision['segments']['G']['excluded']['no_action_coverage'], 3)
        self.assertIn('没有可评估行', warning_texts(decision))

    def test_coverage_is_segment_specific(self):
        rows = [
            make_row('w0-g', 0, segment='G', action='X', propensity=1.0, revenue=5.0),
            make_row('w0-h', 0, segment='H', action='Y', propensity=1.0, revenue=5.0),
            make_row('e-g', 1, segment='G', action='X', candidate='X', propensity=1.0, revenue=6.0),
            make_row('e-h', 1, segment='H', action='Y', candidate='X', propensity=1.0, revenue=6.0),
        ]
        decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=None)
        ev = decision['evaluation']
        self.assertEqual(ev['rows_evaluable'], 1)
        self.assertEqual(ev['excluded']['no_action_coverage'], 1)
        self.assertEqual(decision['segments']['H']['excluded']['no_action_coverage'], 1)


class TimeLeakageBaitTest(unittest.TestCase):
    """时间泄漏诱饵：未来周的极端数据不得改变既往折的任何内容。"""

    def base_rows(self):
        rows = []
        for week in range(6):
            rows.append(make_row(f'x-{week}', week, action='X', candidate='X',
                                 propensity=0.6, revenue=10.0 + week))
            rows.append(make_row(f'y-{week}', week, action='Y', candidate='X',
                                 propensity=0.4, revenue=4.0 + week))
        return rows

    def bait_rows(self):
        return [
            make_row('bait-6', 6, segment='SB', action='AB', candidate='AB',
                     propensity=0.001, revenue=1e6),
            make_row('bait-7', 7, segment='SB', action='AB', candidate='AB',
                     propensity=0.001, revenue=1e6),
            make_row('bait-g-6', 6, action='X', candidate='X', propensity=0.6, revenue=-1e6),
            make_row('bait-g-7', 7, action='X', candidate='X', propensity=0.6, revenue=-1e6),
        ]

    def test_future_rows_do_not_change_earlier_folds(self):
        base, bait = self.base_rows(), self.bait_rows()
        grid = [0.01, 1.0, 100.0]
        full = evaluate(base + bait, train_models(base + bait, 1, grid), clip=10.0)
        truncated = evaluate(base, train_models(base, 1, grid), clip=10.0)
        prefix = [fold for fold in full['folds'] if fold['validation_week'] <= 5]
        self.assertEqual(truncated['folds'], prefix)
        self.assertEqual(len(prefix), 5)

    def test_fold_encoding_only_sees_training_weeks(self):
        rows = self.base_rows() + self.bait_rows()
        decision = evaluate(rows, train_models(rows, 1, [0.01, 1.0, 100.0]), clip=10.0)
        for fold in decision['folds']:
            seen_segments = {row['segment'] for row in rows if row['week'] in fold['train_weeks']}
            seen_actions = {row['action'] for row in rows if row['week'] in fold['train_weeks']}
            self.assertLessEqual(set(fold['model']['segment_levels']), seen_segments)
            self.assertLessEqual(set(fold['model']['action_levels']), seen_actions)
        # 诱饵行的 1/0.001 权重在后续折被裁剪并披露
        self.assertGreater(decision['evaluation']['weights']['rows_clipped'], 0)

    def test_model_selection_uses_only_training_period(self):
        # 未来周加入强非线性诱饵后，既往折的惩罚系数与内部评分保持不变
        base = self.base_rows()
        bait = [make_row(f'nb-{week}', week, action='X', candidate='X',
                         propensity=0.5, revenue=(-1.0) ** week * 1e4) for week in (6, 7, 8)]
        grid = [0.01, 1.0, 100.0]
        full = train_models(base + bait, 1, grid)
        truncated = train_models(base, 1, grid)
        picks_full = [(f['validation_week'], f['model']['penalty'], f['model']['scores'])
                      for f in full['folds'] if f['validation_week'] <= 5]
        picks_trunc = [(f['validation_week'], f['model']['penalty'], f['model']['scores'])
                       for f in truncated['folds']]
        self.assertEqual(picks_full, picks_trunc)


class RowOrderTest(unittest.TestCase):
    """按原始行序重排不会改变折的含义与任何输出。"""

    @classmethod
    def setUpClass(cls):
        cls.records = load_records('data/observations.csv')

    def test_shuffled_input_gives_identical_outputs(self):
        shuffled = copy.deepcopy(self.records)
        random.Random(7).shuffle(shuffled)
        models_a = train_models(self.records, 1, list(DEFAULT_PENALTIES))
        models_b = train_models(shuffled, 1, list(DEFAULT_PENALTIES))
        self.assertEqual(models_a, models_b)
        decision_a = evaluate(self.records, models_a, clip=10.0)
        decision_b = evaluate(shuffled, models_b, clip=10.0)
        self.assertEqual(decision_a, decision_b)
        self.assertEqual(json.dumps(decision_a, sort_keys=True),
                         json.dumps(decision_b, sort_keys=True))


class FixtureDecisionTest(unittest.TestCase):
    """夹具数据上的完整决策文件：总体增益、分客群覆盖、区间与排除质量同处一份文件。"""

    @classmethod
    def setUpClass(cls):
        cls.records = load_records('data/observations.csv')
        cls.decision = evaluate(cls.records, train_models(cls.records, 1, list(DEFAULT_PENALTIES)),
                                clip=10.0)

    def test_independent_crosscheck_numbers(self):
        ev = self.decision['evaluation']
        self.assertEqual(ev['rows_total'], 288)
        self.assertEqual(ev['rows_evaluable'], FIXTURE_EVALUABLE)
        self.assertEqual(ev['matched_rows'], FIXTURE_MATCHED)
        self.assertAlmostEqual(ev['estimators']['ipw']['value'], FIXTURE_IPW, places=9)
        self.assertAlmostEqual(ev['estimators']['snips']['value'], FIXTURE_SNIPS, places=9)
        self.assertAlmostEqual(ev['baseline'], FIXTURE_BASELINE, places=9)
        self.assertAlmostEqual(ev['weights']['ess'], FIXTURE_ESS, places=9)

    def test_decision_file_is_complete(self):
        decision = self.decision
        ev = decision['evaluation']
        self.assertEqual(ev['excluded']['insufficient_history'], 24)
        for reason in ('invalid_propensity', 'zero_propensity', 'no_action_coverage'):
            self.assertIn(reason, ev['excluded'])
        for name in ('ipw', 'snips', 'dr'):
            est = ev['estimators'][name]
            self.assertIsNotNone(est['gain'])
            self.assertIsNotNone(est['ci'])
            self.assertIsNotNone(est['gain_ci'])
        self.assertEqual(set(decision['segments']), {'S0', 'S1', 'S2', 'S3'})
        for segment in decision['segments'].values():
            self.assertEqual(segment['rows'], 72)
            self.assertEqual(segment['evaluable'], 66)
            self.assertAlmostEqual(segment['coverage'], 66 / 72, places=12)
            self.assertIn('no_action_coverage', segment['excluded'])
        self.assertEqual(len(decision['folds']), 11)
        self.assertEqual([fold['validation_week'] for fold in decision['folds']],
                         list(range(1, 12)))
        self.assertEqual(decision['evaluation']['weights']['max_weight'], 5.0)
        self.assertEqual(decision['evaluation']['weights']['rows_clipped'], 0)


class DegenerateDataTest(unittest.TestCase):
    """单周与空数据：全部行不可评估时仍输出完整决策文件。"""

    def test_single_week_all_insufficient_history(self):
        rows = [make_row(f'r{i}', 0, action='X', candidate='X', propensity=0.5,
                         revenue=float(i)) for i in range(5)]
        decision = evaluate(rows, train_models(rows, 1, [1.0]), clip=None)
        ev = decision['evaluation']
        self.assertEqual(ev['rows_evaluable'], 0)
        self.assertEqual(ev['excluded']['insufficient_history'], 5)
        for name in ('ipw', 'snips', 'dr'):
            self.assertIsNone(ev['estimators'][name]['value'])
        self.assertIn('没有可用验证折', warning_texts(decision))

    def test_empty_dataset(self):
        decision = evaluate([], train_models([], 1, [1.0]), clip=None)
        self.assertEqual(decision['rows'], 0)
        self.assertEqual(decision['folds'], [])
        self.assertIsNone(decision['evaluation']['coverage'])
        self.assertIsNone(decision['evaluation']['estimators']['ipw']['value'])

    def test_min_train_weeks_defers_evaluation(self):
        rows = []
        for week in range(4):
            rows.append(make_row(f'x-{week}', week, action='X', candidate='X',
                                 propensity=0.5, revenue=10.0))
            rows.append(make_row(f'y-{week}', week, action='Y', candidate='X',
                                 propensity=0.5, revenue=4.0))
        decision = evaluate(rows, train_models(rows, 2, [1.0]), clip=None)
        ev = decision['evaluation']
        self.assertEqual(ev['excluded']['insufficient_history'], 4)
        self.assertEqual(ev['rows_evaluable'], 4)
        self.assertEqual([fold['validation_week'] for fold in decision['folds']], [2, 3])


class CliRoundTripTest(unittest.TestCase):
    """本地重复训练与加载：模型可保存、可校验、可复现。"""

    def run_cli(self, *argv):
        return subprocess.run([sys.executable, '-m', 'analysis_app', *argv],
                              text=True, capture_output=True)

    def write_csv(self, directory, rows, name='input.csv'):
        path = Path(directory) / name
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_train_then_load_reproduces_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            model_a = Path(directory) / 'model-a.json'
            model_b = Path(directory) / 'model-b.json'
            decision_load = Path(directory) / 'decision-load.json'
            decision_fresh = Path(directory) / 'decision-fresh.json'
            run = self.run_cli('train', '--model', str(model_a))
            self.assertEqual(run.returncode, 0, run.stderr)
            run = self.run_cli('train', '--model', str(model_b))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(model_a.read_bytes(), model_b.read_bytes())
            run = self.run_cli('evaluate', '--model-in', str(model_a),
                               '--output', str(decision_load))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['model_source'], 'load')
            run = self.run_cli('evaluate', '--model-out', str(model_b),
                               '--output', str(decision_fresh))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['model_source'], 'train')
            load_doc = json.loads(decision_load.read_text())
            fresh_doc = json.loads(decision_fresh.read_text())
            self.assertEqual(load_doc.pop('model_source'), 'load')
            self.assertEqual(fresh_doc.pop('model_source'), 'train')
            self.assertEqual(load_doc, fresh_doc)
            self.assertEqual(model_a.read_bytes(), model_b.read_bytes())

    def test_load_rejects_changed_data(self):
        records = load_records('data/observations.csv')
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'model.json'
            run = self.run_cli('train', '--model', str(model_path))
            self.assertEqual(run.returncode, 0, run.stderr)
            changed = copy.deepcopy(records)
            changed[10]['revenue'] += 1.0
            input_path = self.write_csv(directory, changed)
            output_path = Path(directory) / 'decision.json'
            run = self.run_cli('evaluate', '--input', str(input_path),
                               '--model-in', str(model_path), '--output', str(output_path))
            self.assertEqual(run.returncode, 2)
            self.assertIn('指纹', run.stderr)
            self.assertFalse(output_path.exists())

    def test_load_rejects_conflicting_config(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'model.json'
            run = self.run_cli('train', '--model', str(model_path))
            self.assertEqual(run.returncode, 0, run.stderr)
            run = self.run_cli('evaluate', '--model-in', str(model_path),
                               '--min-train-weeks', '2')
            self.assertEqual(run.returncode, 2)
            self.assertIn('不一致', run.stderr)

    def test_evaluate_failure_has_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / 'decision.json'
            run = self.run_cli('evaluate', '--input', str(Path(directory) / 'missing.csv'),
                               '--output', str(output_path))
            self.assertEqual(run.returncode, 2)
            self.assertFalse(output_path.exists())

    def test_invalid_clip_rejected(self):
        run = self.run_cli('evaluate', '--clip', '-1')
        self.assertEqual(run.returncode, 2)
        self.assertIn('裁剪阈值', run.stderr)


if __name__ == '__main__':
    unittest.main()
