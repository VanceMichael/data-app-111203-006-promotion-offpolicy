import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .io import load_records
from .model import DEFAULT_PENALTIES, train_models, verify_models
from .ope import evaluate
from .summary import summarize

DEFAULT_PENALTIES_TEXT = ','.join(str(p) for p in DEFAULT_PENALTIES)


def save_json(path, result):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.analysis-', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_penalties(text):
    try:
        values = [float(item) for item in text.split(',') if item.strip()]
    except ValueError:
        raise ValueError('惩罚系数格式不符：' + text) from None
    if not values or any(value < 0 for value in values):
        raise ValueError('惩罚系数需为非负数且至少一个')
    return sorted(values)


def _cmd_train(args):
    if args.min_train_weeks < 1:
        raise ValueError('min_train_weeks 至少为 1')
    records = load_records(args.input)
    blob = train_models(records, args.min_train_weeks, _parse_penalties(args.penalties))
    save_json(args.model, blob)
    return blob


def _cmd_evaluate(args):
    if args.clip < 0:
        raise ValueError('裁剪阈值需为非负数（0 表示不裁剪）')
    if not 0 < args.confidence < 1:
        raise ValueError('置信水平需在 (0,1) 内')
    records = load_records(args.input)
    if args.model_in:
        try:
            blob = json.loads(args.model_in.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            raise ValueError('模型文件不是合法 JSON') from None
        verify_models(records, blob)
        config = blob['config']
        if args.min_train_weeks is not None and args.min_train_weeks != config['min_train_weeks']:
            raise ValueError('参数与模型文件不一致：min_train_weeks')
        if args.penalties is not None and _parse_penalties(args.penalties) != config['penalties']:
            raise ValueError('参数与模型文件不一致：penalties')
        source = 'load'
    else:
        min_train_weeks = args.min_train_weeks if args.min_train_weeks is not None else 1
        if min_train_weeks < 1:
            raise ValueError('min_train_weeks 至少为 1')
        penalties = (_parse_penalties(args.penalties) if args.penalties is not None
                     else list(DEFAULT_PENALTIES))
        blob = train_models(records, min_train_weeks, penalties)
        source = 'train'
    if args.model_out:
        save_json(args.model_out, blob)
    decision = evaluate(records, blob,
                        clip=(None if args.clip == 0 else args.clip),
                        confidence=args.confidence)
    decision['model_source'] = source
    if args.output:
        save_json(args.output, decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description='本地观测数据分析')
    commands = parser.add_subparsers(dest='command', required=True)
    summary = commands.add_parser('summary', help='按业务维度核对观测口径')
    summary.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    summary.add_argument('--output', type=Path)
    train = commands.add_parser('train', help='前向切分交叉拟合训练收益模型并保存')
    train.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    train.add_argument('--model', type=Path, required=True)
    train.add_argument('--min-train-weeks', type=int, default=1)
    train.add_argument('--penalties', type=str, default=DEFAULT_PENALTIES_TEXT)
    evaluate_cmd = commands.add_parser('evaluate', help='离线策略评估并输出决策文件')
    evaluate_cmd.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    evaluate_cmd.add_argument('--output', type=Path)
    evaluate_cmd.add_argument('--model-in', type=Path, help='加载已训练模型（校验折结构与训练指纹）')
    evaluate_cmd.add_argument('--model-out', type=Path, help='把本次使用的模型另存到指定路径')
    evaluate_cmd.add_argument('--clip', type=float, default=10.0, help='权重裁剪阈值，0 表示不裁剪')
    evaluate_cmd.add_argument('--confidence', type=float, default=0.95)
    evaluate_cmd.add_argument('--min-train-weeks', type=int)
    evaluate_cmd.add_argument('--penalties', type=str)
    args = parser.parse_args()
    try:
        if args.command == 'summary':
            result = summarize(load_records(args.input))
            if args.output:
                save_json(args.output, result)
        elif args.command == 'train':
            result = _cmd_train(args)
        else:
            result = _cmd_evaluate(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
