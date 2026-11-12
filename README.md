# promotion-offpolicy

营销分析记录历史优惠动作、行为概率与净收益，提供按动作的成本和收入汇总，并对候选优惠策略做离线策略评估（OPE）：候选策略在真正投放前先接受支持域检查，再给出逆概率、自归一化与双稳健三种估计的净增益。

## 使用

运行需要 Python 3.11 或更新版本，没有第三方运行依赖。数据在本地处理，不请求外部接口。

```bash
python3 -m analysis_app summary
python3 -m analysis_app summary --input data/observations.csv --output output/summary.json
```

离线策略评估（方法细节见 `docs/ope.md`）：

```bash
# 前向切分交叉拟合训练收益模型并保存；重复训练产物字节一致
python3 -m analysis_app train --model output/model.json
# 加载已保存模型输出决策文件（加载前校验折结构与训练指纹）
python3 -m analysis_app evaluate --model-in output/model.json --output output/decision.json
# 现场训练并同时保存模型也可以
python3 -m analysis_app evaluate --model-out output/model.json --output output/decision.json \
    --clip 10 --confidence 0.95
```

决策文件同时包含总体净增益与区间、分客群覆盖、按原因分布的排除计数、权重裁剪与有效样本数披露；估计不可用的部分输出 null 并保留计数，不会只展示好看的部分。`--clip 0` 关闭权重裁剪，`--min-train-weeks` 与 `--penalties` 控制折计划与模型选择网格，加载模型时这些参数以模型文件为准。

标准输出是完整 JSON，输出文件使用临时文件原子替换；失败退出码为 2，不覆盖既有结果。空数据集保留零计数与空分组，缺失数值不进入数值分母。数据指纹由按 observation_id 排序后的解析记录计算；训练与评估统一按 (week, observation_id) 排序，输入行序不影响结果。

## 验证与容器

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q analysis_app
docker build -t promotion-offpolicy .
docker run --rm promotion-offpolicy
```

容器内也可执行 `python -m unittest discover -s tests -v`。镜像构建阶段取得 Python 镜像，产品启动、测试和演示均无需下载。小规模样例的完整单元测试和 summary 可以连续执行，不需要等待定时任务。

## 代码与数据

- `analysis_app/io.py`：有类型的 CSV 读取、完整性校验及内容指纹。
- `analysis_app/schema.py`：字段类型、主体与聚合维度。
- `analysis_app/domain.py`：当前业务核算函数。
- `analysis_app/summary.py`：确定性的分组与数值口径。
- `analysis_app/model.py`：前向切分交叉拟合的岭回归收益模型，可保存与校验。
- `analysis_app/ope.py`：支持域检查、IPW/SNIPS/DR 估计、裁剪披露与区间。
- `analysis_app/cli.py`：本地命令及原子输出。
- `docs/data.md`：数据契约。
- `docs/ope.md`：离线策略评估方法说明。

演示数据全部为本地合成，不对应现实机构、设备或人员；不能用来推断真实业务效果。
