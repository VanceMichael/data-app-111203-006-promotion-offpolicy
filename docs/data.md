# 数据契约

营销分析记录历史优惠动作、行为概率与净收益，提供按动作的成本和收入汇总。

主测量 `revenue` 的单位为元，主体字段是 `session`，核算维度是 `action`。`observation_id` 是文件内唯一观测编号，主体允许重复观测。数值只接受有限值；空单元格解析为 None，不能直接当成零。额外列不进入当前核算。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `observation_id` | str | 必填 |
| `session` | str | 必填 |
| `action` | str | 必填 |
| `revenue` | float | 必填 |
| `cost` | float | 允许空值，空值不是零 |
| `propensity` | float | 允许空值，空值不是零 |
| `week` | int | 允许空值，空值不是零 |
| `segment` | str | 允许空值，空值不是零 |
| `candidate` | str | 允许空值，空值不是零 |

时间字段 day、week、tick、hour、minute、cycle、start_day、reported_day、known_at、visible_at、opened、observed 均以各字段名所述的相对时间单位表示整数；不同字段并不必然代表同一种时间。缺少的登记码与浓度等信息保持为空。示例只表达产品数据结构，不能代替实际调查资料。
