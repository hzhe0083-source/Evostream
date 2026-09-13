# fabri_cache_acceptance

小型、独立的验收测试报告检查工具（Python 标准库实现）。
供主模型在服务器端快速检查第一次缓存与训练验证生成的 JSON 报告。

## 关键免责声明与边界

> **重要**：本工具仅用于**验证报告的合同格式与约束有效性（Contract / Schema Validator）**，验证通过**不能**作为实际模型代码在物理硬件/GPU 上真实执行的证据。实际执行真实性仍须由主模型在目标环境中亲自验证（日志、Checkpoints、进程与环境指标）。

## 命令行用法 (CLI)

```bash
python -m fabri_cache_acceptance.check REPORT.json
```

- 退出码 `0`：校验通过。
- 退出码 `2`：校验失败（缺失必填字段、类型错误、数值非法或未达标），并在 `stderr` 打印具体失败字段与原因。

## JSON 报告合同约定 (Contract Specification)

报告必须为 JSON Object。允许包含额外的非保留字段（工具会自动忽略）。

| 字段路径 | 类型 | 合同约束 |
|---|---|---|
| `official_single_frame.passed` | boolean | 必须为 `true` |
| `official_single_frame.max_abs_error` | number | 必须为有限数值，且 `max_abs_error >= 0` |
| `cached_full.passed` | boolean | 必须为 `true` |
| `cached_full.max_abs_error` | number | 必须为有限数值，且 `max_abs_error >= 0` |
| `window_rebuild.passed` | boolean | 必须为 `true` |
| `window_rebuild.max_abs_error` | number | 必须为有限数值，且 `max_abs_error >= 0` |
| `reset.passed` | boolean | 必须为 `true` |
| `vision_reuse.passed` | boolean | 必须为 `true` |
| `strict_weight_load.passed` | boolean | 必须为 `true` |
| `training.loss` | number | 必须为有限数值，且 `loss >= 0` |
| `training.backbone_has_grad` | boolean | 必须为 `false`（冻结 backbone） |
| `training.action_head_updated` | boolean | 必须为 `true`（权重正常更新） |
| `padded_actions_zero` | boolean | 必须为 `true` |

### 类型严格性说明

在 Python JSON 解析中，`bool` 是 `int` 的子类（`isinstance(True, int) == True`）。
本校验器执行**严格类型判定**：
- `number` 字段传入布尔值（如 `true`）会被拒绝，绝不能冒充数字；
- `NaN` / `Infinity` / `-Infinity` 等非有限数值均会被拒绝；
- 负数（如负的 error 或 loss）会被拒绝。

## 运行单元测试

```bash
python -m unittest discover -s fabri_cache_acceptance/tests -p "test_*.py"
```
