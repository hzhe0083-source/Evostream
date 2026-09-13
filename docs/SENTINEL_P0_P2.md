# 浅层哨兵：P0–P2 首批基础接口

## 交付边界

本批保留 `moss_action_v6` 策略权重布局、固定周期 planner、动作头与默认评测行为。
新增的是执行命令日志、真实分段视觉接口、离线转移训练和 **离线 shadow replay**。
**尚未接入在线 SentinelWorker，也没有事件中断、等待控制器、恢复 SFT 或性能收益结论。**
现有 async 耗尽时重复旧命令的 baseline 行为仍在，不能当作安全保持。

## P0：最终命令日志

现有 `evaluate.py` 命令加 `--action-ledger-path /absolute/path/actions.json` 即可导出每个 trial 的日志。
记录在夹爪处理之后、`env.step` 之前复制的命令，成功返回后写入有界账本。
记录包含 command_id、step_index、monotonic timestamp、wall-clock env.step duration、fallback 标识。
baseline/ensemble 无可靠单一计划归属，plan_id 为 null，不用并发 planner 计数冒充来源。

- settling 阶段明确排除；trial 内第一个控制步骤编号为 0。
- `duration` 是 **env.step 的墙钟耗时**，不是仿真动作物理时长。
- 导出 trial 的 `control_interval` 与 `step_index` 用于仿真时间对齐。墙钟日志中的调度空隙不自动填成动作持续区间。
- `query_interval` 检测覆盖缺口和容量截断；不能把缺少记录解释成零动作。
- 当前日志不含图像/状态/已接受任务表征，不能单独直接作为完整监控训练数据。

## P1：原生视觉前缀与续算

`vision_split.SplitVision` 是普通辅助类，不重复注册或复制视觉模型权重。
实现参照官方源码固定 revision：
`25e81cb952d5f353a5690f2c1ea09a725815df80`，
[modeling_moss_vl.py](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime/blob/25e81cb952d5f353a5690f2c1ea09a725815df80/modeling_moss_vl.py)。
实际模型源码的 SHA256 由 adapter 记录；结构接口检查并不保证任意未来模型版本等价。

```python
from vision_split import SplitVision

policy.eval()
adapter = SplitVision(policy.backbone.moss)
# frame_inputs 来自原生 processor，并已放到 policy 的设备上。
packet = adapter.encode_prefix(
    frame_inputs['pixel_values'], frame_inputs['grid_thw'],
    stop_after=4, frame_id=frame_id, observation_time=timestamp,
    media_nums_per_sample=frame_inputs.get('media_nums_per_sample'),
)
z = adapter.spatial_features(packet, side=4)  # [16, visual_hidden_dim]
# 到这里：未运行后半 ViT、merger 或 Language Model，未修改 Session KV。
encoded = adapter.encode_suffix(packet)  # packet 只允许消费一次
session.append_preencoded_frame(
    frame_inputs, encoded, frame_id=frame_id,
    timestamp=timestamp, origin_timestamp=episode_origin,
)
```

调用方须使用同一 `episode_origin` 构造 `realtime_frame_segment(timestamp - episode_origin)`，
并负责保留/释放 packet；当前不提供后台补算或在线缓冲 worker。只接受单帧输入。
续算保留原生位置编码、cu_seqlens、此前 DeepStack 特征，调用原生 merger 和 packed-to-batch。
`spatial_features` 把原生 merge-block patch 顺序还原为二维网格，再作固定区域池化；没有可训练投影。

`encode_complete_vision` 与 `append_encoded_vision` 将全量编码和 KV 追加分离；旧 `append_stream_frame` 为兼容包装。
`EncodedVision` 必须携带源 MOSS 核心实例引用 `source_core`（非注册对象引用身份）与可选诊断指纹 `source_fingerprint`。
`append_encoded_vision` 在执行任何 cache 裁切或元数据修改前，强制核验 `encoded.source_core is self.moss`；即使同架构同权重的异核实例也会在修改前被严格拒绝。
该机制建立进程内不可变权重的推理合同；模型就地修改权重而不更新实例是不支持的行为。
`session.plan()` 额外返回 detached `ActionChunk.context`（同次 Query hidden 均值），不自动发布为已接受参考。
该摘要的接受时刻、来源版本绑定仍由后续在线执行器负责；不能把生成完成当作计划已接受。

## P2：离线录制合同

输入是可信本地 `torch.save` 文件：`{'metadata': metadata.to_dict(), 'recordings': [...]}`。
加载使用 Python pickle，**不要加载不可信文件**。
元数据 `TransitionContractMetadata` 必须绑定策略哈希、视觉源码哈希、预处理哈希、浅层权重哈希、退出层、
空间特征规格、帧/动作时间合同和任务摘要提取方式。
预测器自己的 state/config 哈希单独保存，不覆盖浅层视觉权重身份。

每个 recording 必须包含：

```text
episode_id                 全局唯一 episode 标识（建议含 task/camera）
timestamps                 [T] float64，真实观测时间，严格递增
frozen_features            [T, regions, feature_dim]
states / velocities        [T, state_dim]，开始时刻可用状态与速度
actions                    [A, action_dim]，最终执行命令
action_timestamps          [A] float64，动作区间开始
action_end_timestamps      [A] float64，动作区间结束
context                    固定 [C] 或稀疏/逐帧更新 [N, C]（C_exp==1 时支持 [N]）
context_timestamps         标量或 [N]，非减时间戳；同一时间戳仅允许相同重复值（兼容逐帧保持），冲突值拒绝；检索时按 searchsorted(right=True) 取最新 <= t0
```

开始时刻的摘要必须仅来自此前已接受规划，并一直保持到下次接受；提取器检查时间但不能验证伪造的来源。
支持固定 [C] 与稀疏更新 [N, C]（独立于帧数 T），时间戳必须非减；同一时刻重复输入必须值完全一致，冲突即拒绝；因果检索使用最新 <= t0，无先验摘要拒绝。
动作必须严格按真实端点相交并覆盖观测转移区间，不允许从原预测 chunk 或默认 `action_offset=1` 猜测。
跨边界命令按交集时长加权，微小边界命令纳入计算并保留为区间末命令。当前使用 mean+last 摘要，**不能区分所有动作排列**，是首版模型限制。
按 episode 划分训练/验证/测试，不能打散相邻帧后随机分割。

```bash
python train_sentinel.py extract --recordings-path train_recordings.pt --output train_transitions.pt --spans 1 2 4
python train_sentinel.py extract --recordings-path val_recordings.pt --output val_transitions.pt --spans 1 2 4
python train_sentinel.py extract --recordings-path test_recordings.pt --output test_transitions.pt --spans 1 2 4
python train_sentinel.py train --train-data train_transitions.pt --val-data val_transitions.pt --output sentinel.pt
python train_sentinel.py calibrate --model-checkpoint sentinel.pt --val-data val_transitions.pt --output-profile profile.json
# 默认 held_out 模式严格拒绝训练集与校准集交集，必须使用独立测试集：
python train_sentinel.py shadow --dataset test_transitions.pt --model-checkpoint sentinel.pt --profile profile.json --output-log shadow.jsonl
# 若需在重叠数据上做诊断回放，必须显式附加 --allow-data-overlap，此时日志打上 diagnostic 标记与 overlap 列表：
python train_sentinel.py shadow --dataset val_transitions.pt --model-checkpoint sentinel.pt --profile profile.json --output-log shadow_diag.jsonl --allow-data-overlap
```

`extract` **只从已提取的冻结特征构造转移**，不是原始图片/HDF5 特征提取器。
`shadow` 为离线多跨度回放，不修改控制器。正常数据的阈值校准不代表关键异常召回已经验证。
训练采用固定原始特征尺度 Huber；normalizer 只在训练集拟合，区域坐标固定，目标端没有可收缩的学习投影。

## 验收记录与限制

本机环境：PyTorch 2.10.0+cu128、Transformers 5.2.0、RTX 3080 Laptop 16 GiB。
未找到本地 MOSS checkpoint 或训练后的 v6 策略/恢复数据。
直接导入上述官方源码在该环境失败：`ImportError: cannot import name 'OutputRecorder' from transformers.utils.generic`。
没有改全局依赖、HF 缓存或下载权重绕过这一问题。

无权重测试命令：

```bash
python -m pytest test_action_ledger.py test_sentinel.py test_vision_split.py -q
python -m pytest test_contract.py -q
```

本批新增测试最终结果：**46 passed**；compileall 和 git diff --check 通过。
涵盖 EncodedVision 实例身份绑定（同模型多 adapter 接收、异核同结构同参数拒绝且无 cache 污染、缺失所有者拒绝）、校准集 calibration_episodes 强校验、shadow held_out/diagnostic 模式隔离、上下文稀疏更新 [N, C]/[N]/固定 [C]、非减时间戳与完全重复兼容/冲突值拒绝/严格因果搜索、动作区间严格端点相交与微小边界命令贡献、状态速度及特征维度契约校验。
Gemini 初版与本轮修复由主模型亲自复核代码与测试（本轮无独立 review agent 介入）。
视觉测试使用 tiny native-like fixtures，不是官方真实视觉权重；覆盖 2/4/8 层早退计数、DeepStack 续算、空间顺序、预编码追加和状态零污染。
真实 split parity、同选帧集合的真实流式 parity、检测误报/漏报、GPU p99 延迟、在线监控尾延迟和闭环成功率均未验收。

原有 `test_batched_vs_unbatched_query_parity` 对随机初始化敏感：固定 `torch.manual_seed(0)` 跑原测试套件，
在未修改的 `b91fdd2` 与本批代码上均出现相同误差 `0.000511525`，超过 `atol=0.0005`。
未放宽容差、删除或跳过此原测试。初次未固定随机种子的原套件曾 28/28 通过，不能据此声称稳定通过。
