# MOSS-Action VLA

**MOSS-Action 将指令与增量视觉历史保存在持久多模态前缀中，每个规划周期从该前缀临时分叉出一组因果 Action Query，利用 MOSS 内部的文本 Self-Attention 与视觉 Cross-Attention 一次生成连续 micro-chunk，输出后立即回滚 Query KV，只保留干净的感知上下文供下一帧继续追加。**

保留 MOSS-VL 的完整视觉编码器和统一解码层 `0..23`，删除层 `24..47` 与语言输出头。指令只 prefill 一次；每个新帧单独编码并追加到 episode 级原生视觉 K/V cache，无需反复重编码历史画面。

动作生成采用 **Streaming Action Query Decoder**：$K$ 个因果动作查询嵌入进入 24 层 MOSS 骨干网络，通过内部文本注意力读指令与状态速度条件，通过 6 层原生 Gated Cross-Attention 读增量视觉 KV，末端由轻量两层 MLP 直接回归 $K \times 7$ 连续动作块。生成完成后临时查询的 KV 被立即回滚截断，保证已执行的动作记忆不被未执行的旧未来污染。

## 已实现合同

- 24 个保留层总计包含 18 个 Self-Attention 层和 6 个原生 Gated Cross-Attention 层；后者位于 `2, 6, 10, 14, 18, 22`。
- 训练采用单个 planning time 样本，将增量帧序列、状态速度 $\dot{s}_t$ 与 $K$ 个 Action Query 一起送入 24 层前向并全参数反向传播；部署时指令与历史视觉以增量 KV 形式缓存，每次规划时以相同的因果注意力与跨注意力计算 Action Query，输出数值与全量前向对齐。
- Action Query 具有 **Ephemeral 回滚保证**：每次规划完成后，`_ephemeral_cache` 只裁切 18 个 Self-Attention 层中的临时 Query 尾部，保留 6 个 Cross-Attention 层中的全部视觉 KV，感知上下文对后续帧零污染。
- 采用 **Delay-Aware 训练**：每个样本随机采样实际视觉陈旧度（0 到 `max_visual_age_steps` 步）及仿真规划耗时 $L \sim U(10\text{ms}, 60\text{ms})$，前瞻构造每个 Query 的执行延迟 $d_j = \text{age}_v + L + j \Delta t$。
- 状态输入统一使用物理时间归一化的状态速度 $\dot{s}_t = \frac{s_t - s_{t-1}}{\Delta t}$，消除不同规划频率下的数值尺度失配。
- 动作 waypoint 以 10 Hz 生成（与 LIBERO 10 Hz 控制时序严格对齐）；闭环执行器支持朴素 Receding Horizon（暴露真实边界跳变）与可选的 ACT 式指数加权平滑（`--ensemble-lambda`）。
- 部署使用 `Threading.Lock()` 互斥保护同一个流式 Session，彻底杜绝异步感知追加与规划回滚之间的竞态条件。
- 所有 MOSS 加载都设置 `local_files_only=True`；这些脚本不会自动下载模型权重。

## 浅层哨兵首批接口（P0–P2）

新增最终执行命令账本、原生 ViT prefix/suffix 续算接口，以及独立的 `extract/train/calibrate/shadow` 离线监控工具。
**当前 shadow 是离线回放，尚未接入在线事件执行；默认策略与 planner 不变。**
完整数据合同、运行命令、真实权重验收限制与已有 parity 不稳定项见 [docs/SENTINEL_P0_P2.md](docs/SENTINEL_P0_P2.md)。

## 根目录入口

```text
model.py          截断加载、增量 MOSS KV、Ephemeral Action Query、轻量 MLP 头
data.py           官方 LIBERO HDF5 → 单 planning-time / delay-aware / state_velocity
train.py          audit、单步 preflight、全参数 Action Query 训练
evaluate.py       官方 fixed-init LIBERO blocking/async 闭环评测与边界抖动度量
streaming.py      异步 perception worker、chunk planner、chunk executor
test_contract.py  无权重单元测试、Action Query Parity、路径 A/B 回滚不变性测试
```

原始 MOSS-VL 代码、推理、微调、SGLang 和 Flash-Attention 文件也都位于本仓库根目录。原始说明保存在 [docs/MOSS_VL_UPSTREAM.md](docs/MOSS_VL_UPSTREAM.md) 和 [docs/MOSS_VL_UPSTREAM_zh.md](docs/MOSS_VL_UPSTREAM_zh.md)。

基础权重必须使用 `OpenMOSS-Team/MOSS-VL-Realtime`。`MOSS-VL-Instruct-0708` 是离线 checkpoint，加载审计会直接拒绝，避免把离线视频模型误当成流式策略。

## 安装

```bash
pip install -r requirements.txt
```

LIBERO 评测还需要按其官方方式安装 `libero` 包和 MuJoCo 运行环境。训练数据直接读取官方 HDF5，必须包含：

```text
data/demo_N/actions                         [T, 7]
data/demo_N/obs/agentview_rgb              [T, H, W, 3]
data/demo_N/obs/joint_states               [T, 7]
data/demo_N/obs/gripper_states             [T, 2]
data.attrs.problem_info.language_instruction
```

图像只做一次 OpenGL 垂直翻转。状态用训练集 1%/99% 分位归一化；动作保持 LIBERO 原始 OSC `[-1,1]` 语义。每个时刻执行 `instruction + frames[0:t-visual_age] + state[t] + Δstate[t] + K queries → actions[t:t+K]`；每帧间隔默认为 `0.1 s`。可用 `--frame-stride`、`--frame-interval` 调整采样时间，如数据记录语义不同可用 `--action-offset 0` 校准。

动作头采用 **Streaming Action Query Decoder**：K 个临时动作查询嵌入进入 24 层 MOSS 骨干网络（通过内部 Self-Attention 读文本指令与状态条件，通过 Gated Cross-Attention 读增量视觉 KV），由轻量 L1 回归 MLP 一次性预测连续 micro-chunk。生成完成后临时查询的 KV 被立刻截断，保证已执行历史不被旧计划污染。训练时使用 delay-aware 视觉采样模拟异步运行时的视觉延迟。

## 1. 无权重检查

```bash
python test_contract.py
python -m compileall model.py data.py train.py evaluate.py streaming.py test_contract.py
```

安装 PyTorch 后，测试还会运行小型假 MOSS 骨干，检查逐帧 memory、Streaming Flow 状态、异步 perception/action mailbox 和 strict audit；没有 PyTorch 时，线程与 mailbox 测试仍会运行，模型测试会明确跳过。

## 2. 本地 checkpoint 审计

```bash
python train.py audit \
  --moss-checkpoint /absolute/path/to/MOSS-VL-Realtime
```

审计要求保留张量全部精确加载；唯一允许的 unexpected tensors 是原 checkpoint 的 layer `24..47`。命令同时记录所有本地权重 shard 的 SHA256 和截断后的精确参数量。

## 3. 两项 parity（长训练前硬门槛）

```bash
python test_contract.py parity \
  --checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --image /absolute/path/to/test.png \
  --instruction "pick up the red block"

python test_contract.py streaming-parity \
  --checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --image /absolute/path/to/test.png \
  --instruction "pick up the red block"
```

第一项检查完整48层与截断24层的 raw taps；第二项分别检查第一帧、第二帧的“一次完整计算”和“指令 prefill + 帧增量追加”产生相同的最新 `H14/H18/H23`。两项都通过后再开始长训练。

## 4. 全参数训练

先执行一个 forward/backward/optimizer smoke test：

```bash
python train.py preflight \
  --moss-checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --data /absolute/path/to/libero/datasets \
  --frame-stride 1 --frame-interval 0.1
```

再训练视觉编码器、保留的 24 层 MOSS 和 Streaming Action Expert：

```bash
python train.py train \
  --moss-checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --data /absolute/path/to/libero/datasets \
  --output checkpoints/full_stream.pt \
  --frame-stride 1 --frame-interval 0.1 \
  --lr 1e-4 --backbone-lr 1e-5 \
  --batch-size 1 --gradient-accumulation 8 --epochs 10
```

长训练前可先固定一条样本做过拟合检查：

```bash
python train.py train \
  --moss-checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --data /absolute/path/to/libero/datasets \
  --overfit-one \
  --frame-stride 1 --frame-interval 0.1 \
  --batch-size 1 --epochs 500 --max-steps 500 \
  --output checkpoints/overfit_one.pt
```

checkpoint 保存视觉编码器、保留的 24 层 MOSS 和 Action Query Decoder 的完整微调权重；加载时仍验证基础 Realtime checkpoint 的 SHA256。

部署前再对完整微调权重执行一次 KV parity：

```bash
python test_contract.py streaming-parity \
  --checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --policy checkpoints/full_stream.pt \
  --image /absolute/path/to/test.png \
  --instruction "pick up the red block"
```

## 5. LIBERO 闭环

```bash
python evaluate.py \
  --policy checkpoints/full_stream.pt \
  --moss-checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --suite libero_10 --task-ids 0,1,2 \
  --mode async --backbone-mode streaming-kv --control-hz 10 \
  --output evaluation.json
```

评测支持通过 `--ensemble-lambda 2.0` 启用 ACT 式时间加权平滑（不传则最新 chunk 胜出，直接暴露边界跳变用于纯粹度量）。

输出包含每任务成功率、task-macro success、planner latency、两路频率、编码帧数、chunk stalls 和边界跳变（boundary jump / step delta）。`--mode blocking` 是执行调度对照；`--backbone-mode full-recompute` 是关闭增量 KV 的模型计算对照。

用 `--action-delay-ms 100`（或其他延迟）可直接做异步鲁棒性曲线。

`moss_action_v6` 采用增量视觉 KV + 临时 Action Query + 连续 micro-chunk L1 解码，并在训练时使用单 planning time 的 delay-aware 视觉采样；拒绝加载更早格式 checkpoint，需要重新训练。

## 暂不包含

没有 World Model、VA、语言生成、PCGrad 或视觉缓存淘汰策略。每个 episode 必须创建新的 stream session；语言输出头仍未恢复。
