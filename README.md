# MOSS-Action VLA

MOSS-Action 保留 MOSS-VL 的完整视觉编码器和统一解码层 `0..23`，删除层 `24..47` 与语言输出头。指令只 prefill 一次；每个新帧单独编码并追加到 episode 级原生视觉 K/V cache，新的 frame-end control token 读取从开局到当前时刻的语言—视觉上下文。

训练对视觉编码器、MOSS `0..23` 层和 Action Expert 做全参数更新。一个样本就是一个完整 episode，在所有 frame-end 同时监督动作；部署再把同一 causal forward 等价地改成增量 KV。动作侧采用 action-space Streaming Flow：持久 action state 是 Q，每个控制 tick 只积分一次 velocity、立刻输出一个动作；新画面在下一个动作边界更新条件，不再等待 `H50 × 7` 动作块完成 8 次去噪。

## 已实现合同

- 24 个保留层总计包含 18 个 Self-Attention 层和 6 个原生 Gated Cross-Attention 层；后者必须位于 `2,6,10,14,18,22`。
- 训练把完整 episode 作为一条 causal sequence，并对视觉编码器、MOSS `0..23` 和 Action Expert 的全部参数反向传播；没有冻结或 LoRA 模式。部署时指令只 prefill 一次，新帧使用 MOSS 原生 `vision_cache_position` 追加，历史帧不重编码。
- 每个 frame-end 都有一项稳定化条件 Flow Matching 损失：第 `t` 帧只读取 `frames[0:t]`，不会看到未来画面。未来 `H50` 动作只用于构造轨迹位置 `ξ(τ)` 与速度 `ξ̇(τ)`，不作为部署时反复生成的动作块。
- 不切固定四帧窗口，也不把 episode 拆成短片段；`--frame-stride` 只控制原始帧采样密度。
- 从最新 frame-end token 读取未经 final RMSNorm 的原始 `H14/H18/H23`，分别投影成三个动作 memory token。
- 当前 action state、D9 机器人状态和 Flow time 组成动作 Q；Action Expert 对三个 memory token 执行 Cross-Attention 和 FFN，直接预测当前 action-space velocity。
- 部署拆成独立的 MOSS perception worker 与 Streaming Action Flow worker：感知流异步刷新 memory；动作流按控制频率持续生成，始终读取最新 memory 与最新 proprioception。
- 动作从上一动作附近的窄高斯开始，训练目标为 `ξ̇(τ) - k(a-ξ(τ))`；稳定项将受新视觉条件影响而偏离的 action state 拉回当前示范轨迹。
- 所有 MOSS 加载都设置 `local_files_only=True`；这些脚本不会下载模型权重。

## 根目录入口

```text
model.py          截断加载、增量 MOSS KV、Cross-Attention Streaming Flow Expert
data.py           官方 LIBERO HDF5 → 完整 episode/per-frame state/H50/mask
train.py          audit、单步 preflight、全参数流式训练
evaluate.py       官方 fixed-init LIBERO blocking/async 闭环评测
streaming.py      perception/action workers、视觉与动作最新值 mailbox
test_contract.py  无权重单元测试、prefix parity、streaming two-frame parity
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

图像只做一次 OpenGL 垂直翻转。状态用训练集 1%/99% 分位归一化；动作保持 LIBERO 原始 OSC `[-1,1]` 语义。每个时刻执行 `instruction + frames[0:t] + state[t] + action_state[t] → velocity[t]`；每帧间隔默认为 `0.1 s`。可用 `--frame-stride`、`--frame-interval` 调整采样时间，如数据记录语义不同可用 `--action-offset 0` 校准。

Streaming Flow 默认使用 `--initial-action-noise 0.1 --stabilization 10`。前者控制多样性与重启扰动，后者控制偏离示范轨迹后的回拉强度；两者属于实机校准参数，恢复训练时必须保持一致。

动作目标采用 [Streaming Flow Policy](https://arxiv.org/abs/2505.21851) 的 action-space conditional flow：把动作轨迹本身作为 flow trajectory，而不是在每个观测上生成“trajectory of trajectories”。本项目额外允许 MOSS memory 在相邻 action step 之间刷新。

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

长训练前可先固定一条样本和 Flow 噪声做过拟合检查：

```bash
python train.py train \
  --moss-checkpoint /absolute/path/to/MOSS-VL-Realtime \
  --data /absolute/path/to/libero/datasets \
  --overfit-one --fixed-flow-noise \
  --frame-stride 1 --frame-interval 0.1 \
  --batch-size 1 --epochs 500 --max-steps 500 \
  --output checkpoints/overfit_one.pt
```

checkpoint 保存视觉编码器、保留的 24 层 MOSS 和 Action Expert 的完整微调权重；加载时仍验证基础 Realtime checkpoint 的 SHA256。

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

调试过拟合样本时加 `--fixed-noise`，可在相同观测下复现完全相同的 Flow 初始噪声。

输出包含每任务成功率、task-macro success、perception/action-flow latency、两路频率、编码帧数、重复动作比例和 action age。`--mode blocking` 是执行调度对照；`--backbone-mode full-recompute` 是关闭增量 KV 的模型计算对照。

用 `--action-delay-ms 100`（或其他延迟）可直接做异步鲁棒性曲线。

`moss_action_v5` 将动作块 Flow 改为逐动作 Streaming Flow，并采用全参数、完整 episode 的 causal 训练；拒绝加载更早动作头，需要重新训练。

## 暂不包含

没有 World Model、VA、语言生成、PCGrad 或视觉缓存淘汰策略。每个 episode 必须创建新的 stream session；语言输出头仍未恢复。
