# FabriVLA 原生 KV 缓存与备用 MOSS-style 实验

## 当前可选路径与验收状态（2026-09-11）

- **新可选紧凑时间路径已实现**：`--native-memory compact-temporal --timestamp-mode text`，非决策观察256→64视觉tokens并紧凑入LLM，当前/历史决策帧完整；原1D RoPE增加连续时间相位，查询后后台重建、下一查询前完整生效。时间文字仍保留，默认入口不变。
- 新训练协议 `--stream-protocol compact_memory_replay_v1`，支持dense或memory v1.2的显式`--transition-from`；启动器 `scripts/run_compact_10epochs.sh`。本轮未启动新协议训练，也未切换现有双L20运行。
- 主模型本地/服务器各359tests通过，CPU与本机CUDA-eager tiny-Qwen独立脚本通过，真实FastTokenizer三线程隔离复验通过。**尚未取得新路径真实FabriVLA权重FA2测速或闭环能力保持结果**。
- 本地机制验证：`python -m fabri_moss.verify_compact_memory --device cpu --output-dir <新空目录>`。详见 [新路径入口、机制验收与边界](../docs/fabri_cache/COMPACT_TEMPORAL_ACCEPTANCE.md)。
- 现有正式训练是 [memory v1.2阶段](../docs/fabri_cache/MEMORY_STAGE_RUN.md)，09-11 01:53实查第6轮、日志14187/27790、完整last保存14000。下方早期运行数字与窗口方案保留为历史，不作为当前运行状态。

## 原生窗口路径与早期运行记录

**当前主路径：原 LLM 的 Q 读取 history KV + 新增 delta KV，不新增 attention、MLP、门控或 readout 参数。** 保留 FabriVLA step93000 的原 ViT 最终层投影、14 层语言模型、浅深 1024-token 上下文和原 Action Expert。ViT、LLM 与专家均在 GPU；CPU 仅用于功能测试。

- 入口：`python -m fabri_moss.evaluate_async --mode native-cache --history-frames 16 --timestamp-mode text ...`，不传 `--adapter` 或旧 `--window`。
- 每轮取请求时刻全部 ready 新帧，不凑5帧、不截断为5帧；`--history-frames 16` 只控制保留历史，较大新增批次按时间顺序分组全部计算。
- 默认 pending/ready 不设帧数上限；显式 `--max-pending` / `--max-ready` 满载报错而不是静默删帧。持续过载仍会积压，不保证无限内存。
- 默认把相对 episode 起点的场景秒数作为普通文本放在图像之前；采集host时间另用于延迟。`--timestamp-mode none` 保留未加时间的原单帧输入对照。
- 未满窗只处理新 embedding blocks；满窗从保留 embeddings 重建 LLM KV，不重跑 ViT，但有 LLM 重建成本。
- 纯原生 joint KV 为 **56 MiB/帧，16 帧 896 MiB**；不使用旧 cross-cache 或压缩矩阵的容量数字。
- 零新增模型参数。**原生全模型多帧微调已于2026-09-09 00:02:59 +0800从93k启动双GPU10epochs**，现有879,465,144个参数全部解冻，FP32权重/AdamW更新，视觉语言BF16+FA2、专家FP32。推理adapter仍冻结/no_grad；新训练入口 `python -m fabri_moss.train_native`。旧矩阵step50仍停止，旧双卡草稿不复用；多层ViT/历史压缩备用。

**新训练已验收**：本地/服务器各187测试通过；真实16帧末目标loss回传最早历史视觉feature，四组原模型参数均实际更新；双卡2步保存再恢复到4步通过。独立GPU连续/恢复对照非逐位一致，最大权重差6.78e-7，RNG/游标/调度一致。正式数据2250train/250val，每轮169784目标起点，10轮27790updates；00:07检查到step11，完整last.pt保存step1。新模型尚无闭环收益结论。详见 [全模型10轮训练合同与运行记录](../docs/fabri_cache/NATIVE_FULL_FINETUNE.md)。

**本版已亲测验收**：本地/服务器各153测试通过；真实FA2带时间的1+8+2快照全部进入语言层，异步/同分块直接特征与KV差0，8帧内部两组计算但专家只调用一次。闭环实际出现19帧快照、过载丢帧0；2×80步完成、0/2成功，仅算入口检查。时间关闭时的原单帧0差基线另行保留，不能声称加时间prompt仍与原单帧完全相同；BF16跨分块形状数值差边界仍在。

[动态批次、时间语义、命令与完整结果](../docs/fabri_cache/DYNAMIC_TIMESTAMP_CACHE.md) · [原生接口说明](../docs/fabri_cache/NATIVE_KV_CACHE.md) · [本版GPU报告](../docs/fabri_cache/dynamic_results_20260908/verify_gpu_report.json)。下方保留历史矩阵 / consume 实验，**不是当前默认架构**。

## 备用矩阵 Delta 版本（2026-09-08，已停止于 step50）

新增 `memory_mode="delta"`：本批新帧保留精确cross-attention KV；历史通过残差 `V - normalize(K) @ S` 更新固定矩阵，不再在每次规划后全部遗忘。四层矩阵持久容量 **2 MiB**，五帧局部KV另需 **40 MiB**。它是压缩递归记忆，不是全历史softmax的无损增量公式。

- 模型接口：`forward_delta(images_window, frame_ids, prompt, previous=None)` 返回 `deep, shallow, next_state`；`read_delta`读取已投影KV，旧state不被原地修改。
- 异步接口：delta模型使用 `AsyncVisualPlanner(..., stateful=True)`，仅规划成功才提交next_state；结果不持state，reset清空历史。consume默认行为与旧接口保留。
- 连续训练入口：`python -m fabri_moss.train_delta`；每条episode内顺序更新记忆、截断反向传播，完整episode结束才允许参数更新，避免跨优化器使用旧投影记忆。原93k完整权重初始化，首阶段保留原动作专家。
- **真实FA2已校准**。本版本进程须显式加入私有依赖，且训练/评测delta入口检查真实语言和视觉FA2；旧eager短训适配不用于新训练。

完整状态与实际命令见 [delta合同](../docs/fabri_cache/DELTA_MEMORY.md) 和 [训练运行记录](../docs/fabri_cache/DELTA_TRAINING_RUN.md)。下方为已完成的旧consume版本记录，不能将其短训结果当delta训练结果。

```bash
export PYTHONPATH=/root/FabriVLA/diagnostics/fa2_repro_20260908/python_packages:/root/Evo_stream_delta_20260908
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

## 1. 模型与缓存

```text
GPU视觉流：新图像 → 原ViT / pixel shuffle / mlp1 → 每层视觉K/V → ready队列
                                                                    │
GPU规划流：指令+16个readout → 原14层LLM + 门控cross-attention ← 固定快照
                                           │
                               原concat_proj + 原Action Expert
```

- 完整加载FabriVLA checkpoint；保留原ViT、视觉投影、14层LLM、原连续动作专家。第3/6/10/14层后插入复制原层参数初始化的门控交互层。
- `VisionSession` 是旧的同步滑窗接口，默认2帧，供结构测试；`AsyncVisualPlanner` 是新异步消费接口，默认最多5帧。
- 新观测与state/mask在CPU提交边界复制；模型计算在GPU。vision/planner使用独立流，完成事件和`record_stream`管理张量生命周期。
- request_plan取当时ready快照；成功后释放本批KV，返回结果只保留帧编号/时间元数据与动作、感知输出。规划期间新来的帧留给下一轮。失败保留快照供`retry=True`。
- 队列有界。过载丢最旧待编码或未消费帧并计数，不承诺所有观测无损保留；没有新帧时不重复消费旧帧。
- episode/指令reset使用generation隔离旧任务。模型运行中不可训练或改变权重。v1不跨规划缓存语言self-attention KV。

## 2. 服务器 GPU 命令

当前工作目录：`/root/Evo_stream_moss_async_20260908`。

```bash
cd /root/Evo_stream_moss_async_20260908
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# 本轮实测使用物理 GPU 1；设定后 cuda:0 指向物理 GPU 1。
export CUDA_VISIBLE_DEVICES=1
```

`train`、`evaluate_async`、`verify_async` 默认设备已改为`cuda:0`，GPU不可用会报错，不静默退回CPU。实际运行前检查剩余显存和已有任务。下列命令不自动施加显存限额；本轮共享L20验收在外部启动代码中使用4.5GiB PyTorch分配限额，没有停止其他任务。共享GPU的耗时不能当独占基准。

### GPU 异步回放

```bash
/root/fabrivla_env/bin/python -m fabri_moss.verify_async \
  --data-root /root/evo1_metaworld_dataset \
  --adapter /root/Evo_stream_moss_async_20260908/consume_train_gpu/adapter_final.pt \
  --output-dir /root/Evo_stream_moss_async_20260908/replay_gpu_new \
  --window 5 --flow-steps 50 --device cuda:0
```

固定真实帧`[0..4]`作为第一轮，规划时提交`[5..7]`作为下一轮；验证不同线程、实际时间重叠、按截止点消费、缓存/fresh特征对齐、有效动作mask。使用数据集自身指令。此CLI专门验收五帧，要求`window=5,max_pending>=5`；不是实时摄像头吞吐测量。

### 五帧消费训练

```bash
/root/fabrivla_env/bin/python -m fabri_moss.train \
  --data-root /root/evo1_metaworld_dataset \
  --init-adapter /root/Evo_stream_moss_v1_20260908/bridge_resume_cpu/adapter_final.pt \
  --output-dir /root/Evo_stream_moss_async_20260908/consume_gpu_new \
  --stage bridge --context-mode consume --window 5 --frame-stride 1 \
  --min-context-frames 1 --steps 3 --max-episodes 2 \
  --fixed-sample --fixed-noise --device cuda:0
```

这是短训诊断。`consume`把episode采样帧划分为长度1..W、不重叠的片段，使用末帧状态和后续动作监督。当前不模拟真实GPU完成顺序/过载丢帧；正式数据适配需验证该差异。保留`--context-mode window`兼容旧训练。

`bridge`冻结原ViT/投影/LLM/动作专家，梯度仍穿过冻结模块回传新增连接。教师为原生单帧FabriVLA，GT flow loss与教师速度蒸馏使用共同noise/time。有效4维补零24，50步动作尾部repeat-last，复用原checkpoint归一化。

**历史结果限制**：上述旧consume短训沿用eager部署，与FA2 padding-query语义不等价，因此这些权重仍只用于功能检查。此后原生真实FA2已校准并实测445/500成功；新delta训练从原93k和正确FA2重新开始，不续用上述旧适配。见delta运行记录及Obsidian原生基线复测记录。

### GPU 闭环入口

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}
/root/fabrivla_env/bin/python -m fabri_moss.evaluate_async \
  --mode moss \
  --adapter /root/Evo_stream_moss_async_20260908/consume_train_gpu/adapter_final.pt \
  --task reach-v3 --episodes 1 --episode-horizon 400 \
  --exec-horizon 5 --num-inference-timesteps 50 \
  --control-mode realtime --control-hz 30 --window 5 \
  --output-dir /root/Evo_stream_moss_async_20260908/moss_gpu_new \
  --device cuda:0
```

原生对照使用`--mode native`且不传adapter。两者共用本入口的动作时间对齐和环境协议；不等于官方原始同步MT50评测复现。任务指令按官方metadata取，不用自造prompt代替。

- `realtime`尝试按墙钟调度；达不到deadline会记录miss并重定截止点，不声称自动达到30Hz。动作耗尽执行并记录**环境空间零动作**，不是将归一化零动作反变换，也不是已证明的安全保持。
- `step_wait`仅在无可用动作时暂停仿真等计划，保留为功能检查。
- 动作第0步对应计划source_frame_id，当前控制step选择`step-source_frame_id`；过期计划拒绝，不重放旧首步动作。环境render/step仅在控制线程。

## 3. 权重初始化与恢复

- `--init-adapter`：只载适配权重，可显式2→5帧或bridge→expert；只容许声明的`max_frames`差异，其他架构/基座SHA/归一化严格。新优化器、新step和随机状态，保存来源与文件SHA。不允许expert→bridge默默丢掉已训练专家。
- `--resume`：同stage、同模型与**同数据合同**恢复参数、优化器及随机状态。合同包含context模式、stride、window、episode IDs与metadata SHA。不能用resume切换消费协议。旧无合同checkpoint只能按旧window配置续训。
- `--steps`表示本次额外更新次数。CUDA模型保存/恢复CUDA随机状态；CPU检查不为保存RNG而初始化GPU。

当前GPU检查产物：`/root/Evo_stream_moss_async_20260908/consume_train_gpu/adapter_final.pt`。包含新增参数及AdamW状态，不重复保存完整基座。正式模型依然依赖原FabriVLA和InternVL文件。

## 4. 异步接口

```python
from fabri_moss.async_pipeline import AsyncVisualPlanner, Observation, make_moss_callbacks

# model 已严格加载、初始化适配权重并在 GPU eval；启动线程后不可修改。
encode, plan, validate = make_moss_callbacks(model)
with AsyncVisualPlanner(encode, plan, max_frames=5, max_pending=8, validate=validate) as worker:
    worker.reset("episode-1", instruction)
    worker.submit(observation)  # Observation: 单视角图像、CPU归一化state/masks、真实step编号和monotonic捕获时间
    # 摄像头/环境线程可持续 submit，不等待模型编码。
    if worker.wait_ready(timeout=30):
        worker.request_plan()
        result = worker.wait_plan(timeout=30)
        # result.computation.actions: [1,50,24]；result.frames仅包含元数据，不持视觉KV。
```

无隐式相机采集线程：调用方负责捕获真实观测；模型的视觉编码和规划是两个独立工作线程。同步`VisionSession`接口仍保留，不与异步消费混称。单样本、单视角、单tile是当前边界。

## 5. 验证与报告

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /root/fabrivla_env/bin/python -m pytest fabri_moss/tests -q
```

39项单测覆盖并发快照、重试/reset、过载计数、已消费payload回收、时间偏移/零动作回退、消费采样及初始化/恢复。GPU实测另见[验收记录](../docs/fabri_cache/ASYNC_RUN_20260908.md)。训练、回放、闭环输出目录不能无意覆盖已有结果；权重不进入git仓库。
