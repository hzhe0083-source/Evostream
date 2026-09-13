# FabriVLA 迁移核查

日期：2026-09-07。状态：FabriVLA 官方接口与 G0.5 论文身份已确认；已核查结构化 RVQ 动作编码；未实施迁移、未续训。

## 已确认的来源

- 官方仓库：https://github.com/Youi-FabriX/FabriVLA
- 审查固定版本：`95609d272e66001ae26a37185ebdfce698d6ebe6`。
- 官方权重：https://huggingface.co/Youi-FabriX/FabriVLA
- 官方配置：`fabri-vla/configs/train/1-exp_shallow_concat_proj_scratch_100k.yaml`。
- 官方实现：`fabri-vla/src/model/internvl_embedder.py`、`fabri-vla/src/model/action_head.py`。
- 本地已有安装记录：`/home/ryan/Documents/robot/ORA0/docs/FABRIVLA_SERVER_SETUP_2026-09-06.md`。该记录报告了服务器权重及单次前向验证，本轮未连接服务器复验，不能当作本轮运行结果。

## 源码核查结果

1. FabriVLA 使用 `OpenGVLab/InternVL3_5-1B`，配置保留 14 层，融合深层与第 6 层特征；`concat_proj` 将两者拼接后投影回 1024 维。
2. 图像特征替换 `<IMG_CONTEXT>` token 的 embedding，再与文本一起进入 language model。官方 prompt 为图像段在前、指令在后，不是 MOSS 独立视觉 Cross-Attention 缓存结构。
3. 官方动作头通过噪声和动作的插值训练速度预测，并在采样时数值积分；不是离散动作词表输出。配置 horizon=50、per_action_dim=24、state_dim=24、num_inference_timesteps=50。MetaWorld 的实际控制维度不能直接等同于模型补零后的 24 维。
4. 本仓库当前 MOSS 实现依赖 18 个 Self-Attention 层、6 个视觉 Cross-Attention 层及临时 action query KV 回滚，不能只替换 checkpoint 路径。

## 迁移边界与实施约束

- 保留 FabriVLA 原生权重、浅深层融合和连续动作头，作为第一阶段基线。不要为未知动作词典预建输出头。
- 历史图像编码特征缓存、联合序列 self-attention KV 缓存、MOSS 式独立视觉 KV 是三种不同方案；实现与报告必须明确区分，不能将单帧特征复用称为完成 MOSS 迁移。
- 若采用联合因果序列追加帧，需定义新的时间顺序、指令位置、padding、position IDs 和读取动作的上下文。必须用同一新序列语义的完整前向作为缓存 parity 基准；不能宣称与原来的逐帧 prompt 天然等价。
- 若采用独立视觉 Cross-Attention，需要新增或改造注意力模块并续训；原 InternVL self-attention 权重不能直接充当已训练好的 MOSS 视觉注意力权重。
- 第一阶段验收：原 checkpoint 严格加载；无缓存原生基线；单帧及多帧缓存/全量前向对齐；episode reset 与指令切换；训练梯度路径；真实权重前向/反向 smoke；同硬件的完整规划延迟与闭环任务结果。缓存跨优化器更新复用不在允许的训练语义内。
- 第二阶段先检验目标动作 tokenizer 的编码—解码误差，再检查动作定义、归一化、控制频率、词表及输出头兼容性，最后续训与比较。

## G0.5 已确认；剩余是实现资料核查

用户提供 https://arxiv.org/abs/2608.11739 。原文标题为 G0.5: One Autoregressive Stream for Robot Reasoning and Action，不是 π0.5。§3.1 的“词典”是按运动部件分组的 RVQ 动作编码：每个残差轮次，每个激活组的结构标记后跟 8 个 action codes。27 维指统一连续动作空间，不是词表大小。最终动作输出需要离散自回归预测和 codec 解码，FabriVLA 原生连续头只作为过渡基线。详见 [G0.5 定向分析](research/g05/analysis.md)。

不再需要用户解释模型身份。尚需核实可用 codec 权重、token-ID 表、残差轮数和 chunk 时间配置；本轮未获得这些可加载资产。项目页静态抓取只有标题，不能据此断言未开源。不得将自训 RVQ codec 冒称官方 G0.5 词典。

## 本轮验证与保护

- 已读取固定版本官方 README、视觉融合代码、动作头代码及配置；这是静态审查，不是运行验证。
- 外部 Grok 两轮均因额度耗尽失败；GitHub tree API 返回限流，改用固定版本 raw 源码读取成功。
- 未运行训练、模型测试或性能评测；未下载模型权重、未修改服务器。
- 保留任务开始前 README.md、evaluate.py、model.py、streaming.py 的修改及 Sentinel/vision_split 等未跟踪文件。
- 本轮仅新增本说明文档，不更改现有模型路径，不新增无法验证的适配器，不提交或推送。
