# Impact-audio literature map (2026-08-12)

Scope: reading companion for the audio study's direction decision — whether to keep pushing the fly/ground screening task, and along which road. All arXiv PDFs below are downloaded under `papers/`; open-access classics under `papers/X_open-access-classics/`.

How to use: don't read in listing order. Read the three waves in §8. Each entry says **为什么收**（why it relates to *this* project）and **精读看什么**（what to extract）. The payoff of a paper is what it does to our decision, not whether it is a good paper in general.

## 1. Our stuck points (what the literature must speak to)

1. **Event vs context**: frozen M2D event BA ≈ 0.61 (0.67 on timing-eligible pairs), but the same model on strict pre-contact ≈ 0.58; full-clip duration alone reaches BA ≈ 0.93. We have not yet isolated contact-specific information.
2. **No source transfer**: M2D ≈ chance on human-collected external data; nearly all source groups are singletons, so grouped CV degenerates to ordinary stratified CV.
3. **Collection confound**: fly/ground correlates with collector, clip length, and clipping workflow; the physical question is drowned by the collection question.
4. **Representation bottleneck candidates**: 200 ms ≈ five 40 ms M2D patches; mean/std/max pooling destroys onset–decay order; pretrained models were built for multi-second semantic clips, not sub-second transients.
5. **Simulation exists but unproven**: `contact_audio_simulation/` can synthesize contact audio; we don't yet know whether synthetic audio transfers, or whether physics from sound is even estimable in principle.

## 2. Cluster → stuck point

| Cluster | Answers stuck point |
| --- | --- |
| A impact/bat-ball acoustics | what physical features *should* carry outcome info (prior for features) |
| B transient/impact classification | how to represent & pool a sub-200 ms transient without losing order |
| C sports audio | closest-domain evidence: what others got from ball-impact audio |
| D foundation models & adaptation | when frozen > fine-tuned; what PEFT can and cannot fix |
| E confounds/leakage/evaluation | the evaluation machinery we must trust before any model claim |
| F simulation/synthesis | can we generate or augment our way out of the data problem? |
| G physics-from-sound | is "outcome from contact sound" estimable in principle? |
| H baseball/trajectory | what the wider project does with video; where audio might plug in |

## 3. A — Impact & bat-ball acoustics

- **Perceptual analyses of action-related impact sounds** — Bezat, Roussarie, Kronland-Martinet, Ystad et al., 2010. [arXiv](https://arxiv.org/abs/1003.4908) · `papers/A_impact-bat-ball-acoustics/1003.4908.pdf`
  为什么收：动作相关的冲击声（打击、碰撞）感知实验——人靠哪些声学属性区分冲击类别。
  精读看什么：他们分离了哪些属性（衰减、频谱质心、模态、激励）？哪些属性跨强度稳定（对应我们的跨响度鲁棒性）？能否直接变成我们的传统特征分组。

- **RealImpact: A Dataset of Impact Sound Fields for Real Objects** — Clarke, Gao, Wang, Rau et al., 2023. [arXiv](https://arxiv.org/abs/2306.09944) · `papers/A_impact-bat-ball-acoustics/2306.09944.pdf`
  为什么收：真实物体冲击声的规模化数据集+模态分析——冲击声领域怎么做受控采集和物理标注。
  精读看什么：他们的冲击声模态标注流程；单次冲击可辨识的物理量范围（材料、形状、部位）——对比我们想从 200 ms 里读出轨迹结果，量级差在哪。

- **The ObjectFolder Benchmark** — Gao, Dou, Li, Agarwal et al., 2023. [arXiv](https://arxiv.org/abs/2306.00956) · `papers/A_impact-bat-ball-acoustics/2306.00956.pdf`
  为什么收：神经物体冲击声合成与真实录音的对照基准；合成-真实对拍的设计。
  精读看什么：合成冲击声与真实录音之间的域差距如何度量、如何缩小——直接服务我们的仿真增强路线。

## 4. B — Transient & impact-sound classification

- **Adaptive pooling operators for weakly labeled sound event detection** — McFee, Salamon, Bello, 2018. [arXiv](https://arxiv.org/abs/1804.10070) · `papers/B_transient-impact-classification/1804.10070.pdf`
  为什么收：我们的 mean/std/max 全局池化正是丢弃 onset–decay 顺序的嫌疑环节；这篇给出可学习池化的谱系。
  精读看什么：soft/auto-pool 的梯度和效果对比；200 ms≈5 token 时哪种池化保留时序最多；能否直接替换进 M2D head 做一个最小对照实验。

- **A benchmark of SED systems evaluated on synthetic soundscapes** — Ronchini, Serizel, 2022. [arXiv](https://arxiv.org/abs/2202.01487) · `papers/B_transient-impact-classification/2202.01487.pdf`
  为什么收：用合成声景做受控评估的方法论——正是我们 contact-audio-simulation 想做的事。
  精读看什么：合成评估的协议设计（SNR、事件密度、评估指标）；系统排名在真实 vs 合成数据上的一致性结论（合成评估是否可信的证据）。

- **JiTTER: Jigsaw Temporal Transformer for Event Reconstruction for Self-Supervised SED** — Nam, Park, 2025. [arXiv](https://arxiv.org/abs/2502.20857) · `papers/B_transient-impact-classification/2502.20857.pdf`
  为什么收：拼图式时序重建自监督——专门逼模型学事件的时序结构，恰好是我们丢掉的维度。
  精读看什么：jigsaw 任务怎么定义、在多少数据量上有效；能不能用作 M2D 之外的自监督预训练（或替代）候选。

- **Frequency Dynamic Convolutions for Sound Event Detection** — Nam, 2025. [arXiv](https://arxiv.org/abs/2506.12785) · `papers/B_transient-impact-classification/2506.12785.pdf`
  为什么收：频率维度动态卷积保留细粒度谱结构，适合瞬态事件。
  精读看什么：频率动态核与普通卷积的差异在哪些事件类型上显著（瞬态类收益最大？）；复杂度是否适合我们的小数据规模。

## 5. C — Sports audio (closest domain)

- **Sound-Based Spin Estimation in Table Tennis** — Gossard, Schmalzl, Ziegler, Zell, 2024. [arXiv](https://arxiv.org/abs/2409.11760) · `papers/C_sports-audio/2409.11760.pdf`
  为什么收：**头号精读**。任务与我们同构：从极短的球拍击球声推断球的旋转（运动结果），带数据集和实时管线。
  精读看什么：他们用了什么特征/模型（纯音频还是音视频）；准确率天花板在哪；数据集规模与采集多样性；哪些设计可直接搬到 fly/ground——以及他们是否做了采集混杂控制（我们已知的关键坑）。

- **Sports highlights generation based on acoustic events detection: A rugby case study** — Baijal, Cho, Lee, Ko, 2015. [arXiv](https://arxiv.org/abs/1509.06279) · `papers/C_sports-audio/1509.06279.pdf`
  为什么收：球类冲击声事件检测的经典；证明"踢/击球声可检测"的证据链。
  精读看什么：踢球声的特征（时域峰值、频带）与误检来源；广播音频里的冲击检测率——支撑"接触瞬态可检测"但"结果难辨"的对比。

- **Automatic Summarization of Soccer Highlights Using Audio-visual Descriptors** — Raventos, Quijada, Torres, Tarres, 2014. [arXiv](https://arxiv.org/abs/1411.6496) · `papers/C_sports-audio/1411.6496.pdf`
  为什么收：观众声/哨声/击球声做高光检测的经典——音频在体育广播里的信息分布。
  精读看什么：哪些音频事件被证明最可靠；短时能量类特征的泛化局限。

- **Wearable Audio and IMU Based Shot Detection in Racquet Sports** — Sharma, Anand, Srivastava, Kaligounder, 2018. [arXiv](https://arxiv.org/abs/1805.05456) · `papers/C_sports-audio/1805.05456.pdf`
  为什么收：击球声+IMU 融合——击球声与运动量之间的互补关系。
  精读看什么：音频单独能贡献多少（ablation）？可穿戴采集与广播采集的差异——对照我们的人工采集。

- **Improved Soccer Action Spotting using both Audio and Video Streams** — Vanderplaetse, Dupont, 2020. [arXiv](https://arxiv.org/abs/2011.04258) · `papers/C_sports-audio/2011.04258.pdf`
  为什么收：音视频融合定位踢球动作——音频在何时赢过视频、何时输。
  精读看什么：融合策略与音频单模态的消融；对"音频是否有独立价值"的直接证据。

- **Language and Multimodal Models in Sports: A Survey** — Xia, Yang, Zhao, Wang et al., 2024. [arXiv](https://arxiv.org/abs/2406.12252) · `papers/C_sports-audio/2406.12252.pdf`
  为什么收：体育 ML 全领域地图——看清音频在这个版图里占多大、旁边有什么。
  精读看什么：体育音频任务清单（有没有棒球击球声任务？）；数据集与模态分布；找我们可对齐或可合作的线。

- **Automated Detection of Sport Highlights from Audio and Video Sources** — Della Santa, Lalli, 2025. [arXiv](https://arxiv.org/abs/2501.16100) · `papers/C_sports-audio/2501.16100.pdf`
  为什么收：近期体育音视频检测的现状与方法谱系。
  精读看什么：现代管线里音频的作用与预算；和 2014/2015 经典对比，方法论进化了多少。

- **Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection** — Joo, Oh, 2026. [arXiv](https://arxiv.org/abs/2602.03891) · `papers/C_sports-audio/2602.03891.pdf`
  为什么收：双通路音频编码器——近期短时音频编码设计。
  精读看什么：两条通路分别编码什么（事件 vs 上下文？）——正是我们 event/context 问题的架构版本。

## 6. D — Audio foundation models & adaptation

这些我们已经在 V4/V5 实测过，重读的目的是**换一个问题的答案**：不是"谁在 leaderboard 上高"，而是"它的训练目标/结构会保留还是丢弃短瞬态物理信息"。

- **M2D: Masked Modeling Duo** — Niizumi, Takeuchi, Ohishi, Harada et al., 2024. [arXiv](https://arxiv.org/abs/2404.06095) · `papers/D_audio-foundation-and-adaptation/2404.06095.pdf`
  精读看什么：6 秒输入的设计动机；M2D-X 的领域适配设定（我们何时才轮到它）；40 ms 变体与 80 ms 变体的差异。
- **BEATs** — Chen, Wu, Wang, Liu et al., 2022. [arXiv](https://arxiv.org/abs/2212.09058) · `papers/D_audio-foundation-and-adaptation/2212.09058.pdf`
  精读看什么：语义 tokenizer 为什么故意丢弃"冗余"细节——它认为冗余的，是否正是我们的 onset/decay。
- **AudioMAE** — Huang, Xu, Li, Baevski et al., 2022. [arXiv](https://arxiv.org/abs/2207.06405) · `papers/D_audio-foundation-and-adaptation/2207.06405.pdf`
  精读看什么：patch 大小与掩码率；200 ms 下 patch 数≈5 意味着什么。
- **PANNs** — Kong, Cao, Iqbal, Wang et al., 2019. [arXiv](https://arxiv.org/abs/1912.10211) · `papers/D_audio-foundation-and-adaptation/1912.10211.pdf`
  精读看什么：CNN 感受野 vs 短窗口的结构性不兼容（V4 已实测 500 ms 下限）。
- **AST** — Gong, Chung, Glass, 2021. [arXiv](https://arxiv.org/abs/2104.01778) · `papers/D_audio-foundation-and-adaptation/2104.01778.pdf`
  精读看什么：10.24 s 原生输入；短窗口全靠 padding 的代价。
- **HTS-AT** — Chen, Du, Zhu, Ma et al., 2022. [arXiv](https://arxiv.org/abs/2202.00874) · `papers/D_audio-foundation-and-adaptation/2202.00874.pdf`
  精读看什么：层次化 token 与 swin 结构——短窗口下 token 数退化的机制。
- **ATST-Frame** — Li, Shao, Li, 2023. [arXiv](https://arxiv.org/abs/2306.04186) · `papers/D_audio-foundation-and-adaptation/2306.04186.pdf`
  精读看什么：明明是 frame-level 设计，为什么本地 200 ms 输给 M2D、且事件移除后不掉——"声称 frame-level ≠ 保留物理时序"的直接教材。
- **Deep Scattering Spectrum** — Andén, Mallat, 2013. [arXiv](https://arxiv.org/abs/1304.6763) · `papers/D_audio-foundation-and-adaptation/1304.6763.pdf`
  精读看什么：二阶散射对 attack/调制的表征；V4 无决定性优势后，何时才值得重开（数据重设计之后）。
- **Integrated Parameter-Efficient Tuning for General-Purpose Audio Models** — Kim, Heo, Shin, Lim et al., 2022. [arXiv](https://arxiv.org/abs/2211.02227) · `papers/D_audio-foundation-and-adaptation/2211.02227.pdf`
  精读看什么：哪些层冻结、哪些动；PEFT 在小数据上的边际收益——为 Stage 3 预演。
- **Parameter-Efficient Transfer Learning of AST** — Cappellazzo, Falavigna, Brutti, Ravanelli, 2023. [arXiv](https://arxiv.org/abs/2312.03694) · `papers/D_audio-foundation-and-adaptation/2312.03694.pdf`
  精读看什么：AST 家族适配器选择；多少数据下 PEFT ≈ 全量微调（临界点）。
- **Efficient Fine-tuning of AST via Soft Mixture of Adapters** — Cappellazzo, Falavigna, Brutti, 2024. [arXiv](https://arxiv.org/abs/2402.00828) · `papers/D_audio-foundation-and-adaptation/2402.00828.pdf`
  精读看什么：soft MoA 的插值机制；冻结主干+轻适配是否适合我们的 1851 配对样本规模。
- **On the Transferability of Large-Scale Self-Supervision to Few-Shot Audio Classification** — Heggan, Budgett, Hospedales, Yaghoobi, 2024. [arXiv](https://arxiv.org/abs/2402.01274) · `papers/D_audio-foundation-and-adaptation/2402.01274.pdf`
  精读看什么：冻结表征在 few-shot 下的可迁移性实证——什么条件下冻结输给微调。
- **Can Masked Autoencoders Also Listen to Birds?** — Rauch, Heinrich, Moummad, Joly et al., 2025. [arXiv](https://arxiv.org/abs/2504.12880) · `papers/D_audio-foundation-and-adaptation/2504.12880.pdf`
  精读看什么：在窄域（鸟声）复用 MAE 自监督的可行性与陷阱——和"要不要对棒球音频做领域自监督预训练"直接相关。
- **Probing Acoustic Representations for Phonetic Properties** — Ma, Ryant, Liberman, 2020. [arXiv](https://arxiv.org/abs/2010.13007) · `papers/D_audio-foundation-and-adaptation/2010.13007.pdf`
  精读看什么：探针方法论（linear probe、按属性分组、对照探针）——我们 attention probe / 严格 Pre 路线的学理版本。
- **Adapting Language-Audio Models as Few-Shot Audio Learners** — Liang, Liu, Liu, Phan et al., 2023. [arXiv](https://arxiv.org/abs/2305.17719) · `papers/D_audio-foundation-and-adaptation/2305.17719.pdf`
  精读看什么：CLAP 类模型在具体细粒度任务上的适配极限；为什么语义对齐不一定帮上物理任务。
- **Few-Shot Bioacoustic Event Detection with Frame-Level Embedding Learning System** — Zhao, Lu, Zou, 2024. [arXiv](https://arxiv.org/abs/2407.10182) · `papers/D_audio-foundation-and-adaptation/2407.10182.pdf`
  精读看什么：DCASE 2024 少样本生物声学的冠军系统设计——frame-level 表征+少样本事件检测的实操。

## 7. E — Confounds, leakage, evaluation (we must trust these before any claim)

- **On the cross-validation bias due to unsupervised pre-processing** — Moscovich, Rosset, 2019. [arXiv](https://arxiv.org/abs/1901.08974) · `papers/E_domain-shift-confounds-evaluation/1901.08974.pdf`
  精读看什么：scaler/归一化放错位置的偏差量级——审计我们 StandardScaler 的放置。
- **Cross-Validation for Correlated Data** — Rabinowicz, Rosset, 2019. [arXiv](https://arxiv.org/abs/1904.02438) · `papers/E_domain-shift-confounds-evaluation/1904.02438.pdf`
  精读看什么：相关结构下 CV 的理论偏误——我们的 game/lineage 分组 CV 的理论依据与残余风险。
- **Cross validation for model selection: a primer with examples from ecology** — Yates, Aandahl, Richards, Brook, 2022. [arXiv](https://arxiv.org/abs/2203.04552) · `papers/E_domain-shift-confounds-evaluation/2203.04552.pdf`
  精读看什么：nested CV 正确姿势与常见错误（生态学例子的可读性高）。
- **A Note on the Finite Sample Bias in Time Series Cross-Validation** — Lusompa, 2025. [arXiv](https://arxiv.org/abs/2512.05900) · `papers/E_domain-shift-confounds-evaluation/2512.05900.pdf`
  精读看什么：时序/分组切分的有限样本偏差——我们外部测试 n=87 的解释边界。
- **Unsupervised adversarial domain adaptation for ASC** — Gharib, Drossos, Çakir, Serdyuk et al., 2018. [arXiv](https://arxiv.org/abs/1808.05777) · `papers/E_domain-shift-confounds-evaluation/1808.05777.pdf`
  精读看什么：跨设备域适应经典——如果我们要"在 Codex 上训、在人工数据上适应"，这是路线之一。
- **Spectrum Correction: ASC with Mismatched Recording Devices** — Kośmider, 2021. [arXiv](https://arxiv.org/abs/2105.11856) · `papers/E_domain-shift-confounds-evaluation/2105.11856.pdf`
  精读看什么：设备频谱签名校正——轻量、可解释的跨设备方案，代价小值得试。
- **Adversarial DA with Paired Examples for ASC on Different Recording Devices** — Kacprzak, Kowalczyk, 2021. [arXiv](https://arxiv.org/abs/2110.09598) · `papers/E_domain-shift-confounds-evaluation/2110.09598.pdf`
  精读看什么：配对样本（同一场景两台设备）下的域适应——我们有没有可配对的跨采集样本（同一场比赛不同来源）？
- **Device-Robust ASC via Impulse Response Augmentation** — Morocutti, Schmid, Koutini, Widmer, 2023. [arXiv](https://arxiv.org/abs/2305.07499) · `papers/E_domain-shift-confounds-evaluation/2305.07499.pdf`
  精读看什么：IR 增广做设备鲁棒——与我们仿真增广同族；增广域假设（房间/麦克风脉冲）能否描述"采集者差异"。
- **ASC in DCASE 2020: generalization across devices** — Heittola, Mesaros, Virtanen, 2020. [arXiv](https://arxiv.org/abs/2005.14623) · `papers/E_domain-shift-confounds-evaluation/2005.14623.pdf`
  精读看什么：多设备泛化评估的标准设定——我们"跨采集者"评估协议的参照模板。
- **Domain Information Control at Inference Time for ASC** — Masoudian, Koutini, Schedl, Widmer et al., 2023. [arXiv](https://arxiv.org/abs/2306.08010) · `papers/E_domain-shift-confounds-evaluation/2306.08010.pdf`
  精读看什么：测试时不知道设备 ID 怎么办——我们外部测试恰恰没有可信的采集者标签先验。
- **Robust SED in bioacoustic sensor networks** — Lostanlen, Salamon, Farnsworth, Kelling et al., 2019. [arXiv](https://arxiv.org/abs/1905.08352) · `papers/E_domain-shift-confounds-evaluation/1905.08352.pdf`
  精读看什么：野外传感器网络的跨设备噪声与不变性——最接近"真实部署"的参照域。
- **Domain-Invariant Representation Learning of Bird Sounds** — Moummad, Serizel, Benetos, Farrugia, 2024. [arXiv](https://arxiv.org/abs/2409.08589) · `papers/E_domain-shift-confounds-evaluation/2409.08589.pdf`
  精读看什么：细粒度声音的域不变表征（对比/分布对齐）——与 2510.00346 对比选型。
- **Learning Domain-Robust Bioacoustic Representations for Mosquito Species Classification** — Hou, Liu, Shen, Roberts, 2025. [arXiv](https://arxiv.org/abs/2510.00346) · `papers/E_domain-shift-confounds-evaluation/2510.00346.pdf`
  精读看什么：跨采集装置的细粒度分类——**我们的问题在另一个物种上的镜像**，方法可直接对照。
- **Over-Parameterization and Generalization in Audio Classification** — Koutini, Eghbal-zadeh, Henkel, Schlüter et al., 2021. [arXiv](https://arxiv.org/abs/2107.08933) · `papers/E_domain-shift-confounds-evaluation/2107.08933.pdf`
  精读看什么：容量与泛化的关系——支持我们"冻结+小头、不要急着微调"路线的理论证据。
- **Data Leakage in Notebooks: Static Detection and Better Processes** — Yang, Brower-Sinning, Lewis, Kästner, 2022. [arXiv](https://arxiv.org/abs/2209.03345) · `papers/E_domain-shift-confounds-evaluation/2209.03345.pdf`
  精读看什么：泄漏模式分类——按清单审计我们的 pipeline。
- **LeakageDetector** — AlOmar, DeMario, Shagawat, Kreiser, 2025. [arXiv](https://arxiv.org/abs/2503.14723) · `papers/E_domain-shift-confounds-evaluation/2503.14723.pdf`
  精读看什么：开源泄漏审计工具——能否直接跑在我们的流程上。
- **Data Leakage and Evaluation Issues in Micro-Expression Analysis** — Varanka, Li, Peng, Zhao, 2022. [arXiv](https://arxiv.org/abs/2211.11425) · `papers/E_domain-shift-confounds-evaluation/2211.11425.pdf`
  精读看什么：小领域 ML 被泄漏与评估问题毁掉的案例研究——反面教材清单，逐条对照我们的报告。
- **Don't Push the Button! Data Leakage Risks in ML and Transfer Learning** — Apicella, Isgrò, Prevete, 2024. [arXiv](https://arxiv.org/abs/2401.13796) · `papers/E_domain-shift-confounds-evaluation/2401.13796.pdf`
  精读看什么：预训练/微调管线的泄漏通道——我们用了预训练模型，泄漏面在哪。
- **On the (Mis)Use of Machine Learning with Panel Data** — Cerqua, Letta, Pinto, 2024. [arXiv](https://arxiv.org/abs/2411.09218) · `papers/E_domain-shift-confounds-evaluation/2411.09218.pdf`
  精读看什么：面板/分组结构下的估计偏差——把"panel unit"换成"game/source group"读。
- **Evaluating Supervised ML Models: Principles, Pitfalls, Metric Selection** — Liu, Cabrera Martin, Trovati, Xu et al., 2026. [arXiv](https://arxiv.org/abs/2604.13882) · `papers/E_domain-shift-confounds-evaluation/2604.13882.pdf`
  精读看什么：评估设计清单——报告写完后逐项自检。
- **Towards a more realistic evaluation of ML models for bearing fault diagnosis** — Vieira, Bauler, Rosa, Silva, 2025. [arXiv](https://arxiv.org/abs/2509.22267) · `papers/E_domain-shift-confounds-evaluation/2509.22267.pdf`
  精读看什么：振动诊断（与冲击声同族）的评估纠偏——随机切分被高估多少，跨设备/跨工况怎么测。
- **A Toolkit for Detecting Spurious Correlations in Speech Datasets** — Gauder, Riera, Slachevsky, Forno et al., 2026. [arXiv](https://arxiv.org/abs/2604.26676) · `papers/E_domain-shift-confounds-evaluation/2604.26676.pdf`
  精读看什么：spurious correlation 自动审计工具——对标我们的 duration/collector 混杂诊断。
- **Mitigating Stethoscope-Induced Shortcuts in Respiratory Sound Classification** — Koo, Kim, Toikkanen, Kim, 2026. [arXiv](https://arxiv.org/abs/2605.29862) · `papers/E_domain-shift-confounds-evaluation/2605.29862.pdf`
  精读看什么：听诊器（设备）捷径的因果启发消除——**我们的采集混杂问题在医疗音频上的同构案例**，方法可搬。
- **Cross-individual generalizability of ML models for ball speed prediction in baseball pitching** — Takamido, Suzuki, Nakamoto, 2026. [arXiv](https://arxiv.org/abs/2605.05487) · `papers/E_domain-shift-confounds-evaluation/2605.05487.pdf`
  精读看什么：棒球传感域跨个体泛化的最新实证（个体级留出 vs 随机切分的差距）——**我们"跨采集者泛化"的直接参照**。

## 8. F — Simulation, synthesis, augmentation

- **Sound Synthesis, Propagation, and Rendering: A Survey** — Liu, Manocha, 2020. [arXiv](https://arxiv.org/abs/2011.05538) · `papers/F_simulation-synthesis-augmentation/2011.05538.pdf`
  精读看什么：模态合成谱系全图——我们的仿真器属于哪一支、缺哪一支（传播/渲染/麦克风响应）。
- **Rigid-Body Sound Synthesis with Differentiable Modal Resonators** — Diaz, Hayes, Saitis, Fazekas et al., 2022. [arXiv](https://arxiv.org/abs/2210.15306) · `papers/F_simulation-synthesis-augmentation/2210.15306.pdf`
  精读看什么：物理模态+可微=可控冲击声合成；可微性意味着可以做"声音→物理参数"的逆问题——**仿真管线的核心参照**。
- **DiffSound: Differentiable Modal Sound Rendering and Inverse Rendering** — Jin, Xu, Gao, Wu et al., 2024. [arXiv](https://arxiv.org/abs/2409.13486) · `papers/F_simulation-synthesis-augmentation/2409.13486.pdf`
  精读看什么：正向渲染+逆渲染统一框架——从声音反推物理参数在原理上成立，但需要什么条件（多视角、多冲击）。
- **Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos** — Su, Qian, Shlizerman, Torralba et al., 2023. [arXiv](https://arxiv.org/abs/2303.16897) · `papers/F_simulation-synthesis-augmentation/2303.16897.pdf`
  精读看什么：video→impact sound 条件生成——我们手里正好有 video（review evidence assets），能否用视频约束合成。
- **Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes** — Dou, Oh, Luo, Loquercio et al., 2025. [arXiv](https://arxiv.org/abs/2506.09989) · `papers/F_simulation-synthesis-augmentation/2506.09989.pdf`
  精读看什么：物理交互声音生成的最新进展——生成质量/物理一致性天花板。
- **NeuralSound: Learning-based Modal Sound Synthesis with Acoustic Transfer** — Jin, Li, Wang, Manocha, 2021. [arXiv](https://arxiv.org/abs/2108.07425) · `papers/F_simulation-synthesis-augmentation/2108.07425.pdf`
  精读看什么：可学习模态合成+声传递函数——合成声音与目标录音的对齐方法。
- **EnvGAN: Adversarial Synthesis of Environmental Sounds for Data Augmentation** — Madhu, Suresh K, 2021. [arXiv](https://arxiv.org/abs/2104.07326) · `papers/F_simulation-synthesis-augmentation/2104.07326.pdf`
  精读看什么：合成增广的经典收益/局限——增广到底补了什么分布。
- **SoundSpaces 2.0** — Chen, Schissler, Garg, Kobernik et al., 2022. [arXiv](https://arxiv.org/abs/2206.08312) · `papers/F_simulation-synthesis-augmentation/2206.08312.pdf`
  精读看什么：视觉-声学学习仿真平台的组织方式——仿真基建的工程参照。

## 9. G — Physics from sound (is our question estimable at all?)

- **The Sound of Water: Inferring Physical Properties from Pouring Liquids** — Bagad, Tapaswi, Snoek, Zisserman, 2024. [arXiv](https://arxiv.org/abs/2411.11222) · `papers/G_physics-from-sound/2411.11222.pdf`
  精读看什么：从连续声音流反推连续物理量（体积、流速）的表征与误差——比我们 200 ms 单瞬态更丰富的输入下，能做到什么精度。
- **Automatic Impact-sounding Acoustic Inspection of Concrete Structure** — Feng, Xiao, Hoxha, Song et al., 2021. [arXiv](https://arxiv.org/abs/2110.13125) · `papers/G_physics-from-sound/2110.13125.pdf`
  精读看什么：撞击声→材料/结构属性（结构健康监测域）——单次冲击声能携带多少内部状态信息；他们的特征与分类设计。
- **STReSSD: Sim-To-Real from Sound for Stochastic Dynamics** — Matl, Narang, Fox, Bajcsy et al., 2020. [arXiv](https://arxiv.org/abs/2011.03136) · `papers/G_physics-from-sound/2011.03136.pdf`
  精读看什么：声音域的 sim-to-real 差距来源与解法——直接回答"合成冲击声训的模型能否用于真实广播"。

## 10. H — Baseball & trajectory (wider project context)

- **Fine-grained Activity Recognition in Baseball Videos** — Piergiovanni, Ryoo, 2018. [arXiv](https://arxiv.org/abs/1804.03247) · `papers/H_baseball-trajectory-outcome/1804.03247.pdf`
  精读看什么：棒球视频细粒度识别基线——视频单模态能做到什么，音频在哪一步可能加分。
- **Dynamical Chaos in a Simple Model of a Knuckleball** — Nelson, Strauss, 2020. [arXiv](https://arxiv.org/abs/2009.05140) · `papers/H_baseball-trajectory-outcome/2009.05140.pdf`
  精读看什么：棒球飞行模型的敏感依赖——轨迹预测的物理难度上限，帮我们理解"接触时刻的信息最多能决定什么"。
- **Event-based Gaze Control for Real-time Spin Estimation in Professional Ball Games** — Hu, Schilling, Cavinato, Aydin et al., 2026. [arXiv](https://arxiv.org/abs/2606.26780) · `papers/H_baseball-trajectory-outcome/2606.26780.pdf`
  精读看什么：专业球类旋转估计的最新硬件路线——视觉域在做什么，音频的差异化机会在哪。

## 11. Open-access classics (downloaded, non-arXiv)

- **Varma & Simon 2006**, *Bias in error estimation when using cross-validation for model selection*, BMC Bioinformatics. `papers/X_open-access-classics/varma2006_nested_cv_bias.pdf` — nested CV 偏差的原始证据；模型选择与误差估计共用同一 CV 的乐观偏置。
- **Geras & Sutton 2013**, *Multiple-source cross-validation*, ICML (PMLR). `papers/X_open-access-classics/geras2013_multisource_cv.pdf` — 多源 CV 的形式化；为什么随机切分不是跨源评估。
- **Zhang et al. 2022**, *Impact Position Estimation for Baseball Batting with a Force-Irrelevant Vibration Feature*, Sensors. `papers/X_open-access-classics/zhang2022_impact_position_vibration.pdf` — **棒球击球冲击位置估计**：强度无关的模态比特征——本项目传统特征分组设计的直接先例。
- **Russell 2004**, *The sweet spot of a hollow baseball or softball bat*, ASA (HTML). `papers/X_open-access-classics/russell2004_sweetspot.html` — 碰撞<1 ms、球棒模态振动的物理基础。

## 12. Already in repo root (no re-download)

- `2211.06687v4.pdf` — CLAP (Wu et al. 2023), arXiv:2211.06687.
- `2405.07407v1.pdf` — PitcherNet (2024), arXiv:2405.07407.
- `2405.16296v1.pdf` — Baseball pitch trajectory tracking from single-view video (2024), arXiv:2405.16296.
- `Auston_Sterling_ISNN_-_Impact_ECCV_2018_paper.pdf` — ISNN: Impact Sound Neural Network, ECCV 2018.

## 13. Paywalled classics (link only, cited in project docs)

- **Collier 2001**, *The sounds of baseball: The bat–ball collision and the crack of the bat*, JASA. <https://doi.org/10.1121/1.4744893>
- **Collier, Kaliski, Sherwood 2004**, *Vibration and sound radiation of solid wood and tubular metal baseball bats as a function of ball-bat location*, JASA. <https://doi.org/10.1121/1.4808680>
- **Klatzky, Pai, Krotkov 2000**, *Perception of Material from Contact Sounds*, Presence. <https://doi.org/10.1162/105474600566907>
- **Kaufman, Rosset, Perlich 2011**, *Leakage in Data Mining*, KDD. <https://doi.org/10.1145/2020408.2020496>
- **Nathan 2008**, *The effect of spin on the flight of a baseball*, Am. J. Phys. — open PDF at <http://baseball.physics.illinois.edu/traj.html>
- **Adair 2002**, *The Physics of Baseball* (book) — background only.

## 14. 精读顺序（三轮）

**Wave 1 — 决策冲击最大（先读这 6 篇）**
1. 乒乓球旋转估计 (2409.11760) — 同构任务的天花板与设计
2. 棒球投球跨个体泛化 (2605.05487) — 跨个体泛化的现实差距
3. 呼吸音听诊器捷径 (2605.29862) — 设备混杂的因果消除
4. 蚊子跨装置分类 (2510.00346) — 跨采集装置细粒度分类的方法镜像
5. 刚体可微模态谐振器 (2210.15306) + DiffSound (2409.13486) — 仿真路线能否闭环
6. STReSSD sim-to-real (2011.03136) — 合成声音训练的迁移现实

**Wave 2 — 表征与模型路线（为"要不要换表征"收集证据）**
adaptive pooling (1804.10070)、JiTTER (2502.20857)、M2D 重读、ATST-Frame 重读、PEFT 三件套 (2211.02227, 2312.03694, 2402.00828)、few-shot 迁移 (2402.01274)、鸟叫 MAE (2504.12880)、探针方法论 (2010.13007)、过参数化 (2107.08933)、冲击位置振动特征 (Zhang 2022)。

**Wave 3 — 评估与混杂治理（所有结论的最后一道闸）**
Moscovich CV 偏差 (1901.08974)、相关数据 CV (1904.02438)、nested CV primer (2203.04552)、面板数据误用 (2411.09218)、轴承诊断评估纠偏 (2509.22267)、微表情泄漏案例 (2211.11425)、LeakageDetector (2503.14723)、DCASE 2020 设备泛化任务 (2005.14623)、IR 增广 (2305.07499)、频谱校正 (2105.11856)、Varma & Simon、Geras & Sutton。

其余按需：C 组体育音频作为领域背景随时翻阅；G 组回答"原理上可不可行"；H 组为大项目方向提供上下文。

## 15. How this feeds the direction decision

This map is raw material, not a decision. The decision — keep pushing fly/ground, pivot the method, or narrow the claim — belongs to `/grill-with-docs` (we are in a working directory with `CONTEXT.md`; the interview leaves ADRs and updates the glossary). Suggested sequence: finish Wave 1 together → take the concrete "aha or dead-end" findings into a grilling session → then either spec new work (`/to-spec` → `/to-tickets` → `/implement`) or record a narrowing of the research claim.

Collection artifacts: query log `papers/search.log`, candidate pool `papers/candidates.json` (389 candidates), download log `papers/download.log`, manifest `papers/download-manifest.json`, collector script `papers/collect.mjs`.
