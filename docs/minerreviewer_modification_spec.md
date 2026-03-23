
# MinerReviewer 修改说明书（完整替换版）

> 目标：将当前系统从“criteria-driven reviewer pipeline”升级为“memory-driven reviewer system”。
>
> 本说明书供 Claude Code 直接执行。请严格按本说明书进行重构，优先保证：
> 1. 架构方向正确
> 2. 数据结构清晰
> 3. 模块职责边界明确
> 4. 不破坏现有 CLI 的基本可用性
> 5. 所有新增能力都有最小可运行实现和测试

---

# 1. 改动理由（放在最前面）

当前仓库已经实现了一条基本可运行的 reviewer pipeline，大致是：

- retrieve
- mine criteria
- distill criteria
- rewrite criteria
- theme agents
- arbiter
- similarity scoring
- calibrate
- update memory

这条链路可以工作，但它更像一个：

**检索增强的 criteria-driven reviewer**

而不是我们真正要做的：

**experience-accumulating, memory-driven reviewer**

目前存在的核心问题如下。

## 1.1 Memory 不是系统核心，只是附属模块

当前 memory 主要是 policy criteria 的存储和更新。  
这意味着系统虽然会“记住一些标准”，但还不会真正“记住历史审稿经验”。

当前缺失的关键能力包括：

- 不会维护“相似论文 case”
- 不会维护“可复用 critique case”
- 不会维护“失败经验 / 误判经验”
- 不会把一次审稿蒸馏成结构化经验再迁移到下一次

因此它不是一个真正会成长的 reviewer。

---

## 1.2 相似论文召回存在，但还不是论文记忆库

当前检索能召回“相似论文”，但本质上只是 paper retrieval：

- 论文以 `title + abstract` 为主建立索引
- 查询也主要基于 `title + abstract`
- 召回结果是原始论文对象，不是结构化 case

这和真正需要的“论文记忆库 / paper case memory”差距很大。

真正需要的是：

- 一篇历史论文被抽象成 `paper case`
- case 不只包含论文文本，还包含：
  - paper signature
  - 历史 strengths / weaknesses
  - 决定性 critique
  - venue 语境
  - failure pattern
  - 可复用 criteria

这样相似论文召回才会成为真正的经验检索，而不是资料搜索。

---

## 1.3 最终决策目前过度依赖“拟合历史评分分布”

当前决策链路大致是：

`Arbiter -> SimilarReviewScoring -> Calibrator`

其中存在几个明显问题：

1. SimilarReviewScoring 会根据相似历史评论重新调整 `raw_rating` 和 `decision_recommendation`
2. 这使系统更像在“拟合历史评分分布”
3. 对真正新颖、历史近邻稀少的论文不友好
4. 会削弱“基于当前论文内容做判断”的主线

我们不希望做出一个“history-fitted scoring system”，而希望做出一个：

**利用经验辅助判断，但不被历史分布绑架的 reviewer**

因此：

- 相似历史评论应只做一致性检查
- calibration 只做概率校准
- 最终 decision 应由内容判断主导

---

## 1.4 当前 update_memory 太轻，不足以支撑 test-time learning

当前 memory update 更接近：

- 更新 policy criteria 的统计
- 记录分数结果
- 维护少量状态

这不足以支撑“test-time 持续成长”。

真正需要的是：

- 从当前审稿轨迹中提取可复用经验
- 将经验分层写入 short-term / long-term memory
- 区分：
  - policy memory
  - paper case memory
  - critique memory
  - failure memory

只有这样，系统才会真正“越审越会审”。

---

## 1.5 当前系统仍然较强依赖 venue hardcode

例如 ICLR 的标准、theme、decision mapping 等在多个模块中存在显式或隐式硬编码。

这会导致：

- 难以做跨 venue 迁移
- 难以解释“venue-specific adaptation”
- 难以支撑你真正的研究目标

所以本次重构需要明确：

- global memory
- venue-specific memory
- case memory
- failure memory

并且让 venue 成为可配置、可切换、可评测的一级对象。

---

# 2. 总体改造目标

改造后的系统目标架构：

`PaperParser -> MultiChannelRetriever -> CriteriaPlanner -> ThemeAgents -> Arbiter -> Verifier -> ScoreConsistencyChecker -> Calibrator -> ExperienceDistiller -> MemoryEditor`

核心思想：

1. **先从 memory 中检索经验，再执行审稿**
2. **多智能体围绕 criteria 和 case 工作，而不是自由发挥**
3. **审稿结束后，从本次轨迹中蒸馏经验并写回 memory**
4. **历史经验只辅助判断，不直接替代当前论文内容判断**

---

# 3. 非目标（本次不要做）

以下内容不是本轮重构重点：

1. 不做大规模训练框架改造
2. 不做模型微调
3. 不做复杂前端
4. 不做数据库系统全面替换
5. 不追求一次性解决所有 venue 的所有特殊规则
6. 不强求最优性能，优先建立正确架构

---

# 4. 改造后的目标架构

## 4.1 新主流程

```text
Paper
  -> PaperParser
  -> VenueProfiler
  -> MultiChannelRetriever
      -> PolicyMemory
      -> PaperCaseMemory
      -> CritiqueMemory
      -> FailureMemory
      -> Supporting/Competing Papers
  -> CriteriaPlanner
  -> ThemeAgents
  -> Arbiter
  -> Verifier
  -> ScoreConsistencyChecker
  -> Calibrator
  -> Final Review

Trace
  -> ExperienceDistiller
  -> MemoryEditor
  -> ShortTermMemory
  -> LongTermMemory
````

---

## 4.2 模块职责简述

### PaperParser

将论文解析为结构化 `PaperSignature`。

### VenueProfiler

根据 venue config / rubric / profile 输出当前 venue 的审稿偏好。

### MultiChannelRetriever

从多个 memory/source 检索：

* policy
* paper cases
* critique cases
* failure cases
* supporting papers

### CriteriaPlanner

决定当前论文激活哪些 criteria，不再每次从零开始挖所有标准。

### ThemeAgents

分主题生成 strengths / weaknesses / questions / evidence notes。

### Arbiter

综合多个 agents 的结论，生成：

* strengths
* weaknesses
* raw_rating
* decision_recommendation
* decision_rationale

### Verifier

检查：

* 分数和文字是否一致
* critique 是否有证据
* 是否 venue mismatch
* 是否过于模板化

### ScoreConsistencyChecker

利用相似历史 case / review 做一致性检查，只发 warning，不直接改分。

### Calibrator

给出 acceptance / borderline / rejection 的概率估计，不覆盖主 decision。

### ExperienceDistiller

从本次审稿中提取可复用经验。

### MemoryEditor

决定哪些经验进入 short-term / long-term memory，哪些应该 reject / merge / downweight。

---

# 5. 数据结构改造

---

## 5.1 新增 `PaperSignature`

文件建议：`src/common/types.py` 或新增 `src/common/signatures.py`

```python
class PaperSignature(BaseModel):
    paper_type: str | None = None
    tasks: list[str] = []
    domain: str | None = None
    method_family: list[str] = []
    main_claims: list[str] = []
    claim_strength: str | None = None
    datasets: list[str] = []
    evaluation_style: list[str] = []
    has_ablation: bool | None = None
    has_human_eval: bool | None = None
    has_efficiency_eval: bool | None = None
    baseline_coverage: str | None = None
    citation_keywords: list[str] = []
    risk_profile: list[str] = []
```

作用：

* 支撑更强的相似论文召回
* 支撑 criteria planning
* 支撑 case construction

---

## 5.2 新增 `PaperCase`

```python
class PaperCase(BaseModel):
    case_id: str
    paper_id: str | None = None
    venue_id: str | None = None
    year: int | None = None
    title: str
    abstract: str
    paper_signature: PaperSignature
    top_strengths: list[str] = []
    top_weaknesses: list[str] = []
    decisive_issues: list[str] = []
    review_consensus: str | None = None
    decision: str | None = None
    rating: float | None = None
    source_review_ids: list[str] = []
    source_meta_review: str | None = None
    transferable_criteria: list[str] = []
    failure_patterns: list[str] = []
```

作用：

* 构建真正的论文记忆库
* 相似论文召回不再只返回原始 paper
* 为后续 agent 提供可复用经验

---

## 5.3 扩展 `ExperienceCard`

当前 `ExperienceCard` 不能只支持 `policy`。
请扩展为多类型：

```python
class ExperienceCard(BaseModel):
    card_id: str
    kind: Literal["policy", "case", "critique", "failure"]
    scope: Literal["global", "venue", "paper_type", "domain"]
    venue_id: str | None = None
    theme: str
    content: str
    trigger: list[str] = []
    utility: float = 0.5
    confidence: float = 0.5
    use_count: int = 0
    source_ids: list[str] = []
    metadata: dict[str, Any] = {}
```

---

## 5.4 新增 `ScoreConsistencyReport`

```python
class ScoreConsistencyReport(BaseModel):
    similar_review_count: int
    mean_rating: float | None = None
    median_rating: float | None = None
    min_rating: float | None = None
    max_rating: float | None = None
    decision_distribution: dict[str, int] = {}
    consistency_level: str = "unknown"
    warning: str | None = None
    justification_needed: bool = False
```

---

## 5.5 新增 `DecisionVerificationReport`

```python
class DecisionVerificationReport(BaseModel):
    passed: bool
    score_text_alignment: str
    evidence_support_level: str
    venue_alignment_level: str
    warnings: list[str] = []
    requires_revision: bool = False
```

---

## 5.6 新增 `CalibrationResult`

```python
class CalibrationResult(BaseModel):
    calibrated_rating: float | None = None
    acceptance_likelihood: float | None = None
    borderline_likelihood: float | None = None
    rejection_likelihood: float | None = None
    calibration_confidence: float | None = None
```

---

## 5.7 新增 `FinalDecisionReport`

```python
class FinalDecisionReport(BaseModel):
    raw_rating: float
    calibrated_rating: float | None = None
    decision_recommendation: str
    acceptance_likelihood: float | None = None
    borderline_likelihood: float | None = None
    rejection_likelihood: float | None = None
    decision_rationale: str
    score_consistency_report: ScoreConsistencyReport | None = None
    verification_report: DecisionVerificationReport | None = None
```

---

# 6. 文件级改造要求

---

## 6.1 `src/main.py`

### 改造目标

保留现有 CLI 习惯，但增加新能力的入口和兼容。

### 要求

1. 保持现有命令仍然能跑
2. 增加 `build_cases` 命令，用于从 paper + reviews 构建 `PaperCase`
3. 增加可选参数，允许选择：

   * 是否启用 case retrieval
   * 是否启用 verifier
   * 是否启用 score consistency
   * calibration 模式

---

## 6.2 `src/common/types.py`

### 改造目标

升级所有核心对象，支持：

* paper signature
* paper case
* 多类型 memory
* decision verification
* score consistency
* richer calibration result

### 要求

* 保持向后兼容，必要时提供默认值
* 避免破坏现有测试
* 所有新增类型都应加注释

---

## 6.3 新增 `src/pipeline/parse_paper.py`

### 新模块职责

输入论文对象，输出 `PaperSignature`。

### 最小实现

可先基于 LLM 或简单 heuristics 提取：

* paper_type
* tasks
* method_family
* claim_strength
* evaluation_style
* baseline_coverage
* risk_profile

### 要求

* 先做 MVP，不追求完美
* 输出稳定的结构化字段
* 对缺失字段要有默认值

---

## 6.4 `src/pipeline/retrieve.py`

### 现状问题

当前 `Retriever.retrieve()` 返回：

* related_papers
* related_reviews
* unrelated_papers
* venue_policy

这还不是多通道 retrieval。

### 改造目标

改为返回结构化 `RetrievalBundle`，至少包含：

```python
class RetrievalBundle(BaseModel):
    similar_paper_cases: list[PaperCase] = []
    supporting_or_competing_papers: list[Any] = []
    critique_cases: list[ExperienceCard] = []
    policy_cards: list[ExperienceCard] = []
    failure_cards: list[ExperienceCard] = []
    venue_policy: Any | None = None
    trace: dict[str, Any] = {}
```

### 要求

* 相似论文优先从 `PaperCaseMemory` 检索
* 原始 paper retrieval 仍可保留，作为 supporting papers
* `unrelated_papers` 可以移除或降级为分析用途
* 检索策略支持：

  * metadata filter
  * vector retrieval
  * rerank

### 最小实现建议

第一版可先做到：

* `similar_paper_cases`：从 case store 向量检索
* `policy_cards`：从 memory store 取 venue/global policy
* `critique_cases`：先从历史 reviews 中蒸馏少量 case
* `failure_cards`：先接入现有 coverage gaps 逻辑

---

## 6.5 `src/pipeline/mine_criteria.py`

### 现状问题

当前更像“每次从检索材料里现挖标准”，而不是“从 memory 中选择 criteria”。

### 改造目标

保留 criteria mining 能力，但将其降级为：

* 当 memory 覆盖不足时补充新 criteria
* 不再承担主标准来源角色

### 新要求

输出应区分：

* retrieved criteria
* newly mined criteria

建议新增字段：

* source = `memory` | `mined`
* priority
* trigger_condition
* evidence_requirement

---

## 6.6 新增 `src/pipeline/plan_criteria.py`

### 新模块职责

根据：

* current paper signature
* venue profile
* retrieved policy cards
* similar paper cases
* failure cards

生成本次审稿要激活的 criteria plan。

### 输出建议

```python
class ActivatedCriterion(BaseModel):
    theme: str
    criterion: str
    source: str
    priority: int
    trigger_reason: str
    required_evidence: list[str] = []
    owner_agent: str | None = None
```

### 作用

* 让多智能体“围绕计划工作”
* 降低 theme agents 的自由漂移
* 增强解释性

---

## 6.7 `src/pipeline/rewrite_criteria.py`

### 改造目标

保留，但角色从“重写标准文本”升级为：

* 将 activated criteria 改写为 paper-specific critique checklist
* 为各 agent 生成更具体的检查问题

### 要求

每条改写后的 criteria 应尽量包含：

* 关注点
* 触发原因
* 要检查的证据
* 可能的 failure mode

---

## 6.8 `src/agents/theme_agent.py`

### 改造目标

让 theme agents 接受更明确的输入，不再只依赖通用 criteria。

### 新输入建议

* paper
* paper_signature
* activated_criteria
* supporting papers
* similar paper cases
* venue profile

### 建议主题

不要再保留过于泛的 `Quality` 主题，建议向以下方向收敛：

* novelty
* soundness
* experiments
* related_work_evidence
* clarity
* venue_alignment

### 要求

每个 agent 输出：

* strengths
* weaknesses
* open questions
* evidence notes
* confidence

---

## 6.9 `src/agents/arbiter_agent.py`

### 现状问题

Arbiter 已经承担主决策，但缺少显式 rationale 字段。

### 改造目标

Arbiter 仍然负责主判断，但必须输出更可解释的结构。

### 新要求

输出增加：

* `decision_rationale`
* `score_rationale`
* `key_decisive_issues`

### 原则

* 最终 `decision_recommendation` 主来源仍是 Arbiter
* 不要让后续 similarity 模块覆盖它

---

## 6.10 新增 `src/pipeline/verify_decision.py`

### 新模块职责

在 Arbiter 之后，检查：

* 分数和文字是否一致
* critique 是否有足够证据
* 是否 venue mismatch
* 是否模板化、空泛、重复

### 要求

Verifier 不重新写 review，只做：

* 诊断
* warning
* 是否建议 Arbiter 进行一次修订

### 最小实现

第一版可以先只做规则 + LLM 混合判断。

---

## 6.11 `src/pipeline/review_pipeline.py`

### 这是最关键的改造文件

### 当前问题

存在：

* SimilarReviewScoring 直接调分
* calibration 边界不清
* memory update 偏弱
* 缺少 verifier

### 改造要求

将主流程修改为：

1. parse paper
2. retrieve multi-channel bundle
3. plan criteria
4. theme agents
5. arbiter
6. verifier
7. score consistency check
8. calibrate
9. distill experience
10. edit/write memory

### 必须做的改动

* 删除或重写 `_score_with_similar_reviews()`
* 新增 `_check_score_consistency()`
* 将 calibration 结果变成附加输出，而不是主决策覆盖器
* 在 final result 中加入：

  * verification report
  * score consistency report
  * calibration result

---

## 6.12 新增 `src/pipeline/check_score_consistency.py`

### 新模块职责

根据相似历史评论 / case，检查当前 Arbiter 给出的分数和 decision 是否明显偏离近邻分布。

### 注意

它只能做：

* 参考
* 预警
* 解释要求

### 不允许

* 不允许直接覆盖 `raw_rating`
* 不允许直接覆盖 `decision_recommendation`

### 最小规则建议

* 相似 case 少于 3 个时，不做强结论
* 偏差过大时：

  * `warning = "..."`
  * `justification_needed = True`

---

## 6.13 `src/pipeline/calibrate.py`

### 当前问题

当前是 accept vs non-accept 的极简二值映射，不够细。

### 改造目标

升级为：

* ordinal calibration
  或
* three-way calibration

### 输出

至少包括：

* acceptance_likelihood
* borderline_likelihood
* rejection_likelihood

### 原则

Calibration 不覆盖主 decision，只做辅助概率输出。

---

## 6.14 `src/pipeline/update_memory.py`

### 当前问题

只更新 policy，不足以支撑 memory-driven reviewer。

### 改造目标

将其拆为两步：

1. `distill_experience.py`
2. `memory_editor.py`

---

## 6.15 新增 `src/pipeline/distill_experience.py`

### 新模块职责

从本轮 review trace 中蒸馏：

* reusable policy updates
* paper case
* critique case
* failure case

### 例子

* 某条 criteria 在本轮特别有用
* 某个 critique 模式再次出现且有效
* 某类论文常见的 missing baseline 模式
* 某个 venue 中高频失败点

---

## 6.16 新增 `src/pipeline/memory_editor.py`

### 新模块职责

决定：

* 哪些经验写入 short-term memory
* 哪些能进入 long-term memory
* 哪些需要 merge
* 哪些应该丢弃

### 最低要求

实现 admission gate：

* 不能生成什么就写什么
* 有 warning / 低置信的经验先进入 short-term
* 高 utility / 高 confidence / 多次复现的经验才进入 long-term

---

## 6.17 `src/storage/memory_store.py`

### 当前问题

相似判定过于简单，且只适合 policy card。

### 改造目标

升级为支持多类型 memory card，并为相似性判断引入更稳的机制。

### 要求

* 支持 `policy` / `case` / `critique` / `failure`
* 支持按 `kind + theme + scope` 查询
* 相似判定不再只靠简单 token overlap
* 第一版可改成：

  * embedding 相似度
  * 辅助 lexical gate
  * metadata filter

---

## 6.18 新增 `src/storage/case_store.py`

### 新模块职责

专门管理 `PaperCase`。

### 要求

支持：

* add case
* get case
* search similar cases
* filter by venue / year / domain / paper_type
* deduplicate / merge

### 最小实现

先支持 jsonl 或本地存储 + 向量索引接口。

---

# 7. 配置文件改造

---

## 7.1 配置安全问题必须先修

### 要求

* 移除所有明文 API key
* 移除硬编码 host/base_url
* 改为环境变量读取
* 提供 `.example` 配置

### 必须做

* 新增 `configs/iclr.example.yaml`
* README 中不再出现敏感地址
* 所有 key/host 都支持 env override

---

## 7.2 decision scoring 配置改造

将：

```yaml
decision_scoring:
  use_similarity: true
```

替换为：

```yaml
score_consistency:
  enabled: true
  min_similar_reviews: 3
  max_rating_deviation: 1.5
  require_justification_on_mismatch: true
```

---

## 7.3 calibration 配置改造

```yaml
calibration:
  enabled: true
  mode: ordinal   # ordinal | three_way
  use_decision_labels: true
  use_rating_labels: true
```

---

## 7.4 retrieval 配置改造

建议新增：

```yaml
retrieval:
  use_case_memory: true
  case_top_k: 5
  supporting_paper_top_k: 10
  critique_case_top_k: 5
  failure_case_top_k: 5
  metadata_filter_enabled: true
  rerank_enabled: true
```

---

## 7.5 memory 配置改造

```yaml
memory:
  short_term_enabled: true
  long_term_enabled: true
  write_failure_cases: true
  write_paper_cases: true
  write_critique_cases: true
  write_policy_cards: true
  long_term_admission_min_utility: 0.7
  long_term_admission_min_confidence: 0.7
```

---

# 8. 决策与评分模块专项要求

---

## 8.1 改造原则

改造后的决策链路必须是：

`Arbiter -> Verifier -> ScoreConsistencyChecker -> Calibrator`

其中职责边界必须明确：

* Arbiter：主决策
* Verifier：检查和约束
* ScoreConsistencyChecker：参考提醒
* Calibrator：概率输出

---

## 8.2 必须删除的行为

代码中不能再存在：

* 用相似历史评论直接覆盖 `raw_rating`
* 用相似历史评论直接覆盖 `decision_recommendation`
* 用 calibration 直接替换 Arbiter 的主 decision

---

## 8.3 ScoreConsistencyChecker 的定位

它只能输出：

* 相似 case 的评分统计
* 当前分数偏离程度
* warning
* 是否需要 justification

不能做：

* 改分
* 改 decision

---

## 8.4 Verifier 的定位

Verifier 需要检查：

1. score-text alignment
2. evidence support
3. venue alignment
4. critique specificity

它可以要求 Arbiter 二次修订，但不能完全替代 Arbiter。

---

## 8.5 Calibrator 的定位

Calibrator 只做：

* acceptance probability
* borderline probability
* rejection probability

它的结果用于：

* 展示
* 评测
* 风险提示

不用于：

* 覆盖主 decision

---

# 9. 论文记忆库 / case memory 构建要求

---

## 9.1 case 怎么构建

每个 case 需要来自：

* paper
* reviews
* rating / decision
* 有条件的话加 meta-review / rebuttal

构建步骤：

1. parse paper -> `PaperSignature`
2. parse reviews -> critique units
3. aggregate strengths / weaknesses
4. identify decisive issues
5. distill transferable criteria / failure patterns
6. save as `PaperCase`

---

## 9.2 critique units 的构建

建议从 review 中抽取：

* theme
* polarity
* specificity
* evidence-backed or not
* actionable or not

第一版可由 LLM 提取，后续再做规则增强。

---

## 9.3 case retrieval 的使用方式

对当前论文：

1. 先构建 `PaperSignature`
2. 检索最相似 `PaperCase`
3. 将 case 中的：

   * decisive issues
   * transferable criteria
   * failure patterns
     喂给 criteria planner 和 evidence agent

---

# 10. 测试要求

---

## 10.1 必须新增的测试

新增以下测试文件：

* `tests/test_paper_parser.py`
* `tests/test_case_store.py`
* `tests/test_criteria_planner.py`
* `tests/test_score_consistency.py`
* `tests/test_decision_verifier.py`
* `tests/test_calibrator_multiclass.py`
* `tests/test_memory_editor.py`
* `tests/test_review_pipeline_smoke.py`

---

## 10.2 测试重点

### PaperParser

* 输入 paper，能输出结构化 signature
* 缺字段时不会崩

### CaseStore

* 可添加 case
* 可按相似度召回 case
* 可按 venue / type 过滤

### ScoreConsistencyChecker

* 不会改 `raw_rating`
* 不会改 `decision_recommendation`
* 会正确输出 warning

### DecisionVerifier

* 能检测 score-text mismatch
* 能识别 evidence 不足

### Calibrator

* 不再只输出 accept vs reject
* 能输出多路概率

### ReviewPipeline

* 完整主流程可运行
* 输出中包含：

  * verification_report
  * score_consistency_report
  * calibration result

---

# 11. 迁移策略

---

## Phase 1：最低可运行版

先完成：

* PaperSignature
* PaperCase
* CaseStore
* PaperParser
* RetrievalBundle
* ScoreConsistencyChecker
* DecisionVerifier
* 更合理的 Calibrator
* 新主流程串联

目标：

* 不破坏主 CLI
* 先跑通 end-to-end

---

## Phase 2：增强 memory update

完成：

* ExperienceDistiller
* MemoryEditor
* short-term / long-term memory
* paper case write-back
* failure memory write-back

---

## Phase 3：跨 venue 泛化

完成：

* venue profiles
* rubric mapping
* 去 ICLR hardcode
* leave-one-venue-out 评测支持

---

# 12. 验收标准

完成本次重构后，系统应满足以下标准：

## 架构层

1. 存在 `PaperParser`
2. 存在 `PaperCaseMemory`
3. 存在多通道 retrieval
4. 存在 `Verifier`
5. 存在 `ScoreConsistencyChecker`
6. 存在更合理的 calibration
7. 存在 experience distillation + memory editing

## 决策层

8. 不存在“相似历史评论直接改分”的逻辑
9. Arbiter 是主 decision 来源
10. Verifier 只做检查和修正约束
11. ScoreConsistencyChecker 只做提醒
12. Calibrator 只做概率输出

## memory 层

13. Memory 不再只支持 `policy`
14. 支持 `case` / `critique` / `failure`
15. 存在 short-term / long-term 区分
16. 存在 write gate

## 工程层

17. 配置中无明文敏感信息
18. README 与默认配置一致
19. 新增测试可通过
20. 主流程 smoke test 可通过

---

# 13. 实施注意事项

1. 优先做“正确架构”，不要先追求 prompt 微调
2. 保留足够 trace，方便后续 debug 和论文分析
3. 所有新增字段尽量加注释
4. 所有模块输出尽量结构化
5. 对外部依赖保守处理，不要引入过重框架
6. 先实现 MVP，再做性能优化

---

# 14. 给 Claude Code 的执行要求

请按以下顺序执行：

1. 先完成数据结构改造
2. 再完成 `PaperParser` 和 `CaseStore`
3. 再改 `Retriever` 为多通道
4. 再引入 `CriteriaPlanner`
5. 再重构 `review_pipeline`
6. 再加入 `Verifier`、`ScoreConsistencyChecker`、新 `Calibrator`
7. 最后补 `ExperienceDistiller` 和 `MemoryEditor`
8. 补测试
9. 更新 README
10. 清理配置安全问题

每完成一个阶段，请保证：

* 代码可运行
* 不引入明显破坏性改动
* 有最小测试支撑

---

# 15. 最终一句话要求

请将当前项目从：

**一个会检索、会挖 criteria、会多智能体讨论的 reviewer pipeline**

重构为：

**一个能够利用论文案例库、审稿经验库和可演化 memory，在测试时持续积累和迁移经验的 memory-driven reviewer system**


