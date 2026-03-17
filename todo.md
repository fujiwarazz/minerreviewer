# MinerReviewer 修改说明书（交给 Claude Code 执行）

## 0. 改动理由（必须先看）

本次修改的目标，不是简单补几个模块，而是把当前系统从 **criteria-driven reviewer** 改造成 **memory-driven reviewer**。

当前仓库已经具备一条完整的审稿主链路：

`retrieve -> mine_criteria -> distill_criteria -> rewrite_criteria -> theme agents -> arbiter -> similarity scoring -> calibrate -> update_memory`

这个骨架能跑，也有明确的工程分层，但存在以下根本问题：

### 0.1 memory 不是一等公民
当前系统的 memory 只保存 `policy` 类型的卡片，`ExperienceCard.kind` 固定为 `"policy"`，`update_memory()` 也只会把 `policy_criteria` 写回 store。系统没有真正的：
- 论文 case memory
- critique case memory
- failure / reflection memory
- 跨会议可迁移的 criteria memory

结果是：系统每次还是“重新挖 criteria”，而不是“先检索经验，再执行审稿，再把新经验写回”。

### 0.2 检索层还不是 case retrieval
当前 paper retrieval 使用 `title + abstract` 的 embedding 进行检索，返回 `related_papers / related_reviews / unrelated_papers`。这对于“找近邻文献”是够的，但对“找**可复用历史案例**”还不够，因为没有 paper signature、review signature，也没有区分：
- 相似论文案例
- 相似 critique 案例
- 支撑/竞争文献
- failure case

### 0.3 当前 update_memory 太弱
当前 memory update 的逻辑是：如果分数满足 `stable` 或 `borderline` 条件，就把 policy criteria 写回 memory store。它缺少：
- experience distillation
- short-term / long-term memory 分层
- write gate / verifier
- case / failure / critique 写回
- merge / garbage control

### 0.4 score similarity 模块容易把系统带偏
当前流程会在 Arbiter 之后用“相似历史评论”让 LLM 调整分数。这会让系统偏向拟合历史评分分布，而不是提升 critique 质量。这个模块更适合降级成 `ScoreConsistencyChecker`，作为检查器而不是调分器。

### 0.5 目前系统仍然 ICLR-hardcoded
`mine_criteria.py` 里存在显式的 `ICLR_CRITERIA_THEMES`，policy prompt 也直接写死为从 `ICLR reviews` 抽规则。这样无法支撑你想要的跨会议迁移目标。

---

## 1. 这次改动的总目标

把当前项目重构为：

**Training-free, memory-driven, conference-aware multi-agent reviewer with online memory evolution.**

具体来说，系统要具备：

1. **多通道 memory**：policy / paper case / critique / failure
2. **多通道 retrieval**：从不同 memory / corpus 中检索不同用途的对象
3. **criteria planning**：优先从 memory 中选择 criteria，而不是每次从头挖
4. **在线 memory update**：在 test time 蒸馏经验并更新 memory，但不更新模型参数
5. **cross-venue 能力**：不再把 ICLR 写死在代码里
6. **更稳的评测与测试**：能比较 no-memory / static-memory / dynamic-memory

---

## 2. 这次改动的非目标（不要做）

以下内容本次不要做，避免范围失控：

1. **不要引入 finetuning / SFT / RLHF / LoRA**
   - 本项目保持 training-free。

2. **不要直接把所有模块推翻重写**
   - 尽量保留现有骨架和 CLI。
   - 优先增量重构，而不是一次性重写全部逻辑。

3. **不要立即支持很多 venue 的复杂 schema**
   - 先把 ICLR 抽象成通用 venue profile，再提供 1~2 个 profile 示例即可。

4. **不要把 score checker 做成强制调分器**
   - 这次应该把它降级为检查器。

5. **不要删除已有的 tests**
   - 在此基础上扩展。

---

## 3. 修改后的目标架构

新架构应当变成下面这条主线：

```text
Paper -> PaperParser -> VenueProfiler
      -> MultiChannelRetriever
         -> PaperCaseMemory
         -> CritiqueMemory
         -> PolicyMemory
         -> FailureMemory
         -> Literature Corpus
      -> CriteriaPlanner
      -> ThemeAgents
      -> Arbiter
      -> Verifier / MetaReviewer
      -> ScoreConsistencyChecker
      -> Calibrator
      -> Final Review

Trace -> ExperienceDistiller
      -> MemoryEditor
      -> ShortTermMemory
      -> LongTermMemory
```

### 3.1 主流程职责
- **PaperParser**：把当前论文抽成结构化 paper signature
- **VenueProfiler**：加载会议 profile / rubric / style guideline
- **MultiChannelRetriever**：从不同来源取不同对象
- **CriteriaPlanner**：从 memory 中选 criteria，并补充缺失 criteria
- **ThemeAgents**：分主题审稿
- **Arbiter**：汇总 strengths / weaknesses / rating / decision
- **Verifier**：检查证据不足、venue mismatch、模板化意见
- **ScoreConsistencyChecker**：只检查分数是否异常，不直接改分
- **Calibrator**：输出 acceptance likelihood

### 3.2 副流程职责
- **ExperienceDistiller**：把本轮 trace 压成可复用经验
- **MemoryEditor**：决定写到 short-term 还是 long-term
- **LongTermMemory**：保存稳定的 case / critique / policy / failure

---

## 4. 优先级与实施顺序


### P1（核心方法改造）
4. 新增 PaperParser
5. 把 Retriever 改成多通道检索
6. 新增 CriteriaPlanner
7. 新增 ExperienceDistiller + MemoryEditor
8. 重构 update_memory

### P2（增强与稳定性）
9. 新增 Verifier / MetaReviewer
10. SimilarReviewScoring 改为 ScoreConsistencyChecker
11. venue profile 抽象
12. 增加测试与迁移脚本

---

## 5. 文件级修改说明

下面是 Claude Code 应该执行的文件级改动。

---

### 5.1 `configs/iclr.yaml`

#### 要改什么
1. 新增 memory 分层配置
2. 新增 venue profile 配置
3. 新增 parser / retriever / verifier / memory_editor 的配置项
4. 将 `decision_scoring.use_similarity` 改造成 `score_consistency.enabled`

#### 建议改成
```yaml
venue:
  id: "ICLR"
  profile_path: "configs/venues/iclr.yaml"

retrieval:
  top_k_papers: 6
  top_k_reviews: 12
  top_k_cases: 8
  top_k_failures: 5
  top_k_supporting_papers: 6
  unrelated_k: 0
  similarity_threshold: 0.35

paper_parser:
  enabled: true
  max_fulltext_chars: 12000

score_consistency:
  enabled: true
  top_k: 8
  mode: "warn_only"

memory:
  short_term_path: "data/processed/memory_short_term.json"
  long_term_path: "data/processed/memory_long_term.json"
  similarity_threshold: 0.75
  stable_margin: 0.5
  borderline_low: 0.45
  borderline_high: 0.6
  promote_min_uses: 2
  promote_min_confidence: 0.7

llm:
  backend: "openai"
  model: "qwen-plus"
  temperature: 0.1
  base_url_env: "OPENAI_BASE_URL"
  api_key_env: "OPENAI_API_KEY"
```

#### 为什么
- 当前配置有安全风险。
- 当前配置不支持新架构的 memory 分层与多通道检索。
- 需要把实验参数显式化，方便比较 static vs dynamic memory。

#### 验收标准
- 新模块的配置均可读取。

---

### 5.2 新增 `configs/iclr.example.yaml`

#### 要改什么
新增公开可提交的 example 配置文件。

#### 为什么
- 保护敏感信息。
- 给 README 和复现实验提供标准配置模板。

#### 验收标准
- 用户拷贝 `iclr.example.yaml` 后，填入环境变量即可运行。

---

### 5.3 README

#### 要改什么
1. 修正“abstract-only by default”与实际配置不一致的问题。
2. 补充新的模块说明：PaperParser / MultiChannelRetriever / CriteriaPlanner / ExperienceDistiller。
3. 补充 `.env` / 环境变量配置方法。
4. 增加“迁移说明”：旧 memory store 如何升级。
5. 增加“实验模式”说明：no-memory / static-memory / dynamic-memory。

#### 为什么
- 当前 README 与默认配置不一致，会误导用户。
- 文档必须反映新架构。

#### 验收标准
- README 中的 quickstart 可以真实跑通。
- README 与配置行为一致。

---

### 5.4 `pyproject.toml`

#### 要改什么
1. 保留当前入口脚本兼容性
2. 新增更标准的包入口（如果进行目录规范化）
3. 如新增依赖（例如 `python-dotenv`），在这里声明

#### 推荐方案
- 短期保留：
  ```toml
  [project.scripts]
  peerreviewer = "main:main"
  ```
- 如果改包结构，再升级为：
  ```toml
  [project.scripts]
  peerreviewer = "minerreviewer.cli:main"
  ```

#### 为什么
- 先保持 CLI 兼容，减少破坏性改动。

#### 验收标准
- 原有命令仍能运行。
- 新目录结构下入口不出错。

---

### 5.5 `src/common/types.py`

#### 要改什么
扩展数据模型，重点是让 memory 能表示多种对象。

#### 当前问题
- `ExperienceCard.kind` 只有 `"policy"`
- `RetrievalBundle` 不够表达多通道检索结果

#### 新增/修改的数据结构

##### 5.5.1 扩展 `ExperienceCard`
```python
class ExperienceCard(BaseModel):
    card_id: str
    kind: Literal["policy", "case", "critique", "failure"]
    scope: Literal["global", "venue", "paper_type", "domain"] = "venue"
    venue_id: str | None = None
    theme: str
    content: str
    trigger: list[str] = Field(default_factory=list)
    utility: float = 0.5
    confidence: float = 0.5
    use_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    active: bool = True
    version: int = 1
    source_ids: list[str] = Field(default_factory=list)
    source_trace: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

##### 5.5.2 新增 `PaperSignature`
```python
class PaperSignature(BaseModel):
    paper_type: str | None = None
    task: str | None = None
    method_family: str | None = None
    claim_strength: str | None = None
    experiment_style: str | None = None
    baseline_density: str | None = None
    risk_tags: list[str] = Field(default_factory=list)
```

##### 5.5.3 新增 `PaperCase`
```python
class PaperCase(BaseModel):
    case_id: str
    paper_id: str
    venue_id: str | None = None
    year: int | None = None
    title: str
    abstract: str
    paper_signature: PaperSignature
    review_signature: dict[str, Any] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    rating: float | None = None
    decision: str | None = None
```

##### 5.5.4 扩展 `RetrievalBundle`
```python
class RetrievalBundle(BaseModel):
    target_paper: Paper
    similar_paper_cases: list[PaperCase] = Field(default_factory=list)
    supporting_or_competing_papers: list[Paper] = Field(default_factory=list)
    critique_cases: list[ExperienceCard] = Field(default_factory=list)
    policy_cards: list[ExperienceCard] = Field(default_factory=list)
    failure_cards: list[ExperienceCard] = Field(default_factory=list)
    venue_policy: VenuePolicy | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
```

##### 5.5.5 新增 `PlannedCriterion`
```python
class PlannedCriterion(BaseModel):
    criterion_id: str
    text: str
    theme: str
    kind: Literal["content", "policy"]
    priority: int = 1
    trigger_condition: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    agent_owner: str | None = None
    source_ids: list[str] = Field(default_factory=list)
```

#### 为什么
- 这是整个 memory-driven 重构的基础。

#### 验收标准
- 所有旧逻辑在类型层面仍能兼容。
- 新类型能支撑多通道 memory 和 planner。

---

### 5.6 新增 `src/pipeline/parse_paper.py`

#### 要改什么
新增 `PaperParser`，把论文解析成 `PaperSignature`。

#### 最小实现
输入：`Paper`
输出：
- `paper_signature`
- `parsed_claims`
- `experiment_summary`
- `baseline_summary`

#### 推荐实现思路
1. 优先使用 `title + abstract + truncated fulltext`
2. 调一次 LLM，输出结构化 JSON
3. 如果失败，回退到基于 abstract 的启发式抽取

#### 推荐接口
```python
class PaperParser:
    def __init__(self, llm: LLMClient, max_fulltext_chars: int = 12000) -> None: ...
    def parse(self, paper: Paper) -> PaperSignature: ...
```

#### 为什么
- 当前检索只用 `title + abstract`，召回不足。
- 需要 paper signature 做 case retrieval 和 criteria planning。

#### 验收标准
- 对一篇 paper 能稳定输出结构化 signature。
- signature 能被 Retriever 使用。

---

### 5.7 `src/pipeline/retrieve.py`

#### 要改什么
把单通道 Retriever 改为多通道 Retriever。

#### 当前问题
当前 `Retriever.retrieve()`：
- 用 `title + abstract` 做 query
- 返回 `related_papers / related_reviews / unrelated_papers`

这不够支撑 case memory。

#### 新设计
拆成多个子检索函数：

```python
class Retriever:
    def retrieve(...):
        ...

    def retrieve_similar_paper_cases(...): ...
    def retrieve_supporting_or_competing_papers(...): ...
    def retrieve_critique_cases(...): ...
    def retrieve_policy_cards(...): ...
    def retrieve_failure_cards(...): ...
```

#### 行为要求
1. **保留现有 paper/review index 的兼容读取**
2. 新增对 case memory / failure memory 的读取
3. `unrelated_papers` 默认可以取消或降级为调试用途
4. query 不再只用 `title + abstract`，而是：
   - `title + abstract`
   - `paper_signature`
   - 可选的 `task/method tags`

#### 输出要求
返回新的 `RetrievalBundle`。

#### 为什么
- 你要的不是单纯近邻检索，而是不同用途的 memory 检索。

#### 验收标准
- 能区分返回：paper cases、supporting papers、critique cards、policy cards、failure cards。
- trace 中要记录每个通道命中的 ID 和分数。

---

### 5.8 新增 `src/storage/case_store.py`

#### 要改什么
新增 PaperCase 的存储与读取。

#### 接口建议
```python
class CaseStore:
    def __init__(self, path: str | Path) -> None: ...
    def add_case(self, case: PaperCase) -> None: ...
    def list_cases(self, venue_id: str | None = None) -> list[PaperCase]: ...
    def get_case(self, case_id: str) -> PaperCase | None: ...
```

#### 为什么
- 不应把论文 case 强行塞进旧的 `MemoryStore`。
- case 与 critique / policy 的生命周期不同。

#### 验收标准
- 可以存取 PaperCase。
- 可以支持基于 case 的检索。

---

### 5.9 `src/storage/memory_store.py`

#### 要改什么
把当前 MemoryStore 从“policy 卡片版本库”升级成通用经验库。

#### 当前问题
- 只支持按词集合 overlap 做 `_find_similar()`
- 只支持 `venue_id + theme + content`
- 不支持 kind / scope / confidence / use_count 等

#### 修改要求
1. `add_or_update()` 支持 `kind`、`scope`、`trigger`、`source_ids`
2. `list_active()` 支持按 kind / venue / theme / scope 过滤
3. 相似度查找改为：
   - 优先 embedding 相似
   - 次级 lexical gate
4. 增加 `record_use()` / `record_success()` / `record_failure()`
5. 增加 `promote()` / `deactivate()`

#### 推荐接口
```python
class MemoryStore:
    def add_or_update(...): ...
    def search(...): ...
    def record_use(card_id: str): ...
    def record_success(card_id: str): ...
    def record_failure(card_id: str): ...
    def promote(card_id: str): ...
    def deactivate(card_id: str): ...
```

#### 为什么
- 当前 `_find_similar()` 过于脆弱。
- memory 需要支持长期维护。

#### 验收标准
- 可以同时管理 policy / critique / failure。
- 支持从 short-term promote 到 long-term。

---

### 5.10 新增 `src/pipeline/plan_criteria.py`

#### 要改什么
新增 `CriteriaPlanner`，替代“全部现挖”的模式。

#### 目标行为
输入：
- `PaperSignature`
- `RetrievalBundle`
- `VenueProfile`

输出：
- `planned_content_criteria`
- `planned_policy_criteria`
- `missing_criteria_requests`

#### 具体逻辑
1. 从 `policy_cards` 中选 top-k
2. 从 `similar_paper_cases` 中抽常见 weakness / useful criteria
3. 从 `failure_cards` 中补风险检查项
4. 如果覆盖不够，再调用 `CriteriaMiner` 生成补充 criteria
5. 给每条 criterion 增加 priority / agent_owner / required_evidence

#### 为什么
- 当前的 `CriteriaMiner` 太像“每次重新发明 criteria”。
- 需要先利用 memory，再做补充挖掘。

#### 验收标准
- planner 可以在不调用 LLM 的情况下，只用 memory 产出部分 criteria。
- planner 输出的 criteria 带优先级和归属主题。

---

### 5.11 `src/pipeline/mine_criteria.py`

#### 要改什么
将该模块从 ICLR-hardcoded 改成 venue-aware，并让它只负责“补充生成”，不再承担全部 criteria 生成。

#### 必须修改
1. 去掉硬编码 `ICLR_CRITERIA_THEMES`
2. 改为从 `VenueProfile` 读取主题
3. prompt 中不要写死 `ICLR`
4. 保留 `coverage gap` 机制，但把它视为 failure memory 的输入之一

#### 推荐改法
- 保留原有两个函数：
  - `mine_content_criteria()`
  - `mine_policy_criteria()`
- 但增加参数：
  - `venue_profile`
  - `seed_criteria`
  - `target_gaps`

#### 为什么
- 该模块目前把 ICLR review themes 和 review style 写死，不利于迁移。

#### 验收标准
- 更换 venue profile 后，prompt 内容自动切换。
- 没有任何 ICLR 硬编码文案残留在通用逻辑里。

---

### 5.12 `src/pipeline/rewrite_criteria.py`

#### 要改什么
保留，但升级为更强的 criteria concretization 模块。

#### 建议新增能力
1. 根据 `PaperSignature` 改写
2. 根据 `agent_owner` 输出更具体的审查问题
3. 根据 `required_evidence` 生成 evidence checklist

#### 输出目标
从：
- “实验是否充分？”

变成：
- “对于该 benchmark-heavy 论文，是否在与强基线一致的协议下报告了全部主结果，并包含对关键模块的消融？”

#### 为什么
- rewriter 是把抽象标准落地到当前论文的关键一步，应该保留并加强。

#### 验收标准
- 重写后的标准更具体且不泄露过拟合到单篇论文的细节。

---

### 5.13 `src/pipeline/review_pipeline.py`

#### 要改什么
这是本次最大的改动点。

#### 新流程要求
把当前流程：

```text
retrieve -> mine -> distill -> rewrite -> theme -> arbiter -> score -> calibrate -> update_memory
```

改为：

```text
parse -> retrieve -> plan -> mine_missing -> distill -> rewrite -> theme -> arbiter -> verifier -> score_check -> calibrate -> distill_experience -> memory_edit
```

#### 具体重构要求

##### 5.13.1 初始化阶段
新增这些成员：
- `PaperParser`
- `CaseStore`
- `ShortTermMemoryStore`
- `LongTermMemoryStore`
- `VenueProfileLoader`
- `CriteriaPlanner`
- `ExperienceDistiller`
- `MemoryEditor`
- `VerifierAgent`

##### 5.13.2 `_run_review()`
重写为：
1. 读取 config / venue profile
2. parse target paper
3. retrieve multi-channel bundle
4. 用 planner 生成 candidate criteria
5. 用 miner 补齐缺失 criteria
6. 用 distiller 去重/筛选
7. 用 rewriter 具体化
8. 跑 theme agents
9. arbiter 汇总
10. verifier 检查
11. score consistency warning
12. calibrate
13. distill experience
14. memory edit / update
15. 把完整 trace 塞进输出

##### 5.13.3 删除或降级逻辑
- `_score_with_similar_reviews()` 不再直接改分
- 把它改成 `_check_score_consistency()`

##### 5.13.4 trace 必须增强
至少记录：
- paper_signature
- retrieval by channel
- criteria chosen from memory vs mined
- verifier warnings
- experience distilled
- memory writes / rejects

#### 为什么
- 这是把系统从 criteria-driven 改为 memory-driven 的核心。

#### 验收标准
- 旧 CLI 仍能调用 review。
- 输出 trace 明显更丰富。
- 不开启 dynamic memory 时也能运行。

---

### 5.14 新增 `src/agents/verifier_agent.py`

#### 要改什么
新增最终 review 的 verifier / meta-reviewer。

#### 核心职责
检查：
1. critique 是否空泛
2. critique 是否和 evidence 对齐
3. score 是否和文字一致
4. 是否 venue mismatch
5. 是否重复 / 矛盾

#### 输出
```python
class VerificationReport(BaseModel):
    warnings: list[str]
    unsupported_claims: list[str]
    venue_mismatch: list[str]
    score_consistency_warning: str | None
    should_block_writeback: bool = False
```

#### 为什么
- 没有 verifier，memory writeback 很容易把错误经验记进去。

#### 验收标准
- verifier 可选开启。
- trace 中保留 verification report。

---

### 5.15 新增 `src/pipeline/distill_experience.py`

#### 要改什么
从单次审稿 trace 中抽取 reusable experience。

#### 输出对象
- `case_cards`
- `critique_cards`
- `failure_cards`
- `policy_updates`

#### 最小逻辑
1. 从最终 weaknesses / strengths 中抽 critique units
2. 从 verifier warnings 中抽 failure patterns
3. 从 target paper + final review 构造 `PaperCase`
4. 从高 utility 的 criteria 里提炼 reusable policy

#### 为什么
- 这是 test-time memory update 的核心模块。

#### 验收标准
- 单次 review 后，至少能蒸馏出 PaperCase 和 critique/failure 候选对象。

---

### 5.16 新增 `src/pipeline/memory_editor.py`

#### 要改什么
控制经验写回位置和准入门槛。

#### 逻辑要求
1. 新经验先进入 short-term memory
2. 满足条件才 promote 到 long-term
3. 若 verifier 标记高风险，则禁止写回 long-term
4. 能 merge 相似经验

#### 推荐准入规则
- `use_count >= promote_min_uses`
- `confidence >= promote_min_confidence`
- `should_block_writeback == False`

#### 为什么
- 避免 memory 污染。

#### 验收标准
- 能看到 short-term -> long-term 的 promote 行为。
- 低质量经验不会直接进 long-term。

---

### 5.17 `src/pipeline/update_memory.py`

#### 要改什么
这个文件不要再只负责写 policy_criteria。

#### 建议处理方式
二选一：

##### 方案 A（推荐）
保留文件名，但内部改成调用：
- `ExperienceDistiller`
- `MemoryEditor`

##### 方案 B
把此文件缩成兼容包装器：
```python
def update_memory(...):
    return memory_editor.apply(experience_distiller.distill(...))
```

#### 新逻辑必须支持
- case writeback
- critique writeback
- failure writeback
- policy writeback
- write gate
- short-term / long-term 分层

#### 为什么
- 这是当前最弱的一环。

#### 验收标准
- `update_memory()` 不再只依赖 `policy_criteria`。
- trace 中可看到写回的对象类型。

---

### 5.18 `src/pipeline/calibrate.py`

#### 要改什么
保留 calibration，但不要只做 accept vs non-accept 的极简版本。

#### 建议
1. 保持当前 isotonic regression 兼容
2. 新增三分类或序数映射（可选）
3. 输出 calibration artifact 时保留 venue / year / split 信息

#### 为什么
- 目前 calibrator 过于粗糙，容易丢失 borderline 信息。

#### 验收标准
- 校准结果可追踪。
- 不阻塞主流程。

---

### 5.19 把 SimilarReviewScoring 改为 `ScoreConsistencyChecker`

#### 要改什么
无论它现在在 `review_pipeline.py` 里还是未来拆文件，都要把语义改掉：

从：
- 根据相似历史评论调整当前评分

改成：
- 检查当前评分是否和相似历史 case 明显冲突
- 只输出 warning / explanation
- 默认不直接覆盖 raw_rating

#### 推荐输出
```python
class ScoreConsistencyReport(BaseModel):
    warning: str | None = None
    similar_case_ids: list[str] = Field(default_factory=list)
    suggested_delta: float | None = None
```

#### 为什么
- 这一步现在会把系统引向“拟合历史评分”。

#### 验收标准
- 默认不再修改 raw_rating。
- trace 中保留 consistency report。

---

### 5.20 新增 `configs/venues/` 与 `src/common/venue_profile.py`

#### 要改什么
新增 venue profile 抽象。

#### 示例结构
`configs/venues/iclr.yaml`
```yaml
venue_id: "ICLR"
dimensions:
  - novelty
  - soundness
  - empirical_adequacy
  - clarity
  - reproducibility
style_guidelines:
  - "Strengths before weaknesses"
  - "Weaknesses should be actionable"
rating_scale:
  min: 1
  max: 10
decision_labels: ["reject", "borderline", "accept"]
```

#### 为什么
- 不再把 venue 写死在 prompt 和主题常量里。

#### 验收标准
- 更换 venue profile 后，主题和 prompt 会变。

---

### 5.21 `src/main.py`

#### 要改什么
1. build_index 阶段支持 case 索引（可后置实现）
2. review_paper / evaluate 兼容新配置项
3. 允许构建 case memory / migrate memory 的辅助命令（可选）

#### 推荐新增命令
- `peerreviewer migrate_memory --config ...`
- `peerreviewer build_case_index --config ...`

#### 为什么
- 需要迁移旧 memory 并支持 case 建库。

#### 验收标准
- 原有命令仍然能用。
- 新命令能运行。

---

## 6. 测试要求

当前测试覆盖太窄，只包含 distill / leakage / memory store。需要新增以下测试。

### 6.1 保留已有测试
- `test_distill.py`
- `test_leakage_filter.py`
- `test_memory_store.py`

### 6.2 新增测试文件

#### `tests/test_paper_parser.py`
测试 PaperParser 能否输出稳定 schema。

#### `tests/test_case_store.py`
测试 PaperCase 的写入和读取。

#### `tests/test_multi_channel_retrieval.py`
测试 Retriever 是否按通道返回对象。

#### `tests/test_criteria_planner.py`
测试 planner 是否优先使用 memory。

#### `tests/test_score_consistency_checker.py`
测试 consistency checker 不会直接调分。

#### `tests/test_experience_distiller.py`
测试单次审稿是否能蒸馏出 case / critique / failure。

#### `tests/test_memory_editor.py`
测试 short-term promote 到 long-term 的逻辑。

#### `tests/test_review_pipeline_smoke.py`
做一个端到端 smoke test。

#### `tests/test_cross_venue_profile.py`
测试不同 venue profile 下主题与 prompt 是否切换。

### 6.3 验收标准
- `pytest` 全通过。
- 至少有一个 e2e smoke test。
- 新模块都有单测。

---

## 7. 迁移策略

为了避免一次性打爆当前系统，按以下步骤迁移：

### 第一步：安全与配置
- 清空明文 key
- 增加 example config
- 修 README

### 第二步：类型和存储
- 扩 types
- 新增 CaseStore
- 扩 MemoryStore

### 第三步：引入 parser 和 planner
- 新增 PaperParser
- 新增 CriteriaPlanner
- Retriever 改成多通道

### 第四步：经验蒸馏与写回
- 新增 ExperienceDistiller
- 新增 MemoryEditor
- update_memory 改为包装器

### 第五步：增强稳定性
- 新增 Verifier
- SimilarReviewScoring 降级为 checker
- 加测试

---

## 8. 需要保留的现有设计（不要删）

下面这些模块和思想是好的，应当保留并增强：

1. **CriteriaDistiller**
   - 去重和按主题均衡筛选仍然有价值。

2. **CriteriaRewriter**
   - 将通用 criteria 具体化是关键能力，应增强而不是删除。

3. **ThemeAgents 并行执行**
   - 多主题并行仍然是合理设计。

4. **Arbiter 汇总**
   - 仍然需要一个统一仲裁模块。

5. **Calibration**
   - 可以保留，但不应成为主贡献点。

---

## 9. 最终验收清单（Claude Code 完成后必须满足）

### 9.1 架构层面
- [ ] 系统不再只依赖 policy memory
- [ ] 存在 PaperCaseMemory / CritiqueMemory / FailureMemory / PolicyMemory
- [ ] ReviewPipeline 变为 memory-driven

### 9.2 工程层面
- [ ] 仓库中没有明文 key / 内网地址
- [ ] README 与默认配置一致
- [ ] CLI 兼容旧命令

### 9.3 算法层面
- [ ] 检索改为多通道
- [ ] planner 优先使用 memory
- [ ] update_memory 支持 case / critique / failure / policy
- [ ] SimilarReviewScoring 不再直接调分
- [ ] verifier 可阻止低质量经验写回 long-term

### 9.4 测试层面
- [ ] 所有旧测试通过
- [ ] 新增测试通过
- [ ] 至少有一个 review pipeline smoke test

---

## 10. 建议给 Claude Code 的执行方式

为了减少返工，请按下面顺序分 PR / 分 commit 做：

1. `config + security cleanup`
2. `types + stores refactor`
3. `paper parser + multi-channel retrieval`
4. `criteria planner + venue profile`
5. `experience distillation + memory editor`
6. `review pipeline integration`
7. `verifier + score consistency checker`
8. `tests + docs`

每一步都要求：
- 先改类型和接口
- 再改调用方
- 最后补测试

---

## 11. 额外要求

1. **所有新增模块都要有 docstring 和类型标注**
2. **不要在核心逻辑里保留 venue 名称硬编码**
3. **不要引入过多全局状态**
4. **trace 必须比现在更丰富**
5. **能用 dataclass / pydantic 的地方保持统一风格**
6. **优先写兼容包装器，而不是强行改掉所有旧接口**

---

## 12. 一句话总结给 Claude Code

> 当前项目已经有一个可运行的 criteria-driven reviewer，但它的 memory 只是附属功能。请把它重构成一个 memory-driven reviewer：加入论文 case memory、critique/failure memory、多通道检索、criteria planner、experience distiller 和 memory editor；保留现有多 agent 与 arbiter 主骨架，但把 update_memory 做强，把 score similarity 降级为 consistency checker，并去掉 ICLR 和敏感配置的硬编码。


