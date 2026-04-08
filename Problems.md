# Problems

基于 trace `data/pipeline_traces/20260407_223519_QcMdPYBwTu` 的问题总结。

## 目标偏差

当前系统的目标应为：

- 用 `Case` 做相似论文检索和评分锚定
- 用 `Policy` 提供通用审稿标准
- 用 `Critique` 提供常见问题模式
- 用 `Failure` 提供高风险拒稿信号
- 用内容相关 criteria 驱动各个 `ThemeAgent` 对当前论文做实质评审

这条 trace 相比之前已有进步，但仍然存在几个结构性问题，说明系统还没有真正达到上面的分层目标。

## 1. Retrieval 仍然把大量未充分抽象的 memory 带进主流程

### 现象

- `related_papers` 和 `related_reviews` 仍然为 0
- `policy_cards` 数量高达 44，`critique_cases` 高达 12
- `policy_cards` 中仍然混有明显跨论文、跨领域的内容，例如：
  - Mamba / SSM
  - BEGAN / chrominance score

### 问题

- 通用 memory 没有被有效抽象化
- `policy` 和 `critique` 中仍然保留 paper-specific 内容
- 这些内容虽然不一定全部被 planner 选中，但已经进入 prompt 上下文，干扰后续步骤

### 修改方向

- 在 memory 检索前增加通用 memory 过滤
- 对 `policy` / `critique` 增加抽象化检查，避免把仍然强依赖原论文上下文的内容直接作为通用 memory 使用
- 对 `policy_cards` 增加更严格的 top-k 和去噪规则，不能只按 utility 排序

## 2. Planner 虽然改进了，但 `case_memory` 仍在错误迁移 paper-specific critique

### 现象

`04_criteria_planner/output.json` 已经从上一次的“全是 policy_memory”改成：

- `policy_mined`
- `content_mined`
- `case_memory`

这是进步。

但 `case_memory` 仍然包含明显错误迁移的内容，例如：

- `Human evaluation protocol lacks methodological transparency: reports n=127 raters...`
- `... BEGAN ...`

这些明显不是当前 GNN 论文的可迁移 criteria。

### 问题

- `Case -> transferable criteria` 的抽取逻辑过于宽松
- 当前系统把某篇具体论文的 critique 句子当成了可迁移模式
- `case_memory` 缺少“是否脱离原论文仍成立”的判定

### 修改方向

- 对 `PaperCase.transferable_criteria` 增加可迁移性判断
- 只有满足“跨论文仍成立”的内容才允许进入 planner
- 否则保留在 `Case` 中用于评分锚定，不允许转成 criteria

## 3. CriteriaRewriter 会把错误来源“合理化”

### 现象

`05_criteria_rewriter/output.json` 中，来源本来就有问题的 criteria，被改写成了看起来和当前论文高度相关的版本。

### 问题

- rewriter 当前不仅在“具体化” criteria，也在“补全”错误 criteria
- 一旦上游来源有问题，rewriter 会掩盖问题，让错误看起来更合理
- 这会让问题更难排查，因为 trace 表面上看起来像是“合理 criteria”

### 修改方向

- 在 rewriter 之前做来源合法性检查
- 对来自 `case_memory` 的 criteria 增加 stricter validation
- 对明显不属于当前论文的 criteria 直接丢弃，而不是改写

## 4. Theme 设计混杂了“论文评审主题”和“review 写作规范主题”

### 现象

`06_theme_agents/output.json` 中的 themes 同时包含：

- 实质评审主题：
  - `Quality`
  - `Experiments`
  - `Significance`
  - `Clarity`
  - `Originality`
  - `Reproducibility`
  - `Related Work`
- 写作/协议主题：
  - `Tone`
  - `Structure`
  - `Style`
  - `Evidence`
- 以及大小写不统一的重复主题：
  - `Significance`
  - `significance`

### 问题

- `Tone / Structure / Style / Evidence` 中至少有一部分更接近 review protocol / prompt constraints，而不是与 `Quality / Experiments / Originality` 完全同级的内容评审维度
- 它们不应该全部以与内容主题完全相同的方式直接产出 strengths/weaknesses
- 这会导致：
  - 风格约束被误当成论文缺陷来源
  - 主题预算被稀释
  - arbiter 混淆“内容弱点”和“表达规范”
- theme 名称没有规范化，出现重复主题

### 修改方向

- 将 theme 分成两层：
  - 内容评审 themes：`Quality / Experiments / Originality / Significance / Clarity / Reproducibility / Related Work`
  - 写作/协议相关层：`Tone / Structure / Style / Evidence`
- 第二层不应全部以与第一层相同的 reviewer 形态运行；更合理的做法是：
  - 其中与论文质量直接相关的部分，例如 `Clarity/Writing`，保留对分数的影响
  - 更偏 review protocol 的部分，例如 `Tone / Structure / Evidence`，更多作为 prompt constraints、verifier checks 或受控加权信号
- 在 theme 进入执行前统一做 normalize，避免 `Significance` / `significance` 这种重复

## 5. Theme 输出仍然存在“把应检查点写成已验证事实”的倾向

### 现象

`06_theme_agents/output.json` 中，多个 theme 会直接写出高度具体的事实，例如：

- 具体 speedup 数字
- 具体 baseline 名称
- 具体图表和 section 对应关系

其中一部分可能属实，但从整体行为上看，系统有明显 tendency：

- 把“应该检查的点”
- 写成“论文已经满足的事实”

### 问题

- `policy/style` 类 criteria 被部分当作了内容依据
- rewriter 也会把 criteria 具体化到过度确定的程度
- theme prompt 当前鼓励模型补全 paper facts，而不是区分：
  - 已明确看到的事实
  - 需要核验的点

### 修改方向

- 要求 theme agent 明确区分：
  - observed evidence
  - inferred concern
  - missing information
- 对未明确证实的信息使用不确定性语言
- 对 `policy`/`style` 类 criteria 禁止直接转化成论文 strengths

## 6. Arbiter 仍然通过 `policy_cards` 旁路引入未充分抽象的记忆

### 现象

`07_arbiter/output.json` 的 `trace.criteria_used` 里仍然包含大量 `policy_memory_*`

其中包括上一轮已确认有问题的内容，例如：

- `BEGAN`
- `chrominance score`

并且 `rating_rationale` 中还出现了：

- `undefined 'chrominance score' -> undefined coarse-node construction`

### 问题

- 即使 planner 已经改善，arbiter 仍然直接使用 retrieval 阶段的大量 `policy_cards`
- 这等于绕过 planner，再次把未充分抽象的 memory 注入最终决策
- 最终不仅 trace 变脏，连评分理由本身都被带偏

### 修改方向

- arbiter 只应消费 planner/rewriter 之后的合法 criteria
- retrieval 的原始 `policy_cards` 不应直接作为“已使用证据”进入 arbiter 主决策
- 如果保留 memory 参考，也必须经过过滤和显式分层

## 7. Similar cases 仍然主要用于评分，但没有足够的迁移边界

### 现象

- `similar_cases_used = 4`
- score consistency 也仍然基于这批 cases 做主要参考

### 问题

- `Case` 本来应该主要用于：
  - 相似性检索
  - 评分锚定
  - 辅助理解常见优缺点模式
- 但当前系统还在把 case 内容过度迁移成可执行 criteria
- 这使 case 的作用从“anchor”变成了“content source”

### 修改方向

- 明确限制 case 的两种用途：
  - 评分锚定
  - 高层模式参考
- 禁止直接把 paper-specific weakness 转成通用 criteria，除非经过抽象化

## 8. Verifier 仍然过松，没有覆盖真正关键的流程异常

### 现象

`08_verifier/output.json` 仍然是：

- `passed = true`
- `warnings = []`
- `requires_revision = false`

但当前 trace 明明仍然存在：

- 未充分抽象的 memory 残留
- theme 分层不清
- 重复 theme
- arbiter rationale 混入跨论文内容
- score consistency 明确提示决策与多数 similar cases 不一致

### 问题

- verifier 现在只检查表面一致性
- 没有检查流程级异常

### 修改方向

- verifier 增加流程级检查：
  - 是否存在重复 theme
  - 是否存在跨论文实体被错误升格为通用依据
  - 是否存在 policy/style theme 直接输出 substantive judgment
  - 是否存在 arbiter rationale 引入无关实体
  - score consistency 有 warning 时是否要求 revision

## 9. 当前 memory 体系缺少“抽象化/受控保留实体”这一层

### 现象

从整个 trace 看，真正的问题不是 memory 类型设计错了，而是：

- `Case -> Policy/Critique/Failure`

这条链在抽取时没有完成真正的抽象化。

### 问题

- 目前更像是在摘录历史 review 句子
- 而不是在沉淀跨论文、跨方向仍成立的经验模式

### 修改方向

memory 构建应该改成两步：

1. 从 review / case 中生成候选经验模式
2. 再判断候选是否：
   - 具有可迁移性
   - 对于通用 memory，是否仍然强依赖具体论文实体
   - 属于 `policy / critique / failure` 哪一类
   - 值得进入长期或短期 memory

## 9.1 关于“论文实体”的新边界

这次讨论后，边界需要进一步明确：

- 不是所有带论文实体的经验都应被禁止
- 但带实体的经验不能直接以原样进入通用 `Policy / Critique / Failure`

推荐分层如下。

### A. `Case`

允许保留具体论文实体，例如：

- 方法名
- 数据集名
- 模块名
- 具体实验缺陷

因为 `Case` 本来就是历史具体论文案例，它的职责包括：

- 相似检索
- 评分锚定
- 为当前论文生成内容相关 criteria 提供案例参考

### B. `Policy / Critique / Failure`

默认要求抽象化，不应直接依赖具体论文实体才能成立。

例如可以保留成：

- `Policy`: “核心方法必须有足够基线对比”
- `Critique`: “关键模块缺少清晰定义会削弱可复现性”
- `Failure`: “核心理论假设未成立会构成高风险拒稿点”

而不应直接保留成：

- `BEGAN 的 chrominance score 未定义`
- `Bokeh 的 DoF 设置不清楚`
- `Mamba 的 selective SSM 细节...`

### C. 当前论文相关 criteria 生成

这里可以参考带实体的 `Case`，但必须经过“受控迁移”：

- 不是直接复用历史句子
- 而是将历史 case 中的问题映射成当前论文的检查点

也就是说：

- `Case` 可以带实体
- `通用 memory` 应该抽象化
- `当前论文 criteria` 可以借用 case 经验，但要经过当前论文条件化改写

### 直接结论

当前系统真正的问题不是“出现实体”本身，而是：

- 带实体的内容被错误地升格成了通用 memory
- 然后又被 planner / arbiter 当成普适标准使用

这一步必须改。

## 10. 优先修改顺序

建议按下面顺序改：

1. 修 memory 抽象与准入
   - 先堵住 `policy` / `critique` / `case transfer` 中未充分抽象的来源
2. 修 arbiter 输入边界
   - 不允许 retrieval 原始 `policy_cards` 直接旁路进入最终决策
3. 修 theme 分层
   - 将 `Tone / Structure / Style / Evidence` 从“并列 reviewer”改成“受控约束/受控加权层”
4. 修 verifier
   - 增加流程级异常检查
5. 最后再调 retrieval 配比和 planner 权重

## 一句话总结

这条新 trace 说明系统已经从“完全被脏 memory 主导”进步到“当前论文内容 criteria 开始回到主流程”。

但系统仍然没有真正建立起以下边界：

- `Case` 用于锚定，不直接等于可迁移 criteria
- `Policy/Critique/Failure` 进入通用 memory 前必须经过抽象化
- `ThemeAgent` 的核心职责应仍是论文内容评审
- `Tone/Structure/Style/Evidence` 不应全部作为并列 reviewer 运行，而应作为受控约束/受控加权层存在
- `Arbiter` 不应直接消费 retrieval 的原始、未充分抽象的 memory

因此当前最核心的问题已经从“planner 全坏”转移为：

- memory 抽象/准入机制不够严格
- theme 层级设计不清
- arbiter 输入边界不严
