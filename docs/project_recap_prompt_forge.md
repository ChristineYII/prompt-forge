# Prompt Forge 项目复盘

## 1. 我做了什么？哪些是框架/作业给的？

### 我做的（独立设计实现）：
- **6 类 failure taxonomy**：wrong_function / missing_param / hallucinated_param / 
  hallucinated_call / type_mismatch / format_error
- **deterministic evaluator**：function name + 参数集合精确匹配
- **Critic refine 流程**：聚合失败类型分布 → 生成定向优化建议 → 触发新版本
- **客服 Agent demo 数据**：16 个测试用例（覆盖 clear / wrong_function trap / 
  missing_param trap / no-tool 四类）+ 3 个工具 schema
- **双层 temperature 接口**：evaluation / candidate generation / refine 三处
  分别可配置
- FastAPI + SQLAlchemy 后端 + Jinja2 前端

### 框架/工具：
- Gemini 2.5 Flash API
- FastAPI / SQLAlchemy / SQLite

## 2. 简历上的数字怎么来的？现在能复现吗?

- v1 87.5% → v2 93.75% → v3 100%
- 在客服 demo 场景（固定 16 个测试 case）下跑出
- **可复现条件**：DB 里有 customer service scenario 和 16 个测试 case 
  （通过 seed_phase0.py 灌入）
- **对照实验确认**：temperature ∈ {0, 1.0} 跑出相同结果，因为客服 case 都是
  高确定性输入

## 3. 最骄傲的设计决策 + trade-off

### Decision: 做对照实验否决了 temperature=0 的必要性
- 最初担心 temperature 默认 1.0 会让评测有随机性
- 做对照实验：固定 16 case + 同一个 v1 prompt，分别用 temperature=0 / 1.0 
  跑评测
- 结果：两个配置跑出完全相同的准确率（v1 87.5%, v2 93.75%, v3 100%）
- 归因：客服 case 都是高确定性输入（明确 order ID、明确动词），LLM 输出 
  分布已经 peaked，temperature 影响微乎其微

### Trade-off:
- 客服场景下不需要 temperature=0（无差异）
- 但通用场景下输入可能更模糊，仍然在代码里保留 temperature=0 接口

## 4. 最大局限 + 为什么没修

- **统计不显著**：n=16，无法说"显著改善"。理想是扩到 100+ case
- **测试集自动生成质量未量化**：项目支持任意 scenario LLM 动态生成 test cases，
  但目前没有评估生成 case 的质量（覆盖度、陷阱合理性、no-tool 占比）
- **只支持单工具调用**：不支持 parallel / sequential tool use
- **没修是因为**：Phase 1 优先做端到端工作流跑通，多工具和大规模 benchmark 
  是 Phase 2/3 的范畴

## 5. 重做我会改什么？

- **加入 LLM-as-judge 处理 ambiguous case**：deterministic eval 在某些 case 上
  过于严格（如 lookup + escalate 都合理时）
- **加入 prompt 版本对比 UI**：现在只能看 accuracy 数字
- **多工具支持**：parallel / sequential tool calls 评测
- **更多 demo scenario**：除客服外加代码 review、Q&A 等