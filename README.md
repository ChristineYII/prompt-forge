# Prompt Forge

**Agent Skill / MCP Tool 发布前的质量评测与自动修复平台** — 类似 CI 里的单元测试,但针对 Agent 的 system prompt 和工具调用。

自动评测 prompt 质量、按 7 类失败模式归类、生成针对性修复建议,在发布前暴露 prompt 与 tool schema 的不匹配。

---

## 核心闭环

```
Test Generator → Evaluator → Critic → Refined Prompt
                    ↓
            Lever 1: 单 case 信号不 refine
            Lever 2: refine 引入新失败时回滚
```

---

## 7 类失败模式

每类是 `(detector, mitigation_hint, severity)` 三件套,注册在 `lib/failure_modes.py`。新增类型只需 register 一次,evaluate / critic 核心闭环代码不变。

| Failure Mode | Severity | 含义 |
|---|---|---|
| `hallucinated_call` | high | 不该调工具却调了 |
| `wrong_function` | high | 选错工具 |
| `format_error` | high | 该调没调,返回纯文本 |
| `value_error` | high | 类型对、字段对,但值的语义错 |
| `missing_param` | medium | 缺失必填参数 |
| `hallucinated_param` | medium | schema 外的额外参数 |
| `type_mismatch` | low | 参数类型不符 |

---

## Guardrails(stress test 驱动的设计)

早期版本任何失败都 refine。多次 stress test 观察到一种反噬模式:**v1 = 93.8%,refine 一轮后 v2 退化到 81.2%** — 单 case 信号驱动的 refine 容易过拟合,在无关 case 上引入新失败。

- **Lever 1**:总失败 < 2 不 refine
- **Lever 2**:v_{n+1} 引入新失败时回滚到 v_n(类似 A/B 实验的 guardrail metric)

---

## 实测

```bash
python demo_run.py            # production: 单 case 不 refine
python demo_run.py --demo     # demo: 阈值=1,演示 refine 链路
pytest tests/ -v              # 38 个回归测试 / 0.18 秒
```

真实输出见 [`examples/`](examples/)。

Demo 模式经常一次到 100% — 反过来证明 evaluator 的核心价值:**不只是给 accuracy 数字,是给 Critic 一个"值不值得 refine"的判断**。

---

## 架构

`main.py` (FastAPI) · `lib/failure_modes.py` (注册表) · `lib/evaluator.py` (确定性归类) · `lib/prompt_ops.py` (generator + critic) · `tests/` (38 个回归测试)

**Evaluator 用确定性逻辑,不用 LLM-as-Judge** — 工具调用输出是结构化的,能确定性判断对错。LLM judge 会带来 bias(position / verbosity / self-enhancement)和成本。有结构的地方用规则,无结构的地方才用 LLM。

---

## Roadmap

**v0.4 — General Prompt Evaluator**:三层结构 — deterministic checks + behavior-level LLM judge(`expected_behavior` + rubric)+ informational fields(presence only)。把 intent alignment 从"reason 文本等价"提升为 evaluator 一级概念。

**v0.5 — Real-Data Test Curation**:从生产日志 mine 边界 case,人工标注后接入 evaluate → refine 闭环,捕获 LLM 合成测试集捕获不到的真实长尾失败。
