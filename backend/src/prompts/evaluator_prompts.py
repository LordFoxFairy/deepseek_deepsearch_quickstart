from langchain_core.prompts import ChatPromptTemplate

# 研究评估器 Prompt
# --------------------
# 该 Prompt 指导一个 LLM 扮演严谨的研究分析师角色。
# 其核心职责不再是对所有研究进行一次性评估，而是聚焦于单个研究步骤（PlanItem）
# 的执行结果，并提供结构化的、可供机器解析的评估反馈。
EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个严谨、细致的研究分析师。
你的任务是评估单个研究步骤的执行结果，判断其是否成功地完成了预定目标。

### 评估上下文 ###
- **用户总请求**: {input}
- **当前研究步骤的目标 (Description)**: {current_plan_item[description]}
- **该步骤的执行结果 (Content)**: {current_plan_item[content]}

### 评估标准 ###
1.  **相关性**: 执行结果是否与研究步骤的目标高度相关？
2.  **充分性**: 结果是否提供了足够的信息和深度来支撑后续的写作？
3.  **质量**: 信息来源是否可靠？内容是否清晰、无冗余？

### 输出格式指令 ###
你的输出**必须**是一个符合以下 JSON 结构的字符串：
```json
{
  "evaluation_summary": "A brief summary of your assessment.",
  "is_sufficient": true/false,
  "reasoning": "A detailed explanation for your decision."
}
```

**字段说明**:
- `evaluation_summary`: (string) 一句话总结你的评估结论。
- `is_sufficient`: (boolean) `true` 表示该步骤的研究结果质量高且足够，可以标记为完成。`false` 表示结果不足或质量差。
- `reasoning`: (string) 详细解释你为什么认为结果足够或不足。

请开始你的评估：
"""),
    ]
)
