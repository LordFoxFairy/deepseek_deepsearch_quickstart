from langchain_core.prompts import ChatPromptTemplate

# 研究规划器 Prompt
# --------------------
# 该 Prompt 指导 LLM 将用户的原始请求分解为一系列结构化的研究步骤。
# 其核心职责是生成一个包含依赖关系的、机器可读的研究计划（List[PlanItem]），
# 为后续的自动化、迭代式研究奠定基础。
OUTLINE_PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个专业的任务规划专家，负责将复杂的用户请求分解为一系列结构化的、可执行的研究步骤。

你的核心任务是制定一个详尽的研究计划。

### 输出格式指令 ###
你的输出**必须**是一个符合以下 JSON 结构的字符串：
```json
[
  {
    "item_id": "a_unique_identifier_for_this_step",
    "description": "A clear and specific research query or action for this step.",
    "dependencies": ["list_of_item_ids_this_step_depends_on"]
  }
]
```

**字段说明**:
- `item_id`: (string) 一个独特的、简洁的、描述性的ID，例如 `research_market_size`。
- `description`: (string) 一个具体的、可直接作为搜索查询的描述性文本。
- `dependencies`: (List[string]) 一个字符串列表，包含此步骤所依赖的其他步骤的 `item_id`。如果一个步骤是初始步骤，则其依赖列表应为空 `[]`。

**思考过程**:
1.  首先，将用户的总请求分解成几个独立的研究要点。
2.  为每个要点创建一个 `PlanItem` 对象。
3.  仔细思考这些要点之间的逻辑关系。例如，在研究“技术A的应用”之前，可能需要先完成对“技术A的定义”的研究。
4.  根据逻辑关系，正确地填充 `dependencies` 字段。
5.  确保最终输出的是一个格式完全正确的 JSON 字符串。

用户请求:
{query}

当前上下文 (供参考):
{current_state}
{failure_context}
{no_progress_context}

请开始制定你的结构化研究计划：
"""),
    ]
)


# 写作规划器 Prompt
# --------------------
# 该 Prompt 指导 LLM 基于已有的研究成果，构建一份结构化的报告写作大纲。
# 它不仅规划章节内容，更重要的是定义了章节间的逻辑依赖关系，确保了报告生成的连贯性。
WRITER_PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个专业的报告架构师。
你的任务是根据用户的原始请求和已经收集到的所有研究资料，为最终的报告制定一个详细的、结构化的写作大纲。

### 输出格式指令 ###
你的输出**必须**是一个符合以下 JSON 结构的字符串：
```json
[
  {
    "item_id": "a_unique_identifier_for_this_chapter",
    "description": "A concise title or summary for this chapter/section.",
    "dependencies": ["list_of_item_ids_this_chapter_depends_on"]
  }
]
```

**字段说明**:
- `item_id`: (string) 一个独特的、简洁的、描述性的ID，例如 `write_introduction`, `analyze_market_trends`。
- `description`: (string) 这一章节的标题或核心内容描述。
- `dependencies`: (List[string]) 一个字符串列表，包含撰写本章节前必须完成的其他章节的 `item_id`。

**思考过程**:
1.  仔细分析用户的原始请求和所有已收集的研究资料。
2.  构思一个逻辑清晰的报告结构（例如：引言 -> 背景分析 -> 核心发现 -> 挑战与机遇 -> 结论）。
3.  为报告的每一个章节创建一个 `PlanItem` 对象。
4.  确定章节之间的写作顺序，并正确设置 `dependencies`。例如，`write_conclusion` 必须依赖于所有分析性章节。
5.  确保最终输出的是一个格式完全正确的 JSON 字符串。

用户的原始请求:
{query}

已收集的研究资料 (供你参考和规划):
{search_results}

请开始制定你的结构化写作计划：
"""),
    ]
)
