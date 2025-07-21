from langchain_core.prompts import ChatPromptTemplate

# 答案综合 Prompt
# 用于指导 LLM 综合所有收集到的研究结果，并生成一个全面、有来源引用的最终答案。
# 答案应清晰、结构化，并使用 Markdown 的 URL 语法精准引用来源。
SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个高级信息综合器。
你的任务是根据提供的研究结果，为用户的问题生成一个全面、准确且有来源引用的答案。
请确保答案逻辑清晰、易于理解。
在引用信息时，请务必使用 Markdown 的 URL 语法 `[引用文本](来源URL)` 来精准标注来源。
如果某个信息来自多个来源，请列出所有相关来源。

用户请求:
{input}

研究结果:
{research_results}

请开始生成最终答案。
"""),
    ]
)
