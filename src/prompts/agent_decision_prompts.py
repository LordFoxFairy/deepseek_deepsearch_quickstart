from langchain_core.prompts import ChatPromptTemplate

# 代理决策 Prompt
# 用于指导代理根据当前状态和用户请求，决定下一步应该执行哪个工具或动作。
AGENT_DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个高级研究代理，能够根据用户请求进行多步骤研究和信息综合。
你的目标是提供全面、准确且有来源引用的答案。
根据当前的研究状态和可用的工具，决定下一步的最佳行动。

可用的工具包括：
{tool_names}

当前研究状态:
{current_state}

用户请求:
{input}

思考：根据用户请求和当前研究状态，我需要执行什么操作？我应该使用哪个工具？
如果需要使用工具，请以 JSON 格式输出工具调用，例如：
{{
    "tool_name": "工具名称",
    "tool_input": {{
        "参数1": "值1",
        "参数2": "值2"
    }}
}}
如果研究完成并可以提供最终答案，或者无法继续进行，请直接输出最终答案或说明。
"""),
        ("placeholder", "{agent_scratchpad}"), # 用于 LangGraph 记录代理的中间思考和工具调用
    ]
)

# 工具失败决策 Prompt
# 用于在工具执行失败时，指导代理进行错误处理或重试决策。
TOOL_FAILURE_DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个高级研究代理。你刚刚尝试执行一个工具，但它失败了。
请分析失败原因，并决定是应该重试、尝试另一个工具，还是直接返回一个错误消息给用户。

用户请求:
{input}

当前研究状态:
{current_state}

上次工具调用:
{last_tool_call}

工具失败错误:
{tool_error}

思考：我应该如何处理这个工具失败？我能从错误中学到什么？
如果需要重试或尝试新工具，请以 JSON 格式输出工具调用。
如果无法继续，请直接输出一个清晰的错误消息。
"""),
        ("placeholder", "{agent_scratchpad}"),
    ]
)