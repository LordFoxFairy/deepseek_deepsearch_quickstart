from langchain_core.prompts import ChatPromptTemplate

# 工具使用 Prompt
# 用于指导 LLM 如何根据工具的描述和用户请求，生成正确的工具调用参数。
# 代理决策 Prompt 已经决定了使用哪个工具，这里是填充工具参数的细节。
TOOL_USE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个工具使用专家。
你的任务是根据用户请求和当前研究状态，为指定的工具生成正确的输入参数。
只返回工具调用的 JSON 格式，不要包含任何额外的解释或文字。

可用的工具及其描述和输入 Schema 如下：
{tool_schemas}

用户请求:
{input}

当前研究状态:
{current_state}

思考：根据上述信息，我应该如何调用工具？需要哪些参数？
请以 JSON 格式输出工具调用，例如：
{{
    "tool_name": "工具名称",
    "tool_input": {{
        "参数1": "值1",
        "参数2": "值2"
    }}
}}
"""),
    ]
)