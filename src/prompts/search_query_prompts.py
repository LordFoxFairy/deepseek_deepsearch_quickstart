from langchain_core.prompts import ChatPromptTemplate

# 搜索查询生成 Prompt
# 用于指导 LLM 根据用户请求和当前研究状态生成一个或多个精确的搜索查询。
# 生成的查询应该简洁明了，包含关键词，并能帮助找到准确的信息。
SEARCH_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个专业的搜索查询生成器。
你的任务是根据用户的问题和当前的研究进展，生成最有效、最相关的网络搜索查询。

当前研究状态:
{current_state}

用户请求:
{input}

请生成一个或多个搜索查询，每个查询一行。
例如：
搜索查询1
搜索查询2
"""),
    ]
)
