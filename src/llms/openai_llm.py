from langchain_openai import ChatOpenAI

from backend.src.config.settings import settings


def get_chat_model(
        model: str = settings.OPENAI_CHAT_MODEL,
        api_key: str = settings.OPENAI_API_KEY,
        base_url: str = settings.OPENAI_BASE_URL,
        temperature: float = 0.7,
        **kwargs
):
    """
    初始化并返回一个 Langchain ChatOpenAI 模型实例。
    模型名称、API 密钥和基础 URL 可以从应用程序设置中获取，也可以通过函数参数覆盖。
    此函数可用于对接 OpenAI 官方 API 或兼容 OpenAI 接口的模型（如 DeepSeek）。
    """
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs
    )
