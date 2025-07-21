from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 从 .env 文件加载环境变量
load_dotenv()

class Settings(BaseSettings):
    """
    从环境变量加载应用程序设置。
    此类使用 Pydantic 的 BaseSettings 自动加载环境变量并提供类型验证。
    """

    # OpenAI 及兼容 OpenAI 的模型配置
    OPENAI_API_KEY: str = Field(..., description="用于身份验证的 OpenAI 或兼容 OpenAI 的 API 密钥")
    OPENAI_CHAT_MODEL: str = Field("deepseek-chat",
                                   description="默认的聊天模型名称（可以是 OpenAI 或兼容模型，如 DeepSeek）")
    OPENAI_BASE_URL: str = Field("https://api.deepseek.com/chat/completions",
                                 description="OpenAI 或兼容 OpenAI API 的基础 URL")

    # 通用应用程序设置
    APP_NAME: str = Field("DeepSearch Quickstart", description="应用程序名称")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # 忽略 schema 中未定义的额外环境变量
    )


# 创建一个设置实例，供整个应用程序使用
settings = Settings()

if __name__ == "__main__":
    # 此块仅用于测试目的。
    # 它将打印加载的设置以验证是否正确加载。
    print("已加载的应用程序设置:")
    print(f"  应用程序名称: {settings.APP_NAME}")
    print(f"  聊天模型: {settings.OPENAI_CHAT_MODEL}")
    print(f"  API 密钥 (前5位): {settings.OPENAI_API_KEY[:5]}*****")
    print(f"  API 基础 URL: {settings.OPENAI_BASE_URL}")