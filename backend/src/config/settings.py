from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 从 .env 文件加载环境变量
load_dotenv()


class Settings(BaseSettings):
    """
    应用程序的配置模型。
    此类使用 Pydantic 的 BaseSettings 自动从环境变量加载配置，并进行类型验证。
    """

    # --- 大模型服务相关配置 ---
    # 用于向量嵌入的模型
    DASH_SCOPE_API_KEY: str = Field(..., description="用于向量嵌入服务的API密钥")
    DASH_SCOPE_BASE_URL: str = Field("https://api.deepseek.com", description="向量嵌入服务的API基础URL")
    DASH_SCOPE_EMBEDDING_MODEL: str = Field("text-embedding-ada-002", description="执行向量嵌入的模型名称")

    # 用于生成对话和内容的聊天模型
    DEEPSEEK_API_KEY: str = Field(..., description="用于聊天模型的API密钥")
    DEEPSEEK_BASE_URL: str = Field("https://api.deepseek.com", description="聊天模型的API基础URL")
    DEEPSEEK_CHAT_MODEL: str = Field("deepseek-chat", description="默认的聊天模型名称")

    # --- 通用应用程序设置 ---
    APP_NAME: str = Field("DeepSearch Quickstart", description="应用程序的名称")

    model_config = SettingsConfigDict(
        env_file=".env",      # 指定环境变量文件的名称
        extra="ignore",       # 忽略在模型中未定义的额外环境变量
    )


# 创建一个全局的配置实例，供整个应用程序使用
settings = Settings()

if __name__ == "__main__":
    # 此代码块仅用于测试，当直接运行此文件时执行。
    # 它会打印出加载的配置，以验证环境变量是否被正确读取。
    print("已加载的应用程序配置:")
    print(f"  应用程序名称: {settings.APP_NAME}")
    # 注意：为了安全，不应在生产日志中打印完整的API密钥
    print(f"  聊天模型API密钥 (前5位): {settings.DEEPSEEK_API_KEY[:5]}*****")
    print(f"  聊天模型: {settings.DEEPSEEK_CHAT_MODEL}")
    print(f"  聊天模型API基础URL: {settings.DEEPSEEK_BASE_URL}")
    print(f"  嵌入模型: {settings.DASH_SCOPE_EMBEDDING_MODEL}")
