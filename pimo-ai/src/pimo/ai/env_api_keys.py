"""pimo-ai 环境变量 API Key 解析。

各 LLM 厂商的 API Key 约定存储在特定环境变量中（如 ``OPENAI_API_KEY``、
``ANTHROPIC_API_KEY``）。此模块提供按 provider 名查找环境变量的功能，
用于在 stream() 调用前自动注入 api_key。
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# Provider → 环境变量名映射
# =============================================================================

# 每个 provider 对应的环境变量名列表，按优先级排列（排前面的优先）。
# 仅包含显式 API Key 变量，不含 OAuth token 或云平台凭据（如 AWS IAM、
# Google ADC），后者在 get_env_api_key() 内部自行处理。
_PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "github-copilot": ["COPILOT_GITHUB_TOKEN"],
    # ANTHROPIC_OAUTH_TOKEN 优先级高于 ANTHROPIC_API_KEY
    "anthropic": ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"],
    "ant-ling": ["ANT_LING_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "azure-openai-responses": ["AZURE_OPENAI_API_KEY"],
    "nvidia": ["NVIDIA_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "google": ["GEMINI_API_KEY"],
    "google-vertex": ["GOOGLE_CLOUD_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "vercel-ai-gateway": ["AI_GATEWAY_API_KEY"],
    "zai": ["ZAI_API_KEY"],
    "zai-coding-cn": ["ZAI_CODING_CN_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "minimax": ["MINIMAX_API_KEY"],
    "minimax-cn": ["MINIMAX_CN_API_KEY"],
    "moonshotai": ["MOONSHOT_API_KEY"],
    "moonshotai-cn": ["MOONSHOT_API_KEY"],
    "huggingface": ["HF_TOKEN"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "opencode": ["OPENCODE_API_KEY"],
    "opencode-go": ["OPENCODE_API_KEY"],
    "kimi-coding": ["KIMI_API_KEY"],
    "cloudflare-workers-ai": ["CLOUDFLARE_API_KEY"],
    "cloudflare-ai-gateway": ["CLOUDFLARE_API_KEY"],
    "xiaomi": ["XIAOMI_API_KEY"],
    "xiaomi-token-plan-cn": ["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
    "xiaomi-token-plan-ams": ["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
    "xiaomi-token-plan-sgp": ["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
}


def find_env_keys(provider: str) -> list[str] | None:
    """列出指定 provider 已设置值的环境变量名。

    用于 TUI / 配置诊断：不取具体 Key 值，只检查哪些变量存在。

    Args:
        provider: 厂商名。

    Returns:
        已设置的环境变量名列表（按优先级排列）。一个都没有时返回 None。
    """
    env_vars = _PROVIDER_ENV_VARS.get(provider)
    if not env_vars:
        return None
    found = [var for var in env_vars if os.environ.get(var)]
    return found if found else None


def get_env_api_key(provider: str) -> str | None:
    """获取指定 provider 的环境变量 API Key。

    按优先级读取该 provider 关联的环境变量，返回第一个有值的。
    对 google-vertex 和 amazon-bedrock 等非标准 Key 的 provider，
    返回 ``"<authenticated>"`` 表示凭据已就绪。

    Args:
        provider: 厂商名，如 ``"openai"``、``"anthropic"``、``"deepseek"``。

    Returns:
        API Key 字符串，未找到时返回 None。
    """
    env_vars = _PROVIDER_ENV_VARS.get(provider)
    if env_vars:
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                return value

    # Google Vertex AI: 支持显式 Key 或 Application Default Credentials
    if provider == "google-vertex":
        if _has_vertex_adc_credentials():
            return "<authenticated>"

    # Amazon Bedrock: 支持多种 AWS 凭据来源
    if provider == "amazon-bedrock":
        if _has_bedrock_credentials():
            return "<authenticated>"

    return None


# =============================================================================
# 内部辅助
# =============================================================================


def _has_vertex_adc_credentials() -> bool:
    """检查是否存在 Google Vertex AI Application Default Credentials。"""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not project or not location:
        return False
    # 显式凭据文件路径
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path and Path(gac_path).is_file():
        return True
    # 默认 ADC 路径
    default_adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return default_adc.is_file()


def _has_bedrock_credentials() -> bool:
    """检查是否存在 Amazon Bedrock 凭据（任一 AWS 凭据来源）。"""
    return any(
        os.environ.get(var)
        for var in (
            "AWS_PROFILE",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        )
    ) or (
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
