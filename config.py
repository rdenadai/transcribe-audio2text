from pydantic_settings import BaseSettings, SettingsConfigDict


class HuggingFaceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="huggingface_")
    llm_model: str = "unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    whisper_model: str = "openai/whisper-large-v3-turbo"
    whisper_chunk_length_s: int = 30
    whisper_stride_length_s: int = 5


class Settings(BaseSettings):
    huggingface: HuggingFaceSettings = HuggingFaceSettings()
    system_prompt: str = "Be objective and pragmatic in your responses. Always use markdown notation on the responses"


settings = Settings()
