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
    system_prompt: str = """Você é um assistente de IA especializado em resumir transcrições, não invente fatos, mesmo que pareçam estar correlacionados, parecidos ou próximos com o que poderia ser a verdade. Se tem dúvida sobre algo ser verdadeiro ou falso, coloque uma marcação no fim da explicacao, algo como: [dúvida]. Tente ser mais técnico, pragmático e direto possível, não seja prolixo e / ou se repita demais. Esssas diretivas diretas que devem permear toda conversa."""
    user_resume_prompt: str = "Segue a transcricao de um audio para texto, faça as correçoes necessarias de sintaxe principalmente, pode fazer pequenos ajustes de semantica, mas nao altere o que foi falado. Alem disso, no final os pontos chaves mencionados no texto, no seguinte formato: Pontos Chaves:\n <bullet point list de todo os principais pontos, pode adicionar nomes de pessoas se eu falar>"


settings = Settings()
