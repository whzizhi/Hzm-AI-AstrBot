# -*- coding: utf-8 -*-
"""常量与路径配置。"""
from pathlib import Path

# 插件根目录 = chatbot 包的上一级
BASE_DIR = Path(__file__).resolve().parent.parent
PERSONA_DIR = BASE_DIR / "persona"

# 本地 embedding 服务（OpenAI 兼容 /v1/embeddings）
EMBED_URL = "http://172.18.0.1:8000/v1/embeddings"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# 检索阈值（bge-small-zh-v1.5 实测分布校准：相关 0.73~0.83，无关 0.55~0.62）
CORPUS_THRESHOLD = 0.68
CORPUS_TOP_N = 3
VOICE_THRESHOLD = 0.68
VOICE_TOP_N = 2
PHRASE_THRESHOLD = 0.52
PHRASE_TOP_N = 1
CORE_STORY_THRESHOLD = 0.65
CORE_STORY_TOP_N = 2
PREFERENCE_THRESHOLD = 0.45
PREFERENCE_TOP_N = 2
