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

# ==================== 分段分句发送（真人打字节奏） ====================
# 移植自参考项目 Hzm-AI-Bot 的 constants.py；main.py 逐段 yield 时使用。
SPLIT_REPLY_ENABLED = True     # 长回复拆成几句分开发送
SPLIT_MIN_LEN = 10             # 回复短于该长度不拆
SPLIT_MERGE_MIN_CHARS = 8      # 拆分后比这短的分段并入下一段
SPLIT_MAX_PARTS = 4            # 最多拆成几条消息，超出并入最后一条（防刷屏）
SPLIT_DELAY_BASE_MS = 1500     # 句间延迟基础（毫秒）
SPLIT_DELAY_PER_CHAR_MS = 100  # 每字追加（毫秒）
SPLIT_DELAY_MIN_MS = 1800
SPLIT_DELAY_MAX_MS = 5000
SPLIT_DELAY_JITTER = 0.15      # ±15% 随机抖动
