# -*- coding: utf-8 -*-
"""集中管理路径 / 阈值 / 参数的常量（AstrBot 插件版）。

原版路径从 src/plugins/chatbot/ 往上数四级到项目根；
AstrBot 插件版路径从 chatbot/ 往上一级到插件根目录。
"""
from pathlib import Path

# ==================== 路径 ====================
# constants.py -> chatbot -> 插件根
PLUGIN_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT_FILE = PLUGIN_ROOT / "persona" / "core" / "system_prompt.txt"
TRAITS_FILE = PLUGIN_ROOT / "persona" / "core" / "traits.json"
STYLES_FILE = PLUGIN_ROOT / "persona" / "core" / "styles.json"
BEHAVIORS_FILE = PLUGIN_ROOT / "persona" / "behavior" / "behaviors.json"
VOICE_SAMPLES_FILE = PLUGIN_ROOT / "persona" / "speech" / "voice_samples.json"
TERMS_FILE = PLUGIN_ROOT / "persona" / "world" / "terms.json"

MEMORY_DIR = PLUGIN_ROOT / "user_memory"
MEMORY_FILE = MEMORY_DIR / "short_term.json"
LONG_TERM_MEMORY_FILE = MEMORY_DIR / "long_term.json"
SESSION_MEMORY_FILE = MEMORY_DIR / "session.json"

VECTOR_FILE = PLUGIN_ROOT / "persona" / "world" / "corpus_vectors.json"
TRIGGER_VECTOR_FILE = PLUGIN_ROOT / "persona" / "behavior" / "trigger_vectors.json"
VOICE_SAMPLE_VECTOR_FILE = PLUGIN_ROOT / "persona" / "speech" / "voice_sample_vectors.json"
PHRASE_VECTOR_FILE = PLUGIN_ROOT / "persona" / "speech" / "phrase_vectors.json"
PREFERENCE_VECTOR_FILE = PLUGIN_ROOT / "persona" / "world" / "preference_vectors.json"
CORE_STORY_VECTOR_FILE = PLUGIN_ROOT / "persona" / "world" / "core_story_vectors.json"
LEGENDARY_FILE = PLUGIN_ROOT / "persona" / "world" / "legendary.json"
BEHAVIOR_KEYWORDS_FILE = PLUGIN_ROOT / "persona" / "behavior" / "behavior_keywords.json"

# ==================== Embedding ====================
EMBED_URL = "http://172.18.0.1:8000/v1/embeddings"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# ==================== 检索阈值 ====================
RAG_THRESHOLD = 0.48

# ==================== corpus 关键词门 ====================
CORPUS_KEYWORD_FLOOR = 0.12
CORPUS_STRONG_KEYWORD = 0.8

# ==================== 六路检索 ====================
CORPUS_TOP_N = 3
VOICE_SAMPLE_TOP_N = 3
VOICE_SAMPLE_THRESHOLD = 0.60
VOICE_SAMPLE_KEEPALIVE = True
VOICE_SAMPLE_KEEPALIVE_MIN_SIM = 0.60
VOICE_SAMPLE_MIN_K = 1
VOICE_SAMPLE_PREFER_SHORT = True

PHRASE_TOP_N = 2
PHRASE_THRESHOLD = 0.40
PHRASE_PHASES_MAX = 3

PREFERENCE_THRESHOLD = 0.55
PREFERENCE_TOP_N = 2

CORE_STORY_THRESHOLD = 0.42
CORE_STORY_TOP_N = 2

# ==================== RRF 融合 ====================
RRF_K = 60
SOURCE_WEIGHTS = {"behavior": 1.5, "corpus": 1.0, "voice_sample": 1.0, "phrase": 1.2}
RETRIEVAL_TOPK = 6

# ==================== 预算控制 ====================
RETRIEVAL_BUDGET_CHARS = 1200
MAX_RETRIEVAL_ITEM_CHARS = 300
VOICE_SAMPLE_REPLY_TRIM_CHARS = 60
MAX_CONTEXT_CHARS = 8000

# ==================== 短期记忆 ====================
SHORT_MEMORY_LINES = 10

# ==================== 分段分句发送 ====================
SPLIT_REPLY_ENABLED = True
SPLIT_MIN_LEN = 10
SPLIT_MERGE_MIN_CHARS = 8
SPLIT_MAX_PARTS = 4
SPLIT_DELAY_BASE_MS = 1500
SPLIT_DELAY_PER_CHAR_MS = 100
SPLIT_DELAY_MIN_MS = 1800
SPLIT_DELAY_MAX_MS = 5000
SPLIT_DELAY_JITTER = 0.15

# ==================== 读秒窗口 ====================
READ_WINDOW_MIN_SECONDS = 5.0
READ_WINDOW_MAX_SECONDS = 10.0

# ==================== 会话记忆 ====================
SHORT_QUERY_MAX_CHARS = 4
MAX_EVENTS_PER_SESSION = 6
SESSION_STALE_SECONDS = 12 * 3600

# ==================== 感知增强（时间/农历/节日/天气） ====================
# 和风天气免费版（可配可不配，不配则静默跳过天气段，不影响主流程）
WEATHER_BASE_URL = "https://devapi.qweather.com"   # 和风 API Host 根
WEATHER_CITY = ""                                  # 默认城市（空=墨尔本之外按用户城市，不配则无天气）
WEATHER_KEY = ""                                   # 和风 API Key（免费版）
WEATHER_CACHE_SECONDS = 3600                       # 天气进程内缓存 1 小时
WEATHER_GEO_CACHE_SECONDS = 86400                  # 城市名→LocationID 解析缓存 1 天

# ==================== B站联动 ====================
PUSH_INTERVAL = 180               # B站轮询间隔（秒）。别低于 120：带 SESSDATA 的会话调太频繁会被风控
BILI_STATE_FILE = PLUGIN_ROOT / "data" / "bili_state.json"  # 开播/动态去重状态持久化
AUTO_ACCEPT_FRIEND = True         # 好友申请自动通过
BILI_UID = ""                     # 灰泽满本人 B站 UID（开播/动态监听对象，面板配置覆盖）
NOTIFY_FRIENDS_WHITELIST = []     # 推送白名单 QQ 号列表；空 = 广播给全部好友（面板配置覆盖）

# ==================== 视觉（图片描述） ====================
VISION_MODEL = "glm-4.6v"         # 视觉理解模型（原版用智谱，插件版可用腾讯 youtu-vita 或厂商兼容模型）
VISION_MAX_TOKENS = 512           # 视觉描述输出上限
