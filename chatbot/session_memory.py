# -*- coding: utf-8 -*-
"""会话级记忆（episodic memory）：记录"当前聊什么话题 + 本场关键事件"。

短期记忆只有最近 5 轮原文（碎片），长期记忆只有用户画像（事实），
都没有"这一场的调性/话题线"。会话级记忆补上这一层。

数据：user_memory/session.json，按 user_id 存 {"topic","events","last_active"}
"""
import json
import threading
from pathlib import Path
from datetime import datetime

from .constants import SESSION_MEMORY_FILE, MAX_EVENTS_PER_SESSION, SESSION_STALE_SECONDS

_lock = threading.Lock()

# ==================== 存储 ====================

def _load() -> dict:
    if not SESSION_MEMORY_FILE.exists():
        return {}
    try:
        content = SESSION_MEMORY_FILE.read_text(encoding="utf-8").strip()
        return json.loads(content) if content else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    SESSION_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_MEMORY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_session(user_id: str) -> dict:
    """读取某用户的会话状态。没有或已冷场时返回空会话。"""
    data = _load()
    sess = data.get(user_id)
    if not sess:
        return {"topic": "", "events": [], "last_active": ""}
    # 冷场判定：太久没聊（如隔天），旧话题不适用，返回空会话重新起
    last = sess.get("last_active", "")
    if last:
        try:
            dt = datetime.fromisoformat(last)
            if (datetime.now() - dt).total_seconds() > SESSION_STALE_SECONDS:
                return {"topic": "", "events": [], "last_active": ""}
        except ValueError:
            pass
    return sess


# ==================== 提示词 ====================

SESSION_PROBE_PROMPT = """你是会话话题追踪器。用户在聊天中刚发了一条新消息，下面给出【上一轮已知话题】和【最近对话】。

任务：判断这条新消息在当前语境下的完整含义，以及它是否带来了话题转变。

【上一轮已知话题】{prev_topic}（为空表示新会话）

【最近对话】
{history}

【用户刚发的消息】{user_msg}

输出 JSON：
{{
  "topic": "当前话题的一句话概括",
  "topic_changed": true 或 false,
  "new_event": "这条消息值得记住的关键事件，一句话；纯寒暄无事件则 null",
  "expanded_query": "如果这条消息很短（≤4字）或是指代性的（必须结合前面聊的话题才能理解），补全成完整句；其他情况则 null"
}}
要求：
- topic 要能体现对话调性（在干嘛、什么氛围）
- topic 只概括双方实际说的话，不要添加对话里没有的设定
- topic_changed 判定标准：只要用户这条消息是在问/聊一个新的具体主题，即使语气还延续之前的氛围，也视为转话题
- new_event 只记有意义的互动（示好/情绪/承诺/分享），寒暄问候不记
- **expanded_query 表情识别铁律**：如果消息是**纯表情/纯符号**（emoji 或[表情：xx]），必须先按**表情的标准含义**识别，不要从对话历史臆测：
  · 😅 = 无语/无奈/尴尬（不是傲娇调侃）
  · 😭 = 委屈/难过/哭
  · 🥲/😢 = 强颜欢笑/难受
  · 😂 = 笑/好笑
  · 🙏 = 感谢/拜托
  · ❤️/😍/🥰 = 爱意/喜欢
  · 😳 = 害羞/尴尬
  · 🤔 = 疑惑
  补全句应表达"用户发了【表情含义】"这个意思（如"用户发了个无语的表情"），用于检索记忆理解用户情绪，不要加引号。
- 非表情的短消息（如"咋这样""真的吗"）**或指代性消息**（如"能读给我听听吗""那个呢"——单看不知道指什么）才结合语境补全成完整句；普通长消息（自带话题）返回 null
只输出 JSON，不要多余内容。"""


# ==================== LLM 调用 ====================

async def _llm_call(provider, prompt: str, max_tokens: int = 250) -> str:
    """调用 AstrBot Provider。失败返回空串（调用方降级）。"""
    try:
        resp = await provider.text_chat(
            prompt=prompt,
            system_prompt="你是一个会话话题追踪器，只输出 JSON。",
            max_tokens=max_tokens,
        )
        content = (getattr(resp, "completion_text", None) or "").strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content
    except Exception as e:
        print(f"⚠️ 会话记忆 LLM 调用失败: {e}")
        return ""


# ==================== 更新会话（每轮对话后调用） ====================

async def probe_session(user_id: str, user_msg: str, history_text: str, provider) -> str:
    """对话前同步探测会话：判断话题延续/转换、累计事件、扩充短 query。

    返回：短消息的扩充句（长消息原样返回）。
    """
    msg = (user_msg or "").strip()
    if not msg:
        return msg
    prev = get_session(user_id)
    prev_topic = prev.get("topic", "")
    prompt = SESSION_PROBE_PROMPT.format(
        prev_topic=prev_topic or "（新会话）",
        history=history_text or "（无）",
        user_msg=msg,
    )
    content = await _llm_call(provider, prompt)
    expanded = msg
    if not content:
        return expanded
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return expanded

    topic = str(parsed.get("topic", "")).strip()
    if topic.lower() in ("null", "none"):
        topic = ""
    changed = bool(parsed.get("topic_changed"))
    new_event = parsed.get("new_event")
    if new_event and isinstance(new_event, str):
        new_event = new_event.strip()

    # 转话题：清空旧事件，起新话题
    events = [] if changed else list(prev.get("events", []))
    if new_event and new_event != "null":
        if new_event not in events:
            events.append(new_event)
    events = events[-MAX_EVENTS_PER_SESSION:]

    with _lock:
        data = _load()
        data[user_id] = {
            "topic": topic or prev_topic,
            "events": events,
            "last_active": datetime.now().isoformat(),
        }
        _save(data)

    # 检索 query 扩充
    ex = parsed.get("expanded_query")
    if ex and isinstance(ex, str):
        ex = ex.strip()
        if 0 < len(ex) <= 80 and ex != msg:
            expanded = ex
    return expanded


# ==================== 表情消息判定 ====================

_EMOJI_RANGES = [
    (0x1F600, 0x1F64F),
    (0x1F300, 0x1F5FF),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x2600, 0x27BF),
    (0xFE00, 0xFE0F),
    (0x200D, 0x200D),
]


def is_emoji_msg(msg: str) -> bool:
    """判断消息是否纯表情。"""
    text = (msg or "").strip()
    if not text:
        return False
    if text.startswith("[表情：") and text.endswith("]"):
        return True
    for ch in text:
        cp = ord(ch)
        if not any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES) and not ch.isspace():
            return False
    return True


# ==================== 注入上下文 ====================

def build_session_context(user_id: str) -> str:
    """生成【当前会话】注入文本。无有效会话返回空串。"""
    sess = get_session(user_id)
    topic = sess.get("topic", "")
    events = sess.get("events", [])
    if not topic and not events:
        return ""
    parts = []
    if topic:
        parts.append(f"当前话题：{topic}")
    if events:
        parts.append("本场发生：" + "；".join(events))
    return "\n".join(parts)


def is_emotion_only_query(query: str) -> bool:
    """判断 probe 补全的检索 query 是否为"纯情绪"描述。"""
    q = (query or "").strip()
    return q.startswith("用户发") and "表情" in q
