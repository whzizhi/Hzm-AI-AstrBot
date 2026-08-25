# -*- coding: utf-8 -*-
"""记忆系统（AstrBot 插件版）：短期记忆 + 长期记忆。

短期记忆：最近 5 轮对话原文（short_term.json）
长期记忆：用户画像 + 承诺 + 事实（long_term.json）

原版 memory_manager.py 的功能集成到此模块。
"""
import json
import threading
from pathlib import Path

from .constants import (
    PLUGIN_ROOT, MEMORY_FILE, LONG_TERM_MEMORY_FILE,
    SHORT_MEMORY_LINES,
)

# 确保 user_memory 目录存在
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

# 文件锁
_memory_lock = threading.Lock()

# ==================== 长期记忆 prompt ====================
MEMORY_EXTRACT_PROMPT = """你是记忆提取器。从用户和灰泽满的对话中提取值得记住的信息。

当前已知信息：
{current_summary}

本轮对话：
用户：{user_msg}
灰泽满：{reply}

提取以下信息（没有的返回 null）：
- new_impression：对用户的新印象标签（如"爱撒娇""认真粉丝""新来的"）
- new_user_fact：关于用户的新事实（如"喜欢听灰泽满唱歌""在墨尔本""生日是X月X日"）
- new_promise：灰泽满答应了什么（如"答应下次唱歌给TA听""说明天播"）
- new_self_fact：null（灰泽满的自我信息来自真人素材，不从聊天提取）

输出 JSON：
{{
  "new_impression": "印象标签" 或 null,
  "new_user_fact": "用户事实" 或 null,
  "new_promise": "承诺内容" 或 null,
  "new_self_fact": null
}}

要求：
- 只提取用户明确表达的信息，不要推断
- 不提取寒暄/问候/表情
- 如果没有值得记住的信息，所有字段返回 null"""


def _format_profile_summary(memory_card: dict) -> str:
    """把记忆卡格式化为可读摘要（供 LLM 理解当前已知信息）。"""
    if not memory_card:
        return "（暂无记录）"
    parts = []
    impressions = memory_card.get("impressions", [])
    if impressions:
        parts.append("印象：" + "、".join(impressions[-5:]))
    facts = memory_card.get("user_facts", [])
    if facts:
        parts.append("事实：" + "；".join(facts[-5:]))
    promises = memory_card.get("promises", [])
    if promises:
        parts.append("承诺：" + "；".join(promises[-3:]))
    return "\n".join(parts) if parts else "（暂无记录）"


# ==================== 短期记忆 ====================

def load_short_memory() -> dict:
    """读取 short_term.json；文件缺失/空/格式错误时返回空字典。"""
    if not MEMORY_FILE.exists():
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {}


def get_user_history(user_id: str) -> list:
    """获取某用户的短期对话历史（最近 5 轮）。"""
    memory = load_short_memory()
    history = memory.get(user_id, [])
    return list(history) if isinstance(history, list) else []


def append_user_history(user_id: str, user_msg: str, reply: str) -> None:
    """追加一轮对话到短期记忆，保留最近 N 条。全程持锁。"""
    with _memory_lock:
        memory = load_short_memory()
        history = memory.get(user_id, [])
        if isinstance(history, str):
            history = [history] if history else []
        elif not isinstance(history, list):
            history = []
        history.append(f"用户：{user_msg}")
        history.append(f"灰泽满：{reply}")
        if len(history) > SHORT_MEMORY_LINES:
            history = history[-SHORT_MEMORY_LINES:]
        memory[user_id] = history
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)


# ==================== 长期记忆 ====================

def _load_long_term() -> dict:
    """读取长期记忆文件。"""
    if not LONG_TERM_MEMORY_FILE.exists():
        return {}
    try:
        content = LONG_TERM_MEMORY_FILE.read_text(encoding="utf-8").strip()
        return json.loads(content) if content else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_long_term(data: dict) -> None:
    """保存长期记忆文件。"""
    LONG_TERM_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    LONG_TERM_MEMORY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_user_memory(user_id: str) -> dict:
    """获取某用户的长期记忆卡。"""
    data = _load_long_term()
    return data.get(user_id, {})


def update_user_memory(user_id: str, updates: dict) -> None:
    """更新用户的长期记忆（合并新信息）。"""
    with _memory_lock:
        data = _load_long_term()
        card = data.get(user_id, {})

        # 印象标签
        new_imp = updates.get("new_impression")
        if new_imp and isinstance(new_imp, str) and new_imp.strip():
            impressions = card.get("impressions", [])
            if new_imp not in impressions:
                impressions.append(new_imp)
            card["impressions"] = impressions[-10:]  # 最多保留 10 个

        # 用户事实
        new_fact = updates.get("new_user_fact")
        if new_fact and isinstance(new_fact, str) and new_fact.strip():
            facts = card.get("user_facts", [])
            if new_fact not in facts:
                facts.append(new_fact)
            card["user_facts"] = facts[-15:]  # 最多保留 15 条

        # 承诺
        new_promise = updates.get("new_promise")
        if new_promise and isinstance(new_promise, str) and new_promise.strip():
            promises = card.get("promises", [])
            if new_promise not in promises:
                promises.append(new_promise)
            card["promises"] = promises[-5:]  # 最多保留 5 条

        data[user_id] = card
        _save_long_term(data)


def build_memory_context(memory_card: dict) -> str:
    """把记忆卡格式化为注入上下文。"""
    if not memory_card:
        return ""
    parts = []
    impressions = memory_card.get("impressions", [])
    if impressions:
        parts.append("印象标签：" + "、".join(impressions[-5:]))
    facts = memory_card.get("user_facts", [])
    if facts:
        parts.append("已知事实：" + "；".join(facts[-5:]))
    promises = memory_card.get("promises", [])
    if promises:
        parts.append("灰泽满的承诺：" + "；".join(promises[-3:]))
    return "\n".join(parts) if parts else ""


__all__ = [
    "load_short_memory",
    "get_user_history",
    "append_user_history",
    "get_user_memory",
    "update_user_memory",
    "build_memory_context",
    "_format_profile_summary",
    "MEMORY_EXTRACT_PROMPT",
]
