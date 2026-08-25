# -*- coding: utf-8 -*-
"""硬匹配路由：经典梗库 + LLM 语境确认 + 行为意图分类（L3）。

AstrBot 插件版：用 AstrBot Provider 做 LLM 调用。
"""
import json

from .constants import LEGENDARY_FILE

_legendary_cache = None


def load_legendary() -> dict:
    """加载经典梗库。"""
    global _legendary_cache
    if _legendary_cache is not None:
        return _legendary_cache
    if not LEGENDARY_FILE.exists():
        _legendary_cache = {"replies": {}, "confirms": {}}
        return _legendary_cache
    try:
        data = json.loads(LEGENDARY_FILE.read_text(encoding="utf-8"))
        _legendary_cache = {
            "replies": data.get("replies", {}) or {},
            "confirms": data.get("confirms", {}) or {},
        }
    except (json.JSONDecodeError, OSError):
        _legendary_cache = {"replies": {}, "confirms": {}}
    return _legendary_cache


LEGENDARY_REPLIES = load_legendary()["replies"]
LEGENDARY_CONFIRMS = load_legendary()["confirms"]


async def legendary_confirmed(user_msg: str, prompt_template: str,
                               provider, history: str = "") -> bool:
    """LLM 判断关键词命中的消息是否真是目标梗的语境。"""
    try:
        content = prompt_template
        if "{context}" in content:
            content = content.replace("{context}", history or "（无）").replace("{msg}", user_msg)
        else:
            content = content.replace("{msg}", user_msg)
        resp = await provider.text_chat(
            prompt=content,
            system_prompt="你是语境判断器。",
            max_tokens=10,
        )
        reply = (getattr(resp, "completion_text", None) or "").strip()
        return "是" in reply
    except Exception as e:
        print(f"⚠️ 梗确认失败（默认放行）: {e}")
        return True


# ==================== 行为意图分类（L3） ====================
BEHAVIOR_CLASSIFY_PROMPT = """你是{role_name}的行为意图分类器。判断用户刚发的这条消息是否明确落入某个"行为触发场景"。只有明确匹配才选，拿不准一律 null。

可选行为（name：触发情境）：
{behavior_defs}

判定要点：
- 只看用户这条消息本身的内容和语气，结合最近对话判断语境。
- "被夸"：消息确实在夸{role_name}。
- "被质疑/失约被催"：用户在质问、戳穿或催问{role_name}。
- "被越界"：玩笑/幻想触及个人边界。
- 普通闲聊、提问、寒暄、表情 → null。
- 拿不准 → null。

最近对话：
{history}

用户消息：{user_msg}

只输出 JSON：{{"behavior": "<可选行为name>" 或 null}}"""


async def classify_behavior(provider, user_msg: str, history_text: str,
                            behaviors: list) -> str:
    """LLM 判定用户消息落入哪个行为场景；拿不准或失败返回空串。"""
    if not behaviors or not user_msg:
        return ""
    defs = []
    for b in behaviors:
        if not b.get("name"):
            continue
        line = f"- {b['name']}：{b.get('trigger', '')}"
        for s in b.get("samples", [])[:2]:
            u = (s.get("user") or "").strip()
            if u:
                line += f"\n    例：{u}"
        defs.append(line)
    prompt = BEHAVIOR_CLASSIFY_PROMPT.format(
        behavior_defs="\n".join(defs),
        history=history_text or "（无）",
        user_msg=user_msg,
        role_name="灰泽满",
    )
    try:
        resp = await provider.text_chat(
            prompt=prompt,
            system_prompt="你是行为意图分类器，只输出 JSON。",
            max_tokens=20,
        )
        content = (getattr(resp, "completion_text", None) or "").strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(content)
        behavior = str(parsed.get("behavior") or "").strip()
        names = {b.get("name") for b in behaviors}
        return behavior if behavior in names else ""
    except Exception as e:
        print(f"⚠️ 行为意图分类失败（降级不触发）: {e}")
        return ""
