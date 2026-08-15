# -*- coding: utf-8 -*-
"""静态人格 + 名词库 + 行为指令。"""
import json
import re
from pathlib import Path

from .config import PERSONA_DIR


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


_static_prompt_cache = None


def load_static_prompt() -> str:
    """静态层：system_prompt.txt + 【性格基底】 + 【语言风格】。"""
    global _static_prompt_cache
    if _static_prompt_cache is not None:
        return _static_prompt_cache

    parts = []
    sysp = _read_text(PERSONA_DIR / "core" / "system_prompt.txt").strip()
    if sysp:
        parts.append(sysp)

    traits = _read_json(PERSONA_DIR / "core" / "traits.json", [])
    if traits:
        lines = [f"- {t.get('name', '')}: {t.get('description', '')}" for t in traits
                 if t.get("name") or t.get("description")]
        if lines:
            parts.append("【性格基底】\n" + "\n".join(lines))

    styles = _read_json(PERSONA_DIR / "core" / "styles.json", [])
    if styles:
        lines = [f"- {s.get('name', '')}: {s.get('description', '')}" for s in styles
                 if s.get("name") or s.get("description")]
        if lines:
            parts.append("【语言风格】\n" + "\n".join(lines))

    _static_prompt_cache = "\n\n".join(parts)
    return _static_prompt_cache


_terms_cache = None


def load_terms() -> list:
    global _terms_cache
    if _terms_cache is not None:
        return _terms_cache
    data = _read_json(PERSONA_DIR / "world" / "terms.json", {})
    _terms_cache = data.get("terms", []) if isinstance(data, dict) else []
    return _terms_cache


def _term_matches(term: dict, user_msg: str) -> bool:
    kw = (term.get("keyword") or "").lower()
    if kw and kw in user_msg.lower():
        return True
    for alias in term.get("aliases", []):
        if alias and alias.lower() in user_msg.lower():
            return True
    pattern = term.get("pattern") or ""
    if pattern:
        try:
            if re.search(pattern, user_msg):
                return True
        except re.error:
            pass
    return False


def format_terms(user_msg: str) -> str:
    """核心名词：always 全注入 + on-demand 命中才注入。"""
    terms = load_terms()
    if not terms:
        return ""
    picked = []
    for t in terms:
        if t.get("priority") == "always":
            picked.append(t)
        elif _term_matches(t, user_msg):
            picked.append(t)
    if not picked:
        return ""
    lines = [f"- {t.get('keyword', '')}：{t.get('meaning', '')}" for t in picked]
    return "【核心名词】\n" + "\n".join(lines)


_behaviors_cache = None
_behavior_keywords_cache = None


def load_behaviors() -> list:
    global _behaviors_cache
    if _behaviors_cache is not None:
        return _behaviors_cache
    _behaviors_cache = _read_json(PERSONA_DIR / "behavior" / "behaviors.json", [])
    if not isinstance(_behaviors_cache, list):
        _behaviors_cache = []
    return _behaviors_cache


def load_behavior_keywords() -> dict:
    global _behavior_keywords_cache
    if _behavior_keywords_cache is not None:
        return _behavior_keywords_cache
    data = _read_json(PERSONA_DIR / "behavior" / "behavior_keywords.json", {})
    _behavior_keywords_cache = {k: v for k, v in data.items() if not k.startswith("_")} \
        if isinstance(data, dict) else {}
    return _behavior_keywords_cache


def _format_behavior_rule(rule: dict) -> str:
    name = rule.get("name", "")
    trigger = rule.get("trigger", "")
    response = rule.get("response", "")
    samples = rule.get("samples", [])
    parts = []
    if name:
        parts.append(f"【{name}】")
    if trigger:
        parts.append(f"触发情境：{trigger}")
    if response:
        parts.append(f"回应模式：{response}")
    sample_lines = []
    for s in samples:
        u = s.get("user", "")
        r = s.get("reply", "")
        if u and r:
            sample_lines.append(f"  粉丝说：{u} → 灰泽满：{r}")
    if sample_lines:
        parts.append("她这么说过（照着学腔调，不自己发明）：\n" + "\n".join(sample_lines))
    return "\n".join(parts)


def match_behavior(user_msg: str) -> str:
    """行为判别词兜底：命中 behavior_keywords.json → 注入对应行为指令。"""
    behaviors = load_behaviors()
    keywords = load_behavior_keywords()
    if not behaviors or not user_msg:
        return ""
    for b in behaviors:
        name = b.get("name", "")
        kws = keywords.get(name, [])
        if kws and any(k in user_msg for k in kws):
            return "【行为指令】\n" + _format_behavior_rule(b)
    return ""
