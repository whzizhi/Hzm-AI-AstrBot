# -*- coding: utf-8 -*-
"""回复风格后处理：拆句 / 分批延迟 / 输出清洗 / 复读检测。"""
import random
import re

from .constants import (
    SPLIT_MIN_LEN, SPLIT_MAX_PARTS, SPLIT_MERGE_MIN_CHARS,
    SPLIT_DELAY_BASE_MS, SPLIT_DELAY_PER_CHAR_MS,
    SPLIT_DELAY_MIN_MS, SPLIT_DELAY_MAX_MS, SPLIT_DELAY_JITTER,
    VOICE_SAMPLE_REPLY_TRIM_CHARS,
)
from .persona import load_terms


# 第三人称名字保护名单
_OTHER_PERSON_NAMES_CACHE = None


def _get_other_person_names() -> set:
    """取"可能是'她/他'先行词"的第三人称指代集合。"""
    global _OTHER_PERSON_NAMES_CACHE
    if _OTHER_PERSON_NAMES_CACHE is not None:
        return _OTHER_PERSON_NAMES_CACHE
    names = {"女同学", "女仆女同学", "弥希", "真绯瑠", "瑞雅", "塔菲"}
    names.update({"粉丝", "观众", "水友", "同学", "室友", "阿姨", "姐姐", "妹妹",
                  "女生", "女孩", "老师", "邻居", "朋友", "同事", "主播", "家人们", "绿冻"})
    for t in load_terms():
        if t.get("category") in ("person", "family", "relation", "world") and t.get("keyword"):
            names.add(t["keyword"])
            names.update(str(a) for a in t.get("aliases", []) if a)
    _OTHER_PERSON_NAMES_CACHE = names
    return names


def _trim_text(text: str, max_chars: int) -> str:
    """裁剪长文本，超长加省略号。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "……"


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（DP）。"""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def is_echo_reply(reply: str, recent_bot_replies: list, min_ratio: float = 0.6) -> bool:
    """判断新回复是否复读了最近自己说过的话。"""
    reply = (reply or "").strip()
    if not reply:
        return False
    for old in (recent_bot_replies or [])[-3:]:
        old = (old or "").strip()
        if not old:
            continue
        if reply == old:
            return True
        shorter = min(len(reply), len(old))
        if shorter < 8:
            continue
        if _lcs_len(reply, old) >= shorter * min_ratio:
            return True
    return False


def _split_sentences(text: str) -> list:
    """按句末标点切分。"""
    parts = []
    i = 0
    for m in re.finditer(r'[。！？]|…+', text):
        end = m.end()
        if m.group(0).startswith("…"):
            rest = text[end:].lstrip()
            before = text[i:m.start()].rstrip()
            if len(rest) < 5 or len(before) < 4:
                continue
        parts.append(text[i:end])
        i = end
    if i < len(text):
        parts.append(text[i:])
    return [p for p in parts if p.strip()]


def _limit_commas(parts: list) -> list:
    """每段至多 1 个逗号。"""
    result = []
    for p in parts:
        while True:
            commas = [idx for idx, ch in enumerate(p) if ch in "，,"]
            if len(commas) < 2:
                break
            idx = commas[-1]
            result.append(p[:idx].strip())
            p = p[idx + 1:].strip()
        result.append(p)
    return [x for x in result if x]


def split_reply(reply: str, min_len: int = SPLIT_MIN_LEN,
                max_parts: int = SPLIT_MAX_PARTS) -> list:
    """把长回复按句子断开发送（打字感）。"""
    text = reply.strip().rstrip("。")
    if not text or len(text) < min_len:
        return [text]

    parts = _split_sentences(text)
    parts = _limit_commas(parts)
    parts = [p.strip().rstrip("。") for p in parts if p and p.strip()]

    # 纯括号段并入上一段
    merged = []
    for p in parts:
        if merged and re.fullmatch(r'[（(][^）)]*[）)]', p):
            merged[-1] += p
        else:
            merged.append(p)
    parts = merged

    # 短段并入下一段
    merged_short = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and len(parts[i]) < SPLIT_MERGE_MIN_CHARS:
            parts[i + 1] = parts[i] + parts[i + 1]
        else:
            merged_short.append(parts[i])
        i += 1
    parts = merged_short
    if len(parts) > 1 and len(parts[-1]) < SPLIT_MERGE_MIN_CHARS:
        parts[-2] += parts[-1]
        parts.pop()

    if len(parts) <= 1:
        return [text]

    if len(parts) > max_parts:
        merged = "\n".join(p for p in parts[max_parts - 1:] if p).strip()
        parts = parts[:max_parts - 1] + [merged]
    return parts


def split_delay(part_text: str) -> float:
    """句间发送延迟（秒）。"""
    ms = SPLIT_DELAY_BASE_MS + SPLIT_DELAY_PER_CHAR_MS * len(part_text)
    ms = max(SPLIT_DELAY_MIN_MS, min(ms, SPLIT_DELAY_MAX_MS))
    ms *= random.uniform(1 - SPLIT_DELAY_JITTER, 1 + SPLIT_DELAY_JITTER)
    return round(ms / 1000.0, 3)


def clean_reply(reply: str) -> str:
    """输出清洗：去括号前缀、整条至多 1 个括号、省略号归一并限频、第三人称自指兜底。"""
    text = reply.strip()
    # 去掉开头的连续括号前缀
    while text.startswith("（"):
        idx = text.find("）")
        if idx == -1:
            break
        text = text[idx + 1:].lstrip()
    if not text:
        return reply
    # 若仍有 ≥2 个括号，只保留第一个
    matches = list(re.finditer(r'（[^）]*）', text))
    if len(matches) >= 2:
        first = matches[0]
        before = text[:first.start()]
        kept = first.group(0)
        after = re.sub(r'（[^）]*）', '', text[first.end():])
        text = before + kept + after
    # 省略号纪律
    text = re.sub(r'\.{3,}|…+', '……', text)
    text = re.sub(r'^……(?=\S)', '', text)
    text = re.sub(r'^([一-鿿]{1,5})……(?=.{4,})', r'\1，', text)
    if not text:
        text = '……'
    ell_pos = [m.start() for m in re.finditer('……', text)]
    if len(ell_pos) > 2:
        cut = ell_pos[2]
        text = text[:cut] + text[cut:].replace('……', '')
    # 结巴消融
    text = re.sub(r'(.)、\1', r'\1\1', text)
    # 自指"她/他"兜底
    if not any(n in text for n in _get_other_person_names()):
        text = re.sub(r"她", "灰泽满", text)
        text = re.sub(r"(?<!其|无)他(?!人|家|国|乡|方|日)", "灰泽满", text)
    return text


def is_emotion_only_query(query: str) -> bool:
    """判断 probe 补全的检索 query 是否为"纯情绪"描述。"""
    q = (query or "").strip()
    return q.startswith("用户发") and "表情" in q
