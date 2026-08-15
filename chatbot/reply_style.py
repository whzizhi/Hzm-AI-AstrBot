# -*- coding: utf-8 -*-
"""回复分段分句 + 打字节奏延迟（移植自参考项目 Hzm-AI-Bot 的 reply_style.py）。

纯函数，不依赖 AstrBot 运行时：main.py 生成回复后调用 split_reply 拆分，
再逐段 yield + split_delay 模拟真人打字节奏。
"""
import random
import re

from .config import (
    SPLIT_MIN_LEN,
    SPLIT_MAX_PARTS,
    SPLIT_MERGE_MIN_CHARS,
    SPLIT_DELAY_BASE_MS,
    SPLIT_DELAY_PER_CHAR_MS,
    SPLIT_DELAY_MIN_MS,
    SPLIT_DELAY_MAX_MS,
    SPLIT_DELAY_JITTER,
)
from .persona import load_terms


def _split_sentences(text: str) -> list:
    """按句末标点切分（。！？）；省略号仅在"前后都有内容"时才当边界。

    省略号常表"无语/语气"（如"啊……这……"）和犹豫前缀，不能乱切；
    只有省略号后面跟着新内容（≥5 字）**且前面也有足够内容**（≥4 字）才当停顿边界。
    """
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
    """每段至多 1 个逗号；超了从最后一个逗号拆开（逗号去掉），避免长串逗号连句。"""
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
    """把长回复按句子断开发送（打字感）。短回复/单句不拆，返回单元素列表。

    切分规则：
    - 句末标点（。！？）切分；省略号仅在后面还有新内容（≥5 字）时才切
    - 逗号限制：每段至多 1 个逗号，超了从最后一个逗号拆开
    - 聊天习惯不打句号：切分后去掉句尾"。"（保留？！…）
    - 短碎片并入下一段；纯括号段并入上一段；超 max_parts 并入最后一段
    """
    text = reply.strip().rstrip("。")
    if not text or len(text) < min_len:
        return [text]

    parts = _split_sentences(text)
    parts = _limit_commas(parts)
    parts = [p.strip().rstrip("。") for p in parts if p and p.strip()]

    # 纯括号段（（小声嘀咕））并入上一段，避免单独发一条"舞台说明"
    merged = []
    for p in parts:
        if merged and re.fullmatch(r'[（(][^）)]*[）)]', p):
            merged[-1] += p
        else:
            merged.append(p)
    parts = merged

    # 短段并入下一段（防"哦？""那倒是稀奇……"这种微消息单独成条）
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
        # 超限分段用换行连接（防 run-on）
        merged = "\n".join(p for p in parts[max_parts - 1:] if p).strip()
        parts = parts[:max_parts - 1] + [merged]
    return parts


def split_delay(part_text: str) -> float:
    """句间发送延迟（秒）：按段落长度模拟打字 + ±15% 随机抖动，避免机械等长。"""
    ms = SPLIT_DELAY_BASE_MS + SPLIT_DELAY_PER_CHAR_MS * len(part_text)
    ms = max(SPLIT_DELAY_MIN_MS, min(ms, SPLIT_DELAY_MAX_MS))
    ms *= random.uniform(1 - SPLIT_DELAY_JITTER, 1 + SPLIT_DELAY_JITTER)
    return round(ms / 1000.0, 3)


# ---- 输出清洗（移植自参考项目 Hzm-AI-Bot 的 reply_style.clean_reply） ----

# 角色以外的人（第三人称名字）：回复里出现这些名字时，"她/他"可能指别人，不动
_OTHER_PERSON_NAMES_CACHE = None


def _get_other_person_names() -> set:
    """取"角色以外的人"的名字集合：terms 的 person/family/relation 分类 + 补充。

    用于 clean_reply 的"她/他自指兜底"保护——回复里出现这些名字时，"她/他"
    大概率指这个人而不是角色自己，故不替换。动态取自 terms，新增人物不用记两处。
    """
    global _OTHER_PERSON_NAMES_CACHE
    if _OTHER_PERSON_NAMES_CACHE is not None:
        return _OTHER_PERSON_NAMES_CACHE
    names = {"女同学", "女仆女同学", "弥希", "真绯瑠", "瑞雅", "塔菲"}  # 灰泽满专属：不在 terms 的第三人称人物
    for t in load_terms():
        if t.get("category") in ("person", "family", "relation") and t.get("keyword"):
            names.add(t["keyword"])
            names.update(str(a) for a in t.get("aliases", []) if a)
    _OTHER_PERSON_NAMES_CACHE = names
    return names


def clean_reply(reply: str) -> str:
    """输出清洗：去括号前缀、整条至多 1 个括号、省略号归一并限频、第三人称自指兜底。

    人设规则是"每轮回复至多一个括号、省略号是例外"，但模型 few-shot 会学着多用，
    这里做确定性过滤。自指兜底：对话里角色用名字自称，不该出现"她/他"指自己——
    仅当回复里没出现其他人名时才替换（这时的"她/他"几乎必指角色自己）。

    相比参考项目：自称替换的"她/他"正则加了"们"负向保护（避免"她们/他们"→"灰泽满们"）。
    """
    text = reply.strip()
    # 去掉开头的连续括号前缀，如 （咽口水）你看...
    while text.startswith("（"):
        idx = text.find("）")
        if idx == -1:
            break
        text = text[idx + 1:].lstrip()
    if not text:
        return reply  # 剥光了就回原样，避免空回复
    # 若仍有 ≥2 个括号，只保留第一个，其余删除
    matches = list(re.finditer(r'（[^）]*）', text))
    if len(matches) >= 2:
        first = matches[0]
        before = text[:first.start()]
        kept = first.group(0)
        after = re.sub(r'（[^）]*）', '', text[first.end():])
        text = before + kept + after
    # 省略号纪律：归一连续省略号；剥掉开头省略号；开头"词+省略号"犹豫 → 逗号；整条至多 2 个
    text = re.sub(r'\.{3,}|…+', '……', text)
    text = re.sub(r'^……(?=\S)', '', text)
    text = re.sub(r'^([一-鿿]{1,5})……(?=.{4,})', r'\1，', text)
    if not text:
        text = '……'
    ell_pos = [m.start() for m in re.finditer('……', text)]
    if len(ell_pos) > 2:
        cut = ell_pos[2]
        text = text[:cut] + text[cut:].replace('……', '')
    # 结巴消融："那、那"→"那那"
    text = re.sub(r'(.)、\1', r'\1\1', text)
    # 自指"她/他"兜底：仅当回复里没出现其他第三人称名字时才替换
    if not any(n in text for n in _get_other_person_names()):
        text = re.sub(r"她(?!们)", "灰泽满", text)
        text = re.sub(r"(?<!其|无)他(?!们|人|家|国|乡|方|日)", "灰泽满", text)
    return text
