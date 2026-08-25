# -*- coding: utf-8 -*-
"""检索抽象层：六路检索 + RRF 融合 + 预算控制 + 关键词门（AstrBot 插件版）。

六路来源：
- corpus        背景记忆 → 走 RRF
- voice_sample  风格样本 → 走 RRF
- behavior      行为触发 → 走 RRF
- phrase        措辞指纹 → 走 RRF
- preference    偏好事实 → 命中才带，不走 RRF
- core_story    核心记忆 → 命中才带，不走 RRF
"""
import json
import os
from dataclasses import dataclass, field

from .constants import (
    PLUGIN_ROOT,
    VOICE_SAMPLE_VECTOR_FILE, PHRASE_VECTOR_FILE,
    PREFERENCE_VECTOR_FILE, CORE_STORY_VECTOR_FILE,
    BEHAVIOR_KEYWORDS_FILE,
    RAG_THRESHOLD, CORPUS_TOP_N,
    CORPUS_KEYWORD_FLOOR, CORPUS_STRONG_KEYWORD,
    VOICE_SAMPLE_THRESHOLD, VOICE_SAMPLE_TOP_N,
    VOICE_SAMPLE_KEEPALIVE, VOICE_SAMPLE_MIN_K,
    VOICE_SAMPLE_KEEPALIVE_MIN_SIM, VOICE_SAMPLE_PREFER_SHORT,
    PHRASE_THRESHOLD, PHRASE_TOP_N, PHRASE_PHASES_MAX,
    PREFERENCE_THRESHOLD, PREFERENCE_TOP_N,
    CORE_STORY_THRESHOLD, CORE_STORY_TOP_N,
    RRF_K, SOURCE_WEIGHTS, RETRIEVAL_TOPK,
    RETRIEVAL_BUDGET_CHARS, MAX_RETRIEVAL_ITEM_CHARS,
)
from .rag import cosine_similarity, load_vector_db
from .persona import load_trigger_vectors, _format_behavior_rule


@dataclass
class RetrievalItem:
    """一条检索候选。"""
    source: str
    item_id: str
    score: float
    rank: int = 0
    fusion_score: float = 0.0
    text: str = ""
    extra: dict = field(default_factory=dict)


# ==================== 统一打分核心 ====================

def _score_candidates(query_vector, entries, threshold, top_n,
                      source, id_of, text_of, extra_of=None) -> list:
    """通用打分：低于阈值丢弃，按分数降序取 top_n。"""
    if not query_vector:
        return []
    scored = []
    for entry in entries:
        sim = cosine_similarity(query_vector, entry["vector"])
        if sim < threshold:
            continue
        scored.append(RetrievalItem(
            source=source,
            item_id=id_of(entry),
            score=sim,
            text=text_of(entry),
            extra=extra_of(entry) if extra_of else {},
        ))
    scored.sort(key=lambda it: it.score, reverse=True)
    return scored[:top_n]


# ==================== corpus 关键词门 ====================

_GATE_STOP_CHARS = set("灰泽满绿冻直播你我他的了吗呢吧啊嗯哈是不是不什么怎么和去很在就没都")
_GATE_STRIP_TOKENS = ("灰泽满", "灰泽满Hazel", "hzm", "绿冻", "满神", "小满", "满姐")


def _gate_bigrams(text: str) -> set:
    """区分性 bigram：去名字/实体 + 去领域停用字。"""
    for tok in _GATE_STRIP_TOKENS:
        text = text.replace(tok, "")
    text = text.replace(" ", "")
    out = set()
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a in _GATE_STOP_CHARS or b in _GATE_STOP_CHARS:
            continue
        out.add(a + b)
    return out


def _corpus_keyword_overlap(query: str, statement: str) -> float:
    """query 的区分性 bigram 被 statement 覆盖的比例。"""
    qb = _gate_bigrams(query)
    if not qb:
        return 0.0
    tb = _gate_bigrams(statement)
    return len(qb & tb) / len(qb)


def _corpus_gate_pass(query: str, statement: str, sim: float) -> bool:
    """corpus 放行判定：强关键词直接过；否则语义达标 + 有区分性词重叠才过。"""
    ov = _corpus_keyword_overlap(query, statement)
    if ov >= CORPUS_STRONG_KEYWORD:
        return True
    return sim >= RAG_THRESHOLD and ov >= CORPUS_KEYWORD_FLOOR


# ==================== 三路 retriever ====================

def retrieve_corpus(user_query: str, query_vector,
                    threshold: float = RAG_THRESHOLD,
                    top_n: int = CORPUS_TOP_N) -> list:
    """直播记忆检索。带 V6 关键词门。"""
    db = load_vector_db()
    if not db or not user_query or not query_vector:
        return []
    scored = []
    for i, it in enumerate(db):
        sim = cosine_similarity(query_vector, it["vector"])
        if not _corpus_gate_pass(user_query, it["text"], sim):
            continue
        scored.append(RetrievalItem(source="corpus", item_id=str(i), score=sim, text=it["text"]))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]


# 声音样本向量缓存
_sample_vectors = None


def load_voice_sample_vectors() -> list:
    """读 persona/speech/voice_sample_vectors.json（缓存）。"""
    global _sample_vectors
    if _sample_vectors is not None:
        return _sample_vectors
    if os.environ.get("VOICE_SAMPLES", "1") == "0" or not VOICE_SAMPLE_VECTOR_FILE.exists():
        _sample_vectors = []
        return _sample_vectors
    try:
        with open(VOICE_SAMPLE_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", []) if isinstance(data, dict) else []
        _sample_vectors = [s for s in samples
                           if isinstance(s, dict) and s.get("vector") and s.get("reply")]
    except (json.JSONDecodeError, OSError):
        _sample_vectors = []
    return _sample_vectors


def _ensure_min_samples(items: list, samples: list, query_vector) -> list:
    """保底：阈值过滤后为空时，注入全体最高分 1 条。"""
    if items or not VOICE_SAMPLE_KEEPALIVE or not samples:
        return items
    entries = [{"vector": s["vector"], "id": s["id"],
                "text": "", "extra": {"user": s["user"], "reply": s["reply"], "type": s.get("type", "")}}
               for s in samples]
    best = max((cosine_similarity(query_vector, s["vector"]) for s in entries), default=-1.0)
    if best < VOICE_SAMPLE_KEEPALIVE_MIN_SIM:
        return []
    return _score_candidates(query_vector, entries, -1.0, VOICE_SAMPLE_MIN_K,
                             "voice_sample", lambda e: e["id"], lambda e: e["text"],
                             lambda e: e["extra"])


def retrieve_voice_samples(user_query: str, query_vector,
                           threshold: float = VOICE_SAMPLE_THRESHOLD,
                           top_n: int = VOICE_SAMPLE_TOP_N) -> list:
    """风格样本检索。"""
    samples = load_voice_sample_vectors()
    if not samples:
        return []
    entries = [{"vector": s["vector"], "id": s["id"], "text": "",
                "extra": {"user": s["user"], "reply": s["reply"], "type": s.get("type", "")}}
               for s in samples]
    items = _score_candidates(query_vector, entries, threshold, top_n,
                              "voice_sample", lambda e: e["id"], lambda e: e["text"],
                              lambda e: e["extra"])
    return _ensure_min_samples(items, samples, query_vector)


# 行为判别词
_behavior_keywords_cache = None


def load_behavior_keywords() -> dict:
    """加载行为判别词。"""
    global _behavior_keywords_cache
    if _behavior_keywords_cache is not None:
        return _behavior_keywords_cache
    if not BEHAVIOR_KEYWORDS_FILE.exists():
        _behavior_keywords_cache = {}
        return _behavior_keywords_cache
    try:
        data = json.loads(BEHAVIOR_KEYWORDS_FILE.read_text(encoding="utf-8"))
        _behavior_keywords_cache = {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError):
        _behavior_keywords_cache = {}
    return _behavior_keywords_cache


_BEHAVIOR_KEYWORDS = load_behavior_keywords()


def select_behavior_item(user_msg: str, behavior_intent: str, behaviors: list) -> "RetrievalItem | None":
    """L3：把行为意图转成行为注入项。优先 LLM 意图 > 关键词兜底。"""
    if not behaviors or not user_msg:
        return None
    # ① LLM 判定的意图
    if behavior_intent:
        for b in behaviors:
            if b.get("name") == behavior_intent:
                return RetrievalItem(source="behavior", item_id=behavior_intent, score=1.0,
                                     text=_format_behavior_rule(b))
    # ② 关键词兜底
    for b in behaviors:
        name = b.get("name", "")
        kws = _BEHAVIOR_KEYWORDS.get(name)
        if kws and any(k in user_msg for k in kws):
            return RetrievalItem(source="behavior", item_id=name, score=1.0,
                                 text=_format_behavior_rule(b))
    return None


# 措辞指纹向量缓存
_phrase_vectors = None


def load_phrase_vectors() -> list:
    """读 persona/speech/phrase_vectors.json（缓存）。"""
    global _phrase_vectors
    if _phrase_vectors is not None:
        return _phrase_vectors
    if os.environ.get("PHRASES", "1") == "0" or not PHRASE_VECTOR_FILE.exists():
        _phrase_vectors = []
        return _phrase_vectors
    try:
        with open(PHRASE_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        groups = data.get("phrase_groups", []) if isinstance(data, dict) else []
        _phrase_vectors = [g for g in groups if isinstance(g, dict) and g.get("vector") and g.get("phrases")]
    except (json.JSONDecodeError, OSError):
        _phrase_vectors = []
    return _phrase_vectors


def retrieve_phrases(user_query: str, query_vector,
                     threshold: float = PHRASE_THRESHOLD,
                     top_n: int = PHRASE_TOP_N) -> list:
    """措辞指纹检索。"""
    groups = load_phrase_vectors()
    if not groups:
        return []
    entries = [{"vector": g["vector"], "id": g["id"], "text": "",
                "extra": {"meaning": g.get("meaning", ""), "phrases": g.get("phrases", []),
                          "usage": g.get("usage", "")}}
               for g in groups]
    return _score_candidates(query_vector, entries, threshold, top_n,
                             "phrase", lambda e: e["id"], lambda e: e["text"],
                             lambda e: e["extra"])


# ==================== 偏好检索 ====================

_pref_vectors = None


def load_preference_vectors() -> list:
    """读 persona/world/preference_vectors.json（缓存）。"""
    global _pref_vectors
    if _pref_vectors is not None:
        return _pref_vectors
    if not PREFERENCE_VECTOR_FILE.exists():
        _pref_vectors = []
        return _pref_vectors
    try:
        with open(PREFERENCE_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else []
        _pref_vectors = [e for e in entries if e.get("vector") and e.get("text")]
    except (json.JSONDecodeError, OSError):
        _pref_vectors = []
    return _pref_vectors


def retrieve_preferences(user_query: str, query_vector,
                         threshold: float = PREFERENCE_THRESHOLD,
                         top_n: int = PREFERENCE_TOP_N) -> list:
    """偏好语义检索。"""
    entries = load_preference_vectors()
    if not entries or not query_vector:
        return []
    scored = []
    for e in entries:
        sim = cosine_similarity(query_vector, e["vector"])
        if sim < threshold:
            continue
        scored.append((sim, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{
        "id": e.get("id", ""),
        "category": e.get("category", ""),
        "text": e.get("text", ""),
        "score": round(sim, 3),
    } for sim, e in scored[:top_n]]


# ==================== 核心记忆检索 ====================

_core_story_vectors = None


def load_core_story_vectors() -> list:
    """读 persona/world/core_story_vectors.json（缓存）。"""
    global _core_story_vectors
    if _core_story_vectors is not None:
        return _core_story_vectors
    if not CORE_STORY_VECTOR_FILE.exists():
        _core_story_vectors = []
        return _core_story_vectors
    try:
        with open(CORE_STORY_VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        stories = data.get("stories", []) if isinstance(data, dict) else []
        _core_story_vectors = [s for s in stories if s.get("vector") and s.get("text")]
    except (json.JSONDecodeError, OSError):
        _core_story_vectors = []
    return _core_story_vectors


def retrieve_core_stories(user_query: str, query_vector,
                          threshold: float = CORE_STORY_THRESHOLD,
                          top_n: int = CORE_STORY_TOP_N) -> list:
    """核心记忆检索。"""
    stories = load_core_story_vectors()
    if not stories or not query_vector:
        return []
    scored = []
    for s in stories:
        sim = cosine_similarity(query_vector, s["vector"])
        if sim < threshold:
            continue
        scored.append((sim, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{
        "id": s.get("id", ""),
        "category": s.get("category", ""),
        "text": s.get("text", ""),
        "score": round(sim, 3),
    } for sim, s in scored[:top_n]]


# ==================== RRF 融合 ====================

def rrf_fuse(ranked_lists: list, k: int = RRF_K,
             weights: dict = SOURCE_WEIGHTS) -> list:
    """加权 Reciprocal Rank Fusion。"""
    fused = []
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            item.rank = rank
            w = weights.get(item.source, 1.0)
            item.fusion_score = w / (k + rank)
            fused.append(item)
    fused.sort(key=lambda it: (-it.fusion_score, -it.score))
    return fused


# ==================== 预算控制 ====================

def _item_cost(it: RetrievalItem) -> int:
    if it.source == "voice_sample":
        return len(it.extra.get("user", "")) + len(it.extra.get("reply", ""))
    if it.source == "phrase":
        return sum(len(p) for p in it.extra.get("phrases", [])[:PHRASE_PHASES_MAX])
    return len(it.text)


def truncate_by_budget(items: list, budget_chars: int = RETRIEVAL_BUDGET_CHARS,
                       max_item_chars: int = MAX_RETRIEVAL_ITEM_CHARS) -> list:
    """按融合序贪心保留，超预算丢弃低优先级条目。"""
    total, kept = 0, []
    for it in items:
        cost = _item_cost(it)
        if cost > max_item_chars:
            cost = max_item_chars
        if total + cost > budget_chars:
            continue
        kept.append(it)
        total += cost
    return kept


def fuse_and_truncate(corpus_items, sample_items, behavior_items, phrase_items=None) -> list:
    """完整融合流程：RRF → 条数截断 → 字符预算截断。"""
    if phrase_items is None:
        phrase_items = []

    weights = dict(SOURCE_WEIGHTS)
    if VOICE_SAMPLE_PREFER_SHORT:
        for it in sample_items:
            if it.extra.get("length", "short") == "short":
                weights["voice_sample"] = weights.get("voice_sample", 1.0) + 0.3
                break

    fused = rrf_fuse([corpus_items, sample_items, behavior_items, phrase_items], weights=weights)
    fused = fused[:RETRIEVAL_TOPK]
    return truncate_by_budget(fused)
