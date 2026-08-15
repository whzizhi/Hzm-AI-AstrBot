# -*- coding: utf-8 -*-
"""本地 embedding + 余弦相似度 + 五路语义检索。"""
import json
import urllib.request
from pathlib import Path

from .config import (
    PERSONA_DIR,
    EMBED_URL,
    EMBED_MODEL,
    CORPUS_THRESHOLD,
    CORPUS_TOP_N,
    VOICE_THRESHOLD,
    VOICE_TOP_N,
    PHRASE_THRESHOLD,
    PHRASE_TOP_N,
    CORE_STORY_THRESHOLD,
    CORE_STORY_TOP_N,
    PREFERENCE_THRESHOLD,
    PREFERENCE_TOP_N,
)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def embed(texts, embed_url: str = "") -> list:
    """调用本地 embedding 服务。失败返回空列表（降级：跳过动态检索）。"""
    if not texts:
        return []
    url = embed_url or EMBED_URL
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [d["embedding"] for d in data.get("data", [])]
    except Exception:
        return []


def cosine_similarity(v1, v2) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if not n1 or not n2:
        return 0.0
    return dot / (n1 * n2)


def _topk(entries, query_vec, threshold, top_n, text_of):
    if not query_vec:
        return []
    scored = []
    for e in entries:
        v = e.get("vector")
        if not v:
            continue
        sim = cosine_similarity(query_vec, v)
        if sim >= threshold:
            scored.append((sim, e))
    scored.sort(key=lambda x: -x[0])
    return [text_of(e) for _, e in scored[:top_n]]


_corpus_cache = None
_voice_cache = None
_phrase_cache = None
_core_story_cache = None
_preference_cache = None


def load_corpus() -> list:
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = _read_json(PERSONA_DIR / "world" / "corpus_vectors.json", [])
        if not isinstance(_corpus_cache, list):
            _corpus_cache = []
    return _corpus_cache


def load_voice_samples() -> list:
    global _voice_cache
    if _voice_cache is None:
        data = _read_json(PERSONA_DIR / "speech" / "voice_sample_vectors.json", {})
        _voice_cache = data.get("samples", []) if isinstance(data, dict) else []
    return _voice_cache


def load_phrases() -> list:
    global _phrase_cache
    if _phrase_cache is None:
        data = _read_json(PERSONA_DIR / "speech" / "phrase_vectors.json", {})
        _phrase_cache = data.get("phrase_groups", []) if isinstance(data, dict) else []
    return _phrase_cache


def load_core_stories() -> list:
    global _core_story_cache
    if _core_story_cache is None:
        data = _read_json(PERSONA_DIR / "world" / "core_story_vectors.json", {})
        _core_story_cache = data.get("stories", []) if isinstance(data, dict) else []
    return _core_story_cache


def load_preferences() -> list:
    global _preference_cache
    if _preference_cache is None:
        data = _read_json(PERSONA_DIR / "world" / "preference_vectors.json", {})
        _preference_cache = data.get("entries", []) if isinstance(data, dict) else []
    return _preference_cache


def retrieve_corpus(query_vec) -> str:
    items = _topk(load_corpus(), query_vec, CORPUS_THRESHOLD, CORPUS_TOP_N,
                  lambda e: e.get("text", ""))
    if not items:
        return ""
    return "【她记得的】\n" + "\n".join(f"- {t}" for t in items if t)


def retrieve_voice_samples(query_vec) -> str:
    samples = load_voice_samples()
    if not query_vec:
        return ""
    scored = []
    for s in samples:
        v = s.get("vector")
        if not v:
            continue
        sim = cosine_similarity(query_vec, v)
        if sim >= VOICE_THRESHOLD:
            scored.append((sim, s))
    scored.sort(key=lambda x: -x[0])
    picked = scored[:VOICE_TOP_N]
    if not picked:
        return ""
    lines = [f"粉丝说：{s.get('user', '')} → 灰泽满：{s.get('reply', '')}"
             for _, s in picked if s.get("user") and s.get("reply")]
    if not lines:
        return ""
    return "【她的固定说法】（示例）\n" + "\n".join(lines)


def retrieve_phrases(query_vec) -> str:
    groups = load_phrases()
    if not query_vec:
        return ""
    scored = []
    for g in groups:
        v = g.get("vector")
        if not v:
            continue
        sim = cosine_similarity(query_vec, v)
        if sim >= PHRASE_THRESHOLD:
            scored.append((sim, g))
    scored.sort(key=lambda x: -x[0])
    picked = scored[:PHRASE_TOP_N]
    if not picked:
        return ""
    lines = []
    for _, g in picked:
        meaning = g.get("meaning", "")
        phrases = g.get("phrases", [])
        if meaning or phrases:
            lines.append(f"- {meaning}：{'、'.join(phrases) if phrases else ''}".strip("："))
    if not lines:
        return ""
    return "【她的措辞】\n" + "\n".join(lines)


def retrieve_core_stories(query_vec) -> str:
    items = _topk(load_core_stories(), query_vec, CORE_STORY_THRESHOLD, CORE_STORY_TOP_N,
                  lambda e: e.get("text", ""))
    if not items:
        return ""
    return "【她的核心记忆】\n" + "\n".join(f"- {t}" for t in items if t)


def retrieve_preferences(query_vec) -> str:
    items = _topk(load_preferences(), query_vec, PREFERENCE_THRESHOLD, PREFERENCE_TOP_N,
                  lambda e: e.get("text", ""))
    if not items:
        return ""
    return "【灰泽满的偏好】\n" + "\n".join(f"- {t}" for t in items if t)
