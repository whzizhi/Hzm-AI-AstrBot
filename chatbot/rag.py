# -*- coding: utf-8 -*-
"""RAG 基础层：余弦相似度 + 向量库加载（AstrBot 插件版）。

原版用 zhipu client 做 embedding，AstrBot 插件版用本地 embedding 服务。
"""
import json
import urllib.request

from .constants import VECTOR_FILE, EMBED_URL, EMBED_MODEL


def cosine_similarity(v1, v2) -> float:
    """计算两个向量的余弦相似度。任一向量为零向量时返回 0.0。"""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if not n1 or not n2:
        return 0.0
    return dot / (n1 * n2)


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


def load_vector_db():
    """加载直播记忆向量库；文件缺失或格式错误时返回空列表。"""
    if not VECTOR_FILE.exists():
        return []
    try:
        with open(VECTOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []
