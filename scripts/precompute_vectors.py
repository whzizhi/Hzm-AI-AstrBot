# -*- coding: utf-8 -*-
"""重新预计算灰泽满全部向量，写入 persona/ 目录。

读取 persona/ 下的原始数据（statement_final.json / voice_samples.json / phrases.json /
core_stories.json / preferences.json / behaviors.json），调用本地 embedding 服务生成向量，
写回 persona/ 下的 *_vectors.json（与插件内置向量同源、同 embedding 文本口径）。

用法（在插件根目录执行）：
    EMBED_URL=http://127.0.0.1:8000/v1/embeddings python scripts/precompute_vectors.py

依赖：requests；本地 embedding 服务（默认 fastembed + BAAI/bge-small-zh-v1.5）。
"""
import json
import os
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent
PERSONA = BASE / "persona"
EMB_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8000/v1/embeddings")
MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


def embed(texts):
    """批量 embedding（每批 64 条），返回向量列表。"""
    if not texts:
        return []
    out = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        r = requests.post(EMB_URL, json={"model": MODEL, "input": batch}, timeout=180)
        r.raise_for_status()
        out.extend(d["embedding"] for d in r.json()["data"])
    return out


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {path.relative_to(PERSONA)}")


def gen_corpus():
    print("[1/6] corpus 直播记忆 ...")
    data = json.loads((PERSONA / "world" / "statement_final.json").read_text(encoding="utf-8"))
    if isinstance(data, dict) and "statements" in data:
        items = [{"statement": s} for s in data["statements"] if s]
    else:
        items = data
    items = [it for it in items if (it.get("statement") or "").strip()]
    texts = [it["statement"] for it in items]
    vecs = embed(texts)
    result = [{"text": t, "vector": v} for t, v in zip(texts, vecs)]
    write_json(PERSONA / "world" / "corpus_vectors.json", result)
    print(f"  完成 {len(result)} 条")


def gen_voice():
    print("[2/6] voice_sample 声音样本 ...")
    data = json.loads((PERSONA / "speech" / "voice_samples.json").read_text(encoding="utf-8"))
    samples = data.get("samples", []) if isinstance(data, dict) else []
    items = [s for s in samples if (s.get("user") or "").strip()]
    texts = [s["user"] for s in items]
    vecs = embed(texts)
    result = [{
        "id": s.get("id", str(i)),
        "type": s.get("type", ""),
        "length": s.get("length", "short"),
        "user": s["user"],
        "reply": s.get("reply", ""),
        "vector": v,
    } for i, (s, v) in enumerate(zip(items, vecs))]
    write_json(PERSONA / "speech" / "voice_sample_vectors.json",
               {"model": MODEL, "dim": len(vecs[0]) if vecs else 0, "samples": result})
    print(f"  完成 {len(result)} 条")


def gen_phrase():
    print("[3/6] phrase 措辞指纹 ...")
    data = json.loads((PERSONA / "speech" / "phrases.json").read_text(encoding="utf-8"))
    groups = data.get("phrase_groups", []) if isinstance(data, dict) else []
    items = [g for g in groups if (g.get("trigger") or "").strip()]
    texts = [g["trigger"] for g in items]
    vecs = embed(texts)
    result = [{
        "id": g.get("id", str(i)),
        "meaning": g.get("meaning", ""),
        "trigger": g["trigger"],
        "phrases": g.get("phrases", []),
        "usage": g.get("usage", ""),
        "vector": v,
    } for i, (g, v) in enumerate(zip(items, vecs))]
    write_json(PERSONA / "speech" / "phrase_vectors.json",
               {"model": MODEL, "dim": len(vecs[0]) if vecs else 0, "phrase_groups": result})
    print(f"  完成 {len(result)} 组")


def gen_trigger():
    print("[4/6] trigger 行为触发 ...")
    behaviors = json.loads((PERSONA / "behavior" / "behaviors.json").read_text(encoding="utf-8"))
    triggers = list(dict.fromkeys(b.get("trigger", "") for b in behaviors if b.get("trigger")))
    vecs = embed(triggers)
    result = {t: v for t, v in zip(triggers, vecs)}
    write_json(PERSONA / "behavior" / "trigger_vectors.json", result)
    print(f"  完成 {len(result)} 个")


def gen_core_stories():
    print("[5/6] core_story 核心记忆 ...")
    data = json.loads((PERSONA / "world" / "core_stories.json").read_text(encoding="utf-8"))
    stories = data.get("stories", []) if isinstance(data, dict) else []
    items = [s for s in stories if (s.get("text") or "").strip()]
    texts = [s["text"] for s in items]
    vecs = embed(texts)
    result = [{
        "id": s.get("id", str(i)),
        "category": s.get("category", ""),
        "text": s["text"],
        "vector": v,
    } for i, (s, v) in enumerate(zip(items, vecs))]
    write_json(PERSONA / "world" / "core_story_vectors.json",
               {"model": MODEL, "dim": len(vecs[0]) if vecs else 0, "stories": result})
    print(f"  完成 {len(result)} 条")


def gen_preferences():
    print("[6/6] preference 偏好 ...")
    data = json.loads((PERSONA / "world" / "preferences.json").read_text(encoding="utf-8"))
    entries = data.get("entries", []) if isinstance(data, dict) else []
    items = [e for e in entries if (e.get("text") or "").strip()]
    texts = [e["text"] for e in items]
    vecs = embed(texts)
    result = [{
        "id": e.get("id", str(i)),
        "category": e.get("category", ""),
        "text": e["text"],
        "vector": v,
    } for i, (e, v) in enumerate(zip(items, vecs))]
    write_json(PERSONA / "world" / "preference_vectors.json",
               {"model": MODEL, "dim": len(vecs[0]) if vecs else 0, "entries": result})
    print(f"  完成 {len(result)} 条")


def main():
    gen_corpus()
    gen_voice()
    gen_phrase()
    gen_trigger()
    gen_core_stories()
    gen_preferences()
    print("\n全部完成。")


if __name__ == "__main__":
    main()
