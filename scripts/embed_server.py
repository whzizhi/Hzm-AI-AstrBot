# -*- coding: utf-8 -*-
"""本地 embedding 服务：fastembed + bge-small-zh-v1.5，OpenAI 兼容 /v1/embeddings。

供 AstrBot 插件的 RAG 语义检索使用（可选组件，不启动则插件降级为纯静态人格）。
"""
import os
import time
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
# 模型缓存目录：默认在本脚本同级的 models/ 下（可移植）
CACHE_DIR = os.environ.get(
    "EMBED_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
)
PORT = int(os.environ.get("EMBED_PORT", "8000"))

app = FastAPI(title="local-embedding")

print(f"[embed] loading model {MODEL_NAME} -> {CACHE_DIR} ...", flush=True)
_model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
print("[embed] model loaded OK", flush=True)


class EmbedRequest(BaseModel):
    model: str = MODEL_NAME
    input: Union[str, List[str]]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        return {
            "object": "list",
            "data": [],
            "model": req.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    start = time.time()
    vecs = list(_model.embed(texts))
    data = [
        {"object": "embedding", "index": i, "embedding": [float(x) for x in v.tolist()]}
        for i, v in enumerate(vecs)
    ]
    ntok = sum(len(t) for t in texts)
    print(f"[embed] {len(texts)} text(s) -> {len(data)} vec(s) in {time.time()-start:.3f}s", flush=True)
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": ntok, "total_tokens": ntok},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
