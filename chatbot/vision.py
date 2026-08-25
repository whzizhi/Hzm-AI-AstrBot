# -*- coding: utf-8 -*-
"""视觉识别：图片 → 腾讯 youtu-vita（云端）→ 文字描述。

替代旧做法（把 image_urls 直接丢给主 Provider，纯文本模型如 deepseek 会 400）。
"""
import base64

import httpx

from astrbot.api import logger

VISION_BASE_URL = "https://tokenhub.tencentmaas.com/v1"
VISION_MODEL = "youtu-vita"
VISION_PROMPT = "请识别并转写这张图片中的文字内容（如评论、标题、正文、弹幕等），并简要说明图片主题。用中文，控制在 100 字以内。若文字模糊，尽量转写能辨认的部分，并描述图片的颜色、构图，不要只说'看不清'。"
VISION_MAX_TOKENS = 256

# QQ 图片 CDN 需要带 Referer / UA 才能下载，否则 403
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://q.qq.com/",
}


async def _load_image_bytes(source: str):
    """把图片 URL 或本地路径读成字节；失败返回 None。"""
    try:
        if source.startswith(("http://", "https://")):
            async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True, timeout=30) as client:
                resp = await client.get(source)
                resp.raise_for_status()
                return resp.content
        with open(source, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"[vision] 读取图片失败 {source!r}: {e}")
        return None


async def describe_image(api_key: str, source: str, model: str = VISION_MODEL) -> str:
    """下载图片 → youtu-vita → 文字描述。失败返回空字符串。"""
    if not api_key:
        logger.warning("[vision] 未配置 vision_api_key，跳过图片识别")
        return ""

    img_bytes = await _load_image_bytes(source)
    if not img_bytes:
        return ""

    data_uri = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("ascii")

    # TokenHub 是 OpenAI 兼容接口：图片用标准 image_url 内容块（与 AstrBot 主模型同一套格式，
    # 之前的 <image> + 顶层 image 字段是 VITA 原生格式，TokenHub 不认，导致模型没收到图而瞎编）。
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": VISION_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{VISION_BASE_URL.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (data["choices"][0]["message"]["content"] or "").strip()
            logger.info(f"[vision] 描述结果: {content!r}")
            return content
    except Exception as e:
        logger.warning(f"[vision] youtu-vita 调用失败: {e}")
        return ""
