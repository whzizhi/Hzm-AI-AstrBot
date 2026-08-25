# -*- coding: utf-8 -*-
"""读秒窗口（AstrBot 插件版）：把同一用户几秒内的多条消息攒批，静默后统一回复。

架构（参考原版 chat_window.py 三模块）：
- 采集：main.py 的处理器把每条消息 (text, image_urls) 交给 enqueue()
- 会话：enqueue 重置去抖定时器；用户停手超过随机读秒窗口（5~10s）才触发一次回复
- 回复：_flush 在回复前统一"读图 + 归纳"（作为读整批的一部分），再交给 handle_chat，
        复用分批发送（A）；发送中途用户插话 → 取消未发送的分段，优先回新消息

generation 计数用于区分代际：插话取消旧任务后，旧任务不会误清掉新任务正在用的窗口。
参数在 constants.py 配置（READ_WINDOW_MIN/MAX_SECONDS）。
"""
import asyncio
import random

from astrbot.core.message.message_event_result import MessageChain

from .core import handle_chat, summarize_batch
from .reply_style import split_reply, split_delay, clean_reply
from .constants import READ_WINDOW_MIN_SECONDS, READ_WINDOW_MAX_SECONDS


class _UserWindow:
    """单用户的会话窗口：缓冲 + 去抖/发送任务 + 代际计数。"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.pending: list = []          # [(text, image_urls)]
        self.task: asyncio.Task = None   # 当前"窗口等待/发送"任务
        self.generation = 0              # 每次 enqueue 自增


_windows: dict[str, _UserWindow] = {}


def _combine_text(msgs) -> str:
    """合并本批文本。"""
    texts = []
    for t, _ in msgs:
        if t and str(t).strip():
            texts.append(str(t).strip())
    return "\n".join(texts).strip()


async def _describe_images(msgs, vision_cb) -> str:
    """把本批图片源转成视觉描述文本（去重后最多 3 张）。失败返回空串。"""
    descs = []
    seen = set()
    for _, urls in msgs:
        for u in (urls or [])[:3]:
            if u and u not in seen:
                seen.add(u)
                try:
                    desc = await vision_cb(u)
                    if desc:
                        descs.append(desc)
                except Exception:
                    pass
    return "；".join(descs) if descs else ""


class ChatBatcher:
    """AstrBot 插件版读秒攒批器。"""

    def __init__(self, handle_chat_fn, vision_cb, config: dict = None):
        self.config = config or {}
        self.handle_chat_fn = handle_chat_fn
        self.vision_cb = vision_cb

    def enqueue(self, user_id: str, text: str, image_urls: list,
                send_fn) -> None:
        """把一条消息送入攒批窗口。send_fn 为异步回调（发送分段文本）。"""
        w = _windows.setdefault(user_id, _UserWindow(user_id))
        w.pending.append((text, image_urls or []))
        w.generation += 1
        gen = w.generation
        if w.task and not w.task.done():
            w.task.cancel()
        w.task = asyncio.create_task(self._flush(user_id, gen, send_fn))

    async def _flush(self, user_id: str, gen: int, send_fn) -> None:
        """延迟后合并本批并回复。代际不符（有新消息插话）则放弃。"""
        delay = random.uniform(
            float(self.config.get("read_window_min", READ_WINDOW_MIN_SECONDS)),
            float(self.config.get("read_window_max", READ_WINDOW_MAX_SECONDS)),
        )
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        w = _windows.get(user_id)
        if not w or gen != w.generation:
            return  # 已有新消息插入，本批作废

        batch = w.pending[:]
        w.pending.clear()
        if not batch:
            return

        text = _combine_text(batch)
        imgs_desc = await _describe_images(batch, self.vision_cb)
        if imgs_desc:
            text = (text + "\n\n[用户发来的图片内容] " + imgs_desc).strip()

        # 连发多条 → 先归纳再理解（单条跳过归纳，零额外延迟）
        if len(batch) >= 2:
            try:
                summary = await summarize_batch(batch, self.config.get("provider"))
                if summary:
                    text = f"{text}\n\n[用户刚才连发了几条，归纳：{summary}]"
            except Exception:
                pass

        reply = await self.handle_chat_fn(user_id, text)
        reply = clean_reply(reply)

        parts = split_reply(reply) if self.config.get("split_reply_enabled", True) else [reply]
        parts = [clean_reply(p) for p in parts]
        try:
            for p in parts[:-1]:
                await send_fn(p)
                await asyncio.sleep(split_delay(p))
            if parts:
                await send_fn(parts[-1])
        except Exception:
            pass  # 发送中途用户插话/会话关闭：放弃剩余分段
        finally:
            if w is not None and gen == w.generation:
                _windows.pop(user_id, None)


def make_send_fn(event):
    """把 AstrBot 事件包装成异步发送回调。"""
    async def _send(text: str):
        chain = MessageChain()
        chain.message(text)
        await event.send(chain)
    return _send
