# -*- coding: utf-8 -*-
"""
astrbot_plugin_hzm_hello · 插件主入口（v0.4.0 · 无前缀直聊 + 完整记忆系统）

更新日志（v0.4.0）：
1. 同步原版 Hzm-AI-Bot 最新代码：六路检索 + RRF 融合 + 关键词门
2. 新增长期记忆系统（用户画像 + 承诺 + 事实）
3. 新增会话级记忆（话题追踪 + 短 query 扩充）
4. 新增行为意图分类（LLM 判定 + 关键词兜底）
5. 新增经典梗硬匹配（双路由：关键词 + LLM 语境确认）
6. 新增复读机防护
7. 新增术语语境确认
"""

import asyncio
import json
import os
import sys

# AstrBot 按包加载插件时，同目录自定义模块不在 sys.path 上，需显式加入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astrbot.api.star import Star, Context, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import ComponentType
from astrbot.api import logger

from chatbot.core import assemble_system_prompt, handle_chat
from chatbot.reply_style import split_reply, split_delay, clean_reply
from chatbot.memory import get_user_history
from chatbot import vision
from chatbot.chat_window import ChatBatcher, make_send_fn
from chatbot.bili_bridge import BiliMonitor, run_bili_monitor


DEFAULT_EMBED_URL = "http://172.18.0.1:8000/v1/embeddings"
DEFAULT_MAX_HISTORY = 20


@register(
    "astrbot_plugin_hzm_hello",
    "MureasAm (scaffold)",
    "灰泽满插件：无前缀直聊 + 人格组装 + 六路检索 + 记忆系统",
    "0.4.0",
)
class HzmHelloPlugin(Star):
    """插件主类。继承 Star，AstrBot 自动扫描并实例化。"""

    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        # 读秒攒批器：连发消息先攒批再统一回复（handle_chat + 视觉描述回调）
        self.batcher = None
        # B站监听后台任务
        self._bili_task = None
        self._bili_stop = None
        logger.info("[hzm_hello] __init__ 完成")

    # ==================== 生命周期 ====================
    async def initialize(self):
        logger.info(
            f"[hzm_hello] initialize | "
            f"enable_chat={self.config.get('enable_chat', True)} "
            f"enable_echo={self.config.get('enable_echo', False)} "
            f"enable_rag={self.config.get('enable_rag', True)} "
            f"greeting={self.config.get('greeting', '灰泽满：')!r}"
        )
        self.batcher = ChatBatcher(
            handle_chat_fn=self._handle_batch_chat,
            vision_cb=self._describe_one_image,
            config=self.config,
        )

        # B站直播/动态监听（配置了 BILI_UID 才启动；BILI_SESSDATA 用于动态，未配则只监听开播）
        bili_uid = (self.config.get("bili_uid") or "").strip() or ""
        if bili_uid:
            self._bili_stop = asyncio.Event()
            monitor = BiliMonitor(
                uid=bili_uid,
                sessdata=self.config.get("bili_sessdata", ""),
                whitelist=self.config.get("notify_friends_whitelist", []),
                push_interval=int(self.config.get("push_interval", 0) or 0),
                get_platform_client=self._get_platform_client,
            )
            self._bili_task = asyncio.create_task(run_bili_monitor(monitor, self._bili_stop))
            logger.info(f"[hzm_hello] B站监听已启动 uid={bili_uid}")
        else:
            logger.info("[hzm_hello] BILI_UID 未配置，跳过 B站监听")

    async def terminate(self):
        if self._bili_task:
            try:
                if self._bili_stop:
                    self._bili_stop.set()
                self._bili_task.cancel()
                await asyncio.gather(self._bili_task, return_exceptions=True)
            except Exception as e:
                logger.warning(f"[hzm_hello] 停止B站监听失败: {e}")
        logger.info("[hzm_hello] terminate 完成")

    def _get_platform_client(self):
        """返回第一个支持 OneBot 好友列表/私聊的平台客户端（CQHttp bot），无则 None。"""
        try:
            for platform in self.context.platform_manager.get_insts():
                client = platform.get_client()
                if client is not None and hasattr(client, "get_friend_list") \
                        and hasattr(client, "send_private_msg"):
                    return client
        except Exception as e:
            logger.warning(f"[hzm_hello] 获取平台客户端失败: {e}")
        return None

    # ==================== 攒批回调 ====================
    async def _describe_one_image(self, url: str) -> str:
        """攒批里的图片 → 视觉描述（单张）。"""
        vision_api_key = self.config.get("vision_api_key", "")
        try:
            return await vision.describe_image(vision_api_key, url) or ""
        except Exception as e:
            logger.warning(f"[hzm_hello] 图片描述失败: {e}")
            return ""

    async def _handle_batch_chat(self, session_id: str, user_text: str) -> str:
        """攒批 flush 后的完整聊天入口。"""
        return await self._chat_with_provider(user_text, session_id)

    # ==================== 会话记忆（AstrBot 内置） ====================
    async def _get_conversation(self, umo: str):
        """获取（或新建）当前会话的 Conversation。"""
        conv_mgr = self.context.conversation_manager
        cid = await conv_mgr.get_curr_conversation_id(umo)
        if not cid:
            cid = await conv_mgr.new_conversation(umo)
        conversation = await conv_mgr.get_conversation(umo, cid)
        return conv_mgr, cid, conversation

    async def _load_history(self, umo: str) -> list:
        """读取最近 max_history 条会话历史。"""
        try:
            max_history = int(self.config.get("max_history", DEFAULT_MAX_HISTORY) or DEFAULT_MAX_HISTORY)
            if max_history <= 0:
                return []
            _, _, conversation = await self._get_conversation(umo)
            if conversation and conversation.history:
                history = json.loads(conversation.history)
                if isinstance(history, list):
                    return history[-max_history:]
        except Exception as e:
            logger.warning(f"[hzm_hello] 读取会话历史失败: {e}")
        return []

    async def _save_history(self, umo: str, history: list, prompt: str, reply: str) -> None:
        """追加本轮 user/assistant 并裁剪后写回会话历史。"""
        try:
            max_history = int(self.config.get("max_history", DEFAULT_MAX_HISTORY) or DEFAULT_MAX_HISTORY)
            conv_mgr, cid, _ = await self._get_conversation(umo)
            history = list(history or [])
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": reply})
            history = history[-max_history:] if max_history > 0 else []
            await conv_mgr.update_conversation(umo, cid, history=history)
        except Exception as e:
            logger.warning(f"[hzm_hello] 保存会话历史失败: {e}")

    # ==================== 内部：调用当前 LLM Provider ====================
    async def _chat_with_provider(self, user_text: str, session_id: str,
                                   image_urls: list = None) -> str:
        """调用 AstrBot 当前启用的 LLM Provider，生成灰泽满风格回复。"""
        provider = self.context.get_using_provider()
        if provider is None:
            return "（还没连上大脑呢，请先在 AstrBot 面板配置一个 LLM Provider）"

        prompt = user_text or "[用户发了一张图片]"

        # 图片先走云端视觉转成文字
        if image_urls:
            vision_api_key = self.config.get("vision_api_key", "")
            descs = []
            for url in image_urls[:3]:
                desc = await vision.describe_image(vision_api_key, url)
                if desc:
                    descs.append(desc)
            if descs:
                prompt += "\n\n[用户发来的图片内容] " + "；".join(descs)

        embed_url = self.config.get("embed_url") or DEFAULT_EMBED_URL
        enable_rag = self.config.get("enable_rag", True)

        # 使用完整版 handle_chat（含记忆、会话、行为路由）
        try:
            reply = await handle_chat(
                user_id=session_id,
                user_msg=prompt,
                provider=provider,
                embed_url=embed_url,
                enable_rag=enable_rag,
            )
        except Exception as e:
            logger.exception(f"[hzm_hello] handle_chat 失败: {e}")
            return f"哎呀，hzm 脑子卡了一下……（{e}）"

        # 写回本轮对话
        await self._save_history(session_id, [], prompt, reply)
        return reply

    def _extract_image_urls(self, event) -> list:
        """从消息链中提取图片 URL。"""
        urls = []
        try:
            for comp in event.get_messages() or []:
                if getattr(comp, "type", None) == ComponentType.Image:
                    url = getattr(comp, "url", None) or getattr(comp, "path", None)
                    if url:
                        urls.append(url)
        except Exception:
            pass
        return urls

    # ==================== 被动监听：无前缀直聊 ====================
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        """所有非指令普通文本 → 进入读秒攒批窗口，静默后统一回复。"""
        try:
            text = (event.message_str or "").strip()
            image_urls = self._extract_image_urls(event)

            # 无文本也无图片 → 不接管
            if not text and not image_urls:
                return
            # 让指令走各自的处理器
            if text.startswith("/"):
                return

            # 调试回声
            if self.config.get("enable_echo", False):
                yield event.plain_result(f"[echo] {text or '(图片)'}")
                return

            # 主开关
            if not self.config.get("enable_chat", True):
                return

            sender = event.get_sender_id()
            session_id = event.unified_msg_origin
            logger.info(f"[hzm_hello] enqueue | sender={sender} text={text!r} imgs={len(image_urls)}")

            # 读秒攒批：同一用户连续消息合并，静默后统一回复（后台 event.send）
            if self.batcher is None:
                self.batcher = ChatBatcher(
                    handle_chat_fn=self._handle_batch_chat,
                    vision_cb=self._describe_one_image,
                    config=self.config,
                )
            self.batcher.enqueue(session_id, text, image_urls, make_send_fn(event))

            # 阻断框架默认 LLM
            event.stop_event()

        except Exception as e:
            logger.exception(f"[hzm_hello] on_any_message 异常: {e}")
            yield event.plain_result(f"哎呀，hzm 脑子卡了一下……（{e}）")

    # ==================== 指令：快速自检 ====================
    @filter.command("hzm")
    async def cmd_hzm(self, event: AstrMessageEvent):
        """/hzm [文本]：自检用。"""
        try:
            user_text = (event.message_str or "").strip()
            greeting = self.config.get("greeting", "灰泽满：")
            logger.info(f"[hzm_hello] /hzm | text={user_text!r}")
            if user_text:
                yield event.plain_result(f"{greeting}{user_text}？hzm 收到了。")
            else:
                yield event.plain_result(f"{greeting}hzm 在的。（v0.4.0 测试通过 ✓）")
        except Exception as e:
            logger.exception(f"[hzm_hello] /hzm 异常: {e}")
            yield event.plain_result(f"指令处理失败：{e}")

    @filter.command("hzm_status")
    async def cmd_status(self, event: AstrMessageEvent):
        """/hzm_status：查看插件版本、开关、Provider 状态。"""
        try:
            provider = self.context.get_using_provider()
            provider_name = "未配置" if provider is None else type(provider).__name__
            info = (
                "灰泽满插件 · 运行状态\n"
                f"- version    : 0.4.0\n"
                f"- enable_chat: {self.config.get('enable_chat', True)}\n"
                f"- enable_rag : {self.config.get('enable_rag', True)}\n"
                f"- max_history: {self.config.get('max_history', DEFAULT_MAX_HISTORY)}\n"
                f"- enable_echo: {self.config.get('enable_echo', False)}\n"
                f"- provider   : {provider_name}\n"
                "六路检索 + RRF融合 + 记忆系统 + 会话追踪 + 行为路由 ✓"
            )
            yield event.plain_result(info)
        except Exception as e:
            logger.exception(f"[hzm_hello] /hzm_status 异常: {e}")
            yield event.plain_result(f"状态查询失败：{e}")
