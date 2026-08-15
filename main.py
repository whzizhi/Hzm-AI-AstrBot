# -*- coding: utf-8 -*-
"""
astrbot_plugin_hzm_hello · 插件主入口（v0.2.0 · 无前缀直聊）

变更点（相较 v0.1.0）：
1. on_any_message 从"echo 调试"升级为"直接调 LLM 生成灰泽满风格回复"
2. 命中所有普通文本消息（非指令）→ 走当前 Provider → 用 event.stop_event() 阻断框架默认 LLM
3. 新增配置项：enable_chat / system_prompt
4. 未配置 Provider 时给出可读提示，不会静默失败
"""

import asyncio
import os
import sys

# AstrBot 按包加载插件时，同目录自定义模块不在 sys.path 上，需显式加入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astrbot.api.star import Star, Context, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import logger

from chatbot.core import assemble_system_prompt


DEFAULT_EMBED_URL = "http://172.18.0.1:8000/v1/embeddings"


@register(
    "astrbot_plugin_hzm_hello",
    "MureasAm (scaffold)",
    "灰泽满插件：无前缀直聊 + 人格组装 + 语义检索",
    "0.3.0",
)
class HzmHelloPlugin(Star):
    """插件主类。继承 Star，AstrBot 自动扫描并实例化。"""

    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        logger.info("[hzm_hello] __init__ 完成")

    # ==================== 生命周期 ====================
    async def initialize(self):
        logger.info(
            f"[hzm_hello] initialize | "
            f"enable_chat={self.config.get('enable_chat', True)} "
            f"enable_echo={self.config.get('enable_echo', False)} "
            f"greeting={self.config.get('greeting', '灰泽满：')!r}"
        )

    async def terminate(self):
        logger.info("[hzm_hello] terminate 完成")

    # ==================== 内部：调用当前 LLM Provider ====================
    async def _chat_with_provider(self, user_text: str, session_id: str) -> str:
        """调用 AstrBot 当前启用的 LLM Provider，生成灰泽满风格回复。

        失败场景：
        - 未配置 Provider → 返回可读提示
        - Provider 调用抛异常 → 返回可读兜底
        """
        provider = self.context.get_using_provider()
        if provider is None:
            return "（还没连上大脑呢，请先在 AstrBot 面板配置一个 LLM Provider）"

        system_prompt = await asyncio.to_thread(
            assemble_system_prompt,
            user_text,
            self.config.get("enable_rag", True),
            self.config.get("embed_url") or DEFAULT_EMBED_URL,
        )
        try:
            resp = await provider.text_chat(
                prompt=user_text,
                session_id=session_id,
                system_prompt=system_prompt,
                contexts=[],  # 骨架阶段不带历史；接入完整 memory 后再传短期记忆
            )
            reply = (getattr(resp, "completion_text", None) or "").strip()
            return reply or "……（沉默了一下）"
        except Exception as e:
            logger.exception(f"[hzm_hello] provider.text_chat 失败: {e}")
            return f"哎呀，hzm 脑子卡了一下……（{e}）"

    # ==================== 被动监听：无前缀直聊 ====================
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        """所有非指令普通文本 → 灰泽满接管。

        约束：
        - 指令消息（以 / 开头）交给指令处理器，不在这里回复
        - enable_chat=False 时完全不接管，让框架默认 LLM/其他插件处理
        - 接管后 event.stop_event() 阻断后续 handler，避免双回复
        """
        try:
            text = (event.message_str or "").strip()
            if not text:
                return
            # 让 /hzm、/hzm_status、以及其他插件指令走各自的指令处理器
            if text.startswith("/"):
                return

            # 调试回声（默认关）
            if self.config.get("enable_echo", False):
                yield event.plain_result(f"[echo] {text}")
                return

            # 主开关：关闭则不接管，让框架默认 LLM 生效
            if not self.config.get("enable_chat", True):
                return

            sender = event.get_sender_id()
            session_id = event.unified_msg_origin
            logger.info(f"[hzm_hello] chat | sender={sender} text={text!r}")

            reply = await self._chat_with_provider(text, session_id)
            yield event.plain_result(reply)

            # 阻断框架默认 LLM 与后续被动 handler，避免双回复
            event.stop_event()

        except Exception as e:
            logger.exception(f"[hzm_hello] on_any_message 异常: {e}")
            yield event.plain_result(f"哎呀，hzm 脑子卡了一下……（{e}）")

    # ==================== 指令：快速自检 ====================
    @filter.command("hzm")
    async def cmd_hzm(self, event: AstrMessageEvent):
        """/hzm [文本]：自检用。不走 LLM，直接返回固定应答，用于验证插件可达性。"""
        try:
            user_text = (event.message_str or "").strip()
            greeting = self.config.get("greeting", "灰泽满：")
            logger.info(f"[hzm_hello] /hzm | text={user_text!r}")
            if user_text:
                yield event.plain_result(f"{greeting}{user_text}？hzm 收到了。")
            else:
                yield event.plain_result(f"{greeting}hzm 在的。（骨架插件测试通过 ✓）")
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
                f"- version    : 0.3.0\n"
                f"- enable_chat: {self.config.get('enable_chat', True)}\n"
                f"- enable_rag : {self.config.get('enable_rag', True)}\n"
                f"- enable_echo: {self.config.get('enable_echo', False)}\n"
                f"- provider   : {provider_name}\n"
                "人格组装 + 知识库检索已接入 ✓"
            )
            yield event.plain_result(info)
        except Exception as e:
            logger.exception(f"[hzm_hello] /hzm_status 异常: {e}")
            yield event.plain_result(f"状态查询失败：{e}")
