# -*- coding: utf-8 -*-
"""总装：静态人格 + 动态检索 → 最终 system prompt。"""
from . import persona
from . import retrieval


def assemble_system_prompt(user_text: str, enable_rag: bool = True, embed_url: str = "") -> str:
    """组装最终 system prompt。enable_rag=False 时只保留静态层 + always 名词。"""
    parts = [persona.load_static_prompt()]

    terms_block = persona.format_terms(user_text)
    if terms_block:
        parts.append(terms_block)

    if enable_rag:
        behavior_block = persona.match_behavior(user_text)
        if behavior_block:
            parts.append(behavior_block)

        qv = retrieval.embed([user_text], embed_url)
        query_vec = qv[0] if qv else []
        if query_vec:
            for block in (
                retrieval.retrieve_corpus(query_vec),
                retrieval.retrieve_voice_samples(query_vec),
                retrieval.retrieve_phrases(query_vec),
                retrieval.retrieve_core_stories(query_vec),
                retrieval.retrieve_preferences(query_vec),
            ):
                if block:
                    parts.append(block)

    # 极简长度提醒：一句一停，不展开（对齐原版 build_message_list 末条，只约束日常闲聊）
    parts.append("【回复节奏】日常闲聊：一句话说完就停，不再补第二句。30字内。")

    return "\n\n".join(parts)
