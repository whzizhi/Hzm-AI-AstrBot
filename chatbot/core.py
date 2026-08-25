# -*- coding: utf-8 -*-
"""消息处理主循环（AstrBot 插件版）。

组装系统提示词（基础人设 + 人格规则 + 行为指令 + RAG + 长期/短期记忆），
调用 AstrBot Provider 生成回复，并异步更新长期记忆。
"""
import json
import re
import asyncio

from .constants import (
    SYSTEM_PROMPT_FILE, TERMS_FILE,
    VOICE_SAMPLE_REPLY_TRIM_CHARS,
    PHRASE_PHASES_MAX,
)
from .persona import load_persona_rules, build_global_persona_context, load_terms
from .rag import embed
from .retrieval import (
    retrieve_corpus, retrieve_voice_samples, retrieve_phrases,
    retrieve_preferences, retrieve_core_stories, fuse_and_truncate,
    select_behavior_item,
)
from .memory import (
    get_user_history, append_user_history,
    get_user_memory, update_user_memory, build_memory_context,
    _format_profile_summary, MEMORY_EXTRACT_PROMPT,
)
from .session_memory import (
    probe_session, build_session_context, is_emoji_msg, is_emotion_only_query,
)
from .routing import (
    LEGENDARY_REPLIES, LEGENDARY_CONFIRMS, legendary_confirmed, classify_behavior,
)
from .reply_style import (
    split_reply, split_delay, clean_reply, is_echo_reply, _trim_text,
)

# ==================== 基础人设 ====================
if SYSTEM_PROMPT_FILE.exists():
    SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
else:
    SYSTEM_PROMPT = "你是灰泽满，一个虚拟主播。"


def _split_fused(fused_items):
    """把融合结果按源分组。"""
    behaviors, corpus, samples, phrases = [], [], [], []
    for it in fused_items:
        if it.source == "behavior":
            behaviors.append(it)
        elif it.source == "corpus":
            corpus.append(it)
        elif it.source == "voice_sample":
            samples.append(it)
        elif it.source == "phrase":
            phrases.append(it)
    return behaviors, corpus, samples, phrases


def build_terms_note(user_msg: str, denied_terms: set = None) -> str:
    """根据用户消息命中名词库：核心词(always)每次注入 + 命中词注入。"""
    denied = denied_terms or set()
    terms = load_terms()
    if not terms:
        return ""
    msg = user_msg or ""
    notes = []
    for t in terms:
        kw = t.get("keyword", "")
        if not kw or kw in denied:
            continue
        keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
        pattern = t.get("pattern")
        usage = t.get("usage")
        key_hit = any(k in msg for k in keys) or (pattern and re.search(pattern, msg))
        concept_hit = any(w in msg for w in (t.get("usage_triggers") or []))
        hit = t.get("priority") == "always" or key_hit
        if hit:
            parts = [t.get("meaning", "")]
            if t.get("reaction"):
                parts.append(f"被提到时：{t['reaction']}")
            if usage and (key_hit or concept_hit):
                parts.append(f"用词规则：{usage}")
            notes.append(f"{kw}：{'；'.join(parts)}")
        elif usage and concept_hit:
            notes.append(f"用词规则：{usage}")
    return "；".join(notes) if notes else ""


def _hits_on_demand_term(msg: str) -> bool:
    """消息是否命中某个 on-demand 术语。"""
    for t in load_terms():
        kw = t.get("keyword", "")
        if not kw or t.get("priority") == "always":
            continue
        keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
        pattern = t.get("pattern")
        if any(k in msg for k in keys) or (pattern and re.search(pattern, msg)):
            return True
    return False


async def confirm_ambiguous_terms(user_msg: str, provider=None) -> set:
    """LLM 语境确认：剔除误触词条。"""
    msg = (user_msg or "").strip()
    if not msg or not provider:
        return set()
    terms = load_terms()
    candidates = []
    for t in terms:
        kw = t.get("keyword", "")
        if not kw or not t.get("confirm"):
            continue
        keys = [kw] + [str(a) for a in t.get("aliases", []) if a]
        pattern = t.get("pattern")
        if any(k in msg for k in keys) or (pattern and re.search(pattern, msg)):
            candidates.append(t)
    if not candidates:
        return set()
    lines = "\n".join(f"- {t['keyword']}：{t.get('meaning', '')[:80]}" for t in candidates)
    prompt = (
        "你是角色语境的判断器。下面是一批角色黑话/专名词条，用户消息命中了它们的关键词。\n"
        "判断：每个词条在当前语境下是否真的指它定义的含义。\n"
        "只有当语境明显指向其他意思时才剔除。\n\n"
        f"用户消息：{msg}\n\n词条：\n{lines}\n\n"
        '只输出 JSON：{"exclude": ["词条A"]}，都适用输出 {"exclude": []}'
    )
    try:
        resp = await provider.text_chat(
            prompt=prompt,
            system_prompt="你是语境判断器，只输出 JSON。",
            max_tokens=200,
        )
        content = (getattr(resp, "completion_text", None) or "").strip()
        if "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        exclude = set(data.get("exclude", []))
        return {t["keyword"] for t in candidates if t["keyword"] in exclude}
    except Exception as e:
        print(f"⚠️ 术语语境确认失败（放行）: {e}")
        return set()


def build_message_list(user_msg: str, global_persona: str, fused_items: list,
                       memory_context: str, user_history: list,
                       preference_items: list = None,
                       core_stories: list = None,
                       session_context: str = "",
                       query_hint: str = "",
                       denied_terms: set = None) -> list:
    """按优先级组装发送给模型的消息列表。"""
    messages = []
    base_system = SYSTEM_PROMPT
    if global_persona:
        base_system += "\n\n" + global_persona
    messages.append({"role": "system", "content": base_system})

    # 偏好档案
    if preference_items:
        prefs_text = "；".join(
            f"{p.get('category', '')}：{p.get('text', '')}" for p in preference_items
        )
        if prefs_text:
            messages.append({
                "role": "system",
                "content": f"【灰泽满的偏好】{prefs_text}（这是她稳定真实的偏好，若与直播记忆/聊天记录冲突，以本条为准）",
            })

    # 核心记忆
    if core_stories:
        story_text = "；".join(
            f"{s.get('category', '')}：{s.get('text', '')}" for s in core_stories
        )
        if story_text:
            messages.append({
                "role": "system",
                "content": f"【她的核心记忆】{story_text}（这是她过去最深刻的经历，粉丝常拿这些开玩笑。回应时自然带出，别整段复述）",
            })

    # 名词库
    terms_note = build_terms_note(user_msg, denied_terms=denied_terms)
    if terms_note:
        messages.append({
            "role": "system",
            "content": f"【灰泽满的世界】{terms_note}（这些是她世界的词，遇到时按定义理解并带着对应的态度）",
        })

    # 会话级记忆
    if session_context:
        messages.append({
            "role": "system",
            "content": f"【当前会话】{session_context}\n（这是你们这一场对话的调性和发生过的事，回应时要自然地顺着这个语境）",
        })

    behaviors, corpus, samples, phrases = _split_fused(fused_items)

    # 行为指令
    if behaviors:
        behavior_text = "\n\n".join(it.text for it in behaviors if it.text)
        if behavior_text:
            messages.append({
                "role": "system",
                "content": f"【当前情境下的行为指令】请严格按此模式回应：\n{behavior_text}"
            })

    # 直播记忆
    if corpus:
        context = "\n".join(f"- {it.text}" for it in corpus if it.text)
        if context:
            messages.append({
                "role": "system",
                "content": f"【她经历过的相关背景】以下是她过去直播里经历过的事（背景记忆，都是曾经发生的，不是现在）。只能当作'她记得的经历'自然提及，不要模仿里面的叙述口吻，不要整段复述。说话风格看下面的样本：\n{context}"
            })

    # 长期记忆
    if memory_context:
        messages.append({
            "role": "system",
            "content": f"【关于这个绿冻的长期记忆】\n{memory_context}"
        })

    # 短期记忆
    if user_history:
        if isinstance(user_history, list):
            context = "\n".join(user_history)
            context += (
                "\n\n【一致性规则】解释同一件事时，借口要与之前保持一致。\n"
                "【防复读】以上对话中，灰泽满自己说过的话只是历史背景，"
                "用户没有主动追问时，不要反复重复提起。"
            )
            label = "【最近对话记录】"
        else:
            context = f'我说："{user_history}"'
            label = "【关于这个绿冻的上一轮记忆】"
        messages.append({
            "role": "system",
            "content": f"{label}\n{context}"
        })

    # 措辞指纹
    if phrases:
        phrase_blocks = []
        for it in phrases:
            usage = it.extra.get("usage", "")
            phs = it.extra.get("phrases", [])[:PHRASE_PHASES_MAX]
            if phs:
                block = f"· {it.extra.get('meaning', it.item_id)}：{'、'.join(phs)}"
                if usage:
                    block += f"（{usage}）"
                phrase_blocks.append(block)
        if phrase_blocks:
            messages.append({
                "role": "system",
                "content": "【她的固定说法】以下情景她说这些话。表达同类意思时用这些原话组织，不要自创解释性措辞：\n" + "\n".join(phrase_blocks)
            })

    # 声音样本 few-shot
    if samples:
        messages.append({
            "role": "system",
            "content": "【灰泽满的说话方式参考】以下是她真实的对话片段。只学其中的语气、断句、省略号、自称和措辞。括号是她的'心里话标注'，只在情绪顶点才用一个。内容要针对当前话题，不要复述示例里的具体内容。日常回复保持短句（30字内），简短干脆。"
        })
        for it in samples:
            user_part = it.extra.get("user", "")
            reply_part = it.extra.get("reply", "")
            if user_part and reply_part:
                messages.append({"role": "user", "content": user_part})
                messages.append({"role": "assistant", "content": _trim_text(reply_part, VOICE_SAMPLE_REPLY_TRIM_CHARS)})

    # 长度提醒
    messages.append({
        "role": "system",
        "content": "【回复节奏】日常闲聊：一句话说完就停，不再补第二句。30字内。"
    })

    # 短消息语境提示
    if query_hint:
        if is_emoji_msg(user_msg):
            emoji_hint = (
                f"【用户发了表情】{query_hint}\n"
                "用户只发了一个表情，没有任何文字。请按这个表情的真实情绪回应。"
            )
            messages.append({"role": "system", "content": emoji_hint})
        else:
            messages.append({
                "role": "system",
                "content": f"【用户这条消息的语境】{query_hint}\n（上面是这条消息在当前语境下的完整意思，按这个理解回复）"
            })

    messages.append({"role": "user", "content": user_msg})
    return messages


async def assemble_system_prompt(user_msg: str, enable_rag: bool = True,
                                  embed_url: str = "") -> str:
    """组装完整的 system prompt（供 main.py 调用）。

    这是最简版本：只组装静态人格 + 名词库 + 检索结果。
    完整版本（含记忆、会话、行为路由）在 handle_chat 中。
    """
    # 人格规则
    traits, styles, behaviors = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)

    # 名词库
    terms_note = build_terms_note(user_msg)

    # 检索
    retrieval_context = ""
    if enable_rag and user_msg:
        query_vecs = embed([user_msg], embed_url)
        query_vec = query_vecs[0] if query_vecs else None
        if query_vec:
            corpus_items = retrieve_corpus(user_msg, query_vec)
            sample_items = retrieve_voice_samples(user_msg, query_vec)
            phrase_items = retrieve_phrases(user_msg, query_vec)
            fused = fuse_and_truncate(corpus_items, sample_items, [], phrase_items)
            behaviors_f, corpus_f, samples_f, phrases_f = _split_fused(fused)

            parts = []
            if corpus_f:
                lines = [f"- {it.text}" for it in corpus_f if it.text]
                if lines:
                    parts.append("【她记得的】\n" + "\n".join(lines))
            if samples_f:
                lines = [f"粉丝说：{it.extra.get('user', '')} → 灰泽满：{it.extra.get('reply', '')}"
                         for it in samples_f if it.extra.get("user") and it.extra.get("reply")]
                if lines:
                    parts.append("【她的固定说法】（示例）\n" + "\n".join(lines))
            if phrases_f:
                lines = []
                for it in phrases_f:
                    meaning = it.extra.get("meaning", "")
                    phs = it.extra.get("phrases", [])
                    if meaning or phs:
                        lines.append(f"- {meaning}：{'、'.join(phs[:3]) if phs else ''}".strip("："))
                if lines:
                    parts.append("【她的措辞】\n" + "\n".join(lines))
            retrieval_context = "\n\n".join(parts)

    # 组装
    parts = [SYSTEM_PROMPT]
    if global_persona:
        parts.append(global_persona)
    if terms_note:
        parts.append(f"【灰泽满的世界】{terms_note}")
    if retrieval_context:
        parts.append(retrieval_context)

    return "\n\n".join(parts)


async def handle_chat(user_id: str, user_msg: str, provider,
                      embed_url: str = "", enable_rag: bool = True) -> str:
    """处理一条用户消息，返回机器人回复（完整版：含记忆、会话、行为路由）。"""
    query_text = user_msg.strip()
    if not query_text:
        return "……（沉默了一下）"

    user_history = get_user_history(user_id)
    history_text = "\n".join(user_history[-6:]) if user_history else ""

    # --- 会话级记忆：对话前同步探测 ---
    retrieval_query = query_text
    if query_text and provider:
        try:
            retrieval_query = await probe_session(user_id, query_text, history_text, provider)
            if retrieval_query != query_text:
                print(f"[会话记忆] 短 query 扩充: 「{query_text}」→「{retrieval_query}」")
        except Exception as e:
            print(f"⚠️ 会话探测失败: {e}")

    # 命中已知术语：回退用原文
    if query_text and _hits_on_demand_term(query_text):
        retrieval_query = query_text

    session_context = build_session_context(user_id)

    # --- 经典梗硬匹配 ---
    _confirm_history = ""
    for trigger, replies in LEGENDARY_REPLIES.items():
        if trigger in user_msg:
            confirm_tpl = LEGENDARY_CONFIRMS.get(trigger)
            if confirm_tpl:
                if not _confirm_history:
                    _confirm_history = "\n".join(get_user_history(user_id)[-4:])
                if not await legendary_confirmed(user_msg, confirm_tpl, provider, history=_confirm_history):
                    break
            import random
            reply = random.choice(replies)
            append_user_history(user_id, user_msg, reply)
            return reply

    # --- 人格规则 ---
    traits, styles, behaviors = load_persona_rules()
    global_persona = build_global_persona_context(traits, styles)

    # --- 检索 + 融合 ---
    fused_items = []
    preference_items = []
    core_stories = []

    if query_text and not is_emoji_msg(query_text) and enable_rag:
        if is_emotion_only_query(retrieval_query):
            pass  # 纯情绪消息跳过检索
        else:
            # 行为意图分类
            kw_item = select_behavior_item(query_text, "", behaviors)
            if kw_item:
                behavior_items = [kw_item]
            else:
                behavior_intent = ""
                if provider:
                    try:
                        behavior_intent = await classify_behavior(provider, query_text, history_text, behaviors)
                    except Exception:
                        pass
                behavior_item = select_behavior_item(query_text, behavior_intent, behaviors)
                behavior_items = [behavior_item] if behavior_item else []

            # embedding
            query_vecs = embed([retrieval_query or query_text], embed_url)
            query_vec = query_vecs[0] if query_vecs else None

            if query_vec:
                corpus_items = retrieve_corpus(retrieval_query or query_text, query_vec)
                sample_items = retrieve_voice_samples(retrieval_query or query_text, query_vec)
                phrase_items = retrieve_phrases(retrieval_query or query_text, query_vec)
                fused_items = fuse_and_truncate(corpus_items, sample_items, behavior_items, phrase_items)
                preference_items = retrieve_preferences(retrieval_query or query_text, query_vec)
                core_stories = retrieve_core_stories(retrieval_query or query_text, query_vec)

    # --- 记忆 ---
    user_memory_card = get_user_memory(user_id)
    memory_context = build_memory_context(user_memory_card)

    # --- 构建消息列表 ---
    query_hint = retrieval_query if (retrieval_query and retrieval_query != query_text) else ""
    denied_terms = set()
    if provider:
        try:
            denied_terms = await confirm_ambiguous_terms(user_msg, provider)
        except Exception:
            pass

    messages = build_message_list(
        user_msg, global_persona, fused_items, memory_context, user_history,
        preference_items=preference_items, core_stories=core_stories,
        session_context=session_context, query_hint=query_hint,
        denied_terms=denied_terms,
    )

    # --- 调用大模型 ---
    try:
        resp = await provider.text_chat(
            prompt=user_msg,
            system_prompt="\n".join(m["content"] for m in messages if m["role"] == "system"),
            contexts=[m for m in messages if m["role"] in ("user", "assistant")],
        )
        reply = (getattr(resp, "completion_text", None) or "").strip()
        reply = reply or "……（沉默了一下）"
    except Exception as e:
        return f"哎呀，hzm 脑子卡了一下……（{e}）"

    # --- 复读机防护 ---
    recent_bot = [ln[4:] for ln in get_user_history(user_id) if ln.startswith("灰泽满：")]
    if is_echo_reply(reply, recent_bot):
        print(f"[防复读] 检测到复读『{reply[:20]}』，强制重新生成")
        try:
            resp2 = await provider.text_chat(
                prompt=user_msg,
                system_prompt="\n".join(m["content"] for m in messages if m["role"] == "system")
                + f"\n警告：你刚说过『{reply}』，几乎原样复读会让人反感。用完全不同的说法重新回复。",
                contexts=[m for m in messages if m["role"] in ("user", "assistant")],
            )
            reply2 = (getattr(resp2, "completion_text", None) or "").strip()
            if reply2 and not is_echo_reply(reply2, recent_bot):
                reply = reply2
        except Exception:
            pass

    # --- 更新短期记忆 ---
    append_user_history(user_id, user_msg, reply)

    # --- 异步更新长期记忆 ---
    if provider:
        async def _update_memory():
            try:
                prompt = MEMORY_EXTRACT_PROMPT.format(
                    current_summary=_format_profile_summary(user_memory_card),
                    user_msg=user_msg, reply=reply,
                )
                prompt += "\n【强制规则】new_self_fact 一律返回 null。只提取关于用户的信息。"
                resp = await provider.text_chat(
                    prompt=prompt,
                    system_prompt="你是记忆提取器，只输出 JSON。",
                    max_tokens=100,
                )
                content = (getattr(resp, "completion_text", None) or "").strip()
                if content and content != "null":
                    if "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    updates = json.loads(content)
                    if updates:
                        update_user_memory(user_id, updates)
            except Exception as e:
                print(f"[长期记忆] 更新失败: {e}")
        asyncio.create_task(_update_memory())

    return reply
