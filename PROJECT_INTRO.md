# 灰泽满（Hazel）· AstrBot 人格化 AI 对话插件

**Hzm-AI-AstrBot** —— 把虚拟主播「灰泽满」的说话方式、行为习惯与记忆，完整搬进 [AstrBot](https://github.com/AstrBotDevs/AstrBot) 的无前缀直聊插件。

> 一句话：**不是让 AI「会聊天」，而是让 AI「像某个人」。** 项目不靠堆提示词，而是靠"人格数据分层 + 语义检索按需注入 + 输出侧真人感工程"，让回复稳定地像灰泽满本人说出来的话。

---

## 一、这是什么

灰泽满（Hazel）是一个虚拟主播，官方人设是"永远 16 岁的风纪委员"，说话有四川妹子的直率、爱自嘲、嘴硬心软。本项目把这套人设做成一个 **AstrBot 插件**：

- 收到普通消息**无需任何前缀**，直接以灰泽满的身份接管回复；
- 回复**按句分段、模拟真人打字节奏**一条条发出来；
- 生成内容会经过**输出清洗**，把"她/他"自指、括号滥用、省略号泛滥等 AI 腔确定性修掉。

它不绑定任何具体大模型厂商，使用 AstrBot 当前启用的 LLM Provider（本仓库部署示例为 DeepSeek-V4-Flash）。

## 二、背景：从 Hzm-AI-Bot 到 Hzm-AI-AstrBot

本项目由 [MureasAm/Hzm-AI-Bot](https://github.com/MureasAm/Hzm-AI-Bot) 移植而来。原项目是一套基于 NoneBot2 的角色一致性工程：从真人直播语料蒸馏人格数据、多路检索按需注入、分层提示词组装，并用评测闭环量化调参。

本插件保留了原项目**最核心的方法论**（人格分层数据 + 语义检索 + 输出后处理），并针对 AstrBot 的插件生态做了重构与精简：

| 维度 | 原 Hzm-AI-Bot | Hzm-AI-AstrBot |
|---|---|---|
| 框架 | NoneBot2 + OneBot V11 | AstrBot（无前缀直聊） |
| 接入 | NapCatQQ | AstrBot 平台适配层（QQ / 其他均可） |
| LLM | 直连 DeepSeek API | AstrBot 当前 Provider（不绑定厂商） |
| 人格数据 | 同一套 persona/ | **完全复用同一套数据** |
| 检索 | 五路 + RRF 融合 + 预算截断 | 五路语义检索（本地 embedding） |
| 输出 | 读秒攒批 + 分批发送 + 清洗 | 分段分句 + 打字节奏 + 清洗 |

## 三、它是怎么"像灰泽满"的

核心方法论一句话：**样本与数据 > 提示词规则**。给模型看"灰泽满真实怎么说话"，比规定"你要怎么说话"有效得多。

### 3.1 人格分层数据（`persona/`）

人格不是一段话，而是按职责分层的数据（实际条目数实时统计）：

```
persona/
├── core/       核心人设：system_prompt.txt（说话节奏/自我称呼/行为底色）+ traits.json(4 条【性格基底】) + styles.json(3 条【语言风格】)
├── behavior/   行为：behaviors.json(8 条【行为指令】) + behavior_keywords.json(判别词兜底) + trigger_vectors.json
├── speech/     说话：voice_samples.json(83 条【她的固定说法】few-shot) + phrases.json(11 组【措辞指纹】) + 对应向量
└── world/      世界：terms.json(25 条【核心名词】) + corpus(315 条直播记忆) + core_stories(5 条核心记忆) + preferences(23 条偏好) + 对应向量
```

每一层管一件事：**她是谁**（core）→ **怎么反应**（behavior）→ **怎么说**（speech）→ **懂什么、经历过什么**（world）。出问题时能精确定位到某一层。

### 3.2 五路语义检索（本地 embedding，按需注入）

收到消息后，用**本地 embedding 服务**（`fastembed + BAAI/bge-small-zh-v1.5`，OpenAI 兼容 `/v1/embeddings`）把用户消息向量化，再对五个知识库做余弦相似度检索，命中即注入提示词：

1. **直播记忆 corpus** —— 她经历过的相关背景；
2. **声音样本 voice_samples** —— 示范"怎么说话"的 few-shot 原话；
3. **措辞指纹 phrases** —— 同一意思用她的真实原话锚定，不自创措辞；
4. **核心记忆 core_stories** —— 印象最深的结晶；
5. **偏好 preferences** —— 她稳定真实的喜好（与记忆冲突时以偏好为准）。

阈值、每路条数都在 `chatbot/config.py` 里可调。**embedding 服务不启动时插件优雅降级**为"纯静态人格 + 核心名词"，不会报错。

### 3.3 行为指令（判别词兜底）

被夸、被质疑、失约被催等情境，通过 `behavior_keywords.json` 的判别词确定性命中，注入对应【行为指令】（触发情境 + 回应模式 + 真人示范），比 embedding 猜意图更可靠。

### 3.4 组装成 system prompt（`chatbot/core.py`）

`assemble_system_prompt()` 按固定顺序把以上各层拼接成最终 system prompt：静态人格 → 核心名词 → 行为指令 → 五路检索结果 → **【回复节奏】收尾层**（"日常闲聊一句话说完就停，30字内"，对齐原版末条约束）。

## 四、输出侧：真人感工程

这是本插件相比"套壳聊天 bot"最明显的差异，也是近期补齐的重点：

### 4.1 输出清洗 `clean_reply()`（`chatbot/reply_style.py`）

LLM 生成的原文不可信，做**确定性后处理**：

- **自指兜底**：灰泽满按人设用名字自称、绝不用"她/他"指代自己。检测到回复里没有其他人名时，把"她/他"替换成"灰泽满"，从根上杜绝「灰泽满……她……」这种 AI 腔（原版【自我称呼】规则的确定性保障）；
- **括号纪律**：剥掉开头括号前缀，整条至多保留 1 个括号；
- **省略号纪律**：连续省略号归一、剥开头省略号、犹豫转逗号、整条至多 2 个；
- **结巴消融**："那、那"→"那那"。

### 4.2 分段分句 + 真人打字节奏（`split_reply()` / `split_delay()`）

长回复**按句切分**（`。！？` 切句、省略号边界判断、每段至多 1 个逗号、短碎片合并、上限 4 条防刷屏），然后**逐条发送，句间按字数模拟打字延迟**：基础 1.5s + 每字 0.1s，夹在 1.8~5s 之间，并带 ±15% 随机抖动——避免机械等长，像真人一条条打字发出来。

### 4.3 一条消息的完整生命周期

```
用户消息
  → on_any_message（AstrBot 事件，无需前缀）
  → 过滤（指令 /、纯表情、开关）
  → assemble_system_prompt（静态人格 + 名词 + 行为 + 五路检索 + 回复节奏）
  → provider.text_chat（当前 LLM Provider 生成）
  → clean_reply（自指/括号/省略号纪律）
  → split_reply（按句分段，≤4 条）+ 每段再 clean_reply
  → 逐段 yield + asyncio.sleep(split_delay)（打字节奏）
  → event.stop_event()（阻断框架默认 LLM，避免双回复）
```

## 五、技术架构与目录

```
astrbot_plugin_hzm_hello/
├── main.py                   AstrBot 插件入口（Star 子类，无前缀直聊 + 指令）
├── metadata.yaml             插件清单
├── _conf_schema.json         配置项 schema（面板可视化编辑）
├── requirements.txt          运行时无额外依赖（仅 stdlib + AstrBot API）
├── README.md                 使用文档
├── chatbot/                  核心代码包
│   ├── config.py             常量 / 路径 / 检索阈值 / 分段分句参数
│   ├── persona.py            静态人格 + 名词库 + 行为指令
│   ├── retrieval.py          embedding + 余弦相似度 + 五路检索
│   ├── reply_style.py        输出清洗（clean_reply）+ 分段分句（split_reply）+ 打字延迟（split_delay）
│   └── core.py               总装 assemble_system_prompt
├── persona/                  人格与知识库数据（原始数据 + 预计算向量）
└── scripts/
    ├── embed_server.py       本地 embedding 服务（fastembed + bge-small-zh-v1.5）
    └── precompute_vectors.py 重新预计算全部向量（6 步流水线）
```

## 六、部署与使用

### 环境要求

- AstrBot（建议最新版）+ 一个可用的 LLM Provider；
- 可选：本地 embedding 服务（不装则关闭 RAG，降级为静态人格）。

### 本地 embedding 服务（可选）

```bash
python -m venv venv && source venv/bin/activate
pip install fastembed fastapi uvicorn
python scripts/embed_server.py   # 监听 0.0.0.0:8000
```

Docker 部署 AstrBot 时，插件配置 `embed_url` 填 `http://172.18.0.1:8000/v1/embeddings`（宿主机网关）；裸机填 `http://127.0.0.1:8000/v1/embeddings`。

### 修改人格数据后重算向量

```bash
EMBED_URL=http://127.0.0.1:8000/v1/embeddings python scripts/precompute_vectors.py
```

### 配置项（AstrBot 面板可视化编辑）

| 键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enable_chat` | bool | true | 是否接管所有非指令消息 |
| `enable_rag` | bool | true | 是否开启五路语义检索 |
| `embed_url` | string | "" | embedding 服务地址（空 = 内置默认） |
| `split_reply_enabled` | bool | true | 长回复分段分句 + 打字节奏发送 |
| `greeting` | string | 灰泽满： | `/hzm` 回复前缀 |
| `enable_echo` | bool | false | 调试回声 |

### 指令

| 指令 | 说明 |
|---|---|
| `/hzm` | 自检，验证插件可达 |
| `/hzm_status` | 查看版本 / 开关 / Provider 状态 |

## 七、设计亮点小结

1. **人格数据与代码解耦**：换人设只改 `persona/` 数据，不改代码；
2. **按需注入**：5 路检索 + 阈值控制，不把全部知识塞进提示词；
3. **确定性兜底**：模型管不住的"她自指 / 括号 / 省略号 / 长句"，用 `clean_reply` + `split_reply` 从输出侧硬性修掉；
4. **真人打字感**：按句分段 + 按字数延迟 + 随机抖动，回复像人一条条打出来；
5. **优雅降级**：embedding 服务不可用时，插件自动退化为纯静态人格，不报错；
6. **零额外运行时依赖**：插件本体只依赖 AstrBot API 和 Python 标准库。

## 八、局限与后续方向（诚实说明）

- **内容幻觉无法 100% 兜住**：模型偶尔会编造"她玩过某游戏"这类细节（即使提示词写了"不懂就说不懂"），输出清洗只能修形式、不能修事实；后续可接入原版的长期/短期记忆与复读检测来进一步约束；
- **当前无对话历史注入**（`contexts=[]`）：同一场对话的多轮一致性依赖模型自身，接入会话记忆是下一步；
- 检索是预计算向量 + 余弦相似度，未做 RRF 融合与关键词门（原版有）；数据量变大后需要补。

## 九、致谢

- 人格数据与核心方法论：源自 [MureasAm/Hzm-AI-Bot](https://github.com/MureasAm/Hzm-AI-Bot)；
- AstrBot 插件生态：[AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)；
- 移植与工程化：在人工监督下由 AI 协作完成。

---

*本文档为项目介绍，配套使用文档见同目录 [README.md](./README.md)。*
