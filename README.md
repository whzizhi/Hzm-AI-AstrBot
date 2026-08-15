# 灰泽满（Hazel / hzm）· AstrBot 插件

基于 [Hzm-AI-Bot](https://github.com/MureasAm/Hzm-AI-Bot) 的灰泽满人格与知识库，移植到 [AstrBot](https://github.com/AstrBotDevs/AstrBot) 的无前缀直聊插件。

- **静态人格**：system_prompt + 性格基底 + 语言风格 + 核心名词（terms）
- **动态检索**：直播记忆 / 声音样本 / 措辞 / 核心记忆 / 偏好 —— 本地 embedding 语义检索
- **行为指令**：判别词兜底命中（被夸 / 被质疑 / 失约被催 等）

## 注意事项
不建议将此项目部署在 AstrBot 上，并通过腾讯官方开放平台将机器人接入 QQ。若仅用于个人使用，则配置较为便捷；但若计划在官方平台上向其他用户开放服务，则需具备企业账户，且使用 Webhook 方式时还需完成国内域名备案。请了解之后自行斟酌。

## 目录结构

```
astrbot_plugin_hzm_hello/
├── main.py                   AstrBot 插件入口（Star 子类，无前缀直聊）
├── metadata.yaml             插件清单
├── _conf_schema.json         配置项 schema（面板可视化编辑）
├── requirements.txt          插件运行时依赖（无额外依赖）
├── chatbot/                  核心代码包（仿 src/plugins/chatbot/）
│   ├── __init__.py
│   ├── config.py             常量 / 路径 / 检索阈值 / 分段分句参数
│   ├── persona.py            静态人格 + 名词库 + 行为指令
│   ├── retrieval.py          embedding + 余弦相似度 + 五路检索
│   ├── reply_style.py        回复分段分句（split_reply）+ 打字节奏延迟（split_delay）
│   └── core.py               总装 assemble_system_prompt
├── persona/                  人格与知识库数据（原始数据 + 预计算向量）
│   ├── core/                 system_prompt / traits / styles
│   ├── behavior/             behaviors / 判别词 / trigger 向量
│   ├── speech/               voice_samples / phrases（原始 + 向量）
│   └── world/                terms / corpus / stories / preferences（原始 + 向量）
└── scripts/
    └── precompute_vectors.py 重新预计算全部向量
```

## 部署

1. 把本插件目录放进 AstrBot 的 `data/plugins/`（或在面板插件管理导入）。
2. 重启 / 重载插件。
3. 在 AstrBot 面板配置一个 LLM Provider（AstrBot → 服务提供商）。

插件会在每次收到非指令消息时，组装「静态人格 + 命中名词 + 行为指令 + 语义检索结果」作为 system prompt，再调用当前 Provider 生成回复。

## 可选：本地 embedding 服务（RAG 语义检索）

五路语义检索依赖一个 OpenAI 兼容的 `/v1/embeddings` 服务（默认 `fastembed + BAAI/bge-small-zh-v1.5`）。

- **未配置时**：插件会优雅降级为「仅静态人格 + 核心名词」，不会报错。
- **配置方式**：插件配置项 `embed_url` 填你的服务地址：
  - Docker 部署 AstrBot：通常为 `http://172.x.0.1:8000/v1/embeddings`（宿主机网关）
  - 裸机部署：`http://127.0.0.1:8000/v1/embeddings`

搭建参考（脚本已随包提供 `scripts/embed_server.py`）：

```bash
python -m venv venv && source venv/bin/activate
pip install fastembed fastapi uvicorn
python scripts/embed_server.py   # 监听 0.0.0.0:8000，模型自动下载到 scripts/models/
```

## 重新生成向量

改了 `persona/` 下的原始数据后，重新生成向量：

```bash
EMBED_URL=http://127.0.0.1:8000/v1/embeddings python scripts/precompute_vectors.py
```

## 配置项

| 键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enable_chat` | bool | true | 是否接管所有非指令消息 |
| `enable_rag` | bool | true | 是否开启语义检索 |
| `embed_url` | string | "" | embedding 服务地址（空 = 内置默认） |
| `greeting` | string | `灰泽满：` | `/hzm` 回复前缀 |
| `enable_echo` | bool | false | 调试回声 |
| `split_reply_enabled` | bool | true | 长回复分段分句 + 打字节奏发送（真人打字感；句间按字数延迟，参数见 `chatbot/config.py` 的 `SPLIT_*`） |

## 指令

| 指令 | 说明 |
|---|---|
| `/hzm` | 自检，验证插件可达 |
| `/hzm_status` | 查看版本 / 开关 / Provider 状态 |

## 致谢

- 项目骨架与数据：源自 [MureasAm/Hzm-AI-Bot](https://github.com/MureasAm/Hzm-AI-Bot)
- AstrBot 移植与实现：Claude、DeepSeek在人工监督下完成。