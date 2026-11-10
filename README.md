# Memory Agent API — 无会话记忆智能体 Demo

演示"**去掉 session 会话栏**"的个人助手后端：用户只凭 `user_id` 对话，agent 通过**分层记忆 + 动态上下文组装**跨对话记住一切。

- Python 3.12.0 + FastAPI + SQLAlchemy(async) + PostgreSQL（腾讯云 43.143.233.242）
- uv 管理虚拟环境（`.venv`）
- LLM 未配置时自动进入**离线演示模式**（基于检索记忆直接作答），配置 `LLM_BASE_URL`（如阿里云 PAI 的 vLLM DeepSeek）即接入真实模型

## 架构映射（与设计文档一一对应）

| 设计概念 | 本项目实现 |
|---|---|
| L0 工作集（最近N轮） | `chat_messages` 表最近 N 条（生产换 Redis） |
| L1 情景记忆（事件日志） | `chat_messages` append-only，唯一事实源 |
| L2 语义记忆（结构化事实） | `memory_facts` 表：key/value + confidence + valid_from/valid_to + source_message_id |
| L3 用户画像（常驻注入） | `user_profiles` 表，每轮拼进 system prompt |
| 查询改写 | `memory_query._KEY_ALIASES`（"名字"→identity.name） |
| 混合检索+重排 | 关键词 bigram 重叠 + 别名加权 + 置信度 + 时间衰减 |
| 事实抽取管线 | `memory_writer`：LLM 抽取优先，规则抽取降级；疑问句与疑问代词值拒绝入库（宁严勿滥） |
| 冲突消解（新替旧） | `identity.*`/`work.*` 单值事实：新值入库、旧值 `superseded` 保留替代谱系；其余多值并存 |
| 上下文组装引擎 | `context_assembler`：画像 + top-K 事实 + 近期对话 → system prompt |

## 目录结构（对齐 KLD-License-api 风格）

```
app/
├── main.py                    # 入口, lifespan 中建表
├── core/                      # config.py(pydantic-settings) / db.py(引擎)
├── api/v1/
│   ├── deps.py                # get_db / get_llm
│   ├── router.py
│   └── endpoints/
│       ├── chat.py            # POST /api/v1/chat 对话入口
│       └── memory.py          # 记忆查看/遗忘/画像
├── models/                    # chat_message / memory_fact / user_profile
├── schemas/chat.py
├── services/                  # memory_writer(写) / memory_query(读) / context_assembler(组装)
└── llm/llm_client.py          # OpenAI 兼容客户端 + 离线演示模式
```

## 运行

```bash
uv venv --python 3.12.0 .venv
uv pip install --python .venv/bin/python \
  "fastapi" "uvicorn[standard]" "sqlalchemy[asyncio]" "asyncpg" \
  "pydantic" "pydantic-settings" "httpx" "python-dotenv"
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8100
```

## 测试页面

服务启动后直接打开 **http://127.0.0.1:8100/**（`app/static/index.html`，由 FastAPI 静态挂载）：

- 左侧对话区：改 user_id 即切换"用户"；每条回复下方可展开**本轮记忆调试**（注入了哪些事实及相关性分数、写管线产生了 added/reinforced/superseded 哪类变更）
- 右侧记忆面板：L2 事实列表（灰底删除线 = superseded 谱系）+ L3 画像，可随时刷新
- 顶部快捷按钮覆盖典型测试场景：冷启动自我介绍 / 跨对话提问 / 改称呼（冲突消解）/ 多值并存 / 问句防污染

## 快速体验（无 session，全程只有 user_id）

```bash
# 第1轮: 自我介绍 -> 自动抽取事实
curl -s -X POST localhost:8100/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","message":"我叫李明，负责AI项目部，回复请用极简中文"}'

# 第2轮: 隔天直接问 -> 靠记忆回答, 无需重复自我介绍
curl -s -X POST localhost:8100/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","message":"我叫什么名字？我现在负责什么？"}'

# 第3轮: 改称呼 -> 冲突消解(旧记忆 superseded, 不物理删除)
curl -s -X POST localhost:8100/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","message":"以后叫我老李"}'

# 查看记忆谱系(含被取代的历史) / 画像
curl -s "localhost:8100/api/v1/memory/facts?user_id=u1&include_superseded=true"
curl -s "localhost:8100/api/v1/memory/profile/u1"
```

每个 chat 响应都带 `memory` 调试块：本轮注入了哪些事实（含相关性分数）、近期消息用了几条、写入管线产生了哪些变更（added/reinforced/superseded）——**记忆链路完全可观测**。

## 接入真实 LLM

`.env` 中配置 OpenAI 兼容端点即可（规则抽取自动升级为 LLM 抽取）：

```env
LLM_BASE_URL=https://xxxx.pai-eas.aliyuncs.com/api/v1
LLM_API_KEY=sk-xxxx
LLM_MODEL=deepseek-r1-qwen-14b
```

## 安全须知

- `.env` 含真实数据库凭据，**禁止提交版本控制**（仅 `.env.example` 可入库）
- 生产部署前：抽取管线应移到消息队列异步执行；删除语义改为软删除 + 审计；增加跨用户记忆隔离校验
