# 汽车智能对话系统

基于 LangGraph + LangChain + Milvus 的多 Agent 汽车语音对话系统，支持车辆知识问答、智能导航规划、个性化推荐等场景。

---

## 系统架构

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                  Orchestrator Agent                   │
│              （主控，LangGraph ReAct）                 │
└──────┬───────────────────────────────────────────────┘
       │
       ▼ ① 安全过滤
┌──────────────────────┐
│     拒识 Agent        │──→ REJECT ──→ 返回拒绝回复（终止）
│   （接口预留）         │
└──────┬───────────────┘
       │ PASS，进入工具路由
       ▼ ② 工具路由（ReAct 自动选择）
       │
  ┌────┴──────┬──────────────┬──────────────┬──────────────┐
  │           │              │              │              │
  ▼           ▼              ▼              ▼              ▼
rag_query  get_weather   get_navigation  navigate      recommend
（Tool）    （Tool）        （Tool）        （Tool）       （Tool）
                                            │              │
                                            ▼              ▼
                                     ┌──────────┐  ┌──────────────┐
                                     │ 导航规划  │  │  个性化推荐  │
                                     │ Sub Agent│  │  Sub Agent   │
                                     └────┬─────┘  └──────┬───────┘
                                          │                │
                                ┌─────────┴──┐    ┌────────┴─────┐
                                │            │    │              │
                             路况查询      补能站  用户偏好     POI搜索
                             天气查询      导航    天气查询
```

---

## Agent 说明

### Orchestrator Agent（主控）
- 框架：LangGraph `create_react_agent`
- 职责：意图理解、工具路由、多轮对话管理、结果汇总
- 执行顺序：拒识检查 → ReAct 循环 → 保存记忆

### 拒识 Agent（接口预留）
- 职责：安全前置过滤，拦截领域越界 / 安全违规 / 能力边界 / 恶意注入
- 接口：`check(query) → {action: PASS|REJECT, type, reason}`
- 当前：默认 PASS，对接 LLM 分类器后生效

### RAG Agent（通过 `rag_query` Tool 调用）
- 职责：检索汽车用户手册知识
- 实现：HTTP 调用已有 RAG 微服务（接口预留）
- 向量库：Milvus（字段：chunk_id / car_model / chapter / content / embedding）

### 导航规划 Sub Agent（通过 `navigate` Tool 触发）
- 职责：复杂出行规划，综合多因素制定出行方案
- 内部工具：路况查询、天气查询、补能站查询、导航路线
- 触发时机：用户有复杂路线规划需求时由 Orchestrator 调用

### 个性化推荐 Sub Agent（通过 `recommend` Tool 触发）
- 职责：结合用户历史偏好 + 情境（位置/天气/时间）做个性化推荐
- 内部工具：用户偏好查询、POI 搜索、天气查询
- 触发时机：用户询问附近推荐、个性化建议时由 Orchestrator 调用

---

## 记忆系统

三层设计：短期（进程内）→ 中期（Redis 摘要）→ 长期（Redis 偏好 + Milvus 向量），每次请求读取三层拼入上下文，会话内每 N 轮或会话结束时向上沉淀。

### 整体流程

```
请求进入
  │
  ├─① 读取长期记忆（按 user_id）
  │      ├─ Redis 偏好：结构化偏好事实（权威来源）
  │      ├─ Redis 摘要：重要性最高的 5 条历史摘要
  │      └─ Milvus：按当前 query 语义召回相关历史摘要
  │      （三者合并 + 当前 user_id 注入 SystemMessage）
  │
  ├─② 读取短期记忆（按 user_id:session_id）
  │      └─ 最近 10 轮对话拼入消息列表
  │
  ├─③ Agent 执行（ReAct 循环）
  │
  ├─④ 保存本轮到短期记忆
  │      └─ 每累积 20 轮触发增量摘要 → 写入中/长期记忆，并清除已摘要部分
  │
  └─⑤ 会话关闭（DELETE /session）
         └─ 对剩余未摘要轮次再做一次摘要写入，清除短期记忆
```

### 短期记忆

| 项 | 说明 |
|---|---|
| 存储 | 进程内 `deque`（滑动窗口，最近 10 轮）+ 完整历史 list |
| Key | `user_id:session_id` |
| 增量 flush | 每累积 20 轮（`SESSION_FLUSH_EVERY`）自动将最旧 20 轮摘要写入长期记忆，从 list 中移除，内存上限为 `2×flush_every-1` 轮 |
| 生命周期 | 会话关闭或进程重启时清除，不持久化 |

### 中期记忆（Redis 摘要层）

| 项 | 说明 |
|---|---|
| 存储 | Redis Sorted Set，score = 写入时间戳，上限 20 条 |
| 写入 | 会话内增量 flush 或会话关闭时，由 LLM 将对话压缩为 1-3 句摘要 |
| 读取 | 按综合重要性评分（`0.7×时效衰减 + 0.3×语义召回频率`）取前 5 条 |
| 淘汰 | 超限时淘汰评分最低条目，同步删除 Milvus 对应向量 |

### 长期记忆（Redis 偏好 + Milvus 向量）

**偏好层（Redis String）**

| 项 | 说明 |
|---|---|
| 格式 | 结构化 JSON，每条含 `content` + `confidence`（1-10） |
| 冲突消解 | LLM 判断：一致→+1，矛盾→被矛盾方-1，全新→初始值 5，置信度≤2 自动删除 |
| 权威性 | 是当前用户偏好的唯一权威来源，优先级高于摘要 |

**向量层（Milvus）**

| 项 | 说明 |
|---|---|
| 存储 | 摘要文本 + embedding，按 user_id 隔离 |
| 近重复去重 | 写入前检测相似度>0.95 的已有向量，命中则替换 |
| 偏好冲突检测 | 写入前召回相似度>0.6 的相关摘要，由 LLM 判断偏好矛盾，矛盾则删除旧摘要 |
| TTL | 默认 30 天，支持按 `created_at` 批量清理 |

### Redis 数据结构

```
ltm:{user_id}
  ├── preferences  (String/JSON)  结构化偏好事实列表，含置信度
  ├── summaries    (Sorted Set)   历史会话摘要，score = 写入时间戳
  ├── sum_access   (Hash)         md5(摘要) → 语义召回次数（重要性计算用）
  └── sum_ids      (Hash)         md5(摘要) → Milvus 主键 ID（联动删除用）
```

### 三层对比

| 层次 | 实现 | 写入时机 | 内存/容量上限 | 持久化 |
|---|---|---|---|---|
| 短期记忆 | 进程内 deque + list | 每轮对话后 | `2×flush_every-1` 轮 | 否 |
| 中期记忆 | Redis Sorted Set | 每 20 轮或会话关闭 | 20 条摘要 | 是（TTL 30天） |
| 长期记忆（偏好） | Redis String/JSON | 每 20 轮或会话关闭 | 无硬上限，置信度淘汰 | 是（TTL 30天） |
| 长期记忆（向量） | Milvus | 每 20 轮或会话关闭 | 与中期摘要同步 | 是（TTL 30天） |

---

## 技术栈

| 层次 | 技术 |
|---|---|
| LLM | 豆包 deepseek-v3（火山引擎方舟，OpenAI 兼容接口） |
| Agent 框架 | LangGraph `create_react_agent` |
| 工具协议 | MCP（天气、地图，接口预留） |
| 向量数据库 | Milvus |
| 服务框架 | FastAPI + SSE 流式响应 |
| 长期记忆 | Redis（会话摘要 + 用户偏好，TTL 30天） |

---

## 目录结构

```
├── main.py                          # FastAPI 入口，/chat + /chat/stream
├── requirements.txt
├── .env.example                     # 环境变量模板
├── config/
│   └── settings.py                  # LLM / RAG / MCP 配置
├── memory/
│   ├── conversation_memory.py       # 短期记忆：多 session 滑动窗口（进程内 deque）
│   ├── long_term_memory.py          # 长期记忆：偏好冲突消解 + 重要性遗忘 + Milvus 联动
│   ├── milvus_client.py             # Milvus 向量库客户端（摘要向量存储、去重、TTL 清理）
│   └── redis_client.py              # Redis 连接池客户端
├── agents/
│   ├── orchestrator_agent.py        # 主控 ReAct Agent
│   ├── reject_agent.py              # 拒识 Agent（接口预留）
│   ├── navigation_agent.py          # 导航规划 Sub Agent
│   └── recommendation_agent.py     # 个性化推荐 Sub Agent
└── tools/
    ├── rag_tool.py                  # RAG 知识检索（接口预留）
    ├── mcp_weather_tool.py          # MCP 天气（接口预留）
    ├── mcp_map_tool.py              # MCP 地图导航（接口预留）
    ├── road_condition_tool.py       # 路况查询（接口预留）
    ├── fuel_station_tool.py         # 补能站查询（接口预留）
    ├── poi_search_tool.py           # POI 兴趣点搜索（接口预留）
    └── user_preference_tool.py      # 用户偏好查询
```

---

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 ARK_API_KEY、REDIS_URL 等

# 3. 启动服务
python main.py
# 服务运行在 http://localhost:8000
```

**同步接口：**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "我的车空调不制冷是什么原因？"}'
```

**流式接口（SSE）：**
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我规划从北京到天津的路线，顺便看看沿途天气"}'
```
