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

| 层次 | 实现 | 内容 | 生命周期 |
|---|---|---|---|
| 短期记忆 | 内存 deque（按 session_id） | 当前会话对话轮次 | 会话结束清除 |
| 长期记忆 | Milvus 向量检索（接口预留） | 用户偏好摘要、历史记录 | 跨会话持久化 |
| 知识库 | Milvus（RAG 微服务侧） | 汽车用户手册分块 | 静态，定期更新 |

---

## 技术栈

| 层次 | 技术 |
|---|---|
| LLM | 豆包 deepseek-v3（火山引擎方舟，OpenAI 兼容接口） |
| Agent 框架 | LangGraph `create_react_agent` |
| 工具协议 | MCP（天气、地图，接口预留） |
| 向量数据库 | Milvus |
| 服务框架 | FastAPI + SSE 流式响应 |

---

## 目录结构

```
├── main.py                          # FastAPI 入口，/chat + /chat/stream
├── requirements.txt
├── .env.example                     # 环境变量模板
├── config/
│   └── settings.py                  # LLM / RAG / MCP 配置
├── memory/
│   └── conversation_memory.py       # 多 session 滑动窗口记忆管理
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
# 编辑 .env，填入 ARK_API_KEY

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
