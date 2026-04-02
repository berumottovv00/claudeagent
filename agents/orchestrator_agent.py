"""
Orchestrator Agent —— 主控 ReAct Agent
负责意图路由、工具调用协调、多轮对话管理。
执行顺序：拒识检查 → ReAct 循环（Thought→Action→Observation）→ Final Answer
"""
from typing import AsyncGenerator, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import llm_config
from memory import session_memory_manager, long_term_memory_manager
from tools import ALL_TOOLS
from .reject_agent import reject_agent

# ----------------------------------------------------------------
# 系统 Prompt（通过 state_modifier 注入）
# ----------------------------------------------------------------
SYSTEM_PROMPT = """你是一个专业的汽车智能助手，专注解答车辆使用、保养、故障排查及出行服务等问题。
回答要准确、简洁，优先依据工具返回的内容，不确定时如实告知。

工具选择原则：
- 车辆功能/规格/故障/操作说明 → rag_query
- 简单天气查询 → get_weather
- 简单路线查询 → get_navigation
- 复杂出行规划（需综合路况/天气/补能） → navigate（内部由导航规划 Agent 处理）
- 个性化地点/活动推荐 → recommend（内部由推荐 Agent 处理）"""


class OrchestratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        self.tools = ALL_TOOLS
        # LangGraph prebuilt ReAct agent，state_modifier 注入系统 Prompt
        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def _build_messages(self, user_id: str, session_id: str, query: str):
        """构建消息列表：长期记忆上下文 + 短期历史 + 当前问题"""
        messages = []
        ltm_context = long_term_memory_manager.get_context(user_id, query)
        system_parts = [f"当前用户 ID：{user_id}"]
        if ltm_context:
            system_parts.append(f"以下是该用户的历史偏好与会话摘要，供参考：\n{ltm_context}")
        messages.append(SystemMessage(content="\n".join(system_parts)))
        # human message直接
        messages.extend(session_memory_manager.get_history_messages(user_id, session_id))
        messages.append(HumanMessage(content=query))
        return messages

    def close_session(self, user_id: str, session_id: str) -> None:
        """会话结束：将短期记忆摘要写入长期记忆，然后清除短期记忆。"""
        messages = session_memory_manager.get_full_history(user_id, session_id)
        long_term_memory_manager.save_session(user_id, messages, self.llm)
        session_memory_manager.clear(user_id, session_id)

    def run(self, user_id: str, session_id: str, query: str) -> str:
        """同步运行，返回最终答案字符串"""
        # ① 拒识检查
        reject_result = reject_agent.check(query)
        if reject_result.action == "REJECT":
            return reject_agent.get_reject_message()

        # ② 构建消息列表（长期记忆上下文 + 短期历史 + 当前问题）
        messages = self._build_messages(user_id, session_id, query)

        # ③ 执行 ReAct 循环
        result = self._agent.invoke({"messages": messages})
        answer = result["messages"][-1].content

        # ④ 保存本轮对话到短期记忆
        session_memory_manager.save_turn(user_id, session_id, query, answer)

        return answer

    async def astream(self, user_id: str, session_id: str, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """异步流式运行，逐步 yield SSE 事件"""
        # ① 拒识检查
        reject_result = reject_agent.check(query)
        if reject_result.action == "REJECT":
            yield {"type": "final_answer", "content": reject_agent.get_reject_message()}
            return

        # ② 构建消息列表（长期记忆上下文 + 短期历史 + 当前问题）
        messages = self._build_messages(user_id, session_id, query)

        full_answer = ""

        # ③ 流式执行，通过 astream_events 逐步推送
        async for event in self._agent.astream_events(
            {"messages": messages},
            version="v2",
        ):
            kind = event.get("event")

            if kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "tool": event["name"],
                    "input": str(event.get("data", {}).get("input", "")),
                }

            elif kind == "on_tool_end":
                yield {
                    "type": "tool_end",
                    "tool": event["name"],
                    "output": str(event.get("data", {}).get("output", ""))[:200],
                }

            elif kind == "on_chat_model_stream":
                token = event.get("data", {}).get("chunk", {})
                content = getattr(token, "content", "")
                if content:
                    full_answer += content
                    yield {"type": "token", "content": content}

        # ④ 保存本轮对话
        if full_answer:
            session_memory_manager.save_turn(user_id, session_id, query, full_answer)
            yield {"type": "done", "content": full_answer}


orchestrator = OrchestratorAgent()

