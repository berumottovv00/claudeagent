"""
OrchestratorAgent.run() 集成测试
使用真实 LLM、短期记忆、长期记忆（Redis）和拒识服务。
运行前确保 .env 已配置。
"""
import pytest

from agents.orchestrator_agent import OrchestratorAgent
from memory import session_memory_manager

USER_ID = "test-user-001"
SESSION_ID = "test-session-orchestrator"


@pytest.fixture(scope="module")
def agent():
    return OrchestratorAgent()


@pytest.fixture(autouse=True)
def clean_session():
    """每个用例前后清除短期记忆，保证隔离"""
    session_memory_manager.clear(USER_ID, SESSION_ID)
    yield
    session_memory_manager.clear(USER_ID, SESSION_ID)


# ----------------------------------------------------------------
# 1. 基本问答
# ----------------------------------------------------------------

def test_run_basic(agent):
    """正常问题应返回非空答案"""
    answer = agent.run(USER_ID, SESSION_ID, "发动机异响可能是什么原因？")
    assert isinstance(answer, str) and answer


# ----------------------------------------------------------------
# 2. 拒识（REJECT_URL 配置后生效，未配置时跳过）
# ----------------------------------------------------------------

def test_run_rejected(agent):
    """完全越界问题应返回拒识消息"""
    import os
    if not os.environ.get("REJECT_URL"):
        pytest.skip("REJECT_URL 未配置，跳过拒识测试")

    from agents.reject_agent import REJECT_MESSAGE
    answer = agent.run(USER_ID, SESSION_ID, "帮我写一篇高考作文")
    assert answer == REJECT_MESSAGE


# ----------------------------------------------------------------
# 3. 答案写入短期记忆
# ----------------------------------------------------------------

def test_run_saves_short_term_memory(agent):
    """run 后短期记忆应保存本轮对话"""
    query = "空调不制冷是什么原因？"
    answer = agent.run(USER_ID, SESSION_ID, query)

    history = session_memory_manager.get_history_messages(USER_ID, SESSION_ID)
    assert len(history) == 2
    assert history[0].content == query
    assert history[1].content == answer


# ----------------------------------------------------------------
# 4. 多轮对话 —— 上下文连贯
# ----------------------------------------------------------------

def test_run_multi_turn(agent):
    """第二轮能理解第一轮的上下文"""
    agent.run(USER_ID, SESSION_ID, "我的车空调不制冷，可能是什么原因？")
    answer2 = agent.run(USER_ID, SESSION_ID, "刚才说的第一个原因是什么？")

    # 第二轮应能引用第一轮内容
    assert isinstance(answer2, str) and answer2

    history = session_memory_manager.get_history_messages(USER_ID, SESSION_ID)
    assert len(history) == 4  # 两轮，每轮 human + ai


# ----------------------------------------------------------------
# 5. 天气工具调用
# ----------------------------------------------------------------

def test_run_weather_tool(agent):
    """涉及天气的问题应触发 get_weather 工具"""
    answer = agent.run(USER_ID, SESSION_ID, "今天上海天气怎么样？")
    assert isinstance(answer, str) and answer


# ----------------------------------------------------------------
# 6. close_session —— 摘要写入长期记忆后短期清除
# ----------------------------------------------------------------

def test_close_session(agent):
    """close_session 后短期记忆应被清除"""
    agent.run(USER_ID, SESSION_ID, "轮胎气压多少合适？")
    assert session_memory_manager.get_history_messages(USER_ID, SESSION_ID)

    agent.close_session(USER_ID, SESSION_ID)

    # 短期记忆已清除
    assert session_memory_manager.get_history_messages(USER_ID, SESSION_ID) == []