"""
main.py 接口测试
使用 httpx.TestClient 在进程内直接调用，无需启动服务器。
"""
import json

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app, raise_server_exceptions=False)

SESSION_ID = "test-session-001"
USER_ID = "test-user-001"
QUERY = "我的车发动机异响怎么办？"


# ----------------------------------------------------------------
# 1. 健康检查
# ----------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ----------------------------------------------------------------
# 2. 同步对话 —— 不传 session_id / user_id，由服务端自动生成
# ----------------------------------------------------------------

def test_chat_auto_ids():
    resp = client.post("/chat", json={"query": QUERY})
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body and body["session_id"]
    assert "user_id" in body and body["user_id"]
    assert "answer" in body and body["answer"]


# ----------------------------------------------------------------
# 3. 同步对话 —— 传入固定 session_id / user_id（多轮复用）
# ----------------------------------------------------------------

def test_chat_with_ids():
    payload = {"query": QUERY, "session_id": SESSION_ID, "user_id": USER_ID}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == SESSION_ID
    assert body["user_id"] == USER_ID
    assert body["answer"]


def test_chat_multi_turn():
    """同一 session 连续两轮，验证上下文保持"""
    payload1 = {"query": "我的车发动机异响怎么办？", "session_id": SESSION_ID, "user_id": USER_ID}
    resp1 = client.post("/chat", json=payload1)
    assert resp1.status_code == 200

    payload2 = {"query": "刚才说的第一步是什么？", "session_id": SESSION_ID, "user_id": USER_ID}
    resp2 = client.post("/chat", json=payload2)
    assert resp2.status_code == 200
    assert resp2.json()["answer"]


# ----------------------------------------------------------------
# 4. SSE 流式对话
# ----------------------------------------------------------------

def test_chat_stream():
    """解析 SSE 各帧，验证首帧包含 session_id / user_id，最终有 final_answer"""
    payload = {"query": "今天上海天气怎么样", "session_id": SESSION_ID, "user_id": USER_ID}

    with client.stream("POST", "/chat/stream", json=payload) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))

    assert events, "未收到任何 SSE 事件"

    # 首帧
    first = events[0]
    assert first["type"] == "session_id"
    assert first["session_id"] == SESSION_ID
    assert first["user_id"] == USER_ID

    # 至少存在一个 final_answer 帧
    types = [e["type"] for e in events]
    assert "final_answer" in types


def test_chat_stream_auto_ids():
    """不传 id 时，首帧应包含服务端生成的 session_id 和 user_id"""
    with client.stream("POST", "/chat/stream", json={"query": QUERY}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                first = json.loads(line[len("data:"):].strip())
                assert first["session_id"]
                assert first["user_id"]
                break


# ----------------------------------------------------------------
# 5. 清除会话
# ----------------------------------------------------------------

def test_clear_session():
    resp = client.delete(f"/session/{USER_ID}/{SESSION_ID}")
    assert resp.status_code == 200
    assert SESSION_ID in resp.json()["message"]