"""
汽车智能对话系统 —— FastAPI 入口
提供同步对话接口和 SSE 流式对话接口
"""
import json
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import orchestrator
from memory import session_memory_manager

# FastAPI 是现代、高性能的 Python Web 框架，专门用于快速构建 API 接口，是当下 Python 后端开发的主流选择之一。
app = FastAPI(title="汽车智能对话系统", version="1.0.0")


# ----------------------------------------------------------------
# 请求 / 响应模型
# ----------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # 不传则自动生成新 session
    user_id: Optional[str] = None     # 不传则自动生成新 user


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    answer: str


# ----------------------------------------------------------------
# 接口
# ----------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """同步对话接口，等待完整答案后返回"""
    session_id = req.session_id or str(uuid.uuid4())
    user_id = req.user_id or str(uuid.uuid4())
    try:
        answer = orchestrator.run(session_id=session_id, query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{str(e)}")
    return ChatResponse(session_id=session_id, user_id=user_id, answer=answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式对话接口，逐步推送 token 和工具调用状态"""
    session_id = req.session_id or str(uuid.uuid4())
    user_id = req.user_id or str(uuid.uuid4())

    # 定义一个异步生成器函数
    async def event_generator():
        # 先推送 session_id / user_id，供前端保存
        # 一次 yield 就是一段 SSE 消息
        yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id, 'user_id': user_id})}\n\n"
        try:
            # async for 开始监听迭代器，主动等待（非阻塞），不占用 CPU
            async for event in orchestrator.astream(user_id=user_id, session_id=session_id, query=req.query):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，确保实时推送
        },
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """清除指定 session 的对话历史"""
    session_memory_manager.clear(session_id)
    return {"message": f"Session {session_id} 已清除"}


# ----------------------------------------------------------------
# 启动
# ----------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
