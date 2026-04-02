"""
MCP Client 示例 —— 大模型通过 MCP 调用工具的完整链路

完整交互流程：
  ┌─────────┐   1. initialize        ┌─────────┐
  │  Client │ ──────────────────────► │  Server │
  │  (LLM)  │ ◄────────────────────── │         │
  └─────────┘   2. 返回 capabilities  └─────────┘
       │
       │  3. tools/list（发现工具）
       │  4. 用户提问 → LLM 判断调用哪个工具
       │  5. tools/call（携带参数）
       │  6. Server 执行并返回结果
       │  7. LLM 将结果融入最终回答
       ▼

依赖：
  pip install mcp anthropic
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ----------------------------------------------------------------
# 配置
# ----------------------------------------------------------------
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
MODEL = "claude-opus-4-6"


# ----------------------------------------------------------------
# 核心：大模型 + MCP 工具调用循环
# ----------------------------------------------------------------
async def run(user_query: str):
    # 1. 启动 MCP Server 子进程，建立 stdio 通信
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # 2. 握手：Client 向 Server 发送 initialize 请求
            #    Server 返回自己的 protocolVersion 和 capabilities
            await session.initialize()
            print("[MCP] 握手完成")

            # 3. 工具发现：tools/list
            #    Server 返回所有可用工具的 name / description / inputSchema
            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"[MCP] 发现工具：{[t.name for t in tools]}\n")

            # 将 MCP 工具描述转换为 Anthropic API 的 tools 格式
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools
            ]

            # 4. 大模型推理：发送用户问题 + 工具描述，让 LLM 决定调用哪个工具
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
            messages = [{"role": "user", "content": user_query}]

            print(f"用户：{user_query}")
            print("-" * 50)

            # 工具调用循环（LLM 可能连续调用多个工具）
            while True:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    tools=anthropic_tools,
                    messages=messages,
                )

                # 5. 判断 LLM 是否要调用工具
                if response.stop_reason == "tool_use":
                    # 将 LLM 的回复追加到消息历史
                    messages.append({"role": "assistant", "content": response.content})

                    # 处理本轮所有 tool_use 块
                    tool_results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        print(f"[LLM] 决定调用工具：{block.name}")
                        print(f"[LLM] 参数：{json.dumps(block.input, ensure_ascii=False)}")

                        # 6. tools/call：通过 MCP 协议调用 Server 上的工具
                        #    JSON-RPC 请求示例：
                        #    {
                        #      "jsonrpc": "2.0",
                        #      "method": "tools/call",
                        #      "params": {"name": "search_nearby_poi", "arguments": {...}}
                        #    }
                        result = await session.call_tool(block.name, block.input)
                        tool_output = result.content[0].text if result.content else ""

                        print(f"[Server] 返回结果：{tool_output[:200]}...")
                        print()

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_output,
                        })

                    # 7. 将工具结果送回 LLM，继续推理
                    messages.append({"role": "user", "content": tool_results})

                else:
                    # LLM 不再调用工具，输出最终回答
                    final = "".join(
                        block.text for block in response.content
                        if hasattr(block, "text")
                    )
                    print(f"AI：{final}")
                    break


# ----------------------------------------------------------------
# 入口
# ----------------------------------------------------------------
if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "帮我搜索电子科技大学附近的餐厅"
    asyncio.run(run(query))