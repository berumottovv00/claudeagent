import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Literal

import requests

logger = logging.getLogger(__name__)

THRESHOLD = 0.5
REJECT_URL = os.environ.get("REJECT_URL", "")


def _request_reject(query: str, trace_id: str) -> str:
    """调用拒识微服务，返回判断结果。"是" 表示应拒识，"否" 表示放行。"""
    start_time = time.time()
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "query": query,
        "thres": THRESHOLD,
        "trace_id": trace_id
    })
    try:
        response = requests.post(REJECT_URL, headers=headers, data=payload, timeout=5)
        res = response.json()
        text = res["data"]
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"拒识模型的输出：{text}，耗时：{elapsed:.2f}ms")
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"call reject failed: {e}，耗时：{elapsed:.2f}ms")
        text = "是"
    return text


REJECT_MESSAGE = "您好，我是汽车智能助手，专注解答车辆使用、保养、故障等问题，暂不支持该类请求。"


@dataclass
class RejectResult:
    action: Literal["PASS", "REJECT"]


class RejectAgent:
    def check(self, query: str) -> RejectResult:
        trace_id = str(uuid.uuid4())
        text = _request_reject(query, trace_id)

        if text.strip() == "是":
            return RejectResult(action="REJECT")
        return RejectResult(action="PASS")

    def get_reject_message(self) -> str:
        return REJECT_MESSAGE


reject_agent = RejectAgent()