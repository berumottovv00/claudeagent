"""
Redis 连接客户端
使用连接池，支持通过环境变量配置连接参数
"""
import os

import redis


class _RedisDBConfig:
    HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
    PORT = int(os.environ.get("REDIS_PORT", "59000"))
    DBID = int(os.environ.get("REDIS_DB", "0"))
    PASSWORD = os.environ.get("REDIS_PASSWORD", "e65K4t8w2")


class RedisClient:
    def __init__(self):
        if not hasattr(RedisClient, "pool"):
            RedisClient._create_pool()
        self._conn = redis.Redis(connection_pool=RedisClient.pool)

    @staticmethod
    def _create_pool():
        RedisClient.pool = redis.ConnectionPool(
            host=_RedisDBConfig.HOST,
            port=_RedisDBConfig.PORT,
            db=_RedisDBConfig.DBID,
            password=_RedisDBConfig.PASSWORD,
            decode_responses=True,
        )

    def set(self, key, value, ex=None):
        return self._conn.set(key, value, ex=ex)

    def get(self, key):
        return self._conn.get(key)

    def lpush(self, name, *values):
        return self._conn.lpush(name, *values)

    def lrange(self, name, start, end):
        return self._conn.lrange(name, start, end)

    def ltrim(self, name, start, end):
        return self._conn.ltrim(name, start, end)

    def expire(self, name, time):
        return self._conn.expire(name, time)

    def delete(self, *names):
        return self._conn.delete(*names)

    def pipeline(self):
        return self._conn.pipeline()