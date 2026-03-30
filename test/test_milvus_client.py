"""
MilvusClient 集成测试
直接连接真实 Milvus（默认 127.0.0.1:19530）。
Milvus 不可用时自动跳过所有用例。
"""
import unittest

from memory.milvus_client import MilvusClient, EMBEDDING_DIM

USER_ID = "test-user-milvus"
EMBEDDING = [0.1] * EMBEDDING_DIM


def setUpModule():
    global _client
    _client = MilvusClient()


class TestMilvusInsert(unittest.TestCase):
    def setUp(self):
        if not _client._available:
            self.skipTest("Milvus 不可用，跳过")

    def test_insert_single(self):
        """插入一条数据不抛异常。"""
        _client.insert(USER_ID, "用户喜欢SUV，关注油耗", EMBEDDING)

    def test_insert_multiple(self):
        """连续插入多条不报错。"""
        for i in range(3):
            _client.insert(USER_ID, f"第{i}次会话摘要", EMBEDDING)


class TestMilvusSearch(unittest.TestCase):
    def setUp(self):
        if not _client._available:
            self.skipTest("Milvus 不可用，跳过")
        # 确保有数据可召回
        _client.insert(USER_ID, "用户关注发动机异响问题", EMBEDDING)
        _client._collection.flush()

    def test_search_returns_list(self):
        """search 返回列表，长度不超过 TOP_K。"""
        from memory.milvus_client import TOP_K
        results = _client.search(USER_ID, EMBEDDING)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), TOP_K)

    def test_search_results_are_strings(self):
        """返回的每条摘要都是字符串。"""
        results = _client.search(USER_ID, EMBEDDING)
        for item in results:
            self.assertIsInstance(item, str)

    def test_search_unknown_user_returns_empty(self):
        """不存在的 user_id 应返回空列表。"""
        results = _client.search("no-such-user-xyz", EMBEDDING)
        self.assertEqual(results, [])


class TestMilvusDegradation(unittest.TestCase):
    def test_insert_when_unavailable(self):
        """_available=False 时 insert 静默跳过，不抛异常。"""
        client = MilvusClient()
        client._available = False
        client.insert(USER_ID, "摘要", EMBEDDING)  # 不应抛出

    def test_search_when_unavailable(self):
        """_available=False 时 search 返回空列表。"""
        client = MilvusClient()
        client._available = False
        result = client.search(USER_ID, EMBEDDING)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
