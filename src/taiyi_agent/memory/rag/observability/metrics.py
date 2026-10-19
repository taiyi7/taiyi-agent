from collections import defaultdict
import time

"""
生产环境可替换为 Prometheus、OpenTelemetry 或 StatsD。
建议监控指标：
离线：
- 文件处理成功率
- 文件解析失败数
- 平均分块数
- embedding 耗时
- 向量写入耗时
- 索引积压任务数

在线：
- 查询 QPS
- P50/P95/P99 延迟
- dense 检索耗时
- sparse 检索耗时
- rerank 耗时
- 空召回率
- 平均召回数量
- rerank 降级次数
- 查询错误率

质量：
- Recall@K
- MRR
- NDCG
- Citation 命中率
- 用户反馈分数
"""

class RAGMetrics:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timings = defaultdict(list)

    def inc(self, name: str, value: int = 1):
        self.counters[name] += value

    def timer(self, name: str):
        return _Timer(self, name)

    def snapshot(self):
        return {
            "counters": dict(self.counters),
            "timings": {
                key: {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                }
                for key, values in self.timings.items()
                if values
            },
        }


class _Timer:
    def __init__(self, metrics, name):
        self.metrics = metrics
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *_):
        elapsed = (time.perf_counter() - self.start) * 1000
        self.metrics.timings[self.name].append(elapsed)