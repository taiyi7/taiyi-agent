from dotenv import load_dotenv
from openai import OpenAI
from taiyi_agent.memory.rag.config import EmbeddingConfig
load_dotenv()

class RAGEmbedder:
    """
    EmbedderPort协议实现，基于OpenAI embedding接口
    可替换为BGE、Jina等本地模型，外部调用完全不用改动pipeline代码
    """
    def __init__(self, config: EmbeddingConfig) -> None:
        if not config.url:
            raise ValueError("Reranker URL 未配置")
        if not config.model:
            raise ValueError("Embedding model 未配置")
        if not config.api_key:
            raise ValueError("Embedding API key 未配置")

        self.model = config.model
        self.batch_size = config.batch_size

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.url,
            timeout=config.timeout,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文档向量化，自动做简单去空白过滤"""
        # 过滤空字符串，避免接口报错
        valid_texts = [t.strip() for t in texts if t.strip()]
        if not valid_texts:
            return []

        if any(not isinstance(text, str) for text in valid_texts):
            raise TypeError("texts 中的每个元素都必须是字符串")

        embeddings: list[list[float]] = []

        for start in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[start:start + self.batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            # 按 index 排序，确保向量顺序与输入文本一致
            data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            if len(data) != len(batch):
                raise RuntimeError(
                    "Embedding 服务返回的向量数量与输入文本数量不一致"
                )

            embeddings.extend(
                [list(item.embedding) for item in data]
            )

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """query向量化，部分模型这里需要加特殊prompt"""
        text = text.strip()
        if not text:
            raise ValueError("query文本不能为空")

        resp = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return resp.data[0].embedding
