
from typing import Any
from datetime import datetime
import hashlib

from taiyi_agent.memory.rag.models import SourceDocument, TextChunk
from taiyi_agent.context.token_counter import TokenCounter
from taiyi_agent.memory.rag.config import ChunkingConfig

"""
生产版本还应支持：
- Markdown 标题感知
- PDF 页码保留
- parent-child chunk
- token 长度限制
- 代码块和表格不拆分
"""

class DocumentChunker:
    """文档处理器"""
    
    def __init__(
        self,
        config: ChunkingConfig,
        token_counter: TokenCounter | None = None
    ):
        self.chunk_size_tokens = config.chunk_size_tokens
        self.chunk_overlap_tokens = config.chunk_overlap_tokens
        self.separators = config.separators
        self.min_chunk_length = config.min_chunk_length
        self.merge_max_length = config.merge_max_length
        self.token_counter = token_counter or TokenCounter()

        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size 必须大于0")
        if not (0 <= self.chunk_overlap_tokens < self.chunk_size_tokens):
            raise ValueError("chunk_overlap 需要 0 ≤ overlap < chunk_size，一般为 chunk_size 的10%‑20%")

    def split_documents(self, documents: list[SourceDocument]) -> list[TextChunk]:
        """
        批量处理文档
        
        Args:
            documents: 文档列表
            
        Returns:
            所有文档块列表
        """
        all_chunks = []
        for document in documents:
            chunks = self.split_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks

    def split_document(self, document: SourceDocument) -> list[TextChunk]:
        """
        处理文档，分割成块
        
        Args:
            document: 输入文档
            
        Returns:
            文档块列表
        """
        chunks = self._split_text(document.content)
        
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            # 创建块的元数据
            chunk_id = hashlib.sha256(f"{document.doc_id}:{i}:{chunk_content}".encode()).hexdigest()

            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "doc_id": document.doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_uri": document.metadata.get("source_uri"),
                "source_type": document.metadata.get("source_type"),
                "title": document.metadata.get("title"),
                "char_count": len(chunk_content),
                "token_count": self.token_counter.count_text(chunk_content),
                "processed_at": datetime.now().isoformat()
            })
            
            chunk = TextChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                content=chunk_content,
                index=i,
                metadata=chunk_metadata,
            )
            document_chunks.append(chunk)

        print(f"[chunker], chunker成功, doc_id:{document.doc_id}, chunks长度:{len(document_chunks)}")

        return document_chunks

    def _split_text(self, text: str) -> list[str]:
        """
        分割文本为块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        text = text.strip()
        if not text:
            return []

        if self.token_counter.count_text(text) <= self.chunk_size_tokens:
            return [text]
        
        chunks: list[str] = []
        start = 0
        
        while start < len(text):
            remaining = text[start:]

            # 结束条件
            if self.token_counter.count_text(remaining) <= self.chunk_size_tokens:
                chunks.append(remaining)
                break
            
            # 得到不超过 chunk_size_tokens 的字符前缀
            token_window = self.token_counter.truncate_text(
                remaining,
                self.chunk_size_tokens,
            )

            if not token_window:
                raise RuntimeError("无法根据 token 限制生成有效 chunk")

            end = start + len(token_window)

            # 在 token 上限范围内，优先寻找自然分隔符
            split_point = self._find_split_point(text,start,end,)

            if split_point <= start:
                split_point = end

            raw_chunk = text[start:split_point]
            chunk = raw_chunk.strip()

            if chunk:
                chunks.append(chunk)

            # overlap 也按照 token 数计算
            overlap_text = self.token_counter.truncate_text(
                raw_chunk,
                self.chunk_overlap_tokens,
                from_end=True,
            )

            overlap_chars = len(overlap_text)
            next_start = split_point - overlap_chars

            # 防止 overlap 导致循环无法前进
            if next_start <= start:
                next_start = split_point

            start = next_start

        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """
        在指定范围内寻找最佳分割点
        
        Args:
            text: 文本
            start: 开始位置
            end: 结束位置
            
        Returns:
            分割点位置，-1表示未找到
        """
        # 从后往前寻找分隔符
        for separator in self.separators:
            # 在end附近寻找分隔符
            search_start = max(start, end - 100)  # 在最后100个字符中寻找
            
            for i in range(end - len(separator), search_start - 1, -1):
                if text[i:i + len(separator)] == separator:
                    return i + len(separator)
        
        return -1


    def merge_chunks(self, chunks: list[TextChunk], max_tokens: int = 1024) -> list[TextChunk]:
        """
        合并小的文档块
        
        Args:
            chunks: 文档块列表
            max_length: 合并后的最大长度
            
        Returns:
            合并后的文档块列表
        """
        if not chunks:
            return []

        merged_chunks = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            # 检查是否可以合并
            combined_content = (current_chunk.content+ "\n"+ next_chunk.content)

            can_merge = (
                current_chunk.doc_id == next_chunk.doc_id
                and self.token_counter.count_text(combined_content) <= max_tokens
            )
            if can_merge:
                # 合并块
                current_chunk.content += "\n" + next_chunk.content
                current_chunk.metadata["total_chunks"] = current_chunk.metadata.get("total_chunks", 1) + 1
            else:
                # 不能合并，保存当前块
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk

        # 添加最后一个块
        merged_chunks.append(current_chunk)

        return merged_chunks

    def add_chunk_metadata(self, chunks: list[TextChunk], metadata: dict[str, Any]) -> list[TextChunk]:
        """
        为文档块添加额外元数据
        
        Args:
            chunks: 文档块列表
            metadata: 要添加的元数据
            
        Returns:
            更新后的文档块列表
        """
        for chunk in chunks:
            chunk.metadata.update(metadata)
        
        return chunks