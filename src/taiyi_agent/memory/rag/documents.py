"""RAG文档处理模块"""

from typing import Any, Optional
from pydantic import BaseModel
from datetime import datetime
import hashlib

class Document(BaseModel):
    """文档类"""
    content: str
    metadata: dict[str, Any]
    doc_id: Optional[str] = None

    # 在字段校验全部完成之后执行
    def model_post_init(self, __context: Any) -> None:
        if self.doc_id is None:
            # 基于内容生成 sha256 ID
            self.doc_id = hashlib.sha256(self.content.encode("utf-8")).hexdigest()

class DocumentChunk(BaseModel):
    """文档块类"""
    content: str
    metadata: dict[str, Any]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0
    
    def model_post_init(self, __context: Any) -> None:
        if self.chunk_id is None:
            # 基于文档ID和块索引生成ID
            chunk_content = f"{self.doc_id}_{self.chunk_index}_{self.content}"
            self.chunk_id = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap          # 一般设置为 chunk_size 的 10%～20%
        self.separators = separators or ["\n\n", "\n", "。", ".", " "]

    def process_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """
        批量处理文档
        
        Args:
            documents: 文档列表
            
        Returns:
            所有文档块列表
        """
        all_chunks = []
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks

    def process_document(self, document: Document) -> list[DocumentChunk]:
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
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "doc_id": document.doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "processed_at": datetime.now().isoformat()
            })
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                doc_id=document.doc_id,
                chunk_index=i
            )
            document_chunks.append(chunk)
        
        return document_chunks

    def _split_text(self, text: str) -> list[str]:
        """
        分割文本为块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 确定块的结束位置
            end = start + self.chunk_size
            
            if end >= len(text):
                # 最后一块
                chunks.append(text[start:])
                break
            
            # 寻找合适的分割点
            split_point = self._find_split_point(text, start, end)
            
            if split_point == -1:
                # 没找到合适的分割点，强制分割
                split_point = end
            
            chunks.append(text[start:split_point])
            
            # 计算下一块的开始位置（考虑重叠）
            start = max(start + 1, split_point - self.chunk_overlap)
        
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


    def merge_chunks(self, chunks: list[DocumentChunk], max_length: int = 2000) -> list[DocumentChunk]:
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
            combined_length = len(current_chunk.content) + len(next_chunk.content)
            
            if (combined_length <= max_length and 
                current_chunk.doc_id == next_chunk.doc_id):
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

    def filter_chunks(self, chunks: list[DocumentChunk], min_length: int = 50) -> list[DocumentChunk]:
        """
        过滤太短的文档块
        
        Args:
            chunks: 文档块列表
            min_length: 最小长度
            
        Returns:
            过滤后的文档块列表
        """
        return [chunk for chunk in chunks if len(chunk.content.strip()) >= min_length]

    def add_chunk_metadata(self, chunks: list[DocumentChunk], metadata: dict[str, Any]) -> list[DocumentChunk]:
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


def load_text_file(file_path: str, encoding: str = "utf-8") -> Document:
    """
    加载文本文件为 Document
    
    Args:
        file_path: 文件路径
        encoding: 文件编码
        
    Returns:
        Document 对象
    """
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    
    metadata = {
        "source": file_path,
        "type": "text_file",
        "loaded_at": datetime.now().isoformat()
    }
    
    return Document(content=content, metadata=metadata)

def create_document(content: str, **metadata) -> Document:
    """
    创建文档的便捷函数
    
    Args:
        content: 文档内容
        **metadata: 元数据
        
    Returns:
        文档对象
    """
    return Document(content=content, metadata=metadata)