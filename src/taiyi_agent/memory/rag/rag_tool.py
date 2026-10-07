import os
from typing import Any

from taiyi_agent.tool.tool_base import BaseTool
from taiyi_agent.core.llm import TaiyiAgentLLM

class RAGTool(BaseTool):
    """RAG工具
    
    提供完整的 RAG 能力：
    - 添加多格式文档（PDF、Office、图片、音频等）
    - 智能检索与召回
    - LLM 增强问答
    - 知识库管理

    完整处理流程：
    -- 任意格式文档 → MarkItDown转换 → Markdown文本 → 智能分块 → 向量化 → 存储检索

    """
    
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default"
    ):
        # 初始化RAG管道
        self._pipelines: dict[str, dict[str, Any]] = {}
        self.llm = TaiyiAgentLLM()
        
        # 创建默认管道
        default_pipeline = create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            rag_namespace=self.rag_namespace
        )
        self._pipelines[self.rag_namespace] = default_pipeline


    def _convert_to_markdown(path: str) -> str:
        """
        核心功能：将任意格式文档转换为Markdown文本
        
        支持格式：
        - 文档：PDF、Word、Excel、PowerPoint
        - 图像：JPG、PNG、GIF（通过OCR）
        - 音频：MP3、WAV、M4A（通过转录）
        - 文本：TXT、CSV、JSON、XML、HTML
        - 代码：Python、JavaScript、Java等
        """
        if not os.path.exists(path):
            return ""
        
        # 对PDF文件使用增强处理
        ext = (os.path.splitext(path)[1] or '').lower()
        if ext == '.pdf':
            return _enhanced_pdf_processing(path)
        
        # 其他格式使用MarkItDown统一转换
        md_instance = _get_markitdown_instance()
        if md_instance is None:
            return _fallback_text_reader(path)
        
        try:
            result = md_instance.convert(path)
            markdown_text = getattr(result, "text_content", None)
            if isinstance(markdown_text, str) and markdown_text.strip():
                print(f"[RAG] MarkItDown转换成功: {path} -> {len(markdown_text)} chars Markdown")
                return markdown_text
            return ""
        except Exception as e:
            print(f"[WARNING] MarkItDown转换失败 {path}: {e}")
            return _fallback_text_reader(path)
