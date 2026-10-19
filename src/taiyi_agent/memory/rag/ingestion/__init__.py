# ### Ingestion Pipeline（数据接入流水线）典型步骤：

# 1. 文件扫描，遍历目录
# 2. **Document loading：加载文件 → `get_file_content()`**（txt/md/json/csv 提取出字符串）
# 3. 文本清洗
# 4. Text Chunking 文本切分
# 5. 可选：实体 / 关系抽取（GraphRAG 的 ingestion 会多这一步）
# 6. Embedding 向量化
# 7. 写入向量数据库 / 图谱存储