from typing import List, Dict, Optional, Any
import os
import hashlib
import sqlite3
import time
import json
from ..embedding import get_text_embedder, get_dimension
from taiyi_agent.memory.storage.qdrant_store import QdrantVectorStore