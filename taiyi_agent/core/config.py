import os
from typing import Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    '''
    TaiyiAgent 的配置类
    Config 类的职责是将代码中硬编码配置参数集中起来，并支持从环境变量中读取。
    '''
    # LLM配置
    default_model: str = "qwen3.7-max"
    temperature: float = 0.7

    # 系统配置
    debug: bool = False
    log_level:str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()

# cfg = Config.from_env()
# print(cfg.to_dict())
# print(cfg.default_model)