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
        """
        从环境变量创建配置
        方法中这些参数优先从环境变量中提取
        环境变量中没有的，使用get默认值
        方法中没有列出的，使用参数定义默认值
        """
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "1.999")),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()    # self.dict()后续版本废弃
