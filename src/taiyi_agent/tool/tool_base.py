from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class ToolParameter(BaseModel):
    '''工具方法中一个参数的属性定义'''
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class BaseTool(ABC):
    '''所有工具的基类'''

    # 工具定义好后，名字和描述基本不会变化，采用固定数据
    name: str
    description: str

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        '''获取工具参数'''
        pass

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        '''工具执行'''
        pass

    def to_openai_schema(self) -> dict[str, Any]:
        '''
        获取所有工具描述清单, 返回 OpenAI function calling 格式
        生成的格式可以直接用于原生的OpenAI SDK的工具调用
        '''
        parameters = self.get_parameters()

        # 构造properties和required
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param in parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description
            }

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if param.default is not None:
                prop["description"] = f"{param.description}(默认：{param.default})"

            # 如果是数字类型，添加items定义
            if param.type == "array":
                prop["items"] = {"type": "string"}

            properties[param.name] = prop

            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name" : self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

# {
#   "type": "object",      # JSON Schema 基础类型，代表：这一组参数是一个 JSON 对象（字典）
#   "properties": {
#     "city": {"type": "string", "description": "城市名称"},
#     "unit": {"type": "string", "description": "温度单位"}
#   },
#   "required": ["city"]   #这个参数是否是必须提供的
# }
    