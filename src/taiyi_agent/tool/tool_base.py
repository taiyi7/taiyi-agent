from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel
import asyncio
import inspect

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
        ...

    def run(self, parameters: dict[str, Any]) -> str:
        '''
        同步工具可以覆盖此方法。
        纯异步工具只需要覆盖 arun。
        '''
        raise NotImplementedError(
            f"工具 {self.name} 没有同步执行实现，请调用 arun()"
        )

    async def arun(self, parameters: dict[str, Any]) -> str:
        """
        统一的异步执行入口。
        同步工具放到线程池，避免阻塞事件循环。
        """
        if type(self).run is BaseTool.run:
            raise NotImplementedError(f"工具 {self.name} 必须实现 run() 或 arun()")

        result = await asyncio.to_thread(self.run, parameters)
        if inspect.isawaitable(result):
            result = await result
        return str(result)


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