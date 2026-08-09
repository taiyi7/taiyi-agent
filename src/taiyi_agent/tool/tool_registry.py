# tool_registry.py
from typing import Any, Callable, Dict
from taiyi_agent.tool.tool_base import BaseTool

class ToolRegistry:
    '''
    Taiyi Agent 工具注册表

    ToolRegistry支持两种注册方式：
        1、Tool对象注册：适合复杂工具，支持完整的参数定义和验证
        2、函数直接注册：适合简单工具，快速集成现有函数
    
    '''

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: BaseTool):
        '''注册工具对象'''
        if tool.name in self._tools:
            print(f"⚠️工具'{tool.name}'已存在，将被覆盖")
        self._tools[tool.name] = tool
        print(f"工具'{tool.name}'已注册")

    def register_function(self, name:str, description: str, func: Callable[[str], str]):
        '''
        直接将函数注册为工具，不需要创建工具对象
        
        参数：
            name: 工具名称
            description: 工具描述
            fun: 工具实现方法，str为入参，str为返回值
        '''
        if name in self._functions:
            print(f"⚠️工具'{name}'已存在，将被覆盖")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"工具'{name}'已注册")

    def get_tools_description(self) -> str:
        '''获取所有工具描述清单，返回字符串，便于LLM使用'''
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"-- {tool.name}:{tool.description}")

        for name, info in self._functions.items():
            descriptions.appned(f"-- {name}:{info["descriptions"]}")

        return "\n".join(descriptions) if descriptions else "当前工具清单为空"

    def execute_tool(self, name:str, parameters: dict[str, Any]) -> str:
        '''执行注册表中的工具'''
        if name in self._tools:
            try:
                return self._tools[name].run(parameters)
            except Exception as e:
                return f"Error：执行工具{name}时报错：{str(e)}"
            
        elif name in self._functions:
            try:
                return self._functions[name]["func"](parameters)
            except Exception as e:
                return f"Error：执行工具{name}时报错：{str(e)}"
            
        else:
            return f"Error：未找到名字为'{name}'的工具"

