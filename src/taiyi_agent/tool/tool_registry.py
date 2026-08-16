# tool_registry.py
from typing import Any, Callable
from taiyi_agent.tool.tool_base import BaseTool, ToolParameter
from taiyi_agent.tool.function_tool.function_tool import FunctionTool

class ToolRegistry:
    '''
    Taiyi Agent 工具注册表

    ToolRegistry支持两种注册方式：
        1、Tool对象注册：适合复杂工具，支持完整的参数定义和验证
        2、函数直接注册：适合简单工具，快速集成现有函数
    
    '''

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        '''注册工具对象'''
        if tool.name in self._tools:
            print(f"⚠️工具'{tool.name}'已存在，将被覆盖")
        self._tools[tool.name] = tool
        print(f"工具'{tool.name}'已注册")

    def register_function(
        self,
        name:str,
        description: str,
        func: Callable[..., str],
        parameters: list[ToolParameter]
    ):
        '''
        直接将函数注册为工具对象
        
        参数：
            name: 工具名称
            description: 工具描述
            fun: 工具实现方法
            parameters: 工具方法的入参格式为：list[ToolParameter]
        '''
        return self.register_tool(
            FunctionTool(
                name=name,
                description=description,
                func=func,
                parameters=parameters
            )
        )

    def get_tools_description(self) -> str:
        '''获取所有工具描述清单，返回字符串，便于LLM使用'''
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"-- {tool.name}:{tool.description}")

        return "\n".join(descriptions) if descriptions else "当前工具列表为空"

    def build_tools_prompt(self) -> str:
        """针对大模型没有原生工具调用的场景，构造系统提示词，来处理工具调用"""
        sections = []

        for tool in self._tools.values():
            parameter_lines = []
            for parameter in tool.get_parameters():
                required = "必填" if parameter.required else "可选"
                default = (
                    f"，默认值：{parameter.default}"
                    if parameter.default is not None
                    else ""
                )
                parameter_lines.append(
                    f"- {parameter.name}: {parameter.type}，{required}，"
                    f"{parameter.description}{default}"
                )

            sections.append(
                f"工具名：{tool.name}\n"
                f"描述：{tool.description}\n"
                f"参数：\n" + ("\n".join(parameter_lines) or "- 无")
            )

        if not sections:
            return "当前没有可调用工具。"

        return (
            "你可以在必要时调用工具。可用工具如下：\n\n"
            + "\n\n".join(sections)
            + "\n\n需要调用工具时，只能输出一个 JSON 对象，不能包含其他文字：\n"
            '{"type":"tool_call","name":"工具名","arguments":{"参数名":"参数值"}}\n'
            "不需要调用工具时，直接正常回答用户。"
        )

    def execute_tool(self, name:str, parameters: dict[str, Any]) -> str:
        '''执行注册表中的工具'''
        if name in self._tools:
            try:
                return self._tools[name].run(parameters)
            except Exception as e:
                return f"Error：执行工具{name}时报错：{str(e)}"
        else:
            return f"Error：未找到名字为'{name}'的工具"

    def get_openai_tools(self) -> list[dict[str, Any]]:
        '''将基类对象工具的参数转化成 OpenAI 的tools格式'''
        tools_list = []

        for tool in self._tools.values():
            tools_list.append(tool.to_openai_schema())

        return tools_list

