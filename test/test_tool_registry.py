from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.tool.datetime_tool.get_current_time_tool import GetCurrentTimeTool

tool_registry = ToolRegistry()
time_tool = GetCurrentTimeTool()
print(time_tool.name, time_tool.description)
tool_registry.register_tool(time_tool)
print("【已注册工具清单】：\n",tool_registry.get_tools_description())
print(tool_registry.execute_tool("datetime", {"timezone": "Asia/Shanghai"}))


