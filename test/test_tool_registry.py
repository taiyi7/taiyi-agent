from taiyi_agent.tool.tool_registry import ToolRegistry
from taiyi_agent.tool.tool_base import ToolParameter
from taiyi_agent.tool.datetime_tool.get_current_time_tool import GetCurrentTimeTool

tool_registry = ToolRegistry()
time_tool = GetCurrentTimeTool()
print(time_tool.name, time_tool.description)
tool_registry.register_tool(time_tool)
print("【已注册工具清单】：\n",tool_registry.get_tools_description())
print(tool_registry.execute_tool("datetime", {"timezone":"utc"}))

def get_weather(city: str, area: str):
    return f"今天{city}{area}的天气为晴天"

tool_registry.register_function(
    name="get_weather",
    description="查询指定城市的天气",
    func=get_weather,
    parameters=[
        ToolParameter(
            name="city",
            type="string",
            description="城市名称",
            required=True,
        ),
        ToolParameter(
            name="area",
            type="string",
            description="地区名称",
            required=True,
        )
    ],
)

print("【已注册工具清单】：\n", tool_registry.get_tools_description())
print(tool_registry.execute_tool("get_weather", {"city":"广州", "area": "白云"}))
