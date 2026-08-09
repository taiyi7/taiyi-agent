from typing import Any
from taiyi_agent.tool.tool_base import BaseTool, ToolParameter
from datetime import datetime
from zoneinfo import ZoneInfo

# 硬编码实现、继承 Tool 的正式工具类 → name/description 使用类变量
# 运行时动态创建、匿名包装函数 → name/description 使用实例属性
class GetCurrentTimeTool(BaseTool):
    name = "datetime"
    description = (
        "获取当前时间、转换时区，或进行日期时间计算。"
        "时区使用 IANA 名称，例如 Asia/Shanghai、UTC。"
    )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="timezone",
                type="string",
                description="目标时区，示例：Asia/Shanghai、UTC、America/New_York。不传默认Asia/Shanghai（UTC+8）",
                required=False,
                default="Asia/Shanghai"
            )
        ]
    
    def run(self, parameters: dict[str, Any]) -> str:
        '''执行工具获取当前时间'''
        timezone = parameters.get("timezone", "Asia/Shanghai")

        if not isinstance(timezone, str):
            return "Error: timezone 必须是字符串"
        
        try:
            tz = ZoneInfo(timezone)
        except Exception:
            # 时区非法时改为为东八区
            tz = ZoneInfo("Asia/Shanghai")
            timezone = "Asia/Shanghai"

        utc_now = datetime.now(ZoneInfo("UTC"))
        local_now = utc_now.astimezone(tz)

        return (
            f"时区：{timezone}\n"
            f"本地时间(ISO): {local_now.isoformat(timespec='seconds')}\n"
            f"UTC时间(ISO): {utc_now.isoformat(timespec='seconds')}\n"
        )
        
