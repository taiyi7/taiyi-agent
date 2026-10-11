import os
import httpx
from typing import Any
from dotenv import load_dotenv
import json

load_dotenv()

QWEATHER_HOST = "mb4qbn44kk.re.qweatherapi.com"
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")

from taiyi_agent.tool.tool_base import BaseTool, ToolParameter

class WeatherTool(BaseTool):
    name="天气信息搜索API"
    description="使用qweather API进行天气搜索"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="city_name",
                type="string",
                description="需要查询天气的城市名称，例如：广州、北京",
                required=True,
            ),
            ToolParameter(
                name="time_type",
                type="string",
                description="需要查询天气的未来时间,支持如下入参(\"now\"、\"3d\"、\"7d\")",
                required=False,
                default="now",
            )
        ]

    async def arun(self, parameters: dict[str, Any]) -> str:
        city_name = parameters.get("city_name", "广州")
        time_type = parameters.get("time_type", "now")
        return await qweather_query(city_name, time_type)


async def qweather_city_lookup(city_name: str):
    """城市名称查询，获取locationID"""
    city_name = city_name.strip()
    if not city_name:
        return {"code": "400", "location": []}

    async with httpx.AsyncClient(timeout=10) as client:
        url = f"https://{QWEATHER_HOST}/geo/v2/city/lookup"
        resp = await client.get(
            url,
            params={
                "location": city_name,
                "key": QWEATHER_API_KEY
            }
        )

        # 打印原始返回内容，排查问题
        # print("status_code:", resp.status_code)
        # print("full request url:", str(resp.url))
        # print("raw response text:", repr(resp.text))

        # 校验状态码 + 判断是否为空
        if resp.status_code != 200:
            raise Exception(f"城市查询请求失败，状态码：{resp.status_code}, 响应：{resp.text}")
        if not resp.text.strip():
            raise Exception("接口返回空内容")
        
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise Exception(f"返回不是JSON，原始内容:{resp.text}") from exc


async def qweather_query(city_name: str, time_type: str) ->str:
    """
    和风天气统一查询接口
    :param city_name: 城市中文名，如广州、北京
    :param time_type: 查询时间类型，可选值：now(实时)、3d(7天预报)、7d(7天预报)
    :return: 统一返回str
    """
    # 前置参数校验（和tavily空query逻辑对齐）
    city = city_name.strip()
    time_type = time_type.strip().lower()
    valid_types = {"now", "1d", "3d", "7d"}

    if not city:
        return {
            "success": False,
            "msg": "城市名称不能为空，请提供有效的城市名称",
            "data": None
        }
    if time_type not in valid_types:
        return {
            "success": False,
            "msg": f"time_type仅支持 {valid_types}，当前输入：{time_type}",
            "data": None
        }

    # 1. 城市转locationId
    geo_result = await qweather_city_lookup(city)
    if geo_result.get("code") != "200" or len(geo_result.get("location", [])) == 0:
        return {
            "success": False,
            "msg": f"找不到城市：{city_name}",
            "data": None
        }
    location_id = geo_result["location"][0]["id"]
    city_display_name = geo_result["location"][0]["name"]

    print(f"🔍 正在执行{city_display_name}天气搜索")
    # 2. 根据time_type选择接口
    async with httpx.AsyncClient(timeout=10) as client:
        url = ""
        if time_type == "now":
            url = f"https://{QWEATHER_HOST}/v7/weather/now"
        elif time_type == "3d":
            url = f"https://{QWEATHER_HOST}/v7/weather/3d"
        elif time_type == "7d":
            url = f"https://{QWEATHER_HOST}/v7/weather/7d"
        resp = await client.get(
            url,
            params={
                "location": location_id,
                "key": QWEATHER_API_KEY
            }
        )
        data = resp.json()
        if data.get("code") != "200":
            return {
                "success": False,
                "msg": f"天气接口请求失败，code={data.get('code')}",
                "data": None
            }

        weather_text = ""
        if time_type == "now":
            now = data["now"]
            weather_text = (
                f"{city_display_name}天气预报："
                f"实时天气：{now['text']}，温度 {now['temp']}℃，体感温度 {now['feelsLike']}℃，"
                f"湿度 {now['humidity']}%，风向 {now['windDir']}，风力 {now['windScale']}级，"
                f"气压 {now['pressure']}hPa，能见度 {now['vis']}km。"
            )
        else:
            # 1d /3d /7d 预报
            daily_list = data["daily"]
            parts = []
            for day in daily_list:
                day_str = (
                    f"{city_display_name}天气预报："
                    f"日期{day['fxDate']}：白天{day['textDay']}，夜间{day['textNight']}，"
                    f"最高{day['tempMax']}℃，最低{day['tempMin']}℃"
                )
                parts.append(day_str)
            weather_text = "\n".join(parts)

        return weather_text

# weather = WeatherTool()
# import asyncio
# ret = asyncio.run(weather.arun({"city_name":"广州", "time_type": "now"}))
# print(ret)