import json
import os
from typing import Any, Iterator, Optional
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from taiyi_agent.core.exceptions import LLMException
from taiyi_agent.core.tool_call import LLMResponse, ToolCall
import asyncio
from collections.abc import AsyncIterator


load_dotenv()
# 后续演进：
# 1、多模态数据类型及处理

class TaiyiAgentLLM:
    '''
    自开发TaiyiAgent的大模型调用客户端
    可以调用兼容OpenAI的大模型接口
    默认使用流式输出，同时支持非流式输出
    '''
    def __init__(
        self, 
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model_id: Optional[str] = None,
        temperature: float=0.7,
        timeout: Optional[int] = None
    ):
        '''
        初始化LLM客户端，优先使用传入参数，如果没有则从环境中加载
        参数：
            llm_api_key：  大模型API密钥，如未提供则使用.env文件中的"LLM_API_KEY"
            llm_base_url： 大模型API base url，如未提供则使用.env文件中的"LLM_BASE_URL"
            llm_model_id： 大模型模型id，如未提供则使用.env文件中的"LLM_MODEL_ID"
            temperature：  温度参数，如未提供则使用默认值
            timeout：      响应超时时间，如未提供则使用.env文件中的"LLM_TIMEOUT"

        '''
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL")
        self.llm_model_id = llm_model_id or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        if not all([self.llm_api_key, self.llm_base_url]):
            raise LLMException("❗LLM的API key和base url必须被提供或者放在.env文件中")

        self.client =  OpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            timeout=self.timeout
        )
        self.async_client = AsyncOpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            timeout=self.timeout
        )
        print("[TaiyiAgentLLM]初始化成功")

# ========= LLM调用 同步方法 =========

    def invoke(
        self,
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        '''
        同步非流式输出，调用大模型API接口，返回信息包含内容和工具调用信息
        '''
        try:
            payload = self._build_payload(messages, temperature, tools=tools)
            response = self.client.chat.completions.create(**payload)
            print(f"[TaiyiAgentLLM]✅ 本次大模型调用 {self.llm_model_id} -同步-非流式-输出完成：")
            return self._parse_response(response)

        except Exception as e:
            raise LLMException(f"❌ -同步-非流式-输出LLM API时报错:{str(e)}")

    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        '''
        同步流式输出，调用大模型API接口
        '''
        try:
            payload = self._build_payload(messages, temperature, tools=tools)
            stream = self.client.chat.completions.create(**payload, stream=True)

            print(f"TaiyiAgentLLM]✒️ 大模型{self.llm_model_id} -同步-流式-输出中...")
            for chunk in stream:
                if not chunk.choices:
                    # print(f"\n[用量统计] {chunk}")
                    continue                
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
            print("[TaiyiAgentLLM]\n✅ 本次大模型-同步-流式-输出完成")
        except Exception as e:
            raise LLMException(f"❌ -同步-流式-输出LLM API时报错: {str(e)}")


# ========= LLM调用 异步方法 =========

    async def ainvoke(self, messages, temperature=None, *, tools=None):
        '''
        异步非流式输出llm结果
        '''
        try:
            payload = self._build_payload(messages, temperature, tools=tools)
            response = await self.async_client.chat.completions.create(**payload)
            print(f"[TaiyiAgentLLM]✅ 本次大模型调用 {self.llm_model_id} -异步-非流式-输出完成：")
            return self._parse_response(response)
        except Exception as e:
            raise LLMException(f"❌ -异步-非流式-输出LLM API时报错:{str(e)}")

    async def astream(self, messages, temperature=None, *, tools=None) -> AsyncIterator[str]:
        '''
        异步流式输出llm结果
        '''
        try:
            payload = self._build_payload(messages, temperature, tools=tools)
            stream = await self.async_client.chat.completions.create(**payload, stream=True)

            print(f"[TaiyiAgentLLM]✒️ 大模型{self.llm_model_id} 正在-异步-流式-输出中...")
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
            print("[TaiyiAgentLLM]\n✅ 本次大模型-异步-流式-输出完成")
        except Exception as exc:
            # 包装成你自定义异常，和同步版本保持一致
            raise LLMException(f"❌ -异步-流式-输出LLM API时报错: {str(exc)}") from exc

# ========= 辅助方法 =========

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict:
        '''
        构造llm输入
        '''
        payload = {
            "messages": messages,
            "model": self.llm_model_id,
            "temperature": (temperature if temperature is not None else self.temperature),
        }
        if tools is not None:
            payload["tools"] = tools
        return payload

    def _parse_response(self, response: Any) -> LLMResponse:
        '''
        解析 OpenAI chat comletions 响应
        '''
        try:
            message = response.choices[0].message
            tool_calls: list[ToolCall] = []

            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise LLMException(
                        f"工具 {call.function.name} 返回了非法 JSON 参数"
                    ) from exc

                if not isinstance(arguments, dict):
                    raise LLMException(
                        f"工具 {call.function.name} 的参数必须是 JSON 对象"
                    )

                tool_calls.append(
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                    )
                )

            print("[TaiyiAgentLLM] content:", message.content)
            print("[TaiyiAgentLLM] tool:", tool_calls)
            return LLMResponse(
                content=message.content or "",
                tool_calls=tool_calls,
            )
        except LLMException:
            raise
        except Exception as exc:
            raise LLMException(f"解析LLM响应失败:{exc}") from exc