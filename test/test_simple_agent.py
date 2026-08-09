from taiyi_agent.core.llm import TaiyiAgentLLM
from taiyi_agent.agents.simple_agent import SimpleAgent

my_llm = TaiyiAgentLLM()
my_agent = SimpleAgent(name="测试助手",llm=my_llm, system_prompt="你是一个对话助手，请精简的回答用户的问题")
result = my_agent.run("你好")
print(result)
result = my_agent.run("你是谁？")
print(result)
response = my_agent.stream_run("ai agent是什么？")
for chunk in response:
    print(chunk, end='', flush=True)
print("获取历史消息: \n", my_agent.history.get_history())
print("获取系统提示词: \n", my_agent.get_system_prompt())
print("清除历史消息: \n", my_agent.history.clear_history())
result = my_agent.run("大模型上下文是什么？")
print(result)
print("获取历史消息: \n", my_agent.history.get_history())