from taiyi_agent.core.llm import TaiyiAgentLLM

llm = TaiyiAgentLLM()
messages = [{"role": "user", "content": "请介绍你自己"}]
response = llm.stream(messages)
result = ""
for chunk in response:
    print(chunk, end="", flush=True)
    result += chunk