from taiyi_agent.core.config import Config

# 是类方法，但是返回值是个实例对象
cfg1 = Config.from_env()
print("cfg1.temperature 修改前:", cfg1.temperature)
cfg1.temperature  = 1.88
print("cfg1.temperature 修改后:", cfg1.temperature)
print(cfg1.to_dict())

cfg2 = Config.from_env()
print("cfg2.temperature 修改前:",cfg2.temperature)
cfg2.temperature  = 1.66
print("cfg2.temperature 修改后:",cfg2.temperature)
print(cfg2.to_dict())