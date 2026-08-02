
class TaiyiAgentException(Exception):
    '''TaiyiAgent基础异常类'''
    pass

class LLMException(TaiyiAgentException):
    '''LLM API调用异常'''
    pass