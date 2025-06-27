import dashscope


import dashscope
from http import HTTPStatus

class DashscopeProcessor:
    def __init__(self, dashscope_api_key="", model="qwen-turbo-latest", temperature=0.1):
        """
        初始化DashscopeProcessor。

        Args:
            dashscope_api_key (str): DashScope API密钥。
            model (str): 使用的大模型名称，默认为"qwen-turbo-latest"。
            temperature (float): 生成的随机性，默认为0.1。
        """
        dashscope.api_key = dashscope_api_key
        self.model = model
        self.temperature = temperature

    def send_message(self, rag_prompt):
        """
        发送消息到DashScope Qwen大模型并获取响应。

        Args:
            rag_prompt (str): 消息内容

        Returns:
            str: 大模型的响应内容，如果请求失败则返回None。
        """
        system_prompt="你是文档归纳总结专家，根据用户的查询和检索结果，进行归纳和总结，请用中文回答。"
        messages = []
        # 添加系统消息，包含上下文信息
        messages.append({"role": "system", "content": system_prompt})

        # 添加用户查询和检索结果
        messages.append({"role": "user", "content": rag_prompt})

        try:
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                result_format='message'
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                print(f"DashScope API请求失败: {response.status_code}, {response.message}")
                return None
        except Exception as e:
            print(f"调用DashScope API时发生错误: {e}")
            return None
