from models.Llama import Llama3_3_70B
from models.Qwen import Qwen1_5B, Qwen3_1_7B

model = Qwen3_1_7B()

messages = [
    {'role': 'user', 'content': '写一段代码实现冒泡排序算法，返回结果只包含代码，不要任何解释' }
]
response, prompt_tokens, completion_tokens = model.prompt(messages)
print(f"Response: {response}")
print(f"Prompt tokens: {prompt_tokens}")
print(f"Completion tokens: {completion_tokens}")