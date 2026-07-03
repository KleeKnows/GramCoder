from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from .base import BaseModel


class QwenModel(BaseModel):
    def __init__(
        self,
        model_dir='./model/DeepSeek-R1-Distill-Qwen-1.5B',
        temperature=0.32,
        max_new_tokens=32768,
        device=None,
        num_gpus=2  # 指定使用的GPU数量
    ):
        self.model_dir = model_dir
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # 检查可用的 GPU
        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            self.num_gpus = min(num_gpus, total_gpus) if num_gpus else total_gpus
            print(f"Using {self.num_gpus} GPUs")
            
            # 显示GPU信息
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f}GB")
        else:
            self.num_gpus = 0
            print("No GPU available, using CPU")
            return
        
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        
        # 设置显存限制
        max_memory = {}
        for i in range(self.num_gpus):
            max_memory[i] = f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1024**2)}MB"
        # 为未使用的GPU分配0显存
        for i in range(self.num_gpus, total_gpus):
            max_memory[i] = "0MB"
        max_memory["cpu"] = "24GB"
        
        print("\nMemory allocation:")
        for device, mem in max_memory.items():
            if device != "cpu":
                print(f"GPU {device}: {mem}")
        print(f"CPU: {max_memory['cpu']}")
        
        # 初始化分词器和模型
        print(f"\nLoading tokenizer from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        print(f"Loading model from {model_dir}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        # 打印模型分布情况
        if hasattr(self.model, "hf_device_map"):
            print("\nModel distribution across devices:")
            for name, device in self.model.hf_device_map.items():
                print(f"{name}: {device}")
        
        self.model.eval()
        print("Model loaded successfully!")

    def prompt(self, processed_input: list[dict]):
        """处理输入并生成响应"""
        try:
            # 处理输入格式
            if isinstance(processed_input, str):
                # 如果输入是字符串，直接使用
                text = processed_input
                prompt_only = text
            elif isinstance(processed_input, list):
                # 如果输入是对话列表，尝试使用 apply_chat_template
                try:
                    text = self.tokenizer.apply_chat_template(
                        processed_input,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True  # 启用思考模式
                    )
                    prompt_only = text
                except Exception as e:
                    # 如果enable_thinking不支持，尝试不使用
                    try:
                        text = self.tokenizer.apply_chat_template(
                            processed_input,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        prompt_only = text
                    except:
                        # 如果 apply_chat_template 失败，手动构造对话
                        text = self._build_chat_prompt(processed_input)
                        prompt_only = text
            else:
                raise ValueError(f"Unsupported input type: {type(processed_input)}")

            # 获取模型所在的设备
            model_device = next(self.model.parameters()).device
            
            # 准备模型输入
            model_input = self.tokenizer([prompt_only], return_tensors='pt').to(model_device)
            
            # 计算输入token数量
            prompt_tokens = len(model_input['input_ids'][0])
            
            # 生成文本
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_input,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )

            # 提取新生成的token
            output_ids = generated_ids[0][len(model_input['input_ids'][0]):].tolist()
            
            # 解码响应
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # 自动截取</think>之后的内容（使用token ID方式）
            # 151668 是</think>的token ID
            try:
                # 从后向前查找</think> token
                think_end_index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                # 如果没有找到</think>，尝试用字符串匹配
                think_end_index = 0
                if "</think>" in response:
                    response = response.split("</think>", 1)[1].strip()

            # 使用token级别的分割
            thinking_content = self.tokenizer.decode(output_ids[:think_end_index], skip_special_tokens=True).strip("\n")
            response = self.tokenizer.decode(output_ids[think_end_index:], skip_special_tokens=True).strip("\n")
           
            # 计算completion token数量
            completion_tokens = len(output_ids)

            return response, prompt_tokens, completion_tokens
            
        except Exception as e:
            print(f"Error in prompt generation: {str(e)}")
            torch.cuda.empty_cache()
            raise

    def _build_chat_prompt(self, messages: list[dict]) -> str:
        """手动构造对话提示词"""
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt += f"{content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt

    def generate_text(self, prompt: str, max_length=None) -> str:
        """生成文本的便捷方法"""
        if max_length is None:
            max_length = self.max_new_tokens
        
        response, _, _ = self.prompt(prompt)
        return response

    def chat(self, user_message: str, system_prompt: str = "") -> str:
        """对话的便捷方法"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        response, _, _ = self.prompt(messages)
        return response

    def __del__(self):
        """析构函数，确保释放GPU内存"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            torch.cuda.empty_cache()
        except:
            pass


class Qwen1_5B(QwenModel):
    """DeepSeek-R1-Distill-Qwen-1.5B 模型"""
    def __init__(
        self,
        model_dir='./model/DeepSeek-R1-Distill-Qwen-1.5B',
        temperature=0.7,
        device=None,
        num_gpus=None
    ):
        super().__init__(
            model_dir=model_dir,
            temperature=temperature,
            device=device,
            num_gpus=num_gpus
        )

class Qwen3_1_7B(QwenModel):
    """Qwen3-1.7B 模型"""
    def __init__(
        self,
        model_dir='./model/Qwen3-1.7B',
        temperature=0.32,
        device=None,
        num_gpus=None
    ):
        super().__init__(
            model_dir=model_dir,
            temperature=temperature,
            device=device,
            num_gpus=num_gpus
        )