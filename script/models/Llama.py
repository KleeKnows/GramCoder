from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base import BaseModel

class LlamaModel(BaseModel):
    def __init__(
        self,
        model_dir='./model/Meta-Llama-3.3-70B-Instruct',
        temperature=0.32,
        max_new_tokens=10000,
        device=None,
        num_gpus=None  # 新增参数：指定使用的GPU数量
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # 打印模型分布情况
        if hasattr(self.model, "hf_device_map"):
            print("\nModel distribution across devices:")
            for name, device in self.model.hf_device_map.items():
                print(f"{name}: {device}")
        
        self.model.eval()

    def prompt(self, processed_input: list[dict]):
        """处理输入并生成响应"""
        try:
            # 使用tokenizer的chat模板处理输入
            text = self.tokenizer.apply_chat_template(
                processed_input,
                tokenize=False,
                add_generation_prompt=True
            )

            # 获取模型的第一个参数所在的设备
            input_device = next(self.model.parameters()).device
            
            # 准备模型输入
            model_input = self.tokenizer([text], return_tensors='pt')
            model_input = {k: v.to(input_device) for k, v in model_input.items()}
            
            attention_mask = torch.ones(
                model_input['input_ids'].shape,
                dtype=torch.long,
                device=input_device
            )

            # 使用自动混合精度
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        model_input['input_ids'],
                        max_new_tokens=self.max_new_tokens,
                        attention_mask=attention_mask,
                        pad_token_id=self.tokenizer.eos_token_id,
                        temperature=self.temperature,
                        use_cache=True
                    )

            # 提取新生成的token
            new_tokens = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_input['input_ids'], generated_ids)
            ]

            # 解码响应
            response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            # 计算token数量
            prompt_tokens = len(model_input['input_ids'][0])
            completion_tokens = len(new_tokens[0])

            return response, prompt_tokens, completion_tokens
            
        except Exception as e:
            print(f"Error in prompt generation: {str(e)}")
            torch.cuda.empty_cache()
            raise

    def __del__(self):
        """析构函数，确保释放GPU内存"""
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
        except:
            pass

class Llama3_3_70B(LlamaModel):
    def prompt(self, processed_input: list[dict]):
        self.model_dir='./model/Meta-Llama-3.3-70B-Instruct'
        return super().prompt(processed_input)