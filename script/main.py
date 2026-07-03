# Standard library imports
import sys
from datetime import datetime
import argparse
import os

# Local imports
# 1.method
from methods.baselines import DirectStrategy, CoTStrategy, SelfPlanningStrategy, AnalogicalStrategy, DirectStrategy_plus
from methods.MapCoder import MapCoder
from methods.GramCoder import GramCoder
from methods.test import Test

# 2.models
from models.OpenAI import GPT4,GPT4o,O1,GPT5,DeepSeekR1
from models.Llama import Llama3_3_70B
from models.Qwen import Qwen3_1_7B

# 3.dataset
from datasets.MBPP import MBPPDataset
from datasets.HumanEval import HumanEvalDataset
from datasets.CodeContestDataset import CodeContestDataset
from datasets.APPSDataset import APPSDataset

from utils.results import Results

# Custom class to write to both terminal and log file
class TeeStream:
    def __init__(self, file_stream, terminal_stream):
        self.file_stream = file_stream
        self.terminal_stream = terminal_stream
    
    def write(self, message):
        self.file_stream.write(message)
        self.terminal_stream.write(message)
        # Ensure the output is immediately visible
        self.file_stream.flush()
        self.terminal_stream.flush()
    
    def flush(self):
        self.file_stream.flush()
        self.terminal_stream.flush()

class MethodSelect:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "GramCoder":
            return GramCoder
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "DirectPlus":
            return DirectStrategy_plus
        elif prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "SelfPlanning":
            return SelfPlanningStrategy
        elif prompting_name == "Analogical":
            return AnalogicalStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "test":
            return Test
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")

class ModelSelect:
    @staticmethod
    def get_model_class(model_name):
        if model_name == "gpt4":
            return GPT4
        elif model_name == "gpt4o":
            return GPT4o
        elif model_name == "o1":
            return O1
        elif model_name == "gpt5":
            return GPT5
        elif model_name == "llama":
            return Llama3_3_70B
        elif model_name == "r1":
            return DeepSeekR1
        elif model_name == "qwen":
            return Qwen3_1_7B
        else:
            raise Exception(f"Unknown model name {model_name}")

class DataSetSelect:
    @staticmethod
    def get_dataset_class(dataset_name):
        if dataset_name == "MBPP":
            return MBPPDataset
        elif dataset_name == "HumanEval":
            return HumanEvalDataset
        elif dataset_name == "APPS":
            return APPSDataset
        elif dataset_name == "CC":
            return CodeContestDataset
        else:
            raise Exception(f"Unknown dataset name {dataset_name}")


# method，model，dataset，language，temperature，pass_at_k，result_path
parser = argparse.ArgumentParser()

# methods: DramPrompt, Direct, CoT, SelfPlanning, Analogical, MapCoder
parser.add_argument(
    "--method", 
    type=str, 
    default="DramCoder",
    choices=[
        "GramCoder",
        "Direct",
        "DirectPlus",
        "CoT",
        "SelfPlanning",
        "Analogical",
        "MapCoder",
        "test",
    ]
)

# select models：gpt4, gpt4o, claude, llama
parser.add_argument(
    '--model', 
    type=str, 
    default='gpt4o',
    choices=[
        'gpt4',
        'gpt4o',
        'o1',
        'gpt5',
        'llama',
        'r1',
        'qwen',
    ]
    )

# datasets：HumanEval, MBPP, APPS, CC
parser.add_argument(
    "--dataset", 
    type=str, 
    default="MBPP", 
    choices=[
        "HumanEval", 
        "MBPP", 
        "CC",
        "APPS",
        ]
    )

parser.add_argument(
    "--language", 
    type=str, 
    default="Python3",
    choices=[
        "Python3",
        "Java",
        "C",
        "C++",
    ]
)

parser.add_argument(
    "--temperature", 
    type=float, 
    default=0.2
)

# pass@k
parser.add_argument(
    "--pass_at_k", 
    type=int, 
    default=1
)

parser.add_argument(
    "--result_path", 
    type=str, 
    default="./results"
)

args = parser.parse_args()
METHOD = args.method
MODEL = args.model
DATASET = args.dataset
LANGUAGE = args.language
TEMPERATURE = args.temperature
PASS_AT_K = args.pass_at_k

FILE_NAME = f"{MODEL}-{METHOD}-{DATASET}-{LANGUAGE}-{TEMPERATURE}-{PASS_AT_K}"
RESULTS_PATH = f"./output/{FILE_NAME}.jsonl"
LOG_FILE_PATH = f"./log/{FILE_NAME}.log"

# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Set up logging to both file and terminal
log_file = open(LOG_FILE_PATH, 'a')
original_stdout = sys.stdout
original_stderr = sys.stderr

# Create Tee streams that write to both log file and terminal
sys.stdout = TeeStream(log_file, original_stdout)
sys.stderr = TeeStream(log_file, original_stderr)

try:
    print(f"#########################\nRunning start {FILE_NAME}, Time: {datetime.now()}\n##########################\n")
    SolveRun = MethodSelect.get_prompting_class(METHOD)(
        model=ModelSelect.get_model_class(MODEL)(temperature=TEMPERATURE),
        data=DataSetSelect.get_dataset_class(DATASET)(),
        language=LANGUAGE,
        pass_at_k=PASS_AT_K,
        results=Results(RESULTS_PATH),
    )
    SolveRun.run()
finally:
    # Restore original stdout and stderr and close log file
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"Execution complete. Log saved to {LOG_FILE_PATH}")