from .base import Dataset
from .MBPP_evaluate import evaluate_io, evaluate_functional_correctness
from utils.paths import *


class MBPPDataset(Dataset):
    def __init__(
        self,
        path: str = MBPP_DATA_PATH,
    ):
        super().__init__(path)
        self.id_key = "name"

    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        result = evaluate_functional_correctness(
            problem=item,
            completion=cur_imp
        )
        return result == "passed"

    def evaluate_detailed(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        """评估私有测试，返回详细结果"""
        result = evaluate_functional_correctness(
            problem=item,
            completion=cur_imp
        )
        passed = result == "passed"
        
        test_detail = {
            'test_index': 0,
            'passed': passed,
            'test_key': item.get('test', ''),
            'error': None if passed else result.replace("failed: ", "")
        }
        
        return passed, [test_detail]

    def evaluate_sample_io(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        if "sample_io" not in item:
            return True, ""
        if len(item["sample_io"]) == 0:
            return True, ""
        return evaluate_io(
            sample_io=item["sample_io"],
            completion=cur_imp,
        )

    def evaluate_sample_io_detailed(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        """返回每个测试用例的详细结果"""
        if "sample_io" not in item:
            return True, []
        if len(item["sample_io"]) == 0:
            return True, []
        
        from .MBPP_evaluate import evaluate_io_detailed
        return evaluate_io_detailed(
            sample_io=item["sample_io"],
            completion=cur_imp,
        )

    @staticmethod
    def get_prompt(item):
        # function_signature = item['code'].split('\n')[0].strip()
        # return f"{item['text']}\nFunction Signature: {function_signature}"
        return item["prompt"]