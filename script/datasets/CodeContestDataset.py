from .base import Dataset
from .evaluate.evaluation import contest_evaluate, contest_evaluate_detailed, contest_evaluate_public_tests, contest_evaluate_public_tests_detailed
from utils.paths import *

class CodeContestDataset(Dataset):
    def __init__(
        self,
        path: str=CODE_CONTEST_DATA_PATH,
    ):
        super().__init__(path)
        self.id_key = "id"

    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        return contest_evaluate_detailed(
            generated_code=cur_imp,
            id=item["id"],
            tests=item["test_list"],
            lang=language
        )
    
    def evaluate_detailed(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        """评估私有测试用例，返回详细结果"""
        return contest_evaluate_detailed(
            generated_code=cur_imp,
            id=item["id"],
            tests=item["test_list"],
            lang=language
        )
    
    def evaluate_sample_io(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        return contest_evaluate_public_tests(
            generated_code=cur_imp,
            id=item["id"],
            tests=item["sample_io"],
            lang=language
        )

    def evaluate_sample_io_detailed(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        """返回每个测试用例的详细结果"""
        return contest_evaluate_public_tests_detailed(
            generated_code=cur_imp,
            id=item["id"],
            tests=item["sample_io"],
            lang=language
        )

    @staticmethod
    def get_prompt(item):
        return f"{item['description']}\n\n-------\nImportant Note: You must follow the input output format. Input must be taken from standard input and output must be given to standard output. The code will be tested against multiple test cases and all the test cases must be passed."