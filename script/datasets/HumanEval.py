from .base import Dataset
from utils.paths import *

# evaluation
from typing import *
import contextlib
import signal


class HumanEvalDataset(Dataset):
    def __init__(
        self,
        path: str = HUMAN_WST_DATA_PATH,
    ):
        super().__init__(path)
        self.id_key = "task_id"

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
        return evaluate_io_detailed(
            sample_io=item["sample_io"],
            completion=cur_imp,            
        )


    @staticmethod
    def get_prompt(item):
        if "prompt" in item:
            return f"{item['prompt']}"
        elif "text" in item:
            return f"{item['text']}"
        else:
            raise Exception("No prompt or text in item")

# timeout
def timeout_handler(_, __):
    raise TimeoutError()

import os, json
def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)

from threading import Thread
class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret
    

def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()
    else:
        return result_container[0]

# evaluate function    
def evaluate_io(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    stop_early: bool = False,
):
    test_log = ""
    passed = True
    for io in sample_io:
        try:
            code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
                completion + "\n" + io + "\n"
            function_with_timeout(
                exec,
                (code, globals()),
                timeout
            )
            test_log += f"passed in test case: {io}\n"
        except Exception as e:
            if stop_early:
                return False, f"failed in test case: {io}\n"
            passed = False
            test_log += f"failed in test case: {io}\n"

    return passed, test_log

def evaluate_io_detailed(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
):
    """
    返回每个测试用例的详细结果
    返回值: (passed: bool, results: List[dict])
    """
    detailed_results = []
    passed = True
    
    for idx, io in enumerate(sample_io):
        try:
            code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
                completion + "\n" + io + "\n"
            function_with_timeout(
                exec,
                (code, globals()),
                timeout
            )
            test_result = {
                'test_index': idx,
                'passed': True,
                'test_case': io,
                'error': None,
            }
        except Exception as e:
            passed = False
            test_result = {
                'test_index': idx,
                'passed': False,
                'test_case': io,
                'error': str(e),
            }
        
        detailed_results.append(test_result)
    
    return passed, detailed_results

def evaluate_io_et(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    prompt: str = "",
):
    io = "\n".join(sample_io)
    try:
        code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
            prompt + completion + "\n" + io + "\n"
        function_with_timeout(
            exec,
            (code, globals()),
            timeout
        )
        return True
    except Exception as e:
        return False

def evaluate_functional_correctness(
    problem: Dict,
    completion: str,
    timeout: int = 5,
    test_key: str = "test",
):
    # if problem["name"] == "mbpp_61_count_Substrings":
    #     pass
    try:
        code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
            completion + "\n" + problem[test_key] + \
            "\n" + f"check({problem['entry_point']})"

        function_with_timeout(
            exec,
            (code, globals()),
            timeout
        )
        return "passed"
    except Exception as e:
        return f"failed: {e}"


def evaluate_functional_correctness2(
    problem: Dict,
    completion: str,
    timeout: float = 10,
) -> Dict:

    check_program = (
        # problem["prompt"] +
        "from typing import *\n" +
        completion + "\n" +
        problem["test"] + "\n" +
        f"check({problem['entry_point']})"
    )
    # print(check_program)

    try:
        exec(check_program)
        return "passed"
    except TimeoutException:
        return "timed out"
    except BaseException as e:
        return f"failed: {e}"

class TimeoutException(Exception):
    pass