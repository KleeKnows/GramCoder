from utils.utils import read_jsonl


class Dataset(object):
    def __init__(
        self,
        path: str,
    ):
        self.path = path
        self.data = None
        self.id_key = ""
        self.load()

    def load(self):
        self.data = read_jsonl(self.path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        raise NotImplementedError

    def evaluate_detailed(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        """返回私有测试用例的详细结果"""
        raise NotImplementedError

    def evaluate_sample_io(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        raise NotImplementedError

    def evaluate_sample_io_detailed(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        """返回公开测试用例的详细结果"""
        raise NotImplementedError

    @staticmethod
    def get_prompt(item):
        raise NotImplementedError