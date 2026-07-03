from rank_bm25 import BM25Okapi
from multiprocessing import Pool
from typing import List, Tuple, Dict
import json
import os
import logging
from functools import partial
import math

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class CodeSearcher:
    def __init__(self, dataset_path: str):
        """初始化代码搜索器
        
        Args:
            dataset_path: 训练数据集的路径
        """
        self.codes = []
        self.load_dataset(dataset_path)
        self.bm25 = None
        self._build_index()
        
    def load_dataset(self, dataset_path: str):
        """加载数据集"""
        logger.info(f"正在从 {dataset_path} 加载数据集...")
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.codes = []
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        self.codes.append({
                            'code': item['code'],
                            'func_name': item['func_name'],
                            'docstring': item['docstring']
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"跳过无效的JSON行: {line[:100]}...")
                        continue
                    
            logger.info(f"已加载 {len(self.codes)} 条代码记录")
        except Exception as e:
            logger.error(f"加载数据集时出错: {str(e)}")
            raise
        
    def _build_index(self):
        """构建BM25索引"""
        # 组合代码、函数名和文档字符串进行索引
        tokenized_codes = [
            (code['code'] + ' ' + code['func_name'] + ' ' + code['docstring']).lower().split()
            for code in self.codes
        ]
        self.bm25 = BM25Okapi(tokenized_codes)
        logger.info("BM25索引构建完成")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关代码片段"""
        if not self.bm25:
            logger.warning("索引未构建")
            return []
            
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'code': self.codes[idx]['code'],
                'func_name': self.codes[idx]['func_name'],
                'docstring': self.codes[idx]['docstring'],
                'score': scores[idx]
            })
            
        return results

def main():
    searcher = CodeSearcher("src/methods/retriever/dataset/CSN_train.jsonl")
    results = searcher.search("how to read json file in python", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Function name: {result['func_name']}")
        print(f"Relevance score: {result['score']:.4f}")
        print("code:")
        print(result['code'])
        print("docstring:")
        print(result['docstring'])

if __name__ == "__main__":
    main()