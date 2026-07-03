# Standard library imports
import json
import os
import re
import sys
import time
from typing import List, Dict, Tuple, Any, Optional
from copy import deepcopy
import xml.etree.ElementTree as ET

# Package/library imports
import tiktoken
import networkx as nx

# Import BM25 code searcher
# from .retriever.BM25 import CodeSearcher

from .base import BaseMethod, Results
from models.base import BaseModel

from datasets.base import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.HumanEval import HumanEvalDataset
from datasets.MBPP import MBPPDataset
from datasets.CodeContestDataset import CodeContestDataset


from datasets.MBPP_evaluate import evaluate_io

num_mapping = {
    1: "one (1)",
    2: "two (2)",
    3: "three (3)",
    4: "four (4)",
    5: "five (5)",
}

# KB + Exemplars + Example plan + Problem plan + Code Generation + Sample IO testing + Code Improvement


class GramCoder(BaseMethod):
    def __init__(
        self,
        m: int = 2,
        n: int = 3,
        k: int = 2,
        t: int = 4,
        code_dataset_path: str = "src/methods/retriever/dataset/CSN_train.jsonl",
        relevance_threshold: float = 80,
        confidence_threshold: int = 90,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.m = m   # Analyst solution number
        self.n = n   # Programmer plan number
        self.k = k   # Retrieved code number
        self.t = t   # refiner times for each plan
        self.code_dataset_path = code_dataset_path
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        self.retriever_available = False
        # Initialize code searcher
        # try:
        #     self.code_searcher = CodeSearcher(self.code_dataset_path)
        #     self.retriever_available = True
        # except Exception as e:
        #     print(f"Warning: Could not initialize code searcher: {str(e)}")
        #     self.retriever_available = False

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    def parse_xml(self, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            root = ET.fromstring(response)
        except:
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
            except:
                root = ET.fromstring('<root>\n' + response)
        return self.xml_to_dict(root)

    def parse_code(self, response: str) -> str:
        if response is None:
            return ''
        
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        if "```Python" in response:
            code_pattern = r'```Python((.|\n)*?)```'
        if "```Python3" in response:
            code_pattern = r'```Python3((.|\n)*?)```'
        if "```python" in response:
            code_pattern = r'```python((.|\n)*?)```'
        if "```python3" in response:
            code_pattern = r'```python3((.|\n)*?)```'
        if "```C" in response:
            code_pattern = r'```C((.|\n)*?)```'
        if "```c" in response:
            code_pattern = r'```c((.|\n)*?)```'
        if "```C++" in response:
            code_pattern = r'```C\+\+((.|\n)*?)```'
        if "```c++" in response:
            code_pattern = r'```c\+\+((.|\n)*?)```'
        if "```cpp" in response:
            code_pattern = r'```cpp((.|\n)*?)```'
        if "```Cpp" in response:
            code_pattern = r'```Cpp((.|\n)*?)```'
        if "```Java" in response:
            code_pattern = r'```Java((.|\n)*?)```'
        if "```java" in response:
            code_pattern = r'```java((.|\n)*?)```'
        if "```Node" in response:
            code_pattern = r'```Node((.|\n)*?)```'
        if "```node" in response:
            code_pattern = r'```node((.|\n)*?)```'
        if "```Rust" in response:
            code_pattern = r'```Rust((.|\n)*?)```'
        if "```rust" in response:
            code_pattern = r'```rust((.|\n)*?)```'
        if "```PHP" in response:
            code_pattern = r'```PHP((.|\n)*?)```'
        if "```php" in response:
            code_pattern = r'```php((.|\n)*?)```'
        if "```Go" in response:
            code_pattern = r'```Go((.|\n)*?)```'
        if "```go" in response:
            code_pattern = r'```go((.|\n)*?)```'
        if "```Ruby" in response:
            code_pattern = r'```Ruby((.|\n)*?)```'
        if "```ruby" in response:
            code_pattern = r'```ruby((.|\n)*?)```'
        if "```C#" in response:
            code_pattern = r'```C#((.|\n)*?)```'
        if "```c#" in response:
            code_pattern = r'```c#((.|\n)*?)```'
        if "```csharp" in response:
            code_pattern = r'```csharp((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str.strip()

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text 
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        """
        如果sample_io为空或不存在, 返回空字符串
        """
        if not sample_io:
            return ""
            
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return str(sample_io)

    def run_single_pass(self, item: dict):
        """Main process to generate a single solution attempt."""
        # 创建有向图
        CoderGraph = nx.DiGraph()
        CoderGraph.add_node("problem", content=self.data.get_prompt(item))
        intermediate_nodes = {}

        # Track token usage and API calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        item['api_calls'] = item.get('api_calls', 0)
        
        # 安全地获取样本测试用例，提供默认值防止KeyError
        sample_io = item.get('sample_io', [])
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(sample_io)}\n"
        
        for h in range(1, self.m + 1):
            # Step 1: Analyst analyzes the problem and identifies core algorithms
            print(f"\n=== ATTEMPT {h} / {self.m} ===\n")
            if h == 1:
                node_algorithm_output = self._generate_algorithm(item, sample_io_prompt, intermediate_nodes, CoderGraph)
            else:
                node_algorithm_output = self._re_generate_algorithm(item, sample_io_prompt, intermediate_nodes, CoderGraph)
            total_prompt_tokens += node_algorithm_output["prompt_tokens"]
            total_completion_tokens += node_algorithm_output["completion_tokens"]
            item['api_calls'] += node_algorithm_output["api_calls"]
        
            CoderGraph = node_algorithm_output["codergraph"]
            intermediate_nodes = node_algorithm_output["intermediate_nodes"]

            analyst_solution_output = self._generate_solution(item, sample_io_prompt, CoderGraph)
            total_prompt_tokens += analyst_solution_output["prompt_tokens"]
            total_completion_tokens += analyst_solution_output["completion_tokens"]
            item['api_calls'] += analyst_solution_output["api_calls"]

            sorted_solutions = analyst_solution_output["sorted_solutions"]
            intermediate_nodes["possible_solutions"] = sorted_solutions

            print("(OoO) sorted_solutions:")
            # print(sorted_solutions)
            
            # 循环处理每个示例，从置信度最高的开始 
            for i, analyst_solution in enumerate(sorted_solutions):
                print(f"\n=== PROCESSING EXAMPLE {i+1} (Confidence: {analyst_solution.get('confidence', 0)}) ===\n")
                
                CoderGraph.add_node("solution", content=analyst_solution)
                CoderGraph.add_edge("algorithm", "solution")
                
                # Step 2: Break down the task into sub-functions
                analyst_plan_output = self._generate_plan(item, sample_io_prompt, analyst_solution, CoderGraph)
                total_prompt_tokens += analyst_plan_output["prompt_tokens"]
                total_completion_tokens += analyst_plan_output["completion_tokens"]
                item['api_calls'] += analyst_plan_output["api_calls"]
                
                # 保存programmer结果
                analyst_plan = analyst_plan_output["plan"]

                validate_plan_output = self._validate_plan(
                    item,
                    sample_io_prompt,
                    analyst_plan,
                )

                validated_plan = validate_plan_output["plan"]
                CoderGraph.add_node("plan", content=validated_plan)
                CoderGraph.add_edge("solution", "plan")
                
                # Step 3: Generator creates initial code based on the Programmer's plan
                generator_output = self._generate_code(item, sample_io_prompt, analyst_solution, CoderGraph)
                total_prompt_tokens += generator_output["prompt_tokens"]
                total_completion_tokens += generator_output["completion_tokens"]
                item['api_calls'] += generator_output["api_calls"]
                
                # 保存generator结果
                generator_result = generator_output["code"]
                # Test the generated code against sample test cases
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    generator_result,
                    self.language
                )
                # 如果生成的代码通过了测试，直接使用这个结果
                if passed:
                    print(f"Solution {i+1} generated code passed the tests.")
                    code = generator_result
                    break
                else:
                    plan = CoderGraph.nodes["plan"]['content']
                    code = generator_result
                    for j in range(1, self.t + 1):
                        print(f"Debugging attempt {j}...")
                        refiner_output = self._validate_code(
                            item,
                            analyst_solution,
                            plan,
                            code,
                            test_log
                        )
                        total_prompt_tokens += refiner_output["prompt_tokens"]
                        total_completion_tokens += refiner_output["completion_tokens"]
                        item['api_calls'] += refiner_output["api_calls"]
                        plan = refiner_output["plan"]
                        code = refiner_output["code"]
                        passed, test_log = self.data.evaluate_sample_io(
                            item,
                            code,
                            self.language
                        )

                        if passed:
                            break
            if passed:
                print("(OvO) Passed the tests.")
                break
        
        print("________________________\n\n", flush=True)
        return code, total_prompt_tokens, total_completion_tokens

    def _coarse_grained_retriever(self, query: str, top_k: int) -> dict:
        """Coarse-grained retriever: 通过 BM25 检索相似问题的辅助方法"""
        print("\n--- COARSE-GRAINED RETRIEVER ---")
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        if self.retriever_available:
            try:
                bm25_results = self.code_searcher.search(query, top_k=top_k)
                
                # 过滤低于阈值的结果
                filtered_results = []
                for result in bm25_results:
                    if result['score'] >= self.relevance_threshold:
                        filtered_results.append(result)
                
                print(filtered_results)
                if filtered_results:
                    print(f"Found {len(filtered_results)} code examples via BM25, generating solutions...")

                    retrieved_items = []
                    for result in filtered_results:
                        retrieved_items.append({
                            "func_name": result['func_name'],
                            "description": result['docstring'],
                            "code": result['code'],
                        })
                    
                    prompt_for_coarse = [
                        {
                            "role": "user",
                            "content": f"""Analyze retrieved code snippets relevant to the following original problem and generate a pseudocode based on the retrieved code.
# Original Problem:
{query}

# Retrieved Code:
{json.dumps(retrieved_items, indent=2)}

### Task1:
For each code snippet: Analyze how closely it matches the original problem requirements, and evaluate the relevance score (0.0-1.0) based on how well the code can be reused for the original problem.
### Task2:
Select the code snippet with the highest relevance score:
(1) If relevance score <= 0.95:
- Skip this code snippet
- Recall {num_mapping[self.k]} relevant and distinct problems (different from original problem mentioned above) and analyse how to solve these problems;
- Based on the analysis, generate a pseudocode to solve the original problem.
(2) If relevance score > 0.95: 
- Analyse how the retrieved code solve the related problem;
- Based on the retrieved code, generate a pseudocode to solve the original problem.

---------------
Important: your response must only contain the pseudocode to solve the original problem.
"""
                        }
                    ]

                    response, pr_tok, com_tok = self.gpt_chat(
                        processed_input = prompt_for_coarse
                    )
                    api_calls = 1
                    prompt_tokens += pr_tok
                    completion_tokens += com_tok
                                     
            except Exception as e:
                print(f"Error in BM25 search: {str(e)}")
        
        return {
            "results": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }

    def _fine_grained_retriever(self, item: dict, step: str) -> dict:
        """
        Fine-grained retriever: searches for relevant code snippets for a specific step.
        """
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        # 使用步骤内容作为查询
        query = step
        
        # 尝试使用 BM25 搜索器检索代码
        retrieved_snippets = []
        
        if self.retriever_available:
            print(f"Searching for code snippets for step: {query}...")
            try:
                bm25_results = self.code_searcher.search(query, top_k=self.k)
                
                # 过滤掉相关性得分低的结果
                for result in bm25_results:
                    if result['score'] >= self.relevance_threshold:
                        retrieved_snippets.append({
                            "description": f"Function: {result['func_name']} - {result['docstring']}",
                            "code": result['code']
                        })
                
                print(f"Found {len(retrieved_snippets)} relevant code snippets with BM25")
                
            except Exception as e:
                print(f"Error in BM25 search: {str(e)}")
        
        # 使用 LLM 生成相关代码片段
        prompt_for_code_snippet = [
            {
                "role": "user",
                "content": f"""Given a specific step in a programming plan and some reference code snippets, generate a code snippet that implement this step.
# Step to implement:
{step}
## Reference code snippets:
{json.dumps(retrieved_snippets, indent=2) if retrieved_snippets else " "}

Based on reference code snippets, generate a {self.language} code snippet that implement this step.
Code snippet should:
1. Be clear and concise
2. Be inspired by the reference code snippets if available
---------------
- Only return the code snippet, do not add any other explanation or words.
- Your response must be structured as follows, do not output explanation:
```
{self.language} code snippet that implement this step.
```
"""
            }
        ]

        response, pr_tok, com_tok = self.gpt_chat(prompt_for_code_snippet)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok
        
        return {
            "snippets": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }

    def _generate_algorithm(self, item: dict, sample_io_prompt: str, intermediate_nodes:dict, CoderGraph) -> dict:
        """Reflect on the problem and generate a concise problem statement, input/output format, and key constraints.
        """
        print("\n=== RUNNING generate_analysis ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        # 构建 LLM 输入提示词
        prompt_for_generate_analysis = [
            {
                "role": "user",
                "content": f"""You are an experienced competitive programming coach. For a algorithmic problem, perform a structured analysis that goes through: Precise Understanding → Initial Algorithm Judgment. Think according to the following structure:
# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}

1. Problem Understanding
(1) Given a complex programming problem, concisely summarize its core requirements and essential nature in your own words. Aim to capture the essence of the problem without unnecessary verbatim repetition.
(2) Consider Sample Test cases, improve understanding:
- Formalize input/output, variable definitions, objective function/conditions
- Identify any potential ambiguities that need clarification

2. Structural Observations & Problem Classification
Recall {self.k} relevant problems and analyse how to solve them. Then, categorize the original problem as either a 'simple problem' or a 'complex problem', and select up to {num_mapping[self.m]} appropriate standard algorithms and data structures for the original problem:
## For simple problem:
select basic algorithms (e.g., brute force, simulation, direct implementation);
## For complex problem:
select up to {num_mapping[self.m]} appropriate advanced algorithms and data structures:
**Algorithms**
(1) Search & Two Pointers (Search):
- Breadth-First Search (BFS): Shortest path (unweighted graphs), traversal, level-order/state BFS
- Depth-First Search (DFS): Traversal, connectivity, topological sort, backtracking, pruning
- Binary Search: On sorted sequences, on answer space
- Two Pointers / Sliding Window: Optimal range, substring counting, shortest covering
(2) Greedy Algorithms: 
Strategies that make locally optimal choices to achieve a global optimum
(3) Recursion & Backtracking
(4) Divide and Conquer:
Merge Sort; Quick Sort; Divide and Conquer DP Optimization
Divide and Conquer + Counting (e.g., inversions)/DSU (e.g., in some offline tree problems)
(5) Dynamic Programming (DP):
Basic DP; Memoization; Bitmask DP (State Compression DP); Tree DP; 

**Data Structure-Specific Techniques**
(1) Linear Structures:
- Stack: Monotone Stack
- Queue: Deque (Double-Ended Queue), Priority Queue (Heap)
(2) Graph Algorithms
- Graph Traversal: BFS; DFS
- Shortest Path Algorithms: Dijkstra's Algorithm (Single Source, Non-negative weights); Bellman-Ford / SPFA (Single Source, Negative weights); Floyd-Warshall (All-Pairs Shortest Path); 0-1 BFS (Edge weights 0 or 1)
- Minimum Spanning Tree (MST): Prim's Algorithm; Kruskal's Algorithm
- Connectivity: Bridges; Articulation Points; Strongly Connected Components (SCC); Biconnected Components (BCC)
- Network Flow:
Max Flow: Augmenting Path Algorithms (Ford-Fulkerson, Edmonds-Karp), Dinic's, ISAP
Min Cut: Max-Flow Min-Cut Theorem
- Topological Sort
- Bipartite Graph: Bipartite Matching(Hungarian Algorithm, Network Flow); Bipartite Check
(3) String Algorithms
- Pattern Matching: KMP Algorithm (Knuth-Morris-Pratt); Z-Algorithm; Rabin-Karp Algorithm (String Hashing); Manacher's Algorithm (Longest Palindromic Substring)
- Advanced String Structures: Suffix Array; Suffix Automaton (SAM); Suffix Tree (ST)
(4) Tree-Specific Techniques
- Binary Search Tree (BST): Balanced BSTs (AVL, Treap, Splay, Red-Black Tree)
- Segment Tree: Lazy Propagation, Value Segment Tree / Persistent Segment Tree
- Fenwick Tree (Binary Indexed Tree - BIT)
- Trie (Prefix Tree)
- Disjoint Set Union (DSU / Union-Find)
- Mergeable Heap (Leftist Heap, Skew Heap)

---------------
Important:
- Think step by step to ensure the correct answer.
- Your response must be concise and follow the format below:
# Understanding
Concise summary of the problem's core requirements and essential nature in your own words.
# analysis
Provide problem categorization and up to {num_mapping[self.m]} appropriate algorithms for the original problem.
""",
            },
        ]

        print("--- Prompt for Analyse: ")
        print(prompt_for_generate_analysis[0]['content'], flush=True)

        analysis, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_generate_analysis
        )
        
        print("\n--- Analyse Response: ")
        print(analysis)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        CoderGraph.add_node("analysis", content=analysis)
        CoderGraph.add_edge("understanding", "analysis")
        intermediate_nodes["possible_algorithms"] = analysis

        prompt_for_generate_algorithm = [
            {
                "role": "user",
                "content": f"""You are an experienced competitive programming coach. Given a problem, choose the most appropriate algorithm based on the analysis.
# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}

Possible_algorithms:
{analysis}
# Your task:
1. Algorithm Selection
Considering the problem's requirements and the provided algorithms, choose the best-suited one. 
- Consider more efficient methods only if they do not compromise correctness.
- Pick the final algorithm you would implement, and justify why it is superior to the alternatives (e.g., efficiency, simpler implementation, avoids double-counting).
2. Write a useful tutorial about the above mentioned algorithms.
- Provide a high level generic tutorial for solving this types of problem.

---------------
Important:
- Think step by step to ensure the correct answer.
- Your response must be concise and follow the format below:
# algorithm
Select the algorithm needed to solve the original problem (do not output extra words).
# totorial
write a concise tutorial about the selected algorithm.
""",
            },
        ]

        algorithm, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_generate_algorithm
        )
        
        print("\n--- Analyse Response: ")
        print(algorithm)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok
        
        # Post processing
        CoderGraph.add_node("algorithm", content=algorithm)
        CoderGraph.add_edge("analysis", "algorithm")

        return {
            "intermediate_nodes": intermediate_nodes,
            "codergraph": CoderGraph,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

    def _re_generate_algorithm(self, item: dict, sample_io_prompt: str, intermediate_nodes:dict, CoderGraph) -> dict:
        """Reflect on the problem and generate a concise problem statement, input/output format, and key constraints.
        """
        print("\n=== RUNNING generate_analysis ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0

        prompt_for_generate_algorithm = [
            {
                "role": "user",
                "content": f"""You are an experienced competitive programming coach. Given a problem, choose the most appropriate algorithm based on the analysis.
# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}
# Your task:
Rethink the original problem and identify explicit and implicit requirements
1. Check the original algorithm
You have tried the previous solutions, but it did not pass the test cases. Analyze the possible reasons for failure based on the test log.
# Previous Solution:
{CoderGraph.nodes["algorithm"]['content']}
# Test Log:
{intermediate_nodes.get("test_log", "")}
2. Choose the final algorithm you would implement again, and prioritize functional correctness over efficiency (prefer simpler implementation).
3. Write a useful tutorial about the above mentioned algorithms.
- Provide a high level generic tutorial for solving this types of problem.

### Multiple Hypothesis Generation
Before settling on an approach:
- Write multiple possible interpretations of the question
- Consider various solution strategies
- Think about potential alternative perspectives
- Keep multiple working hypotheses active
- Avoid premature commitment to a single interpretation
---------------
Important:
- Think step by step to ensure the correct answer. Consider more efficient methods only if they do not compromise correctness.
- Your response must be concise and follow the format below:
# algorithm
Select the algorithm needed to solve the original problem (do not output extra words).
# totorial
write a concise tutorial about the selected algorithm.
""",
            },
        ]

        algorithm, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_generate_algorithm
        )
        
        print("\n--- Analyse Response: ")
        print(algorithm)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok
        
        # Post processing
        CoderGraph.add_node("algorithm", content=algorithm)
        CoderGraph.add_edge("analysis", "algorithm")

        return {
            "intermediate_nodes": intermediate_nodes,
            "codergraph": CoderGraph,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

    def _generate_solution(self, item: dict, sample_io_prompt: str, CoderGraph) -> dict:
        """Reflect on the problem and generate a concise problem statement, input/output format, and key constraints.
        """
        print("\n=== RUNNING ANALYST_SOLUTION ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        prompt_for_generate_solution = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem, analyse it and then identify the algorithm based on the retrieval result.
# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}
---------------
{CoderGraph.nodes["algorithm"]['content']}

# Your task:
Based on the selected algorithm, recall {num_mapping[self.n]} relevant and equivalent problems, then provide possible high-level approaches for solving the original problem using the selected algorithm. For each similar problem,
1. Describe the problem;
2. Generate {self.language} code step by step to solve that problem, if the code is too long, output a short pseudocode to describe the code;
3. Generate a concise plan to solve that problem;
4. Based on the example problem, generate a high-level generic solution strategy for solving the original problem using the selected algorithm, including:
- Core idea.
- Data structure and main tasks.
Prioritize direct methods and simple data structures to ensure correctness.
5. Provide a confidence score of the solution.
---------------
Important:
- Think step by step to ensure the correct answer.
- Output {num_mapping[self.n]} solutions prioritizing correctness.
- Your response must follow the following xml format:

<root>
<problem>
# Recall {num_mapping[self.n]} relevant and distinct problems. Write each problem in the following format, do not generate explanation or format.
<example_description>Describe the problem.</example_description>
<example_code>Generate code to solve this problem. If the code is too long, output a short pseudocode.</example_code>
<example_plan>Generate a plan to solve this problem.</example_plan>
<solution>Provide a high-level generic solution strategy for solving the original problem using the selected algorithm, including core idea, data structure and main tasks.</solution>
<confidence>Based on the selected algorithm and the solution, provide a **integer** type confidence score (0-100) indicating how confident you are that this algorithm and solution strategy will solve the problem correctly. Do not generate extra words.</confidence>
</problem>

# similarly add more problems here...

</root>
""",
            },
        ]

        print("--- Prompt for Analyst_Solution: ")
        print(prompt_for_generate_solution[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_generate_solution
        )
        
        print("\n--- Analyst Response: ")
        print(response)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # Post processing
        response = self.replace_tag(response, 'example_description')
        response = self.replace_tag(response, 'example_code')
        response = self.replace_tag(response, 'example_plan')
        response = self.replace_tag(response, 'solution')
        response = self.replace_tag(response, 'confidence')
        parsed_response = self.parse_xml(response)

        # 列表存储analyst_solutions，确保可以排序
        processed_solutions = []

        for example_no, example in enumerate(parsed_response["problem"], start=1):
            example['confidence'] = int(str(example['confidence']).strip())
            processed_solutions.append(example)
            
        # 按照confidence排序
        if processed_solutions:
            sorted_solutions = sorted(
                processed_solutions, 
                key=lambda x: x['confidence'], 
                reverse=True
            )
        else:
            print("Warning: No valid solutions with confidence found, using original solutions")
            sorted_solutions = [parsed_response] if isinstance(parsed_response, dict) else parsed_response


        return {
            "sorted_solutions": sorted_solutions,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

    def _generate_plan(self, item: dict, sample_io_prompt: str, analyst_solution: str, CoderGraph) -> dict:
        """Breaks down the task and generates a plan with confidence levels."""
        print("\n=== RUNNING GENERATOR_PLAN ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0

        # 构建输入提示词
        prompt_for_plan = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem and its solution, break down the problem into sub-tasks and generate a step-by-step plan to solve the problem.
# Example Problem:
{analyst_solution['example_description']}
Example plan: {analyst_solution['example_plan']}

# Problem to be solved:
{self.data.get_prompt(item)}
{sample_io_prompt}
---------------
Solution:
{CoderGraph.nodes["algorithm"]['content']}
{analyst_solution['solution']}

# Your task:
1. Break down the original problem into sub-tasks and generate a detailed step-by-step plan to solve sub-tasks. Each step should be clear and specific.
- Consider all possible valid inputs and edge cases.
2. Use the sample input to logically apply plan step by step to get the output. Compare the generated output with the sample output and improve the plan utill it works correctly.
3. For each step in the plan, also provide a confidence score (0-100) indicating how confident you are that this step can be correctly implemented in code.
- If the score is below {self.confidence_threshold}, recall how to implement similar functionality, then generate a pseudocode snippet that implement this step.

Let's think step by step to ensure the correct answer.
---------------
Important:
- You should give only the plan to solve the problem. Do not add extra explanation or words.
- Do not add any other information or words.
- Your response must follow the following xml format:

<root>
<step>
<description>Description of the step.</description>
<confidence>Confidence score (0-100) indicating how confident you are that this step can be correctly implemented in code.</confidence>
<snippet>If the confidence score is below {self.confidence_threshold}, recall {self.k} code snippets that implement similar functionality and use them as a reference to generate a pseudocode to implement this step.</snippet>
</step>

# Similarly for other steps...

</root>
"""
            }
        ]

        plan_response, pr_tok, com_tok = self.gpt_chat(prompt_for_plan)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # 解析计划
        plan_response = self.replace_tag(plan_response, 'description')
        plan_response = self.replace_tag(plan_response, 'confidence')
        plan_response = self.replace_tag(plan_response, 'snippet')
        parsed_plan = self.parse_xml(plan_response)
        
        # 提取计划步骤和置信度
        plan_steps = []
        step_confidences = []
        step_snippets = []
        
        # 从XML结构中提取步骤信息
        steps = parsed_plan.get("step", [])
        if not isinstance(steps, list):
            steps = [steps]
        
        for step_idx, step in enumerate(steps):
            if isinstance(step, dict):
                description = step.get("description", "")
                confidence = step.get("confidence", "100")
                snippet = step.get("snippet", "")
                
                # 确保confidence是整数
                try:
                    confidence = int(str(confidence).strip())
                except (ValueError, AttributeError):
                    confidence = 100
                
                plan_steps.append(description)
                step_confidences.append(confidence)
                step_snippets.append(snippet)
            else:
                # 如果step不是字典，可能是字符串，直接添加
                plan_steps.append(str(step))
                step_confidences.append(100)
                step_snippets.append("")
        
        # 评估每个步骤并检索相关代码
        # step_snippets = []
        
        # for step_idx, (step, confidence) in enumerate(zip(plan_steps, step_confidences)):
        #     # 如果置信度低于confidence_threshold，检索相关代码
        #     if confidence < self.confidence_threshold:
        #         print(f"Low confidence ({confidence}) for step {step_idx+1}, retrieving code snippets...")
        #         retriever_result = self._fine_grained_retriever(item, step)
        #         api_calls += retriever_result["api_calls"]
        #         prompt_tokens += retriever_result["prompt_tokens"]
        #         completion_tokens += retriever_result["completion_tokens"]
                
        #         if retriever_result["snippets"]:
        #             step_snippets.append({
        #                 "step_idx": step_idx,
        #                 "step": step,
        #                 "snippets": retriever_result["snippets"]
        #             })
        
        # 为每个步骤添加相关代码片段
        enhanced_plan = ""
        for step_idx, step in enumerate(plan_steps):
            # 添加步骤
            enhanced_plan += f"Step {step_idx+1}: {step}\n"
            snippets = step_snippets[step_idx]
            if snippets:
                enhanced_plan += f"  Example code snippet for step: {snippets}\n"

            # --- If using external retrieval results:
            # for step_snippet in step_snippets:
            #     if step_snippet["step_idx"] == step_idx:
            #         snippets = step_snippet["snippets"]
            #         enhanced_plan += f"  Example code snippet for step: {snippets}\n"
        
        print("--- plan Response: ")
        print(enhanced_plan, flush=True)
        
        return {
            "plan": enhanced_plan,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }
      
    def _generate_code(self, item: dict, sample_io_prompt: str, analyst_solution: dict, CoderGraph) -> dict:
        """The Generator creates initial code based on the Programmer's plan."""
        print("\n=== RUNNING GENERATOR ===\n")
        
        solution_prompt = f"## Solution strategy to solve the original problem:\n{analyst_solution['solution']}"
        algorithm = CoderGraph.nodes["algorithm"]['content']
        plan = CoderGraph.nodes["plan"]['content']
        
        # Standard input/output handling prompt if needed
        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset:
            std_input_prompt = """## Note:
- Strictly follow the sample input and output format. 
- The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take the input using `input()` function then call the function with specified parameters and finally print the output of the function. 
- For array input parse the array then pass it to the function. Parsing technique is given in the sample input output format section.
- Do not add extra print statement otherwise it will failed the test cases.
"""
        else:
            std_input_prompt = ""

        prompt_for_generator = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given the generated plan, generate {self.language} code step by step to solve the problem, and evaluate whether the code can achieve the intended functionality. If the confidence score is below {self.confidence_threshold}, recall {self.k} code snippets that implement similar functionality and use them as a reference to regenerate the code.
# Problem to be solved:
{self.data.get_prompt(item)}
{sample_io_prompt}
## Solution:
{solution_prompt}
(1)Algorithm:
{algorithm}
(2)plan:
{plan}

Let's think step by step to ensure the correct code.
---------------
Important:
{std_input_prompt}
- Confidence score regarding the solvability of the problem must be an integer between 0 and 100.
- Your response must contain only the {self.language} code to solve this problem. The generated code must be inside a triple backtick (```) code block.
"""
            }
        ]

        # print("--- Prompt for Generator: ")
        # print(prompt_for_generator[0]['content'], flush=True)

        code, pr_tok, com_tok = self.gpt_chat(
            prompt_for_generator
        )
        api_calls = 1

        code = self.parse_code(code)

        print("---Generator Response: ")
        print(code, flush=True)

        return {
            "code": code,
            "prompt_tokens": pr_tok,
            "completion_tokens": com_tok,
            "api_calls": api_calls
        }
    
    def _validate_plan(self, item: dict, sample_io_prompt:str, plan: str) -> dict:
        """Checks if the plan is likely to solve the problem."""
        print("\n=== RUNNING PLAN_REFINER ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        prompt_for_evaluator = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem and a proposed plan to solve it, evaluate whether the plan is likely to solve the problem correctly.
# Problem to be solved:
{self.data.get_prompt(item)}
### Test cases:
{sample_io_prompt}
### Plan:
{plan}
# Your Task:
1. For public test cases
- Use public test sample inputs to logically apply the plan step by step to get the output. Compare the generated logical results with the expected output and check if the plan works correctly;
- Analyse the failed test case and identify which step in the plan is incorrect;
- Update the plan of the problem until all sample test cases can be solved correctly.
2. For edge cases
Analyse all possible valid inputs and edge cases: smallest, largest, all zeros/ones, extreme parameter values, etc. 
Improve the plan to handle all possible valid inputs and edge cases correctly.

---------------
Important:
- Think step by step to ensure the correct answer.
- Your response must contain only the plan. Do not add any explanation or code.
"""
            }
        ]
        plan, pr_tok, com_tok = self.gpt_chat(
            prompt_for_evaluator
        )
        api_calls = 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        return {
            "plan": plan,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }
    
    def _validate_code(self, item: dict, analyst_solution: dict, plan: str, code: str, test_log: str) -> dict:
        """The refiner corrects errors by analyzing failing test cases."""
        print("\n=== RUNNING REFINER ===\n")
        
        solution_prompt = f"## Solution strategy to solve the original problem:\n{analyst_solution['solution']}"
           
        # Standard input/output handling prompt if needed
        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset:
            std_input_prompt = """## Note:
- Strictly follow the sample input and output format. 
- The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take the input using `input()` function then call the function with specified parameters and finally print the output of the function. 
- For array input parse the array then pass it to the function. Parsing technique is given in the sample input output format section.
- Do not add extra print statement otherwise it will failed the test cases.
"""
        else:
            std_input_prompt = ""
        
        # Try to improve the code    
        prompt_for_refiner = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. You have generated {self.language} code to solve the given problem, but the generated code can not pass sample test cases. Check if the generated code follows the original plan, and improve your code to solve the problem correctly.

# Problem to be solved:
{self.data.get_prompt(item)}
{solution_prompt}

### Original Plan:
{plan}
### Buggy Code:
{code}
### Test Report:
{test_log}

# Your Task:
1. Analyse the original problem again, and check if your understanding of the problem is correct;
2. Check if the generated code follows the original plan: 
Take the failed sample as the input, and logically deduce the execution process of the code. 
- If the code does not follow the plan: improve the code to solve the problem correctly, focusing on the failed sample in test_log.
- If the code follows the plan: recheck the original plan focusing on the failed sample in test_log
(1) Use a failed sample input to apply the plan step by step to get the output. Compare the generated output with the sample output, analyse the failed test case and identify which step in the plan is incorrect;
(2) Think step by step to improve the plan and code, ensuring the corrected code can handle all valid inputs correctly. 

Let's think step by step to improve {self.language} code to solve the problem correctly.
---------------
Important:
{std_input_prompt}
- Do not add explanation. 
- Output the **modified plan** and the **{self.language} code** enclosed within triple backticks (```) to solve this problem.
- Your response must follow the format below:

<root>
<plan>Analyse and improve the plan.</plan>
<code>{self.language} code enclosed within triple backticks (```).</code>
</root>
"""
            }
        ]
        
        # print(f"--- Prompt for refiner: ")
        # print(prompt_for_refiner_check[0]['content'], flush=True)

        response, prompt_tokens, completion_tokens = self.gpt_chat(
            prompt_for_refiner
        )
        api_calls = 1

        response = self.replace_tag(response, 'plan')
        response = self.replace_tag(response, 'code')
        parsed_response = self.parse_xml(response)

        # 提取改进后的plan和code
        improved_plan = parsed_response.get('plan', '')
        improved_code = parsed_response.get('code', '')

        improved_code = self.parse_code(improved_code)

        print(f"--- Refiner Response: ")
        print(response, flush=True)

        return {
            "plan": improved_plan,
            "code": improved_code,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }