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
        t: int = 3,
        code_dataset_path: str = "src/methods/retriever/dataset/CSN_train.jsonl",
        relevance_threshold: float = 60,
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

        # Track token usage and API calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        item['api_calls'] = item.get('api_calls', 0)
        
        # 安全地获取样本测试用例，提供默认值防止KeyError
        sample_io = item.get('sample_io', [])
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(sample_io)}\n"
        
#         print("(O-O) Simple Prompt:")
#         # Generate code
#         generate_prompt = [
#             {
#                 "role": "user",
#                 "content": f"""You are an expert programmer.
# # Original Problem:
# {self.data.get_prompt(item)}
# {sample_io_prompt}

# Generate {self.language} code step by step to solve the problem, and evaluate whether the code can achieve the intended functionality. If the confidence score is below 95, recall {self.k} code snippets that implement similar functionality and use them as a reference to regenerate the code.

# Let's think step by step to ensure the correct answer.
# ---------------
# Important:
# - Before you generate the response, note all possible valid inputs and edge cases.
# - Your response must contain only {self.language} code to solve the problem using the selected algorithm.
# - Do not add any other text or comments.

# """,
#             },
#         ]

#         initial_code, pr_tok, com_tok = self.gpt_chat(
#             processed_input=generate_prompt
#         )
#         total_prompt_tokens += pr_tok
#         total_completion_tokens += com_tok

#         print(initial_code)

#         passed, test_log = self.data.evaluate_sample_io(
#             item,
#             initial_code,
#             self.language
#         )
#         # 如果生成的代码通过了测试，直接使用这个结果
#         if passed:
#             print(f"DramPromt generated code passed the tests.")
#             code = initial_code
#             return code, total_prompt_tokens, total_completion_tokens
        
        for h in range(1, self.m + 1):
            # Step 1: Analyst analyzes the problem and identifies core algorithms
            if h == 1:
                analyst_algorithm_output = self._analyst_algorithm(item, sample_io_prompt, CoderGraph)
                total_prompt_tokens += analyst_algorithm_output["prompt_tokens"]
                total_completion_tokens += analyst_algorithm_output["completion_tokens"]
                item['api_calls'] += analyst_algorithm_output["api_calls"]
            
                CoderGraph = analyst_algorithm_output["codergraph"]

                analyst_solution_output = self._analyst_solution(item, sample_io_prompt, CoderGraph)
                total_prompt_tokens += analyst_solution_output["prompt_tokens"]
                total_completion_tokens += analyst_solution_output["completion_tokens"]
                item['api_calls'] += analyst_solution_output["api_calls"]

                sorted_solutions = analyst_solution_output["sorted_solutions"]

                print("(OvO) sorted_solutions:")
                # print(sorted_solutions)
            
            else:
                prompt_for_analyst_algorithm = [
                    {
                        "role": "user",
                        "content": f"""For the original problem, provide relevant problems and try to explore different solution strategyes from the original solution.

### Original Problem:
{self.data.get_prompt(item)}

## Original Algorithm and Solution:
{analyst_algorithm}
{analyst_solution}
## Test Log:
{test_log}

Your task:
# Task1: Analyse Problem
1. Take the failed sample in test_log as examples, and analyze whether there are any misunderstandings or omissions in the interpretation of the original problem.
2. Rethink the original problem and identify explicit and implicit requirements;
3. Analyze the transformation process from sample input to expected output, focus on the failed sample in test_log and edge cases. 
- Validate understanding by comparing manual solution with the expected output.

# Task2: Check the original algorithm
1. Take the failed sample in test_log as examples, and verify the correctness of the original algorithm.
2. Select the appropriate algorithm needed to solve the original problem
- Prioritize functional correctness over efficiency: If there is a simulation method or a direct brute force method available, prefer it;


Let's think step by step to ensure the correct answer.
---------------
Important:
- Before you generate the response, note all possible valid inputs and edge cases.
- Your response must be concise and follow the following xml format-

<root>
<understanding>
Rethink the original problem
- Ensure you can understand how all test cases yield the expected output.
</understanding>

<analysis>
- For public test cases, briefly explain how the specified input yields the expected output, and analyze whether there are any misunderstandings or omissions in the interpretation of the original problem.
</analysis>

<algorithm>
(1) Prefer Brute-force for simple problems;
(2) Specialized algorithms for typical algorithmic problems: Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, etc. 
- Consider more efficient methods only if they do not compromise correctness.
- Then write a concise tutorial about the selected algorithm and why choose this algorithm to solve this type of problem. Do not generate code.
</algorithm>
</root>
""",
                    },
                ]

                response, pr_tok, com_tok = self.gpt_chat(
                    processed_input=prompt_for_analyst_algorithm
                )
                
                print("\n--- RE-Analyst Response: ")
                print(response)
                # Post processing
                response = self.replace_tag(response, 'understanding')
                response = self.replace_tag(response, 'analysis')
                response = self.replace_tag(response, 'algorithm') 
                parsed_response = self.parse_xml(response)
                
                algorithm = parsed_response.get('algorithm', '')
                understanding = parsed_response.get('understanding', '')
                analysis = parsed_response.get('analysis', '')
                CoderGraph.add_node("understanding", content=understanding)
                CoderGraph.add_node("analysis", content=analysis)
                CoderGraph.add_node("algorithm", content=algorithm)
                CoderGraph.add_edge("problem", "understanding")
                CoderGraph.add_edge("understanding", "analysis")
                CoderGraph.add_edge("analysis", "algorithm")

                analyst_solution_output = self._analyst_solution(item, sample_io_prompt, CoderGraph)
                total_prompt_tokens += analyst_solution_output["prompt_tokens"]
                total_completion_tokens += analyst_solution_output["completion_tokens"]
                item['api_calls'] += analyst_solution_output["api_calls"]

                sorted_solutions = analyst_solution_output["sorted_solutions"]

                

            analyst_analysis = CoderGraph.nodes["analysis"]['content'] if "analysis" in CoderGraph.nodes else ""    
            analyst_algorithm = CoderGraph.nodes["algorithm"]['content'] if "algorithm" in CoderGraph.nodes else ""

            # 循环处理每个示例，从置信度最高的开始 
            for i, analyst_solution in enumerate(sorted_solutions):
                print(f"\n=== PROCESSING EXAMPLE {i+1} (Confidence: {analyst_solution.get('confidence', 0)}) ===\n")
                
                CoderGraph.add_node("solution", content=analyst_solution['solution'], confidence=analyst_solution['confidence'])
                CoderGraph.add_edge("algorithm", "solution")

                # Step 2: Break down the task into sub-functions
                analyst_plan_output = self._analyst_plan(item, analyst_solution, analyst_algorithm, sample_io_prompt, analyst_analysis)
                total_prompt_tokens += analyst_plan_output["prompt_tokens"]
                total_completion_tokens += analyst_plan_output["completion_tokens"]
                item['api_calls'] += analyst_plan_output["api_calls"]
                
                # 保存programmer结果
                analyst_plan = analyst_plan_output["plan"]
                CoderGraph.add_node("plan", content=analyst_plan)
                CoderGraph.add_edge("solution", "plan")
                
                # Step 3: Generator creates initial code based on the Programmer's plan
                generator_output = self._run_generator(item, analyst_solution, analyst_algorithm, analyst_plan, sample_io_prompt)
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
                    plan = analyst_plan
                    code = generator_result
                    for j in range(1, self.t + 1):
                        print(f"Debugging attempt {j}...")
                        refiner_output = self._run_refiner(
                            item,
                            analyst_solution,
                            analyst_algorithm,
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

    def _analyst_algorithm(self, item: dict, sample_io_prompt: str, CoderGraph) -> dict:
        """Reflect on the problem and generate a concise problem statement, input/output format, and key constraints.
        """
        print("\n=== RUNNING ANALYST_REFLECT ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        # 构建 LLM 输入提示词
        prompt_for_analyst_reflect = [
            {
                "role": "user",
                "content": f"""You are an expert programmer.
# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}

Your task:
# Task1: Problem Understanding
1. Think about the original problem and develop an initial understanding.
2. Work Through the Sample Test Cases:
(1) Demonstrate how the problem would be logically solved using the provided public test cases based on the problem description.
(2) Validate understanding by comparing logical results with the expected output.If there is a discrepancy between the theoretical output and actual results, analyze whether there are any misunderstandings or omissions in the interpretation of the original problem.
(3) Update the understanding of the problem until all sample test cases can be solved correctly.
3. Understand Key Constraints and Implicit Requirements: 
- Based on the analysis of sample test cases, generate a concise problem statement to clearly identify the problem requirements.
- Define the structure and constraints of the input and the expected format of the output.

# Task 2: Analysis
1. For public test cases, briefly explain how the specified input yields the expected output.
2. Describe a key idea to Solving the Problem (Without Coding):
The general idea a human might use to solve the problem, using mathematical or logical reasoning.
- Highlight any hidden conditions or constraints in the problem, such as edge cases, geometric rules or precision requirements.

Let's think step by step to ensure the correct answer.
---------------
Important:
- Before you generate the response, note all possible valid inputs and edge cases. Do not output too many details.
- Your response must be concise and follow the following xml format-

<root>
<understanding>
Write a detailed explanation of the problem(Requirements, Input and Output Format)
</understanding>

<analysis>
- For public test cases, briefly explain how the specified input yields the expected output, and analyze whether there are any misunderstandings or omissions in the interpretation of the original problem.
</analysis>

<idea>
Write the concise idea a human might use to solve the problem, using mathematical or logical reasoning.(include discussion of how to handle edge cases or special input scenarios)
</idea>
</root>
""",
            },
        ]

        # print("--- Prompt for Analyst: ")
        # print(prompt_for_analyst_reflect[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_analyst_reflect
        )
        
        print("\n--- Analyst Response: ")
        print(response)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # Post processing
        response = self.replace_tag(response, 'understanding')
        response = self.replace_tag(response, 'analysis')
        response = self.replace_tag(response, 'idea')
        parsed_response = self.parse_xml(response)
        
        understanding = parsed_response.get('understanding', '')
        analysis = parsed_response.get('analysis', '')
        idea = parsed_response.get('idea', '')

        CoderGraph.add_node("understanding", content=understanding)
        CoderGraph.add_node("analysis", content=analysis)
        CoderGraph.add_node("idea", content=idea)
        CoderGraph.add_edge("understanding", "analysis")
        CoderGraph.add_edge("analysis", "idea")

        prompt_for_analyst_algorithm = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem, analyse it and then identify the algorithm based on the retrieval result.
### Original Problem:
{self.data.get_prompt(item)}
### Problem Analysis:
{understanding}
### Test cases:
{analysis}
Initial Idea:
{idea}

Your task:
Based on analysis, recall how to solve similar problems, then select a appropriate algorithm for original problem.
# Algorithm Selection
1. Output the appropriate algorithm needed to solve the original problem
(1) If the problem is not a typical algorithmic problem: use Brute-force;
(2) If necessary, use specialized algorithms for typical algorithmic problems: Greedy, Dynamic Programming, Divide-and-conquer, Backtracking, Recursive, Binary search, DFS and BFS, etc.
Explain why choose this algorithm to solve this type of problem
2. Write a concise tutorial about the selected algorithm.

Let's think step by step to ensure the correct answer.
---------------
Important:
- Before you generate the response, note all possible valid inputs and edge cases.
- Your response must be concise and follow the following format-
# algorithm
Identify the algorithm needed to solve the original problem and explain why choose this algorithm to solve this type of problem.

# totorial
write a concise tutorial about the selected algorithm.
""",
            },
        ]

        algorithm, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_analyst_algorithm
        )
        
        print("\n--- Analyst Response: ")
        print(algorithm)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # Post processing
        CoderGraph.add_node("algorithm", content=algorithm)
        CoderGraph.add_edge("analysis", "algorithm")

        return {
            "analyst_algorithm": algorithm,
            "codergraph": CoderGraph,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

    def _analyst_solution(self, item: dict, sample_io_prompt: str, CoderGraph) -> dict:
        """Reflect on the problem and generate a concise problem statement, input/output format, and key constraints.
        """
        print("\n=== RUNNING ANALYST_SOLUTION ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        prompt_for_analyst_solution = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem, analyse it and then identify the algorithm based on the retrieval result.
## Original Problem:
{self.data.get_prompt(item)}
Problem Analysis:
{CoderGraph.nodes["understanding"]['content']}
### Test cases:
{CoderGraph.nodes["analysis"]['content']}

### Algorithm:
{CoderGraph.nodes["algorithm"]['content']}

Your task:
Based on analysis, recall {num_mapping[self.n]} equivalent and distinct problems(different from original problem mentioned above).

Let's think step by step to ensure the correct answer.
---------------
Important:
- Before you generate the response, note all possible valid inputs and edge cases.
- Output {self.n} problems.
- Your response must be concise and follow the following xml format-

<root>
<problem>
# Based on analysis, recall {num_mapping[self.n]} equivalent problems with detailed descriptions (different from original problem mentioned above). For each problem,
<example_description>Describe the problem in detail.</example_description>
<example_code>Generate example pseudocode to solve that problem(using the selected algorithm).</example_code>
<example_plan>Generate a plan to solve that problem.</example_plan>
<solution>Provide a high-level generic solution strategy for solving the original problem using the selected algorithm, including the data structure and main tasks. Do not generate code.</solution>
<confidence>Based on the selected algorithm and the solution, provide a **integer** type confidence score (0-100) indicating how confident you are that this algorithm and solution strategy will solve the problem correctly. Do not generate extra words.</confidence>
</problem>

# similarly add more problems here...

</root>
""",
            },
        ]

        print("--- Prompt for Analyst_Solution: ")
        print(prompt_for_analyst_solution[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_analyst_solution
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
            "codergraph": CoderGraph,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

    def _analyst_plan(self, item: dict, analyst_solution: dict, analyst_algorithm: str, sample_io_prompt: str, analyst_analysis: str) -> dict:
        """Breaks down the task and generates a plan with confidence levels."""
        print("\n=== RUNNING ANALYST_PLAN ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0

        # 构建输入提示词
        prompt_for_plan = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem and a relevant example with its solution, break down the problem into sub-tasks and generate a step-by-step plan to solve the problem.
## Example Problem:
{analyst_solution['example_description']}
{analyst_solution['example_code']}
Example plan: {analyst_solution['example_plan']}

## Problem to be solved:
{self.data.get_prompt(item)}
{sample_io_prompt}
{analyst_analysis}
### Solution:
{analyst_algorithm}
{analyst_solution['solution']}

# Your task:
1. Break down the original problem into sub-tasks and generate a detailed step-by-step plan to solve sub-tasks. Each step should be clear and specific.
2. Use the sample input to logically apply plan step by step to get the output. Compare the generated output with the sample output and improve the plan utill it works correctly.
3. For each step in the plan, also provide a confidence score (0-100) indicating how confident you are that this step can be correctly implemented in code.
- If the score is below {self.confidence_threshold}, recall {self.k} code snippets that implement similar functionality, then generate a pseudocode that implement this step.

Let's think step by step to ensure the correct answer.
---------------
Important:
- You should give only the plan to solve the problem. Do not add extra explanation or words.
- Do not add any other information or words.

Your response must follow the following xml format-

<root>
<step>
<description>Description of the step.</description>
<confidence>Confidence score (0-100) indicating how confident you are that this step can be correctly implemented in code.</confidence>
<pseudocode>If the confidence score is below {self.confidence_threshold}, recall {self.k} code snippets that implement similar functionality and use them as a reference to generate a pseudocode to implement this step.</pseudocode>
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
        plan_response = self.replace_tag(plan_response, 'pseudocode')
        parsed_plan = self.parse_xml(plan_response)
        
        # 提取计划步骤和置信度
        plan_steps = []
        step_confidences = []
        step_pseudocode = []
        
        # 从XML结构中提取步骤信息
        steps = parsed_plan.get("step", [])
        if not isinstance(steps, list):
            steps = [steps]
        
        for step_idx, step in enumerate(steps):
            if isinstance(step, dict):
                description = step.get("description", "")
                confidence = step.get("confidence", "100")
                pseudocode = step.get("pseudocode", "")
                
                # 确保confidence是整数
                try:
                    confidence = int(str(confidence).strip())
                except (ValueError, AttributeError):
                    confidence = 100
                
                plan_steps.append(description)
                step_confidences.append(confidence)
                step_pseudocode.append(pseudocode)
            else:
                # 如果step不是字典，可能是字符串，直接添加
                plan_steps.append(str(step))
                step_confidences.append(100)
                step_pseudocode.append("")
        
        # 评估每个步骤并检索相关代码
        step_snippets = []
        
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
            snippets = step_pseudocode[step_idx]
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

    def _run_generator(self, item: dict, analyst_solution: dict, analyst_algorithm: str, programmer_result: dict, sample_io_prompt: str) -> dict:
        """The Generator creates initial code based on the Programmer's plan."""
        print("\n=== RUNNING GENERATOR ===\n")
        
        solution_prompt = f"## Solution strategy to solve the original problem:\n{analyst_solution['solution']}"
        plan = programmer_result
        
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
                "content": f"""You are an expert programmer. See the plan to solve the problem and implement code to solve it.
# Problem to be solved:
{self.data.get_prompt(item)}
{sample_io_prompt}
## Algorithm:
{analyst_algorithm}
{solution_prompt}
# plan:
{plan}

Let's think step by step to ensure the correct code.
---------------
{std_input_prompt}
## Important:
- Your response must contain only the {self.language} code to solve this problem.
- The generated code must be inside a triple backtick (```) code block.
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

    def _run_refiner(self, item: dict, analyst_solution: dict, analyst_algorithm: str, plan: str, code: str, test_log: str) -> dict:
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
        prompt_for_refiner_check = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. You have generated {self.language} code to solve the given problem, but the generated code can not pass sample test cases. Check if the generated code follows the original plan, and improve your code to solve the problem correctly.

# Problem to be solved:
{self.data.get_prompt(item)}

### Original Plan:
{plan}
### Buggy Code:
{code}
### Test Report:
{test_log}

# Your Task:
1. Check if the generated code follows the original plan: 
- Take the failed sample as the input, and logically deduce the execution process of the code. 
- Analyse whether the code follows the plan step by step, and whether the output is correct.
2. If the code follows the plan: output "Recheck plan", do not output other information.
- If the code does not follow the plan, improve the code to solve the problem correctly.

Let's think step by step to modify {self.language} Code for solving this problem.
---------------
{std_input_prompt}
## Important:
- Do not add explanation. Do not generate same code.
- Output "Recheck plan" or the improved **{self.language} code** to solve this problem. Do not output any other information.
"""
            }
        ]
        
        # print(f"--- Prompt for refiner: ")
        # print(prompt_for_refiner_check[0]['content'], flush=True)

        response, prompt_tokens, completion_tokens = self.gpt_chat(
            prompt_for_refiner_check
        )
        api_calls = 1

        if response.strip() == "Recheck plan":
            print("---Recheck plan")

            prompt_for_refiner = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. You have generated {self.language} code to solve the given problem, but the generated code can not pass sample test cases. Check if original plan works, and improve your code to solve the problem correctly.

# Problem to be solved:
{self.data.get_prompt(item)}
Algorithm:
{solution_prompt}

### Buggy Plan:
{plan}
### Buggy Code:
{code}
### Test Report:
{test_log}

# Your Task: Check the original plan
1. Use a failed sample input to apply the plan step by step to get the output. Compare the generated output with the sample output and check if the plan works correctly;
2. Analyse the failed test case and identify which step in the plan is incorrect and improve the plan;
3. Generate modified plan and code. 

Let's think step by step to modify {self.language} Code for solving this problem.
---------------
{std_input_prompt}
## Important:
- Do not add explanation. Do not generate same code.
- Output the case analysis results, **modified plan** and the **{self.language} code** to solve this problem.
- Your response must follow the following xml format-

<root>
<case_analysis>Analyse the failed test case and identify which step in the plan is incorrect.</case_analysis>
<plan>Generate a modified plan.</plan>
<code>Executeable {self.language} code inside ``` block to solve this problem.</code>
</root>
"""
            }
        ]
        
            # print(f"--- Prompt for refiner: ")
            # print(prompt_for_refiner[0]['content'], flush=True)

            response, prompt_tokens, completion_tokens = self.gpt_chat(
                prompt_for_refiner
            )
            api_calls = 1

            response = self.replace_tag(response, 'case_analysis')
            response = self.replace_tag(response, 'plan')
            response = self.replace_tag(response, 'code')
            parsed_response = self.parse_xml(response)

            # 提取改进后的plan和code
            case_analysis = parsed_response.get('case_analysis', '')
            improved_plan = parsed_response.get('plan', '')
            improved_code = parsed_response.get('code', '')

            improved_code = self.parse_code(improved_code)

            print(f"---refiner Response: ")
            print(response, flush=True)
        
        else:
            improved_plan = plan
            improved_code = self.parse_code(response)

        return {
            "plan": improved_plan,
            "code": improved_code,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }