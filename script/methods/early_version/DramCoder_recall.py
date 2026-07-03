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

# Import BM25 code searcher
from .retriever.BM25 import CodeSearcher

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


class DramCoder(BaseMethod):
    def __init__(
        self,
        m: int = 2,
        n: int = 3,
        k: int = 2,
        t: int = 4,
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
        self.t = t   # Debugger times for each plan
        self.code_dataset_path = code_dataset_path
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        # Initialize code searcher
        try:
            self.code_searcher = CodeSearcher(self.code_dataset_path)
            self.retriever_available = True
        except Exception as e:
            print(f"Warning: Could not initialize code searcher: {str(e)}")
            self.retriever_available = False

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
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        languages = [
            "Python", "Python3", "python", "python3", 
            "C", "c", "C++", "c++", "Java", "java", 
            "Node", "node", "Rust", "rust", "PHP", "php", 
            "Go", "go", "Ruby", "ruby", "C#", "c#", "csharp"
        ]
        
        for lang in languages:
            if f"```{lang}" in response:
                escaped_lang = re.escape(lang)
                code_pattern = f'```{escaped_lang}((.|\n)*?)```'
                break

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str

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
        print("", flush=True)
        
        # Track token usage and API calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        item['api_calls'] = item.get('api_calls', 0)
        
        # 安全地获取样本测试用例，提供默认值防止KeyError
        sample_io = item.get('sample_io', [])
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(sample_io)}\n"
        
        for h in range(1, self.m + 1):
            # Step 1: Analyst analyzes the problem and identifies core algorithms
            if h == 1:
                analyst_output = self._run_analyst(item, sample_io_prompt)
                analyst_solutions = analyst_output["analyst_results"]
                total_prompt_tokens += analyst_output["prompt_tokens"]
                total_completion_tokens += analyst_output["completion_tokens"]
                item['api_calls'] += analyst_output["api_calls"]
                
                # print("(OvO) analyst_solutions:")
                # print(analyst_solutions)
                
                # 列表存储analyst_solutions，确保可以排序
                processed_solutions = []
                
                analyst_algorithm = analyst_solutions["analysis"]
                print("(OvO) analyst_algorithm:")
                print(analyst_algorithm)
                
                for example_no, example in enumerate(analyst_solutions["problem"], start=1):
                    example['confidence'] = int(str(example['confidence']).strip())
                    processed_solutions.append(example)
                    
                # 按照confidence排序
                if processed_solutions:
                    sorted_analyst_results = sorted(
                        processed_solutions, 
                        key=lambda x: x['confidence'], 
                        reverse=True
                    )
                else:
                    print("Warning: No valid solutions with confidence found, using original solutions")
                    sorted_analyst_results = [analyst_solutions] if isinstance(analyst_solutions, dict) else analyst_solutions
                
                print("(OvO) sorted_analyst_results:")
                print(sorted_analyst_results)
                
                solution = sorted_analyst_results[0]["solution"]

            else:
                analyst_output = self._re_analyst(item, sample_io_prompt, solution, analyst_algorithm, test_log)
                analyst_solutions = analyst_output["analyst_results"]
                total_prompt_tokens += analyst_output["prompt_tokens"]
                total_completion_tokens += analyst_output["completion_tokens"]
                item['api_calls'] += analyst_output["api_calls"]
                
                # print("(OvO) Re-analyst_solutions:")
                # print(analyst_solutions)
                
                # 列表存储analyst_solutions，确保可以排序
                processed_solutions = []
                
                analyst_algorithm = analyst_solutions["analysis"]
                print("(OvO) analyst_algorithm:")
                print(analyst_algorithm)
                
                for example_no, example in enumerate(analyst_solutions["problem"], start=1):
                    example['confidence'] = int(str(example['confidence']).strip())
                    processed_solutions.append(example)
                    
                # 按照confidence排序
                if processed_solutions:
                    sorted_analyst_results = sorted(
                        processed_solutions, 
                        key=lambda x: x['confidence'], 
                        reverse=True
                    )
                else:
                    print("Warning: No valid solutions with confidence found, using original solutions")
                    sorted_analyst_results = [analyst_solutions] if isinstance(analyst_solutions, dict) else analyst_solutions
                
                print("(OvO) sorted_Re-analyst_results:")
                print(sorted_analyst_results)
                
                solution = sorted_analyst_results[0]["solution"]
            
            # 循环处理每个示例，从置信度最高的开始 
            for i, analyst_result in enumerate(sorted_analyst_results):
                print(f"\n=== PROCESSING EXAMPLE {i+1} (Confidence: {analyst_result.get('confidence', 0)}) ===\n")
                
                # Step 2: Programmer breaks down the task into sub-functions
                programmer_output = self._run_programmer(item, analyst_result, analyst_algorithm, sample_io_prompt)
                total_prompt_tokens += programmer_output["prompt_tokens"]
                total_completion_tokens += programmer_output["completion_tokens"]
                item['api_calls'] += programmer_output["api_calls"]
                
                # 保存programmer结果
                programmer_result = programmer_output["plan"]
                
                # Step 3: Generator creates initial code based on the Programmer's plan
                generator_output = self._run_generator(item, analyst_result, analyst_algorithm, programmer_result, sample_io_prompt)
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
                    plan = programmer_result
                    code = generator_result
                    for j in range(1, self.t + 1):
                        print(f"Debugging attempt {j}...")
                        debugger_output = self._run_debugger(
                            item,
                            analyst_result,
                            analyst_algorithm,
                            plan,
                            code,
                            test_log
                        )
                        total_prompt_tokens += debugger_output["prompt_tokens"]
                        total_completion_tokens += debugger_output["completion_tokens"]
                        item['api_calls'] += debugger_output["api_calls"]
                        plan = debugger_output["plan"]
                        code = debugger_output["code"]
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

    def _run_analyst(self, item: dict, sample_io_prompt: str) -> dict:
        """The Analyst analyzes the problem and identifies core algorithms.   
        Use BM25 to retrieve similar problems first, if the retrieval results are not good, then use LLM to generate examples.
        1. Analysis
        (1) Pseudocode from retrieved code
        (2) Identify the algorithm
        (3) Write a useful tutorial about the selected algorithm
        2. Solution
        (1) Problem description
        (2) Code
        (3) plan
        (4) Solution with Confidence score
        """
        print("\n=== RUNNING ANALYST ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        # 构建检索查询
        query = self.data.get_prompt(item)
        analyst_retriever_results = "0"
        
        # 使用 retriever 获取相关问题
        if self.retriever_available:
            print(f"Searching for similar problems with query: {query}...")
            try:
                # 检索问题
                retriever_output = self._coarse_grained_retriever(query, top_k=1)

                analyst_retriever_results = retriever_output.get("results", {})
                api_calls += retriever_output.get("api_calls", 0)
                prompt_tokens += retriever_output.get("prompt_tokens", 0)
                completion_tokens += retriever_output.get("completion_tokens", 0)
            except Exception as e:
                print(f"Error during problem retrieval: {str(e)}")
        
        # 检索结果部分
        print("\nAnalyst retrieved results:")
        print(analyst_retriever_results)
        
        # 使用 LLM 生成示例
        print(f"\n\n--- LLM based retrieval ---")
        
        # 构建 LLM 输入提示词
        prompt_for_analyst = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem, analyse it and then select the algorithm based on the retrieval result.
# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}

Your task:
# Task1: Analyse Problem
1. Think about the original problem: Develop an initial understanding about the problem;
2. Take the sample test cases as examples: Identify explicit and implicit requirements, cosidering how to handle edge cases.

### When you first encounters a query or task:
- First clearly rephrase the human message in your own words
- Form preliminary impressions about what is being asked
- Consider the broader context of the question
- Identify any immediate connections to relevant knowledge
- Identify any potential ambiguities that need clarification
### Multiple Hypothesis Generation
Before settling on an approach:
- Write multiple possible interpretations of the question
- Consider various solution strategyes
- Think about potential alternative perspectives
- Keep multiple working hypotheses active
- Avoid premature commitment to a single interpretation

# Task2: Based on the retrieval result, identify the algorithm needed to solve the original problem
### Retrieval result:
{analyst_retriever_results}
1. Select appropriate algorithm based on the retrieval result:
(1) If the problem is not a typical algorithmic problem: use Brute-force;
(2) If necessary, use specialized algorithms for typical algorithmic problems: Greedy, Dynamic Programming, Divide-and-conquer, Backtracking, Recursive, Binary search, DFS/BFS, stack/queue, etc.
2. Write a concise tutorial about the selected algorithm and why choose this algorithm to solve this type of problem.

# Task3: Based on analysis, recall {num_mapping[self.n]} equivalent problems with detailed descriptions (different from original problem mentioned above), then propose solution strategies for original problem prioritizing correctness. For each problem,
1. Describe the problem in detail;
2. Generate code step by step to solve the problem using the selected algorithm;
3. Generate a plan to solve the problem;
4. Provide a solution strategy for solving the original problem;
5. Provide a confidence score.

Let's think step by step to ensure the correct answer.
---------------
Important:
- Before you generate the response, note all possible valid inputs and edge cases.
- Output one analysis and {self.n} problems.
- Your response must be concise and follow the following xml format-

<root>
<analysis>
Output the selected algorithm;
Then write a concise tutorial about the selected algorithm and why choose this algorithm to solve this type of problem. Do not generate code.
</analysis>

<problem>
# Based on analysis, recall {num_mapping[self.n]} equivalent problems with detailed descriptions (different from original problem mentioned above). For each problem,
<example_description>Describe the problem.</example_description>
<example_code>Generate {self.language} code step by step to solve the problem using the selected algorithm. If the code is too long, output a short pseudocode to describe the code.</example_code>
<example_plan>Generate a plan to solve the problem.</example_plan>
<solution>Provide a high-level generic solution strategy for solving the original problem using the selected algorithm, including data structures and main tasks. Do not generate code.</solution>
<confidence>Provide a confidence score (an integer between 0 and 100) indicating how confident you are that this algorithm and solution strategy will solve the problem correctly. Do not generate extra words.</confidence>
</problem>

# similarly add more problems here...

</root>
""",
            },
        ]

        # print("--- Prompt for Analyst (LLM-based): ")
        # print(prompt_for_analyst[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_analyst
        )
        
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # Post processing
        response = self.replace_tag(response, 'analysis')
        response = self.replace_tag(response, 'example_description')
        response = self.replace_tag(response, 'example_code')
        response = self.replace_tag(response, 'example_plan')
        response = self.replace_tag(response, 'solution')
        response = self.replace_tag(response, 'confidence')
        
        # print("\n--- Analyst Response (LLM-based): ")
        # print(response, flush=True)

        analyst_results = self.parse_xml(response)
        # return analyst_results: example_description, example_code, example_plan, solution, confidence
        return {
            "analyst_results": analyst_results,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

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
- Recall {num_mapping[self.k]} similar problems that are essentially the same as the original problem;
- Analyse how to solve these problems;
- Generate a pseudocode to solve the original problem based on the analysis.
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
        
    def _re_analyst(self, item: dict, sample_io_prompt: str, solution: str, analyst_algorithm: str, test_log: str) -> dict:
        """The Analyst analyzes the problem and identifies core algorithms.
        对原问题重新理解与分析。
        """
        print("\n=== RUNNING RE-ANALYST ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        # 使用 LLM 生成示例
        print(f"\n\n--- LLM based retrieval ---")
        
        # 构建 LLM 输入提示词
        prompt_for_re_analyst = [
            {
                "role": "user",
                "content": f"""For the original problem, try to explore more solution strategies.

# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}

## Original Algorithm and Solution:
{analyst_algorithm}
{solution}

## Test Log:
{test_log}

Your task:
# Task1: Analyse Problem
1. Rethink the original problem and identify explicit and implicit requirements;
2. Analyze the transformation process from sample input to expected output, focus on the failed sample in test_log and edge cases. 

# Task2: Check the original algorithm
1. Prioritize functional correctness over efficiency: If there is a simulation method or a direct brute force method available, prefer it;
2. Take the failed sample in test_log as examples, and verify the correctness of the original solution.

# Task3: Based on analysis, recall {num_mapping[self.n]} equivalent problems with detailed descriptions (different from original problem mentioned above), then propose solution strategies for original problem prioritizing correctness. For each problem,
1. Describe the problem in detail;
2. Generate code step by step to solve the problem using the selected algorithm;
3. Generate a plan to solve the problem;
4. Provide a solution strategy for solving the original problem;
5. Provide a confidence score.

Let's think step by step to ensure the correct answer.
---------------
Important:
- Consider all possible valid inputs and edge cases.
- Output one analysis and {self.n} problems. 
- Your response must be concise and follow the following xml format-

<root>
<analysis>
Select the algorithm needed to solve the original problem again:
(1) Prefer Brute-force for simple problems;
(2) Specialized algorithms for typical algorithmic problems: Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, etc. 
- Consider more efficient methods only if they do not compromise correctness.
- Then write a concise tutorial about the selected algorithm and why choose this algorithm to solve this type of problem. Do not generate code.
</analysis>

<problem>
# Rethink the original problem, understand the requirements, recall {num_mapping[self.n]} equivalent problems with detailed descriptions (different from original problem mentioned above). For each problem,
<example_description>Describe the problem.</example_description>
<example_code>Generate {self.language} code step by step to solve the problem using the selected algorithm. If the code is too long, output a short pseudocode to describe the code.</example_code>
<example_plan>Generate a plan to solve the problem.</example_plan>
<solution>Provide a high-level generic solution strategy for solving the original problem using the selected algorithm, including data structures and main tasks. Do not generate code.</solution>
<confidence>Provide a confidence score (an integer between 0 and 100) indicating how confident you are that this algorithm and solution strategy will solve the problem correctly. Do not generate extra words.</confidence>
</problem>

# similarly add more problems here...

</root>
""",
            },
        ]

        # print("--- Prompt for Analyst (LLM-based): ")
        # print(prompt_for_re_analyst[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_re_analyst
        )
        
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # Post processing
        response = self.replace_tag(response, 'analysis')
        response = self.replace_tag(response, 'example_description')
        response = self.replace_tag(response, 'example_code')
        response = self.replace_tag(response, 'example_plan')
        response = self.replace_tag(response, 'solution')
        response = self.replace_tag(response, 'confidence')
        
        # print("\n--- Analyst Response (LLM-based): ")
        # print(response, flush=True)

        analyst_results = self.parse_xml(response)

        # return analyst_results: example_description, example_code, example_plan, solution, confidence
        return {
            "analyst_results": analyst_results,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

    def _run_programmer(self, item: dict, analyst_result: dict, analyst_algorithm: str, sample_io_prompt: str) -> dict:
        """The Programmer breaks down the task and generates a plan with confidence levels."""
        print("\n=== RUNNING PROGRAMMER ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0

        # 构建输入提示词
        prompt_for_plan = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem and a relevant example with its solution, break down the problem into sub-tasks and generate a step-by-step plan to solve the problem.
## Example Problem:
{analyst_result['example_description']}
Example plan: {analyst_result['example_plan']}

## Problem to be solved:
{self.data.get_prompt(item)}
{sample_io_prompt}
## Solution:
{analyst_algorithm}
{analyst_result['solution']}

# Your task:
1. Break down the original problem into sub-tasks and generate a detailed step-by-step plan to solve sub-tasks. Each step should be clear and specific.
2. Use the sample input to logically apply plan step by step to get the output. Compare the generated output with the sample output and improve the plan utill it works correctly.
3. For each step in the plan, also provide a confidence score (0-100) indicating how confident you are that this step can be correctly implemented in code.

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
</step>

# Similarly for other steps...

</root>
"""
            }
        ]

        print("--- Prompt for Programmer: ")
        print(prompt_for_plan[0]['content'], flush=True)

        # 获取计划
        plan_response, pr_tok, com_tok = self.gpt_chat(prompt_for_plan)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # 解析计划
        plan_response = self.replace_tag(plan_response, 'description')
        plan_response = self.replace_tag(plan_response, 'confidence')
        parsed_plan = self.parse_xml(plan_response)
        
        # 提取计划步骤和置信度
        plan_steps = []
        step_confidences = []
        
        # 从XML结构中提取步骤信息
        steps = parsed_plan.get("step", [])
        if not isinstance(steps, list):
            steps = [steps]
        
        for step_idx, step in enumerate(steps):
            if isinstance(step, dict):
                description = step.get("description", "")
                confidence = step.get("confidence", "100")
                
                # 确保confidence是整数
                try:
                    confidence = int(str(confidence).strip())
                except (ValueError, AttributeError):
                    confidence = 100
                
                plan_steps.append(description)
                step_confidences.append(confidence)
            else:
                # 如果step不是字典，可能是字符串，直接添加
                plan_steps.append(str(step))
                step_confidences.append(100)
        
        # 评估每个步骤并检索相关代码
        step_snippets = []
        
        for step_idx, (step, confidence) in enumerate(zip(plan_steps, step_confidences)):
            # 如果置信度低于confidence_threshold，检索相关代码
            if confidence < self.confidence_threshold:
                print(f"Low confidence ({confidence}) for step {step_idx+1}, retrieving code snippets...")
                retriever_result = self._fine_grained_retriever(item, step)
                api_calls += retriever_result["api_calls"]
                prompt_tokens += retriever_result["prompt_tokens"]
                completion_tokens += retriever_result["completion_tokens"]
                
                if retriever_result["snippets"]:
                    step_snippets.append({
                        "step_idx": step_idx,
                        "step": step,
                        "snippets": retriever_result["snippets"]
                    })
        
        # 为每个步骤添加相关代码片段
        enhanced_plan = ""
        for step_idx, step in enumerate(plan_steps):
            # 添加步骤
            enhanced_plan += f"Step {step_idx+1}: {step}\n"
            # 如果有相关代码片段，添加到对应步骤下
            for step_snippet in step_snippets:
                if step_snippet["step_idx"] == step_idx:
                    snippets = step_snippet["snippets"]
                    enhanced_plan += f"  Example code snippet for step: {snippets}\n"
        
        print("--- Programmer Response: ")
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

    def _run_generator(self, item: dict, analyst_result: dict, analyst_algorithm: str, programmer_result: dict, sample_io_prompt: str) -> dict:
        """The Generator creates initial code based on the Programmer's plan."""
        print("\n=== RUNNING GENERATOR ===\n")
        
        solution_prompt = f"## Solution strategy to solve the original problem:\n{analyst_result['solution']}"
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
                "content": f"""You are an expert programmer. Given the generated plan, generate {self.language} code step by step to solve the problem, and evaluate whether the code can achieve the intended functionality. If the confidence score is below 95, recall {self.k} code snippets that implement similar functionality and use them as a reference to regenerate the code.
# Problem to be solved:
{self.data.get_prompt(item)}
{sample_io_prompt}
## Algorithm:
{analyst_algorithm}
{solution_prompt}
# plan:
{plan}

Let's think step by step to ensure the correctness of the code.
---------------
{std_input_prompt}
## Important:
- Your response must contain only the {self.language} code to solve this problem.
- Confidence score regarding the solvability of the problem must be an integer between 0 and 100."""
            }
        ]

        print("--- Prompt for Generator: ")
        print(prompt_for_generator[0]['content'], flush=True)

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

    def _run_debugger(self, item: dict, analyst_result: dict, analyst_algorithm: str, plan: str, code: str, test_log: str) -> dict:
        """The Debugger corrects errors by analyzing failing test cases."""
        print("\n=== RUNNING DEBUGGER ===\n")
        
        solution_prompt = f"## Solution strategy to solve the original problem:\n{analyst_result['solution']}"
           
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
        prompt_for_debugger = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. You have generated {self.language} code to solve the given problem, but the generated code can not pass sample test cases. Check if the generated code follows the original plan, and improve your code to solve the problem correctly.

# Problem to be solved:
{self.data.get_prompt(item)}
## Algorithm:
{analyst_algorithm}
{solution_prompt}

## Buggy Plan:
{plan}
## Buggy Code:
{code}
## Test Report:
{test_log}

# Your Task:
Check if the generated code follows the original plan: Take the failed sample as the input, and logically deduce the execution process of the code. 

1. If the code does not follow the plan: improve the code to solve the problem correctly, focusing on the failed sample in test_log.
2. If the code follows the plan: recheck the original plan focusing on the failed sample in test_log
(1) Use a failed sample input to apply the plan step by step to get the output. Compare the generated output with the sample output and check if the plan works correctly;
(2) Analyse the failed test case and identify which step in the plan is incorrect: Improve the plan;
(3) Generate modified plan and code. 

Let's think step by step to modify {self.language} Code for solving this problem.
---------------
{std_input_prompt}
## Important:
- Do not add explanation.
- Output the **modified plan** and the **{self.language} code** to solve this problem.
- Your response must follow the following xml format-

<root>
<plan>Generate a modified plan.</plan>
<code>Executeable {self.language} code to solve this problem.</code>
</root>
"""
            }
        ]
        
        # print(f"--- Prompt for Debugger: ")
        # print(prompt_for_debugger[0]['content'], flush=True)

        response, prompt_tokens, completion_tokens = self.gpt_chat(
            prompt_for_debugger
        )
        api_calls = 1

        response = self.replace_tag(response, 'plan')
        response = self.replace_tag(response, 'code')
        parsed_response = self.parse_xml(response)

        # 提取改进后的plan和code
        improved_plan = parsed_response.get('plan', '')
        improved_code = parsed_response.get('code', '')

        improved_code = self.parse_code(improved_code)

        print(f"---Debugger Response: ")
        print(response, flush=True)

        return {
            "plan": improved_plan,
            "code": improved_code,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }
