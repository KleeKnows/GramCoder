# generator+doctor

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
        k: int = 2,
        m: int = 3,
        n: int = 3,
        t: int = 3,
        code_dataset_path: str = "src/methods/retriever/dataset/CSN_train.jsonl",
        relevance_threshold: float = 90,
        confidence_threshold: int = 70,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k   # Retrieved code number
        self.m = m   # Backtracing number
        self.n = n   # Solution number
        self.t = t   # Refiner times for each plan
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
        if sample_io==[] -> ""
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
        statistic_log_path = "GramCoder_statistic.log"
        # Creat Reasoning Graph
        CoderGraph = nx.DiGraph()
        CoderGraph.add_node("problem", content=self.data.get_prompt(item))
        intermediate_nodes = {}

        # Track token usage and API calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        item['api_calls'] = item.get('api_calls', 0)
        
        # Get sample_io
        sample_io = item.get('sample_io', [])
        sample_io_prompt = f"## Sample Test cases: \n{self.get_sample_io_str(sample_io)}\n"
        
        for h in range(self.m + 1):
            print(f"\n=== ATTEMPT {h} / {self.m} ===\n")
            if h == 0:
                # Step 1&2: Analyzes the problem and identifies core algorithms
                node_algorithm_output = self._generate_algorithm(item, sample_io_prompt, intermediate_nodes, CoderGraph)
                total_prompt_tokens += node_algorithm_output["prompt_tokens"]
                total_completion_tokens += node_algorithm_output["completion_tokens"]
                item['api_calls'] += node_algorithm_output["api_calls"]
            
                CoderGraph = node_algorithm_output["codergraph"]
                intermediate_nodes = node_algorithm_output["intermediate_nodes"]

                # Step 3: Proposes possible solutions
                node_solution_output = self._generate_solution(item, sample_io_prompt, CoderGraph)
                total_prompt_tokens += node_solution_output["prompt_tokens"]
                total_completion_tokens += node_solution_output["completion_tokens"]
                item['api_calls'] += node_solution_output["api_calls"]

                sorted_solutions = node_solution_output["sorted_solutions"]
                intermediate_nodes["possible_solutions"] = sorted_solutions

                print("(OoO) sorted_solutions:")
                # print(sorted_solutions)                
                # Loop through each proposed solution 
                for i, analyst_solution in enumerate(sorted_solutions):
                    print(f"\n=== PROCESSING EXAMPLE {i+1} (Confidence: {analyst_solution.get('confidence', 0)}) ===\n")
                    
                    CoderGraph.add_node("solution", content=analyst_solution)
                    CoderGraph.add_edge("algorithm", "solution")
                    intermediate_nodes["solution"] = analyst_solution['solution']
                    
                    # Step 4: Break down the task into sub-functions
                    node_plan_output = self._generate_plan(item, sample_io_prompt, analyst_solution, CoderGraph)
                    total_prompt_tokens += node_plan_output["prompt_tokens"]
                    total_completion_tokens += node_plan_output["completion_tokens"]
                    item['api_calls'] += node_plan_output["api_calls"]

                    validate_plan_output = self._validate_plan(
                        item,
                        sample_io_prompt,
                        node_plan_output["plan"],
                    )

                    validated_plan = validate_plan_output["plan"]
                    CoderGraph.add_node("plan", content=validated_plan)
                    CoderGraph.add_edge("solution", "plan")
                    
                    # Step 5: Generator creates initial code based on the Programmer's plan
                    generator_output = self._generate_code(item, sample_io_prompt, analyst_solution, CoderGraph)
                    total_prompt_tokens += generator_output["prompt_tokens"]
                    total_completion_tokens += generator_output["completion_tokens"]
                    item['api_calls'] += generator_output["api_calls"]
                    
                    generator_result = generator_output["code"]
                    # Test the generated code against sample test cases
                    passed, test_log = self.data.evaluate_sample_io(
                        item,
                        generator_result,
                        self.language
                    )
                    print("### Test log:\n", test_log)
                    # If passed, use this result
                    if passed:
                        print(f"Solution {i+1} generated code passed the tests.")
                        print("m=====:", h, "n=====:", i+1)
                        with open(statistic_log_path, "a", encoding="utf-8") as f:
                            f.write(f"m=====:{h}, n=====:{i+1}\n")

                        code = generator_result
                        break
                    else:
                        plan = CoderGraph.nodes["plan"]['content']
                        code = generator_result
                        for j in range(1, self.t + 1):
                            print(f"Debugging attempt {j}...")
                            print("### Test log:\n", test_log)
                            refiner_output = self._doctor_code_refining(
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
                                print("m=====:", h, "n=====:", i+1, "t=====:", j+1)
                                with open(statistic_log_path, "a", encoding="utf-8") as f:
                                    f.write(f"m={h}, n={i+1}, t={j+1}\n")
                                break
            else:
                backtracing_output = self._doctor_backtracing(item, sample_io_prompt, intermediate_nodes, CoderGraph)
                
                analyst_solution = backtracing_output["analyst_solution"]
                CoderGraph = backtracing_output["codergraph"]
                intermediate_nodes = backtracing_output["intermediate_nodes"]
                # Step 4: Break down the task into sub-functions
                node_plan_output = self._generate_plan(item, sample_io_prompt, analyst_solution, CoderGraph)
                total_prompt_tokens += node_plan_output["prompt_tokens"]
                total_completion_tokens += node_plan_output["completion_tokens"]
                item['api_calls'] += node_plan_output["api_calls"]

                validate_plan_output = self._validate_plan(
                    item,
                    sample_io_prompt,
                    node_plan_output["plan"],
                )

                validated_plan = validate_plan_output["plan"]
                CoderGraph.add_node("plan", content=validated_plan)
                CoderGraph.add_edge("solution", "plan")
                
                # Step 5: Generator creates initial code based on the Programmer's plan
                generator_output = self._generate_code(item, sample_io_prompt, analyst_solution, CoderGraph)
                total_prompt_tokens += generator_output["prompt_tokens"]
                total_completion_tokens += generator_output["completion_tokens"]
                item['api_calls'] += generator_output["api_calls"]
                
                generator_result = generator_output["code"]
                # Test the generated code against sample test cases
                passed, test_log = self.data.evaluate_sample_io(
                    item,
                    generator_result,
                    self.language
                )
                print("### Test log:\n", test_log)
                # If passed, use this result
                if passed:
                    print(f"Backtracing {h} generated code passed the tests.")
                    print("=====m:", h)
                    with open(statistic_log_path, "a", encoding="utf-8") as f:
                        f.write(f"m={h}\n")
                    code = generator_result
                else:
                    plan = CoderGraph.nodes["plan"]['content']
                    code = generator_result
                    for j in range(1, self.t + 1):
                        print(f"Code Refining attempt {j}...")
                        print("### Test log:\n", test_log)
                        refiner_output = self._doctor_code_refining(
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
                            print("m=====:", h, "t=====:", j+1)
                            with open(statistic_log_path, "a", encoding="utf-8") as f:
                                f.write(f"m={h}, t={j+1}\n")
                            break
            if passed:
                print("m=====:", h)
                print("(OvO) Passed the tests.")
                break
            else:
                with open(statistic_log_path, "a", encoding="utf-8") as f:
                    f.write("Failed\n")
        
        print("________________________\n\n", flush=True)
        return code, total_prompt_tokens, total_completion_tokens

    def _coarse_grained_retriever(self, query: str, top_k: int) -> dict:
        """Coarse-grained retriever: by BM25"""
        print("\n--- COARSE-GRAINED RETRIEVER ---")
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        if self.retriever_available:
            try:
                bm25_results = self.code_searcher.search(query, top_k=top_k)
                
                # filter results
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
        
        # Use step as query
        query = step
        retrieved_snippets = []
        
        if self.retriever_available:
            print(f"Searching for code snippets for step: {query}...")
            try:
                bm25_results = self.code_searcher.search(query, top_k=self.k)
                
                # Filter results
                for result in bm25_results:
                    if result['score'] >= self.relevance_threshold:
                        retrieved_snippets.append({
                            "description": f"Function: {result['func_name']} - {result['docstring']}",
                            "code": result['code']
                        })
                
                print(f"Found {len(retrieved_snippets)} relevant code snippets with BM25")
                
            except Exception as e:
                print(f"Error in BM25 search: {str(e)}")
        
        # Use LLM to generate
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
        print("\n=== RUNNING GENERATOR_ALGORITHM ===\n")
        
        prompt_tokens = 0
        completion_tokens = 0
        api_calls = 0
        
        # Prompt
        prompt_for_generate_analysis = [
            {
                "role": "user",
                "content": f"""You are an experienced competitive programming coach. For a algorithmic problem, perform a structured analysis that goes through: Precise Understanding → Initial Algorithm Judgment. Think according to the following structure:
# Original Problem:
{self.data.get_prompt(item)}
{sample_io_prompt}

1. Problem Understanding
(1) Problem Restatement & Core Requirements Summary:
Clearly rephrase the human message in your own words, ensuring a thorough understanding of the original problem statement.
- Detailed Explanation of Original Problem: Elaborate on the given problem in your own words, explaining its components, context, and any inherent complexities. Do not omit any constraints or details present in the original description.
- Core Requirements Summary: Summarize the essential requirements and the fundamental nature of the problem. 
(2) Consider Sample Test cases, improve understanding:
- Formalize input/output, variable definitions, objective function/conditions
- Key considerations and constraints
- Identify any potential ambiguities or unclear aspects within the problem description that need clarification

2. Problem Classification
Categorize the original problem as either a 'simple problem' or a 'complex problem'. 

3. Recall {num_mapping[self.k]} relevant methods to solve the problem
## For simple problem:
select basic algorithms (e.g., brute force, simulation, direct implementation);
## For complex problem:
select appropriate advanced algorithms, such as:
(1) Search:
- Binary Search: On sorted sequences, on answer space
- Two Pointers or Sliding Window: Optimal range, substring counting, shortest covering
- BFS or DFS
(2) Greedy Algorithms: locally optimal choices to achieve a global optimum
(3) Recursion & Backtracking
(4) Divide and Conquer
(5) Dynamic Programming (DP): Basic DP; Memoization; Bitmask DP; Tree DP; 
(6) Math Techniques, and so on

**Data Structure-Specific Techniques**
(1) Linear Structures: Stack; Queue
(2) Graph Algorithms
- Graph Traversal: Breadth-First Search (BFS); Depth-First Search (DFS)
- Shortest Path Algorithms: Dijkstra's Algorithm (Single Source, Non-negative weights); Bellman-Ford / SPFA (Single Source, Negative weights); Floyd-Warshall (All-Pairs Shortest Path); 0-1 BFS (Edge weights 0 or 1)
- Minimum Spanning Tree (MST): Prim's Algorithm; Kruskal's Algorithm
- Network Flow
- Topological Sort
- Bipartite Graph: Bipartite Matching(Hungarian Algorithm, Network Flow)
(3) String Algorithms
- Pattern Matching: KMP Algorithm (Knuth-Morris-Pratt); Z-Algorithm; Rabin-Karp Algorithm (String Hashing); Manacher's Algorithm (Longest Palindromic Substring)
- Advanced String Structures: Suffix Array; Suffix Automaton (SAM); Suffix Tree (ST)
(4) Tree-Specific Techniques
- Binary Search Tree (BST): Balanced BSTs (AVL, Treap, Splay, Red-Black Tree)
- Segment Tree: Value Segment Tree
- Fenwick Tree (Binary Indexed Tree - BIT)
- Trie (Prefix Tree)
- Disjoint Set Union (DSU / Union-Find)
- Mergeable Heap (Leftist Heap, Skew Heap)
and so on.

4. Algorithm Selection
(1) Considering the problem's requirements, choose suited algorithms. Prioritize functional correctness over efficiency.
- Briefly explain why the algorithms well suited for the originalproblem.
- Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem.
---------------
Important:
- Think step by step to ensure the correct answer. Before outputting, carefully review each part to ensure accuracy.
- Your response must follow the following xml format-
<root>
<restatement>
Detailed Explanation of Original Problem;
Concise summary of the problem's core requirements and essential nature in your own words.
</restatement>
<analysis>
Provide problem categorization for the original problem.
</analysis>
<example>
Recall {num_mapping[self.k]} methods to solve problem.
</example>
</algorithm>
Select algorithms to solve the original problem and write a concise tutorial about the selected algorithm.
<algorithm>
</root>
""",
            },
        ]

        print("--- Prompt for Analyse: ")
        print(prompt_for_generate_analysis[0]['content'], flush=True)

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_generate_analysis
        )
        
        print("\n--- Analyse Response: ")
        print(response)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok

        # Post processing
        response = self.replace_tag(response, 'restatement')
        response = self.replace_tag(response, 'analysis')
        response = self.replace_tag(response, 'example')
        response = self.replace_tag(response, 'algorithm')
        parsed_response = self.parse_xml(response)
        
        restatement = parsed_response.get('restatement', '')
        analysis = parsed_response.get('analysis', '')
        example = parsed_response.get('example', '')
        algorithm = parsed_response.get('algorithm', '')

        # Post processing
        CoderGraph.add_node("restatement", content=restatement)
        CoderGraph.add_edge("problem", "restatement")
        CoderGraph.add_node("analysis", content=analysis)
        CoderGraph.add_edge("restatement", "analysis")
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
        print("\n=== RUNNING GENERATOR_SOLUTION ===\n")
        
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
{CoderGraph.nodes["restatement"]['content']}
{CoderGraph.nodes["algorithm"]['content']}

# Your task:
Based on the selected algorithm, recall {num_mapping[self.n]} solution approaches with relevant code snippets that leverage this algorithm to solve the original problem. For each possible approach:
1. Recall {self.language} code draft to solve the problem;
2. Generate a step-by-step plan base the example code;
3. Based on the example, generate a high-level generic solution strategy for solving the original problem using the selected algorithm, including:
- Core idea
- Key tasks and relevant data structures
4. Reflect on your solution, provide a confidence score of the solution.
---------------
Important:
- Think step by step to ensure the correct answer. Before outputting, carefully review each part to ensure accuracy.
- Output {num_mapping[self.n]} solutions. Prioritize direct methods and simple data structures to ensure correctness.
- Your response must follow the following xml format:

<root>
# Recall {num_mapping[self.n]} distinct solution approaches with code snippets, then provide high-level approaches. Write each approach in the following format:
<approach>
<example_code>Recall relevant code to solve the problem.</example_code>
<example_plan>Generate example plan.</example_plan>
<solution>Provide a high-level generic solution strategy for solving the original problem using the selected algorithm, including core idea, key tasks and data structure.</solution>
<confidence>Based on the selected algorithm and the solution, provide a **integer** type confidence score (0-100) indicating how confident you are that this algorithm and solution strategy will solve the problem correctly. Do not generate extra words.</confidence>
</approach>

# similarly add more approaches here...

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
        response = self.replace_tag(response, 'example_code')
        response = self.replace_tag(response, 'example_plan')
        response = self.replace_tag(response, 'solution')
        response = self.replace_tag(response, 'confidence')
        parsed_response = self.parse_xml(response)

        # Store analyst_solutions as list
        processed_solutions = []

        for example_no, example in enumerate(parsed_response["approach"], start=1):
            example['confidence'] = int(str(example['confidence']).strip())
            processed_solutions.append(example)
            
        # Sort by confidence
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

        # Prompt
        prompt_for_plan = [
            {
                "role": "user",
                "content": f"""You are an expert programmer. Given a problem and its solution, break down the problem into sub-tasks and generate a step-by-step plan to solve the problem.
# Problem to be solved:
{self.data.get_prompt(item)}
{sample_io_prompt}

{CoderGraph.nodes["restatement"]['content']}
# Example plan: {analyst_solution['example_plan']}
---------------
Solution:
{CoderGraph.nodes["algorithm"]['content']}
{analyst_solution['solution']}

# Your task:
1. Based on the solution, break down the original problem into sub-tasks and generate a detailed step-by-step plan to solve it. Each step should be clear and specific.
2. Consider all possible valid inputs. Use possible inputs to logically apply plan step by step to get the output. Improve the plan utill it works correctly.

Let's think step by step to ensure the correct answer. You should give only the plan to solve the problem. Do not add extra explanation or words.
"""
            }
        ]

        enhanced_plan, pr_tok, com_tok = self.gpt_chat(prompt_for_plan)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok
        
        print("--- plan Response: ")
        print(enhanced_plan, flush=True)
        
        return {
            "plan": enhanced_plan,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }
    
    def _validate_plan(self, item: dict, sample_io_prompt:str, plan: str) -> dict:
        """Checks if the plan is likely to solve the problem."""
        print("\n=== RUNNING PLAN_REFINEMENT ===\n")
        
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
1. Anslyse whether the plan is correct to solve this problem.
2. For public test cases:
(1) Use public test sample inputs to logically apply the plan step by step to get the output, and analyse whether each step can achieve correct function. 
(2) Compare the generated logical results with the expected output and check if the plan works correctly:
- If not, analyse which step in the plan is incorrect, and apdate the plan until the problem can be solved correctly.
3. Analyse all possible valid inputs and improve the plan to handle all possible valid inputs correctly.
4. Add relevant code snippets for hard steps in the plan if necessary.
For each step in the plan, provide a confidence score (0-100) indicating how confident you are that this step can be correctly implemented in code.
- If the score is below {self.confidence_threshold}, recall how to implement similar functionality, then generate a reference code snippet that implement this step.
---------------
Important:
- Think step by step to ensure the correct answer.
- Your response must contain only the **refined plan with reference code snippets for hard steps**. 
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
    
    def _generate_code(self, item: dict, sample_io_prompt: str, analyst_solution: dict, CoderGraph) -> dict:
        """The Generator creates initial code based on the Programmer's plan."""
        print("\n=== RUNNING GENERATOR_CODE ===\n")
        
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
Solution:
{algorithm}
{solution_prompt}

# plan:
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
    
    def _doctor_backtracing(self, item: dict, sample_io_prompt: str, intermediate_nodes:dict, CoderGraph) -> dict:
        """Backtracing.
        """
        print("\n=== RUNNING DOCTOR_BACKTRACING ===\n")
        
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
## Test Log:
{intermediate_nodes.get("test_log", "")}

# Your task:
Phase 1: Critical Retrospective and Error Location Analysis
1. Re-evaluate the Original Problem and Requirements
Thoroughly re-analyze the original problem. Identify both explicit and implicit requirements, constraints, and edge cases. Ensure your understanding of the problem's core nature is accurate and complete.
2. Diagnose Previous Attempts
(1) Diagnose Previous Algorithm Chosen:
## Previous Algorithm:
{CoderGraph.nodes["algorithm"]['content']}
Analyze whether the previous algorithm might have been unsuitable. Consider if it fundamentally misaligned with the problem's requirements, was too complex/inefficient, or overlooked simpler, more direct solutions. **Determine if the error lies in the algorithm's fundamental choice.**
(2) Diagnose Previous Solution Approach Failures:
## Previous Solution:
{intermediate_nodes.get("solution", "")}
## Previous Plan:
{CoderGraph.nodes["plan"]['content']}
The previous solution failed test cases. **Analyze the specific reasons for this failure based on the test log.** This might involve:
- Misinterpretation of an algorithm step.
- Incorrect handling of edge cases.
- Subtle logical flaws.
- Any other discrepancies with the problem's constraints.
**Identify the specific step or reasoning in the *solution approach* where the error most likely occurred.**

Phase 2: Categorize Error Source
Based on the problem re-evaluation and the failure of the previous attempt, identify the primary source of error. Categorize the error as one of the following:
A. Problem Misunderstanding: The core requirements, constraints, or nuances of the original problem were misinterpreted.
B. Algorithm Selection Error
C. Solution Approach Error: The high-level approach for applying the chosen algorithm was flawed, leading to incorrect logic or failure to handle specific scenarios (e.g., edge cases, wrong specific data structures).
D. Plan Error: The solution logic was sound, but there were subtle mistakes in the detailed plan steps, causing the failure. 

Phase 3: Final Algorithm and Solution Redesign
(1) Propose Final Algorithm:
Based on the re-evaluation in Phase 1, **select the final, most appropriate algorithm you would implement.** 
- Prioritize functional correctness and simplicity of implementation over raw efficiency at this stage. Justify your choice by explaining how it addresses the identified issues and fits the problem's requirements.
- Write a useful tutorial about the above mentioned algorithms to solve this type of problem.
(2) Generate Example Plan:
Based on the chosen final algorithm, generate an example step-by-step plan to solve the problem using the algorithm. This approach should explicitly address and rectify the identified failure points from the previous attempt. 
(2) Outline New Solution Approach:
**Based on the chosen final algorithm, outline a new, high-level solution approach.** Focus on clarity and correctness.

---------------
Important:
- Think step by step to ensure the correct answer. Before outputting, carefully review each part to ensure accuracy.
- Your response must follow the following xml format-
<root>
<analysis>
# Error Location Analysis
Based on the problem re-evaluation and the failure of the previous attempt, identify the primary source of error. Provide a detailed explanation for your chosen error category and elaborate on the specific reasons for failure.
</analysis>
<algorithm>
Select the final, most appropriate algorithm, and write a concise tutorial about the selected algorithm.
</algorithm>
<example_plan>
Provide an example step-by-step plan to implement the algorithm for the problem.
</example_plan>
<solution>
Outline a new, high-level solution approach to solve the original problem.
</solution>
</root>
""",
            },
        ]

        response, pr_tok, com_tok = self.gpt_chat(
            processed_input=prompt_for_generate_algorithm
        )
        
        print("\n--- Backtracing Response: ")
        print(response)
        api_calls += 1
        prompt_tokens += pr_tok
        completion_tokens += com_tok
        
        # Post processing
        response = self.replace_tag(response, 'analysis')
        response = self.replace_tag(response, 'algorithm')
        response = self.replace_tag(response, 'example_plan')
        response = self.replace_tag(response, 'solution')
        parsed_response = self.parse_xml(response)

        analysis = parsed_response.get('analysis', '')
        algorithm = parsed_response.get('algorithm', '')
        analyst_solution = {}
        analyst_solution["example_plan"] = parsed_response.get('example_plan', '')
        analyst_solution["solution"] = parsed_response.get('solution', '')

        CoderGraph.add_node("analysis", content=analysis)
        CoderGraph.add_node("algorithm", content=algorithm)
        CoderGraph.add_node("solution", content=analyst_solution)
        intermediate_nodes["solution"] = parsed_response.get('solution', '')

        return {
            "analyst_solution": analyst_solution,
            "intermediate_nodes": intermediate_nodes,
            "codergraph": CoderGraph,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls,
        }

    def _doctor_code_refining(self, item: dict, analyst_solution: dict, plan: str, code: str, test_log: str) -> dict:
        """The refiner corrects errors by analyzing failing test cases."""
        print("\n=== RUNNING DOCTOR_REFINING ===\n")
        
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
                "content": f"""You are an expert programmer. You have generated {self.language} code to solve the given problem, but the generated code can not pass sample test cases. Check if the generated code follows the original plan, and analyse whether the plan needs improvement.

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
Check if the generated code follows the original plan by logically deducing the execution process of the code taking the failed sample as the input. 
1. If the code does not follow the plan: 
- Identify the specific parts of the code that contain logical errors or incorrect functional functions in the code. 
- Provide a step-by-step approach to improve code.
2. If the code follows the plan: recheck the original plan focusing on the failed sample in test_log
(1) Use a failed sample input to apply the plan step by step to get the output. Compare the generated output with the sample output, analyse the failed test case and identify which step in the plan is incorrect;
(2) If a step in the plan is incorrect, think step by step to improve the plan.
(3) Try other methods to solve the problem if the plan is hard to be improved.

Let's think step by step to improve the plan to solve the problem correctly.
---------------
Important:
{std_input_prompt}
- Output the explanation and **modified plan**.
- Your response must follow the the following xml format-
<root>
<explain>
(1) Explain the reason why the code failed to pass the test cases, and analyse whether the plan needs improvement.
(2) Provide a step-by-step approach to fix the issues:
- How to improve the plan.
- How to modify the code ensuring the corrected code can handle all valid inputs correctly. 
- Keep the fix steps concise for minor issues, but provide more detailed steps if major revisions are required.
</explain>
<plan>
Provide the improved plan or remain original plan to solve the problem correctly.
</plan>
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

        response = self.replace_tag(response, 'explain')
        response = self.replace_tag(response, 'plan')
        parsed_response = self.parse_xml(response)

        # get refined plan and code
        explanation = parsed_response.get('explain', '')
        improved_plan = parsed_response.get('plan', '')

        print(f"--- Code Refiner Response: ")
        print(response, flush=True)

        prompt_for_code_refiner = [
            {
                "role": "user",
                "content": f"""You are given a coding problem:
{self.data.get_prompt(item)}
{solution_prompt}
### Buggy Code:
{code}
However, the code above failed to produce the expected output:
### Test Report:
{test_log}

# Your Task: Fix the {self.language} code using the following approach:
{explanation}
Follow the improved plan below to rewrite the code step by step so that it can handle all valid inputs correctly:
{improved_plan}

---------------
Important:
{std_input_prompt}
- Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words.
"""
            }
        ]
        
        # print(f"--- Prompt for refiner: ")
        # print(prompt_for_refiner_check[0]['content'], flush=True)

        code, prompt_tokens, completion_tokens = self.gpt_chat(
            prompt_for_code_refiner
        )
        api_calls += 1

        improved_code = self.parse_code(code)

        print(improved_code, flush=True)

        return {
            "plan": improved_plan,
            "code": improved_code,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_calls": api_calls
        }

# 1216: combine comprehension and algorithm