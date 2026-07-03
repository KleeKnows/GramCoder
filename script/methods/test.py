from .base import BaseMethod, Results
from models.base import BaseModel

from datasets.base import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.HumanEval import HumanEvalDataset
from datasets.MBPP import MBPPDataset
from datasets.CodeContestDataset import CodeContestDataset
from datasets.MBPP_evaluate import evaluate_io

class Test(BaseMethod):
    def run_single_pass(self, item: dict):
        code = """
import sys
from collections import defaultdict, deque

input = sys.stdin.read
def solve():
    data = input().split()
    index = 0
    
    n = int(data[index])
    index += 1
    q = int(data[index])
    index += 1
    
    illusion_rates = [0] * (n + 1)
    for i in range(1, n + 1):
        illusion_rates[i] = int(data[index])
        index += 1
    
    # Construct the tree
    adjacency_list = defaultdict(list)
    for _ in range(n - 1):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    # Preprocessing with BFS to find parent and depth
    parent = [-1] * (n + 1)
    depth = [0] * (n + 1)
    
    def bfs(root):
        queue = deque([root])
        visited = [False] * (n + 1)
        visited[root] = True
        while queue:
            node = queue.popleft()
            for neighbor in adjacency_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1
                    queue.append(neighbor)
    
    bfs(1)
    
    # Function to find the path from u to v
    def find_path(u, v):
        path = []
        while u != v:
            if depth[u] > depth[v]:
                path.append(u)
                u = parent[u]
            else:
                path.append(v)
                v = parent[v]
        path.append(u)
        return path
    
    # Process queries
    results = []
    for _ in range(q):
        query_type = int(data[index])
        index += 1
        u = int(data[index])
        index += 1
        if query_type == 1:
            # Update query
            c = int(data[index])
            index += 1
            illusion_rates[u] = c
        elif query_type == 2:
            # Path query
            v = int(data[index])
            index += 1
            path = find_path(u, v)
            total_energy = 0
            for i in range(len(path) - 1):
                x = path[i]
                y = path[i + 1]
                total_energy += max(abs(illusion_rates[x] + illusion_rates[y]), abs(illusion_rates[x] - illusion_rates[y]))
            results.append(total_energy)
    
    # Output results
    for result in results:
        print(result)

solve()
"""
        passed, test_log = self.data.evaluate_sample_io(
            item,
            code,
            self.language
        )
        print("Sample"+str(passed))
        print(test_log)
        return code, 0, 0