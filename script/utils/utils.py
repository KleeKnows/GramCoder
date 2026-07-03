import json
import re
import tiktoken

# Read an jsonl file and convert it into a python list of dictionaries.
def read_jsonl(filename):
    """Reads a jsonl file and yields each line as a dictionary"""
    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lines.append(json.loads(line))
    return lines

# Write a python list of dictionaries into a jsonl file
def write_jsonl(filename, lines):
    """Writes a python list of dictionaries into a jsonl file"""
    with open(filename, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")

"""
Retriving the code blocks from the response.
"""
def parse_response(response: str) -> str:
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

    return code_str


"""
Taking plan in numbered list format and return the list of plans.
Exmaple:
Input:
1. Plan A
2. Plan B
3. Plan C
Output: ["Plan A", "Plan B", "Plan C"]
"""
def extract_plans(planing: str) -> list[str]:
    plans = []
    for line in planing.strip().split("\n"):
        splits = line.split(". ")
        if len(splits) < 2:
            continue
        if splits[0].isnumeric():
            plans.append(splits[1])

    return plans

# token_count.py
def token_count(messages, model="gpt-3.5-turbo"):  
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    # if model == "gpt-3.5-turbo" or model == "gpt" or model == "gpt-35-instant":
    #     # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
    #     return token_count(messages, model="gpt-3.5-turbo-0301")
    # elif model == "gpt-4":
    #     # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
    #     return token_count(messages, model="gpt-4-0314")

    # if "gpt-3.5" in model:
    #     # every message follows <|start|>{role/name}\n{content}<|end|>\n
    #     tokens_per_message = 4
    #     tokens_per_name = -1  # if there's a name, the role is omitted
    # elif "gpt-4" in model:
        # tokens_per_message = 3
        # tokens_per_name = 1
    # else:
    #     raise NotImplementedError(
    #         f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    tokens_per_message = 4
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens