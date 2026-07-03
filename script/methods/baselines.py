from typing import List
import tiktoken
import os
import re
from copy import deepcopy

from .base import BaseMethod, Results
from models.base import BaseModel
# Dataset
from datasets.base import Dataset
from datasets.HumanEval import HumanEvalDataset
from datasets.CodeContestDataset import CodeContestDataset
from datasets.APPSDataset import APPSDataset

class DirectStrategy(BaseMethod):
    def run_single_pass(self, item: dict):
        processed_input = [
            {
                "role": "user",
                "content": f'{self.data.get_prompt(item)}\n\Generate {self.language} code to solve the above mentioned problem.\nYour response must contain only the code, do not add any other text or comments.',
            },
        ]
        print("processed_input:", processed_input)
        return self.gpt_chat(processed_input=processed_input)

class DirectStrategy_plus(BaseMethod):
    def run_single_pass(self, item: dict):
        processed_input = [
            {
                "role": "user",
                "content": f"""{self.data.get_prompt(item)}
Generate {self.language} code step by step to solve the problem, and evaluate whether the code can achieve the intended functionality. If the confidence score is below 95, recall two code snippets that implement similar functionality and use them as a reference to regenerate the code.

Let's think step by step to ensure the correct answer.
---------------
Important:
- Before you generate the response, note all possible valid inputs and edge cases.
- Your response must contain only the code to solve the problem.
- Do not add any other text or comments.
""",
            },
        ]
        return self.gpt_chat(processed_input=processed_input)

class CoTStrategy(BaseMethod):
    def run_single_pass(self, item: dict):
        prompt = None
        if type(self.data) == HumanEvalDataset:
            prompt = """
def encrypt(s):
    '''
    Create a function encrypt that takes a string as an argument and
    returns a string encrypted with the alphabet being rotated. 
    The alphabet should be rotated in a manner such that the letters 
    shift down by two multiplied to two places.
    For example:
    encrypt('hi') returns 'lm'
    encrypt('asdfghjkl') returns 'ewhjklnop'
    encrypt('gf') returns 'kj'
    encrypt('et') returns 'ix'
    '''
    # Let's think step by step.

    # Define the alphabet as a string
    d = 'abcdefghijklmnopqrstuvwxyz'
    
    # Initialize an empty string to store the encrypted result
    out = ''
    
    # Iterate through each character in the input string
    for c in s:
        # Check if the character is a letter in the alphabet
        if c in d:
            # Find the index of the current letter in the alphabet
            index = d.index(c)
            
            # Rotate the alphabet by two multiplied to two places
            # Use modulo 26 to handle wrapping around the alphabet
            rotated_index = (index + 2 * 2) % 26
            
            # Append the encrypted letter to the result string
            out += d[rotated_index]
        else:
            # If the character is not a letter, append it unchanged to the result string
            out += c
    
    # Return the final encrypted string
    return out


-------

def check_if_last_char_is_a_letter(txt):
    ''' 
    Create a function that returns True if the last character of a given string is an alphabetical character and is not a part of a word, and False otherwise. Note: 'word' is a group of characters separated by space.
    Examples:
    check_if_last_char_is_a_letter('apple pie') → False
    check_if_last_char_is_a_letter('apple pi e') → True
    check_if_last_char_is_a_letter('apple pi e ') → False
    check_if_last_char_is_a_letter('') → False
    '''    
    # Let's think step by step.

    # Split the input string by space to get a list of words
    words = txt.split(' ')
    
    # Extract the last word from the list
    last_word = words[-1]
    
    # Check if the last character of the last word is an alphabetical character
    # and is not part of a longer word (length of last word is 1)
    is_last_char_letter = len(last_word) == 1 and (97 <= ord(last_word.lower()) <= 122)
    
    # Return True if the conditions are met, False otherwise
    return is_last_char_letter

-------

def file_name_check(file name):
    '''
    Create a function which takes a string representing a file's name, and returns 'Yes' if the the file's name is valid, and returns 'No' otherwise. A file's name is considered to be valid if and only if all the following conditions are met: - There should not be more than three digits ('0'-'9') in the file's name. - The file's name contains exactly one dot '.' - The substring before the dot should not be empty, and it starts with a letter from the latin alphapet ('a'-'z' and 'A'-'Z'). - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    Examples:
    file_name_check('example.txt') => 'Yes'
    file_name_check('1example.dll') => 'No' (the name should start with a latin alphapet letter)
    '''    
    # Let's think step by step.

    # Define a list of valid file extensions
    valid_suffixes = ['txt', 'exe', 'dll']
    
    # Split the file name into two parts using the dot as a separator
    parts = file_name.split(sep='.')
    
    # Check if there are exactly two parts after splitting
    if len(parts) != 2:
        return 'No'
    
    # Check if the second part (suffix) is in the list of valid suffixes
    if not parts[1] in valid_suffixes:
        return 'No'
    
    # Check if the first part (prefix) is not empty and starts with a letter from the Latin alphabet
    if len(parts[0]) == 0 or not parts[0][0].isalpha():
        return 'No'
    
    # Count the number of digits in the prefix and check if it's not more than three
    num_digits = len([x for x in parts[0] if x.isdigit()])
    if num_digits > 3:
        return 'No'
    
    # If all conditions are met, return 'Yes', indicating a valid file name
    return 'Yes'

-------

def fruit_distribution(s,n):
    '''
    In this task, you will be given a string that represents a number of apples and oranges that are distributed in a basket of fruit this basket contains apples, oranges, and mango fruits. Given the string that represents the total number of the oranges and apples and an integer that represent the total number of the fruits in the basket return the number of the mango fruits in the basket.
    for examble:
    fruit_distribution('5 apples and 6 oranges', 19) = 19 - 5 - 6 = 8
    fruit_distribution('0 apples and 1 oranges',3) = 3 - 0 - 1 = 2
    fruit_distribution('2 apples and 3 oranges', 100) = 100 - 2 - 3 = 95
    fruit_distribution('100 apples and 1 oranges',120) = 120 - 100 - 1 = 19
    '''
    # Let's think step by step.
    
    # Initialize an empty list to store the numeric values (apples and oranges) extracted from the string
    lis = list()

    # Split the input string by space and iterate through each word
    for i in s.split(' '):
        # Check if the word is a numeric value (digit)
        if i.isdigit():
            # Convert the numeric value to an integer and append it to the list
            lis.append(int(i))

    # Calculate the number of mango fruits by subtracting the sum of apples and oranges from the total
    return n - sum(lis)

-------

def prime_fib(n: int):
    '''
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    Examples:
    >>> prime_fib(1) 2
    >>> prime_fib(2) 3
    >>> prime_fib(3) 5
    >>> prime_fib(4) 13
    >>> prime_fib(5) 89
    '''
    # Let's think step by step.

    # Import the math module for the square root function
    import math
    
    # Define a helper function to check if a number is prime
    def is_prime(p):
        if p < 2:
            return False
        for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
            if p % k == 0:
                return False
        return True

    # Initialize the Fibonacci sequence with the first two numbers
    f = [0, 1]
    
    # Continue generating Fibonacci numbers until finding the n-th prime Fibonacci number
    while True:
        f.append(f[-1] + f[-2])
        
        # Check if the latest Fibonacci number is prime
        if is_prime(f[-1]):
            n -= 1
        
        # If the desired n-th prime Fibonacci number is found, return it
        if n == 0:
            return f[-1]

"""


        if type(self.data) == APPSDataset:
            prompt = """
An accordion is a string (yes, in the real world accordions are musical instruments, but let's forget about it for a while) which can be represented as a concatenation of: an opening bracket (ASCII code $091$), a colon (ASCII code $058$), some (possibly zero) vertical line characters (ASCII code $124$), another colon, and a closing bracket (ASCII code $093$). The length of the accordion is the number of characters in it.

For example, [::], [:||:] and [:|||:] are accordions having length $4$, $6$ and $7$. (:|:), {:||:}, [:], ]:||:[ are not accordions. 

You are given a string $s$. You want to transform it into an accordion by removing some (possibly zero) characters from it. Note that you may not insert new characters or reorder existing ones. Is it possible to obtain an accordion by removing characters from $s$, and if so, what is the maximum possible length of the result?


-----Input-----

The only line contains one string $s$ ($1 \le |s| \le 500000$). It consists of lowercase Latin letters and characters [, ], : and |.


-----Output-----

If it is not possible to obtain an accordion by removing some characters from $s$, print $-1$. Otherwise print maximum possible length of the resulting accordion.


-----Examples-----
Input
|[a:b:|]

Output
4

Input
|]:[|:]

Output
-1

2
1 2

## Let's think step by step and generate code to solve the problem.

```python
# Take user input and assign it to the variable 's'
s = input()

# Calculate the length of the string 's' and assign it to the variable 'n'
n = len(s)

# Initialize variables to store the indices of '[' and ']'
ind = -1
bind = -1

# Variable to track whether '[' or ']' characters have been encountered
f = False

# Step 1: Find the index of the first '[' character after encountering ':'
for i in range(n):
    if s[i] == '[':
        f = True
    elif s[i] == ':':
        if f:
            ind = i
            break

# Reset the flag variable
f = False

# Step 2: Find the index of the last ']' character before encountering ':'
for i in range(n - 1, -1, -1):
    if s[i] == ']':
        f = True
    elif s[i] == ':':
        if f:
            bind = i
            break

# Check conditions to determine if it's possible to obtain an accordion
if ind == -1 or bind == -1:
    # Print -1 if '[' or ']' characters were not found
    print(-1)
elif ind >= bind:
    # Print -1 if the order of '[' and ']' characters is incorrect
    print(-1)
else:
    # Initialize the length of the accordion to 4 (opening and closing brackets, and two colons)
    ans = 4
    # Step 3: Count the number of '|' characters between '[' and ']' indices (inclusive)
    for i in range(ind + 1, bind):
        if s[i] == '|':
            ans += 1
    # Print the calculated length of the resulting accordion
    print(ans)
```

-------

Anton has the integer x. He is interested what positive integer, which doesn't exceed x, has the maximum sum of digits.

Your task is to help Anton and to find the integer that interests him. If there are several such integers, determine the biggest of them. 


-----Input-----

The first line contains the positive integer x (1 ≤ x ≤ 10^18) — the integer which Anton has. 


-----Output-----

Print the positive integer which doesn't exceed x and has the maximum sum of digits. If there are several such integers, print the biggest of them. Printed integer must not contain leading zeros.


-----Examples-----
Input
100

Output
99

Input
48

Output
48

Input
521

Output
499

## Let's think step by step and generate code to solve the problem.

```python
# Take user input as a list of integers representing the digits of the given positive integer x
num = list(map(int, input()))

# Initialize a variable 'best' to store the current best candidate for the maximum sum of digits
best = num[:]

# Step 1: Iterate over the digits in reverse order
for i in range(-1, -len(num) - 1, -1):
    # If the current digit is 0, continue to the next digit
    if num[i] == 0:
        continue
    
    # Decrease the current digit by 1
    num[i] -= 1

    # Step 2: Set all digits to the right of the current digit to 9
    for j in range(i + 1, 0):
        num[j] = 9
    
    # Step 3: Compare the sum of the current number with the sum of the best number
    if sum(num) > sum(best):
        # Update 'best' if the current number has a greater sum
        best = num[:]

# Convert the list of digits to a string, remove leading zeros, and print the result
s = ''.join(map(str, best)).lstrip('0')
print(s)
```

-------

Apart from having lots of holidays throughout the year, residents of Berland also have whole lucky years. Year is considered lucky if it has no more than 1 non-zero digit in its number. So years 100, 40000, 5 are lucky and 12, 3001 and 12345 are not.

You are given current year in Berland. Your task is to find how long will residents of Berland wait till the next lucky year.


-----Input-----

The first line contains integer number n (1 ≤ n ≤ 10^9) — current year in Berland.


-----Output-----

Output amount of years from the current year to the next lucky one.


-----Examples-----
Input
4

Output
1

Input
201

Output
99

Input
4000

Output
1000



-----Note-----

In the first example next lucky year is 5. In the second one — 300. In the third — 5000.

## Let's think step by step and generate code to solve the problem.

```python
def main():
    # Take user input as a string representing the current year in Berland
    s = input()

    # Get the length of the input string (number of digits in the current year)
    n = len(s)

    # Step 1: Create a target lucky year 't' by incrementing the first digit by 1 and padding with zeros
    t = int(str(int(s[0]) + 1) + '0' * (n - 1))

    # Step 2: Calculate the number of years from the current year to the next lucky one
    result = t - int(s)

    # Print the result
    print(result)

# Call the main function
main()
```
"""

        if type(self.data) == CodeContestDataset:
            prompt = """
Three little pigs from all over the world are meeting for a convention! Every minute, a triple of 3 new pigs arrives on the convention floor. After the n-th minute, the convention ends.

The big bad wolf has learned about this convention, and he has an attack plan. At some minute in the convention, he will arrive and eat exactly x pigs. Then he will get away.

The wolf wants Gregor to help him figure out the number of possible attack plans that involve eating exactly x pigs for various values of x (1 ≤ x ≤ 3n). Two attack plans are considered different, if they occur at different times or if the sets of little pigs to eat are different.

Note that all queries are independent, that is, the wolf does not eat the little pigs, he only makes plans!

Input

The first line of input contains two integers n and q (1 ≤ n ≤ 10^6, 1 ≤ q ≤ 2⋅ 10^5), the number of minutes the convention lasts and the number of queries the wolf asks.

Each of the next q lines contains a single integer x_i (1 ≤ x_i ≤ 3n), the number of pigs the wolf will eat in the i-th query.

Output

You should print q lines, with line i representing the number of attack plans if the wolf wants to eat x_i pigs. Since each query answer can be large, output each answer modulo 10^9+7.

Examples

Input


2 3
1
5
6


Output


9
6
1


Input


5 4
2
4
6
8


Output


225
2001
6014
6939

Note

In the example test, n=2. Thus, there are 3 pigs at minute 1, and 6 pigs at minute 2. There are three queries: x=1, x=5, and x=6.

If the wolf wants to eat 1 pig, he can do so in 3+6=9 possible attack plans, depending on whether he arrives at minute 1 or 2.

If the wolf wants to eat 5 pigs, the wolf cannot arrive at minute 1, since there aren't enough pigs at that time. Therefore, the wolf has to arrive at minute 2, and there are 6 possible attack plans.

If the wolf wants to eat 6 pigs, his only plan is to arrive at the end of the convention and devour everybody.

Remember to output your answers modulo 10^9+7!

## Let's think step by step and generate code to solve the problem.

```python
def inverses(n, P):
    inv = [1] * (n + 1)
    for i in range(2, n + 1):
        inv[i] = inv[P % i] * (P - P // i) % P
    return inv

mod = 10**9 + 7

def init():
    fac[0] = ifac[0] = 1
    inv = inverses(N, mod)
    for i in range(1, N):
        fac[i] = (fac[i - 1] * i) % mod
        ifac[i] = (ifac[i - 1] * inv[i]) % mod

def comb(n, k):
    return (fac[n] * ifac[n - k] * ifac[k]) % mod

N = 3 * 10**6 + 10

fac = [0] * N
ifac = [0] * N

init()

def main():
    n, Q = map(int, input().split())
    sz = 3 * (n + 1)

    # Create a list to store coefficients of polynomial p(x)
    p = [0] * (sz + 1)

    # Initialize coefficients of p(x) using binomial coefficients
    for i in range(1, sz + 1):
        p[i] = comb(sz, i)

    # Backward iteration to calculate coefficients of p(x-1) and p(x-2)
    for i in range(sz, 2, -1):
        p[i - 1] = ((p[i - 1] - 3 * p[i]) % mod + mod) % mod
        p[i - 2] = ((p[i - 2] - 3 * p[i]) % mod + mod) % mod

    # Remove extra elements at the beginning of p
    p = p[3:]

    # Answer queries based on calculated coefficients
    for _ in range(Q):
        x = int(input())
        print(p[x])

# Call the main function
main()
```

-------

This is the easy version of the problem. The only difference from the hard version is that in this version all coordinates are even.

There are n fence-posts at distinct coordinates on a plane. It is guaranteed that no three fence posts lie on the same line.

There are an infinite number of cows on the plane, one at every point with integer coordinates.

Gregor is a member of the Illuminati, and wants to build a triangular fence, connecting 3 distinct existing fence posts. A cow strictly inside the fence is said to be enclosed. If there are an odd number of enclosed cows and the area of the fence is an integer, the fence is said to be interesting.

Find the number of interesting fences.

Input

The first line contains the integer n (3 ≤ n ≤ 6000), the number of fence posts which Gregor can choose to form the vertices of a fence.

Each of the next n line contains two integers x and y (0 ≤ x,y ≤ 10^7, x and y are even), where (x,y) is the coordinate of a fence post. All fence posts lie at distinct coordinates. No three fence posts are on the same line.

Output

Print a single integer, the number of interesting fences. Two fences are considered different if they are constructed with a different set of three fence posts.

Examples

Input


3
0 0
2 0
0 4


Output


1


Input


5
0 0
2 16
30 14
4 6
2 10


Output


3

Note

In the first example, there is only 1 fence. That fence is interesting since its area is 4 and there is 1 enclosed cow, marked in red.

In the second example, there are 3 interesting fences. 

  * (0,0) — (30,14) — (2,10) 
  * (2,16) — (30,14) — (2,10) 
  * (30,14) — (4,6) — (2,10) 

## Let's think step by step and generate code to solve the problem.

```python
def main():
    # Read the number of fence posts
    n = int(input())

    # Array to store the count of fence posts based on parity of coordinates
    cnt = [[0, 0], [0, 0]]

    # Loop to read the coordinates of each fence post
    for _ in range(n):
        x, y = map(int, input().split())

        # Divide coordinates by 2 to handle even coordinates
        x >>= 1
        y >>= 1

        # Update the count based on the parity of coordinates
        cnt[x & 1][y & 1] += 1

    # Variable to store the number of interesting fences
    ans = 0

    # Loop to iterate over all combinations of parity for x and y coordinates
    for i in range(2):
        for j in range(2):
            k = cnt[i][j]

            # Calculate the number of interesting fences using combinations
            ans += 1 * k * (k - 1) // 2 * (k - 2) // 3 + 1 * k * (k - 1) // 2 * (n - k)

    # Print the result
    print(ans)

# Call the main function
main()
```

-------

Gregor is learning about RSA cryptography, and although he doesn't understand how RSA works, he is now fascinated with prime numbers and factoring them.

Gregor's favorite prime number is P. Gregor wants to find two bases of P. Formally, Gregor is looking for two integers a and b which satisfy both of the following properties.

  * P mod a = P mod b, where x mod y denotes the remainder when x is divided by y, and 
  * 2 ≤ a < b ≤ P. 



Help Gregor find two bases of his favorite prime number!

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 1000).

Each subsequent line contains the integer P (5 ≤ P ≤ {10}^9), with P guaranteed to be prime.

Output

Your output should consist of t lines. Each line should consist of two integers a and b (2 ≤ a < b ≤ P). If there are multiple possible solutions, print any.

Example

Input


2
17
5


Output


3 5
2 4

Note

The first query is P=17. a=3 and b=5 are valid bases in this case, because 17 mod 3 = 17 mod 5 = 2. There are other pairs which work as well.

In the second query, with P=5, the only solution is a=2 and b=4.

## Let's think step by step and generate code to solve the problem.

def main():
    # Read the number of test cases
    t = int(input())

    # List to store prime numbers for each test case
    primes = [int(input()) for _ in range(t)]

    # Iterate over each test case to find bases
    for prime in primes:
        # Print bases a and b for each test case
        print(f"2 {prime - 1}")

# Call the main function
main()

"""
        
        
        input_for_planning = [
            {
                "role": "user",
                "content": f"{prompt}\n{self.data.get_prompt(item)}\n## Let's think step by step and generate {self.language} code to solve the problem.\n# ----------------\nImportant: Your response must contain only the {self.language} code to solve this problem inside ``` block.",
            },
        ]
        return self.gpt_chat(
            processed_input=input_for_planning
        )

class SelfPlanningStrategy(BaseMethod):
    def run_single_pass(self, item: dict):
        planning_prompt = None
        if type(self.data) == HumanEvalDataset:
            planning_prompt = """
def encrypt(s):
    '''
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    For example:
    encrypt('hi') returns 'lm'
    encrypt('asdfghjkl') returns 'ewhjklnop'
    encrypt('gf') returns 'kj'
    encrypt('et') returns 'ix'
    Let's think step by step.
    1. Create a alphabet, bias two places multiplied by two.
    2. Loop the input, find the latter bias letter in alphabet.
    3. Return result.
    ''' 

-------

def check_if_last_char_is_a_letter(txt):
    ''' 
    Create a function that returns True if the last character of a given string is an alphabetical character and is not a part of a word, and False otherwise. Note: 'word' is a group of characters separated by space.
    Examples:
    check_if_last_char_is_a_letter('apple pie') → False
    check_if_last_char_is_a_letter('apple pi e') → True
    check_if_last_char_is_a_letter('apple pi e ') → False
    check_if_last_char_is_a_letter('') → False
    Let's think step by step.
    1. If the string is empty, return False.
    2. If the string does not end with a alphabetical character, return False.
    3. Split the given string into a list of words.
    4. Check if the length of the last word is equal to 1.
    '''

-------

def file_name_check(file name):
    '''
    Create a function which takes a string representing a file's name, and returns 'Yes' if the the file's name is valid, and returns 'No' otherwise. A file's name is considered to be valid if and only if all the following conditions are met: - There should not be more than three digits ('0'-'9') in the file's name. - The file's name contains exactly one dot '.' - The substring before the dot should not be empty, and it starts with a letter from the latin alphapet ('a'-'z' and 'A'-'Z'). - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    Examples:
    file_name_check('example.txt') => 'Yes'
    file_name_check('1example.dll') => 'No' (the name should start with a latin alphapet letter)
    Let's think step by step.
    1. Check if the file name is valid according to the conditions.
    2. Return 'Yes' if valid, otherwise return 'NO'.
    '''

-------

def fruit_distribution(s,n):
    '''
    In this task, you will be given a string that represents a number of apples and oranges that are distributed in a basket of fruit this basket contains apples, oranges, and mango fruits. Given the string that represents the total number of the oranges and apples and an integer that represent the total number of the fruits in the basket return the number of the mango fruits in the basket.
    for examble:
    fruit_distribution('5 apples and 6 oranges', 19) = 19 - 5 - 6 = 8
    fruit_distribution('0 apples and 1 oranges',3) = 3 - 0 - 1 = 2
    fruit_distribution('2 apples and 3 oranges', 100) = 100 - 2 - 3 = 95
    fruit_distribution('100 apples and 1 oranges',120) = 120 - 100 - 1 = 19
    Let's think step by step.
    1. Extract the numbers of oranges and apples from given string.
    2. Calculate the sum of oranges and apples.
    3. Deduct the sum from the total number of fruits.
    4. Return the number of mangoes.
    '''

-------

def prime_fib(n: int):
    '''
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    Examples:
    >>> prime_fib(1) 2
    >>> prime_fib(2) 3
    >>> prime_fib(3) 5
    >>> prime_fib(4) 13
    >>> prime_fib(5) 89
    Let's think step by step.
    1. Create a function to check if a number is prime.
    2. Generate a Fibonacci sequence.
    3. Check if each number in the Fibonacci sequence is prime, decrement the counter.
    4. If the counter is 0, return the Fibonacci number.
    '''

-------
    
def compare_one(a, b):
    ''' 
    Create a function that takes integers, floats, or strings representing real numbers, and returns the larger variable in its given variable type. Return None if the values are equal. Note: If a real number is represented as a string, the floating point might be . or ,
    Examples:
    compare_one(1, 2.5) → 2.5
    compare_one(1, '2,3') → '2,3'
    compare_one('5,1', '6') → '6'
    compare_one('1', 1) → None
    Let's think step by step.
    1. Store the original inputs.
    2. Check if inputs are strings and convert to floats.
    3. Compare the two inputs and return the larger one in its original data type.
    ''' 

-------
    
def sort_even(l: list):
    ''' 
    This function takes a list l and returns a list l' such that l' is identical to l in the odd indicies, while its values at the even indicies are equal to the values of the even indicies of l, but sorted. 
    Examples: 
    >>> sort_even([1, 2, 3])
    [1, 2, 3]
    >>> sort_even([5, 6, 3, 4])
    [3, 6, 5, 4]
    Let's think step by step. 
    1. Create a list of all the even indices of the given list. 
    2. Sort the list of even indices. 
    3. Return a new list that is identical to the original list in the odd indicies, and equal to the sorted even indices in the even indicies. 
    ''' 

-------

def search(lst):
    '''
    You are given a non-empty list of positive integers. Return the greatest integer that is greater than zero, and has a frequency greater than or equal to the value of the integer itself. The frequency of an integer is the number of times it appears in the list. If no such a value exist, return -1.
    Examples:
    search([4, 1, 2, 2, 3, 1]) == 2
    search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
    search([5, 5, 4, 4, 4]) == -1
    Let's think step by step.
    1. Create a frequency dict.
    2. Sort the input list.
    3. Loop the input list, if frequency no lesser than the integer, set result.
    4. Return the result.
    '''

"""
        

        if type(self.data) == APPSDataset:
            planning_prompt = """
An accordion is a string (yes, in the real world accordions are musical instruments, but let's forget about it for a while) which can be represented as a concatenation of: an opening bracket (ASCII code $091$), a colon (ASCII code $058$), some (possibly zero) vertical line characters (ASCII code $124$), another colon, and a closing bracket (ASCII code $093$). The length of the accordion is the number of characters in it.

For example, [::], [:||:] and [:|||:] are accordions having length $4$, $6$ and $7$. (:|:), {:||:}, [:], ]:||:[ are not accordions. 

You are given a string $s$. You want to transform it into an accordion by removing some (possibly zero) characters from it. Note that you may not insert new characters or reorder existing ones. Is it possible to obtain an accordion by removing characters from $s$, and if so, what is the maximum possible length of the result?


-----Input-----

The only line contains one string $s$ ($1 \le |s| \le 500000$). It consists of lowercase Latin letters and characters [, ], : and |.


-----Output-----

If it is not possible to obtain an accordion by removing some characters from $s$, print $-1$. Otherwise print maximum possible length of the resulting accordion.


-----Examples-----
Input
|[a:b:|]

Output
4

Input
|]:[|:]

Output
-1

2
1 2

## Let's think step by step.

1. Take user input as a string `s`.
2. Calculate the length of the string `s` and assign it to the variable `n`.
3. Initialize variables `ind` and `bind` to store the indices of '[' and ']'.
4. Initialize a flag variable `f` to track whether '[' or ']' characters have been encountered.
5. Iterate through the string to find the index of the first '[' character after encountering ':' (indicated by the flag).
6. Reset the flag variable.
7. Iterate through the string in reverse to find the index of the last ']' character before encountering ':'.
8. Check conditions to determine if it's possible to obtain an accordion:
   - If '[' or ']' characters were not found, print -1.
   - If the order of '[' and ']' characters is incorrect, print -1.
9. If conditions are met:
   - Initialize the length of the accordion (`ans`) to 4 (opening and closing brackets, and two colons).
   - Count the number of '|' characters between '[' and ']' indices (inclusive).
   - Print the calculated length of the resulting accordion (`ans`).

-------

Anton has the integer x. He is interested what positive integer, which doesn't exceed x, has the maximum sum of digits.

Your task is to help Anton and to find the integer that interests him. If there are several such integers, determine the biggest of them. 


-----Input-----

The first line contains the positive integer x (1 ≤ x ≤ 10^18) — the integer which Anton has. 


-----Output-----

Print the positive integer which doesn't exceed x and has the maximum sum of digits. If there are several such integers, print the biggest of them. Printed integer must not contain leading zeros.


-----Examples-----
Input
100

Output
99

Input
48

Output
48

Input
521

Output
499

## Let's think step by step.

1. Take user input as a list of integers representing the digits of the given positive integer x (`num`).
2. Initialize a variable `best` to store the current best candidate for the maximum sum of digits, initially set to the digits of x.
3. Iterate over the digits in reverse order.
4. If the current digit is 0, continue to the next digit.
5. Decrease the current digit by 1.
6. Set all digits to the right of the current digit to 9.
7. Compare the sum of the current number with the sum of the best number.
8. Update `best` if the current number has a greater sum.
9. Convert the list of digits to a string, remove leading zeros, and print the result.

-------

Apart from having lots of holidays throughout the year, residents of Berland also have whole lucky years. Year is considered lucky if it has no more than 1 non-zero digit in its number. So years 100, 40000, 5 are lucky and 12, 3001 and 12345 are not.

You are given current year in Berland. Your task is to find how long will residents of Berland wait till the next lucky year.


-----Input-----

The first line contains integer number n (1 ≤ n ≤ 10^9) — current year in Berland.


-----Output-----

Output amount of years from the current year to the next lucky one.


-----Examples-----
Input
4

Output
1

Input
201

Output
99

Input
4000

Output
1000



-----Note-----

In the first example next lucky year is 5. In the second one — 300. In the third — 5000.

## Let's think step by step.

1. Define the `main` function.
2. Take user input as a string representing the current year in Berland (`s`).
3. Get the length of the input string (number of digits in the current year) and assign it to `n`.
4. Create a target lucky year (`t`) by incrementing the first digit by 1 and padding with zeros.
5. Convert the target lucky year to an integer.
6. Calculate the number of years from the current year to the next lucky one (`result`).
7. Print the result.
8. Call the `main` function.
"""

        if type(self.data) == CodeContestDataset:
            planning_prompt = """
Three little pigs from all over the world are meeting for a convention! Every minute, a triple of 3 new pigs arrives on the convention floor. After the n-th minute, the convention ends.

The big bad wolf has learned about this convention, and he has an attack plan. At some minute in the convention, he will arrive and eat exactly x pigs. Then he will get away.

The wolf wants Gregor to help him figure out the number of possible attack plans that involve eating exactly x pigs for various values of x (1 ≤ x ≤ 3n). Two attack plans are considered different, if they occur at different times or if the sets of little pigs to eat are different.

Note that all queries are independent, that is, the wolf does not eat the little pigs, he only makes plans!

Input

The first line of input contains two integers n and q (1 ≤ n ≤ 10^6, 1 ≤ q ≤ 2⋅ 10^5), the number of minutes the convention lasts and the number of queries the wolf asks.

Each of the next q lines contains a single integer x_i (1 ≤ x_i ≤ 3n), the number of pigs the wolf will eat in the i-th query.

Output

You should print q lines, with line i representing the number of attack plans if the wolf wants to eat x_i pigs. Since each query answer can be large, output each answer modulo 10^9+7.

Examples

Input


2 3
1
5
6


Output


9
6
1


Input


5 4
2
4
6
8


Output


225
2001
6014
6939

Note

In the example test, n=2. Thus, there are 3 pigs at minute 1, and 6 pigs at minute 2. There are three queries: x=1, x=5, and x=6.

If the wolf wants to eat 1 pig, he can do so in 3+6=9 possible attack plans, depending on whether he arrives at minute 1 or 2.

If the wolf wants to eat 5 pigs, the wolf cannot arrive at minute 1, since there aren't enough pigs at that time. Therefore, the wolf has to arrive at minute 2, and there are 6 possible attack plans.

If the wolf wants to eat 6 pigs, his only plan is to arrive at the end of the convention and devour everybody.

Remember to output your answers modulo 10^9+7!

## Let's think step by step.

1. Read the number of minutes `n` and the number of queries `Q`.
2. Implement the `inverses` function to calculate inverses using modular arithmetic.
3. Initialize factorials and inverses using the `init` function from the `combi` namespace.
4. Initialize coefficients of polynomial `p(x)` using binomial coefficients up to `3 * (n + 1)`.
5. Backward iteration to calculate coefficients of `p(x-1)` and `p(x-2)`.
6. Remove extra elements at the beginning of the `p` vector.
7. Answer queries based on calculated coefficients.
8. Print each answer modulo 10^9+7.

-------

This is the easy version of the problem. The only difference from the hard version is that in this version all coordinates are even.

There are n fence-posts at distinct coordinates on a plane. It is guaranteed that no three fence posts lie on the same line.

There are an infinite number of cows on the plane, one at every point with integer coordinates.

Gregor is a member of the Illuminati, and wants to build a triangular fence, connecting 3 distinct existing fence posts. A cow strictly inside the fence is said to be enclosed. If there are an odd number of enclosed cows and the area of the fence is an integer, the fence is said to be interesting.

Find the number of interesting fences.

Input

The first line contains the integer n (3 ≤ n ≤ 6000), the number of fence posts which Gregor can choose to form the vertices of a fence.

Each of the next n line contains two integers x and y (0 ≤ x,y ≤ 10^7, x and y are even), where (x,y) is the coordinate of a fence post. All fence posts lie at distinct coordinates. No three fence posts are on the same line.

Output

Print a single integer, the number of interesting fences. Two fences are considered different if they are constructed with a different set of three fence posts.

Examples

Input


3
0 0
2 0
0 4


Output


1


Input


5
0 0
2 16
30 14
4 6
2 10


Output


3

Note

In the first example, there is only 1 fence. That fence is interesting since its area is 4 and there is 1 enclosed cow, marked in red.

In the second example, there are 3 interesting fences. 

  * (0,0) — (30,14) — (2,10) 
  * (2,16) — (30,14) — (2,10) 
  * (30,14) — (4,6) — (2,10) 

## Let's think step by step.

1. Read the number of fence posts `n`.
2. Initialize a 2x2 array `cnt` to store the count of fence posts based on the parity of coordinates.
3. Loop to read the coordinates of each fence post.
4. Divide the coordinates by 2 to handle even coordinates.
5. Update the count based on the parity of coordinates.
6. Initialize a variable `ans` to store the number of interesting fences.
7. Loop to iterate over all combinations of parity for x and y coordinates.
8. Calculate the number of interesting fences using combinations.
9. Print the result.

-------

Gregor is learning about RSA cryptography, and although he doesn't understand how RSA works, he is now fascinated with prime numbers and factoring them.

Gregor's favorite prime number is P. Gregor wants to find two bases of P. Formally, Gregor is looking for two integers a and b which satisfy both of the following properties.

  * P mod a = P mod b, where x mod y denotes the remainder when x is divided by y, and 
  * 2 ≤ a < b ≤ P. 



Help Gregor find two bases of his favorite prime number!

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 1000).

Each subsequent line contains the integer P (5 ≤ P ≤ {10}^9), with P guaranteed to be prime.

Output

Your output should consist of t lines. Each line should consist of two integers a and b (2 ≤ a < b ≤ P). If there are multiple possible solutions, print any.

Example

Input


2
17
5


Output


3 5
2 4

Note

The first query is P=17. a=3 and b=5 are valid bases in this case, because 17 mod 3 = 17 mod 5 = 2. There are other pairs which work as well.

In the second query, with P=5, the only solution is a=2 and b=4.

## Let's think step by step.

1. Read the number of test cases `t`.
2. Initialize an array `array` to store prime numbers for each test case.
3. Read the prime numbers for each test case and store them in the array.
4. Iterate over each test case to find bases.
5. Print bases `a` and `b` for each test case, where `a = 2` and `b = P - 1`.
6. Repeat steps 4-5 for all test cases.
"""
        
        
        input_for_planning = [
            {
                "role": "user",
                "content": f"{planning_prompt}\n{self.data.get_prompt(item)}\n## Let's think step by step.",
            },
        ]
        planning, prompt_tokens, completion_tokens = self.gpt_chat(
            processed_input=input_for_planning
        )

        input_for_implementation = [
            {
                "role": "user",
                "content": f"Problem:\n{self.data.get_prompt(item)}\nPlanning:\n{planning}\nGenerate {self.language} code without any explanation by following the mentioned plans.\n# ----------------\nImportant: Your response must contain only the {self.language} code to solve this problem inside ``` block.",
            },
        ]

        implementation, prompt_tokens_2, completion_tokens_2 = self.gpt_chat(
            processed_input=input_for_implementation
        )

        return implementation, \
                prompt_tokens + prompt_tokens_2, \
                completion_tokens + completion_tokens_2

# self-generate exemplars and knowledge
class AnalogicalStrategy(BaseMethod):
    def parse_code(self, code: str):
        if "Python3 code to solve the original problem:" in code:
            code = code.split("Python3 code to solve the original problem:")[1].strip()
        
        code_pattern = r'```((.|\n)*?)```'
        if "```python" in code:
            code_pattern = r'```python((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, code, re.DOTALL)

        if len(code_blocks) == 0:
            if "```" in code:
                code = code.replace("```", "")
            return code

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code = code_blocks[-1]

        return code
    
    def run_single_pass(self, item: dict):
        input = [
            {
                "role": "user",
                "content": 
f"""Your goal is to write {self.language} code to solve competitive programming problems. Given a problem , explain the core concepts in it and provide other relevant problems. Then solve the original problem.

# Problem:
{self.data.get_prompt(item)}

# Instruction: (Your response must include the following points sequentially)

## Algorithms:
Identify the core concepts or algorithms used to solve the problem.

## Tutorial:
Write a useful tutorial about these algorithms.

## Example Problems: 
Provide three examples of relevant competitive programming problems that involve these algorithms. For each problem , describe the problem , explain the solution in detail , and then write the correct Python3 code.

## {self.language} code to solve the original problem: 
Include the following points in your response: 
- Explanation of the solution: 
- {self.language} code to solve the problem (inside ```  ``` block):""",
            },
        ]
        print(input[0]['content'])

        response, prompt_tokens, completion_tokens = self.gpt_chat(
            processed_input=input
        )

        print(response)

        return response, prompt_tokens, completion_tokens
