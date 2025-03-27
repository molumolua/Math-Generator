base_instruction = """
You are a mathematics expert specializing in simplifying math problems.

Your task is to transform the **Original Problem** into a slightly simpler, more accessible version by focusing on its **solution**. You will simplify the problem by modifying, removing, or transferring one step from the **Original Solution** into the problem's conditions, ensuring the simplified problem remains **correct and solvable**.

Follow these steps to do it:

1. **Analyze Original Solution**:
   - Identify the key concepts and techniques used in the **Original Solution**, step by step.
   - Recognize the intermediate algebraic steps that can be eliminated or transferred into the problem's conditions.

2. **Simplification Process**:
   - Choose ONE step from the **Original Solution** to simplify, remove, or transfer:
     - **Simplify**: Identify a step that requires advanced techniques and simplify it to use more basic concepts.
     - **Remove**: Adjust or delete a single step in the solution process, ensuring the remaining steps can still lead to a complete and solvable problem.
     - **Transfer**: Incorporate a step from the solution directly into the problem's conditions, reducing the number of steps needed in the solution.
   - Ensure the simplified problem and its solution remain **consistent** with the original problem type but at a lower level of complexity.

3. **Present the Simplified Problem and Solution**:
   - **Simplified Problem**:
     - Rewrite the problem based on the simplified solution, ensuring the problem logically matches the simplified steps. If transferring a step, include the relevant information in the problem.
   - **Simplified Solution**:
     - Provide a clear, correct solution reflecting the simplified steps.
     - Ensure the solution directly follows from the simplified problem, with no added assumptions.

**Constraints**:
- The simplified problem must remain mathematically sound and consistent with the original problem type.
- The simplification must focus on reducing the steps or techniques used in the **Original Solution** rather than just changing numbers or coefficients.
- The simplified problem and solution should both be clear, complete, and logical.

**Provide**:
1. **Analyze Original Solution**:
   - List the mathematical ideas and techniques from the **Original Solution**, step by step.
2. **Simplification Process**:
   - Explain which step from the solution was simplified, removed, or transferred, and why.
3. **Simplified Problem**:
   - Present the simplified version of the original problem, reflecting the changes to the solution.
4. **Simplified Solution**:
   - Offer a step-by-step solution to the simplified problem using the modified solution steps.
   - Output the answer in the $\boxed{}$.
"""
example_prompt='''
---
# Example Input:

**Original Problem**:
For certain vectors $\mathbf{p}$ and $\mathbf{q}$, the vectors $3 \mathbf{p} + \mathbf{q}$ and $5 \mathbf{p} - 3 \mathbf{q}$ are orthogonal. Also, the vectors $2 \mathbf{p} + \mathbf{q}$ and $4 \mathbf{p} - 2 \mathbf{q}$ are orthogonal. If $\theta$ is the angle between $\mathbf{p}$ and $\mathbf{q}$, then find $\cos \theta$.

**Original Solution**:
1. Since $2 \mathbf{p} + \mathbf{q}$ and $4 \mathbf{p} - 2 \mathbf{q}$ are orthogonal, $(2 \mathbf{p} + \mathbf{q}) \cdot (4 \mathbf{p} - 2 \mathbf{q}) = 0.$ Expanding, we get \[8 \mathbf{p} \cdot \mathbf{p} - 2 \mathbf{q} \cdot \mathbf{q} = 0,\] so $\|\mathbf{q}\|^2 = 4 \|\mathbf{p}\|^2,$ and $\|\mathbf{q}\| = 2 \|\mathbf{p}\|.$
2. Since $3 \mathbf{p} + \mathbf{q}$ and $5 \mathbf{p} - 3 \mathbf{q}$ are orthogonal, $(3 \mathbf{p} + \mathbf{q}) \cdot (5 \mathbf{p} - 3 \mathbf{q}) = 0.$ Expanding, we get \[15 \mathbf{p} \cdot \mathbf{p} - 4 \mathbf{p} \cdot \mathbf{q} - 3 \mathbf{q} \cdot \mathbf{q} = 0.\]
3. Since $\mathbf{q} \cdot \mathbf{q} = 4 \mathbf{p} \cdot \mathbf{p},$ \[4 \mathbf{p} \cdot \mathbf{q} = 3 \mathbf{p} \cdot \mathbf{p}.\]
4. Then \[\cos \theta = \frac{\mathbf{p} \cdot \mathbf{q}}{\|\mathbf{p}\| \|\mathbf{q}\|} = \frac{\frac{3}{4} \mathbf{p} \cdot \mathbf{p}}{2 \|\mathbf{p}\|^2} = \boxed{\frac{3}{8}}.\]

---
# Example Output:

1. **Analyze Original Solution**:
   - *Step 1*: Use orthogonality of $2 \mathbf{p} + \mathbf{q}$ and $4 \mathbf{p} - 2 \mathbf{q}$ to find a relationship between the magnitudes of $\mathbf{p}$ and $\mathbf{q}$.
   - *Step 2*: Use the orthogonality of $3 \mathbf{p} + \mathbf{q}$ and $5 \mathbf{p} - 3 \mathbf{q}$ to relate $\mathbf{p} \cdot \mathbf{q}$ to $\|\mathbf{p}\|$.
   - *Step 3*: Substitute the magnitude relationship into the dot product and solve for $\cos \theta$.

2. **Simplification Process**:
   Incorporate the magnitude relationship $|\mathbf{q}| = 2 |\mathbf{p}|$ directly into the problem statement. This eliminates the need for Step 1 in the solution.
   
3. **Simplified Problem**:
   Given that the vectors $3 \mathbf{p} + \mathbf{q}$ and $5 \mathbf{p} - 3 \mathbf{q}$ are orthogonal, and $|\mathbf{q}| = 2 |\mathbf{p}|$, find $\cos \theta$, where $\theta$ is the angle between $\mathbf{p}$ and $\mathbf{q}$.

4. **Simplified Solution**:
   - *Step 1*: The orthogonality condition $(3 \mathbf{p} + \mathbf{q}) \cdot (5 \mathbf{p} - 3 \mathbf{q}) = 0$ simplifies to:
     \[ 15 \mathbf{p} \cdot \mathbf{p} - 4 \mathbf{p} \cdot \mathbf{q} - 3 \mathbf{q} \cdot \mathbf{q} = 0. \]
   - *Step 2*: We know that $\mathbf{q} \cdot \mathbf{q} = 4 \mathbf{p} \cdot \mathbf{p}$, so substitute this into the equation:
     \[ 15 \mathbf{p} \cdot \mathbf{p} - 4 \mathbf{p} \cdot \mathbf{q} - 12 \mathbf{p} \cdot \mathbf{p} = 0, \]
     which simplifies to:
     \[ 3 \mathbf{p} \cdot \mathbf{p} = 4 \mathbf{p} \cdot \mathbf{q}. \]
   - *Step 3*: Solve for $\cos \theta$:
     \[ \cos \theta = \frac{\mathbf{p} \cdot \mathbf{q}}{\|\mathbf{p}\| \|\mathbf{q}\|}. \]
     From $3 \mathbf{p} \cdot \mathbf{p} = 4 \mathbf{p} \cdot \mathbf{q}$, we get:
     \[ \mathbf{p} \cdot \mathbf{q} = \frac{3}{4} \mathbf{p} \cdot \mathbf{p}. \]
     Substitute this into the cosine formula:
     \[ \cos \theta = \frac{\frac{3}{4} \mathbf{p} \cdot \mathbf{p}}{2 \|\mathbf{p}\|^2} = \frac{3}{8}. \]
   - Final answer: $\boxed{\frac{3}{8}}$.

---
'''
def createSimpleQuestionPrompt(problem, solution):
    prompt = base_instruction
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt
 
 
# compare_prompt = """
# You will be given two problems along with their respective answers. Your task is to compare the difficulty of the two problems and determine if:

# 1. Problem 1 is **easier** than Problem 2  
# 2. Problem 1 is **harder** than Problem 2  
# 3. Both problems are of **comparable** difficulty  

# You must base your comparison on the following factors:
# 1. **The complexity of the problem**  
# 2. **The types of operations involved**  
# 3. **The amount of prior knowledge required**  
# 4. **The difficulty of solving each problem**  

# Your output should strictly include:  
# - `"easier"` if Problem 1 is easier  
# - `"harder"` if Problem 1 is harder  
# - `"comparable"` if both problems are of roughly the same difficulty  

# Additionally, provide a reasoning section titled `Reasoning Steps:` where you clearly explain your decision, listing the specific factors that influenced your judgment. Avoid giving conclusions beyond the three specified options.

# ---

# # Input Format Example:

# Problem 1: [Description of Problem 1]  
# Answer 1: [Answer to Problem 1]

# Problem 2: [Description of Problem 2]  
# Answer 2: [Answer to Problem 2]

# ---

# # Output Format Example:

# "easier"
# Reasoning Steps:
# 1. [Step 1 of your reasoning]
# 2. [Step 2 of your reasoning]
# 3. [Conclusion explaining why Problem 1 is easier than Problem 2]

# **Important Notes:**  
# - Only output one of the three possible conclusions: `"easier"`, `"harder"`, or `"comparable"`.  
# - Provide reasoning directly below your conclusion under the heading `Reasoning Steps:`, clearly detailing the comparison factors.  

# ---

# """
# example='''
# # Example

# ## Example Input:
# Problem 1: Solve for x in the equation 2x + 3 = 7  
# Answer 1: x = 2

# Problem 2: Find the roots of the quadratic equation x^2 - 5x + 6 = 0  
# Answer 2: x = 2, 3

# ## Example Output:
# "easier"
# Reasoning Steps:
# 1. Problem 1 involves a simple linear equation, requiring basic algebraic manipulation to isolate the variable.
# 2. Problem 2 involves solving a quadratic equation, which requires factoring or applying the quadratic formula—an additional layer of complexity.
# 3. Conclusion: Since Problem 2 requires more advanced techniques, Problem 1 is easier.

# ---
# '''
compare_prompt_incontext='''
You are an expert in evaluating and comparing mathematical problems based on their difficulty.

Your task is to assess the relative difficulty between two given problems by analyzing their descriptions and solutions.


**Comparison Criteria**:

- Analyze how intricate and involved each problem is.
- Examine the mathematical operations and techniques required.
- Consider the foundational knowledge needed to approach each problem.

     
**Output your evaluation in the following structured format**:
1. **Reasoning Steps**:
   - Provide a list of reasons that support your following conclusion based on **Comparison Criteria**.
2. **Conclusion**:
   - Output only one of the following based on your analysis:
     - `"former"` if **Problem 1** is harder than **Problem 2**.
     - `"later"` if **Problem 2** is harder than **Problem 1**.
     - `"comparable"` if both problems have similar difficulty levels.
---
'''

compare_example_1='''

# Example Input:
**Problem 1**:
Express the following as a common fraction: $\\sqrt[3]{4\\div 13.5}$
**Solution 1**:
Writing $13.5$ as $\\frac{27}{2}$, we get \\[\\sqrt[3]{4\\div 13.5} = \\sqrt[3]{\\frac{4}{27/2}} = \\sqrt[3]{4\\cdot \\frac{2}{27}} = \\sqrt[3]{\\frac{8}{27}} = \\sqrt[3]{\\frac{2^3}{3^3}} = \\boxed{\\frac23}.\\]
**Problem 2**:
A certain organism begins as three cells. Each cell splits and becomes two cells at the end of two days. At the end of another two days, every cell of the organism splits and becomes two cells. This process lasts for a total of 8 days, and no cells die during this time. How many cells are there at the end of the $8^\\text{th}$ day?
**Solution 2**:
This is a geometric sequence with a first term of $3$ and a common ratio of $2$. At the end of the eighth day, we are at the 5th term of this sequence, so there are $3\\cdot2^4=\\boxed{48}$ cells then.

# Example Output:
1. **Reasoning Steps**:
- **Complexity of the Problem**: Problem 1 involves manipulating a fractional expression under a cube root, including conversion and simplification of fractions, whereas Problem 2 involves understanding and applying a simple geometric progression. Problem 1's operations are more intricate.

- **Types of Operations Involved**: Problem 1 requires fractional division, cube root extraction, and simplification of expressions, which are more advanced operations, while Problem 2 only requires applying the formula for the nth term of a geometric sequence.

- **Amount of Prior Knowledge Required**: Problem 1 requires knowledge of fractional arithmetic, cube roots, and simplification techniques, which are typically more advanced topics. Problem 2 requires understanding of basic geometric sequences, which is generally introduced earlier.

2. **Conclusion**:
"former"
---
'''
compare_example_2='''

# Example Input:
**Problem 1**:
Evaluate: $-\\left(14\\div 2\\cdot 9-60+3\\cdot 9\\right)$.
**Solution 1**:
Recall that the order of operations says that we must perform the multiplication and division before we do the addition and subtraction.  Also, the operations inside the parentheses must be done before we negate the entire expression.  Therefore, we have \\begin{align*}-\\left(14\\div 2\\cdot 9-60+3\\cdot 9\\right)&=-\\left(7\\cdot 9-60+3\\cdot 9\\right) \\\\ &=-\\left(63-60+3\\cdot 9\\right) \\\\ &=-\\left(63-60+27\\right) \\\\ &=-\\left(63+(-60)+27\\right) \\\\ &=-\\left(63+27+(-60)\\right) \\\\ &=-\\left(90+(-60)\\right) \\\\ &=-\\left(90-60\\right) \\\\ &=-\\left(30\\right) \\\\ &=\\boxed{-30}.\\end{align*}
**Problem 2**:
The expression $\\frac{4k+8}{4}$ simplifies to an expression of the form $ak+b$ where $a$ and $b$ are integers. Find $\\frac{a}{b}$ .
**Solution 2**:
We need to look for a common factor of 4 and 8 to cancel. 4 and 8 are both divisible by 4 so we can cancel 4 from the numerator and denominator of the fraction. \\[\\frac{4k+8}{4}=\\frac{4\\cdot(1k+2)}{4\\cdot1}=\\frac{4}{4}\\cdot\\frac{1k+2}{1}=\\frac{1k+2}{1}\\] Dividing by one leaves an expression the same, so the it is now $1k+2$ . Checking the form that the answer should be expressed in, we can see that $1k+2$ is of the form $ak+b$ with $a$ and $b$ integers, since 1 and 2 are both integers. So we divide 1 by 2 to get $\\boxed{\\frac{1}{2}}$ .

# Example Output:
1. **Reasoning Steps**:
- **Complexity of the Problem**: Both problems involve straightforward arithmetic operations. Problem 1 requires applying the order of operations (PEMDAS/BODMAS) to evaluate an expression, while Problem 2 involves simplifying a linear algebraic expression. The complexity is similar because both require basic manipulation of expressions without deeper mathematical structures.

- **Types of Operations Involved**: Problem 1 involves division, multiplication, addition, and subtraction within a nested expression. Problem 2 involves factoring and simplifying a fraction, which includes recognizing common factors and performing division. The types of operations are elementary in both cases, making them comparable in this aspect.

- **Amount of Prior Knowledge Required**: Both problems require a basic understanding of arithmetic operations and algebraic manipulation. Problem 1 requires knowledge of the order of operations, while Problem 2 requires understanding how to simplify an algebraic fraction. The prior knowledge required for both problems is at the level of early high school mathematics, indicating a comparable level of difficulty.

2. **Conclusion**:
"comparable"
---
'''
compare_example_3='''

# Example Input:
**Problem 1**:
My school's math club has 6 boys and 8 girls.  I need to select a team to send to the state math competition.  We want 6 people on the team.  In how many ways can I select the team to have 3 boys and 3 girls?
**Solution 1**:
We are picking 3 boys out of 6, so there are $\\binom{6}{3} = 20$ options for the boys on the team.  We are picking 3 girls out of 8, so there are $\\binom{8}{3} = 56$ options for the girls on the team.  This gives a total of $20 \\times 56 = \\boxed{1120}$ choices.
**Problem 2**:
Seven people arrive to dinner, but the circular table only seats six.  If two seatings such that one is a rotation of the other are considered the same, then in how many different ways can we choose six people and seat them at the table?
**Solution 2**:
There are 7 ways to choose the person who is left standing.  To seat the 6 remaining people, there are 6 seats from which the first person can choose, 5 seats left for the second, and so on down to 1 seat for the last person.  This suggests that there are $6\\cdot 5\\cdot 4\\cdot 3\\cdot 2\\cdot 1 = 6!$ ways to seat the six people.  However, each seating can be rotated six ways, so each seating is counted six times in this count.  Therefore, for each group of 6 people, there are $6!/6 = 5!$ ways to seat them around the table.  There are 7 different possible groups of 6 to seat (one for each person left standing), giving a total of $7\\cdot 5! = \\boxed{840}$ ways to seat the seven people.

# Example Output:
1. **Reasoning Steps**:
- **Complexity of the Problem**: Problem 2 involves both selecting a subset of individuals and arranging them in a specific formation (a circular arrangement), which adds complexity compared to Problem 1, which only involves selecting subsets.

- **Types of Operations Involved**: Problem 1 uses combinations, which are straightforward counting operations. Problem 2 also involves combinations but requires additional reasoning about permutations and rotational symmetry, increasing operational complexity.

- **Amount of Prior Knowledge Required**: Both problems require knowledge of combinatorics, specifically combinations. However, Problem 2 also requires understanding of permutations and how rotational symmetries affect counting, necessitating more advanced prior knowledge.\n   4. **Difficulty of Solving Each Problem**: Problem 1 is a direct application of the combination formula and simple multiplication. Problem 2 requires careful consideration of rotational equivalence in permutations, making it more challenging to solve correctly.

2. **Conclusion**:
"later"
---
'''

compare_prompt='''
You are an expert in evaluating and comparing mathematical problems based on their difficulty.

Your task is to assess the relative difficulty between two given problems by analyzing their descriptions and solutions.


**Comparison Criteria**:

- Analyze how intricate and involved each problem is.
- Examine the mathematical operations and techniques required.
- Consider the foundational knowledge needed to approach each problem.
- Assess the overall challenge in finding a solution.

Please provide the following sections in your answer:
1. **Reasoning Steps**:
   - Provide a list of reasons analyze the relative difficulty.
   - Each reason should correspond to one of the comparison criteria.
   - Ensure clarity and relevance in each reasoning step.
   
2. **Conclusion**:
   - Provide only one of the following based on **Reasoning Steps**:
     - `"former is harder"` if **Problem 1** is harder than **Problem 2**.
     - `"later is harder"` if **Problem 2** is harder than **Problem 1**.
     - `"comparable"` if both problems have similar difficulty levels.
   
**Constraints**:
- **Do not include any additional text or sections** beyond the specified format.
- Each reasoning step should provide unique insight based on the different comparison criteria.
- Use **bold** formatting for each section title.
'''

def createComparePrompt(problem1, answer1, problem2, answer2):
    """
    将给定的两个问题与答案插入到 compare_prompt 中，并返回完整的 Prompt。
    """
    prompt = compare_prompt
    prompt += f"\n**Problem 1**: {problem1}\n**Solution 1**: {answer1}\n"
    prompt += f"\n**Problem 2**: {problem2}\n**Solution 2**: {answer2}\n"
    prompt +="Please Analysis the relative difficulty between the given problems."
    return prompt

base_instructionV3 = """
You are a mathematics expert specializing in simplifying math problems.

Your task is to transform the **Original Problem** into a more accessible version by simplifying core mathematical concepts or techniques used in **Original Solution**.

Please provide the following sections in your answer:

1. **Simplification Process**:
   - Break down the original solution and identify significant core concepts or techniques that can be simplified, replaced, or deleted to generate a new problem.
   - Clearly explain **how** you are reducing complexity or using more basic methods.
   - **Adjust the original problem statement** to align with the simplified approach.
   
2. **Simplified Problem**:
   - Provide the **revised** problem statement **without any introductory or explanatory sentences**.

3. **Simplified Solution**:
   - Present the simplified solution in a logical sequence, ensuring the correctness.

**Format Requirements**:
- Use **bold** formatting for each section title.
- Ensure that the final answer is enclosed within \\boxed{{}}.

**Constraints**:
- **Ensure that the simplified problem has a unique answer**.
- **You must change the wording or structure of the original problem statement** enough to reflect the simpler approach.
"""

def createSimpleQuestionPromptV3(problem, solution):
    prompt = base_instructionV3
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt
 
 
def createAnsqerPrompt(problem):
   return f"{problem}\nPlease reason step by step,and put your final answer within \\boxed{{}}"


complexification_prompt = """
You are a mathematics expert specializing in increasing the complexity of math problems.

Your task is to transform the **Original Problem** into a more challenging version by introducing advanced mathematical concepts or techniques.

Please provide the following sections in your answer:

1. **Complexification Process**:
   - Break down the original solution and identify opportunities to introduce more advanced concepts.
   - Clearly explain **how** you are increasing the complexity by adding these techniques or ideas.
   - **Adjust the original problem statement** to reflect the more sophisticated approach.

2. **Complexified Problem**:
   - Provide the **revised** problem statement **without any introductory or explanatory sentences**.

3. **Complexified Solution**:
   - Present the complexified solution in a logical sequence, ensuring the correctness.

**Format Requirements**:
- Use **bold** formatting for each section title.
- Ensure that the final answer is enclosed within \\boxed{{}}.
"""

complexification_prompt_noprocess = """
Your task is to transform the **Original Problem** into a more challenging version.

Please provide the following sections in your answer:

1. **Complexified Problem**:
   - Provide the **revised** problem statement without any introductory or explanatory sentences.

2. **Complexified Solution**:
   - Present the complexified solution in a logical sequence, ensuring the correctness.
   - Ensure that the final answer is enclosed within \\boxed{{}}.

**Format Requirements**:
- Use **bold** formatting for each section title.
"""


def createComplexQuestionPrompt(problem, solution):
    prompt = complexification_prompt_noprocess
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt

def createComplexQuestionProcessPrompt(problem, solution):
    prompt = complexification_prompt
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt


add_process_prompt = '''
You are a mathematics expert tasked with increasing the complexity of a given problem.

Supposing you have alreadly transfromed the **Original Problem** into **Complexified Problem**, your task is to supply the complification process.

**Reversed Process** is the reversed version of your output, and you should use it to guide the output of complexification process.

Only provide following content in your answer:
- Break down the original solution and identify opportunities to introduce more advanced concepts.
- Clearly explain **how** you are increasing the complexity by adding these techniques or ideas.
- Provide the method how to adjust the original problem statement to reflect the more sophisticated approach.

Do not output any titles,section labels,original problem or complexified problem and avoid to be verbose.
'''


def createAddProcessPrompt(problem_1, solution_1,problem_2,solution_2,reverse_process):
    prompt = add_process_prompt
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem_1)
    prompt += "\n**Original Solution**:\n{}\n".format(solution_1)
    prompt += "\n**Complexified Problem**:\n{}\n".format(problem_2)
    prompt += "\n**Complexified Solution**:\n{}\n".format(solution_2)
    prompt += "\n**Reversed Process**:\n{}\n".format(reverse_process)
    return prompt


add_process_prompt_2 = '''
You are a mathematics expert specializing in simplifying math problems.

Supposing you have alreadly transfromed the **Original Problem** into **Simplified Problem**, your task is to supply the simplification process.

Only provide following content in your answer:
- Break down the original solution and identify significant core concepts or techniques that can be simplified, replaced, or deleted to generate a new problem.
- Clearly explain **how** you are reducing complexity or using more basic methods.
- Provide the method how to adjust the original problem statement to align with the simplified approach.

Do not output any titles,section labels,original problem or simplified Problem and avoid to be verbose.
'''


def createAddProcessPrompt_2(problem_1, solution_1,problem_2,solution_2):
    prompt = add_process_prompt_2
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem_1)
    prompt += "\n**Original Solution**:\n{}\n".format(solution_1)
    prompt += "\n**Simplified Problem**:\n{}\n".format(problem_2)
    prompt += "\n**Simplified Solution**:\n{}\n".format(solution_2)
    return prompt


add_think_prompt = '''
You are a mathematics expert tasked with increasing the complexity of a problem.

Supposing you have alreadly transformed **Original Problem** into **Complexified Problem**, explain the reasoning process of transforming the original problem into the more complex one inside <think></think>.

**Constraint**:
- Suppose you do not know **Complexified Problem** and focus only on the transforming **Original Problem** into the complexified one.
- Repeat given **Complexified Problem** and **Complexified Solution** in the output.
'''


def createAddThinkPrompt(problem_1, solution_1,problem_2,solution_2):
    prompt = add_think_prompt
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem_1)
    prompt += "\n**Original Solution**:\n{}\n".format(solution_1)
    prompt += "\n**Complexified Problem**:\n{}\n".format(problem_2)
    prompt += "\n**Complexified Solution**:\n{}\n".format(solution_2)
    return prompt



base_instruction_think = """
Your task is to transform the **Original Question** and corresponding **Original Answer** into a simpler **Simplified Question** and corresponding **Simplified Answer**.

**Simplification Criteria**:
- Break down the **Original Answer** and identify key concepts or techniques that can be simplified, replaced, or deleted to create a new question.
- Reduce the complexity by applying more basic methods or replacing advanced concepts with simpler ones.

**Original Question**:
{Question}

**Original Answer**:
{Answer}

**Output Requirements**:
The output should include the following sections and use **bold** formatting for each section title:
1. **Simplification Process**:
   - Explain **how** the simplification process reduces complexity or uses more basic methods based on the simplification criteria.

2. **Simplified Question**:
   - Provide the **revised** Question statement without any introductory or explanatory sentences.

3. **Simplified Answer**:
   - Present the simplified answer step by step, ensuring it remains correct and logically sound.
   - Ensure that the final answer is enclosed within \\boxed{{}}.
"""

def createThinkSimpleQuestionPrompt(problem, solution):
    prompt = base_instruction_think.format(Question=problem,Answer=solution)
    return prompt


compare_think_prompt='''
Your task is to assess the relative difficulty between two given problems by analyzing their descriptions and solutions.

Please provide **one** of the following conclusion in your answer:
   
- `"former one is harder."` if **Problem 1** is harder than **Problem 2**.
- `"later one is harder."` if **Problem 2** is harder than **Problem 1**.
- `"comparable"` if both problems have similar difficulty levels.
'''

def createCompareThinkPrompt(problem1, answer1, problem2, answer2):
    """
    将给定的两个问题与答案插入到 compare_prompt 中，并返回完整的 Prompt。
    """
    prompt = compare_think_prompt
    prompt += f"\n**Problem 1**: {problem1}\n**Solution 1**: {answer1}\n"
    prompt += f"\n**Problem 2**: {problem2}\n**Solution 2**: {answer2}\n"
    prompt +="Please Analysis the relative difficulty between the given problems."
    return prompt

complex_think_prompt_now='''
Your task is to transform the **Original Question** and corresponding **Original Answer** into a more difficult **Hard Question** and corresponding **Hard Answer**.

**Increased Difficulty Criteria**:
- Break down the **Original Answer** and identify key concepts or techniques that can be extended, enhanced, or replaced with more advanced ones to create a new question.
- Increase the difficulty by incorporating additional mathematical methods, concepts, or theories that enhance the original approach.

**Original Question**:
{Question}

**Original Answer**:
{Answer}

**Output Requirements**:
The output should include the following sections and use **bold** formatting for each section title:

1. **Hard Question**:
   - Provide the **revised** Question statement that introduces more complexity by adding new elements, constraints, or advanced concepts.

2. **Hard Answer**:
   - Present the more difficult answer step by step, ensuring it logically incorporates the more advanced methods or techniques.
   - Ensure that the final answer is enclosed within \\boxed{{}}.

'''

compare_think_prompt_test='''
You are an expert in evaluating and comparing mathematical problems based on their difficulty.

Your task is to assess the relative difficulty between two given problems by analyzing their descriptions and solutions.

**Comparison Criteria**:

- Analyze how intricate and involved each problem is.
- Examine the mathematical operations and techniques required.
- Consider the foundational knowledge needed to approach each problem.
- Assess the overall challenge in finding a solution.

Please only provide **one** of the following conclusion in your answer:
   
- `"former one is harder."` if **Problem 1** is harder than **Problem 2**.
- `"later one is harder."` if **Problem 2** is harder than **Problem 1**.
- `"tie"` if both problems have equal difficulty levels.
'''

def createDetailedCompareThinkPrompt(problem1, answer1, problem2, answer2):
    """
    将给定的两个问题与答案插入到 compare_prompt 中，并返回完整的 Prompt。
    """
    prompt = compare_think_prompt_test
    prompt += f"\n**Problem 1**: {problem1}\n**Solution 1**: {answer1}\n"
    prompt += f"\n**Problem 2**: {problem2}\n**Solution 2**: {answer2}\n"
    prompt +="Please Analysis the relative difficulty between the given problems."
    return prompt


def createNewComplexQuestionPrompt(problem, solution):
    prompt = complex_think_prompt_now.format(Question=problem,Answer=solution)
    return prompt
