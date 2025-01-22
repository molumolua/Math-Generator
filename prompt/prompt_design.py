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

construct_prompt='''
You are a professional mathematics problem analyst. Your task is to analyze the provided JSON-formatted solution and accurately reconstruct the original mathematical problem. 
Follow the detailed steps ,and ONLY provide the below four sections in the response to ensure a comprehensive and precise analysis:

1. **Analyze the JSON Structure**:
    - Carefully read each object within the JSON array.
    - Examine the "conditions," "method," and "conclusion" fields in each object.
    - Understand the mathematical concepts and logical reasoning presented in each step.

2. **Extract Key Information**:
    - Identify the main elements involved (e.g., number of family members, types of family members, objects to be arranged).
    - Note any specific constraints or requirements (e.g., certain members must be seated next to each other).
    - Recognize the mathematical methods used (e.g., complementary counting, factorial calculations).

3. **Reconstruct the Problem Statement**:
    - Integrate the extracted information into a coherent and complete problem statement.
    - Ensure that all necessary conditions and requirements are included.
    - Use clear and formal language appropriate for a practice problem.
    - ONLY ask one question based on the final conclusion in the solution.

4. **Verify Consistency and Completeness**:
    - Ensure that the reconstructed problem encompasses all conditions, methods, and conclusions presented in the JSON solution.
    - Confirm that the problem's answer aligns with the conclusions derived in the JSON.

Please perform the analysis based on the following JSON solution and provide the original mathematical problem statement as specified:

[
    {
        "conditions": [
            "The Smith family consists of 4 sons.",
            "The Smith family consists of 3 daughters.",
            "There are 7 chairs arranged in a row.",
            "At least 2 boys must be seated next to each other."
        ],
        "method": "Using complementary counting to focus on counting the arrangements where no two boys are next to each other, which simplifies the problem. Considering one possible arrangement (BGBGBGB) where no two boys are adjacent.",
        "conclusion": "Only one arrangement (BGBGBGB) ensures no two boys are adjacent."
    },
    {
        "conditions": [
            "Only one arrangement (BGBGBGB) ensures no two boys are adjacent."
        ],
        "method": "Calculate the factorial of the number of boys and girls for the specific arrangement BGBGBGB. That is $4!$ for boys and $3!$ for girls.",
        "conclusion": "Total seating arrangements for BGBGBGB is $4! \\times 3! = 144$."
    },
    {
        "conditions": [
            "The Smith family consists of 4 sons.",
            "The Smith family consists of 3 daughters.",
            "There are 7 chairs arranged in a row.",
            "Total seating arrangements for BGBGBGB is $4! \\times 3! = 144$."
        ],
        "method": "Calculate the total number of unrestricted seating arrangements for all 7 children, and subtract the unwanted seating arrangements (where no two boys are next to each other) from it. Total unrestricted arrangements calculated by $7!$.",
        "conclusion": "Desired seating arrangements with at least two boys next to each other is $7! - (4! \\times 3!) = 5040-144 = 4896."
    }
]
'''
base_instructionV2 = """
You are a mathematics expert specializing in simplifying math problems.

Your task is to transform the **Original Problem** into a slightly simpler, more accessible version by focusing on its **solution**. You will simplify the problem by deleting certain initial conditions or peripheral information, allowing the solution to start from an intermediate conclusion, while ensuring the problem remains **correct and solvable** and that the original question and final conclusion are preserved.

Follow these steps to do it:

1. **Analyze Original Solution**:
   - Identify the key concepts and techniques used in the **Original Solution**, step by step.
   - Recognize the initial conditions or peripheral information that can be removed or omitted, allowing the solution to begin from an intermediate result.

2. **Simplification Process**:
   - **Start from Intermediate Conclusion**: Modify the problem so that it provides an intermediate conclusion or result from which the solver can continue to reach the final answer.
   - **Delete Non-Essential Information**: Identify and remove specific initial conditions or peripheral information from the problem that are not essential for reaching the final conclusion if we **start from Intermediate Conclusion**.
   - Ensure that by removing these elements, the problem remains **consistent**, **mathematically sound**, and the **final conclusion** remains unchanged.

3. **Present the Simplified Problem and Solution**:
   - **Simplified Problem**:
     - Rewrite the problem based on the simplified approach, ensuring that the problem starts from the intermediate conclusion.
     - Exclude the deleted initial conditions or peripheral information from the problem.
   - **Simplified Solution**:
     - Provide a clear, correct solution that begins from the intermediate conclusion provided in the simplified problem.
     - Ensure the solution logically leads to the original final conclusion without requiring the deleted initial conditions or peripheral information.

**Format Requirements**:
- Use a numbered list for the sections as shown below.
- Use bold formatting for each section title.
- Do **not** use markdown headings (e.g., ###) or any other formatting styles.
- Ensure that the final answer is enclosed in a LaTeX boxed format.

**Constraints**:
- The simplified problem must remain mathematically sound and consistent with the original problem type.
- The simplification must focus on removing specific initial conditions or peripheral information to reduce the complexity of the problem-solving process.
- The simplified problem and solution should both be clear, complete, and logical, maintaining the integrity of the original question and final conclusion.

**Provide**:
1. **Analyze Original Solution**:
   - List the mathematical ideas and techniques from the **Original Solution**, step by step.
2. **Simplification Process**:
   - Explain which initial conditions or peripheral information were deleted and how the problem now starts from an intermediate conclusion.
3. **Simplified Problem**:
   - Present the simplified version of the original problem, reflecting the removal of initial conditions or peripheral information and the introduction of the intermediate conclusion.
4. **Simplified Solution**:
   - Offer a step-by-step solution to the simplified problem starting from the intermediate conclusion and leading to the final answer.
   - Output the answer in the LaTeX boxed format.
"""
        
def createSimpleQuestionPromptV2(problem, solution):
    prompt = base_instructionV2
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
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
   - Present the simplified solution in a logical sequence, ensuring the format is similar to the **original solution**.

**Format Requirements**:
- Use **bold** formatting for each section title.
- Ensure that the final answer is enclosed in a LaTeX boxed format containing **only the numerical value**.

**Constraints**:
- **Ensure that the simplified problem has a unique answer**.
- **You must change the wording or structure of the original problem statement** enough to reflect the simpler approach.
"""

def createSimpleQuestionPromptV3(problem, solution):
    prompt = base_instructionV3
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt
 
 
base_instructionV4 = """
You are a mathematics expert specializing in simplifying math problems.

Your task is to transform the **Original Problem** into a slightly simpler, more accessible version by focusing on its **sub-conclusion** derived directly from the **given conditions** and the **Original Solution**.

Follow these steps to do it:

1. **Analyze Original Solution**:
   - Identify the key mathematical ideas, techniques, and steps used in the **Original Solution**.
   - Determine the  sub-conclusion from these steps, ensuring it is reached primarily from the provided initial conditions and direct logic—rather than deriving it from the final conclusion.
   - Note any explicit or implicit relationships among variables or constants that are necessary for deriving this  sub-conclusion.

2. **Simplification Process**:
   - **Extract Maximal Sub-Conclusion**: Use the identified sub-conclusion as the question in **Simplified Problem** (i.e., the most substantial intermediate result necessary before reaching the final conclusion).
   - **Retain Essential Conditions**: Preserve any conditions strictly necessary to maintain the correctness and solvability of the problem at the level of the sub-conclusion.
   - Ensure the problem remains **consistent** and **mathematically sound** by focusing on the sub-conclusion and removing only non-essential elements.

3. **Present the Simplified Problem and Solution**:
   - **Simplified Problem**:
     - Formulate the question based on the newly identified sub-conclusion.
     - Ensure that the problem is clearly stated, leaving no room for misinterpretation.
     - Present **only one** single, clear mathematical question asking for ONLY one object.
   - **Simplified Solution**:
     - Provide a concise, correct solution that starts from conditions in **Simplified Problem**.
        

**Format Requirements**:
- Use a numbered list for the sections as shown above.
- Use **bold formatting** for each section title.
- Do **not** use markdown headings (e.g., ###) or any other formatting styles.
- Ensure that the final answer is enclosed in a LaTeX boxed format.
- Ensure that the **Simplified Problem** asks only one question.

**Constraints**:
- The simplified problem must remain mathematically sound and consistent with the original problem type.
- The simplification must focus on extracting the  sub-conclusion that emerges directly from the given conditions (not from the final or a secondary conclusion) and removing only non-essential elements.
- **Do not remove any initial conditions that are essential for deriving the sub-conclusion, including any necessary relationships among variables or constants.**
- The simplified problem and solution should both be clear, complete, and logical, preserving the integrity of the original question.
- The sub-conclusion must be sufficient to generate a reasonable, stand-alone mathematical problem.

**Provide**:
1. **Analyze Original Solution**:
   - List the mathematical ideas and techniques from the **Original Solution** in the order they appear, highlighting where any variables or constants are introduced or used.
2. **Simplification Process**:
   - Explain which initial conditions or details were removed and why.
   - Emphasize how the problem now focuses on the sub-conclusion derived from the original conditions, ensuring any necessary relationships remain.
3. **Simplified Problem**:
   - State the simplified version of the original problem, showing how the necessary variables, constants, and their interrelations remain properly defined.
4. **Simplified Solution**:
   - Give the solution steps starting at the sub-conclusion and making use of any required relationships.
   - End with the final answer in **only one** LaTeX boxed format.
"""
        
def createSimpleQuestionPromptV4(problem, solution):
    prompt = base_instructionV4
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt

def createAnsqerPrompt(problem):
   return f"Problem: {problem}\nProvide a detailed solution and Output the final number of the answer in the latex boxed format."


complexification_prompt = """
You are a mathematics expert specializing in increasing the complexity of math problems.

Your task is to transform the **Original Problem** into a more challenging version by introducing advanced mathematical concepts or techniques.

Please provide the following sections in your answer:

1. **Complexification Process**:
   - Break down the original solution and identify opportunities to introduce more advanced concepts, such as higher-level algebra, calculus, or abstract methods.
   - Clearly explain **how** you are increasing the complexity by adding these techniques or ideas.
   - **Adjust the original problem statement** to reflect the more sophisticated approach.

2. **Complexified Problem**:
   - Provide the **revised** problem statement **without any introductory or explanatory sentences**.

3. **Complexified Solution**:
   - Present the complexified solution in a logical sequence, ensuring that you demonstrate the use of the more advanced concepts or techniques introduced in the new problem statement.

**Format Requirements**:
- Use **bold** formatting for each section title.
- Ensure that the final answer is enclosed in a LaTeX boxed format containing **only the numerical value**.
  
**Constraints**:
- **Ensure that the complexified problem has a unique and challenging answer**.
- **You must change the wording or structure of the original problem statement** enough to reflect the more advanced approach.
"""

complexification_prompt_noprocess = """
You are a mathematics expert specializing in increasing the complexity of math problems.

Your task is to transform the **Original Problem** into a more challenging version by introducing advanced mathematical concepts or techniques.

Please provide the following sections in your answer:

1. **Complexified Problem**:
   - Provide the **revised** problem statement **without any introductory or explanatory sentences**.

2. **Complexified Solution**:
   - Present the complexified solution in a logical sequence, ensuring that you demonstrate the use of the more advanced concepts or techniques introduced in the new problem statement.

**Format Requirements**:
- Use **bold** formatting for each section title.
- Ensure that the final answer is enclosed in a LaTeX boxed format containing **only the numerical value**.

**Constraints**:
- **Ensure that the complexified problem has a unique and challenging answer**.
- **You must change the wording or structure of the original problem statement** enough to reflect the more advanced approach.
"""


def createComplexQuestionPrompt(problem, solution):
    prompt = complexification_prompt_noprocess
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt
