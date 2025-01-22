problem_prompt = """
You are a Mathematics Expert specializing in extracting key information from mathematical problems.

Your task is to extract key mathematical premises and the question from a given problem.

**Instructions:**

1. **Mathematical Premises Extraction:**
   - The `**Mathematical Premises**` section **MUST** be a JSON array formatted in JSON.
   - Each item in the array should be a string that describes a single mathematical premise.
   - Ensure that the mathematical format and language in each premise match those in the `**Original Problem**`.
   - **Do not omit** non-text elements such as tables and code present in the `**Original Problem**`.
   - **Do not omit** any input provided in the `**Original Problem**`.

2. **Extraction Methodology:**
   - Analyze each token in the problem to distinguish between the definition part and the question part within the `**Original Problem**`.
   - Identify each premise within the definition part and add it to the `**Mathematical Premises**` section.
   - Strive to uncover any implied premises present in the `**Original Problem**`.
   - Ensure that each item in the `**Mathematical Premises**` array contains **only one** mathematical premise.

3. **Mathematical Question Extraction:**
   - Populate the `**Mathematical Question**` section with the question part identified from the problem, ensuring it is free of any assumptions or conditions.
   - Ensure that the extracted question does **not** contain any assumptions or conditions. All such conditions should be moved to the `**Mathematical Premises**` section.

4. **Reconstruction Assurance:**
   - Ensure that the original problem can be fully reconstructed using only the extracted `**Mathematical Premises**` and the `**Mathematical Question**`.

**Provide:**
1. **Mathematical Premises**
2. **Mathematical Question**

---
# Example Input:
**Original Problem**:
The Smith family has 4 sons and 3 daughters. In how many ways can they be seated in a row of 7 chairs such that at least 2 boys are next to each other?

# Example Output:
1. **Mathematical Premises:**
[
  "The Smith family has 4 sons.",
  "The Smith family has 3 daughters.",
  "They are to be seated in a row of 7 chairs."
  "At least 2 boys are next to each other."
]
2. **Mathematical Question:**
In how many ways can they be seated?

---
"""
def createProblemPrompt(problem):
    prompt = problem_prompt
    prompt += "\n\n**Original Problem**:\n{}\n".format(problem)
    return prompt

identify_prompt='''
You are a Mathematics Expert specializing in extracting key information during the problem-solving process.

Your task is to extract triples of {premises, process, conclusions} for each step in the given mathematical solution found in `**Original Solution**`.

**Output both sections below and follow these steps to do it**:

1. **Analysis**:
   - **Purpose**: Provide a comprehensive overview of the information extracted from the `**Original Question**`, `**Initial Premises**`, and `**Original Solution**`.
   - **Content to Include**:
     - Brief summary of the original question.
     - Overview of the solution's main steps or strategies.
     - Key mathematical concepts or theorems used.
     - The used premises of each step.
     - Prepare for the extraction of triples.

2. **Extraction of Triples**:
   - Extract detailed premises, process, and conclusions from each step.
   - Ensure each triple is fine-grained.

**Constraints**:
1. **Consistency**:
   - Premises must strictly equal the conclusions from earlier steps or `**Initial Premises**`.
   - Do not reuse former conclusions or initial premises in later conclusions.

2. **Clarity**:
   - Each conclusion should contain only one mathematical result.
   - Identify any implied premises used in each step.
   - Reflect the connections between steps accurately.

3. **Restrictions**:
   - Do not use information outside the corresponding premises for process and conclusions.
   - Process should detail all calculations and results.
   - Conclusions should be clear intermediate results.
   - The final conclusions must fully answer the **Original Question**.
   - Ensure the solution can be reconstructed solely from **Extraction of Triples**.

4. **Output Requirements**:
   - The result in **Extraction of Triples** must be **ONLY** a JSON array without additional text.
   - Each triple should represent a single step and include: 
        - premises: a JSON list of strings containing **ALL** the premises in the corresponding process.
        - process: a string indicating the calculation process.
        - conclusions: a JSON list of strings containing **ALL** the conclusions from corresponding process.

**Constraints**:
- **Do not include any additional text or explanations outside the two sections**.
- Ensure that both **Analysis** and **Extraction of Triples** are present in the output.
'''
example_identify='''
---
# Example Input:
**Initial Premises**:
[
    "The Smith family consists of 4 sons.",
    "The Smith family consists of 3 daughters.",
    "There are 7 chairs arranged in a row.",
    "At least 2 boys must be seated next to each other."
]
**Original Question**:
In how many ways can they be seated?
**Original Solution**:
This problem is a perfect candidate for complementary counting.  It will be fairly difficult to try to count this directly, since there are lots of possible cases (just two are BBBBGGG and BGGBBGB, where B is a boy and G is a girl).  But there is only one way to assign genders to the seating so that no two boys are next to each other, and that is BGBGBGB. If we seat the children as BGBGBGB, then there are $4!$ orderings for the 4 boys, and $3!$ orderings for the 3 girls, giving a total of $4! \\times 3! = 144$ seatings for the 7 children. These are the seatings that we don't want, so to count the seatings that we do want, we need to subtract these seatings from the total number of seatings without any restrictions.  Since there are 7 kids, there are $7!$ ways to seat them. So the answer is $7! - (4! \\times 3!) = 5040-144 = \\boxed{4896}$.

# Example Output:
1. **Analysis**:
- **Brief summary of the original question**: The problem seeks to determine the number of ways the Smith family, consisting of 4 sons and 3 daughters, can be seated on 7 chairs arranged in a row, with the condition that at least two boys must be seated next to each other.
- **Overview of the solution's main steps or strategies**: The solution utilizes the complementary counting technique. It first calculates the total number of unrestricted seating arrangements and then subtracts the number of arrangements where no two boys are seated next to each other to find the desired count.
- **Key mathematical concepts or theorems used**:
   - Permutation: Calculating the number of ways to arrange individuals.
   - Complementary Counting: Finding the desired count by subtracting the count of unwanted arrangements from the total possible arrangements.
- **The used premises of each step**:
   - The Smith family consists of 4 sons and 3 daughters.
   - There are 7 chairs arranged in a row.
   - At least two boys must be seated next to each other.
- **Prepare for the extraction of triples**: The solution involves identifying the total number of seating arrangements, determining the number of arrangements where no two boys are adjacent, and subtracting the latter from the former to obtain the final answer.

2. **Extraction of Triples**:
[
    {
        "premises": [
            "The Smith family consists of 4 sons.",
            "The Smith family consists of 3 daughters.",
            "There are 7 chairs arranged in a row.",
            "At least 2 boys must be seated next to each other."
        ],
        "process": "Calculate the total number of unrestricted seating arrangements as 7! = 5040.",
        "conclusions": [
            "Total number of unrestricted seatings is 5040."
        ]
    },
    {
        "premises": [
            "The Smith family consists of 4 sons.",
            "The Smith family consists of 3 daughters.",
            "There are 7 chairs arranged in a row.",
            "At least 2 boys must be seated next to each other."
        ],
        "process": "Determine the number of seatings where no two boys are adjacent by arranging children in the pattern BGBGBGB and calculate 4! × 3! = 144.",
        "conclusions": [
            "Number of seatings with no two boys adjacent is 144."
        ]
    },
    {
        "premises": [
            "Total number of unrestricted seatings is 5040.",
            "Number of seatings with no two boys adjacent is 144."
        ],
        "process": "Subtract the number of invalid seatings from the total to find the number of valid seatings: 5040 - 144 = 4896.",
        "conclusions": [
            "Number of valid seatings is 4896."
        ]
    }
]

---
'''

def createConstructPrompt(premise,question, solution):
    prompt = identify_prompt + example_identify
    prompt += "\n\n**Initial Premises**:\n{}\n".format(premise)
    prompt += "\n**Original Question**:\n{}\n".format(question)
    prompt += "\n**Original Solution**:\n{}\n".format(solution)
    return prompt



reconstruct_prompt='''
You are a Mathematics Expert specializing in creating clear and coherent problem statements along with detailed solutions.

Your task is to generate a well-structured **Problem** and **Solution** based on the provided `**Initial Premise**` and `**Triples**`.

ONLY output the following sections and adhere to the guidelines below:

1. **Problem**:
   - Craft a precise and concise problem statement using the `**Initial Premise**` and the final conclusion from the `**Triples**`.
   - Ensure the problem is self-contained and clearly states what needs to be found or proven.
   - **Avoid introducing unnecessary elements or terminology that are not essential to the problem. Specifically, do not mention concepts like "series" or other irrelevant terms.**

2. **Solution**:
   - Develop a step-by-step solution to the problem, drawing from the `process` described in each triple of the `**Triples**`.
   - Present the solution in a logical and natural mathematical narrative, suitable for educational purposes.
   - Include all necessary calculations, justifications, and conclusions as outlined in the `**Triples**`.
   - **Ensure the solution focuses solely on the essential mathematical steps without incorporating extraneous concepts or terminology.**
   - Output the final answer in the $\boxed{}$

**Constraints**:

1. **Structure**:
   - The output must consist of only two sections: **Problem** and **Solution**.
   - Use appropriate mathematical notation and formatting for clarity.

2. **Consistency**:
   - The **Problem** must be directly derived from the `**Initial Premise**` and the final conclusion in the `**Triples**`.
   - The **Solution** must logically follow the steps outlined in the `process` of each triple, ensuring a coherent flow from start to finish.

3. **Clarity**:
   - Both the **Problem** and **Solution** should be clearly and precisely written, making them easy to understand.
   - Avoid unnecessary complexity; focus on presenting the information in an accessible manner.

4. **Restrictions**:
   - Do not introduce any new information beyond what is provided in the `**Initial Premise**` and `**Triples**`.
   - **Do not include irrelevant terminology or concepts unless they are essential to the problem.**
   - Ensure all mathematical expressions are correctly formatted and accurately reflect the intended calculations.

5. **Output Requirements**:
   - The output must strictly include only the **Problem** and **Solution** sections, without any additional text or commentary.
   - Maintain a clear separation between the **Problem** and **Solution** sections for readability.

'''
def createReConstructPrompt(Initial_premises,Triples):
    prompt = reconstruct_prompt
    prompt += "\n\n**Initial Premises**:\n{}\n".format(Initial_premises)
    prompt += "\n**Triples**:\n{}\n".format(Triples)
    return prompt
 
check_prompt='''
You are an expert in mathematical problem verification.

Your task is to evaluate the given **Problem** and **Solution** based on the following criteria:

1. **Problem Validity**:
   - Determine if the **Problem** is clearly and reasonably stated.
   - Ensure that it poses a well-defined mathematical question with a specific, unambiguous answer.

2. **Solution Integrity**:
   - Check if the **Solution** relies solely on information and premises provided in the **Problem**.
   - Identify if any external assumptions, theorems, or conclusions not mentioned in the **Problem** are used.

**Output your evaluation in the following structured format**:
1. **Check1**: PASS or FAIL 
2. **Check2**: PASS or FAIL
3. **Analysis**:
   - Problem Validity: [Provide a brief explanation supporting PASS or FAIL]
   - Solution Integrity: [Provide a brief explanation supporting PASS or FAIL]

**Constraints**:
- **Do not include any additional text or sections** beyond the specified format.
- Ensure that the keywords `Check1`, `Check2`, and `Analysis` are present exactly as shown.
- The explanations under `Analysis` should be concise and directly related to each check.

'''
def createCheckPrompt(problem,solution):
    prompt = check_prompt
    prompt += "\n\n**Problem**:\n{}\n".format(problem)
    prompt += "\n**Solution**:\n{}\n".format(solution)
    return prompt
 
