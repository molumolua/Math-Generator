from prompt.openai_access import get_oai_completion
from prompt.prompt_design import createAnsqerPrompt
from util.util import reject_sample
def reject_sample_check(problem, solution, model, logger):
    try:
        logger.debug(f"Checking reject sample for problem: {problem}")
        prompt = createAnsqerPrompt(problem)
        gpt_answer = get_oai_completion(prompt, model=model)
        if gpt_answer:
            result = reject_sample(gpt_answer, solution)
            logger.debug(f"Reject sample result: {result}")
            return result
        else:
            logger.warning("GPT answer is empty.")
            return False
    except Exception as e:
        logger.error(f"Error in reject_sample_check: {e}")
        return False
