from openai import OpenAI
import openai
import time
from util.config import OPENAI_API_KEY
import os
import time
import logging
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
client = OpenAI(api_key=OPENAI_API_KEY)
def get_oai_completion(prompt,model,temperature):
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error fetching answer: {e}")
        return None

def call_chatgpt(prompt,model):
    success = False
    re_try_count = 10
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion(prompt,model)
            success = True
        except:
            time.sleep(5)
            print('retry for sample:', prompt)
    return ans


def get_answer_from_chat_model(prompt, logger=None, eng='gpt-3.5-turbo', temperature=0.0, timeout=20, max_try=3):
    """
    向聊天模型发送单个请求，并返回回答。

    Args:
        prompt (str): 提示词。
        logger (logging.Logger): 日志记录器。
        eng (str): 使用的模型名称。
        temperature (float): 温度参数。
        timeout (int): 请求超时时间（秒）。
        max_try (int): 最大重试次数。

    Returns:
        str: 模型的回答。
    """
    if eng not in [
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",
        "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo-1106","gpt-4o"
    ]:
        raise ValueError(f"Unsupported model: {eng}")

    is_success = False
    num_exception = 0

    while not is_success:
        if max_try > 0 and num_exception >= max_try:
            logger.error(f"Max retries reached for question: {q}") if logger else None
            return ""
        try:
            response = get_oai_completion(prompt,eng,temperature)
            return response
        except Exception as e:
            num_exception += 1
            sleep_time = min(num_exception, 2)
            if logger:
                is_print_exc = num_exception % 10 == 0
                logger.error(f"Exception for question '{q}': {e}", exc_info=is_print_exc)
                logger.info(f"Retry {num_exception}/{max_try} after sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            is_success = False

def wrapper(idx_args, func):
    """
    包装函数，用于多进程返回索引和结果。

    Args:
        idx_args (tuple): (索引, 参数)
        func (callable): 要调用的函数。

    Returns:
        tuple: (索引, 结果)
    """
    idx, args = idx_args
    res = func(args)
    return idx, res

def batch_get_chat_api(examples, eng, pre_fun, post_fun,
                       logger=None, n_processes=4, temperature=0.7, timeout=20, max_try=3, **kwargs):
    """
    批量处理聊天模型的 API 请求。

    Args:
        examples (list): 示例数据列表，每个元素是包含 'question' 键的字典。
        eng (str): 使用的模型名称。
        pre_fun (callable): 前处理函数。
        post_fun (callable): 后处理函数。
        logger (logging.Logger): 日志记录器。
        n_processes (int): 使用的进程数。
        temperature (float): 温度参数。
        timeout (int): 请求超时时间（秒）。
        max_try (int): 最大重试次数。
        **kwargs: 其他可选参数。

    Returns:
        None
    """
    get_answer_func = partial(
        get_answer_from_chat_model,
        logger=logger,
        eng=eng,
        temperature=temperature,
        timeout=timeout,
        max_try=max_try,
        **kwargs
    )

    prompts = [f"{pre_fun(example)}" for example in examples]

    idx2res = {}
    with Pool(n_processes) as pool:
        tasks = enumerate(prompts)
        wrapped_func = partial(wrapper, func=get_answer_func)
        for idx, response in tqdm(pool.imap_unordered(wrapped_func, tasks), total=len(prompts), desc="Processing"):
            idx2res[idx] = response

    for idx, example in enumerate(examples):
        post_fun(example, idx2res.get(idx, ""))