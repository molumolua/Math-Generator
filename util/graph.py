import networkx as nx
import matplotlib.pyplot as plt
import json
import sys
from collections import Counter
import random

# 定义字符串（确保使用原始字符串以避免转义字符的问题）
data_str = r'''
[
    {
        "premises": [
            "The series is a geometric series with first term 1 and common ratio 7",
            "The series has 2005 terms, from 7^0 to 7^{2004}"
        ],
        "process": "Apply the formula for the sum of a geometric series, S_n = a \\frac{r^n - 1}{r - 1}, where a is the first term, r is the common ratio, and n is the number of terms.",
        "conclusions": ["1 + 7 + 7^2 + \\cdots + 7^{2004} = \\frac{7^{2005}-1}{6}"]
    },
    {
        "premises": [
            "We are interested in the remainder when the sum of this series is divided by 1000",
            "1 + 7 + 7^2 + \\cdots + 7^{2004} = \\frac{7^{2005}-1}{6}"
        ],
        "process": "Use Fermat-Euler's theorem, noting that \\varphi(1000) = 400 to simplify 7^{2005} mod 1000. 7^{2005} \\equiv 7^{400 \\cdot 5 + 5} \\pmod{1000} simplifies to 7^5 due to 7^{400} \\equiv 1 \\pmod{1000}.",
        "conclusions": ["\\frac{7^{2005}-1}{6} \\equiv \\frac{7^5 - 1}{6} \\pmod{1000}"]
    },
    {
        "premises": [
            "\\frac{7^{2005}-1}{6} \\equiv \\frac{7^5 - 1}{6} \\pmod{1000}"
        ],
        "process": "Calculate 7^5 - 1 and then divide by 6, finally applying modulo 1000 to the result.",
        "conclusions": ["The remainder when 1 + 7 + 7^2 + \\cdots + 7^{2004} is divided by 1000 is 801"]
    }
]
'''

def str2graph(data_str, show=False, show_seed=2025):

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError as e:
        # 捕获错误后，替换单个反斜杠为双反斜杠
        try:
            fixed_json_str = data_str.replace('\\', '\\\\')
            data = json.loads(fixed_json_str)
        except json.JSONDecodeError as e:
            print("str2graph",data_str)
            sys.exit(1)

    if show:
        # 查看解析后的数据
        for step in data:
            print("前提:", step["premises"])
            print("方法:", step["process"])
            print("结论:", step["conclusions"])
            print("-" * 50)

    # 创建一个有向图
    G = nx.DiGraph()

    # 遍历每个步骤，添加节点和边
    for step in data:
        conclusions = step["conclusions"]
        premises = step["premises"]
        process = step.get("process", "")

        for conclusion in conclusions:
            G.add_node(conclusion, type='conclusion')

            for premise in premises:
                G.add_node(premise, type='premise')
                # 添加边，并存储process作为边的属性
                G.add_edge(premise, conclusion, process=process)
    if show:
        # 图初始设置
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常
        plt.figure(figsize=(14, 10))
        # 布局
        pos = nx.spring_layout(G, k=0.5, seed=show_seed) 
        # 绘制点
        nodes = [node for node, attr in G.nodes(data=True) if attr['type'] in ['node', 'premise', 'conclusion']]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='lightblue', node_shape='s', node_size=300)
        # 绘制边
        edges = G.edges(data=True)
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->', arrowsize=20, edge_color='gray')
        # 绘制标签
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_family='sans-serif')

        # 创建图例
        import matplotlib.patches as mpatches
        premise_patch = mpatches.Patch(color='lightblue', label='Premise')
        conclusion_patch = mpatches.Patch(color='lightblue', label='Conclusion')
        plt.legend(handles=[premise_patch, conclusion_patch], fontsize=12)

        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return G

def constrcutDatafromMethod(data_str,selected_process):
    steps = []
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError as e:
        # 捕获错误后，替换单个反斜杠为双反斜杠
        try:
            fixed_json_str = data_str.replace('\\', '\\\\')
            data = json.loads(fixed_json_str)
        except json.JSONDecodeError as e:
            print("construct Data from Method",data_str)
            sys.exit(1)
     # 遍历每个步骤，添加节点和边
    if data[-1].get("process", "") in selected_process:
        return None
    for step in data:
        conclusions = step["conclusions"]
        premises = step["premises"]
        process = step.get("process", "")
        if process not in selected_process:
            step = {
                "premises": premises,
                "process": process,
                "conclusions": conclusions
            }
            steps.append(step)
    json_str = json.dumps(steps, ensure_ascii=False, indent=4)
    return json_str

def findEraseProcess(G,enable_random=False):
    # 1. 找到所有入度为0的节点
    source_nodes = [node for node, deg in G.in_degree() if deg == 0]

    # print("入度为0的节点（源节点）:")
    # for node in source_nodes:
    #     print(f" - {node}")

    # 2. 找到出度最小的入度为0的节点
    if not source_nodes:
        return None
    else:
        # 计算这些源节点的出边 (去重)
        source_out = dict()
        for node in source_nodes:
            # 获取所有出边的process属性并去重
            unique_processes = set()
            for succ in G.successors(node):
                process = G.edges[node, succ].get('process')
                if process is not None:
                    unique_processes.add(process)
            source_out[node] = unique_processes
        min_out_degree = min([len(process) for process in source_out.values()])
        min_out_degree_nodes = [node for node, process in source_out.items() if len(process) == min_out_degree]

        all_processes = ['@'.join(sorted(list(source_out[node]))) for node in min_out_degree_nodes]
        
        # # 计算这些源节点的出度 (不去重)
        # source_out_degrees = {node: G.out_degree(node) for node in source_nodes}

        # # 找出最小的出度(为了尽可能少的删除步骤)
        # min_out_degree = min(source_out_degrees.values())
        # # 找出所有具有最小出度的源节点
        # min_out_degree_nodes = [node for node, deg in source_out_degrees.items() if deg == min_out_degree]
        

        # # 3. 对于每个这样的节点，列出其直接连接的节点及对应的process
        # # 并收集所有相关的process
        # all_processes = []
        # for node in min_out_degree_nodes:
        #     successors = list(G.successors(node))
        #     process_list=[]
        #     for succ in successors:
        #         process = G.edges[node, succ]['process']
        #         process_list.append(process)
        #     all_processes.append('@'.join(process_list))
        # if len(all_processes)<1:
        #     return None
        
        return [process.split("@") for process in list(set(all_processes))]
        
        # if enable_random:
        #     all_processes=[process.split("@") for process in list(set(all_processes))]
        #     return random.choice(all_processes)
        
        # process_counter = Counter(all_processes)
        # min_freq = min(process_counter.values())
        # # 找出所有具有最小频率的process (对这个问题变化最少)
        # most_common_processes = [process.split("@") for process, count in process_counter.items() if count == min_freq]
        # selected_process = random.choice(most_common_processes)
        # return selected_process

# # 构建图
# G = str2graph(data_str, show=False, show_seed=2025)
# selected_process=findEraseProcess(G,enable_random=True)
# simplified_data_str=constrcutDatafromMethod(data_str,selected_process)
# simplified_G = str2graph(simplified_data_str,show=False)
# simplified_premise=[node for node, deg in simplified_G.in_degree() if deg == 0]


