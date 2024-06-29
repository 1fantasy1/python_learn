"""装载问题回溯算法"""
def backtracking_loading(weights, max_weight):
    """
    装载问题的回溯算法实现

    参数:
    weights (list): 每个货物的重量列表
    max_weight (int): 卡车的最大载重量

    返回:
    tuple: 最大装载重量和装载方案的布尔列表，每个位置表示对应货物是否装载
    """
    n = len(weights)
    current_solution = [False] * n
    best_solution = [False] * n
    max_loaded_weight = 0

    def backtrack(index, current_weight):
        nonlocal max_loaded_weight

        if index == n:
            if current_weight > max_loaded_weight:
                max_loaded_weight = current_weight
                best_solution[:] = current_solution[:]
            return

        # 尝试不装载第index个货物
        backtrack(index + 1, current_weight)

        # 尝试装载第index个货物
        if current_weight + weights[index] <= max_weight:
            current_solution[index] = True
            backtrack(index + 1, current_weight + weights[index])
            current_solution[index] = False

    backtrack(0, 0)
    return max_loaded_weight, best_solution


# 示例使用
weights = [10, 20, 30, 40, 50]
max_weight = 100

max_loaded_weight, solution = backtracking_loading(weights, max_weight)

print(f"最大装载重量为: {max_loaded_weight}")
print("装载方案为:", solution)