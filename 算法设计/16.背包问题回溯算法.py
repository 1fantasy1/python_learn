"""背包问题回溯算法"""
def knapsack_backtracking(weights, values, capacity):
    """
    0/1背包问题的回溯算法实现

    参数:
    weights (list): 每种物品的重量列表
    values (list): 每种物品的价值列表
    capacity (int): 背包的容量限制

    返回:
    tuple: 最大总价值和装载方案的布尔列表，每个位置表示对应物品是否装入背包
    """
    n = len(weights)
    current_solution = [False] * n
    best_solution = [False] * n
    max_value = 0

    def backtrack(index, current_weight, current_value):
        nonlocal max_value

        if index == n:
            if current_weight <= capacity and current_value > max_value:
                max_value = current_value
                best_solution[:] = current_solution[:]
            return

        # 尝试不装入第index个物品
        backtrack(index + 1, current_weight, current_value)

        # 尝试装入第index个物品
        if current_weight + weights[index] <= capacity:
            current_solution[index] = True
            backtrack(index + 1, current_weight + weights[index], current_value + values[index])
            current_solution[index] = False

    backtrack(0, 0, 0)
    return max_value, best_solution


# 示例使用
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

max_value, solution = knapsack_backtracking(weights, values, capacity)

print(f"背包的最大总价值为: {max_value}")
print("装载方案为:", solution)
