"""最小生成树"""
import heapq

def prim(graph):
    """
    普里姆算法求解最小生成树问题

    参数:
    graph (dict): 图的邻接表表示，键为节点，值为列表，表示与该节点相邻的节点和边的权重。

    返回:
    list: 最小生成树的边列表，每个元素为 (节点1, 节点2, 权重) 的三元组。
    """
    min_heap = []   # 最小堆，用于选择最小权重的边
    visited = set() # 记录已经访问的节点
    minimum_spanning_tree = [] # 最小生成树的边集合

    # 选择起始节点（这里假设从节点0开始）
    start_node = 0
    visited.add(start_node)

    # 将起始节点的所有边加入最小堆
    for neighbor, weight in graph[start_node]:
        heapq.heappush(min_heap, (weight, start_node, neighbor))

    while min_heap:
        weight, u, v = heapq.heappop(min_heap)

        if v not in visited:
            visited.add(v)
            minimum_spanning_tree.append((u, v, weight))

            # 将新加入节点的所有边加入最小堆
            for neighbor, weight in graph[v]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (weight, v, neighbor))

    return minimum_spanning_tree

# 示例图的邻接表表示：键为节点，值为与之相邻的节点和边的权重
graph = {
    0: [(1, 2), (2, 1)],
    1: [(0, 2), (2, 4), (3, 5)],
    2: [(0, 1), (1, 4), (3, 3)],
    3: [(1, 5), (2, 3)]
}

# 调用普里姆算法求解最小生成树
mst = prim(graph)

# 打印最小生成树的边
print("最小生成树 (Prim算法):")
for edge in mst:
    print(edge)