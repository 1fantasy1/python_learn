""" 顺序检索 """
def sequential_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 示例使用
arr = [5, 3, 7, 1, 9]
target = 7
result = sequential_search(arr, target)
if result != -1:
    print(f"目标元素 {target} 在索引 {result} 处找到。")
else:
    print(f"目标元素 {target} 不在数组中。")