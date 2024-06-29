"""改进顺序检索"""
def bidirectional_sequential_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        if arr[left] == target:
            return left
        if arr[right] == target:
            return right
        left += 1
        right -= 1
    return -1

# 示例使用
arr = [5, 3, 7, 1, 9]
target = 7
result = bidirectional_sequential_search(arr, target)
if result != -1:
    print(f"目标元素 {target} 在索引 {result} 处找到。")
else:
    print(f"目标元素 {target} 不在数组中。")
