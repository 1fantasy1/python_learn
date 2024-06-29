"""二分检索"""
def binary_search_iterative(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1

# 示例使用
arr = [1, 3, 5, 7, 9, 11, 13]
target = 7
result = binary_search_iterative(arr, target)
if result != -1:
    print(f"目标元素 {target} 在索引 {result} 处找到。")
else:
    print(f"目标元素 {target} 不在数组中。")
