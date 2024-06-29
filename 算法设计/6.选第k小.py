"""选第k小"""
def quickselect(arr, low, high, k):
    if low == high:
        return arr[low]

    pivot_index = partition(arr, low, high)

    if pivot_index == k:
        return arr[pivot_index]
    elif pivot_index > k:
        return quickselect(arr, low, pivot_index - 1, k)
    else:
        return quickselect(arr, pivot_index + 1, high, k)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# 示例使用
arr = [3, 6, 8, 10, 1, 2, 1]
k = 4  # 选择第4小的元素（索引从0开始，即第5个元素）
print(f"数组: {arr}")
result = quickselect(arr, 0, len(arr) - 1, k)
print(f"数组中的第{k + 1}小的元素是: {result}")
