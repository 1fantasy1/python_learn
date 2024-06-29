"""最大子段和"""
def max_subarray_sum(nums):
    if not nums:
        return 0

    current_max = global_max = nums[0]

    for num in nums[1:]:
        current_max = max(num, current_max + num)
        global_max = max(global_max, current_max)

    return global_max

# 示例使用
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(f"最大子段和为: {max_subarray_sum(nums)}")
