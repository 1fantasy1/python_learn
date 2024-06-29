"""最大延迟调度"""
def max_lateness_scheduling(tasks):
    # 排序任务按照截止时间
    sorted_tasks = sorted(tasks, key=lambda x: x[2])  # 按照截止时间排序

    current_time = 0
    max_lateness = 0
    schedule = []

    for task in sorted_tasks:
        start_time = current_time
        finish_time = current_time + task[1]
        lateness = max(0, finish_time - task[2])  # 计算延迟时间
        max_lateness = max(max_lateness, lateness)
        schedule.append((task[0], start_time, finish_time, lateness))
        current_time = finish_time

    return max_lateness, schedule

# 示例使用
tasks = [
    ('Task1', 3, 6),  # (任务名, 执行时间, 截止时间)
    ('Task2', 2, 8),
    ('Task3', 5, 7),
    ('Task4', 4, 10),
    ('Task5', 2, 5)
]

max_lateness, schedule = max_lateness_scheduling(tasks)

print(f"最大延迟时间为: {max_lateness}")
print("任务调度表:")
print("任务名\t开始时间\t完成时间\t延迟时间")
for task in schedule:
    print(f"{task[0]}\t{task[1]}\t\t{task[2]}\t\t{task[3]}")
