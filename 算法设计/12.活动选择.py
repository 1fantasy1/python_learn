"""活动选择"""
def activity_selection(start, end):
    activities = sorted(zip(start, end), key=lambda x: x[1])  # 按结束时间排序
    selected_activities = []

    if activities:
        selected_activities.append(activities[0])
        last_end_time = activities[0][1]

        for activity in activities[1:]:
            if activity[0] >= last_end_time:  # 当前活动的开始时间大于等于上一个选中活动的结束时间
                selected_activities.append(activity)
                last_end_time = activity[1]

    return selected_activities


# 示例使用
start_times = [1, 3, 0, 5, 8, 5]
end_times = [2, 4, 6, 7, 9, 9]
selected = activity_selection(start_times, end_times)

print("最大的互不相交活动集合为:")
for activity in selected:
    print(activity)
