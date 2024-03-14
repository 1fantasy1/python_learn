def find_unique_numbers(numbers):
    unique_numbers = []
    for num in numbers:
        if num not in unique_numbers:
            unique_numbers.append(num)
    return unique_numbers


def main():
    numbers = input("请输入一些数字，用空格分隔：").split()
    numbers = [int(num) for num in numbers]  # 将输入的字符串转换为整数列表
    unique_numbers = find_unique_numbers(numbers)

    if unique_numbers:
        print("唯一数字为：", unique_numbers)
    else:
        print("没有唯一数字。")


if __name__ == "__main__":
    main()
