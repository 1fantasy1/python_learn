import re
from docx import Document


def load_questions_from_docx(file_path):
    """
    从 docx 文件加载题库并解析为结构化数据
    """
    doc = Document(file_path)
    questions = []
    current_type = None
    current_question = None
    is_collecting_options = False

    for paragraph in doc.paragraphs:
        line = paragraph.text.strip()

        # 判断题型
        if line.startswith("一、单项选择题"):
            current_type = "single"
            continue
        elif line.startswith("二、多项选择题"):
            current_type = "multiple"
            continue
        elif line.startswith("三、判断题"):
            current_type = "judge"
            continue

        # 判断题目
        if re.match(r"^\d+、", line):  # 题目开头
            if current_question:  # 保存上一道题
                questions.append(current_question)
            current_question = {
                "type": current_type,
                "question": re.sub(r"^\d+、", "", line),
                "options": [],
                "answer": ""
            }
            is_collecting_options = True  # 准备收集选项
            continue

        # 判断选项
        if is_collecting_options and re.match(r"^[A-D]\.", line):  # 选项
            current_question["options"].append(line)
            continue

        # 判断答案
        if line.startswith("正确答案："):
            answer = line.replace("正确答案：", "").strip()
            if current_type == "multiple":
                current_question["answer"] = answer.split("、")
            elif current_type == "judge":
                current_question["answer"] = True if answer in ["A", "正确"] else False
            else:
                current_question["answer"] = answer
            is_collecting_options = False  # 停止收集选项

    if current_question:  # 保存最后一道题
        questions.append(current_question)

    return questions


def start_quiz(questions):
    """
    开始答题（命令行形式）
    """
    score = 0
    print("欢迎使用答题系统！\n")

    for idx, q in enumerate(questions, 1):
        print(f"题目 {idx}: {q['question']}")

        if q["type"] in ("single", "multiple"):
            for option in q["options"]:
                print(option)

        # 根据题目类型进行不同的答题交互
        if q["type"] == "single":  # 单选题
            user_answer = input("请输入你的答案 (A/B/C/D): ").strip().upper()
            if user_answer == q["answer"]:
                print("答对了！\n")
                score += 1
            else:
                print(f"答错了！正确答案是 {q['answer']}。\n")

        elif q["type"] == "multiple":  # 多选题
            user_input = input("请输入你的答案 (多个选项用逗号分隔，如 A,B): ").strip().upper()
            user_answer = [ans.strip() for ans in user_input.split(',')]
            # 判断与正确答案集合是否完全一致
            if set(user_answer) == set(q["answer"]):
                print("答对了！\n")
                score += 1
            else:
                print(f"答错了！正确答案是 {','.join(q['answer'])}。\n")

        elif q["type"] == "judge":  # 判断题
            user_input = input("请输入你的答案(正确/错误 或 True/False): ").strip()
            if user_input in ["True", "正确", "A"]:
                user_answer = True
            else:
                user_answer = False
            if user_answer == q["answer"]:
                print("答对了！\n")
                score += 1
            else:
                correct_str = "正确" if q["answer"] else "错误"
                print(f"答错了！正确答案是 {correct_str}。\n")

    print(f"答题结束！你的总分是：{score}/{len(questions)}")

if __name__ == "__main__":
    file_path = "机器学习练习题.docx"
    question_bank = load_questions_from_docx(file_path)
    start_quiz(question_bank)