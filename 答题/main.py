from docx import Document
import re


def load_questions_from_docx(file_path):
    """
    从 docx 文件加载题库并解析为结构化数据
    """
    doc = Document(file_path)
    questions = []
    current_type = None
    current_question = None

    for paragraph in doc.paragraphs:
        line = paragraph.text.strip()

        # 判断题型（单选、多选、判断）
        if line.startswith("一、单项选择题"):
            current_type = "single"
        elif line.startswith("二、多项选择题"):
            current_type = "multiple"
        elif line.startswith("三、判断题"):
            current_type = "judge"
        elif re.match(r"^\d+、", line):  # 题目开头
            if current_question:  # 保存上一道题
                questions.append(current_question)
            current_question = {"type": current_type, "question": "", "options": [], "answer": ""}
            question_text = re.sub(r"^\d+、", "", line)
            current_question["question"] = question_text
        elif re.match(r"^[A-D]?\.\s?", line):  # 选项
            current_question["options"].append(line)
        elif line.startswith("正确答案："):  # 答案
            answer = line.replace("正确答案：", "").strip()
            if current_type == "multiple":
                current_question["answer"] = answer.split("、")
            elif current_type == "judge":
                current_question["answer"] = True if answer in ["A", "正确"] else False
            else:
                current_question["answer"] = answer

    if current_question:  # 保存最后一道题
        questions.append(current_question)

    return questions


def start_quiz(questions):
    """
    开始答题
    """
    score = 0
    print("欢迎使用答题系统！\n")

    for idx, q in enumerate(questions, 1):
        print(f"题目 {idx}: {q['question']}")

        if q["type"] in ["single", "multiple"]:
            for option in q["options"]:
                print(option)

        if q["type"] == "single":
            user_answer = input("请输入你的答案 (A/B/C/D): ").strip().upper()
            if user_answer == q["answer"]:
                print("答对了！\n")
                score += 1
            else:
                print(f"答错了！正确答案是 {q['answer']}。\n")

        elif q["type"] == "multiple":
            user_answer = input("请输入你的答案 (多个选项用逗号分隔，如 A,B): ").strip().upper().split(',')
            if set(user_answer) == set(q["answer"]):
                print("答对了！\n")
                score += 1
            else:
                print(f"答错了！正确答案是 {','.join(q['answer'])}。\n")

        elif q["type"] == "judge":
            user_answer = input("请输入你的答案 (True/False): ").strip().capitalize()
            if (user_answer == "True" and q["answer"]) or (user_answer == "False" and not q["answer"]):
                print("答对了！\n")
                score += 1
            else:
                print(f"答错了！正确答案是 {q['answer']}。\n")

    print(f"答题结束！你的总分是：{score}/{len(questions)}")


if __name__ == "__main__":
    # 替换为您的文件路径
    file_path = "机器学习练习题.docx"
    question_bank = load_questions_from_docx(file_path)
    start_quiz(question_bank)
