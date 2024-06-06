from faster_whisper import WhisperModel
import torch
import math
# from transformers import WhisperTokenizer, WhisperForConditionalGeneration
#
# model_name = "Systran/faster-whisper-large-v3"
#
# tokenizer = WhisperTokenizer.from_pretrained("./faster-whisper-large-v3", local_files_only=True)
# model = WhisperForConditionalGeneration.from_pretrained("./faster-whisper-large-v3", local_files_only=True)
from transformers import WhisperTokenizer, WhisperForConditionalGeneration

model_name = "Systran/faster-whisper-large-v3"

# Set local_files_only to False to allow downloading the model 最优化理论 it's not found locally
tokenizer = WhisperTokenizer.from_pretrained(model_name, local_files_only=False)
model = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=False)

def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = math.floor((seconds % 1) * 1000)
    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
    return output
# 制作字幕文件
def make_srt(file_path, model_name="small"):
    # 判断是否有可用的CUDA设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 如果有CUDA设备，使用CUDA和float16计算类型，否则使用CPU和int8计算类型
    if device == "cuda":
        model = WhisperModel(model_name, device="cuda", compute_type="float16", download_root="./model_from_whisper",
                             local_files_only=False)
    else:
        model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root="./model_from_whisper",
                             local_files_only=False)
    # 或者可以在GPU上运行，使用INT8计算类型
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    # 转录音频文件，使用beam size为5
    segments, info = model.transcribe(file_path, beam_size=5)

    # 打印检测到的语言和概率
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    count = 0
    # 打开文件以写入字幕
    with open('./video.srt', 'w') as f:
        for segment in segments:
            count += 1
            duration = f"{convert_seconds_to_hms(segment.start)} --> {convert_seconds_to_hms(segment.end)}\n"
            text = f"{segment.text.lstrip()}\n\n"

            # 将格式化的字符串写入文件
            f.write(f"{count}\n{duration}{text}")
            print(f"{duration}{text}", end='')

    # 以只读方式打开文件以读取字幕内容
    with open("./video.srt", 'r', encoding="utf-8") as file:
        srt_data = file.read()

    return "转写完毕"

# 测试 make_srt 函数
result = make_srt("output.wav")
print(result)
