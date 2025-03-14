import subprocess

input_file = "1.ts"
output_file = "output.mp4"

# FFmpeg 命令，使用 NVENC 硬件加速
command = [
    "ffmpeg", "-hwaccel", "cuda",  # 使用 CUDA 硬件加速
    "-i", input_file,             # 输入文件
    "-c:v", "h264_nvenc",         # 使用 NVIDIA 的 NVENC 编码器
    "-preset", "fast",            # 设置编码速度（fast、medium、slow）
    "-c:a", "aac",                # 音频编码器
    output_file                   # 输出文件
]

# 执行命令
try:
    subprocess.run(command, check=True)
    print("转换完成！")
except subprocess.CalledProcessError as e:
    print(f"转换失败: {e}")
