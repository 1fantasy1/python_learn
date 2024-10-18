import torch
import whisper
from whisper.utils import get_writer

audio_path = "E:\python\Python documentation\python_learn\AI\水槽のブランコ.flac"
name = "large-v2"

model = whisper.load_model(name, download_root="./models")

result = model.transcribe(audio_path, language=None, initial_prompt=None, verbose=True)

output_format = "srt"
output_dir = "E:\python\Python documentation\python_learn\AI"
file_name = "test05 .srt"

writer = get_writer(output_format, output_dir)
writer(result, file_name)


