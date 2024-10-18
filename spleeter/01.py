from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
#模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('iic/speech_frcrn_ans_cirm_16k')

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
result = ans(
    'E:\python\Python documentation\python_learn\spleeter\水槽のブランコ.flac',
    output_path='01.wav')