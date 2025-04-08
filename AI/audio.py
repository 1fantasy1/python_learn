
from http import HTTPStatus
import os
from dashscope.audio.asr import Recognition
import dashscope

dashscope.api_key = "sk-02a5e836dbaa436592ac4f4881a2b5a9"


def main():
    try:
        recognition = Recognition(
            model='paraformer-realtime-v2',
            format='flac',  # 与实际文件格式一致
            sample_rate=16000,
            language='ja',  # 显式指定日语
            callback=None
        )

        # ✅ 直接传递文件路径字符串（保持原始调用方式）
        file_path = r'1.flac'
        result = recognition.call(file_path)

        if result.status_code == HTTPStatus.OK:
            print('识别结果:')
            print(result.get_sentence())
        else:
            print(f'错误代码 {result.status_code}: {result.message}')

        # 显示性能指标
        print(f'[追踪信息] 请求ID: {recognition.get_last_request_id()}')
        print(f'首包延迟: {recognition.get_first_package_delay()}ms')
        print(f'末包延迟: {recognition.get_last_package_delay()}ms')

    except Exception as e:
        print(f'发生异常: {str(e)}')


if __name__ == "__main__":
    main()
