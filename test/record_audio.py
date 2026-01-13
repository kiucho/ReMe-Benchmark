#!/usr/bin/env python3
"""
macOS 麦克风录音脚本
需要安装: pip install pyaudio wave
"""

import pyaudio
import wave
import sys
import os
from datetime import datetime


class AudioRecorder:
    """macOS 音频录制器"""

    def __init__(self, output_dir="recordings"):
        """
        初始化录音器

        Args:
            output_dir: 录音文件保存目录
        """
        self.output_dir = output_dir
        self.chunk = 1024  # 每次读取的音频块大小
        self.format = pyaudio.paInt16  # 16位深度
        self.channels = 1  # 单声道
        self.rate = 44100  # 采样率 44.1kHz

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def record(self, duration=5, filename=None):
        """
        录制音频

        Args:
            duration: 录制时长（秒）
            filename: 输出文件名，如果为None则自动生成

        Returns:
            str: 保存的文件路径
        """
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"

        filepath = os.path.join(self.output_dir, filename)

        # 初始化PyAudio
        audio = pyaudio.PyAudio()

        try:
            # 打开音频流（这会触发macOS的麦克风权限请求）
            print("正在请求麦克风权限...")
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            print(f"开始录音，时长: {duration} 秒")
            print("录音中...")

            frames = []

            # 录制音频
            for i in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)

                # 显示进度
                progress = (i + 1) / (self.rate / self.chunk * duration) * 100
                sys.stdout.write(f"\r进度: {progress:.1f}%")
                sys.stdout.flush()

            print("\n录音完成！")

            # 停止并关闭流
            stream.stop_stream()
            stream.close()

            # 保存为WAV文件
            print(f"正在保存到: {filepath}")
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"✓ 文件已保存: {filepath}")
            return filepath

        except Exception as e:
            print(f"\n错误: {e}")
            print("\n提示:")
            print("1. 请确保已安装 pyaudio: pip install pyaudio")
            print("2. 在macOS上，首次运行会弹出权限请求对话框")
            print("3. 如果权限被拒绝，请前往 系统偏好设置 > 安全性与隐私 > 隐私 > 麦克风")
            return None

        finally:
            audio.terminate()

    def record_interactive(self):
        """交互式录音"""
        print("=" * 50)
        print("macOS 麦克风录音工具")
        print("=" * 50)

        try:
            duration = input("\n请输入录音时长（秒，默认5秒）: ").strip()
            duration = int(duration) if duration else 5

            filename = input("请输入文件名（留空自动生成）: ").strip()
            filename = filename if filename else None
            if filename and not filename.endswith('.wav'):
                filename += '.wav'

            print()
            self.record(duration=duration, filename=filename)

        except KeyboardInterrupt:
            print("\n\n录音已取消")
        except ValueError:
            print("输入无效，请输入数字")


def main():
    """主函数"""
    recorder = AudioRecorder()

    if len(sys.argv) > 1:
        # 命令行模式
        try:
            duration = int(sys.argv[1])
            filename = sys.argv[2] if len(sys.argv) > 2 else None
            recorder.record(duration=duration, filename=filename)
        except ValueError:
            print("用法: python record_audio.py [时长(秒)] [文件名(可选)]")
            print("示例: python record_audio.py 10 my_recording.wav")
    else:
        # 交互式模式
        recorder.record_interactive()


if __name__ == "__main__":
    main()
