import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from RealtimeSTT import AudioToTextRecorder
from colorama import Fore, Style
import colorama
import time
import logging
import numpy as np

if __name__ == '__main__':
    colorama.init()
    logging.basicConfig(level=logging.WARNING)
    
    full_sentences = []
    transcription_times = []
    
    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def update_display():
        clear_console()
        print("=== 连续转录结果 ===")
        for i, sentence in enumerate(full_sentences):
            color = Fore.YELLOW if i % 2 == 0 else Fore.CYAN
            print(f"{color}{sentence}{Style.RESET_ALL}")
        print("=" * 30)
        print(f"{Fore.GREEN}继续说话...{Style.RESET_ALL}")
    
    def process_text(text):
        """处理转录结果"""
        current_time = time.strftime("%H:%M:%S")
        
        # 记录转录完成时间
        end_time = time.time()
        
        print(f"\n{Fore.BLUE}[{current_time}] ● 收到转录结果: '{text}'{Style.RESET_ALL}")
        
        if text.strip():
            # 计算转录时间（从转录开始到现在）
            if hasattr(process_text, 'transcription_start_time') and process_text.transcription_start_time:
                duration = end_time - process_text.transcription_start_time
                transcription_times.append({
                    'text': text.strip()[:40] + "..." if len(text.strip()) > 40 else text.strip(),
                    'duration': duration,
                    'timestamp': current_time
                })
                
                print(f"{Fore.MAGENTA}转录耗时: {duration:.2f}秒{Style.RESET_ALL}")
                process_text.transcription_start_time = None
            
            full_sentences.append(f"[{current_time}] {text.strip()}")
            update_display()
            print(f"{Fore.GREEN}转录完成，等待下一段语音...{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}收到空转录结果{Style.RESET_ALL}")
    
    # 初始化转录开始时间属性
    process_text.transcription_start_time = None
    
    def print_statistics():
        if not transcription_times:
            print(f"{Fore.YELLOW}没有转录时间数据{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}{'='*15} 转录时间统计 {'='*15}{Style.RESET_ALL}")
        
        for i, item in enumerate(transcription_times, 1):
            print(f"{Fore.WHITE}{i:2d}. [{item['timestamp']}] {item['text']:<40} - {Fore.GREEN}{item['duration']:.2f}秒{Style.RESET_ALL}")

        if transcription_times:
            durations = [item['duration'] for item in transcription_times]
            avg_time = sum(durations) / len(durations)
            min_time = min(durations)
            max_time = max(durations)
            
            print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}总句数: {len(transcription_times)}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}平均转录时间: {avg_time:.2f}秒{Style.RESET_ALL}")
            print(f"{Fore.GREEN}最快转录时间: {min_time:.2f}秒{Style.RESET_ALL}")
            print(f"{Fore.GREEN}最慢转录时间: {max_time:.2f}秒{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    
    # 配置参数
    config = {
        'model': 'large-v3',
        'language': 'zh',
        'download_root': 'D:\\model',
        'device': 'cuda',
        'compute_type': 'int8',
        'batch_size': 16,
        'beam_size': 1,
        'post_speech_silence_duration': 1.0,
        'min_length_of_recording': 0.5,
        'pre_recording_buffer_duration': 1.0,
        'silero_sensitivity': 0.6,
        'webrtc_sensitivity': 2,
        'level': logging.WARNING,
    }
    
    print("初始化中...")
    
    try:
        recorder = AudioToTextRecorder(**config)
        
        # 劫持录音器的转录方法来记录真实的转录开始时间
        original_transcribe = None
        if hasattr(recorder, 'transcription_worker') and hasattr(recorder.transcription_worker, '_transcribe_audio'):
            original_transcribe = recorder.transcription_worker._transcribe_audio
            
            def timed_transcribe(*args, **kwargs):
                # 记录转录开始时间
                process_text.transcription_start_time = time.time()
                print(f"{Fore.BLUE}🔄 开始转录...{Style.RESET_ALL}")
                return original_transcribe(*args, **kwargs)
            
            recorder.transcription_worker._transcribe_audio = timed_transcribe
        
        print(f"{Fore.GREEN}系统已就绪，开始说话！{Style.RESET_ALL}")
        print(f"{Fore.CYAN}调试模式：将显示详细的检测信息{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}提示：请大声清晰地说话{Style.RESET_ALL}")
        
        # 启动连续录音模式
        recorder.text(process_text)
        
        # 保持运行，并显示状态信息
        counter = 0
        while True:
            time.sleep(1)
            counter += 1
            if counter % 10 == 0:
                print(f"{Fore.YELLOW}● 系统运行中... ({counter}秒) - 请说话测试{Style.RESET_ALL}")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}正在退出...{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}发生错误: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    finally:
        if 'recorder' in locals():
            recorder.shutdown()
        
        # 打印转录时间统计
        print_statistics()