from faster_whisper import WhisperModel, BatchedInferencePipeline
from typing import Iterable, List, Optional, Union, Callable
import torch.multiprocessing as mp
from scipy.signal import resample
import signal as system_signal
from ctypes import c_bool
import soundfile as sf
import faster_whisper
import collections
import numpy as np
import traceback
import threading
import webrtcvad
import datetime
import platform
import logging
import struct
import queue
import torch
import time
import copy
import os
import gc


logger = logging.getLogger("faststt")
logger.propagate = False


SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0
TIME_SLEEP = 0.02


INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0


class ContinuousAudioRecorder:
    
    def __init__(self, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE, 
                 input_device_index=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.input_device_index = input_device_index
        self.audio_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.is_running = False
        self.reader_thread = None
        
    def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.shutdown_event.clear()
        self.reader_thread = threading.Thread(
            target=self._audio_worker,
            daemon=True
        )
        self.reader_thread.start()
        logger.info("音频录制已开始")
    
    def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        self.shutdown_event.set()
        
        if self.reader_thread:
            self.reader_thread.join(timeout=5)
        
        logger.info("音频录制已停止")
    
    def get_audio_data(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _audio_worker(self):

        import pyaudio
        
        audio_interface = None
        stream = None
        
        try:
            audio_interface = pyaudio.PyAudio()
            if self.input_device_index is None:
                try:
                    default_device = audio_interface.get_default_input_device_info()
                    self.input_device_index = default_device['index']
                except:
                    self.input_device_index = 0
            

            stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.buffer_size * 2,
                input_device_index=self.input_device_index,
            )
            
            logger.info(f"音频流已开启 (设备: {self.input_device_index}, 采样率: {self.sample_rate})")

            audio_buffer = bytearray()
            target_buffer_size = self.buffer_size * 4
            
            while not self.shutdown_event.is_set():
                try:
                    data = stream.read(self.buffer_size * 2, exception_on_overflow=False)
                    audio_buffer.extend(data)
                    
                    while len(audio_buffer) >= target_buffer_size:
                        chunk = audio_buffer[:target_buffer_size]
                        audio_buffer = audio_buffer[target_buffer_size:]
                        
                        if not self.shutdown_event.is_set():
                            self.audio_queue.put(bytes(chunk))
                        
                except Exception as e:
                    if not self.shutdown_event.is_set():
                        logger.error(f"音频读取错误: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"音频工作线程错误: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if audio_interface:
                audio_interface.terminate()


class VoiceActivityDetector:
    
    def __init__(self, sample_rate=SAMPLE_RATE, 
                 silero_sensitivity=0.6, 
                 webrtc_sensitivity=2,     
                 energy_threshold=150):
        self.sample_rate = sample_rate
        self.silero_sensitivity = silero_sensitivity
        self.webrtc_sensitivity = webrtc_sensitivity
        self.energy_threshold = energy_threshold

        self.webrtc_vad = webrtcvad.Vad()
        self.webrtc_vad.set_mode(webrtc_sensitivity)
        

        try:
            self.silero_vad, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False
            )
            logger.info("Silero VAD加载成功")
        except Exception as e:
            logger.warning(f"Silero VAD加载失败，仅使用WebRTC VAD: {e}")
            self.silero_vad = None
        
        self.debug_counter = 0
        
    def is_speech(self, audio_chunk):
        self.debug_counter += 1
        debug_this_round = (self.debug_counter % 50 == 0)
        
        try:
            energy = self._calculate_energy(audio_chunk)
            if energy < self.energy_threshold:
                if debug_this_round:
                    logger.info(f"能量过低: {energy:.1f} < {self.energy_threshold}")
                return False

            webrtc_result = self._webrtc_detect(audio_chunk)

            silero_result = False
            if self.silero_vad is not None:
                silero_result = self._silero_detect(audio_chunk)

            if self.silero_vad is not None:
                final_result = webrtc_result or silero_result
            else:
                final_result = webrtc_result
            
            if debug_this_round:
                logger.info(f"VAD检测: 能量={energy:.1f}, WebRTC={webrtc_result}, Silero={silero_result}, 最终={final_result}")
            
            return final_result
            
        except Exception as e:
            if debug_this_round:
                logger.error(f"VAD检测错误: {e}")
            return False
    
    def _calculate_energy(self, audio_chunk):
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            return energy
        except:
            return 0
    
    def _webrtc_detect(self, chunk):
        try:
            if len(chunk) < 320:
                return False
            frame_length = 320
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(chunk) - frame_length, frame_length):
                frame = chunk[i:i + frame_length]
                if len(frame) == frame_length:
                    total_frames += 1
                    try:
                        if self.webrtc_vad.is_speech(frame, self.sample_rate):
                            speech_frames += 1
                    except:
                        continue

            return total_frames > 0 and (speech_frames / total_frames) > 0.2
            
        except Exception as e:
            return False
    
    def _silero_detect(self, chunk):

        try:
            audio_float = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_float) < 512:
                return False
                

            if len(audio_float) > 16000:
                audio_float = audio_float[:16000]
                
            prob = self.silero_vad(torch.from_numpy(audio_float), self.sample_rate).item()
            

            threshold = 1.0 - self.silero_sensitivity  
            result = prob > threshold
            
            return result
            
        except Exception as e:
            return False

class TranscriptionWorker:
    
    def __init__(self, model_path, download_root, compute_type, 
                 gpu_device_index, device, beam_size, batch_size, normalize_audio):
        self.model_path = model_path
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.normalize_audio = normalize_audio
        
        self.model = None
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.worker_thread = None
        self.model_ready = threading.Event()
        
    def start(self):
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.model_ready.wait()
        
    def stop(self):
        self.shutdown_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def transcribe_async(self, audio_data, language, callback):
        self.transcription_queue.put((audio_data, language, callback))
    
    def _worker_loop(self):
        try:
            self._init_model()
            self.model_ready.set()
            
            logger.info("转录工作器已准备就绪")
            
            while not self.shutdown_event.is_set():
                try:
                    item = self.transcription_queue.get(timeout=1.0)
                    if item is None:
                        break
                        
                    audio_data, language, callback = item
                    result = self._transcribe_audio(audio_data, language)
                    if callback and result:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"回调函数错误: {e}")
                    
                    self.transcription_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"转录工作器错误: {e}")
                    
        except Exception as e:
            logger.error(f"转录工作器初始化失败: {e}")
            self.model_ready.set() 
    
    def _init_model(self):
        """初始化模型"""
        logger.info("正在初始化转录模型...")
        
        self.model = WhisperModel(
            model_size_or_path=self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            device_index=self.gpu_device_index,
            download_root=self.download_root,
        )
        
        if self.batch_size > 0:
            self.model = BatchedInferencePipeline(model=self.model)

        try:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            warmup_audio_path = os.path.join(current_dir, "warmup_audio.wav")
            
            if os.path.exists(warmup_audio_path):
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                segments, info = self.model.transcribe(warmup_audio_data, language="en", beam_size=1)
                list(segments)
            else:
                warmup_audio = np.zeros(16000, dtype=np.float32)
                segments, info = self.model.transcribe(warmup_audio, language="en", beam_size=1)
                list(segments)
                
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
        
        logger.info("转录模型初始化完成")
    
    def _transcribe_audio(self, audio, language):
        """转录音频"""
        if audio is None or len(audio) == 0:
            return ""
        
        try:
            # 音频预处理
            if self.normalize_audio:
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = (audio / peak) * 0.95
            
            # 执行转录
            if self.batch_size > 0:
                segments, info = self.model.transcribe(
                    audio, language=language, beam_size=self.beam_size,
                    batch_size=self.batch_size
                )
            else:
                segments, info = self.model.transcribe(
                    audio, language=language, beam_size=self.beam_size
                )
            
            transcription = " ".join(seg.text for seg in segments).strip()
            return transcription
            
        except Exception as e:
            logger.error(f"转录错误: {e}")
            return ""


class AudioToTextRecorder:
    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 download_root: str = None,
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 level=logging.WARNING,
                 batch_size: int = 16,
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = INIT_POST_SPEECH_SILENCE_DURATION,
                 min_length_of_recording: float = INIT_MIN_LENGTH_OF_RECORDING,
                 pre_recording_buffer_duration: float = INIT_PRE_RECORDING_BUFFER_DURATION,
                 beam_size: int = 5,
                 normalize_audio: bool = False,
                 **kwargs):

        self.model_path = model
        self.download_root = download_root
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.normalize_audio = normalize_audio
        self.post_speech_silence_duration = post_speech_silence_duration
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.is_shut_down = False
        self.continuous_mode_active = False
        self._setup_logging(level)
        self._init_components()
    
    def _setup_logging(self, level):
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter("faststt: %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    def _init_components(self):
        logger.info("初始化faststt组件...")
        
        self.audio_recorder = ContinuousAudioRecorder(
            sample_rate=SAMPLE_RATE,
            buffer_size=BUFFER_SIZE,
            input_device_index=self.input_device_index
        )
        
        self.vad = VoiceActivityDetector(
            sample_rate=SAMPLE_RATE,
            silero_sensitivity=0.6,
            webrtc_sensitivity=2
        )
        
        self.transcription_worker = TranscriptionWorker(
            model_path=self.model_path,
            download_root=self.download_root,
            compute_type=self.compute_type,
            gpu_device_index=self.gpu_device_index,
            device=self.device,
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            normalize_audio=self.normalize_audio
        )
        
        self.transcription_worker.start()
        
        logger.info("faststt初始化完成")
    
    def text(self, on_transcription_finished: Optional[Callable] = None):
        if on_transcription_finished:
            return self._start_continuous_mode(on_transcription_finished)
        else:
            return self._single_transcription()
    
    def _single_transcription(self):

        logger.info("开始单次录音转录")
        
        self.audio_recorder.start()
        
        try:
            audio_data = self._record_single_segment()
            if audio_data is not None and len(audio_data) > 0:
                result = self._transcribe_sync(audio_data)
                return result
            return ""
        finally:
            self.audio_recorder.stop()
    
    def _start_continuous_mode(self, callback):
        if self.continuous_mode_active:
            logger.warning("连续模式已经在运行")
            return
        
        logger.info("启动连续录音模式")
        self.continuous_mode_active = True

        self.audio_recorder.start()

        processing_thread = threading.Thread(
            target=self._continuous_processing_loop,
            args=(callback,),
            daemon=True
        )
        processing_thread.start()
        
        return "" 
    
    def _continuous_processing_loop(self, callback):
        logger.info("连续处理循环已启动")
        recording_state = "waiting" 
        speech_frames = 0
        silence_frames = 0
        

        audio_buffer = collections.deque()
        pre_buffer = collections.deque(
            maxlen=int(self.pre_recording_buffer_duration * SAMPLE_RATE / (BUFFER_SIZE * 4))
        )
        

        start_threshold = 3   
        stop_threshold = 15   
        
        try:
            while self.continuous_mode_active and not self.is_shut_down:
                audio_chunk = self.audio_recorder.get_audio_data(timeout=0.1)
                if audio_chunk is None:
                    continue
                

                pre_buffer.append(audio_chunk)
                

                is_speech = self.vad.is_speech(audio_chunk)
                

                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if recording_state == "recording":
                        speech_frames = max(0, speech_frames - 1)
                

                if recording_state == "waiting":
                    if speech_frames >= start_threshold:
                        logger.info(f"🎤 开始录音 (语音帧: {speech_frames})")
                        recording_state = "recording"
                        recording_start_time = time.time()

                        audio_buffer.extend(list(pre_buffer))
                        
                elif recording_state == "recording":
                    audio_buffer.append(audio_chunk)
                    
                    if silence_frames >= stop_threshold:
                        recording_duration = time.time() - recording_start_time
                        if recording_duration >= self.min_length_of_recording:
                            logger.info(f"结束录音 (时长: {recording_duration:.1f}s, 静音帧: {silence_frames})")
                            

                            self._process_audio_segment(audio_buffer, callback)
                            

                            recording_state = "waiting"
                            speech_frames = 0
                            silence_frames = 0
                            audio_buffer.clear()
                            
        except Exception as e:
            logger.error(f"连续处理循环错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_audio_segment(self, audio_buffer, callback):
        """处理音频段落"""
        try:

            audio_bytes = b''.join(audio_buffer)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
            self.transcription_worker.transcribe_async(audio_float, self.language, callback)
            
        except Exception as e:
            logger.error(f"处理音频段落错误: {e}")
    
    def _record_single_segment(self):
        audio_buffer = []
        speech_detected = False
        silence_start_time = None
        recording_start_time = None
        
        pre_buffer = collections.deque(
            maxlen=int(self.pre_recording_buffer_duration * SAMPLE_RATE / (BUFFER_SIZE * 4))
        )
        
        logger.info("等待语音输入...")
        
        while True:
            audio_chunk = self.audio_recorder.get_audio_data(timeout=0.1)
            if audio_chunk is None:
                continue
            
            pre_buffer.append(audio_chunk)
            is_speech = self.vad.is_speech(audio_chunk)
            
            if is_speech and not speech_detected:
                logger.info("检测到语音，开始录音")
                speech_detected = True
                recording_start_time = time.time()
                audio_buffer.extend(list(pre_buffer))
                
            elif speech_detected:
                audio_buffer.append(audio_chunk)
                
                if is_speech:
                    silence_start_time = None
                else:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    
                    silence_duration = time.time() - silence_start_time
                    recording_duration = time.time() - recording_start_time
                    
                    if (silence_duration >= self.post_speech_silence_duration and
                        recording_duration >= self.min_length_of_recording):
                        
                        logger.info(f"录音完成 (时长: {recording_duration:.1f}s)")
                        break

        if audio_buffer:
            audio_bytes = b''.join(audio_buffer)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
            return audio_float
        
        return None
    
    def _transcribe_sync(self, audio_data):
        result_queue = queue.Queue()
        
        def callback(result):
            result_queue.put(result)
        
        self.transcription_worker.transcribe_async(audio_data, self.language, callback)
        
        try:
            return result_queue.get(timeout=30)
        except queue.Empty:
            logger.error("转录超时")
            return ""
    
    def stop_continuous_mode(self):
        """停止连续模式"""
        logger.info("停止连续录音模式")
        self.continuous_mode_active = False
    
    def shutdown(self):
        """关闭录制器"""
        if self.is_shut_down:
            return
        
        logger.info("正在关闭faststt...")
        self.is_shut_down = True
        if self.continuous_mode_active:
            self.stop_continuous_mode()
        self.audio_recorder.stop()
        self.transcription_worker.stop()
        
        logger.info("faststt已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()