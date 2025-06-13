from curses.ascii import alt
from gettext import install
from turtle import update
from webbrowser import get

import pip


alt-get,update
alt-get,install libportaudio2
pip install sounddevice
pip install numpy

import sounddevice as sd
import numpy as np
import whisper
import requests
import json
import google.generativeai as genai

# Gemini APIキーを設定
# 取得したAPIキーを設定
gemini_api_key = "AIzaSyDPjdazU05Wrrcn2rTMOtuR6JOAsDa1Frs"
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

def record_audio(duration=5, fs=16000, device=None):
    """音声を録音し、NumPy配列として返す。"""
    print("録音開始...")
    # デバイスが指定されていない場合はデフォルトの入力デバイスを取得
    if device is None:
        try:
            # First, check if *any* input devices are available
            input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
            if not input_devices:
                print("No input audio devices found.")
                print("Available devices:")
                print(sd.query_devices())
                return None # No input device, return None

            # If devices are found, try to get the default input device
            device_info = sd.query_devices(kind='input')
            device = device_info['index']
            print(f"Using default input device: {device_info['name']} (index {device})")
        except Exception as e:
            print(f"Error getting default input device: {e}")
            print("Available devices:")
            print(sd.query_devices())
            return None # Error getting default device, return None

    try:
        # channels=1 for mono recording
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device)
        sd.wait()
        print("録音終了")
        return recording.flatten()
    except Exception as e:
        print(f"Error during recording: {e}")
        # If there's an error during recording even after finding a device,
        # it might indicate the selected device is not working or busy.
        # Print available devices again for troubleshooting.
        print("Available devices:")
        print(sd.query_devices())
        return None

def audio_to_text_whisper(audio_data, fs=16000):
    """Whisperを使って音声をテキストに変換する。"""
    choice = "large"
    try:
        whisper_model = whisper.load_model(choice)
        # Ensure the audio data is in the correct format and sample rate for whisper
        # Whisper models expect 16kHz audio
        if fs != 16000:
             print(f"Warning: Input audio sample rate is {fs}, converting to 16000 for Whisper.")
             # Simple resampling - for better quality consider a proper resampling library like `scipy.signal.resample`
             audio_data = np.interp(np.arange(0, len(audio_data), fs/16000), np.arange(0, len(audio_data)), audio_data)

        result = whisper_model.transcribe(audio_data.astype(np.float32), language="ja")
        return result["text"]
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return "音声認識に失敗しました。"


def get_gemini_response(text):
    """Gemini APIにテキストを送信して回答を得る。"""
    if not text or text.strip() == "":
        return "入力がありませんでした。"
    try:
        response = model.generate_content(text)
        # Check if the response has a 'text' attribute
        if hasattr(response, 'text'):
            return response.text
        else:
            # Handle cases where Gemini might not return text (e.g., safety filters)
            print("Gemini did not return text content.")
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                print(f"Block reason: {response.prompt_feedback.block_reason}")
            return "Geminiからの回答がありませんでした。"

    except Exception as e:
        return f"Gemini APIエラー: {e}"

def text_to_speech_voicevox(text, speaker=1):
    """VOICEVOX APIを使ってテキストを音声に変換し、再生する。"""
    if not text or text.strip() == "":
        print("VOICEVOX: 入力テキストがありません。")
        return

    query_payload = {"text": text, "speaker": speaker}
    try:
        # Check if VOICEVOX engine is running
        try:
            requests.get("http://localhost:50021/version")
        except requests.exceptions.ConnectionError:
            print("VOICEVOX engine is not running. Please start the engine.")
            return

        res1 = requests.post("http://localhost:50021/audio_query", params=query_payload)
        res1.raise_for_status()  # エラーレスポンスの場合に例外を発生させる
        res2 = requests.post(
            "http://localhost:50021/synthesis",
            headers={"Content-Type": "application/json"},
            params={"speaker": speaker},
            data=json.dumps(res1.json()),
        )
        res2.raise_for_status()
        # Play the audio data
        # Determine the sample rate from the audio query response if possible,
        # otherwise use a default like 24000 which is common for VOICEVOX
        audio_query_data = res1.json()
        sample_rate = audio_query_data.get('outputSamplingRate', 24000)

        wav_data = np.frombuffer(res2.content, dtype=np.int16)
        sd.play(wav_data, sample_rate)
        sd.wait()
    except requests.exceptions.RequestException as e:
        print(f"VOICEVOX APIエラー: {e}")
    except json.JSONDecodeError as e:
        print(f"VOICEVOX APIレスポンスエラー: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in text_to_speech_voicevox: {e}")

# --- メイン処理 ---

# 1. 音声録音
# Noneを渡すことで、record_audio関数内でデフォルトデバイスを試行します
recorded_audio = record_audio(device=None)

if recorded_audio is not None:
    # 2. Whisperによる音声認識
    recognized_text = audio_to_text_whisper(recorded_audio)
    print(f"認識結果: {recognized_text}")

    # 3. Gemini APIによる回答生成
    gemini_response = get_gemini_response(recognized_text)
    print(f"Geminiの回答: {gemini_response}")

    # 4. VOICEVOXによる音声合成と出力
    if gemini_response:
        # speaker=1 をデフォルトとして指定していますが、必要に応じて変更してください
        # text_to_speech_voicevox(gemini_response, speaker=speaker_id) # ← もし最初のセルで定義したspeaker_idを使いたい場合
        text_to_speech_voicevox(gemini_response, speaker=1) # ← デフォルトのspeaker=1を使う場合
else:
    print("音声録音がスキップされました。オーディオ入力デバイスが見つからないか、アクセスできませんでした。")
    print("システムにマイクが接続されているか、環境にオーディオ入力が設定されているか確認してください。")