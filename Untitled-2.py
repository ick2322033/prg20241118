#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 必要なライブラリのインポート
import sys
import sounddevice as sd # type: ignore
import numpy as np
import scipy.io.wavfile as wav
import requests
import json
import google.generativeai as genai # type: ignore
import io
import wave
import openai # type: ignore
import os

def record_audio(duration=5, sample_rate=44100):
    """指定された時間だけ音声を録音し、NumPy配列として返します。"""
    print("録音を開始します...")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("録音を終了しました。")
        return recording, sample_rate
    except sd.PortAudioError as e:
        print(f"録音中にエラーが発生しました: {e}")
        print("必要なオーディオデバイスが見つからないか、アクセスできません。")
        print("システムで利用可能なオーディオデバイスを確認し、必要に応じて設定してください。")
        sys.exit(1)  # エラー終了

def save_audio(data, sample_rate, filename="recorded_audio.wav"):
    """NumPy配列の音声データをWAVファイルとして保存します。"""
    wav.write(filename, sample_rate, data)
    print(f"音声を {filename} に保存しました。")
    return filename

def transcribe_audio_whisper(audio_file, api_key):
    """OpenAI Whisper APIを用いて音声ファイルを文字起こしします。"""
    openai.api_key = api_key
    try:
        with open(audio_file, "rb") as f:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=f,
                language="ja"  # 必要に応じて変更
            )
        print(f"Whisper APIによる文字起こし結果: {transcript['text']}")
        return transcript['text']
    except openai.error.OpenAIError as e:
        print(f"Whisper APIの呼び出しでエラーが発生しました: {e}")
        print(f"エラー詳細: {e}")  # より詳細なエラー情報を表示
        return None
    except FileNotFoundError:
        print(f"エラー：指定されたファイルが見つかりません: {audio_file}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None

def generate_response(prompt, api_key):
    """Gemini APIを用いてテキストを生成します。"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini APIの呼び出しでエラーが発生しました: {e}")
        return None

def synthesize_speech(text, speaker, endpoint):
    """VOICEVOX APIを用いてテキストから音声を合成し、バイトデータを返します。"""
    params = {
        "text": text,
        "speaker": speaker
    }
    try:
        res = requests.post(f"{endpoint}/audio_query", params=params)
        res.raise_for_status()
        query_data = res.json()

        headers = {"Content-Type": "application/json"}
        res = requests.post(f"{endpoint}/synthesis", headers=headers, data=json.dumps(query_data))
        res.raise_for_status()
        return res.content
    except requests.exceptions.RequestException as e:
        print(f"VOICEVOX APIへの接続でエラーが発生しました: {e}")
        print(f"エラー詳細: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"VOICEVOX APIからのレスポンスのJSON解析でエラーが発生しました: {e}")
        print(f"エラー詳細: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None

def play_audio_bytes(audio_bytes):
    """バイト形式の音声データを再生します。"""
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            num_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16)
        try:
            sd.play(audio_array, sample_rate)
            sd.wait()
        except sd.PortAudioError as e:
            print(f"音声再生中にエラーが発生しました: {e}")
            print("オーディオデバイスがビジー状態か、正しく設定されていません。")
            print("別のアプリケーションがオーディオデバイスを使用していないか確認し、必要であればシステムのオーディオ設定を調整してください。")
    except wave.Error as e:
        print(f"WAVデータの処理中にエラーが発生しました: {e}")
        print("無効な音声データがVOICEVOX APIから返された可能性があります。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    print("音声録音、Whisper APIによる文字起こし、Gemini APIによる応答、VOICEVOXによる音声合成・再生を開始します。")

    # APIキーとエンドポイントの入力
    VOICEVOX_ENDPOINT = input("VOICEVOX APIのエンドポイントを入力してください (例: http://localhost:50021): ")
    GEMINI_API_KEY = input("Gemini APIキーを入力してください: ")
    OPENAI_API_KEY = input("OpenAI APIキーを入力してください: ")

    # 1. 音声録音
    recorded_data, sample_rate = record_audio()
    if recorded_data is None:
        print("録音に失敗しました。プログラムを終了します。")
        sys.exit(1)
    recorded_filename = save_audio(recorded_data, sample_rate)

    # 2. Whisper APIによる文字起こし
    recognized_text = transcribe_audio_whisper(recorded_filename, OPENAI_API_KEY)
    if recognized_text is None:
        print("文字起こしに失敗しました。プログラムを終了します。")
        os.remove(recorded_filename)  # エラー終了前に一時ファイルを削除
        sys.exit(1)

    # 3. Gemini APIによる応答生成
    prompt = f"質問: {recognized_text}\n回答:"
    gemini_response = generate_response(prompt, GEMINI_API_KEY)
    if gemini_response is None:
        print("応答生成に失敗しました。プログラムを終了します。")
        os.remove(recorded_filename)  # エラー終了前に一時ファイルを削除
        sys.exit(1)
    print(f"Geminiの応答: {gemini_response}")

    # 4. VOICEVOX APIによる音声合成
    audio_output = synthesize_speech(gemini_response, speaker=1, endpoint=VOICEVOX_ENDPOINT)
    if audio_output is None:
        print("音声合成に失敗しました。プログラムを終了します。")
        os.remove(recorded_filename)  # エラー終了前に一時ファイルを削除
        sys.exit(1)

    # 5. 音声再生
    print("応答を再生します...")
    play_audio_bytes(audio_output)

    # 一時ファイルの削除
    os.remove(recorded_filename)
    print(f"一時ファイル {recorded_filename} を削除しました。")

    print("プログラムを終了します。")
