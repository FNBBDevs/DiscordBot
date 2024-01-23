import os
import sys
import json
import asyncio
import edge_tts
import argparse
import subprocess
from subprocess import Popen, PIPE


async def say(message: str, member: str, curr_dir: str):
    CURRENT_DIR = curr_dir + "\\BOT\\_utils\\bruhtts\\"

    sys.path.append(CURRENT_DIR)

    MODELS_PATH = os.path.join(CURRENT_DIR, "models")
    AUDIOS_INPUT_PATH = os.path.join(CURRENT_DIR, "audio/inputs")
    AUDIOS_OUTPUT_PATH = os.path.join(CURRENT_DIR, "audio/outputs")

    MODEL_NAMES = names = [
        os.path.join(root, file)
        for root, _, files in os.walk(MODELS_PATH, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    AUDIO_PATHS = [
        os.path.join(root, name)
        for root, _, files in os.walk(AUDIOS_INPUT_PATH, topdown=False)
        for name in files
        if name.endswith(tuple({
            "wav",
            "mp3",
            "flac",
            "ogg",
            "opus",
            "m4a",
            "mp4",
            "aac",
            "alac",
            "wma",
            "aiff",
            "webm",
            "ac3",
        }))
        and root == AUDIOS_INPUT_PATH
        and "_output" not in name
    ]

    json_path = os.path.join(CURRENT_DIR, "rvc", "lib", "tools", "tts_voices.json")
    with open(json_path, "r") as file:
        tts_voices_data = json.load(file)
    RVC_VOICES = [voice.get("ShortName", "").strip() for voice in tts_voices_data  if voice.get("ShortName").startswith("en-US")]

    TTS_PATH = os.path.join(CURRENT_DIR, "rvc", "lib", "tools", "tts.py")
    INFER_PATH = os.path.join(CURRENT_DIR, "rvc", "infer", "infer.py")

    USER_TEXT = message
    USER_SELECTED_VOICE = "en-US-AndrewNeural"
    MEMBER = member
    USER_SELECTED_MODEL = [model for model in MODEL_NAMES if MEMBER.value.lower() in model.lower()]
    if USER_SELECTED_MODEL:
        USER_SELECTED_MODEL = USER_SELECTED_MODEL[0]
    else:
        return False
    USER_SELECTED_INDEX = ""

    TTS_COMMAND = [
        "poetry",
        "run",
        "python",
        TTS_PATH,
        USER_TEXT,
        USER_SELECTED_VOICE,
        AUDIOS_OUTPUT_PATH + "\\tts_output.wav"
    ]

    # pitch
    f0up_key = 0
    filter_radius = 3
    index_rate = 0.75
    hop_length = 128
    f0method = "rmvpe"

    INFER_COMMAND = [
        "poetry",
        "run",
        "python",
        INFER_PATH,
        str(f0up_key),
        str(filter_radius),
        str(index_rate),
        str(hop_length),
        f0method,
        AUDIOS_OUTPUT_PATH + "\\tts_output.wav",
        AUDIOS_OUTPUT_PATH + "\\tts_rvc_output.wav",
        USER_SELECTED_MODEL,
        USER_SELECTED_INDEX,
        "",
        MODELS_PATH
    ]

    try:
        subprocess.run(TTS_COMMAND)
        subprocess.run(INFER_COMMAND)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
