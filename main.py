import asyncio
import subprocess
import os
import uuid
import json
import logging
import re
import io
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator
import uvicorn
import requests
import tomllib
import threading
import numpy as np
import soundfile as sf
import librosa

from text_normalizer import TextNormalizer

_http_session = requests.Session()
_voices_lock = threading.Lock()

# ---------- 加载配置文件 ----------
CONFIG_FILE = "config.toml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        default_toml = """# VoxCPM TTS Server 配置文件

[server]
host = "0.0.0.0"
port = 8000
log_level = "INFO"

[cli]
path = "./build/bin/voxcpm2-cli"
base_model = "VoxCPM-0.5B-BaseLM-Q8_0.gguf"
acoustic_model = "VoxCPM-0.5B-Acoustic-F16.gguf"
use_cpu = true

[storage]
temp_dir = "./temp_audio"
voices_dir = "./uploaded_voices"
voices_db = "./voices_db.json"

[download]
timeout = 30
"""
        with open(CONFIG_FILE, "w") as f:
            f.write(default_toml)
        print(f"⚠️  默认配置文件已创建：{CONFIG_FILE}，请编辑后重新启动。")
        raise SystemExit(0)

    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)

config = load_config()

# ---------- 从配置读取参数 ----------
SERVER_HOST = config["server"]["host"]
SERVER_PORT = config["server"]["port"]
LOG_LEVEL = config["server"].get("log_level", "INFO")

CLI_PATH = config["cli"]["path"]
BASE_MODEL = config["cli"]["base_model"]
ACOUSTIC_MODEL = config["cli"]["acoustic_model"]
USE_CPU = config["cli"]["use_cpu"]

TEMP_DIR = config["storage"]["temp_dir"]
VOICES_DIR = config["storage"]["voices_dir"]
VOICES_DB_PATH = config["storage"]["voices_db"]

DOWNLOAD_TIMEOUT = config["download"]["timeout"]

# ---------- 初始化目录 ----------
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
Path(VOICES_DIR).mkdir(parents=True, exist_ok=True)

# ---------- 语音数据库管理 ----------
def load_voices_db():
    if os.path.exists(VOICES_DB_PATH):
        with open(VOICES_DB_PATH) as f:
            return json.load(f)
    return {}

def save_voices_db(db):
    with open(VOICES_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

voices_db = load_voices_db()

# ---------- TextNormalizer ----------
normalizer = TextNormalizer()

# ---------- FastAPI ----------
app = FastAPI(title="VoxCPM TTS Server", version="2.0")

# ---------- 请求模型 ----------
class SpeechRequest(BaseModel):
    input: str
    instructions: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    voice: str | None = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    temperature: float = 1.0
    seed: int | None = None
    normalize: bool = False
    retry_badcase: bool = True
    trim_silence: bool = True
    stream: bool = False

    @field_validator('cfg_value')
    @classmethod
    def check_cfg(cls, v):
        if v < 1.0 or v > 3.0:
            raise ValueError('cfg_value must be in range 1.0-3.0')
        return v

    @field_validator('inference_timesteps')
    @classmethod
    def check_timesteps(cls, v):
        if v < 4 or v > 30:
            raise ValueError('inference_timesteps must be in range 4-30')
        return v

# ---------- 辅助函数 ----------
MAX_SEGMENT_LENGTH = 300

def split_text(text: str, max_len: int = MAX_SEGMENT_LENGTH) -> list[str]:
    sentences = re.split(r'(?<=[。！？.!?\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]
    segments, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_len:
            current += s
        else:
            if current:
                segments.append(current)
            current = s
    if current:
        segments.append(current)
    return segments

def validate_reference_audio(path: str):
    try:
        audio, sr = sf.read(path)
        duration = len(audio) / sr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")
    if duration < 1.0:
        raise HTTPException(status_code=400, detail=f"参考音频过短：{duration:.1f}s < 1s")
    if duration > 30.0:
        raise HTTPException(status_code=400, detail=f"参考音频过长：{duration:.1f}s > 30s")

def download_file(url: str, dest_path: str):
    try:
        resp = _http_session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"下载失败：{url} ({e})")
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path

def is_temp_download(path: str) -> bool:
    return path.startswith(os.path.join(TEMP_DIR, "ref_"))

def resolve_audio_path(ref_audio: str) -> str:
    if ref_audio.startswith(("http://", "https://")):
        uid = str(uuid.uuid4())[:8]
        local = os.path.join(TEMP_DIR, f"ref_{uid}.wav")
        return download_file(ref_audio, local)
    if not os.path.exists(ref_audio):
        raise HTTPException(status_code=400, detail=f"本地参考音频不存在：{ref_audio}")
    return ref_audio

def detect_badcase(wav_path: str) -> bool:
    try:
        audio, sr = sf.read(wav_path)
        duration = len(audio) / sr
        if duration < 0.5:
            return True
    except Exception:
        pass
    return False

def run_voxcpm_cli(
    text, output_path,
    ref_wav_path=None,
    prompt_wav_path=None,
    prompt_text=None,
    cfg_value=2.0,
    timesteps=10,
    temperature=1.0,
    seed=None,
    stream=False,
):
    cmd = [
        CLI_PATH, "-t", text, "-o", output_path,
        BASE_MODEL, ACOUSTIC_MODEL,
    ]

    if ref_wav_path:
        cmd += ["-r", ref_wav_path]
    if prompt_wav_path:
        cmd += ["--prompt-wav", prompt_wav_path]
    if prompt_text:
        cmd += ["--prompt-text", prompt_text]
    cmd += ["--cfg", str(cfg_value)]
    cmd += ["--timesteps", str(timesteps)]
    cmd += ["--temperature", str(temperature)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if USE_CPU:
        cmd += ["--cpu"]
    if stream:
        cmd += ["--stream"]

    logging.info(f"Running: {' '.join(cmd)}")

    if stream:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"CLI 失败：{result.stderr}")
    if not os.path.exists(output_path):
        raise RuntimeError("未生成输出文件")

def generate_and_read(text, output_path, ref_audio_path, prompt_wav_path,
                      prompt_text, request):
    seed = request.seed
    max_retries = 3 if request.retry_badcase else 1

    for attempt in range(max_retries):
        current_seed = seed + attempt if seed is not None else None
        attempt_path = output_path if attempt == 0 else output_path.replace(".wav", f"_retry{attempt}.wav")

        run_voxcpm_cli(
            text=text,
            output_path=attempt_path,
            ref_wav_path=ref_audio_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=request.cfg_value,
            timesteps=request.inference_timesteps,
            temperature=request.temperature,
            seed=current_seed,
        )

        if request.retry_badcase and detect_badcase(attempt_path):
            logging.warning(f"检测到异常音频，正在重试（第 {attempt+1} 次）")
            if attempt < max_retries - 1:
                os.remove(attempt_path)
                continue

        if attempt_path != output_path:
            os.rename(attempt_path, output_path)
        break
    else:
        raise RuntimeError(f"重试 {max_retries} 次后音频仍异常")

    try:
        wav, sr = sf.read(output_path)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    if os.path.exists(output_path):
        os.remove(output_path)
    return wav, sr

# ---------- API 端点 ----------
@app.post("/v1/audio/voices")
async def upload_voice(
    audio_sample: UploadFile = File(...),
    consent: str = Form(...),
    name: str = Form(...),
    ref_text: Optional[str] = Form(None),
):
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if name in voices_db:
        logging.warning(f"覆盖已有语音：{name}")

    ext = os.path.splitext(audio_sample.filename)[1] or ".wav"
    audio_path = os.path.join(VOICES_DIR, f"{name}{ext}")
    with open(audio_path, "wb") as f:
        f.write(await audio_sample.read())

    await asyncio.to_thread(validate_reference_audio, audio_path)

    with _voices_lock:
        voices_db[name] = {
            "audio_path": audio_path,
            "ref_text": ref_text or "",
            "consent": consent,
            "created_at": datetime.now().isoformat(),
        }
    await asyncio.to_thread(save_voices_db, voices_db)
    return {"status": "success", "voice": name}

@app.get("/v1/audio/voices")
async def list_voices():
    return {
        "voices": [
            {
                "name": name,
                "ref_text": info.get("ref_text", ""),
                "consent": info.get("consent", ""),
                "created_at": info.get("created_at", ""),
            }
            for name, info in voices_db.items()
        ]
    }

@app.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str):
    with _voices_lock:
        if name not in voices_db:
            raise HTTPException(status_code=404, detail=f"语音 '{name}' 不存在")
        audio_path = voices_db[name].get("audio_path")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        del voices_db[name]
    await asyncio.to_thread(save_voices_db, voices_db)
    return {"status": "success"}

def _resolve_mode(request: SpeechRequest):
    """根据请求参数推断模式，返回 (ref_audio_path, prompt_wav_path, prompt_text, voice_desc)。"""
    ref_audio_path = None
    prompt_wav_path = None
    prompt_text = None
    voice_desc = None

    if request.voice and request.ref_audio:
        raise HTTPException(status_code=400, detail="voice 与 ref_audio 互斥，不能同时提供")

    # 1. Registered Voice
    if request.voice:
        if request.voice not in voices_db:
            raise HTTPException(status_code=400, detail=f"语音 '{request.voice}' 不存在")
        entry = voices_db[request.voice]
        ref_audio_path = entry["audio_path"]
        if entry.get("ref_text"):
            prompt_text = entry["ref_text"]
            prompt_wav_path = ref_audio_path
            voice_desc = request.instructions
            if voice_desc:
                prompt_text = None
                ref_audio_path = prompt_wav_path
                prompt_wav_path = None
        else:
            voice_desc = request.instructions

    # 2. Hi-Fi Clone
    elif request.ref_audio and request.ref_text:
        ref_audio_path = resolve_audio_path(request.ref_audio)
        prompt_wav_path = ref_audio_path
        prompt_text = request.ref_text
        voice_desc = request.instructions
        if voice_desc:
            prompt_text = None
            ref_audio_path = prompt_wav_path
            prompt_wav_path = None

    # 3. Controllable Clone + style
    elif request.ref_audio and request.instructions:
        ref_audio_path = resolve_audio_path(request.ref_audio)
        voice_desc = request.instructions

    # 4. Controllable Clone (no style)
    elif request.ref_audio:
        ref_audio_path = resolve_audio_path(request.ref_audio)

    # 5. Voice Design
    elif request.instructions:
        voice_desc = request.instructions

    # 6. Random Voice: 什么都不传

    return ref_audio_path, prompt_wav_path, prompt_text, voice_desc


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="输入文本为空")

    if request.stream:
        raise HTTPException(status_code=400, detail="流式请求请使用 POST /v1/audio/speech/stream")

    ref_audio_path, prompt_wav_path, prompt_text, voice_desc = await asyncio.to_thread(_resolve_mode, request)

    downloaded_refs: list[str] = []
    if ref_audio_path and is_temp_download(ref_audio_path):
        downloaded_refs.append(ref_audio_path)

    if ref_audio_path:
        await asyncio.to_thread(validate_reference_audio, ref_audio_path)

    # ---------- 文本归一化 ----------
    text = request.input
    if request.normalize:
        text = await asyncio.to_thread(normalizer.normalize, text)

    # ---------- 长文本分段 ----------
    segments = split_text(text)

    if voice_desc:
        segments = [f"({voice_desc}){seg}" for seg in segments]

    uid = str(uuid.uuid4())[:8]
    sr = None

    try:
        if len(segments) > 1:
            logging.info(f"文本已分为 {len(segments)} 段")
            all_wavs = []
            for i, seg in enumerate(segments):
                tmp = os.path.join(TEMP_DIR, f"tts_{uid}_{i}.wav")
                seg_prompt_wav = prompt_wav_path if i == 0 else None
                seg_prompt_text = prompt_text if i == 0 else None
                wav, sr_seg = await asyncio.to_thread(
                    generate_and_read,
                    seg, tmp,
                    ref_audio_path,
                    seg_prompt_wav,
                    seg_prompt_text,
                    request,
                )
                if sr is None:
                    sr = sr_seg
                elif sr != sr_seg:
                    wav = await asyncio.to_thread(librosa.resample, wav, orig_sr=sr_seg, target_sr=sr)
                all_wavs.append(wav)
            full_wav = np.concatenate(all_wavs)
        else:
            tmp = os.path.join(TEMP_DIR, f"tts_{uid}.wav")
            full_wav, sr = await asyncio.to_thread(
                generate_and_read,
                segments[0], tmp,
                ref_audio_path,
                prompt_wav_path,
                prompt_text,
                request,
            )

        if request.trim_silence:
            full_wav, _ = librosa.effects.trim(full_wav, top_db=20)

        buf = io.BytesIO()
        sf.write(buf, full_wav, sr, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    finally:
        for f in downloaded_refs:
            if os.path.exists(f):
                os.remove(f)


@app.post("/v1/audio/speech/stream")
async def speech_stream(request: SpeechRequest):
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="输入文本为空")

    ref_audio_path, prompt_wav_path, prompt_text, voice_desc = await asyncio.to_thread(_resolve_mode, request)

    downloaded_refs: list[str] = []
    if ref_audio_path and is_temp_download(ref_audio_path):
        downloaded_refs.append(ref_audio_path)

    if ref_audio_path:
        await asyncio.to_thread(validate_reference_audio, ref_audio_path)

    if request.retry_badcase:
        logging.info("流式模式不支持 retry_badcase，已忽略")

    text = request.input
    if request.normalize:
        text = await asyncio.to_thread(normalizer.normalize, text)

    if voice_desc:
        text = f"({voice_desc}){text}"

    if len(text) > MAX_SEGMENT_LENGTH:
        logging.warning(f"流式模式文本较长（{len(text)} 字符），可能不稳定，建议使用非流式端点")

    uid = str(uuid.uuid4())[:8]
    tmp = os.path.join(TEMP_DIR, f"stream_{uid}.wav")

    def stream_generator():
        proc = None
        try:
            proc = run_voxcpm_cli(
                text=text,
                output_path=tmp,
                ref_wav_path=ref_audio_path,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                cfg_value=request.cfg_value,
                timesteps=request.inference_timesteps,
                temperature=request.temperature,
                seed=request.seed,
                stream=True,
            )
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode()
                logging.error(f"流式 CLI 失败：{stderr}")
        except GeneratorExit:
            if proc:
                proc.kill()
        except Exception as e:
            logging.exception("流式生成错误")
        finally:
            if proc and proc.stdout:
                proc.stdout.close()
            if proc and proc.stderr:
                proc.stderr.close()
            if os.path.exists(tmp):
                os.remove(tmp)
            for f in downloaded_refs:
                if os.path.exists(f):
                    os.remove(f)

    return StreamingResponse(stream_generator(), media_type="audio/wav")

@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- 启动 ----------
if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, LOG_LEVEL))
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
