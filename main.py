import subprocess
import os
import time
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from fastapi.responses import Response
import uvicorn
import requests
import tomllib  # Python 3.11+ 内置

# ---------- 加载配置文件 ----------
CONFIG_FILE = "config.toml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        # 创建默认配置文件
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
VOICES_DB = config["storage"]["voices_db"]

DOWNLOAD_TIMEOUT = config["download"]["timeout"]

# ---------- 初始化目录 ----------
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
Path(VOICES_DIR).mkdir(parents=True, exist_ok=True)

# ---------- 语音数据库管理 ----------
def load_voices_db():
    if os.path.exists(VOICES_DB):
        with open(VOICES_DB, 'r') as f:
            return json.load(f)
    return {}

def save_voices_db(db):
    with open(VOICES_DB, 'w') as f:
        json.dump(db, f, indent=2)

voices_db = load_voices_db()

# ---------- FastAPI ----------
app = FastAPI(title="VoxCPM TTS Server (TOML Config)", version="1.0")

# ---------- 请求模型 ----------
class SpeechRequest(BaseModel):
    input: str
    task_type: Optional[str] = Field(default=None, description="VoiceDesign or Base")
    instructions: Optional[str] = Field(default=None, description="Voice description or style instruction")
    ref_audio: Optional[str] = Field(default=None, description="Reference audio URL or local path")
    ref_text: Optional[str] = Field(default=None, description="Transcript of reference audio")
    voice: Optional[str] = Field(default=None, description="Registered voice ID")
    non_streaming_mode: Optional[bool] = Field(default=True, description="Ignored")

# ---------- 辅助函数 ----------
def download_file(url: str, dest_path: str):
    response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download: {url}")
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path

def run_voxcpm_cli(text, output_path, ref_wav_path=None, voice_desc=None):
    """
    调用 voxcpm2-cli 生成音频。
    """
    cmd = [CLI_PATH, "-t", text, "-o", output_path, BASE_MODEL, ACOUSTIC_MODEL]
    
    if voice_desc:
        full_text = f"({voice_desc}){text}"
        cmd[cmd.index("-t") + 1] = full_text
    
    if ref_wav_path:
        cmd.extend(["-r", ref_wav_path])
    
    if USE_CPU:
        cmd.append("--cpu")
    
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logging.error(f"CLI stderr: {result.stderr}")
        raise RuntimeError(f"CLI failed: {result.stderr}")
    
    if not os.path.exists(output_path):
        raise RuntimeError("Output file not generated")

# ---------- API 端点 ----------
@app.post("/v1/audio/voices")
async def upload_voice(
    audio_sample: UploadFile = File(...),
    consent: str = Form(...),
    name: str = Form(...),
    ref_text: Optional[str] = Form(None)
):
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if name in voices_db:
        logging.warning(f"Overwriting existing voice: {name}")
    
    ext = os.path.splitext(audio_sample.filename)[1] or ".wav"
    audio_path = os.path.join(VOICES_DIR, f"{name}{ext}")
    with open(audio_path, "wb") as f:
        content = await audio_sample.read()
        f.write(content)
    
    voices_db[name] = {
        "audio_path": audio_path,
        "ref_text": ref_text or "",
        "consent": consent,
        "created_at": datetime.now().isoformat()
    }
    save_voices_db(voices_db)
    
    return {
        "status": "success",
        "voice": name,
        "message": f"Voice '{name}' uploaded successfully"
    }

@app.get("/v1/audio/voices")
async def list_voices():
    result = []
    for name, info in voices_db.items():
        result.append({
            "name": name,
            "ref_text": info.get("ref_text", ""),
            "consent": info.get("consent", ""),
            "created_at": info.get("created_at", ""),
            "audio_path": info.get("audio_path", "")
        })
    return {"voices": result}

@app.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str):
    if name not in voices_db:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    audio_path = voices_db[name].get("audio_path")
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    del voices_db[name]
    save_voices_db(voices_db)
    return {"status": "success", "message": f"Voice '{name}' deleted"}

@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty input text")
    
    ref_audio_path = None
    voice_desc = None
    
    # 模式解析
    if request.task_type == "VoiceDesign":
        if not request.instructions:
            raise HTTPException(status_code=400, detail="Missing 'instructions' for VoiceDesign")
        voice_desc = request.instructions
    elif request.task_type == "Base":
        if not request.ref_audio:
            raise HTTPException(status_code=400, detail="Missing 'ref_audio' for Base")
        if request.ref_audio.startswith(("http://", "https://")):
            uid = str(uuid.uuid4())[:8]
            local_ref = os.path.join(TEMP_DIR, f"ref_{uid}.wav")
            try:
                download_file(request.ref_audio, local_ref)
                ref_audio_path = local_ref
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Download ref_audio failed: {str(e)}")
        else:
            if not os.path.exists(request.ref_audio):
                raise HTTPException(status_code=400, detail=f"Local ref_audio not found: {request.ref_audio}")
            ref_audio_path = request.ref_audio
        if request.instructions:
            voice_desc = request.instructions
    elif request.voice:
        if request.voice not in voices_db:
            raise HTTPException(status_code=400, detail=f"Voice '{request.voice}' not found")
        ref_audio_path = voices_db[request.voice]["audio_path"]
        if request.instructions:
            voice_desc = request.instructions
    elif request.ref_audio:
        if request.ref_audio.startswith(("http://", "https://")):
            uid = str(uuid.uuid4())[:8]
            local_ref = os.path.join(TEMP_DIR, f"ref_{uid}.wav")
            try:
                download_file(request.ref_audio, local_ref)
                ref_audio_path = local_ref
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Download ref_audio failed: {str(e)}")
        else:
            if not os.path.exists(request.ref_audio):
                raise HTTPException(status_code=400, detail=f"Local ref_audio not found: {request.ref_audio}")
            ref_audio_path = request.ref_audio
        if request.instructions:
            voice_desc = request.instructions
    else:
        raise HTTPException(status_code=400, detail="Must provide one of: voice, ref_audio, or task_type='VoiceDesign'")
    
    uid = str(uuid.uuid4())[:8]
    temp_wav = os.path.join(TEMP_DIR, f"tts_{uid}_{int(time.time())}.wav")
    try:
        run_voxcpm_cli(request.input, temp_wav, ref_audio_path, voice_desc)
        with open(temp_wav, "rb") as f:
            audio_data = f.read()
        return Response(content=audio_data, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("TTS generation error")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        if ref_audio_path and ref_audio_path.startswith(TEMP_DIR) and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- 启动 ----------
if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, LOG_LEVEL))
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
