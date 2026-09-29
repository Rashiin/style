"""
Local virtual try-on demo for AIStylist.

Runs Qwen-Image-Edit (open weights, Apache 2.0) inside ComfyUI on a single GPU,
with every outbound network connection blocked once the models are on disk.
Designed for a free Kaggle GPU (T4, 16 GB) using a GGUF-quantized model.

Typical use (see kaggle_tryon_demo.ipynb):

    import tryon_demo as td
    td.setup()                       # install ComfyUI + download models (needs internet)
    td.start_server(offline=True)    # ComfyUI runs with the network blocked
    td.prove_offline()               # shows that outbound requests fail
    result = td.run_passes(person, passes)
    td.make_comparison(...)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths and pinned versions
# --------------------------------------------------------------------------- #

WORK = Path(os.environ.get("TRYON_WORK", "/tmp/tryon" if Path("/kaggle").exists() else "./work"))
COMFY_DIR = WORK / "ComfyUI"
GUARD_DIR = WORK / "offline_guard"
OUT_DIR = Path(os.environ.get("TRYON_OUT", "/kaggle/working/tryon_output" if Path("/kaggle").exists() else "./tryon_output"))

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
COMFY_TAG = "v0.37.4"
GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"

HOST, PORT = "127.0.0.1", 8188

# Candidate Hugging Face sources. Each entry is (repo_id, filename regex); the
# first repo that contains a matching file wins, so a renamed file or a missing
# repo falls through to the next candidate instead of failing the whole setup.
MODEL_SOURCES = {
    "2511": {
        "unet": [
            ("unsloth/Qwen-Image-Edit-2511-GGUF", r"Q4_K_M\.gguf$"),
            ("QuantStack/Qwen-Image-Edit-2511-GGUF", r"Q4_K_M\.gguf$"),
            ("unsloth/Qwen-Image-Edit-2511-GGUF", r"Q4_0\.gguf$"),
        ],
        "lora": [
            ("lightx2v/Qwen-Image-Edit-2511-Lightning", r"4steps.*bf16\.safetensors$"),
            ("lightx2v/Qwen-Image-Lightning", r"Edit-2511.*4steps.*bf16\.safetensors$"),
        ],
    },
    "2509": {
        "unet": [
            ("QuantStack/Qwen-Image-Edit-2509-GGUF", r"Q4_K_M\.gguf$"),
            ("QuantStack/Qwen-Image-Edit-2509-GGUF", r"Q4_0\.gguf$"),
        ],
        "lora": [
            ("lightx2v/Qwen-Image-Lightning", r"Edit-2509.*4steps.*bf16\.safetensors$"),
        ],
    },
}
COMMON_SOURCES = {
    "text_encoder": [("Comfy-Org/Qwen-Image_ComfyUI", r"text_encoders/qwen_2\.5_vl_7b_fp8_scaled\.safetensors$")],
    "vae": [("Comfy-Org/Qwen-Image_ComfyUI", r"vae/qwen_image_vae\.safetensors$")],
}
COMFY_SUBDIR = {"unet": "unet", "lora": "loras", "text_encoder": "text_encoders", "vae": "vae"}

# Filled in by setup(); read by build_graph().
MODELS: dict[str, str | None] = {"version": None, "unet": None, "lora": None, "text_encoder": None, "vae": None}
if (WORK / "models.json").exists():
    MODELS.update(json.loads((WORK / "models.json").read_text()))


def _run(cmd: list[str], **kw) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------------------- #
# Setup: ComfyUI + models (the only step that needs internet)
# --------------------------------------------------------------------------- #

def _install_comfy() -> None:
    if not (COMFY_DIR / "main.py").exists():
        _run(["git", "clone", "--depth", "1", "--branch", COMFY_TAG, COMFY_REPO, str(COMFY_DIR)])
    gguf_dir = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"
    if not gguf_dir.exists():
        _run(["git", "clone", GGUF_REPO, str(gguf_dir)])
        _run(["git", "-C", str(gguf_dir), "checkout", "-q", GGUF_COMMIT])
    # Keep the preinstalled CUDA build of torch: drop torch lines from requirements.
    reqs = [
        line for line in (COMFY_DIR / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#") and not re.match(r"^(torch|torchvision|torchaudio)\b", line)
    ]
    reqs += ["gguf>=0.13.0", "huggingface_hub>=0.25"]
    _run([sys.executable, "-m", "pip", "install", "-q", *reqs])


def _find_file(candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    from huggingface_hub import list_repo_files

    for repo, pattern in candidates:
        try:
            files = list_repo_files(repo)
        except Exception as e:  # repo missing or renamed
            print(f"  - {repo}: not available ({type(e).__name__})")
            continue
        matches = sorted(f for f in files if re.search(pattern, f))
        if matches:
            return repo, matches[0]
        print(f"  - {repo}: no file matching {pattern}")
    return None


def _download(kind: str, repo: str, filename: str, attempts: int = 5) -> str:
    from huggingface_hub import constants, hf_hub_download

    # The Xet transfer backend can stall near the end of large files without raising.
    # Plain HTTP times out on a stall instead, and the retry resumes the partial file.
    constants.HF_HUB_DISABLE_XET = True
    constants.HF_HUB_DOWNLOAD_TIMEOUT = max(constants.HF_HUB_DOWNLOAD_TIMEOUT, 60)

    print(f"Downloading {kind}: {repo}/{filename}  (free disk: {shutil.disk_usage(WORK).free / 1e9:.0f} GB)")
    for attempt in range(1, attempts + 1):
        try:
            path = Path(hf_hub_download(repo, filename, local_dir=WORK / "hf" / repo.replace("/", "__")))
            break
        except Exception as e:
            if attempt == attempts:
                raise
            print(f"  ! download interrupted ({type(e).__name__}: {e}); resuming, attempt {attempt + 1}/{attempts}")
            time.sleep(5)
    target_dir = COMFY_DIR / "models" / COMFY_SUBDIR[kind]
    target_dir.mkdir(parents=True, exist_ok=True)
    link = target_dir / path.name
    if not link.exists():
        link.symlink_to(path)
    return path.name


def setup(version: str = "2511") -> dict:
    """Install ComfyUI and download the models. Falls back to 2509 if 2511 files are missing."""
    WORK.mkdir(parents=True, exist_ok=True)
    _install_comfy()

    order = [version] + [v for v in MODEL_SOURCES if v != version]
    for v in order:
        print(f"Looking for Qwen-Image-Edit-{v} files...")
        unet = _find_file(MODEL_SOURCES[v]["unet"])
        if unet:
            MODELS["version"] = v
            MODELS["unet"] = _download("unet", *unet)
            lora = _find_file(MODEL_SOURCES[v]["lora"])
            MODELS["lora"] = _download("lora", *lora) if lora else None
            if not lora:
                print("  ! No Lightning LoRA found: falling back to 20-step sampling (slower).")
            break
    else:
        raise RuntimeError("No Qwen-Image-Edit GGUF file found in any candidate repo.")

    for kind, candidates in COMMON_SOURCES.items():
        found = _find_file(candidates)
        if not found:
            raise RuntimeError(f"Could not find the {kind} file.")
        MODELS[kind] = _download(kind, *found)

    (WORK / "models.json").write_text(json.dumps(MODELS, indent=2))
    print(json.dumps(MODELS, indent=2))
    return MODELS


# --------------------------------------------------------------------------- #
# Offline guard: block every non-loopback connection in the ComfyUI process
# --------------------------------------------------------------------------- #

_GUARD_SOURCE = '''
import ipaddress, socket, sys

_LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

def _is_local(host):
    if host is None:
        return True
    host = host.decode() if isinstance(host, bytes) else str(host)
    if host in _LOCAL:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

class OfflineGuardError(OSError):
    pass

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_getaddrinfo = socket.getaddrinfo

def _check(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address[0]):
        raise OfflineGuardError(f"[offline-guard] blocked outbound connection to {address[0]}")

def connect(self, address):
    _check(self, address)
    return _orig_connect(self, address)

def connect_ex(self, address):
    _check(self, address)
    return _orig_connect_ex(self, address)

def getaddrinfo(host, *args, **kwargs):
    if not _is_local(host):
        raise OfflineGuardError(f"[offline-guard] blocked DNS lookup for {host}")
    return _orig_getaddrinfo(host, *args, **kwargs)

socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
print("[offline-guard] active: outbound network is blocked in this process", file=sys.stderr, flush=True)
'''


def _offline_env() -> dict:
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    (GUARD_DIR / "sitecustomize.py").write_text(_GUARD_SOURCE)
    # Drop proxy settings too: a proxy on localhost would otherwise be an allowed way out.
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    env["PYTHONPATH"] = str(GUARD_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DO_NOT_TRACK="1")
    return env


def prove_offline() -> None:
    """Run a request to OpenAI under the same guard ComfyUI runs with; it must fail."""
    code = textwrap.dedent("""
        import urllib.request
        try:
            urllib.request.urlopen("https://api.openai.com/v1/models", timeout=10)
            print("NETWORK REACHABLE (guard not active!)")
        except Exception as e:
            print("Request to api.openai.com failed as expected:", e)
    """)
    out = subprocess.run([sys.executable, "-c", code], env=_offline_env(), capture_output=True, text=True)
    print(out.stderr.strip())
    print(out.stdout.strip())


# --------------------------------------------------------------------------- #
# ComfyUI server
# --------------------------------------------------------------------------- #

_server: subprocess.Popen | None = None


def _api(path: str, payload: dict | None = None) -> dict:
    url = f"http://{HOST}:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI {path} -> {e.code}: {e.read().decode()[:4000]}") from None


def start_server(offline: bool = True, extra_args: list[str] | None = None, timeout: int = 300) -> None:
    global _server
    if MODELS["unet"] is None and (WORK / "models.json").exists():
        MODELS.update(json.loads((WORK / "models.json").read_text()))
    try:
        _api("/system_stats")
        print("ComfyUI is already running; reusing it.")
        return
    except Exception:
        pass
    env = _offline_env() if offline else dict(os.environ)
    log = open(WORK / "comfyui.log", "w")
    cmd = [sys.executable, "main.py", "--listen", HOST, "--port", str(PORT), *(extra_args or [])]
    _server = subprocess.Popen(cmd, cwd=COMFY_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _server.poll() is not None:
            raise RuntimeError("ComfyUI exited early:\n" + (WORK / "comfyui.log").read_text()[-3000:])
        try:
            _api("/system_stats")
            break
        except Exception:
            time.sleep(2)
    else:
        raise TimeoutError("ComfyUI did not start in time; see comfyui.log")
    guard_line = [l for l in (WORK / "comfyui.log").read_text().splitlines() if "offline-guard" in l]
    print(f"ComfyUI is up in {time.time() - t0:.0f}s.", guard_line[0] if guard_line else "(offline guard OFF)")


def stop_server() -> None:
    if _server and _server.poll() is None:
        _server.terminate()
        _server.wait(30)


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #

def build_graph(person: str, products: list[str], prompt: str, seed: int = 42,
                megapixels: float = 1.0, steps: int | None = None, cfg: float | None = None) -> dict:
    """ComfyUI API graph: picture 1 = person, pictures 2-3 = products."""
    if not 1 <= len(products) <= 2:
        raise ValueError("One pass takes 1 or 2 product images (3 pictures in total).")
    lightning = MODELS["lora"] is not None
    steps = steps or (4 if lightning else 20)
    cfg = cfg if cfg is not None else (1.0 if lightning else 2.5)

    g: dict[str, dict] = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODELS["unet"]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["text_encoder"], "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "person": {"class_type": "LoadImage", "inputs": {"image": person}},
        "person_scaled": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["person", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["person_scaled", 0], "vae": ["vae", 0]}},
    }
    model_ref = ["unet", 0]
    if lightning:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model_ref, "lora_name": MODELS["lora"], "strength_model": 1.0}}
        model_ref = ["lora", 0]
    g["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model_ref, "shift": 3.0}}
    g["cfgnorm"] = {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}}

    images = {"image1": ["person_scaled", 0]}
    for i, name in enumerate(products, start=2):
        g[f"product{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        images[f"image{i}"] = [f"product{i}", 0]
    for key, text in (("positive", prompt), ("negative", "")):
        g[key] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], **images}}

    g["sampler"] = {"class_type": "KSampler", "inputs": {
        "model": ["cfgnorm", 0], "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple",
        "positive": ["positive", 0], "negative": ["negative", 0],
        "latent_image": ["latent", 0], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "tryon"}}
    return g


def _stage(path: str | Path) -> str:
    """Copy an image into ComfyUI/input under a unique name and return that name."""
    src = Path(path)
    name = f"{uuid.uuid4().hex[:8]}_{src.name}"
    shutil.copy(src, COMFY_DIR / "input" / name)
    return name


def _report_progress(t0: float) -> None:
    """Print the generator's latest status line so a long run does not look frozen."""
    try:
        text = (WORK / "comfyui.log").read_text(errors="ignore")[-4000:]
    except OSError:
        return
    lines = [l.strip() for l in re.split(r"[\r\n]+", text) if l.strip()]
    steps = [l for l in lines if re.search(r"\d+%\|", l)]
    last = steps[-1] if steps else (lines[-1] if lines else "")
    print(f"  ... {time.time() - t0:.0f}s elapsed | {last[:120]}", flush=True)


def generate(person: str | Path, products: list[str | Path], prompt: str, seed: int = 42, **kw) -> tuple[Path, float]:
    graph = build_graph(_stage(person), [_stage(p) for p in products], prompt, seed=seed, **kw)
    t0 = time.time()
    prompt_id = _api("/prompt", {"prompt": graph, "client_id": "tryon-demo"})["prompt_id"]
    last_report = t0
    while True:
        hist = _api(f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"Generation failed: {json.dumps(msgs, indent=1)[:4000]}")
            if status.get("completed"):
                break
        if time.time() - last_report > 30:
            _report_progress(t0)
            last_report = time.time()
        time.sleep(1)
    elapsed = time.time() - t0
    img = entry["outputs"]["save"]["images"][0]
    path = COMFY_DIR / "output" / img.get("subfolder", "") / img["filename"]
    return path, elapsed


@dataclass
class PassResult:
    products: list[str]
    prompt: str
    output: Path
    seconds: float
    extra: dict = field(default_factory=dict)


def run_passes(person: str | Path, passes: list[dict], seed: int = 42, name: str = "set",
               keep_regions: list[tuple[float, float, float, float]] | None = None, **gen_kw) -> list[PassResult]:
    """
    Run one or more passes; each pass dresses the output of the previous one.
    passes = [{"products": [path, ...], "prompt": "..."}, ...]
    keep_regions: boxes (x0, y0, x1, y1) as fractions of the image that are copied
    back from the original photo after generation (e.g. a site logo).
    gen_kw: passed to build_graph (megapixels, steps, cfg).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current = Path(person)
    results = []
    for i, p in enumerate(passes, start=1):
        out, secs = generate(current, p["products"], p["prompt"], seed=seed, **gen_kw)
        final = OUT_DIR / f"{name}_pass{i}.png"
        _finish(Path(person), out, final, keep_regions)
        print(f"{name} pass {i}: {secs:.1f}s -> {final}")
        results.append(PassResult([str(x) for x in p["products"]], p["prompt"], final, secs))
        current = final
    return results


def _finish(original: Path, generated: Path, dest: Path, keep_regions) -> None:
    from PIL import Image

    out = Image.open(generated).convert("RGB")
    if keep_regions:
        src = Image.open(original).convert("RGB").resize(out.size, Image.LANCZOS)
        w, h = out.size
        for x0, y0, x1, y1 in keep_regions:
            box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            out.paste(src.crop(box), box)
    out.save(dest)


# --------------------------------------------------------------------------- #
# Report: side-by-side comparison + timing
# --------------------------------------------------------------------------- #

def make_comparison(person: str | Path, products: list[str | Path], ours: str | Path,
                    openai: str | Path | None = None, dest: str | Path | None = None, height: int = 900) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    def fit(path, h):
        im = Image.open(path).convert("RGB")
        return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    cols = [("Customer photo", fit(person, height))]
    tile_h = (height - 10 * (len(products) - 1)) // len(products)
    tiles = [fit(p, tile_h) for p in products]
    stack = Image.new("RGB", (max(t.width for t in tiles), height), "white")
    y = 0
    for t in tiles:
        stack.paste(t, ((stack.width - t.width) // 2, y))
        y += t.height + 10
    cols.append(("Products", stack))
    cols.append(("Local model", fit(ours, height)))
    if openai:
        cols.append(("OpenAI", fit(openai, height)))

    pad, head = 20, 60
    width = sum(c[1].width for c in cols) + pad * (len(cols) + 1)
    sheet = Image.new("RGB", (width, height + head + pad), "white")
    draw = ImageDraw.Draw(sheet)
    x = pad
    for title, im in cols:
        sheet.paste(im, (x, head))
        tw = draw.textlength(title, font=font)
        draw.text((x + (im.width - tw) / 2, 15), title, fill="black", font=font)
        x += im.width + pad
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = Path(dest or OUT_DIR / f"comparison_{Path(ours).stem}.jpg")
    sheet.save(dest, quality=92)
    return dest


def clean_corner(path: str | Path, box: tuple[float, float, float, float] = (0.84, 0.0, 1.0, 0.1)) -> Path:
    """Paint over a badge in a product photo (box as image fractions) with the nearby background colour."""
    from PIL import Image, ImageDraw

    im = Image.open(path).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    sample = im.getpixel((max(x0 - 10, 0), min(y1 + 10, h - 1)))
    ImageDraw.Draw(im).rectangle((x0, y0, x1, y1), fill=sample)
    dest = WORK / f"clean_{Path(path).stem}.png"
    im.save(dest)
    return dest


def gpu_info() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def write_report(results: dict[str, list[PassResult]], offline: bool = True) -> Path:
    lines = [
        "# AIStylist local try-on demo",
        "",
        f"- GPU: {gpu_info()}",
        f"- Model: Qwen-Image-Edit-{MODELS['version']} ({MODELS['unet']})",
        f"- Acceleration LoRA: {MODELS['lora'] or 'none'}",
        f"- Outbound network during generation: {'blocked' if offline else 'allowed'}",
        "",
        "| Set | Pass | Products | Seconds |",
        "|---|---|---|---|",
    ]
    for name, passes in results.items():
        for i, r in enumerate(passes, start=1):
            lines.append(f"| {name} | {i} | {', '.join(Path(p).name for p in r.products)} | {r.seconds:.1f} |")
    lines += ["", "The first pass includes loading the model into GPU memory; later passes show the steady-state time."]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "report.md"
    path.write_text("\n".join(lines))
    print("\n".join(lines))
    return path
