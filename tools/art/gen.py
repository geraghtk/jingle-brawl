"""Generate game art with Flux schnell through a local ComfyUI (http://127.0.0.1:8188), then cut each
picture out of its plain background (flood fill from the border) into a transparent, trimmed PNG.

    python gen.py prompts.json out_dir [--only name,name] [--seeds 3]   (see README.md)

specs.json: {"style": "...", "items": {"name": "subject", ...}}. Writes out_dir/raw/<name>-<seed>.png
and out_dir/cut/<name>-<seed>.png (1024 -> trimmed, square, 512).
"""
import json, sys, time, urllib.request, io, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

API = "http://127.0.0.1:8188"


def workflow(prompt, seed, size=1024):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": size, "height": size, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                                                  "latent_image": ["4", 0], "seed": seed, "steps": 4, "cfg": 1.0,
                                                  "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "yd"}},
    }


def call(path, body=None):
    req = urllib.request.Request(API + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600).read()


def render(prompt, seed):
    pid = json.loads(call("/prompt", {"prompt": workflow(prompt, seed)}))["prompt_id"]
    while True:
        h = json.loads(call(f"/history/{pid}"))
        if pid in h and h[pid].get("outputs"):
            img = h[pid]["outputs"]["7"]["images"][0]
            q = f"filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
            return Image.open(io.BytesIO(call(f"/view?{q}"))).convert("RGB")
        if pid in h and h[pid].get("status", {}).get("status_str") == "error":
            raise RuntimeError(h[pid]["status"])
        time.sleep(0.5)


def cut(img, tol=28, out=512, pad=0.06):
    """The subject out of a plain background: pixels joined to the border and close to its colour go."""
    a = np.asarray(img).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    near = np.abs(a - bg).max(axis=2) <= tol
    lab, _ = ndimage.label(near)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    back = np.isin(lab, list(edge))
    back = ndimage.binary_opening(back, iterations=1)
    alpha = (~back).astype(np.float32)
    alpha = ndimage.gaussian_filter(alpha, 1.0)                  # a soft edge, not a jagged one
    keep = ndimage.binary_fill_holes(alpha > 0.5)
    lab2, n = ndimage.label(keep)                                # drop specks: keep the big pieces
    if n > 1:
        sizes = ndimage.sum(keep, lab2, range(1, n + 1))
        big = [i + 1 for i, s in enumerate(sizes) if s >= 0.02 * sizes.max()]
        alpha = alpha * np.isin(lab2, big) if big else alpha
    rgba = np.dstack([np.asarray(img), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
    im = Image.fromarray(rgba, "RGBA")
    box = im.getbbox()
    im = im.crop(box) if box else im
    side = int(max(im.size) * (1 + 2 * pad))
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return sq.resize((out, out), Image.LANCZOS)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 2
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "cut").mkdir(parents=True, exist_ok=True)
    for name, subject in spec["items"].items():
        if only and name not in only:
            continue
        for s in range(seeds):
            raw = out / "raw" / f"{name}-{s}.png"
            if not raw.exists():
                render(spec["style"].replace("{subject}", subject), 1000 + s).save(raw)
            cut(Image.open(raw).convert("RGB")).save(out / "cut" / f"{name}-{s}.png")
            print(name, s, flush=True)


if __name__ == "__main__":
    main()
