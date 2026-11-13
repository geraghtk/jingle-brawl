"""The Ring Toss reindeer (web/public/art/ringtoss/reindeer.webp): Flux schnell in the local ComfyUI,
wide (its antlers are the targets, so they spread far), no sticker border, cut out of a plain
background like the rest (gen.py's cut, without squaring).

    python reindeer.py out_dir [--seeds 6] [--variant a]     (--variant bg: the backdrop)

Then pick a take, measure its antler tips and star (512-px wide coordinates) and set the pegs in
web/src/core/minigames/ringtoss/ringtoss.ts and the placement (DEER) in web/src/host/RingTossStage.tsx.
"""
import io, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from gen import call

W, H = 1536, 896

STYLE = ("A flat vector cartoon illustration of {subject}. Clean bold shapes, smooth soft cel shading, warm "
         "friendly colors, simple and polished like a modern mobile game. Perfectly symmetrical, seen straight "
         "from the front. No outline border, no sticker edge, no white rim around the character, no frame. "
         "Isolated on a plain solid flat medium grey background, no shadow, no text, no letters.")
SUBJECTS = {
    "h": "a feisty cartoon reindeer champion facing forward, standing chest-deep in a big rounded white snowdrift that "
         "hides its legs, its head, neck, shoulders and furry chest above the snow, smug grin with mouth closed, "
         "determined furrowed eyebrows, battle-ready swagger, red nose, a tall cone-shaped party hat with gold "
         "stripes perched on top of its head pointing straight up to a sharp tip, and gigantic wide antlers "
         "spreading far to both sides like a coat rack, each side with two tall upright prongs with rounded tips",
    "g": "a feisty cartoon reindeer champion facing forward, its head, neck and broad furry shoulders and chest, "
         "the chest continuing down and cut off by the bottom edge of the picture, smug grin with mouth closed, "
         "determined furrowed eyebrows, battle-ready swagger, red nose, a tall cone-shaped party hat with gold "
         "stripes perched on top of its head pointing straight up to a sharp tip, and gigantic wide antlers "
         "spreading far to both sides like a coat rack, each side with two tall upright prongs with rounded tips",
    "f": "a whimsical battle-ready cartoon reindeer from the chest up facing the viewer, with its neck, shoulders and "
         "furry chest and the tops of its front legs, a red knitted scarf, a cheeky confident smirk with the mouth "
         "closed, determined furrowed eyebrows, a shiny red nose, a tall pointy striped red and gold party hat cone "
         "standing straight up on top of its head, and enormous antlers that spread very far out sideways, much "
         "wider than its body, each antler with two long prongs pointing straight up with round knobbly tips",
    "d": "a whimsical battle-ready reindeer head facing the viewer, a cheeky confident smirk with the mouth closed, "
         "determined furrowed eyebrows, half-lidded fearless eyes, a shiny red nose, a small sticking-plaster on one "
         "cheek, wearing a tall pointy striped red and gold party hat cone standing straight up on top of its head, "
         "and enormous antlers that spread very far out sideways, much wider than the whole head, each antler with "
         "two long prongs pointing straight up with round knobbly tips",
    "e": "a feisty cartoon reindeer champion head facing forward, smug grin with mouth closed, one eyebrow raised, "
         "battle-ready swagger, red nose, a tall cone-shaped party hat with gold stripes perched on top of its head "
         "pointing straight up to a sharp tip, and gigantic wide antlers spreading far to both sides like a coat "
         "rack, each side with two tall upright prongs with rounded tips",
    "c": "a cute friendly reindeer head facing the viewer, soft warm brown fur, big gentle eyes, a shiny round red "
         "nose and a small sweet closed-mouth smile, and enormous antlers that spread very far out sideways, much "
         "wider than the whole head, like a wide coat rack, each antler with two long prongs pointing straight up "
         "with round knobbly tips, and a small glowing golden star floating in the air just above the top of its "
         "head between the antlers, not attached to anything",
    "a": "a cheerful reindeer head and neck facing the viewer, shiny red nose, big friendly eyes, and enormous "
         "antlers that spread very far out sideways, much wider than the whole head, like a wide coat rack, each "
         "antler with two long prongs pointing straight up with round knobbly tips, and a golden star on a short "
         "spring standing on top of its head between the antlers",
    "b": "a friendly reindeer with a red nose facing forward, only the head and shoulders, wearing a red scarf, "
         "with gigantic extra-wide antlers stretching out horizontally to both sides like outstretched arms, three "
         "times wider than its head, each side ending in two upright prongs with rounded tips, and a small golden "
         "star topper on its head in the middle",
}


def workflow(prompt, seed):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                                                  "latent_image": ["4", 0], "seed": seed, "steps": 4, "cfg": 1.0,
                                                  "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "yd-deer"}},
    }


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


def cut_wide(img, tol=28):
    """The character out of its plain background, trimmed (not squared), with a soft edge."""
    a = np.asarray(img).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    near = np.abs(a - bg).max(axis=2) <= tol
    lab, _ = ndimage.label(near)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    back = ndimage.binary_opening(np.isin(lab, list(edge)), iterations=1)
    keep = ndimage.binary_fill_holes(~back)
    lab2, n = ndimage.label(keep)
    if n > 1:
        sizes = ndimage.sum(keep, lab2, range(1, n + 1))
        keep = np.isin(lab2, [i + 1 for i, s in enumerate(sizes) if s >= 0.02 * sizes.max()])
    alpha = ndimage.gaussian_filter(keep.astype(np.float32), 0.8)
    im = Image.fromarray(np.dstack([np.asarray(img), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)]), "RGBA")
    return im.crop(im.getbbox())


BACKDROP = ("A flat vector cartoon illustration of a calm snowy winter clearing at dusk, seen from ground level: "
            "a wide smooth empty snowfield in the foreground filling the lower half, a soft line of distant pine trees "
            "and gentle hills on the horizon, a pale blue and lavender evening sky with a few soft stars. Simple, "
            "uncluttered, soft muted colors, lots of empty space in the middle, like a modern mobile game backdrop. "
            "No characters, no animals, no text, no letters, no frame.")


def backdrops(out, seeds):
    (out / "bg").mkdir(parents=True, exist_ok=True)
    for s in range(seeds):
        f = out / "bg" / f"bg-{s}.png"
        if not f.exists():
            render(BACKDROP, 3000 + s).save(f)
        print("bg", s, flush=True)


def main():
    out = Path(sys.argv[1])
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 6
    only = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else None
    if only == "bg":
        return backdrops(out, seeds)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "cut").mkdir(parents=True, exist_ok=True)
    for v, subject in SUBJECTS.items():
        if only and v != only:
            continue
        for s in range(seeds):
            raw = out / "raw" / f"deer-{v}-{s}.png"
            if not raw.exists():
                render(STYLE.replace("{subject}", subject), 2000 + s).save(raw)
            cut_wide(Image.open(raw).convert("RGB")).save(out / "cut" / f"deer-{v}-{s}.png")
            print(v, s, flush=True)


if __name__ == "__main__":
    main()
