import os

# On CPU-only runners (e.g. GitHub Actions hosted runners, typically 2-4
# vCPUs), torch/OpenBLAS/MKL/Paddle each default to spawning as many threads
# as there are visible cores. When several of these libraries are active in
# the same process (as they are during OCR: torch for the VLM, Paddle for
# YOLO), this causes severe thread oversubscription/contention that can make
# a single model.generate() call appear to hang indefinitely, even though
# nothing is actually deadlocked - it's just extreme scheduling thrash.
# Capping thread counts up front (before torch/paddle are imported anywhere,
# including in the main.py subprocess this script launches, which inherits
# this environment) avoids that failure mode. Override via env if a given
# deployment has more cores to spare.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, os.environ.get("BOT_CPU_THREADS", "2"))

import re
import sys
import json
import shutil
import zipfile
import asyncio
import subprocess
import io
from pathlib import Path

# Tall manhwa/webtoon strips (and recomposed tiled pages) routinely exceed
# PIL's default DecompressionBombWarning/Error threshold (~89 megapixels).
# This is expected for this bot's use case, not an actual decompression-bomb
# attack, so raise the ceiling instead of letting PIL warn/crash on
# legitimate large pages. Setting it here (before any PIL.Image import)
# ensures every `from PIL import Image` elsewhere in this file inherits it,
# since PIL.Image.MAX_IMAGE_PIXELS is a module-level attribute.
from PIL import Image as _PILImage
_PILImage.MAX_IMAGE_PIXELS = None  # disable the pixel-count safety limit entirely

from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery

# ================= Configuration & Secrets =================
API_ID = int(os.environ.get("API_ID", 0))
API_HASH = os.environ.get("API_HASH", "")
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AUTH_USERS_STR = os.environ.get("AUTHORIZED_USERS", "")

if not API_ID or not API_HASH or not BOT_TOKEN:
    print("❌ Error: API_ID, API_HASH, or BOT_TOKEN missing from repository secrets!")
    sys.exit(1)

AUTHORIZED_USERS = [int(u.strip()) for u in AUTH_USERS_STR.split(",") if u.strip().isdigit()]

app = Client("YomiTranslatorBot", bot_token=BOT_TOKEN, api_id=API_ID, api_hash=API_HASH)

# ================= Persistent Storage Paths =================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "bot_data"
FONTS_DIR = DATA_DIR / "fonts"
PROMPTS_DIR = DATA_DIR / "prompts"
USERS_FILE = DATA_DIR / "user_settings.json"

for d in [DATA_DIR, FONTS_DIR, PROMPTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================= Language Catalogs =================
SOURCE_LANGS = [
    ("Auto Detect", "auto"), ("Japanese", "ja"), ("Korean", "ko"),
    ("English", "en"), ("Chinese", "zh"),
]

TARGET_LANGS = [
    ("English", "English"), ("Hindi (Roman)", "Roman Hindi"), ("Urdu (Roman)", "Roman Urdu"),
    ("Urdu", "Urdu"), ("Hindi", "Hindi"), ("French", "French"), ("Spanish", "Spanish"),
]

PROVIDERS = ["Google", "Gemini", "Anthropic", "OpenAI", "xAI", "DeepSeek", "Z.ai", "Moonshot AI", "Xiaomi MiMo", "OpenRouter", "OpenAI-Compatible"]

OUTPUT_FORMATS = [
    ("ZIP Package", "zip"), ("CBZ Archive", "cbz"),
    ("PDF Document", "pdf"), ("Raw Images", "img"),
]

UPLOAD_MODES = [
    ("🖼 Raw Images", "raw"), ("📦 ZIP / CBZ", "archive"), ("📄 PDF", "pdf"),
]

CONTENT_TYPES = [
    ("🍥 Manhwa (long strip)", "manhwa"), ("📖 Manga (page-by-page)", "manga"),
    ("💬 Comic (Western)", "comic"), ("📝 Novel (text only)", "novel"),
]

# Tuned to cut far less often than before: most bubble/panel detectors handle
# tiles up to ~3000-3500px fine, so there's no need to slice a 9000px strip into
# 6 pieces. Fewer tiles = fewer seams = fewer chances of duplicated/misaligned art.
MANHWA_TILE_HEIGHT = 3200
MANHWA_TILE_OVERLAP = 400
MANHWA_TILE_TRIGGER_HEIGHT = 3800
MANHWA_SAFE_CUT_FLAT_THRESHOLD = 3.5

# Min/Max tiles-per-page bounds. Bubble/panel safety always wins over hitting
# these counts: we will happily land BELOW min (or anywhere under max) rather
# than ever slice through a bubble. Max is a ceiling, not a target — landing
# under it (even close to Min) is completely fine.
MANHWA_TILE_MIN_CUTS = 1
MANHWA_TILE_MAX_CUTS = 10
# "Auto" search radius: when the fixed radius doesn't find a safe row, keep
# widening the search (in tile_search_radius steps) instead of giving up.
MANHWA_TILE_SEARCH_RADIUS_AUTO_MAX_EXTEND = 40

DEFAULT_SYSTEM_PROMPT_NAME = "Default Localization Engine"
DEFAULT_SYSTEM_PROMPT_TEXT = (
    "You are a professional multi-language manga and comic localization engine.\n"
    "Adapt the source script lines into high-fidelity, smooth, natural spoken {target_lang} dialogue.\n\n"
    "Operational Constraints:\n"
    "1. Contextual Structural Alignment: Localize structural layout flows organically rather than a rigid word-for-word interpretation.\n"
    "2. Preservation Matrix: Retain relational honorifics, names, and titles native to the source if they add contextual narrative immersion.\n"
    "3. Balanced Pacing: Limit overly aggressive, archaic, or excessively modern slang unless explicitly demanded by scene gravity.\n"
    "4. Volumetric Bounds: Match output textual block size to original bubble sizing to avoid clipping/layout overlap.\n\n"
    "Produce strictly the final targeted localization stream mapping blocks directly."
)
DEFAULT_USER_PROMPT_NAME = "Standard Workflow"
DEFAULT_USER_PROMPT_TEXT = "Standard clean workflow processing baseline."

# ================= Persistent JSON State =================
_state_lock = asyncio.Lock()

def _load_all_settings():
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_all_settings(data):
    USERS_FILE.write_text(json.dumps(data, indent=2))

user_settings = _load_all_settings()
pending_files = {}   
active_jobs = {}      
awaiting_reply = {}   

def default_config():
    return {
        "source_lang": "auto",
        "source_lang_label": "Auto Detect",
        "target_lang": "Roman Hindi",
        "target_lang_label": "Hindi (Roman)",
        "font_name": None,           
        "provider": "OpenAI-Compatible",
        "api_url": "https://api.highwayapi.ai/openai",
        "api_key": "",
        "model_name": "gemini-3.1-flash-lite",
        "output_format": "zip",
        "content_type": "manhwa",
        "content_type_label": "🍥 Manhwa (long strip)",
        
        # Prompts
        "system_prompt_name": DEFAULT_SYSTEM_PROMPT_NAME,
        "system_prompt_text": DEFAULT_SYSTEM_PROMPT_TEXT,
        "user_prompt_name": DEFAULT_USER_PROMPT_NAME,
        "user_prompt_text": DEFAULT_USER_PROMPT_TEXT,

        # Tiling Settings
        "tile_enabled": None, "tile_height": None, "tile_search_radius": None, 
        "tile_trigger_height": None, "tile_flat_threshold": None, "tile_seam_band_px": None, 
        "tile_seam_diff_threshold": None, "tile_min_cuts": None, "tile_max_cuts": None,
        "tile_search_radius_auto": None,

        # UI Modifiable Flags (Appearance, Detect)
        "min_font_size": None, "max_font_size": None, "auto_vertical_text": None,
        "line_spacing_mult": None, "subpixel_rendering": None, "font_hinting": None,
        "use_ligatures": None, "hyphenate_before_scaling": None, "hyphen_penalty": None,
        "hyphenation_min_word_length": None, "badness_exponent": None, "padding_pixels": None,
        "supersampling_factor": None, "detach_trailing_punctuation": None,
        "confidence": None, "conjoined_confidence": None, "panel_confidence": None,
        "seg_model": None, "conjoined_detection": None, "bubble_detector_model": None,
        "ocr_method": None, "osb_enabled": True,

        # Manual Translation Mode (two-pass human-in-the-loop workflow, separate
        # from translation_mode's one-step/two-step OCR strategy)
        "manual_translation_mode": False,

        # ALL OTHER main.py flags mapped for JSON Import/Export
        "temperature": None, "top_p": None, "top_k": None, "max_tokens": None,
        "translation_mode": None, "reasoning_effort": None, "effort": None, 
        "verbosity": None, "reading_direction": None, "enable_web_search": None,
        "enable_code_execution": None, "use_custom_sampling": None, "media_resolution": None,
        "media_resolution_bubbles": None, "media_resolution_context": None, "image_detail": None,
        "send_full_page_context": None, "parallel_requests": None, "batch_parallel_within_pages": None,
        "batch_previous_context_images": None, "batch_previous_context_texts": None,
        "retry_failed_once": None, "overlap_llm_with_inpaint": None,
        "use_otsu_threshold": None, "thresholding_value": None, "roi_shrink_px": None, 
        "inpaint_colored_bubbles": None, "whiteout_conjoined_bubbles": None,
        "upscale_method": None, "image_upscale_mode": None, "image_upscale_factor": None,
        "auto_scale": None, "jpeg_quality": None, "png_compression": None,
        "bubble_min_side_pixels": None, "context_image_max_side_pixels": None,
        "verbose": None, "cpu": None, "cleaning_only": None, "upscaling_only": None, "test_mode": None,
        "osb_inpainting_method": None, "osb_flux_backend": None,
        "osb_flux_low_vram": None, "osb_flux_sdcpp_cache_mode": None, "osb_flux_sdcpp_diffusion_quant": None,
        "osb_flux_sdcpp_text_encoder_quant": None, "osb_flux_upscale_small_crops": None,
        "osb_flux_group_regions": None, "osb_flux_steps": None, "osb_flux_luminance_correction": None,
        "osb_sfx_skip_inpaint": None,
        "osb_flux_residual_threshold": None, "osb_seed": None, "osb_max_font_size": None,
        "osb_min_font_size": None, "osb_use_ligatures": None, "osb_outline_width": None,
        "osb_line_spacing": None, "osb_use_subpixel": None, "osb_font_hinting": None,
        "osb_bbox_expansion": None, "osb_render_expansion_narrow": None, "osb_render_expansion_tiny": None,
        "osb_render_expansion_aspect_threshold": None, "osb_render_expansion_area_threshold": None,
        "osb_text_box_proximity_ratio": None, "osb_confidence": None, "osb_filter_page_numbers": None,
        "osb_page_filter_margin": None, "osb_page_filter_min_area": None, "osb_min_area_ignore_ratio": None,
        "osb_min_side_pixels": None
    }

def get_user_config(user_id):
    uid = str(user_id)
    if uid not in user_settings:
        user_settings[uid] = default_config()
        _save_all_settings(user_settings)
    else:
        defaults = default_config()
        cfg = user_settings[uid]
        missing = {k: v for k, v in defaults.items() if k not in cfg}
        if missing:
            cfg.update(missing)
            _save_all_settings(user_settings)
    return user_settings[uid]

async def save_user_config(user_id):
    async with _state_lock:
        _save_all_settings(user_settings)

# ================= Prompt Library Helpers =================
def _prompt_lib_file(kind):
    return PROMPTS_DIR / f"{kind}_prompts.json"

def load_prompt_library(kind):
    f = _prompt_lib_file(kind)
    if f.exists():
        try: return json.loads(f.read_text())
        except Exception: pass
    default_name = DEFAULT_SYSTEM_PROMPT_NAME if kind == "system" else DEFAULT_USER_PROMPT_NAME
    default_text = DEFAULT_SYSTEM_PROMPT_TEXT if kind == "system" else DEFAULT_USER_PROMPT_TEXT
    lib = {default_name: default_text}
    f.write_text(json.dumps(lib, indent=2))
    return lib

def save_prompt_library(kind, lib):
    _prompt_lib_file(kind).write_text(json.dumps(lib, indent=2))
    git_commit_data(f"Update {kind} prompt library")

def add_prompt(kind, name, text):
    lib = load_prompt_library(kind)
    lib[name] = text
    save_prompt_library(kind, lib)

def delete_prompt(kind, name):
    lib = load_prompt_library(kind)
    if name in lib and len(lib) > 1:
        lib.pop(name)
        save_prompt_library(kind, lib)
        return True
    return False

# ================= Font Library Helpers =================
def list_fonts():
    return sorted([f.name for f in FONTS_DIR.glob("*") if f.suffix.lower() in (".ttf", ".otf")])

def delete_font(name):
    f = FONTS_DIR / name
    if f.exists():
        f.unlink()
        git_commit_data(f"Remove font {name}")
        return True
    return False

def git_commit_data(message):
    try:
        subprocess.run(["git", "add", str(DATA_DIR)], cwd=str(BASE_DIR), check=False, capture_output=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(BASE_DIR), capture_output=True)
        if result.returncode == 0: return
        subprocess.run(["git", "-c", "user.email=bot@yomisubs.local", "-c", "user.name=YomiSubsBot", "commit", "-m", message], cwd=str(BASE_DIR), check=False, capture_output=True)
        subprocess.run(["git", "push"], cwd=str(BASE_DIR), check=False, capture_output=True)
    except Exception as e:
        print(f"⚠️ Git persistence skipped: {e}")

PROMPT_NAME_MAX_LEN = 32  

# ================= Safe Telegram UI Helpers =================
async def safe_edit(message, text, reply_markup=None):
    try:
        await message.edit_text(text, reply_markup=reply_markup)
    except Exception as e:
        err = str(e).lower()
        if "not modified" in err:
            return  
        try:
            await message.edit_text(text + " ", reply_markup=reply_markup)
        except Exception as e2:
            print(f"⚠️ safe_edit failed: {e2}")

async def safe_answer(query, text=None, show_alert=False):
    try:
        if text: await query.answer(text, show_alert=show_alert)
        else: await query.answer()
    except Exception as e:
        print(f"⚠️ safe_answer failed: {e}")

# ================= Auth Gate Filter =================
async def auth_check(_, __, message):
    return message.from_user.id in AUTHORIZED_USERS
auth_filter = filters.create(auth_check)

async def auth_check_cb(_, __, query):
    return query.from_user.id in AUTHORIZED_USERS
auth_filter_cb = filters.create(auth_check_cb)

# ================= Keyboard Builders =================
def kb_main_menu(cfg=None):
    rows = [
        [InlineKeyboardButton("📚 Content Type", callback_data="menu_content_type")],
        [InlineKeyboardButton("🌐 Language Settings", callback_data="menu_lang")],
        [InlineKeyboardButton("🔡 Font Track", callback_data="menu_font")],
        [InlineKeyboardButton("🎨 Appearance", callback_data="menu_appearance")],
    ]
    if cfg is not None and cfg.get("content_type") != "novel":
        rows.append([InlineKeyboardButton("🎯 Detection Settings", callback_data="menu_detection")])
    if cfg is not None and cfg.get("content_type") == "manhwa":
        rows.append([InlineKeyboardButton("✂️ Tiling Settings", callback_data="menu_tiling")])
    rows += [
        [InlineKeyboardButton("🧠 Generation Settings", callback_data="xf_group_generation")],
        [InlineKeyboardButton("🧹 Cleaning & Upscaling", callback_data="xf_group_cleaning")],
        [InlineKeyboardButton("⚙️ Batch & Performance", callback_data="xf_group_batch")],
    ]
    if cfg is not None and cfg.get("osb_enabled", True):
        rows.append([InlineKeyboardButton("🫧 OSB Tuning", callback_data="xf_group_osb")])
    rows += [
        [InlineKeyboardButton("⚙️ Provider & API", callback_data="menu_api")],
        [InlineKeyboardButton("📝 Prompt Library", callback_data="menu_prompt")],
        [InlineKeyboardButton("📦 Output Format", callback_data="menu_output")],
        [InlineKeyboardButton("💾 Backup (Import/Export JSON)", callback_data="menu_backup")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_content_type_select(cfg):
    rows = []
    for label, code in CONTENT_TYPES:
        mark = "✅ " if cfg.get("content_type") == code else ""
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"ctypeset_{code}")])
    rows.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)

def kb_lang_root(cfg):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Source: {cfg['source_lang_label']}", callback_data="lang_src_open")],
        [InlineKeyboardButton(f"Target: {cfg['target_lang_label']}", callback_data="lang_tgt_open")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
    ])

def kb_source_select(cfg):
    rows = []
    for label, code in SOURCE_LANGS:
        mark = "✅ " if cfg["source_lang"] == code else ""
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"srcset_{code}")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_lang")])
    return InlineKeyboardMarkup(rows)

def kb_target_select(cfg):
    rows = []
    for label, value in TARGET_LANGS:
        mark = "✅ " if cfg["target_lang"] == value else ""
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"tgtset_{value}")])
    custom_mark = "✅ " if cfg["target_lang_label"] == "Custom" else ""
    rows.append([InlineKeyboardButton(f"{custom_mark}Custom", callback_data="tgtset_custom")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_lang")])
    return InlineKeyboardMarkup(rows)

def kb_font_menu(cfg):
    rows = []
    for name in list_fonts():
        mark = "✅ " if cfg.get("font_name") == name else ""
        rows.append([
            InlineKeyboardButton(f"{mark}{name}", callback_data=f"fontsel_{name}"),
            InlineKeyboardButton("🗑", callback_data=f"fontdel_{name}"),
        ])
    rows.append([InlineKeyboardButton("➕ Add Font", callback_data="font_add")])
    rows.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)

def kb_appearance_menu(cfg):
    rows = [
        [InlineKeyboardButton("🔡 Font & Sizing", callback_data="menu_appear_font")],
        [InlineKeyboardButton("📐 Layout & Quality", callback_data="menu_appear_layout")],
        [InlineKeyboardButton("♻️ Reset All to Original/Default", callback_data="appear_reset_all")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
    ]
    return InlineKeyboardMarkup(rows)

def _appear_val_label(cfg, field, default_display):
    v = cfg.get(field)
    return f"Original/Default ({default_display})" if v is None else str(v)

def kb_appear_font_menu(cfg):
    min_fs = _appear_val_label(cfg, "min_font_size", "8")
    max_fs = _appear_val_label(cfg, "max_font_size", "16")
    ls_mult = _appear_val_label(cfg, "line_spacing_mult", "1.0")
    hint = _appear_val_label(cfg, "font_hinting", "none")
    lig = cfg.get("use_ligatures")
    lig_label = "Original/Default (Off)" if lig is None else ("✅ On" if lig else "❌ Off")
    avt = cfg.get("auto_vertical_text")
    avt_label = "Original/Default (Off)" if avt is None else ("✅ On" if avt else "❌ Off")

    rows = [
        [InlineKeyboardButton(f"🔡 Min Font Size (px): {min_fs}", callback_data="appear_field_min_font_size")],
        [InlineKeyboardButton(f"🔠 Max Font Size (px): {max_fs}", callback_data="appear_field_max_font_size")],
        [InlineKeyboardButton(f"📏 Line Spacing: {ls_mult}", callback_data="appear_field_line_spacing_mult")],
        [InlineKeyboardButton(f"🔎 Font Hinting: {hint}", callback_data="appear_hinting_open")],
        [InlineKeyboardButton(f"🔗 Ligatures: {lig_label}", callback_data="appear_bool_use_ligatures")],
        [InlineKeyboardButton(f"↕️ Auto-Vertical Text: {avt_label}", callback_data="appear_bool_auto_vertical_text")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_appearance")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_appear_layout_menu(cfg):
    padding = _appear_val_label(cfg, "padding_pixels", "5.0")
    supersample = _appear_val_label(cfg, "supersampling_factor", "4")
    badness = _appear_val_label(cfg, "badness_exponent", "3.0")
    hyphen_pen = _appear_val_label(cfg, "hyphen_penalty", "1000.0")
    hyphen_len = _appear_val_label(cfg, "hyphenation_min_word_length", "8")
    subpx = cfg.get("subpixel_rendering")
    subpx_label = "Original/Default (On)" if subpx is None else ("✅ On" if subpx else "❌ Off")
    hyph_scale = cfg.get("hyphenate_before_scaling")
    hyph_scale_label = "Original/Default (On)" if hyph_scale is None else ("✅ On" if hyph_scale else "❌ Off")
    detach_punct = cfg.get("detach_trailing_punctuation")
    detach_punct_label = "Original/Default (On)" if detach_punct is None else ("✅ On" if detach_punct else "❌ Off")

    rows = [
        [InlineKeyboardButton(f"📦 Bubble Padding (px): {padding}", callback_data="appear_field_padding_pixels")],
        [InlineKeyboardButton(f"✨ Supersampling (1-4): {supersample}", callback_data="appear_field_supersampling_factor")],
        [InlineKeyboardButton(f"📊 Line Badness Exponent: {badness}", callback_data="appear_field_badness_exponent")],
        [InlineKeyboardButton(f"➖ Hyphen Penalty: {hyphen_pen}", callback_data="appear_field_hyphen_penalty")],
        [InlineKeyboardButton(f"🔤 Hyphenation Min Word Length: {hyphen_len}", callback_data="appear_field_hyphenation_min_word_length")],
        [InlineKeyboardButton(f"🖥 Subpixel Rendering: {subpx_label}", callback_data="appear_bool_subpixel_rendering")],
        [InlineKeyboardButton(f"✂️ Hyphenate Before Scaling: {hyph_scale_label}", callback_data="appear_bool_hyphenate_before_scaling")],
        [InlineKeyboardButton(f"❗ Detach Trailing Punctuation: {detach_punct_label}", callback_data="appear_bool_detach_trailing_punctuation")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_appearance")],
    ]
    return InlineKeyboardMarkup(rows)

APPEAR_BOOL_LABELS = {
    "use_ligatures": ("Ligatures", "menu_appear_font"),
    "auto_vertical_text": ("Auto-Vertical Text", "menu_appear_font"),
    "subpixel_rendering": ("Subpixel Rendering", "menu_appear_layout"),
    "hyphenate_before_scaling": ("Hyphenate Before Scaling", "menu_appear_layout"),
    "detach_trailing_punctuation": ("Detach Trailing Punctuation", "menu_appear_layout"),
}

def kb_appear_bool_select(cfg, field):
    val = cfg.get(field)
    def mark(v): return "✅ " if val == v else ""
    _, back_target = APPEAR_BOOL_LABELS.get(field, ("Setting", "menu_appearance"))
    rows = [
        [InlineKeyboardButton(f"{mark(True)}On", callback_data=f"appearboolset_{field}_on")],
        [InlineKeyboardButton(f"{mark(False)}Off", callback_data=f"appearboolset_{field}_off")],
        [InlineKeyboardButton(f"{'✅ ' if val is None else ''}Original/Default", callback_data=f"appearboolset_{field}_default")],
        [InlineKeyboardButton("🔙 Back", callback_data=back_target)],
    ]
    return InlineKeyboardMarkup(rows)

FONT_HINTING_OPTIONS = ["none", "slight", "normal", "full"]

def kb_font_hinting_select(cfg):
    current = cfg.get("font_hinting")
    rows = []
    for opt in FONT_HINTING_OPTIONS:
        mark = "✅ " if current == opt else ""
        rows.append([InlineKeyboardButton(f"{mark}{opt}", callback_data=f"hintingset_{opt}")])
    rows.append([InlineKeyboardButton(f"{'✅ ' if current is None else ''}Original/Default (none)", callback_data="hintingset_default")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_appear_font")])
    return InlineKeyboardMarkup(rows)

# ================= Detection Settings Keyboards (YOLO/OCR) =================
SEG_MODEL_OPTIONS = ["yolo", "sam2", "sam3"]
BUBBLE_DETECTOR_OPTIONS = ["yolo_1", "yolo_2"]
OCR_METHOD_OPTIONS = ["LLM", "manga-ocr", "paddleocr-vl-1.6", "paddleocr-classic", "paddleocr-classic-v5"]
OCR_METHOD_LABELS = {
    "LLM": "LLM (vision model)",
    "manga-ocr": "manga-ocr (Japanese only)",
    "paddleocr-vl-1.6": "PaddleOCR-VL (VLM, slow on CPU)",
    "paddleocr-classic": "PaddleOCR Classic (JP/KR/CN/EN)",
    "paddleocr-classic-v5": "PaddleOCR Classic v5 (JP/KR/CN/EN, newer & more accurate)",
}
TRANSLATION_MODE_OPTIONS = ["one-step", "two-step"]

# ================= Value Validation (prevents invalid CLI args reaching main.py) =================
# Mirrors the argparse `choices=[...]` in MangaTranslator/main.py. Any cfg value that
# isn't in these sets (e.g. from a hand-edited or stale imported JSON file) gets
# reset to None (= engine default) instead of being passed to the CLI and crashing
# the whole job with "invalid choice: '...'".
VALID_CHOICES = {
    "translation_mode": set(TRANSLATION_MODE_OPTIONS),
    "seg_model": set(SEG_MODEL_OPTIONS),
    "bubble_detector_model": set(BUBBLE_DETECTOR_OPTIONS),
    "ocr_method": set(OCR_METHOD_OPTIONS),
    "reading_direction": {"rtl", "ltr"},
    "reasoning_effort": {"xhigh", "high", "medium", "low", "minimal", "none"},
    "effort": {"high", "medium", "low"},
    "verbosity": {"high", "medium", "low"},
    "media_resolution": {"auto", "high", "medium", "low"},
    "media_resolution_bubbles": {"auto", "high", "medium", "low"},
    "media_resolution_context": {"auto", "high", "medium", "low"},
    "image_detail": {"auto", "original", "high", "low"},
    "upscale_method": {"model", "model_lite", "lanczos", "none"},
    "image_upscale_mode": {"off", "initial", "final"},
    "osb_inpainting_method": {"flux_klein_9b", "flux_klein_4b", "flux_kontext", "opencv", "none"},
    "osb_flux_backend": {"sdcpp", "sdnq", "nunchaku"},
    "osb_flux_sdcpp_cache_mode": {"spectrum", "cache-dit", "taylorseer", "dbcache", "none"},
    "osb_font_hinting": {"none", "slight", "normal", "full"},
}

def sanitize_cfg_values(cfg):
    """Reset any cfg field to None (engine default) if it holds a value that isn't
    a legal choice for that field's CLI flag. Returns [(field, bad_value), ...] that
    were cleared, so the caller can tell the user exactly what was wrong."""
    cleared = []
    for field, allowed in VALID_CHOICES.items():
        val = cfg.get(field)
        if val is not None and val not in allowed:
            cleared.append((field, val))
            cfg[field] = None
    return cleared

# ================= Extended Settings Registry (auto-derived from main.py argparse) =================
# Every field here mirrors an actual --flag in MangaTranslator/main.py exactly (type, default,
# choices), so values entered through these menus can never desync from what the engine accepts.
FIELD_REGISTRY = {
    'temperature': {'group': 'generation', 'label': '🌡 Temperature', 'vtype': 'val', 'argtype': 'float', 'default': 0.1, 'choices': None, 'hint': '0.0-2.0'},
    'top_p': {'group': 'generation', 'label': '🎯 Top P', 'vtype': 'val', 'argtype': 'float', 'default': 0.95, 'choices': None, 'hint': '0.0-1.0'},
    'top_k': {'group': 'generation', 'label': '🔢 Top K', 'vtype': 'val', 'argtype': 'int', 'default': 1, 'choices': None, 'hint': 'positive integer'},
    'max_tokens': {'group': 'generation', 'label': '📏 Max Tokens', 'vtype': 'val', 'argtype': 'int', 'default': None, 'choices': None, 'hint': '2048-32768'},
    'use_custom_sampling': {'group': 'generation', 'label': '🎛 Custom Sampling', 'vtype': 'bool_invert', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'reasoning_effort': {'group': 'generation', 'label': '🧠 Reasoning Effort', 'vtype': 'val', 'argtype': 'str', 'default': 'medium', 'choices': ['xhigh', 'high', 'medium', 'low', 'minimal', 'none'], 'hint': None},
    'effort': {'group': 'generation', 'label': '⚡ Effort', 'vtype': 'val', 'argtype': 'str', 'default': 'medium', 'choices': ['high', 'medium', 'low'], 'hint': None},
    'verbosity': {'group': 'generation', 'label': '💬 Verbosity', 'vtype': 'val', 'argtype': 'str', 'default': 'low', 'choices': ['high', 'medium', 'low'], 'hint': None},
    'reading_direction': {'group': 'generation', 'label': '↔️ Reading Direction', 'vtype': 'val', 'argtype': 'str', 'default': 'rtl', 'choices': ['rtl', 'ltr'], 'hint': None},
    'enable_web_search': {'group': 'generation', 'label': '🌐 Web Search', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'enable_code_execution': {'group': 'generation', 'label': '💻 Code Execution', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'media_resolution': {'group': 'generation', 'label': '🖼 Media Resolution', 'vtype': 'val', 'argtype': 'str', 'default': 'auto', 'choices': ['auto', 'high', 'medium', 'low'], 'hint': None},
    'media_resolution_bubbles': {'group': 'generation', 'label': '🫧 Media Resolution (Bubbles)', 'vtype': 'val', 'argtype': 'str', 'default': 'auto', 'choices': ['auto', 'high', 'medium', 'low'], 'hint': None},
    'media_resolution_context': {'group': 'generation', 'label': '📄 Media Resolution (Context)', 'vtype': 'val', 'argtype': 'str', 'default': 'auto', 'choices': ['auto', 'high', 'medium', 'low'], 'hint': None},
    'image_detail': {'group': 'generation', 'label': '🔍 Image Detail', 'vtype': 'val', 'argtype': 'str', 'default': 'auto', 'choices': ['auto', 'original', 'high', 'low'], 'hint': None},
    'send_full_page_context': {'group': 'generation', 'label': '📃 Send Full Page Context', 'vtype': 'bool_invert', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'inpaint_colored_bubbles': {'group': 'cleaning', 'label': '🎨 Inpaint Colored Bubbles', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'use_otsu_threshold': {'group': 'cleaning', 'label': '⬜ Use Otsu Threshold', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'thresholding_value': {'group': 'cleaning', 'label': '🎚 Thresholding Value', 'vtype': 'val', 'argtype': 'int', 'default': 200, 'choices': None, 'hint': '0-255'},
    'roi_shrink_px': {'group': 'cleaning', 'label': '📐 ROI Shrink (px)', 'vtype': 'val', 'argtype': 'int', 'default': 5, 'choices': None, 'hint': '0-10'},
    'whiteout_conjoined_bubbles': {'group': 'cleaning', 'label': '⬜ Whiteout Conjoined Bubbles', 'vtype': 'bool_invert', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'upscale_method': {'group': 'cleaning', 'label': '⬆️ Upscale Method', 'vtype': 'val', 'argtype': 'str', 'default': 'model_lite', 'choices': ['model', 'model_lite', 'lanczos', 'none'], 'hint': None},
    'image_upscale_mode': {'group': 'cleaning', 'label': '⬆️ Image Upscale Mode', 'vtype': 'val', 'argtype': None, 'default': 'off', 'choices': ['off', 'initial', 'final'], 'hint': None},
    'image_upscale_factor': {'group': 'cleaning', 'label': '✖️ Image Upscale Factor', 'vtype': 'val', 'argtype': 'float', 'default': 2.0, 'choices': None, 'hint': '1.0-8.0'},
    'jpeg_quality': {'group': 'cleaning', 'label': '🖼 JPEG Quality', 'vtype': 'val', 'argtype': 'int', 'default': 95, 'choices': None, 'hint': '1-100'},
    'png_compression': {'group': 'cleaning', 'label': '🗜 PNG Compression', 'vtype': 'val', 'argtype': 'int', 'default': 2, 'choices': None, 'hint': '0-6'},
    'auto_scale': {'group': 'cleaning', 'label': '📏 Auto Scale', 'vtype': 'bool_invert', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'bubble_min_side_pixels': {'group': 'cleaning', 'label': '🫧 Bubble Min Side (px)', 'vtype': 'val', 'argtype': 'int', 'default': 128, 'choices': None, 'hint': 'positive integer'},
    'context_image_max_side_pixels': {'group': 'cleaning', 'label': '📄 Context Image Max Side (px)', 'vtype': 'val', 'argtype': 'int', 'default': 1024, 'choices': None, 'hint': 'positive integer'},
    'parallel_requests': {'group': 'batch', 'label': '⚙️ Parallel Requests', 'vtype': 'val', 'argtype': 'int', 'default': 1, 'choices': None, 'hint': '1-20'},
    'batch_parallel_within_pages': {'group': 'batch', 'label': '⚙️ Parallel Within Pages', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'batch_previous_context_images': {'group': 'batch', 'label': '🖼 Previous Context Images', 'vtype': 'val', 'argtype': 'int', 'default': 0, 'choices': None, 'hint': '0-10'},
    'batch_previous_context_texts': {'group': 'batch', 'label': '📝 Previous Context Texts', 'vtype': 'val', 'argtype': 'int', 'default': 3, 'choices': None, 'hint': '0-50'},
    'retry_failed_once': {'group': 'batch', 'label': '🔁 Retry Failed Once', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'overlap_llm_with_inpaint': {'group': 'batch', 'label': '⏱ Overlap LLM With Inpaint', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'verbose': {'group': 'batch', 'label': '🔊 Verbose Logging', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'cpu': {'group': 'batch', 'label': '🖥 Force CPU', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'cleaning_only': {'group': 'batch', 'label': '🧹 Cleaning Only', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'upscaling_only': {'group': 'batch', 'label': '⬆️ Upscaling Only', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'test_mode': {'group': 'batch', 'label': '🧪 Test Mode', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'osb_inpainting_method': {'group': 'osb', 'label': '🎨 OSB Inpainting Method', 'vtype': 'val', 'argtype': 'str', 'default': 'flux_klein_4b', 'choices': ['flux_klein_9b', 'flux_klein_4b', 'flux_kontext', 'opencv', 'none'], 'hint': None},
    'osb_flux_backend': {'group': 'osb', 'label': '⚙️ OSB Flux Backend', 'vtype': 'val', 'argtype': 'str', 'default': 'sdnq', 'choices': ['sdcpp', 'sdnq', 'nunchaku'], 'hint': None},
    'osb_flux_low_vram': {'group': 'osb', 'label': '💾 OSB Flux Low VRAM', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'osb_flux_sdcpp_cache_mode': {'group': 'osb', 'label': '💾 OSB Flux SDCPP Cache Mode', 'vtype': 'val', 'argtype': 'str', 'default': 'none', 'choices': ['spectrum', 'cache-dit', 'taylorseer', 'dbcache', 'none'], 'hint': None},
    'osb_flux_sdcpp_diffusion_quant': {'group': 'osb', 'label': '🔢 OSB Flux SDCPP Diffusion Quant', 'vtype': 'val', 'argtype': 'str', 'default': 'Q4_K_M', 'choices': None, 'hint': 'text, e.g. Q4_K_M'},
    'osb_flux_sdcpp_text_encoder_quant': {'group': 'osb', 'label': '🔢 OSB Flux SDCPP Text Encoder Quant', 'vtype': 'val', 'argtype': 'str', 'default': None, 'choices': None, 'hint': 'text'},
    'osb_flux_upscale_small_crops': {'group': 'osb', 'label': '⬆️ OSB Flux Upscale Small Crops', 'vtype': 'bool_invert', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'osb_flux_group_regions': {'group': 'osb', 'label': '🧩 OSB Flux Group Regions', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'osb_sfx_skip_inpaint': {'group': 'osb', 'label': '🔇 OSB SFX Skip Inpaint', 'vtype': 'bool_invert', 'argtype': None, 'default': True, 'choices': None, 'hint': None},
    'osb_flux_steps': {'group': 'osb', 'label': '🔁 OSB Flux Steps', 'vtype': 'val', 'argtype': 'int', 'default': 4, 'choices': None, 'hint': 'positive integer'},
    'osb_flux_luminance_correction': {'group': 'osb', 'label': '💡 OSB Flux Luminance Correction', 'vtype': 'bool_invert', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'osb_flux_residual_threshold': {'group': 'osb', 'label': '🎚 OSB Flux Residual Threshold', 'vtype': 'val', 'argtype': 'float', 'default': 0.15, 'choices': None, 'hint': 'decimal'},
    'osb_seed': {'group': 'osb', 'label': '🌱 OSB Seed', 'vtype': 'val', 'argtype': 'int', 'default': 1, 'choices': None, 'hint': 'integer'},
    'osb_max_font_size': {'group': 'osb', 'label': '🔠 OSB Max Font Size (px)', 'vtype': 'val', 'argtype': 'int', 'default': 64, 'choices': None, 'hint': 'positive integer'},
    'osb_min_font_size': {'group': 'osb', 'label': '🔡 OSB Min Font Size (px)', 'vtype': 'val', 'argtype': 'int', 'default': 10, 'choices': None, 'hint': 'positive integer'},
    'osb_use_ligatures': {'group': 'osb', 'label': '🔗 OSB Ligatures', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'osb_outline_width': {'group': 'osb', 'label': '⭕ OSB Outline Width', 'vtype': 'val', 'argtype': 'float', 'default': 3.0, 'choices': None, 'hint': 'decimal'},
    'osb_line_spacing': {'group': 'osb', 'label': '📏 OSB Line Spacing', 'vtype': 'val', 'argtype': 'float', 'default': 1.0, 'choices': None, 'hint': 'decimal'},
    'osb_use_subpixel': {'group': 'osb', 'label': '🖥 OSB Subpixel Rendering', 'vtype': 'bool_true', 'argtype': None, 'default': True, 'choices': None, 'hint': None},
    'osb_font_hinting': {'group': 'osb', 'label': '🔎 OSB Font Hinting', 'vtype': 'val', 'argtype': 'str', 'default': 'none', 'choices': ['none', 'slight', 'normal', 'full'], 'hint': None},
    'osb_bbox_expansion': {'group': 'osb', 'label': '📦 OSB Bbox Expansion', 'vtype': 'val', 'argtype': 'float', 'default': 0.1, 'choices': None, 'hint': 'decimal'},
    'osb_render_expansion_narrow': {'group': 'osb', 'label': '↔️ OSB Render Expansion (Narrow)', 'vtype': 'val', 'argtype': 'float', 'default': 1.0, 'choices': None, 'hint': 'decimal'},
    'osb_render_expansion_tiny': {'group': 'osb', 'label': '🔬 OSB Render Expansion (Tiny)', 'vtype': 'val', 'argtype': 'float', 'default': 1.0, 'choices': None, 'hint': 'decimal'},
    'osb_render_expansion_aspect_threshold': {'group': 'osb', 'label': '📐 OSB Aspect Threshold', 'vtype': 'val', 'argtype': 'float', 'default': 0.4, 'choices': None, 'hint': 'decimal'},
    'osb_render_expansion_area_threshold': {'group': 'osb', 'label': '📐 OSB Area Threshold', 'vtype': 'val', 'argtype': 'float', 'default': 0.005, 'choices': None, 'hint': 'decimal'},
    'osb_text_box_proximity_ratio': {'group': 'osb', 'label': '📏 OSB Text Box Proximity Ratio', 'vtype': 'val', 'argtype': 'float', 'default': 0.02, 'choices': None, 'hint': 'decimal'},
    'osb_confidence': {'group': 'osb', 'label': '🎯 OSB Confidence', 'vtype': 'val', 'argtype': 'float', 'default': 0.6, 'choices': None, 'hint': '0.0-1.0'},
    'osb_filter_page_numbers': {'group': 'osb', 'label': '🔢 OSB Filter Page Numbers', 'vtype': 'bool_true', 'argtype': None, 'default': None, 'choices': None, 'hint': None},
    'osb_page_filter_margin': {'group': 'osb', 'label': '📐 OSB Page Filter Margin', 'vtype': 'val', 'argtype': 'float', 'default': 0.1, 'choices': None, 'hint': 'decimal'},
    'osb_page_filter_min_area': {'group': 'osb', 'label': '📐 OSB Page Filter Min Area', 'vtype': 'val', 'argtype': 'float', 'default': 0.05, 'choices': None, 'hint': 'decimal'},
    'osb_min_area_ignore_ratio': {'group': 'osb', 'label': '📐 OSB Min Area Ignore Ratio', 'vtype': 'val', 'argtype': 'float', 'default': 0.0, 'choices': None, 'hint': 'decimal'},
    'osb_min_side_pixels': {'group': 'osb', 'label': '🫧 OSB Min Side (px)', 'vtype': 'val', 'argtype': 'int', 'default': 128, 'choices': None, 'hint': 'positive integer'},
}

FIELD_GROUPS = {
    'generation': ['temperature', 'top_p', 'top_k', 'max_tokens', 'use_custom_sampling', 'reasoning_effort', 'effort', 'verbosity', 'reading_direction', 'enable_web_search', 'enable_code_execution', 'media_resolution', 'media_resolution_bubbles', 'media_resolution_context', 'image_detail', 'send_full_page_context'],
    'cleaning': ['inpaint_colored_bubbles', 'use_otsu_threshold', 'thresholding_value', 'roi_shrink_px', 'whiteout_conjoined_bubbles', 'upscale_method', 'image_upscale_mode', 'image_upscale_factor', 'jpeg_quality', 'png_compression', 'auto_scale', 'bubble_min_side_pixels', 'context_image_max_side_pixels'],
    'batch': ['parallel_requests', 'batch_parallel_within_pages', 'batch_previous_context_images', 'batch_previous_context_texts', 'retry_failed_once', 'overlap_llm_with_inpaint', 'verbose', 'cpu', 'cleaning_only', 'upscaling_only', 'test_mode'],
    'osb': ['osb_inpainting_method', 'osb_flux_backend', 'osb_flux_low_vram', 'osb_flux_sdcpp_cache_mode', 'osb_flux_sdcpp_diffusion_quant', 'osb_flux_sdcpp_text_encoder_quant', 'osb_flux_upscale_small_crops', 'osb_flux_group_regions', 'osb_sfx_skip_inpaint', 'osb_flux_steps', 'osb_flux_luminance_correction', 'osb_flux_residual_threshold', 'osb_seed', 'osb_max_font_size', 'osb_min_font_size', 'osb_use_ligatures', 'osb_outline_width', 'osb_line_spacing', 'osb_use_subpixel', 'osb_font_hinting', 'osb_bbox_expansion', 'osb_render_expansion_narrow', 'osb_render_expansion_tiny', 'osb_render_expansion_aspect_threshold', 'osb_render_expansion_area_threshold', 'osb_text_box_proximity_ratio', 'osb_confidence', 'osb_filter_page_numbers', 'osb_page_filter_margin', 'osb_page_filter_min_area', 'osb_min_area_ignore_ratio', 'osb_min_side_pixels'],
}

FIELD_GROUP_TITLES = {
    "generation": "🧠 Generation Settings",
    "cleaning": "🧹 Cleaning & Upscaling",
    "batch": "⚙️ Batch & Performance",
    "osb": "🫧 OSB (Outside-Bubble Text) Tuning",
}

# ================= Generic Extended-Settings UI (data-driven from FIELD_REGISTRY) =================
def _fmt_field_value(key):
    meta = FIELD_REGISTRY[key]
    def _get(cfg):
        v = cfg.get(key)
        if v is None:
            d = meta["default"]
            return f"Original/Default ({d})" if d is not None else "Original/Default"
        return str(v)
    return _get

def kb_field_group_menu(cfg, group):
    keys = FIELD_GROUPS[group]
    rows = []
    for key in keys:
        meta = FIELD_REGISTRY[key]
        val = cfg.get(key)
        if meta["vtype"] in ("bool_true", "bool_invert"):
            if val is None:
                shown = "Original/Default"
            else:
                shown = "✅ On" if val else "❌ Off"
        elif meta["choices"]:
            shown = val if val is not None else f"Original/Default ({meta['default']})"
        else:
            shown = val if val is not None else f"Original/Default ({meta['default']})" if meta["default"] is not None else "Original/Default"
        rows.append([InlineKeyboardButton(f"{meta['label']}: {shown}", callback_data=f"xf_open_{group}_{key}")])
    rows.append([InlineKeyboardButton("♻️ Reset Group to Original/Default", callback_data=f"xf_reset_{group}")])
    rows.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)

def kb_field_bool_select(cfg, key):
    meta = FIELD_REGISTRY[key]
    val = cfg.get(key)
    def mark(v): return "✅ " if val == v else ""
    rows = [
        [InlineKeyboardButton(f"{mark(True)}On", callback_data=f"xfboolset_{key}_on")],
        [InlineKeyboardButton(f"{mark(False)}Off", callback_data=f"xfboolset_{key}_off")],
        [InlineKeyboardButton(f"{'✅ ' if val is None else ''}Original/Default", callback_data=f"xfboolset_{key}_default")],
        [InlineKeyboardButton("🔙 Back", callback_data=f"xf_group_{meta['group']}")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_field_choice_select(cfg, key):
    meta = FIELD_REGISTRY[key]
    current = cfg.get(key)
    rows = []
    for opt in meta["choices"]:
        mark = "✅ " if current == opt else ""
        rows.append([InlineKeyboardButton(f"{mark}{opt}", callback_data=f"xfchoiceset_{key}::{opt}")])
    default_disp = meta["default"] if meta["default"] is not None else "engine default"
    rows.append([InlineKeyboardButton(f"{'✅ ' if current is None else ''}Original/Default ({default_disp})", callback_data=f"xfchoiceset_{key}::__default__")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data=f"xf_group_{meta['group']}")])
    return InlineKeyboardMarkup(rows)

async def open_field_editor(query, cfg, group, key):
    meta = FIELD_REGISTRY[key]
    if meta["vtype"] in ("bool_true", "bool_invert"):
        await safe_edit(query.message, f"{meta['label']}:", reply_markup=kb_field_bool_select(cfg, key))
        return True
    if meta["choices"]:
        await safe_edit(query.message, f"{meta['label']}:", reply_markup=kb_field_choice_select(cfg, key))
        return True
    return False  # numeric/text field -> caller sets awaiting_reply and prompts

def _detect_val_label(cfg, field, default_display):
    v = cfg.get(field)
    return f"Original/Default ({default_display})" if v is None else str(v)

def kb_detection_menu(cfg):
    conf = _detect_val_label(cfg, "confidence", "0.6")
    conj_conf = _detect_val_label(cfg, "conjoined_confidence", "0.35")
    panel_conf = _detect_val_label(cfg, "panel_confidence", "0.25")
    seg = _detect_val_label(cfg, "seg_model", "yolo")
    bubble_model = _detect_val_label(cfg, "bubble_detector_model", "yolo_1")
    ocr_raw = cfg.get("ocr_method")
    ocr = "Original/Default (LLM)" if ocr_raw is None else OCR_METHOD_LABELS.get(ocr_raw, ocr_raw)
    conj_det = cfg.get("conjoined_detection")
    conj_det_label = "Original/Default (On)" if conj_det is None else ("✅ On" if conj_det else "❌ Off")

    trans_mode = _detect_val_label(cfg, "translation_mode", "one-step")
    manual_mode_label = "✅ On (edit translations yourself)" if cfg.get("manual_translation_mode") else "❌ Off (AI translates automatically)"

    rows = [
        [InlineKeyboardButton(f"🎯 Bubble Confidence: {conf}", callback_data="detect_field_confidence")],
        [InlineKeyboardButton(f"🔗 Conjoined Confidence: {conj_conf}", callback_data="detect_field_conjoined_confidence")],
        [InlineKeyboardButton(f"🖼 Panel Confidence: {panel_conf}", callback_data="detect_field_panel_confidence")],
        [InlineKeyboardButton(f"🧩 Conjoined Bubble Detection: {conj_det_label}", callback_data="detect_bool_conjoined_detection")],
        [InlineKeyboardButton(f"🧠 Segmentation Model: {seg}", callback_data="detect_seg_open")],
        [InlineKeyboardButton(f"🫧 Bubble Detector Model: {bubble_model}", callback_data="detect_bubblemodel_open")],
        [InlineKeyboardButton(f"👁 OCR Method: {ocr}", callback_data="detect_ocr_open")],
        [InlineKeyboardButton(f"🔀 Translation Mode: {trans_mode}", callback_data="detect_transmode_open")],
        [InlineKeyboardButton(f"✍️ Manual Translation Mode: {manual_mode_label}", callback_data="detect_bool_manual_translation_mode")],
        [InlineKeyboardButton("♻️ Reset All to Original/Default", callback_data="detect_reset_all")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_translation_mode_select(cfg):
    current = cfg.get("translation_mode")
    rows = []
    for opt in TRANSLATION_MODE_OPTIONS:
        mark = "✅ " if current == opt else ""
        label = f"{opt} (combines OCR/Translate)" if opt == "one-step" else f"{opt} (separates OCR/Translate — better for weaker LLMs)"
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"transmodeset_{opt}")])
    rows.append([InlineKeyboardButton(f"{'✅ ' if current is None else ''}Original/Default (one-step)", callback_data="transmodeset_default")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_detection")])
    return InlineKeyboardMarkup(rows)

def kb_detect_bool_select(cfg, field):
    val = cfg.get(field)
    def mark(v): return "✅ " if val == v else ""
    rows = [
        [InlineKeyboardButton(f"{mark(True)}On", callback_data=f"detectboolset_{field}_on")],
        [InlineKeyboardButton(f"{mark(False)}Off", callback_data=f"detectboolset_{field}_off")],
        [InlineKeyboardButton(f"{'✅ ' if val is None else ''}Original/Default", callback_data=f"detectboolset_{field}_default")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_detection")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_manual_mode_select(cfg):
    val = bool(cfg.get("manual_translation_mode"))
    rows = [
        [InlineKeyboardButton(f"{'✅ ' if val else ''}✍️ On — I'll edit translations myself", callback_data="manualmodeset_on")],
        [InlineKeyboardButton(f"{'✅ ' if not val else ''}🤖 Off — AI translates automatically", callback_data="manualmodeset_off")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_detection")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_seg_model_select(cfg):
    current = cfg.get("seg_model")
    rows = []
    for opt in SEG_MODEL_OPTIONS:
        mark = "✅ " if current == opt else ""
        rows.append([InlineKeyboardButton(f"{mark}{opt}", callback_data=f"segset_{opt}")])
    rows.append([InlineKeyboardButton(f"{'✅ ' if current is None else ''}Original/Default (yolo)", callback_data="segset_default")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_detection")])
    return InlineKeyboardMarkup(rows)

def kb_bubble_model_select(cfg):
    current = cfg.get("bubble_detector_model")
    rows = []
    for opt in BUBBLE_DETECTOR_OPTIONS:
        mark = "✅ " if current == opt else ""
        rows.append([InlineKeyboardButton(f"{mark}{opt}", callback_data=f"bubblemodelset_{opt}")])
    rows.append([InlineKeyboardButton(f"{'✅ ' if current is None else ''}Original/Default (yolo_1)", callback_data="bubblemodelset_default")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_detection")])
    return InlineKeyboardMarkup(rows)

def kb_ocr_method_select(cfg):
    current = cfg.get("ocr_method")
    rows = []
    for opt in OCR_METHOD_OPTIONS:
        mark = "✅ " if current == opt else ""
        label = OCR_METHOD_LABELS.get(opt, opt)
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"ocrmethodset_{opt}")])
    rows.append([InlineKeyboardButton(f"{'✅ ' if current is None else ''}Original/Default (LLM)", callback_data="ocrmethodset_default")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_detection")])
    return InlineKeyboardMarkup(rows)

# ================= Tiling Settings Keyboards (Manhwa) =================
def _tile_val_label(cfg, field, default_display):
    v = cfg.get(field)
    return f"Original/Default ({default_display})" if v is None else str(v)

def kb_tiling_menu(cfg):
    enabled = cfg.get("tile_enabled")
    enabled_label = "Original/Default (On)" if enabled is None else ("✅ On" if enabled else "❌ Off")
    height = _tile_val_label(cfg, "tile_height", str(MANHWA_TILE_HEIGHT))
    trigger = _tile_val_label(cfg, "tile_trigger_height", str(MANHWA_TILE_TRIGGER_HEIGHT))
    flat = _tile_val_label(cfg, "tile_flat_threshold", str(MANHWA_SAFE_CUT_FLAT_THRESHOLD))
    band = _tile_val_label(cfg, "tile_seam_band_px", str(SEAM_CHECK_BAND_PX))
    diff = _tile_val_label(cfg, "tile_seam_diff_threshold", str(SEAM_DUPLICATE_DIFF_THRESHOLD))
    min_cuts = _tile_val_label(cfg, "tile_min_cuts", str(MANHWA_TILE_MIN_CUTS))
    max_cuts = _tile_val_label(cfg, "tile_max_cuts", str(MANHWA_TILE_MAX_CUTS))

    radius_auto = cfg.get("tile_search_radius_auto")
    if radius_auto:
        radius = "🔁 Auto (expands until a safe cut is found)"
    else:
        radius = _tile_val_label(cfg, "tile_search_radius", str(MANHWA_TILE_OVERLAP))

    rows = [
        [InlineKeyboardButton(f"✂️ Tiling: {enabled_label}", callback_data="tile_bool_tile_enabled")],
        [InlineKeyboardButton(f"📏 Trigger Height (px): {trigger}", callback_data="tile_field_tile_trigger_height")],
        [InlineKeyboardButton(f"📐 Tile Height (px): {height}", callback_data="tile_field_tile_height")],
        [InlineKeyboardButton(f"⬇️ Min Cuts: {min_cuts}", callback_data="tile_field_tile_min_cuts")],
        [InlineKeyboardButton(f"⬆️ Max Cuts: {max_cuts}", callback_data="tile_field_tile_max_cuts")],
        [InlineKeyboardButton(f"🔍 Safe-Cut Search Radius (px): {radius}", callback_data="tile_search_radius_open")],
        [InlineKeyboardButton(f"⬜ Flatness Threshold: {flat}", callback_data="tile_field_tile_flat_threshold")],
        [InlineKeyboardButton(f"📊 Seam Check Band (px): {band}", callback_data="tile_field_tile_seam_band_px")],
        [InlineKeyboardButton(f"🎯 Seam Duplicate Diff Threshold: {diff}", callback_data="tile_field_tile_seam_diff_threshold")],
        [InlineKeyboardButton("♻️ Reset All to Original/Default", callback_data="tile_reset_all")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_tile_search_radius_menu(cfg):
    auto = cfg.get("tile_search_radius_auto")
    radius = _tile_val_label(cfg, "tile_search_radius", str(MANHWA_TILE_OVERLAP))
    rows = [
        [InlineKeyboardButton(f"{'✅ ' if auto else ''}🔁 Auto (recommended)", callback_data="tileradiusauto_on")],
        [InlineKeyboardButton(f"{'✅ ' if not auto else ''}✏️ Manual Value: {radius}", callback_data="tileradiusauto_off")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_tiling")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_tile_bool_select(cfg, field):
    val = cfg.get(field)
    def mark(v): return "✅ " if val == v else ""
    rows = [
        [InlineKeyboardButton(f"{mark(True)}On", callback_data=f"tileboolset_{field}_on")],
        [InlineKeyboardButton(f"{mark(False)}Off", callback_data=f"tileboolset_{field}_off")],
        [InlineKeyboardButton(f"{'✅ ' if val is None else ''}Original/Default", callback_data=f"tileboolset_{field}_default")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_tiling")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_api_menu(cfg):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Provider: {cfg['provider']}", callback_data="api_provider_open")],
        [InlineKeyboardButton(f"Base URL: {cfg['api_url'] or 'not set'}", callback_data="api_field_api_url")],
        [InlineKeyboardButton("API Key: " + ("••••••" if cfg['api_key'] else "not set"), callback_data="api_field_api_key")],
        [InlineKeyboardButton(f"Model ID: {cfg['model_name'] or 'not set'}", callback_data="api_field_model_name")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
    ])

def kb_provider_select(cfg):
    rows = []
    for p in PROVIDERS:
        mark = "✅ " if cfg["provider"] == p else ""
        rows.append([InlineKeyboardButton(f"{mark}{p}", callback_data=f"provset_{p}")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_api")])
    return InlineKeyboardMarkup(rows)

def kb_prompt_root():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🖥 System Prompt", callback_data="prompt_open_system")],
        [InlineKeyboardButton("👤 User Prompt", callback_data="prompt_open_user")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
    ])

def kb_prompt_list(kind, cfg):
    lib = load_prompt_library(kind)
    selected_name = cfg["system_prompt_name"] if kind == "system" else cfg["user_prompt_name"]
    rows = []
    for name in lib.keys():
        mark = "✅ " if name == selected_name else ""
        row = [InlineKeyboardButton(f"{mark}{name}", callback_data=f"promptsel_{kind}_{name}")]
        if len(lib) > 1:
            row.append(InlineKeyboardButton("🗑", callback_data=f"promptdel_{kind}_{name}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("➕ Add New", callback_data=f"prompt_add_{kind}")])
    rows.append([InlineKeyboardButton("✏️ Edit Selected", callback_data=f"prompt_edit_{kind}")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="menu_prompt")])
    return InlineKeyboardMarkup(rows)

def kb_output_menu(cfg):
    rows = []
    for label, code in OUTPUT_FORMATS:
        mark = "✅ " if cfg["output_format"] == code else ""
        rows.append([InlineKeyboardButton(f"{mark}{label}", callback_data=f"outset_{code}")])
    osb_mark = "✅ " if cfg.get("osb_enabled", True) else "❌ "
    rows.append([InlineKeyboardButton(f"{osb_mark}OSB (Outside-Bubble Text)", callback_data="osb_toggle")])
    rows.append([InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)

def kb_backup_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📤 Export Settings (JSON)", callback_data="backup_export")],
        [InlineKeyboardButton("📥 Import Settings (JSON)", callback_data="backup_import")],
        [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")],
    ])

def kb_upload_mode():
    rows = [[InlineKeyboardButton(label, callback_data=f"uploadmode_{code}")] for label, code in UPLOAD_MODES]
    return InlineKeyboardMarkup(rows)

def kb_cancel_only():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🛑 Cancel", callback_data="job_cancel")]])

def kb_resume_options():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Continue", callback_data="job_continue")],
        [InlineKeyboardButton("📤 Send Translated Files", callback_data="job_send_partial")],
    ])

# ================= Base Commands =================
@app.on_message(filters.command("start") & auth_filter)
async def start_cmd(client, message: Message):
    await message.reply_text(
        "⚡ **Yomi Subs Core Engine Online** ⚡\n\n"
        "Welcome back, Senpai!\n"
        "👉 Send `/translate` to begin a new job.\n"
        "⚙️ Send `/settings` to configure language, font, API, and prompts."
    )

@app.on_message(filters.command("settings") & auth_filter)
async def settings_cmd(client, message: Message):
    cfg = get_user_config(message.from_user.id)
    await message.reply_text("🛠 **Settings**\nChoose a category to configure:", reply_markup=kb_main_menu(cfg))

@app.on_message(filters.command("translate") & auth_filter)
async def translate_cmd(client, message: Message):
    user_id = message.from_user.id
    if user_id in active_jobs:
        await message.reply_text("⚠️ A job is already running. Cancel or complete it first.")
        return
    pending_files[user_id] = {"mode": None, "files": [], "collecting": False}
    await message.reply_text(
        "📋 **What are you going to upload?**\nSelect the input type below:",
        reply_markup=kb_upload_mode()
    )

@app.on_message(filters.command("end") & auth_filter)
async def end_cmd(client, message: Message):
    user_id = message.from_user.id
    queue = pending_files.get(user_id)
    if not queue or not queue.get("collecting"):
        await message.reply_text("⚠️ No active upload session. Start with `/translate`.")
        return
    queue["collecting"] = False
    if not queue["files"]:
        await message.reply_text("❌ No files received. Try `/translate` again.")
        pending_files.pop(user_id, None)
        return
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Start Translation", callback_data="start_pipeline")]])
    await message.reply_text(
        f"✅ **{len(queue['files'])} file(s) queued.**\nReady to start translation?",
        reply_markup=kb
    )

@app.on_message(filters.command("cancel") & auth_filter)
async def cancel_cmd(client, message: Message):
    user_id = message.from_user.id
    if user_id in active_jobs:
        active_jobs[user_id]["cancel"] = True
        await message.reply_text("🛑 Cancellation requested. Job is stopping...")
    else:
        pending_files.pop(user_id, None)
        awaiting_reply.pop(user_id, None)
        manual_job = manual_jobs.pop(user_id, None)
        if manual_job:
            job_root = manual_job.get("job_root")
            if job_root and os.path.exists(job_root):
                shutil.rmtree(job_root, ignore_errors=True)
        await message.reply_text("✅ Session cleared.")

# ================= Safe Background Task Runner =================
def run_job(coro, status_msg, user_id):
    async def _runner():
        try:
            await coro
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                await safe_edit(status_msg, f"❌ **Job crashed unexpectedly:**\n`{type(e).__name__}: {e}`\n\nSend `/cancel` and try `/translate` again.")
            except Exception:
                pass
            active_jobs.pop(user_id, None)
    return asyncio.create_task(_runner())

# ================= Callback Query Router =================
@app.on_callback_query(auth_filter_cb)
async def handle_callbacks(client, query: CallbackQuery):
    data = query.data
    user_id = query.from_user.id
    cfg = get_user_config(user_id)

    await safe_answer(query)

    if data == "main_menu":
        await safe_edit(query.message, "🛠 **Settings**\nChoose a category to configure:", reply_markup=kb_main_menu(cfg))
        return

    if data == "menu_content_type":
        await safe_edit(
            query.message,
            f"📚 **Content Type**\nCurrent: `{cfg.get('content_type_label', 'Manhwa')}`\n\n"
            f"🍥 **Manhwa**: long vertical-scroll strips. Tall stitched pages "
            f"are automatically tiled before detection so bubbles don't get "
            f"crushed into invisibility.\n"
            f"📖 **Manga**: normal single manga pages, right-to-left panels.\n"
            f"💬 **Comic**: Western-style single-page comics.\n"
            f"📝 **Novel**: text-only prose, no bubbles/panels - skips the "
            f"image detection/rendering pipeline entirely.",
            reply_markup=kb_content_type_select(cfg)
        )
        return

    if data.startswith("ctypeset_"):
        code = data.split("_", 1)[1]
        label = next((l for l, c in CONTENT_TYPES if c == code), code)
        cfg["content_type"] = code
        cfg["content_type_label"] = label
        await save_user_config(user_id)
        await safe_answer(query, f"Content type set to {label}")
        await safe_edit(
            query.message,
            f"📚 **Content Type**\nCurrent: `{cfg['content_type_label']}`",
            reply_markup=kb_content_type_select(cfg)
        )
        return

    if data == "menu_lang":
        await safe_edit(query.message, 
            f"🌐 **Language Settings**\nSource: `{cfg['source_lang_label']}`\nTarget: `{cfg['target_lang_label']}`\n\nTap a field to change it:",
            reply_markup=kb_lang_root(cfg)
        )
        return

    if data == "lang_src_open":
        await safe_edit(query.message, "🌐 **Select Source Language:**", reply_markup=kb_source_select(cfg))
        return

    if data == "lang_tgt_open":
        await safe_edit(query.message, "🌐 **Select Target Language:**", reply_markup=kb_target_select(cfg))
        return

    if data.startswith("srcset_"):
        code = data.split("_", 1)[1]
        label = next((l for l, c in SOURCE_LANGS if c == code), code)
        cfg["source_lang"] = code
        cfg["source_lang_label"] = label
        await save_user_config(user_id)
        await safe_answer(query, f"Source set to {label}")
        await safe_edit(query.message, 
            f"🌐 **Language Settings**\nSource: `{cfg['source_lang_label']}`\nTarget: `{cfg['target_lang_label']}`\n\nTap a field to change it:",
            reply_markup=kb_lang_root(cfg)
        )
        return

    if data.startswith("tgtset_"):
        value = data.split("_", 1)[1]
        if value == "custom":
            awaiting_reply[user_id] = {"type": "custom_lang"}
            await safe_edit(query.message, "✍️ **Reply to this message with your target language name.**")
            return
        label = next((l for l, v in TARGET_LANGS if v == value), value)
        cfg["target_lang"] = value
        cfg["target_lang_label"] = label
        await save_user_config(user_id)
        await safe_answer(query, f"Target set to {label}")
        await safe_edit(query.message, 
            f"🌐 **Language Settings**\nSource: `{cfg['source_lang_label']}`\nTarget: `{cfg['target_lang_label']}`\n\nTap a field to change it:",
            reply_markup=kb_lang_root(cfg)
        )
        return

    if data == "menu_font":
        fonts = list_fonts()
        body = "🔡 **Font Track**\n"
        body += f"Selected: `{cfg.get('font_name') or 'none'}`\n\n"
        body += "Library:\n" + ("\n".join(f"• {f}" for f in fonts) if fonts else "_empty_")
        await safe_edit(query.message, body, reply_markup=kb_font_menu(cfg))
        return

    if data == "font_add":
        awaiting_reply[user_id] = {"type": "font_upload"}
        await safe_edit(query.message, "📤 **Upload your font now** (.ttf or .otf).\nSend it as a document reply, or just send the file directly in chat.")
        return

    if data.startswith("fontsel_"):
        name = data.split("_", 1)[1]
        cfg["font_name"] = name
        await save_user_config(user_id)
        await safe_answer(query, f"Font set to {name}")
        await safe_edit(query.message, 
            f"🔡 **Font Track**\nSelected: `{cfg['font_name']}`\n\nLibrary:\n" +
            "\n".join(f"• {f}" for f in list_fonts()),
            reply_markup=kb_font_menu(cfg)
        )
        return

    if data.startswith("fontdel_"):
        name = data.split("_", 1)[1]
        delete_font(name)
        if cfg.get("font_name") == name:
            cfg["font_name"] = None
            await save_user_config(user_id)
        await safe_answer(query, f"Deleted {name}")
        fonts = list_fonts()
        body = "🔡 **Font Track**\n" + f"Selected: `{cfg.get('font_name') or 'none'}`\n\nLibrary:\n" + \
               ("\n".join(f"• {f}" for f in fonts) if fonts else "_empty_")
        await safe_edit(query.message, body, reply_markup=kb_font_menu(cfg))
        return

    if data == "menu_appearance":
        await safe_edit(
            query.message,
            "🎨 **Appearance Settings**\n"
            "Controls how translated text is rendered inside speech bubbles.\n"
            "_\"Original/Default\" = untouched, exactly like before this menu existed._",
            reply_markup=kb_appearance_menu(cfg)
        )
        return

    if data == "menu_appear_font":
        await safe_edit(
            query.message,
            "🔡 **Font & Sizing**",
            reply_markup=kb_appear_font_menu(cfg)
        )
        return

    if data == "menu_appear_layout":
        await safe_edit(
            query.message,
            "📐 **Layout & Quality**",
            reply_markup=kb_appear_layout_menu(cfg)
        )
        return

    if data.startswith("appear_field_"):
        field = data.split("appear_field_", 1)[1]
        pretty = {
            "min_font_size": "Min Font Size (px)",
            "max_font_size": "Max Font Size (px)",
            "line_spacing_mult": "Line Spacing Multiplier",
            "padding_pixels": "Bubble Padding (px)",
            "supersampling_factor": "Supersampling Factor (1-4)",
            "badness_exponent": "Line Badness Exponent (2-4)",
            "hyphen_penalty": "Hyphen Penalty (100-2000)",
            "hyphenation_min_word_length": "Hyphenation Min Word Length (4-10)",
        }.get(field, field)
        int_fields = {"min_font_size", "max_font_size", "supersampling_factor", "hyphenation_min_word_length"}
        hint = "a whole number, e.g. `14`" if field in int_fields else "a decimal, e.g. `1.2`"
        awaiting_reply[user_id] = {"type": "appear_field", "extra": {"field": field}}
        await safe_edit(
            query.message,
            f"✍️ **Reply to this message with the new {pretty}** ({hint}).\n"
            f"Reply with `default` to reset to Original/Default."
        )
        return

    if data == "appear_hinting_open":
        await safe_edit(query.message, "🔎 **Font Hinting Mode:**", reply_markup=kb_font_hinting_select(cfg))
        return

    if data.startswith("hintingset_"):
        choice = data.split("_", 1)[1]
        cfg["font_hinting"] = None if choice == "default" else choice
        await save_user_config(user_id)
        await safe_answer(query, "Font Hinting updated")
        await safe_edit(query.message, "🔎 **Font Hinting Mode:**", reply_markup=kb_font_hinting_select(cfg))
        return

    if data.startswith("appear_bool_"):
        field = data.split("appear_bool_", 1)[1]
        label, _ = APPEAR_BOOL_LABELS.get(field, (field, "menu_appearance"))
        await safe_edit(query.message, f"⚙️ **{label}:**", reply_markup=kb_appear_bool_select(cfg, field))
        return

    if data.startswith("appearboolset_"):
        rest = data[len("appearboolset_"):]
        field, choice = rest.rsplit("_", 1)
        cfg[field] = None if choice == "default" else (choice == "on")
        await save_user_config(user_id)
        await safe_answer(query, "Updated")
        label, _ = APPEAR_BOOL_LABELS.get(field, (field, "menu_appearance"))
        await safe_edit(query.message, f"⚙️ **{label}:**", reply_markup=kb_appear_bool_select(cfg, field))
        return

    if data == "appear_reset_all":
        for field in (
            "min_font_size", "max_font_size", "auto_vertical_text", "line_spacing_mult",
            "subpixel_rendering", "font_hinting", "use_ligatures", "hyphenate_before_scaling",
            "hyphen_penalty", "hyphenation_min_word_length", "badness_exponent",
            "padding_pixels", "supersampling_factor", "detach_trailing_punctuation",
        ):
            cfg[field] = None
        await save_user_config(user_id)
        await safe_answer(query, "Appearance reset to Original/Default")
        await safe_edit(
            query.message,
            "🎨 **Appearance Settings**\nAll settings reset to Original/Default.",
            reply_markup=kb_appearance_menu(cfg)
        )
        return

    if data == "menu_api":
        await safe_edit(query.message, 
            "⚙️ **Provider & API Configuration**",
            reply_markup=kb_api_menu(cfg)
        )
        return

    if data == "api_provider_open":
        await safe_edit(query.message, "⚙️ **Select Provider:**", reply_markup=kb_provider_select(cfg))
        return

    if data.startswith("provset_"):
        provider = data.split("_", 1)[1]
        cfg["provider"] = provider
        await save_user_config(user_id)
        await safe_answer(query, f"Provider set to {provider}")
        await safe_edit(query.message, "⚙️ **Provider & API Configuration**", reply_markup=kb_api_menu(cfg))
        return

    if data.startswith("api_field_"):
        field = data.split("api_field_", 1)[1] 
        pretty = {"api_url": "Base URL", "api_key": "API Key", "model_name": "Model ID"}.get(field, field)
        awaiting_reply[user_id] = {"type": "api_field", "extra": {"field": field}}
        await safe_edit(query.message, f"✍️ **Reply to this message with the new {pretty}.**")
        return

    if data == "menu_prompt":
        await safe_edit(query.message, "📝 **Prompt Library**\nSystem prompt = model behaviour. User prompt = your custom focus.", reply_markup=kb_prompt_root())
        return

    if data == "prompt_open_system" or data == "prompt_open_user":
        kind = "system" if data.endswith("system") else "user"
        selected = cfg["system_prompt_name"] if kind == "system" else cfg["user_prompt_name"]
        await safe_edit(query.message, f"📝 **{kind.title()} Prompts**\nSelected: `{selected}`", reply_markup=kb_prompt_list(kind, cfg))
        return

    if data.startswith("promptsel_"):
        _, kind, name = data.split("_", 2)
        lib = load_prompt_library(kind)
        if name in lib:
            if kind == "system":
                cfg["system_prompt_name"] = name
                cfg["system_prompt_text"] = lib[name]
            else:
                cfg["user_prompt_name"] = name
                cfg["user_prompt_text"] = lib[name]
            await save_user_config(user_id)
            await safe_answer(query, f"Selected: {name}")
        await safe_edit(query.message, f"📝 **{kind.title()} Prompts**\nSelected: `{name}`", reply_markup=kb_prompt_list(kind, cfg))
        return

    if data.startswith("promptdel_"):
        _, kind, name = data.split("_", 2)
        ok = delete_prompt(kind, name)
        if ok:
            lib = load_prompt_library(kind)
            fallback_name = next(iter(lib))
            if kind == "system" and cfg["system_prompt_name"] == name:
                cfg["system_prompt_name"] = fallback_name
                cfg["system_prompt_text"] = lib[fallback_name]
                await save_user_config(user_id)
            elif kind == "user" and cfg["user_prompt_name"] == name:
                cfg["user_prompt_name"] = fallback_name
                cfg["user_prompt_text"] = lib[fallback_name]
                await save_user_config(user_id)
            await safe_answer(query, f"Deleted {name}")
        else:
            await safe_answer(query, "Can't delete the last remaining prompt.", show_alert=True)
        await safe_edit(query.message, f"📝 **{kind.title()} Prompts**", reply_markup=kb_prompt_list(kind, cfg))
        return

    if data.startswith("prompt_add_"):
        kind = data.split("prompt_add_", 1)[1]
        awaiting_reply[user_id] = {"type": "prompt_name", "extra": {"kind": kind}}
        await safe_edit(query.message, f"✍️ **Reply to this message with a name for the new {kind} prompt.**")
        return

    if data.startswith("prompt_edit_"):
        kind = data.split("prompt_edit_", 1)[1]
        name = cfg["system_prompt_name"] if kind == "system" else cfg["user_prompt_name"]
        current_text = cfg["system_prompt_text"] if kind == "system" else cfg["user_prompt_text"]
        awaiting_reply[user_id] = {"type": "prompt_body", "extra": {"kind": kind, "name": name, "parts": [], "editing": True}}
        preview = current_text if len(current_text) <= 500 else current_text[:500] + "…"
        await safe_edit(
            query.message,
            f"✏️ **Editing `{name}` ({kind} prompt).**\n\n"
            f"Current text:\n```\n{preview}\n```\n\n"
            f"✍️ **Reply to this message with the new full text.** This will overwrite `{name}` in place.\n"
            f"_If it's too long, split it across multiple messages (each as a reply), then reply `/donedone`._"
        )
        return

    if data == "menu_output":
        await safe_edit(query.message, f"📦 **Output Format**\nCurrent: `{cfg['output_format']}`", reply_markup=kb_output_menu(cfg))
        return

    if data.startswith("outset_"):
        code = data.split("_", 1)[1]
        cfg["output_format"] = code
        await save_user_config(user_id)
        await safe_answer(query, f"Output set to .{code}")
        await safe_edit(query.message, f"📦 **Output Format**\nCurrent: `{cfg['output_format']}`", reply_markup=kb_output_menu(cfg))
        return

    if data == "osb_toggle":
        cfg["osb_enabled"] = not cfg.get("osb_enabled", True)
        await save_user_config(user_id)
        state = "ON" if cfg["osb_enabled"] else "OFF"
        await safe_answer(query, f"OSB text detection turned {state}")
        await safe_edit(query.message, f"📦 **Output Format**\nCurrent: `{cfg['output_format']}`", reply_markup=kb_output_menu(cfg))
        return

    if data == "menu_detection":
        await safe_edit(
            query.message,
            "🎯 **Detection Settings (YOLO/OCR)**\n"
            "Controls how speech bubbles and panels get detected before "
            "translation, and which OCR path reads the text.\n\n"
            "• **Bubble Confidence**: threshold for the main bubble detector.\n"
            "• **Conjoined Confidence**: threshold for detecting bubbles fused "
            "together (secondary YOLO pass).\n"
            "• **Panel Confidence**: threshold for panel-boundary detection.\n"
            "• **Conjoined Bubble Detection**: on/off for that secondary pass.\n"
            "• **Segmentation Model**: yolo / sam2 / sam3.\n"
            "• **Bubble Detector Model**: which primary detector weights to use.\n"
            "• **OCR Method**: vision LLM vs local manga-ocr/paddleocr-vl/"
            "paddleocr-classic/paddleocr-classic-v5 (local options need "
            "`two-step` translation mode). PaddleOCR (Classic, v4 or v5) "
            "supports Japanese, Korean, Chinese, and English; v5 is "
            "newer and more accurate.\n\n"
            "_\"Original/Default\" = untouched, exactly like before this menu existed._",
            reply_markup=kb_detection_menu(cfg)
        )
        return

    if data.startswith("detect_field_"):
        field = data.split("detect_field_", 1)[1]
        pretty = {
            "confidence": "Bubble Confidence (0.0-1.0)",
            "conjoined_confidence": "Conjoined Confidence (0.0-1.0)",
            "panel_confidence": "Panel Confidence (0.0-1.0)",
        }.get(field, field)
        awaiting_reply[user_id] = {"type": "detect_field", "extra": {"field": field}}
        await safe_edit(
            query.message,
            f"✍️ **Reply to this message with the new {pretty}** (a decimal, e.g. `0.6`).\n"
            f"Reply with `default` to reset to Original/Default."
        )
        return

    if data == "detect_bool_conjoined_detection":
        await safe_edit(query.message, "🧩 **Conjoined Bubble Detection:**", reply_markup=kb_detect_bool_select(cfg, "conjoined_detection"))
        return

    if data.startswith("detectboolset_"):
        rest = data[len("detectboolset_"):]
        field, choice = rest.rsplit("_", 1)
        cfg[field] = None if choice == "default" else (choice == "on")
        await save_user_config(user_id)
        await safe_answer(query, "Updated")
        await safe_edit(query.message, "🧩 **Conjoined Bubble Detection:**", reply_markup=kb_detect_bool_select(cfg, field))
        return

    if data == "detect_bool_manual_translation_mode":
        await safe_edit(
            query.message,
            "✍️ **Manual Translation Mode**\n\n"
            "When **On**, after OCR the bot sends you a JSON file with every "
            "detected bubble's original text. Fill in your own translations "
            "and reply to that message with the edited file — the bot will "
            "then render your translations onto the pages instead of calling "
            "the AI translation step. Useful for weaker/free LLMs, or when "
            "you just want full manual control over the wording.\n\n"
            "When **Off** (default), translation runs automatically as usual.",
            reply_markup=kb_manual_mode_select(cfg)
        )
        return

    if data.startswith("manualmodeset_"):
        choice = data.split("_", 1)[1]
        cfg["manual_translation_mode"] = (choice == "on")
        await save_user_config(user_id)
        await safe_answer(query, "Manual Translation Mode " + ("enabled" if choice == "on" else "disabled"))
        await safe_edit(query.message, "✍️ **Manual Translation Mode:**", reply_markup=kb_manual_mode_select(cfg))
        return

    if data == "detect_seg_open":
        await safe_edit(query.message, "🧠 **Segmentation Model:**", reply_markup=kb_seg_model_select(cfg))
        return

    if data.startswith("segset_"):
        choice = data.split("_", 1)[1]
        cfg["seg_model"] = None if choice == "default" else choice
        await save_user_config(user_id)
        await safe_answer(query, "Segmentation model updated")
        await safe_edit(query.message, "🧠 **Segmentation Model:**", reply_markup=kb_seg_model_select(cfg))
        return

    if data == "detect_bubblemodel_open":
        await safe_edit(query.message, "🫧 **Bubble Detector Model:**", reply_markup=kb_bubble_model_select(cfg))
        return

    if data.startswith("bubblemodelset_"):
        choice = data.split("_", 1)[1]
        cfg["bubble_detector_model"] = None if choice == "default" else choice
        await save_user_config(user_id)
        await safe_answer(query, "Bubble detector model updated")
        await safe_edit(query.message, "🫧 **Bubble Detector Model:**", reply_markup=kb_bubble_model_select(cfg))
        return

    if data == "detect_ocr_open":
        await safe_edit(query.message, "👁 **OCR Method:**", reply_markup=kb_ocr_method_select(cfg))
        return

    if data.startswith("ocrmethodset_"):
        choice = data.split("_", 1)[1]
        cfg["ocr_method"] = None if choice == "default" else choice
        await save_user_config(user_id)
        await safe_answer(query, "OCR method updated")
        await safe_edit(query.message, "👁 **OCR Method:**", reply_markup=kb_ocr_method_select(cfg))
        return

    if data == "detect_transmode_open":
        await safe_edit(query.message, "🔀 **Translation Mode:**", reply_markup=kb_translation_mode_select(cfg))
        return

    if data.startswith("transmodeset_"):
        choice = data.split("_", 1)[1]
        cfg["translation_mode"] = None if choice == "default" else choice
        await save_user_config(user_id)
        await safe_answer(query, "Translation mode updated")
        await safe_edit(query.message, "🔀 **Translation Mode:**", reply_markup=kb_translation_mode_select(cfg))
        return

    if data == "detect_reset_all":
        for field in (
            "confidence", "conjoined_confidence", "panel_confidence", "seg_model",
            "conjoined_detection", "bubble_detector_model", "ocr_method", "translation_mode",
        ):
            cfg[field] = None
        cfg["manual_translation_mode"] = False
        await save_user_config(user_id)
        await safe_answer(query, "Detection settings reset to Original/Default")
        await safe_edit(
            query.message,
            "🎯 **Detection Settings (YOLO/OCR)**\nAll settings reset to Original/Default.",
            reply_markup=kb_detection_menu(cfg)
        )
        return
    # ---- Extended settings: group menu open ----
    if data.startswith("xf_group_"):
        group = data[len("xf_group_"):]
        title = FIELD_GROUP_TITLES.get(group, group)
        await safe_edit(query.message, f"{title}\nTap a setting to change it.", reply_markup=kb_field_group_menu(cfg, group))
        return

    # ---- Extended settings: open a specific field editor ----
    if data.startswith("xf_open_"):
        rest = data[len("xf_open_"):]
        group, key = rest.split("_", 1)
        handled = await open_field_editor(query, cfg, group, key)
        if not handled:
            meta = FIELD_REGISTRY[key]
            awaiting_reply[user_id] = {"type": "xf_field", "extra": {"key": key}}
            hint = meta["hint"] or ("a whole number" if meta["argtype"] == "int" else "a decimal number" if meta["argtype"] == "float" else "text")
            await safe_edit(
                query.message,
                f"✍️ **Reply to this message with the new {meta['label']}** ({hint}).\n"
                f"Reply with `default` to reset to Original/Default."
            )
        return

    # ---- Extended settings: bool set ----
    if data.startswith("xfboolset_"):
        rest = data[len("xfboolset_"):]
        key, choice = rest.rsplit("_", 1)
        cfg[key] = None if choice == "default" else (choice == "on")
        await save_user_config(user_id)
        await safe_answer(query, "Updated")
        await safe_edit(query.message, f"{FIELD_REGISTRY[key]['label']}:", reply_markup=kb_field_bool_select(cfg, key))
        return

    # ---- Extended settings: choice set ----
    if data.startswith("xfchoiceset_"):
        rest = data[len("xfchoiceset_"):]
        key, choice = rest.split("::", 1)
        cfg[key] = None if choice == "__default__" else choice
        await save_user_config(user_id)
        await safe_answer(query, "Updated")
        await safe_edit(query.message, f"{FIELD_REGISTRY[key]['label']}:", reply_markup=kb_field_choice_select(cfg, key))
        return

    # ---- Extended settings: reset whole group ----
    if data.startswith("xf_reset_"):
        group = data[len("xf_reset_"):]
        for key in FIELD_GROUPS[group]:
            cfg[key] = None
        await save_user_config(user_id)
        await safe_answer(query, "Group reset to Original/Default")
        title = FIELD_GROUP_TITLES.get(group, group)
        await safe_edit(query.message, f"{title}\nAll settings in this group reset to Original/Default.", reply_markup=kb_field_group_menu(cfg, group))
        return


    if data == "menu_tiling":
        await safe_edit(
            query.message,
            "✂️ **Tiling Settings (Manhwa)**\n"
            "Controls how tall, stitched long-strip pages get sliced into "
            "detector-sized windows before bubble detection, so art doesn't "
            "get crushed down and bubbles don't vanish.\n\n"
            "• **Trigger Height**: pages taller than this get tiled at all.\n"
            "• **Tile Height**: target height per tile.\n"
            "• **Min / Max Cuts**: bounds on how many tiles a page can be split "
            "into. We never go below Min or above Max — but a bubble is NEVER "
            "cut just to hit these numbers. If a page fits safely in fewer "
            "tiles than Min, that's fine.\n"
            "• **Safe-Cut Search Radius**: how far to search for a blank row "
            "near the target cut line, so a cut never lands mid-bubble. "
            "**Auto** keeps widening the search until it finds one instead "
            "of giving up.\n"
            "• **Flatness Threshold**: how blank a row must be to count as "
            "a safe cut.\n"
            "• **Seam Check Band / Diff Threshold**: how the final duplicate-"
            "text check compares pixels just above/below a seam.\n\n"
            "_\"Original/Default\" = untouched, exactly like before this menu existed._",
            reply_markup=kb_tiling_menu(cfg)
        )
        return

    if data.startswith("tile_field_"):
        field = data.split("tile_field_", 1)[1]
        pretty = {
            "tile_height": "Tile Height (px)",
            "tile_search_radius": "Safe-Cut Search Radius (px)",
            "tile_trigger_height": "Trigger Height (px)",
            "tile_flat_threshold": "Flatness Threshold",
            "tile_seam_band_px": "Seam Check Band (px)",
            "tile_seam_diff_threshold": "Seam Duplicate Diff Threshold",
            "tile_min_cuts": "Min Cuts",
            "tile_max_cuts": "Max Cuts",
        }.get(field, field)
        int_fields = {
            "tile_height", "tile_search_radius", "tile_trigger_height", "tile_seam_band_px",
            "tile_min_cuts", "tile_max_cuts",
        }
        hint = "a whole number, e.g. `1600`" if field in int_fields else "a decimal, e.g. `3.5`"
        if field in ("tile_min_cuts", "tile_max_cuts"):
            hint = "a whole number of tiles, e.g. `2`"
        awaiting_reply[user_id] = {"type": "tile_field", "extra": {"field": field}}
        await safe_edit(
            query.message,
            f"✍️ **Reply to this message with the new {pretty}** ({hint}).\n"
            f"Reply with `default` to reset to Original/Default."
        )
        return

    if data == "tile_bool_tile_enabled":
        await safe_edit(query.message, "✂️ **Tiling Enabled:**", reply_markup=kb_tile_bool_select(cfg, "tile_enabled"))
        return

    if data == "tile_search_radius_open":
        await safe_edit(
            query.message,
            "🔍 **Safe-Cut Search Radius**\n\n"
            "• **Auto** (recommended): if no safe row is found nearby, the search "
            "keeps widening step-by-step until one is — a bubble is never cut just "
            "because a fixed radius ran out.\n"
            "• **Manual**: fixed radius in px, same as before.",
            reply_markup=kb_tile_search_radius_menu(cfg)
        )
        return

    if data == "tileradiusauto_on":
        cfg["tile_search_radius_auto"] = True
        await save_user_config(user_id)
        await safe_answer(query, "Search radius set to Auto")
        await safe_edit(query.message, "🔍 **Safe-Cut Search Radius**", reply_markup=kb_tile_search_radius_menu(cfg))
        return

    if data == "tileradiusauto_off":
        cfg["tile_search_radius_auto"] = False
        await save_user_config(user_id)
        await safe_answer(query, "Search radius set to Manual")
        await safe_edit(query.message, "🔍 **Safe-Cut Search Radius**", reply_markup=kb_tile_search_radius_menu(cfg))
        return

    if data.startswith("tileboolset_"):
        rest = data[len("tileboolset_"):]
        field, choice = rest.rsplit("_", 1)
        cfg[field] = None if choice == "default" else (choice == "on")
        await save_user_config(user_id)
        await safe_answer(query, "Updated")
        await safe_edit(query.message, "✂️ **Tiling Enabled:**", reply_markup=kb_tile_bool_select(cfg, field))
        return

    if data == "tile_reset_all":
        for field in (
            "tile_enabled", "tile_height", "tile_search_radius", "tile_trigger_height",
            "tile_flat_threshold", "tile_seam_band_px", "tile_seam_diff_threshold",
            "tile_min_cuts", "tile_max_cuts", "tile_search_radius_auto",
        ):
            cfg[field] = None
        await save_user_config(user_id)
        await safe_answer(query, "Tiling settings reset to Original/Default")
        await safe_edit(
            query.message,
            "✂️ **Tiling Settings (Manhwa)**\nAll settings reset to Original/Default.",
            reply_markup=kb_tiling_menu(cfg)
        )
        return

    if data == "menu_backup":
        await safe_edit(
            query.message,
            "💾 **Backup Settings**\n"
            "Export your full settings (language, font, appearance, tiling, "
            "API, prompts, output format) as a JSON file you can save, or "
            "import a previously exported JSON file to restore/copy a config.\n\n"
            "**Note:** By exporting this JSON, you can also manually edit all 70+ hidden parameters from `main.py`!",
            reply_markup=kb_backup_menu()
        )
        return

    if data == "backup_export":
        await safe_answer(query, "Preparing export...")
        export_path = BASE_DIR / "workspace" / f"settings_export_{user_id}.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(cfg, indent=2))
        await client.send_document(
            query.message.chat.id,
            document=str(export_path),
            caption="📤 Your exported settings. Use **Import Settings** to restore this later or modify advanced flags."
        )
        try:
            export_path.unlink()
        except Exception:
            pass
        return

    if data == "backup_import":
        awaiting_reply[user_id] = {"type": "settings_import"}
        await safe_edit(
            query.message,
            "📥 **Send the exported settings `.json` file now** (as a document).\n"
            "It will replace your current settings entirely."
        )
        return

    if data.startswith("uploadmode_"):
        mode = data.split("_", 1)[1]
        pending_files[user_id] = {"mode": mode, "files": [], "collecting": True}
        mode_label = dict(UPLOAD_MODES).get(f"uploadmode_{mode}", mode)
        hint = {
            "raw": "Now send all your images. Once done, send `/end`.",
            "archive": "Now send your ZIP or CBZ file(s). You can send multiple. Once done, send `/end`.",
            "pdf": "Now send your PDF file(s). You can send multiple. Once done, send `/end`.",
        }.get(mode, "Send your files, then send /end.")
        await safe_edit(query.message, f"📥 **Upload mode:** `{mode}`\n{hint}")
        return

    if data == "start_pipeline":
        await safe_edit(query.message, "🔄 Initializing translation pipeline...")
        active_jobs[user_id] = {"cancel": False, "status_msg": query.message}
        run_job(execute_manga_pipeline(client, query.message, user_id), query.message, user_id)
        return

    if data == "job_cancel":
        job = active_jobs.get(user_id)
        if job:
            job["cancel"] = True
            await safe_answer(query, "Cancelling...")
        return

    if data == "job_continue":
        job_state = paused_jobs.get(user_id)
        if not job_state:
            await safe_answer(query, "No paused job found.", show_alert=True)
            return
        await safe_edit(query.message, "▶️ Resuming translation...")
        active_jobs[user_id] = {"cancel": False, "status_msg": query.message}
        run_job(resume_manga_pipeline(client, query.message, user_id), query.message, user_id)
        return

    if data == "job_send_partial":
        job_state = paused_jobs.get(user_id)
        if not job_state:
            await safe_answer(query, "No paused job found.", show_alert=True)
            return
        await send_partial_results(client, query.message, user_id)
        return

# ================= File Ingestion During /translate Collection =================
@app.on_message((filters.document | filters.photo) & auth_filter)
async def receive_files(client, message: Message):
    user_id = message.from_user.id
    pending_reply = awaiting_reply.get(user_id)

    if pending_reply and pending_reply["type"] == "font_upload" and message.document:
        doc_name = message.document.file_name or ""
        if not doc_name.lower().endswith((".ttf", ".otf")):
            await message.reply_text("❌ Only .ttf or .otf files are allowed.")
            return
        dest = FONTS_DIR / doc_name
        await message.download(file_name=str(dest))
        git_commit_data(f"Add font {doc_name}")
        awaiting_reply.pop(user_id, None)
        cfg = get_user_config(user_id)
        cfg["font_name"] = doc_name
        await save_user_config(user_id)
        await message.reply_text(f"✅ Font `{doc_name}` added to library and selected.", reply_markup=kb_font_menu(cfg))
        return

    if pending_reply and pending_reply["type"] == "manual_translations_reply" and message.document:
        doc_name = message.document.file_name or ""
        if not doc_name.lower().endswith(".json"):
            await message.reply_text("❌ Please reply with the edited `.json` file (as a document).")
            return

        job = manual_jobs.get(user_id)
        if not job:
            awaiting_reply.pop(user_id, None)
            await message.reply_text("❌ No pending manual translation job found. Send `/translate` again.")
            return

        if user_id in active_jobs:
            await message.reply_text("⚠️ A job is already running. Wait for it to finish or `/cancel` first.")
            return

        dest_path = os.path.join(job["job_root"], "received_translations.json")
        await message.download(file_name=dest_path)

        awaiting_reply.pop(user_id, None)
        status_msg = await message.reply_text("🔄 Rendering your manual translations...")
        active_jobs[user_id] = {"cancel": False, "status_msg": status_msg}
        run_job(execute_manual_pipeline_pass2(client, status_msg, user_id, dest_path), status_msg, user_id)
        return

    if pending_reply and pending_reply["type"] == "settings_import" and message.document:
        doc_name = message.document.file_name or ""
        if not doc_name.lower().endswith(".json"):
            await message.reply_text("❌ Only `.json` files are allowed.")
            return
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, doc_name)
            await message.download(file_name=dest)
            try:
                with open(dest, "r") as f:
                    imported = json.load(f)
            except json.JSONDecodeError as e:
                await message.reply_text(
                    f"❌ Invalid JSON — not imported.\n"
                    f"{e.msg} at line {e.lineno}, column {e.colno} (char {e.pos}).\n"
                    f"Fix the file and re-upload; nothing was changed."
                )
                return
            except Exception as e:
                await message.reply_text(f"❌ Couldn't read that file: {e}\nNothing was imported.")
                return

        if not isinstance(imported, dict):
            await message.reply_text("❌ That file doesn't look like an exported settings JSON (expected an object).")
            return

        awaiting_reply.pop(user_id, None)
        merged = default_config()
        unknown_keys = [k for k in imported.keys() if k not in merged]
        known_imported = {k: v for k, v in imported.items() if k in merged}
        merged.update(known_imported)
        cleared = sanitize_cfg_values(merged)
        user_settings[str(user_id)] = merged
        await save_user_config(user_id)
        cfg = get_user_config(user_id)
        report_lines = ["✅ **Settings imported and applied.**"]
        if cleared:
            report_lines.append("\n⚠️ **Invalid values were reset to default:**")
            for field, bad_value in cleared:
                allowed = ", ".join(sorted(VALID_CHOICES[field]))
                report_lines.append(f"• `{field}` = `{bad_value}` → not a valid choice (allowed: {allowed})")
        if unknown_keys:
            shown = ", ".join(unknown_keys[:10])
            more = " ..." if len(unknown_keys) > 10 else ""
            report_lines.append(f"\nℹ️ Ignored {len(unknown_keys)} unrecognized key(s): `{shown}`{more}")
        report_lines.append(
            "\nNote: fonts and prompt library text referenced by name still need "
            "to exist in this bot's library — re-upload/re-add them if missing."
        )
        await message.reply_text("\n".join(report_lines), reply_markup=kb_main_menu(cfg))
        return

    queue = pending_files.get(user_id)
    if not queue or not queue.get("collecting"):
        await message.reply_text("ℹ️ Send `/translate` first and select an upload type.")
        return

    mode = queue["mode"]
    if mode == "raw" and not message.photo and not (message.document and (message.document.mime_type or "").startswith("image/")):
        await message.reply_text("❌ Only images are allowed in this mode.")
        return
    if mode == "archive" and not (message.document and message.document.file_name and message.document.file_name.lower().endswith((".zip", ".cbz"))):
        await message.reply_text("❌ Only ZIP/CBZ files are allowed in this mode.")
        return
    if mode == "pdf" and not (message.document and message.document.file_name and message.document.file_name.lower().endswith(".pdf")):
        await message.reply_text("❌ Only PDF files are allowed in this mode.")
        return

    queue["files"].append(message)
    await message.reply_text(f"✅ Queued ({len(queue['files'])} total). Send more, or send `/end`.")

# ================= Generic Reply Capture (settings inputs) =================
@app.on_message(filters.text & filters.reply & auth_filter)
async def handle_reply_capture(client, message: Message):
    user_id = message.from_user.id
    pending_reply = awaiting_reply.get(user_id)
    if not pending_reply:
        return

    cfg = get_user_config(user_id)
    kind = pending_reply["type"]
    text = message.text.strip()

    if kind == "custom_lang":
        cfg["target_lang"] = text
        cfg["target_lang_label"] = "Custom"
        await save_user_config(user_id)
        awaiting_reply.pop(user_id, None)
        await message.reply_text(f"✅ Target language set to `{text}`.", reply_markup=kb_lang_root(cfg))
        return

    if kind == "api_field":
        field = pending_reply["extra"]["field"]
        cfg[field] = text
        await save_user_config(user_id)
        awaiting_reply.pop(user_id, None)
        pretty = {"api_url": "Base URL", "api_key": "API Key", "model_name": "Model ID"}.get(field, field)
        await message.reply_text(f"✅ {pretty} updated.", reply_markup=kb_api_menu(cfg))
        return

    if kind == "appear_field":
        field = pending_reply["extra"]["field"]
        pretty = {
            "min_font_size": "Min Font Size (px)",
            "max_font_size": "Max Font Size (px)",
            "line_spacing_mult": "Line Spacing Multiplier",
            "padding_pixels": "Bubble Padding (px)",
            "supersampling_factor": "Supersampling Factor",
            "badness_exponent": "Line Badness Exponent",
            "hyphen_penalty": "Hyphen Penalty",
            "hyphenation_min_word_length": "Hyphenation Min Word Length",
        }.get(field, field)
        font_fields = {"min_font_size", "max_font_size", "line_spacing_mult"}
        back_kb = kb_appear_font_menu(cfg) if field in font_fields else kb_appear_layout_menu(cfg)

        if text.lower() == "default":
            cfg[field] = None
            await save_user_config(user_id)
            awaiting_reply.pop(user_id, None)
            await message.reply_text(f"✅ {pretty} reset to Original/Default.", reply_markup=back_kb)
            return

        int_fields = {"min_font_size", "max_font_size", "supersampling_factor", "hyphenation_min_word_length"}
        is_int_field = field in int_fields
        try:
            value = int(text) if is_int_field else float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            hint = "a positive whole number (e.g. `14`)" if is_int_field else "a positive decimal (e.g. `1.2`)"
            await message.reply_text(f"❌ Please reply with {hint}, or `default` to reset. Try again.")
            return

        range_checks = {
            "supersampling_factor": (1, 4),
            "badness_exponent": (2, 4),
            "hyphen_penalty": (100, 2000),
            "hyphenation_min_word_length": (4, 10),
            "padding_pixels": (2, 12),
        }
        if field in range_checks:
            lo, hi = range_checks[field]
            if not (lo <= value <= hi):
                await message.reply_text(f"❌ {pretty} should be between {lo} and {hi}. Try again.")
                return

        if field in ("min_font_size", "max_font_size"):
            other_field = "max_font_size" if field == "min_font_size" else "min_font_size"
            other_value = cfg.get(other_field)
            if other_value is not None:
                if field == "min_font_size" and value > other_value:
                    await message.reply_text(f"❌ Min Font Size ({value}) can't be greater than Max Font Size ({other_value}). Try again.")
                    return
                if field == "max_font_size" and value < other_value:
                    await message.reply_text(f"❌ Max Font Size ({value}) can't be less than Min Font Size ({other_value}). Try again.")
                    return

        cfg[field] = value
        await save_user_config(user_id)
        awaiting_reply.pop(user_id, None)
        await message.reply_text(f"✅ {pretty} set to `{value}`.", reply_markup=back_kb)
        return
    if kind == "xf_field":
        key = pending_reply["extra"]["key"]
        meta = FIELD_REGISTRY[key]
        group = meta["group"]
        back_kb = kb_field_group_menu(cfg, group)

        if text.lower() == "default":
            cfg[key] = None
            await save_user_config(user_id)
            awaiting_reply.pop(user_id, None)
            await message.reply_text(f"✅ {meta['label']} reset to Original/Default.", reply_markup=back_kb)
            return

        argtype = meta["argtype"]
        try:
            if argtype == "int":
                value = int(text)
            elif argtype == "float":
                value = float(text)
            else:
                value = text
        except ValueError:
            hint = meta["hint"] or ("a whole number" if argtype == "int" else "a decimal number")
            await message.reply_text(f"❌ Please reply with {hint}, or `default` to reset. Try again.")
            return

        cfg[key] = value
        await save_user_config(user_id)
        awaiting_reply.pop(user_id, None)
        await message.reply_text(f"✅ {meta['label']} set to `{value}`.", reply_markup=back_kb)
        return


    if kind == "detect_field":
        field = pending_reply["extra"]["field"]
        pretty = {
            "confidence": "Bubble Confidence",
            "conjoined_confidence": "Conjoined Confidence",
            "panel_confidence": "Panel Confidence",
        }.get(field, field)

        if text.lower() == "default":
            cfg[field] = None
            await save_user_config(user_id)
            awaiting_reply.pop(user_id, None)
            await message.reply_text(f"✅ {pretty} reset to Original/Default.", reply_markup=kb_detection_menu(cfg))
            return

        try:
            value = float(text)
            if not (0.0 <= value <= 1.0):
                raise ValueError
        except ValueError:
            await message.reply_text(f"❌ Please reply with a decimal between 0.0 and 1.0 (e.g. `0.6`), or `default` to reset. Try again.")
            return

        cfg[field] = value
        await save_user_config(user_id)
        awaiting_reply.pop(user_id, None)
        await message.reply_text(f"✅ {pretty} set to `{value}`.", reply_markup=kb_detection_menu(cfg))
        return

    if kind == "tile_field":
        field = pending_reply["extra"]["field"]
        pretty = {
            "tile_height": "Tile Height (px)",
            "tile_search_radius": "Safe-Cut Search Radius (px)",
            "tile_trigger_height": "Trigger Height (px)",
            "tile_flat_threshold": "Flatness Threshold",
            "tile_seam_band_px": "Seam Check Band (px)",
            "tile_seam_diff_threshold": "Seam Duplicate Diff Threshold",
            "tile_min_cuts": "Min Cuts",
            "tile_max_cuts": "Max Cuts",
        }.get(field, field)

        if text.lower() == "default":
            cfg[field] = None
            await save_user_config(user_id)
            awaiting_reply.pop(user_id, None)
            await message.reply_text(f"✅ {pretty} reset to Original/Default.", reply_markup=kb_tiling_menu(cfg))
            return

        int_fields = {
            "tile_height", "tile_search_radius", "tile_trigger_height", "tile_seam_band_px",
            "tile_min_cuts", "tile_max_cuts",
        }
        is_int_field = field in int_fields
        try:
            value = int(text) if is_int_field else float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            hint = "a positive whole number (e.g. `1600`)" if is_int_field else "a positive decimal (e.g. `3.5`)"
            await message.reply_text(f"❌ Please reply with {hint}, or `default` to reset. Try again.")
            return

        if field in ("tile_height", "tile_trigger_height"):
            other_field = "tile_trigger_height" if field == "tile_height" else "tile_height"
            other_value = cfg.get(other_field)
            if other_value is not None:
                if field == "tile_height" and value > other_value:
                    await message.reply_text(f"❌ Tile Height ({value}) shouldn't exceed Trigger Height ({other_value}). Try again.")
                    return
                if field == "tile_trigger_height" and value < other_value:
                    await message.reply_text(f"❌ Trigger Height ({value}) shouldn't be less than Tile Height ({other_value}). Try again.")
                    return

        # Min Cuts must never exceed Max Cuts, and vice versa.
        if field in ("tile_min_cuts", "tile_max_cuts"):
            other_field = "tile_max_cuts" if field == "tile_min_cuts" else "tile_min_cuts"
            other_value = cfg.get(other_field)
            if other_value is None:
                other_value = MANHWA_TILE_MAX_CUTS if other_field == "tile_max_cuts" else MANHWA_TILE_MIN_CUTS
            if field == "tile_min_cuts" and value > other_value:
                await message.reply_text(f"❌ Min Cuts ({value}) can't be more than Max Cuts ({other_value}). Try again.")
                return
            if field == "tile_max_cuts" and value < other_value:
                await message.reply_text(f"❌ Max Cuts ({value}) can't be less than Min Cuts ({other_value}). Try again.")
                return

        cfg[field] = value
        await save_user_config(user_id)
        awaiting_reply.pop(user_id, None)
        await message.reply_text(f"✅ {pretty} set to `{value}`.", reply_markup=kb_tiling_menu(cfg))
        return

    if kind == "prompt_name":
        prompt_kind = pending_reply["extra"]["kind"]
        if len(text) > PROMPT_NAME_MAX_LEN:
            await message.reply_text(f"❌ Name can't be more than {PROMPT_NAME_MAX_LEN} characters (got `{len(text)}`). Please reply again.")
            return
        awaiting_reply[user_id] = {"type": "prompt_body", "extra": {"kind": prompt_kind, "name": text, "parts": []}}
        await message.reply_text(
            f"✍️ **Reply to this message with the {prompt_kind} prompt text for** `{text}`.\n\n"
            f"_If the prompt is longer than Telegram's 4096 character limit, split it across multiple messages "
            f"(reply each part to this same message). Once you've sent it all, reply `/donedone`._"
        )
        return

    if kind == "prompt_body":
        prompt_kind = pending_reply["extra"]["kind"]
        name = pending_reply["extra"]["name"]
        parts = pending_reply["extra"].setdefault("parts", [])

        if text.strip() == "/donedone":
            if not parts:
                await message.reply_text("❌ No prompt text received yet. Send the text first, then `/donedone`.")
                return
            full_text = "".join(parts)
            is_editing = pending_reply["extra"].get("editing", False)
            add_prompt(prompt_kind, name, full_text)
            if prompt_kind == "system":
                cfg["system_prompt_name"] = name
                cfg["system_prompt_text"] = full_text
            else:
                cfg["user_prompt_name"] = name
                cfg["user_prompt_text"] = full_text
            await save_user_config(user_id)
            awaiting_reply.pop(user_id, None)
            verb = "updated" if is_editing else "added and selected"
            await message.reply_text(
                f"✅ Prompt `{name}` {verb}. ({len(full_text)} characters, {len(parts)} part(s) combined.)",
                reply_markup=kb_prompt_list(prompt_kind, cfg)
            )
            return

        parts.append(text)
        total_len = sum(len(p) for p in parts)
        await message.reply_text(
            f"➕ Part {len(parts)} received ({len(text)} chars, total so far: {total_len}).\n"
            f"Send more, or reply `/donedone` to save."
        )
        return

# ================= Job State for Pause/Resume =================
paused_jobs = {} 

# ================= Job State for Manual Translation Mode =================
# Keyed by user_id. Holds everything needed to resume into Pass 2 once the
# user replies with their edited translations JSON. Populated at the end of
# execute_manual_pipeline_pass1 and consumed/cleared by
# handle_manual_translations_reply -> execute_manual_pipeline_pass2.
manual_jobs = {}

BASE_STAGING = str(BASE_DIR / "workspace")

def build_status_text(mode_label, stage, current_file_idx, total_files, current_image, total_images_in_file, percent):
    bar_filled = int(percent / 10)
    bar = "▓" * bar_filled + "░" * (10 - bar_filled)
    return (
        f"📊 **Translation Status**\n"
        f"Mode: `{mode_label}`\n"
        f"File: `{current_file_idx}/{total_files}`\n"
        f"Stage: {stage}\n"
        f"Progress: [{bar}] {percent}%\n"
        f"Image: `{current_image}/{total_images_in_file}`"
    )

def extract_archive(path, dest_dir):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

def extract_pdf(path, dest_dir):
    import fitz  
    zoom = 200 / 72  
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(path)
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_path = os.path.join(dest_dir, f"page_{i:03d}.jpg")
            pix.save(out_path)
    finally:
        doc.close()

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def _stitch_group_vertically(input_dir, base_name, slice_files):
    from PIL import Image
    slice_files = sorted(slice_files, key=_natural_key)
    imgs = [Image.open(os.path.join(input_dir, f)).convert("RGB") for f in slice_files]
    total_width = max(im.width for im in imgs)
    total_height = sum(im.height for im in imgs)
    stitched = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    y_offset = 0
    for im in imgs:
        x_offset = (total_width - im.width) // 2
        stitched.paste(im, (x_offset, y_offset))
        y_offset += im.height
    out_name = f"{base_name}__stitched.jpg"
    out_path = os.path.join(input_dir, out_name)
    stitched.save(out_path, quality=95)
    for im in imgs:
        im.close()
    for f in slice_files:
        try:
            os.remove(os.path.join(input_dir, f))
        except Exception:
            pass
    return out_name

def stitch_sliced_images(input_dir):
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS)]
    groups = {}
    singles = []
    for f in all_files:
        stem = os.path.splitext(f)[0]
        if "__" in stem:
            base = stem.split("__", 1)[0]
            groups.setdefault(base, []).append(f)
        else:
            singles.append(f)

    for base, slice_files in groups.items():
        if len(slice_files) > 1:
            _stitch_group_vertically(input_dir, base, slice_files)

def flatten_and_order(input_dir, content_type="manhwa", cfg=None):
    for root, _, files in os.walk(input_dir, topdown=False):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                shutil.move(os.path.join(root, f), os.path.join(input_dir, f))
        if root != input_dir:
            try:
                os.rmdir(root)
            except Exception:
                pass

    stitch_sliced_images(input_dir)

    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS)], key=_natural_key)
    ordered_map = {}
    for idx, fname in enumerate(images, start=1):
        ext = os.path.splitext(fname)[1]
        new_name = f"{idx:03d}{ext}"
        if new_name != fname:
            shutil.move(os.path.join(input_dir, fname), os.path.join(input_dir, new_name))
        ordered_map[idx] = new_name

    tile_manifest = None
    if content_type == "manhwa":
        tile_manifest = tile_tall_pages(input_dir, ordered_map, cfg=cfg)

    return ordered_map, tile_manifest

# ================= Long-Strip Tiling (Manhwa) =================
def _compute_row_flatness(im):
    import numpy as np
    gray = im.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    return arr.std(axis=1)

def _find_safe_cut_row(row_flatness, target_y, search_window, min_y, max_y, flat_threshold=MANHWA_SAFE_CUT_FLAT_THRESHOLD):
    if row_flatness[target_y] < flat_threshold:
        return target_y
    for delta in range(1, search_window + 1):
        up = target_y - delta
        down = target_y + delta
        if down <= max_y and row_flatness[down] < flat_threshold:
            return down
        if up >= min_y and row_flatness[up] < flat_threshold:
            return up
    return None  

def _find_safe_cut_row_expanding(row_flatness, target_y, min_y, max_y, flat_threshold, base_window, auto, max_extend_steps):
    """
    Looks for a safe (flat/non-bubble) row near target_y. If none is found within
    base_window and `auto` is on, keeps widening the search window step by step
    (instead of giving up) so a bubble is never cut just because a fixed radius
    ran out. Returns the row index, or None if truly nothing safe exists in
    [min_y, max_y] even at full width — in which case the caller must NOT cut here.
    """
    window = base_window
    cut = _find_safe_cut_row(row_flatness, target_y, window, min_y, max_y, flat_threshold)
    if cut is not None or not auto:
        return cut
    steps = 0
    while cut is None and steps < max_extend_steps:
        window += base_window
        steps += 1
        cut = _find_safe_cut_row(row_flatness, target_y, window, min_y, max_y, flat_threshold)
    return cut

def _any_safe_row_exists(row_flatness, min_y, max_y, flat_threshold):
    segment = row_flatness[min_y:max_y + 1]
    return bool((segment < flat_threshold).any())

def tile_tall_pages(input_dir, ordered_map, cfg=None):
    from PIL import Image

    cfg = cfg or {}
    if cfg.get("tile_enabled") is False:
        return {} 

    tile_height = cfg.get("tile_height") or MANHWA_TILE_HEIGHT
    tile_search_radius = cfg.get("tile_search_radius") or MANHWA_TILE_OVERLAP
    tile_trigger_height = cfg.get("tile_trigger_height") or MANHWA_TILE_TRIGGER_HEIGHT
    tile_flat_threshold = cfg.get("tile_flat_threshold") or MANHWA_SAFE_CUT_FLAT_THRESHOLD
    radius_auto = cfg.get("tile_search_radius_auto")
    if radius_auto is None:
        radius_auto = True  # Auto is the recommended/default behavior
    min_cuts = cfg.get("tile_min_cuts") or MANHWA_TILE_MIN_CUTS
    max_cuts = cfg.get("tile_max_cuts") or MANHWA_TILE_MAX_CUTS
    # "cuts" here means resulting tiles for the page.
    min_tiles = max(1, min_cuts)
    max_tiles = max(min_tiles, max_cuts)

    manifest = {}
    for idx, fname in list(ordered_map.items()):
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            continue
        with Image.open(path) as im:
            width, height = im.size
            if height <= tile_trigger_height:
                continue  

            im = im.convert("RGB")
            row_flatness = _compute_row_flatness(im)

            tile_files = []
            tile_heights = []
            forced_cut_rows = []  # kept for compatibility; stays empty now since we never force a bubble cut
            y = 0
            tile_n = 0
            max_extend_steps = MANHWA_TILE_SEARCH_RADIUS_AUTO_MAX_EXTEND

            while y < height:
                remaining = height - y
                tiles_so_far = tile_n  # completed tiles before this one
                # Bubble safety always wins: we only ever try to land a cut short of
                # max_tiles. If we've already reached max_tiles - 1 completed tiles,
                # this must be the final tile — ride out to the end of the image.
                at_last_allowed_tile = (tiles_so_far >= max_tiles - 1)

                target_bottom = min(y + tile_height, height)

                if target_bottom >= height or at_last_allowed_tile:
                    cut = height
                else:
                    cut = _find_safe_cut_row_expanding(
                        row_flatness, target_bottom, y + 1, height - 1,
                        tile_flat_threshold, tile_search_radius, radius_auto, max_extend_steps
                    )
                    if cut is None:
                        # No safe row near the target even after (auto-)expanding the
                        # search. Never cut through a bubble to hit Min Cuts — instead
                        # push the target further down and keep looking, as long as
                        # doing so still respects Max Cuts (checked via at_last_allowed_tile
                        # above on the next loop iteration) and a safe row remains ahead.
                        extended_target = target_bottom
                        while cut is None and extended_target < height:
                            extended_target = min(extended_target + tile_search_radius, height)
                            if extended_target >= height:
                                cut = height
                                break
                            if not _any_safe_row_exists(row_flatness, y + 1, height - 1, tile_flat_threshold):
                                # No safe row anywhere left in the remainder of the
                                # image — ride out to the end rather than force a cut.
                                cut = height
                                break
                            cut = _find_safe_cut_row_expanding(
                                row_flatness, extended_target, y + 1, height - 1,
                                tile_flat_threshold, tile_search_radius, radius_auto, max_extend_steps
                            )
                    if cut is None:
                        cut = height

                tile = im.crop((0, y, width, cut))
                tile_name = f"{os.path.splitext(fname)[0]}_tile{tile_n:03d}.jpg"
                tile.save(os.path.join(input_dir, tile_name), quality=95)
                tile_files.append(tile_name)
                tile_heights.append(cut - y)
                tile_n += 1
                if cut >= height:
                    break
                y = cut

            manifest[idx] = {
                "tiles": tile_files,
                "heights": tile_heights,
                "width": width,
                "original_height": height,
                "original_name": fname,
                "forced_cut_rows": forced_cut_rows,  
            }
        os.remove(path)  
    return manifest

SEAM_CHECK_BAND_PX = 40
SEAM_DUPLICATE_DIFF_THRESHOLD = 6.0

def _seam_looks_duplicated(recomposed_im, seam_y, band_px=SEAM_CHECK_BAND_PX, diff_threshold=SEAM_DUPLICATE_DIFF_THRESHOLD):
    import numpy as np
    width, height = recomposed_im.size
    top = max(0, seam_y - band_px)
    bottom = min(height, seam_y + band_px)
    if bottom - top < band_px * 2:
        return False  
    region = recomposed_im.crop((0, top, width, bottom)).convert("L")
    arr = np.asarray(region, dtype=np.float32)
    upper_band = arr[:band_px]
    lower_band = arr[band_px:]
    if upper_band.shape != lower_band.shape:
        return False
    diff = np.abs(upper_band - lower_band).mean()
    return diff < diff_threshold

def recompose_tiled_page(translated_dir, page_idx, manifest_entry, cfg=None):
    from PIL import Image
    cfg = cfg or {}
    seam_band_px = cfg.get("tile_seam_band_px") or SEAM_CHECK_BAND_PX
    seam_diff_threshold = cfg.get("tile_seam_diff_threshold") or SEAM_DUPLICATE_DIFF_THRESHOLD

    tiles = manifest_entry["tiles"]
    heights = manifest_entry["heights"]
    width = manifest_entry["width"]
    total_height = manifest_entry["original_height"]
    forced_cut_rows = set(manifest_entry.get("forced_cut_rows", []))

    # Collect ALL tile paths that exist on disk for this page up front, so that
    # even on a partial-failure path (some tile never got translated) we can
    # still clean up the tiles that DID make it - otherwise they're left behind
    # in translated_dir as loose images and get swept into produced_files,
    # causing the same page to appear twice (once as a leftover tile, once
    # wherever it resurfaces) in the final PDF/CBZ output.
    found_paths = []
    missing = False
    for tile_name in tiles:
        stem = os.path.splitext(tile_name)[0]
        found = None
        for ext in IMAGE_EXTS:
            candidate = os.path.join(translated_dir, stem + ext)
            if os.path.exists(candidate):
                found = candidate
                break
        if found is None:
            missing = True
        else:
            found_paths.append(found)

    if missing:
        # Incomplete tile set for this page - can't safely recompose. Remove
        # whatever tiles DID get translated so they don't leak into the output
        # as stray extra images, then bail.
        for p in found_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    translated_tile_paths = found_paths

    try:
        recomposed = Image.new("RGB", (width, total_height), (255, 255, 255))
        seam_ys = []
        y_cursor = 0
        for i, tile_path in enumerate(translated_tile_paths):
            with Image.open(tile_path) as tile_im:
                tile_im = tile_im.convert("RGB")
                if tile_im.size != (width, heights[i]):
                    tile_im = tile_im.resize((width, heights[i]))
                recomposed.paste(tile_im, (0, y_cursor))
                y_cursor += tile_im.height
                if i < len(translated_tile_paths) - 1:
                    seam_ys.append(y_cursor)

        flagged_seams = []
        for seam_y in seam_ys:
            was_forced = any(abs(seam_y - forced_y) <= 2 for forced_y in forced_cut_rows)
            effective_threshold = seam_diff_threshold * (1.4 if was_forced else 1.0)
            if _seam_looks_duplicated(recomposed, seam_y, band_px=seam_band_px, diff_threshold=effective_threshold):
                flagged_seams.append(seam_y)

        out_name = manifest_entry["original_name"]
        out_path = os.path.join(translated_dir, out_name)
        recomposed.save(out_path, quality=95)
    finally:
        # ALWAYS remove the source tiles, even if recompose raised partway
        # through (corrupt tile image, resize failure, disk error, etc).
        # Leaving them behind means they get picked up as loose "extra"
        # images later and the page effectively appears twice in the
        # final PDF/CBZ/ZIP.
        for p in translated_tile_paths:
            try:
                os.remove(p)
            except Exception:
                pass
    return (out_path, flagged_seams)

def build_dynamic_system_instruction(cfg):
    system_text = cfg["system_prompt_text"].replace("{target_lang}", cfg["target_lang"])
    return (
        f"{system_text}\n\n"
        f"User Specific Instructions Focus: {cfg['user_prompt_text']}\n"
        f"Produce strictly the final targeted localization stream mapping blocks directly."
    )

# ================= MangaTranslator CLI Mappings =================
_cli_help_cache = {"text": None}

def _get_main_help_text():
    if _cli_help_cache["text"] is None:
        try:
            result = subprocess.run(
                ["python", "MangaTranslator/main.py", "--help"],
                capture_output=True, text=True, timeout=30
            )
            _cli_help_cache["text"] = (result.stdout or "") + (result.stderr or "")
        except Exception:
            _cli_help_cache["text"] = ""
    return _cli_help_cache["text"]

def cli_supports_flag(flag):
    return flag in _get_main_help_text()

# Master CLI Flag Mapper
CLI_MAPPINGS = {
    "min_font_size": ("--min-font-size", "val"), "max_font_size": ("--max-font-size", "val"),
    "line_spacing_mult": ("--line-spacing-mult", "val"), "padding_pixels": ("--padding-pixels", "val"),
    "supersampling_factor": ("--supersampling-factor", "val"), "font_hinting": ("--font-hinting", "val"),
    "auto_vertical_text": ("--auto-vertical-text", "bool_true"), "use_ligatures": ("--use-ligatures", "bool_true"),
    "subpixel_rendering": ("--no-subpixel-rendering", "bool_invert"), "hyphenate_before_scaling": ("--no-hyphenate-before-scaling", "bool_invert"),
    "detach_trailing_punctuation": ("--no-detach-trailing-punctuation", "bool_invert"), "hyphen_penalty": ("--hyphen-penalty", "val"),
    "hyphenation_min_word_length": ("--hyphenation-min-word-length", "val"), "badness_exponent": ("--badness-exponent", "val"),
    "temperature": ("--temperature", "val"), "top_p": ("--top-p", "val"), "top_k": ("--top-k", "val"),
    "max_tokens": ("--max-tokens", "val"), "translation_mode": ("--translation-mode", "val"),
    "use_custom_sampling": ("--no-custom-sampling", "bool_invert"), "ocr_method": ("--ocr-method", "val"),
    "reasoning_effort": ("--reasoning-effort", "val"), "effort": ("--effort", "val"), "verbosity": ("--verbosity", "val"),
    "reading_direction": ("--reading-direction", "val"), "enable_web_search": ("--enable-web-search", "bool_true"),
    "enable_code_execution": ("--enable-code-execution", "bool_true"), "media_resolution": ("--media-resolution", "val"),
    "media_resolution_bubbles": ("--media-resolution-bubbles", "val"), "media_resolution_context": ("--media-resolution-context", "val"),
    "image_detail": ("--image-detail", "val"), "send_full_page_context": ("--no-full-page-context", "bool_invert"),
    "confidence": ("--confidence", "val"), "conjoined_confidence": ("--conjoined-confidence", "val"),
    "panel_confidence": ("--panel-confidence", "val"), "seg_model": ("--seg-model", "val"),
    "bubble_detector_model": ("--bubble-detector-model", "val"), "conjoined_detection": ("--no-conjoined-detection", "bool_invert"),
    "inpaint_colored_bubbles": ("--inpaint-colored-bubbles", "bool_true"), "use_otsu_threshold": ("--use-otsu-threshold", "bool_true"),
    "thresholding_value": ("--thresholding-value", "val"), "roi_shrink_px": ("--roi-shrink-px", "val"),
    "whiteout_conjoined_bubbles": ("--no-whiteout-conjoined-bubbles", "bool_invert"), "upscale_method": ("--upscale-method", "val"),
    "image_upscale_mode": ("--image-upscale-mode", "val"), "image_upscale_factor": ("--image-upscale-factor", "val"),
    "jpeg_quality": ("--jpeg-quality", "val"), "png_compression": ("--png-compression", "val"),
    "auto_scale": ("--no-auto-scale", "bool_invert"), "bubble_min_side_pixels": ("--bubble-min-side-pixels", "val"),
    "context_image_max_side_pixels": ("--context-image-max-side-pixels", "val"), "parallel_requests": ("--parallel-requests", "val"),
    "batch_parallel_within_pages": ("--batch-parallel-within-pages", "bool_true"), "batch_previous_context_images": ("--batch-previous-context-images", "val"),
    "batch_previous_context_texts": ("--batch-previous-context-texts", "val"),
    "retry_failed_once": ("--retry-failed-once", "bool_true"), "overlap_llm_with_inpaint": ("--overlap-llm-with-inpaint", "bool_true"),
    "verbose": ("--verbose", "bool_true"),
    "cpu": ("--cpu", "bool_true"), "cleaning_only": ("--cleaning-only", "bool_true"), "upscaling_only": ("--upscaling-only", "bool_true"),
    "test_mode": ("--test-mode", "bool_true"), "osb_inpainting_method": ("--osb-inpainting-method", "val"),
    "osb_flux_backend": ("--osb-flux-backend", "val"), "osb_flux_low_vram": ("--osb-flux-low-vram", "bool_true"),
    "osb_flux_sdcpp_cache_mode": ("--osb-flux-sdcpp-cache-mode", "val"), "osb_flux_sdcpp_diffusion_quant": ("--osb-flux-sdcpp-diffusion-quant", "val"),
    "osb_flux_sdcpp_text_encoder_quant": ("--osb-flux-sdcpp-text-encoder-quant", "val"), "osb_flux_upscale_small_crops": ("--osb-no-flux-upscale-small-crops", "bool_invert"),
    "osb_flux_group_regions": ("--osb-flux-group-regions", "bool_true"), "osb_flux_steps": ("--osb-flux-steps", "val"),
    "osb_sfx_skip_inpaint": ("--osb-no-sfx-skip-inpaint", "bool_invert"),
    "osb_flux_luminance_correction": ("--osb-no-luminance-correction", "bool_invert"), "osb_flux_residual_threshold": ("--osb-flux-residual-threshold", "val"),
    "osb_seed": ("--osb-seed", "val"), "osb_max_font_size": ("--osb-max-font-size", "val"),
    "osb_min_font_size": ("--osb-min-font-size", "val"), "osb_use_ligatures": ("--osb-use-ligatures", "bool_true"),
    "osb_outline_width": ("--osb-outline-width", "val"), "osb_line_spacing": ("--osb-line-spacing", "val"),
    "osb_use_subpixel": ("--osb-use-subpixel", "bool_true"), "osb_font_hinting": ("--osb-font-hinting", "val"),
    "osb_bbox_expansion": ("--osb-bbox-expansion", "val"), "osb_render_expansion_narrow": ("--osb-render-expansion-narrow", "val"),
    "osb_render_expansion_tiny": ("--osb-render-expansion-tiny", "val"), "osb_render_expansion_aspect_threshold": ("--osb-render-expansion-aspect-threshold", "val"),
    "osb_render_expansion_area_threshold": ("--osb-render-expansion-area-threshold", "val"), "osb_text_box_proximity_ratio": ("--osb-text-box-proximity-ratio", "val"),
    "osb_confidence": ("--osb-confidence", "val"), "osb_filter_page_numbers": ("--osb-filter-page-numbers", "bool_true"),
    "osb_page_filter_margin": ("--osb-page-filter-margin", "val"), "osb_page_filter_min_area": ("--osb-page-filter-min-area", "val"),
    "osb_min_area_ignore_ratio": ("--osb-min-area-ignore-ratio", "val"), "osb_min_side_pixels": ("--osb-min-side-pixels", "val")
}

# ================= Shared Engine Runner (live logs + stall timeout) =================
# Local OCR methods (manga-ocr / paddleocr-vl / paddleocr-classic / paddleocr-classic-v5) can silently hang on a CPU-only
# Actions runner - either a slow/rate-limited first-time weight download, or
# just very slow CPU inference on a VLM like PaddleOCR-VL. Previously all
# engine output was captured via PIPE and only printed after the process
# exited (process.communicate()), so a stuck run produced ZERO log output
# until killed - making it impossible to tell "hung" from "slow" from the
# Actions log. This helper streams stdout/stderr live, line by line, and kills
# the process if too long passes without a new translated image appearing.
ENGINE_STALL_TIMEOUT = int(os.environ.get("ENGINE_STALL_TIMEOUT", "900"))  # 15 min default

async def run_engine_subprocess(cmd, env, status_msg, progress_dir=None, total_images=1,
                                 mode_label="Engine", stage_label="🧠 Running", file_idx=1,
                                 total_files=1, cancel_check=None, base_pct=40, pct_span=40):
    """
    Runs `cmd` with live-streamed output and a stall timeout based on new
    files appearing in `progress_dir` (if given). Returns a dict:
      {"returncode": int|None, "timed_out": bool, "tail": str}
    `cancel_check()` (if given) should return True if the user cancelled;
    on cancel this kills the process and returns {"cancelled": True, ...}.
    """
    recent_lines = []
    MAX_RECENT_LINES = 60

    async def _stream_reader(stream, label):
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip("\n")
            print(f"[{label}] {text}", flush=True)
            recent_lines.append(text)
            if len(recent_lines) > MAX_RECENT_LINES:
                del recent_lines[0]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
    )
    stdout_task = asyncio.create_task(_stream_reader(process.stdout, "stdout"))
    stderr_task = asyncio.create_task(_stream_reader(process.stderr, "stderr"))

    loop = asyncio.get_event_loop()
    start_time = loop.time()
    last_progress_count = -1
    last_progress_time = start_time
    timed_out = False
    cancelled = False

    while process.returncode is None:
        if cancel_check and cancel_check():
            process.kill()
            cancelled = True
            break

        now = loop.time()
        if progress_dir and os.path.exists(progress_dir):
            done_count = len([f for f in os.listdir(progress_dir) if f.lower().endswith(IMAGE_EXTS)])
        else:
            done_count = 0

        if done_count != last_progress_count:
            last_progress_count = done_count
            last_progress_time = now

        stalled_for = now - last_progress_time
        if stalled_for >= ENGINE_STALL_TIMEOUT:
            timed_out = True
            process.kill()
            break

        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            pct = base_pct + int((done_count / max(total_images, 1)) * pct_span)
            mins_stalled = int(stalled_for // 60)
            stall_note = f" (no new output image for {mins_stalled}m)" if mins_stalled >= 2 else ""
            await safe_edit(
                status_msg,
                build_status_text(mode_label, f"{stage_label}{stall_note}", file_idx, total_files, done_count, total_images, min(pct, base_pct + pct_span)),
                reply_markup=kb_cancel_only()
            )

    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    tail = "\n".join(recent_lines[-40:])
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "tail": tail,
    }

# ================= Manual Translation Mode: Pass 1 (OCR-only checkpoint) =================
async def execute_manual_pipeline_pass1(client, status_msg: Message, user_id: int):
    """
    Manual Translation Mode, Pass 1. Mirrors execute_manga_pipeline's file
    download/extract/flatten steps, but instead of running the full
    translate+render engine, calls main.py with --manual-ocr-checkpoint so
    each page is only detected/cleaned/OCR'd (no translation API call) and
    checkpointed to disk. Every input file's combined_translations.json is
    then merged into one JSON and sent to the user as a Telegram document;
    the job is parked in `manual_jobs` awaiting their edited reply instead of
    being torn down like a normal completed job.
    """
    cfg = get_user_config(user_id)
    queue = pending_files.get(user_id)

    if cfg.get("content_type") == "novel":
        await safe_edit(
            status_msg,
            "📝 **Novel content type isn't supported by this pipeline yet.**\n\n"
            "Switch **📚 Content Type** to Manhwa/Manga/Comic first."
        )
        active_jobs.pop(user_id, None)
        return

    mode = queue["mode"]
    files = queue["files"]
    MODE_LABELS = {"raw": "Raw Images", "archive": "ZIP/CBZ Extraction", "pdf": "PDF Extraction"}
    mode_label = MODE_LABELS.get(mode, mode)

    job_root = os.path.join(BASE_STAGING, str(user_id))
    if os.path.exists(job_root):
        shutil.rmtree(job_root)
    os.makedirs(job_root, exist_ok=True)
    font_dir_for_run = os.path.join(job_root, "fonts")
    os.makedirs(font_dir_for_run, exist_ok=True)
    if cfg.get("font_name"):
        src_font = FONTS_DIR / cfg["font_name"]
        if src_font.exists():
            shutil.copy(src_font, font_dir_for_run)

    total_files = len(files)
    dynamic_system_instruction = build_dynamic_system_instruction(cfg)

    subprocess_env = os.environ.copy()
    # Without this, Python fully buffers stdout/stderr when piped (as they
    # are in run_engine_subprocess), so nothing streams to the Actions log
    # live - a stuck/slow local-OCR run would look totally silent.
    subprocess_env['PYTHONUNBUFFERED'] = '1'
    subprocess_env['INPUT_LANG'] = cfg['source_lang']
    subprocess_env['PROVIDER'] = cfg['provider']
    subprocess_env['API_URL'] = cfg['api_url']
    subprocess_env['API_KEY'] = cfg['api_key']
    subprocess_env['MODEL_NAME'] = cfg['model_name']
    subprocess_env['SPECIAL_INS'] = dynamic_system_instruction

    last_minute_cleared = sanitize_cfg_values(cfg)
    if last_minute_cleared:
        await save_user_config(user_id)
        bad_list = ", ".join(f"{f}='{v}'" for f, v in last_minute_cleared)
        await safe_edit(
            status_msg,
            f"⚠️ Corrected invalid setting(s) before running: {bad_list} "
            f"(reset to engine default). Continuing..."
        )

    combined_pages = []  # merged across all files, in file/page order
    file_entries = []    # per-file bookkeeping for Pass 2 (input_dir, checkpoint_dir, output_format)

    for file_idx, source_message in enumerate(files, start=1):
        job = active_jobs.get(user_id)
        if job and job["cancel"]:
            active_jobs.pop(user_id, None)
            manual_jobs.pop(user_id, None)
            await safe_edit(status_msg, "🛑 **Manual mode Pass 1 cancelled.**")
            if os.path.exists(job_root):
                shutil.rmtree(job_root, ignore_errors=True)
            return

        input_dir = os.path.join(job_root, f"input_{file_idx:03d}")
        os.makedirs(input_dir, exist_ok=True)
        checkpoint_dir = os.path.join(job_root, ".manual_checkpoint", f"file_{file_idx:03d}")
        preview_dir = os.path.join(job_root, "preview", f"file_{file_idx:03d}")
        os.makedirs(preview_dir, exist_ok=True)

        await safe_edit(status_msg, build_status_text(mode_label, "📥 Downloading payload", file_idx, total_files, 0, 0, 5))
        downloaded_path = await source_message.download(file_name=os.path.join(job_root, f"src_{file_idx:03d}"))

        if source_message.document and source_message.document.file_name:
            ext = os.path.splitext(source_message.document.file_name)[1] or ".jpg"
        elif source_message.photo:
            ext = ".jpg"
        else:
            ext = ""

        if ext and not downloaded_path.lower().endswith(ext.lower()):
            fixed_path = downloaded_path + ext
            os.rename(downloaded_path, fixed_path)
            downloaded_path = fixed_path

        await safe_edit(status_msg, build_status_text(mode_label, "📂 Extracting", file_idx, total_files, 0, 0, 15))
        if mode == "archive" or downloaded_path.lower().endswith(('.zip', '.cbz')):
            extract_archive(downloaded_path, input_dir)
        elif mode == "pdf" or downloaded_path.lower().endswith('.pdf'):
            extract_pdf(downloaded_path, input_dir)
        else:
            shutil.move(downloaded_path, os.path.join(input_dir, os.path.basename(downloaded_path)))

        # Manual mode currently targets single/multi-page manga/comic-style
        # jobs page-by-page; still run flatten_and_order (handles naming,
        # ordering, and manhwa tiling if applicable) so page order matches
        # what the rest of the bot expects.
        ordered_map, tile_manifest = flatten_and_order(input_dir, content_type=cfg.get("content_type", "manhwa"), cfg=cfg)
        total_images = len(ordered_map)

        if total_images == 0:
            await safe_edit(status_msg, f"⚠️ File {file_idx}/{total_files}: no valid images found, skipping.")
            continue

        if tile_manifest:
            tiled_pages = len(tile_manifest)
            total_tiles = sum(len(v["tiles"]) for v in tile_manifest.values())
            await safe_edit(
                status_msg,
                f"✂️ File {file_idx}/{total_files}: {tiled_pages} tall page(s) tiled into "
                f"{total_tiles} tile(s) for OCR. They'll be recomposed automatically after "
                f"you send back your edited translations."
            )

        cmd = [
            "python",
            "-W", "ignore:Image size:UserWarning",
            "MangaTranslator/main.py",
            "--input", input_dir,
            "--output", preview_dir,
            "--batch",
            "--manual-ocr-checkpoint", checkpoint_dir,
            "--font-dir", font_dir_for_run,
            "--input-language", cfg['source_lang'],
            "--output-language", cfg['target_lang'],
            "--provider", cfg['provider'],
            "--openai-compatible-url", cfg['api_url'],
            "--openai-compatible-api-key", cfg['api_key'],
            "--model-name", cfg['model_name'],
            "--special-instructions", dynamic_system_instruction
        ]

        if cfg.get("osb_enabled", True) and cli_supports_flag("--osb-enable"):
            cmd.append("--osb-enable")
            if cli_supports_flag("--osb-font-dir"):
                cmd += ["--osb-font-dir", font_dir_for_run]

        for key, config_meta in CLI_MAPPINGS.items():
            flag, val_type = config_meta
            val = cfg.get(key)
            if val is not None and cli_supports_flag(flag):
                if val_type == "val":
                    cmd += [flag, str(val)]
                elif val_type == "bool_true" and val is True:
                    cmd.append(flag)
                elif val_type == "bool_invert" and val is False:
                    cmd.append(flag)

        await safe_edit(status_msg,
            build_status_text(mode_label, "🧠 OCR running (Manual mode Pass 1 — no translation yet)", file_idx, total_files, 0, total_images, 40),
            reply_markup=kb_cancel_only()
        )

        try:
            result = await run_engine_subprocess(
                cmd, subprocess_env, status_msg,
                progress_dir=preview_dir, total_images=total_images,
                mode_label=mode_label, stage_label="🧠 OCR running (Manual mode Pass 1 — no translation yet)",
                file_idx=file_idx, total_files=total_files,
                cancel_check=lambda: (active_jobs.get(user_id) or {}).get("cancel", False),
                base_pct=40, pct_span=40,
            )

            if result["cancelled"]:
                active_jobs.pop(user_id, None)
                manual_jobs.pop(user_id, None)
                await safe_edit(status_msg, "🛑 **Manual mode Pass 1 cancelled.**")
                if os.path.exists(job_root):
                    shutil.rmtree(job_root, ignore_errors=True)
                return

            if result["timed_out"]:
                tail = result["tail"] or "no output captured before stall"
                await safe_edit(
                    status_msg,
                    f"⏱️ **Manual mode Pass 1 stalled for over {ENGINE_STALL_TIMEOUT // 60} min "
                    f"with no new output — killed** (file {file_idx}/{total_files}).\n"
                    f"This usually means the local OCR model (manga-ocr/paddleocr-vl/paddleocr-classic/paddleocr-classic-v5) is stuck "
                    f"downloading weights or hanging on CPU inference.\n"
                    f"**Last engine output:**\n```\n{tail[-800:]}\n```"
                )
                active_jobs.pop(user_id, None)
                if os.path.exists(job_root):
                    shutil.rmtree(job_root, ignore_errors=True)
                return

            if result["returncode"] != 0:
                tail = result["tail"] or "no output captured"
                await safe_edit(
                    status_msg,
                    f"❌ **Manual mode Pass 1 exited with error on file {file_idx}/{total_files}** (code {result['returncode']}):\n```\n{tail[-800:]}\n```"
                )
                active_jobs.pop(user_id, None)
                if os.path.exists(job_root):
                    shutil.rmtree(job_root, ignore_errors=True)
                return
        except Exception as exec_err:
            await safe_edit(status_msg, f"❌ Manual mode Pass 1 error on file {file_idx}/{total_files}: {exec_err}")
            active_jobs.pop(user_id, None)
            if os.path.exists(job_root):
                shutil.rmtree(job_root, ignore_errors=True)
            return

        combined_json_path = os.path.join(checkpoint_dir, "combined_translations.json")
        if not os.path.exists(combined_json_path):
            await safe_edit(
                status_msg,
                f"❌ **File {file_idx}/{total_files}: Pass 1 produced no combined_translations.json.**\n"
                f"Skipping this file's pages."
            )
            continue

        try:
            with open(combined_json_path, "r", encoding="utf-8") as f:
                file_combined = json.load(f)
        except Exception as e:
            await safe_edit(status_msg, f"❌ File {file_idx}/{total_files}: failed to read its translations JSON: {e}")
            continue

        # Namespace each page's source_file with the file index, so a
        # multi-file job's merged JSON can't collide on identically-named
        # pages (e.g. "001.jpg" appearing in two different uploaded ZIPs),
        # and so Pass 2 can route each page back to the right file's
        # checkpoint directory.
        for page in file_combined.get("pages", []):
            page["file_idx"] = file_idx
            page["display_name"] = f"file{file_idx:03d}/{page.get('source_file')}"
            combined_pages.append(page)

        file_entries.append({
            "file_idx": file_idx,
            "checkpoint_dir": checkpoint_dir,
            "output_format": cfg['output_format'],
            "chat_id": source_message.chat.id,
            "tile_manifest": tile_manifest or None,
        })

    if not combined_pages:
        await safe_edit(status_msg, "⚠️ **Manual mode Pass 1 finished but no pages were OCR'd.** Check the files you sent.")
        active_jobs.pop(user_id, None)
        pending_files.pop(user_id, None)
        if os.path.exists(job_root):
            shutil.rmtree(job_root, ignore_errors=True)
        return

    merged = {
        "version": 1,
        "input_language": cfg['source_lang'],
        "output_language": cfg['target_lang'],
        "instructions": (
            "Fill in the 'translation' field for each bubble with your translated "
            "text, then reply to the message this file was sent with, attaching "
            "your edited copy of this JSON. Leave 'translation' empty to keep the "
            "original OCR'd text for that bubble."
        ),
        "pages": combined_pages,
    }
    merged_json_path = os.path.join(job_root, "combined_translations.json")
    with open(merged_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total_bubbles = sum(len(p.get("bubbles", [])) for p in combined_pages)
    sent_msg = await client.send_document(
        status_msg.chat.id,
        document=merged_json_path,
        caption=(
            f"✍️ **Manual Translation Mode — Pass 1 complete!**\n"
            f"📄 Pages: `{len(combined_pages)}` across `{len(file_entries)}` file(s)\n"
            f"💬 Bubbles detected: `{total_bubbles}`\n\n"
            f"Fill in each bubble's `translation` field in this JSON, then **reply "
            f"to this message** with your edited file to continue."
        )
    )

    manual_jobs[user_id] = {
        "job_root": job_root,
        "file_entries": file_entries,
        "cfg_snapshot": dict(cfg),
        "combined_json_message_id": sent_msg.id,
        "total_files": total_files,
    }
    awaiting_reply[user_id] = {"type": "manual_translations_reply", "extra": {"reply_to_message_id": sent_msg.id}}

    await safe_edit(
        status_msg,
        f"✅ **Pass 1 done — {len(combined_pages)} page(s) OCR'd, {total_bubbles} bubble(s) found.**\n"
        f"📤 Translations JSON sent above. Reply to it with your edited copy when ready.\n"
        f"Send `/cancel` to abandon this manual job."
    )
    active_jobs.pop(user_id, None)


# ================= Manual Translation Mode: Pass 2 (render from checkpoint) =================
async def execute_manual_pipeline_pass2(client, status_msg: Message, user_id: int, translations_json_path: str):
    """
    Manual Translation Mode, Pass 2. Reads the user's edited translations
    JSON (already validated/split back into per-file JSONs by the caller),
    calls main.py with --manual-render-checkpoint + --manual-translations-json
    for each file's checkpoint directory (no detection/cleaning/OCR re-run),
    and packages/sends each file's rendered output the same way the normal
    pipeline does via package_output.
    """
    job = manual_jobs.get(user_id)
    if not job:
        await safe_edit(status_msg, "❌ No pending manual translation job found. Send `/translate` again.")
        active_jobs.pop(user_id, None)
        return

    cfg = job["cfg_snapshot"]
    job_root = job["job_root"]
    file_entries = job["file_entries"]
    total_files = len(file_entries)

    # Pass 1 already created job_root/fonts and copied the user's selected
    # font into it (see execute_manual_pipeline_pass1). Reuse the same dir
    # here so Pass 2's render call doesn't fall back to main.py's default
    # "./fonts" (which doesn't exist and causes blank/untranslated renders).
    font_dir_for_run = os.path.join(job_root, "fonts")
    os.makedirs(font_dir_for_run, exist_ok=True)
    if cfg.get("font_name"):
        src_font = FONTS_DIR / cfg["font_name"]
        if src_font.exists() and not (Path(font_dir_for_run) / cfg["font_name"]).exists():
            shutil.copy(src_font, font_dir_for_run)

    try:
        with open(translations_json_path, "r", encoding="utf-8") as f:
            edited = json.load(f)
    except Exception as e:
        await safe_edit(status_msg, f"❌ Couldn't parse your edited JSON: {e}\nFix it and reply again with the corrected file.")
        active_jobs.pop(user_id, None)
        awaiting_reply[user_id] = {"type": "manual_translations_reply", "extra": {"reply_to_message_id": job.get("combined_json_message_id")}}
        return

    # Split the merged JSON back into one per-file JSON (matching what each
    # file's own main.py --manual-render-checkpoint invocation expects),
    # keyed by the file_idx namespacing added during Pass 1.
    pages_by_file = {}
    for page in edited.get("pages", []):
        file_idx = page.get("file_idx")
        pages_by_file.setdefault(file_idx, []).append(page)

    active_jobs[user_id] = {"cancel": False, "status_msg": status_msg}
    all_translated_outputs = []
    render_dir = os.path.join(job_root, "rendered")
    os.makedirs(render_dir, exist_ok=True)

    for entry in file_entries:
        job_ctrl = active_jobs.get(user_id)
        if job_ctrl and job_ctrl["cancel"]:
            break

        file_idx = entry["file_idx"]
        pages = pages_by_file.get(file_idx, [])
        if not pages:
            continue

        per_file_json = {"version": 1, "pages": pages}
        per_file_json_path = os.path.join(job_root, f"translations_file_{file_idx:03d}.json")
        with open(per_file_json_path, "w", encoding="utf-8") as f:
            json.dump(per_file_json, f, ensure_ascii=False, indent=2)

        file_render_dir = os.path.join(render_dir, f"file_{file_idx:03d}")
        os.makedirs(file_render_dir, exist_ok=True)

        await safe_edit(status_msg, build_status_text("Manual Render", "🎨 Rendering translations", file_idx, total_files, 0, len(pages), 40))

        cmd = [
            "python",
            "-W", "ignore:Image size:UserWarning",
            "MangaTranslator/main.py",
            "--input", entry["checkpoint_dir"],
            "--output", file_render_dir,
            "--batch",
            "--manual-render-checkpoint", entry["checkpoint_dir"],
            "--manual-translations-json", per_file_json_path,
            "--font-dir", font_dir_for_run,
            "--input-language", cfg['source_lang'],
            "--output-language", cfg['target_lang'],
        ]

        if cfg.get("osb_enabled", True) and cli_supports_flag("--osb-font-dir"):
            cmd += ["--osb-font-dir", font_dir_for_run]

        try:
            result = await run_engine_subprocess(
                cmd, os.environ.copy(), status_msg,
                progress_dir=file_render_dir, total_images=max(len(pages), 1),
                mode_label="Manual Render", stage_label="🎨 Rendering translations",
                file_idx=file_idx, total_files=total_files,
                cancel_check=None,  # no cancel button on this pass currently
                base_pct=40, pct_span=40,
            )
            if result["returncode"] != 0:
                tail = result["tail"] or "no output captured"
                await safe_edit(status_msg, f"❌ **Manual render failed for file {file_idx}/{total_files}:**\n```\n{tail[-800:]}\n```")
                continue
        except Exception as exec_err:
            await safe_edit(status_msg, f"❌ Manual render error on file {file_idx}/{total_files}: {exec_err}")
            continue

        tile_manifest = entry.get("tile_manifest")
        if tile_manifest:
            await safe_edit(status_msg, build_status_text("Manual Render", "🧵 Recomposing tiled pages", file_idx, total_files, len(pages), len(pages), 70))
            recompose_failures = []
            for page_idx, manifest_entry in tile_manifest.items():
                try:
                    result = recompose_tiled_page(file_render_dir, page_idx, manifest_entry, cfg=cfg)
                except Exception:
                    result = None
                if result is None:
                    recompose_failures.append(page_idx)
            if recompose_failures:
                await safe_edit(
                    status_msg,
                    f"⚠️ File {file_idx}/{total_files}: {len(recompose_failures)} tiled page(s) "
                    f"had a tile missing/failed after render, so those page(s) may be incomplete."
                )

        produced_files = [f for f in os.listdir(file_render_dir) if f.lower().endswith(IMAGE_EXTS)] if os.path.exists(file_render_dir) else []
        if not produced_files:
            await safe_edit(status_msg, f"⚠️ File {file_idx}/{total_files}: render produced no images, skipping.")
            continue

        try:
            output_path = package_output(file_render_dir, job_root, file_idx, entry["output_format"])
            all_translated_outputs.append((file_idx, output_path))
            format_label = "Raw Images (.ZIP)" if entry["output_format"] == "img" else f".{entry['output_format'].upper()}"
            await client.send_document(
                entry["chat_id"],
                document=output_path,
                caption=(
                    f"💥 **File {file_idx}/{total_files} done (Manual Translation)!**\n"
                    f"📦 Format: `{format_label}`\n"
                    f"🖼 Frames: `{len(produced_files)}`"
                )
            )
        except Exception as send_err:
            await safe_edit(status_msg, f"❌ **Failed to send file {file_idx}/{total_files}:**\n`{send_err}`")
            continue

    if all_translated_outputs:
        await safe_edit(status_msg, f"✅ **{len(all_translated_outputs)}/{total_files} file(s) rendered from your manual translations and sent!**")
    else:
        await safe_edit(status_msg, "⚠️ **Manual render finished but no files were produced/sent.** Check the logs.")

    active_jobs.pop(user_id, None)
    manual_jobs.pop(user_id, None)
    pending_files.pop(user_id, None)
    awaiting_reply.pop(user_id, None)
    if os.path.exists(job_root):
        shutil.rmtree(job_root, ignore_errors=True)


# ================= Main Pipeline Runner =================
async def execute_manga_pipeline(client, status_msg: Message, user_id: int):
    cfg = get_user_config(user_id)
    queue = pending_files.get(user_id)

    if not queue or not queue["files"]:
        await safe_edit(status_msg, "❌ Error: No files found in queue. Send `/translate` again.")
        active_jobs.pop(user_id, None)
        return

    if cfg.get("manual_translation_mode"):
        await execute_manual_pipeline_pass1(client, status_msg, user_id)
        return

    if cfg.get("content_type") == "novel":
        await safe_edit(
            status_msg,
            "📝 **Novel content type isn't supported by this pipeline yet.**\n\n"
            "This bot's engine (MangaTranslator) is built around detecting and "
            "redrawing speech bubbles in comic/manga art - it has no bubbles to "
            "find in prose text, so running it here would waste API calls and "
            "likely produce broken output.\n\n"
            "Switch **📚 Content Type** to Manhwa/Manga/Comic for image-based "
            "chapters. Novel/text support would need a separate OCR+translate "
            "path - let the maintainer know if you want that built."
        )
        active_jobs.pop(user_id, None)
        return

    mode = queue["mode"]
    files = queue["files"]
    MODE_LABELS = {"raw": "Raw Images", "archive": "ZIP/CBZ Extraction", "pdf": "PDF Extraction"}
    mode_label = MODE_LABELS.get(mode, mode)

    job_root = os.path.join(BASE_STAGING, str(user_id))
    translated_dir = os.path.join(job_root, "translated")
    font_dir_for_run = os.path.join(job_root, "fonts")
    is_resuming = user_id in paused_jobs
    if os.path.exists(job_root) and not is_resuming:
        shutil.rmtree(job_root)
    os.makedirs(translated_dir, exist_ok=True)
    os.makedirs(font_dir_for_run, exist_ok=True)

    if cfg.get("font_name"):
        src_font = FONTS_DIR / cfg["font_name"]
        if src_font.exists():
            shutil.copy(src_font, font_dir_for_run)

    total_files = len(files)
    all_translated_outputs = []  
    failure_reasons = []  
    resume_from = paused_jobs.pop(user_id, {}).get("stopped_at_file") or 1

    for file_idx, source_message in enumerate(files, start=1):
        if file_idx < resume_from:
            continue  

        job = active_jobs.get(user_id)
        if job and job["cancel"]:
            await handle_job_cancelled(client, status_msg, user_id, translated_dir)
            return

        input_dir = os.path.join(job_root, f"input_{file_idx:03d}")
        os.makedirs(input_dir, exist_ok=True)

        await safe_edit(status_msg, build_status_text(mode_label, "📥 Downloading payload", file_idx, total_files, 0, 0, 5))
        downloaded_path = await source_message.download(file_name=os.path.join(job_root, f"src_{file_idx:03d}"))

        if source_message.document and source_message.document.file_name:
            ext = os.path.splitext(source_message.document.file_name)[1] or ".jpg"
        elif source_message.photo:
            ext = ".jpg"
        else:
            ext = ""

        if ext and not downloaded_path.lower().endswith(ext.lower()):
            fixed_path = downloaded_path + ext
            os.rename(downloaded_path, fixed_path)
            downloaded_path = fixed_path

        await safe_edit(status_msg, build_status_text(mode_label, "📂 Extracting", file_idx, total_files, 0, 0, 15))
        if mode == "archive" or downloaded_path.lower().endswith(('.zip', '.cbz')):
            extract_archive(downloaded_path, input_dir)
        elif mode == "pdf" or downloaded_path.lower().endswith('.pdf'):
            extract_pdf(downloaded_path, input_dir)
        else:
            shutil.move(downloaded_path, os.path.join(input_dir, os.path.basename(downloaded_path)))

        ordered_map, tile_manifest = flatten_and_order(input_dir, content_type=cfg.get("content_type", "manhwa"), cfg=cfg)
        total_images = len(ordered_map)

        if total_images == 0:
            await safe_edit(status_msg, f"⚠️ File {file_idx}/{total_files}: no valid images found, skipping.")
            continue

        if tile_manifest:
            tiled_pages = len(tile_manifest)
            total_tiles = sum(len(v["tiles"]) for v in tile_manifest.values())
            await safe_edit(
                status_msg,
                build_status_text(mode_label, f"✂️ Tiling {tiled_pages} tall page(s) into {total_tiles} tile(s) for detection", file_idx, total_files, 0, total_images, 20)
            )
            total_images = len([f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS)])

        dynamic_system_instruction = build_dynamic_system_instruction(cfg)

        subprocess_env = os.environ.copy()
        subprocess_env['PYTHONUNBUFFERED'] = '1'
        subprocess_env['INPUT_LANG'] = cfg['source_lang']
        subprocess_env['PROVIDER'] = cfg['provider']
        subprocess_env['API_URL'] = cfg['api_url']
        subprocess_env['API_KEY'] = cfg['api_key']
        subprocess_env['MODEL_NAME'] = cfg['model_name']
        subprocess_env['SPECIAL_INS'] = dynamic_system_instruction

        file_translated_dir = os.path.join(translated_dir, f"file_{file_idx:03d}")
        os.makedirs(file_translated_dir, exist_ok=True)

        # Tall manhwa/webtoon pages (and their recomposed tiles) routinely exceed
        # PIL's default ~89-megapixel DecompressionBombWarning threshold. This is
        # expected content for this bot, not an actual attack, so we silence it
        # via a `-W` interpreter flag on the command itself (not PYTHONWARNINGS,
        # which caused "Invalid -W option ignored" errors here before due to
        # env-var parsing/quoting differences across shells). The filter format
        # is "action:message:category:module", and DecompressionBombWarning is
        # a UserWarning subclass, so filtering on UserWarning with the message
        # prefix "Image size" catches it without needing PIL importable yet.
        cmd = [
            "python",
            "-W", "ignore:Image size:UserWarning",
            "MangaTranslator/main.py",
            "--input", input_dir,
            "--output", file_translated_dir,
            "--batch",
            "--font-dir", font_dir_for_run,
            "--input-language", cfg['source_lang'],
            "--output-language", cfg['target_lang'],
            "--provider", cfg['provider'],
            "--openai-compatible-url", cfg['api_url'],
            "--openai-compatible-api-key", cfg['api_key'],
            "--model-name", cfg['model_name'],
            "--special-instructions", dynamic_system_instruction
        ]

        if cfg.get("osb_enabled", True) and cli_supports_flag("--osb-enable"):
            cmd.append("--osb-enable")
            if cli_supports_flag("--osb-font-dir"):
                cmd += ["--osb-font-dir", font_dir_for_run]

        # SAFETY NET: strip any choice-restricted field holding an invalid value
        # (e.g. a stale/hand-edited translation_mode like "contextual") before it
        # can reach main.py's argparse and abort the whole job with "invalid choice".
        last_minute_cleared = sanitize_cfg_values(cfg)
        if last_minute_cleared:
            await save_user_config(user_id)
            bad_list = ", ".join(f"{f}='{v}'" for f, v in last_minute_cleared)
            await safe_edit(
                status_msg,
                f"⚠️ Corrected invalid setting(s) before running: {bad_list} "
                f"(reset to engine default). Continuing..."
            )

        # INJECT ALL CUSTOM PARAMETERS
        for key, config_meta in CLI_MAPPINGS.items():
            flag, val_type = config_meta
            val = cfg.get(key)
            if val is not None and cli_supports_flag(flag):
                if val_type == "val":
                    cmd += [flag, str(val)]
                elif val_type == "bool_true" and val is True:
                    cmd.append(flag)
                elif val_type == "bool_invert" and val is False:
                    cmd.append(flag)

        await safe_edit(status_msg, 
            build_status_text(mode_label, "🧠 OCR + Translation running", file_idx, total_files, 0, total_images, 40),
            reply_markup=kb_cancel_only()
        )

        try:
            result = await run_engine_subprocess(
                cmd, subprocess_env, status_msg,
                progress_dir=file_translated_dir, total_images=total_images,
                mode_label=mode_label, stage_label="🧠 OCR + Translation running",
                file_idx=file_idx, total_files=total_files,
                cancel_check=lambda: (active_jobs.get(user_id) or {}).get("cancel", False),
                base_pct=40, pct_span=40,
            )

            if result["cancelled"]:
                await handle_job_cancelled(client, status_msg, user_id, translated_dir, file_idx, total_files, ordered_map)
                return

            if result["timed_out"]:
                tail = result["tail"] or "no output captured before stall"
                await safe_edit(
                    status_msg,
                    f"⏱️ **File {file_idx}/{total_files} stalled for over {ENGINE_STALL_TIMEOUT // 60} min "
                    f"with no new output image — killed.**\n"
                    f"This usually means the local OCR model (manga-ocr/paddleocr-vl/paddleocr-classic/paddleocr-classic-v5) is stuck "
                    f"downloading weights or hanging on CPU inference.\n"
                    f"**Last engine output:**\n```\n{tail[-800:]}\n```"
                )
                active_jobs.pop(user_id, None)
                return

            if result["returncode"] != 0:
                tail = result["tail"] or "no output captured"
                await safe_edit(
                    status_msg,
                    f"❌ **Engine exited with error on file {file_idx}/{total_files}** (code {result['returncode']}):\n```\n{tail[-800:]}\n```"
                )
                active_jobs.pop(user_id, None)
                return
        except Exception as exec_err:
            await safe_edit(status_msg, f"❌ Engine error on file {file_idx}/{total_files}: {exec_err}")
            active_jobs.pop(user_id, None)
            return

        await safe_edit(status_msg, build_status_text(mode_label, "📦 Packaging output", file_idx, total_files, total_images, total_images, 90))

        produced_files = []
        if os.path.exists(file_translated_dir):
            produced_files = [f for f in os.listdir(file_translated_dir) if f.lower().endswith(IMAGE_EXTS)]

        if not produced_files:
            debug_tail = (result["tail"] or "no output captured")[-500:]
            failure_reasons.append((file_idx, debug_tail))
            await safe_edit(
                status_msg,
                f"❌ **File {file_idx}/{total_files} produced no output.**\n"
                f"Engine exited cleanly (code 0) but wrote no translated images.\n```\n{debug_tail}\n```\n"
                f"Skipping to next file."
            )
            continue

        if tile_manifest:
            await safe_edit(status_msg, build_status_text(mode_label, "🧵 Recomposing tiled pages", file_idx, total_files, total_images, total_images, 85))
            recompose_failures = []
            duplicate_suspected_pages = []
            for page_idx, manifest_entry in tile_manifest.items():
                try:
                    result = recompose_tiled_page(file_translated_dir, page_idx, manifest_entry, cfg=cfg)
                except Exception:
                    # recompose_tiled_page already cleans up its own tiles
                    # in a finally block even on error, so this just needs
                    # to record the failure and keep going instead of
                    # letting one bad page kill the whole file.
                    result = None
                if result is None:
                    recompose_failures.append(page_idx)
                    continue
                result_path, flagged_seams = result
                if flagged_seams:
                    duplicate_suspected_pages.append(page_idx)
            if recompose_failures:
                await safe_edit(
                    status_msg,
                    f"⚠️ File {file_idx}/{total_files}: {len(recompose_failures)} tiled page(s) "
                    f"had a tile that failed to translate, so those pages may be incomplete. "
                    f"Continuing with the rest."
                )
            if duplicate_suspected_pages:
                pages_list = ", ".join(str(p) for p in duplicate_suspected_pages)
                await safe_edit(
                    status_msg,
                    f"⚠️ File {file_idx}/{total_files}: possible duplicated text detected on "
                    f"page(s) {pages_list} (unusually tall panel with no clean cut point found). "
                    f"Please double-check these pages in the output."
                )
            produced_files = [f for f in os.listdir(file_translated_dir) if f.lower().endswith(IMAGE_EXTS)]

        try:
            if cfg['output_format'] == 'img':
                image_files = sorted(
                    [f for f in os.listdir(file_translated_dir) if f.lower().endswith(IMAGE_EXTS)]
                )
                archive_base = os.path.join(job_root, f"raw_images_{file_idx:03d}")
                shutil.make_archive(archive_base, "zip", file_translated_dir)
                raw_zip_path = f"{archive_base}.zip"
                await client.send_document(
                    source_message.chat.id,
                    document=raw_zip_path,
                    caption=(
                        f"💥 **File {file_idx}/{total_files} done!**\n"
                        f"📦 Format: `Raw Images (.ZIP)`\n"
                        f"🖼 Frames: `{len(image_files)}`"
                    )
                )
                all_translated_outputs.append((file_idx, raw_zip_path))
            else:
                output_path = package_output(file_translated_dir, job_root, file_idx, cfg['output_format'])
                all_translated_outputs.append((file_idx, output_path))
                await client.send_document(
                    source_message.chat.id,
                    document=output_path,
                    caption=(
                        f"💥 **File {file_idx}/{total_files} done!**\n"
                        f"📦 Format: `.{cfg['output_format'].upper()}`\n"
                        f"🖼 Frames: `{total_images}`"
                    )
                )
        except Exception as send_err:
            await safe_edit(status_msg, f"❌ **Failed to send file {file_idx}/{total_files}:**\n`{send_err}`")
            active_jobs.pop(user_id, None)
            return

    if not all_translated_outputs:
        if failure_reasons:
            last_idx, last_reason = failure_reasons[-1]
            summary = (
                f"⚠️ **Job finished — 0/{total_files} file(s) produced output.**\n\n"
                f"**Last failure (file {last_idx}/{total_files}):**\n```\n{last_reason}\n```"
            )
            if len(failure_reasons) > 1:
                summary += f"\n\n_{len(failure_reasons)} file(s) failed total._"
            await safe_edit(status_msg, summary)
        else:
            await safe_edit(status_msg, "⚠️ **Job finished but no files were produced/sent.** Check the engine logs.")
    else:
        await safe_edit(status_msg, f"✅ **{len(all_translated_outputs)}/{total_files} file(s) translated and sent!**")
    active_jobs.pop(user_id, None)
    pending_files.pop(user_id, None)
    paused_jobs.pop(user_id, None)
    if os.path.exists(job_root):
        shutil.rmtree(job_root, ignore_errors=True)

def package_output(source_dir, job_root, file_idx, output_format):
    archive_base = os.path.join(job_root, f"output_{file_idx:03d}")
    if output_format in ("zip", "cbz"):
        shutil.make_archive(archive_base, "zip", source_dir)
        payload = f"{archive_base}.zip"
        if output_format == "cbz":
            cbz_payload = payload.replace(".zip", ".cbz")
            os.rename(payload, cbz_payload)
            payload = cbz_payload
        return payload
    elif output_format == "pdf":
        images = sorted(Path(source_dir).glob("*.*"))
        pdf_path = f"{archive_base}.pdf"
        try:
            from PIL import Image
            imgs = [Image.open(p).convert("RGB") for p in images if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
            if imgs:
                imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
                return pdf_path
        except Exception:
            pass
        shutil.make_archive(archive_base, "zip", source_dir)
        return f"{archive_base}.zip"
    else:
        shutil.make_archive(archive_base, "zip", source_dir)
        return f"{archive_base}.zip"

async def handle_job_cancelled(client, status_msg, user_id, translated_dir, file_idx=None, total_files=None, ordered_map=None):
    paused_jobs[user_id] = {
        "translated_dir": translated_dir,
        "stopped_at_file": file_idx,
        "total_files": total_files,
    }
    active_jobs.pop(user_id, None)
    await safe_edit(status_msg, 
        "🛑 **Translation stopped.**\nProgress so far has been saved.\nWhat would you like to do?",
        reply_markup=kb_resume_options()
    )

async def resume_manga_pipeline(client, status_msg, user_id):
    await execute_manga_pipeline(client, status_msg, user_id)

async def send_partial_results(client, status_msg, user_id):
    state = paused_jobs.get(user_id)
    if not state:
        await safe_edit(status_msg, "❌ No paused job data found.")
        return
    translated_dir = state["translated_dir"]
    if os.path.exists(translated_dir) and os.listdir(translated_dir):
        archive_base = translated_dir + "_partial"
        shutil.make_archive(archive_base, "zip", translated_dir)
        await client.send_document(
            status_msg.chat.id,
            document=f"{archive_base}.zip",
            caption="📤 Partial translated files (progress so far)."
        )
        os.remove(f"{archive_base}.zip")
    else:
        await safe_edit(status_msg, "⚠️ No translated files available yet.")
    paused_jobs.pop(user_id, None)
    pending_files.pop(user_id, None)

if __name__ == "__main__":
    app.run()
