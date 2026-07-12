import os
import re
import sys
import json
import shutil
import zipfile
import asyncio
import subprocess
from pathlib import Path
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery

# ================= Configuration & Secrets =================
API_ID = int(os.environ.get("API_ID", 0))
API_HASH = os.environ.get("API_HASH", "")
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AUTH_USERS_STR = os.environ.get("AUTHORIZED_USERS", "")

if not API_ID or not API_HASH or not BOT_TOKEN:
    print("❌ Error: API_ID, API_HASH, ya BOT_TOKEN repository secrets mein missing hain!")
    sys.exit(1)

AUTHORIZED_USERS = [int(u.strip()) for u in AUTH_USERS_STR.split(",") if u.strip().isdigit()]

app = Client("YomiTranslatorBot", bot_token=BOT_TOKEN, api_id=API_ID, api_hash=API_HASH)

# ================= Persistent Storage Paths =================
# Runner par ye repo ke andar rehte hain, taake font/prompt library commit hoke persist ho.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "bot_data"
FONTS_DIR = DATA_DIR / "fonts"
PROMPTS_DIR = DATA_DIR / "prompts"
USERS_FILE = DATA_DIR / "user_settings.json"

for d in [DATA_DIR, FONTS_DIR, PROMPTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================= Language Catalogs =================
SOURCE_LANGS = [
    ("Auto Detect", "auto"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("English", "en"),
    ("Chinese", "zh"),
]

TARGET_LANGS = [
    ("English", "English"),
    ("Hindi (Roman)", "Roman Hindi"),
    ("Urdu (Roman)", "Roman Urdu"),
    ("Urdu", "Urdu"),
    ("Hindi", "Hindi"),
    ("French", "French"),
    ("Spanish", "Spanish"),
]

PROVIDERS = ["Gemini", "Anthropic", "OpenAI", "OpenAI-Compatible"]

OUTPUT_FORMATS = [
    ("ZIP Package", "zip"),
    ("CBZ Archive", "cbz"),
    ("PDF Document", "pdf"),
    ("Raw Images", "img"),
]

UPLOAD_MODES = [
    ("🖼 Raw Images", "raw"),
    ("📦 ZIP / CBZ", "archive"),
    ("📄 PDF", "pdf"),
]

# ================= Content Type Profiles =================
# Different source formats need fundamentally different processing:
#  - Manhwa: long vertical-scroll strips, often delivered as sliced fragments
#    (001__001.jpg, 001__002.jpg...). After stitching, a single page can be
#    10,000+ px tall, which crushes YOLO speech-bubble detection since the
#    detector resizes the image down to its fixed input resolution (e.g.
#    640-1280px) - bubbles become sub-pixel and vanish. Fix: tile the tall
#    page into overlapping detection-sized windows, run the translator per
#    tile, then recompose.
#  - Manga/Comic: normal single-page images/spreads, no tiling needed -
#    default single-page pipeline as-is.
#  - Novel: text-heavy prose, no speech bubbles or panel art at all. Image
#    pipeline (detection/cleaning/rendering) is actively wrong here - it
#    would try to detect bubbles that don't exist. This mode should route to
#    a plain OCR + translate pass without the bubble/render stages.
CONTENT_TYPES = [
    ("🍥 Manhwa (long strip)", "manhwa"),
    ("📖 Manga (page-by-page)", "manga"),
    ("💬 Comic (Western)", "comic"),
    ("📝 Novel (text only)", "novel"),
]

# Tiling parameters for long-strip Manhwa. Each tile is TILE_HEIGHT tall with
# TILE_OVERLAP px shared between consecutive tiles so a bubble that straddles
# a cut line still appears whole in at least one tile.
MANHWA_TILE_HEIGHT = 1600
MANHWA_TILE_OVERLAP = 200
# Only kick in tiling once a stitched page exceeds this height - short
# Manhwa pages behave fine as a single image and tiling would just add
# unnecessary subprocess calls.
MANHWA_TILE_TRIGGER_HEIGHT = 2200

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
pending_files = {}   # user_id -> {"mode": "raw"/"archive"/"pdf", "files": [Message,...], "collecting": bool}
active_jobs = {}      # user_id -> {"cancel": bool, "status_msg": Message}
awaiting_reply = {}   # user_id -> {"type": "custom_lang"/"font_upload"/"prompt_name"/"prompt_body"/"api_field", "extra": {...}}

def default_config():
    return {
        "source_lang": "auto",
        "source_lang_label": "Auto Detect",
        "target_lang": "Roman Hindi",
        "target_lang_label": "Hindi (Roman)",
        "font_name": None,           # currently selected font filename
        "provider": "OpenAI-Compatible",
        "api_url": "https://api.highwayapi.ai/openai",
        "api_key": "",
        "model_name": "gemini-3-flash-preview",
        "output_format": "zip",
        # Which processing profile to use - see CONTENT_TYPES. Controls
        # whether long-strip tiling kicks in (manhwa), whether the image
        # pipeline runs at all (novel = text-only, skipped), etc.
        "content_type": "manhwa",
        "content_type_label": "🍥 Manhwa (long strip)",
        # OSB (Outside Speech Bubble) text detection - catches Manhwa-style
        # narration/SFX/dialogue placed outside drawn bubble shapes. Defaults
        # ON since most scraped Manhwa needs it; requires the HF_TOKEN env
        # var / --osb setup described in the MangaTranslator README to
        # actually download the AnimeText_yolo model on first use.
        "osb_enabled": True,
        "system_prompt_name": DEFAULT_SYSTEM_PROMPT_NAME,
        "system_prompt_text": DEFAULT_SYSTEM_PROMPT_TEXT,
        "user_prompt_name": DEFAULT_USER_PROMPT_NAME,
        "user_prompt_text": DEFAULT_USER_PROMPT_TEXT,
    }

def get_user_config(user_id):
    uid = str(user_id)
    if uid not in user_settings:
        user_settings[uid] = default_config()
        _save_all_settings(user_settings)
    else:
        # Backfill any new config keys added in later versions (e.g. osb_enabled)
        # for users whose settings were saved before that key existed, so we
        # never hit a KeyError on an old config.
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
    # kind: "system" or "user"
    return PROMPTS_DIR / f"{kind}_prompts.json"

def load_prompt_library(kind):
    f = _prompt_lib_file(kind)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
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

# ================= Git Persistence (GitHub Actions runner) =================
def git_commit_data(message):
    """Commit bot_data/ changes so fonts & prompts persist across ephemeral runner jobs."""
    try:
        subprocess.run(["git", "add", str(DATA_DIR)], cwd=str(BASE_DIR), check=False,
                        capture_output=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(BASE_DIR),
                                 capture_output=True)
        if result.returncode == 0:
            return  # nothing changed
        subprocess.run(["git", "-c", "user.email=bot@yomisubs.local",
                         "-c", "user.name=YomiSubsBot",
                         "commit", "-m", message], cwd=str(BASE_DIR), check=False,
                        capture_output=True)
        subprocess.run(["git", "push"], cwd=str(BASE_DIR), check=False, capture_output=True)
    except Exception as e:
        print(f"⚠️ Git persistence skipped: {e}")


# ================= Limits =================
PROMPT_NAME_MAX_LEN = 32  # only limit that exists anywhere in this bot

# ================= Safe Telegram UI Helpers =================
# Prevents "stuck button" bug: Telegram keeps a button's loading spinner active
# until callback_query.answer() is called, and if edit_text() throws (e.g.
# "message is not modified" or a network hiccup) the handler used to crash
# before ever answering the query, leaving the old menu frozen on screen.
async def safe_edit(message, text, reply_markup=None):
    try:
        await message.edit_text(text, reply_markup=reply_markup)
    except Exception as e:
        err = str(e).lower()
        if "not modified" in err:
            return  # content identical, nothing to do
        # Any other failure: try to at least refresh the markup, else swallow
        try:
            await message.edit_text(text + " ", reply_markup=reply_markup)
        except Exception as e2:
            print(f"⚠️ safe_edit failed: {e2}")

async def safe_answer(query, text=None, show_alert=False):
    try:
        if text:
            await query.answer(text, show_alert=show_alert)
        else:
            await query.answer()
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
def kb_main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📚 Content Type", callback_data="menu_content_type")],
        [InlineKeyboardButton("🌐 Language Settings", callback_data="menu_lang")],
        [InlineKeyboardButton("🔡 Font Track", callback_data="menu_font")],
        [InlineKeyboardButton("⚙️ Provider & API", callback_data="menu_api")],
        [InlineKeyboardButton("📝 Prompt Library", callback_data="menu_prompt")],
        [InlineKeyboardButton("📦 Output Format", callback_data="menu_output")],
    ])

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
    await message.reply_text("🛠 **Settings**\nChoose a category to configure:", reply_markup=kb_main_menu())

@app.on_message(filters.command("translate") & auth_filter)
async def translate_cmd(client, message: Message):
    user_id = message.from_user.id
    if user_id in active_jobs:
        await message.reply_text("⚠️ Ek job already chal rahi hai. Pehle usse cancel ya complete karo.")
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
        await message.reply_text("⚠️ Koi active upload session nahi hai. `/translate` se shuru karo.")
        return
    queue["collecting"] = False
    if not queue["files"]:
        await message.reply_text("❌ Koi file receive nahi hui. `/translate` dobara try karo.")
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
        await message.reply_text("🛑 Cancellation requested. Job rukne wali hai...")
    else:
        pending_files.pop(user_id, None)
        awaiting_reply.pop(user_id, None)
        await message.reply_text("✅ Session cleared.")

# ================= Safe Background Task Runner =================
# asyncio.create_task() silently swallows exceptions if the task's result is
# never awaited/checked. That was the cause of jobs freezing at "Downloading
# payload" with no error shown - any exception in the pipeline just vanished.
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

    # Answer immediately so the button never stays stuck in a loading state,
    # even if something below raises. Branches that want a custom toast text
    # call safe_answer(query, "...") again later, which is harmless (Telegram
    # ignores a second answer silently on the client side after the first).
    await safe_answer(query)

    # ---------- Main Menu ----------
    if data == "main_menu":
        await safe_edit(query.message, "🛠 **Settings**\nChoose a category to configure:", reply_markup=kb_main_menu())
        return

    # ---------- Content Type Menu ----------
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

    # ---------- Language Menu ----------
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

    # ---------- Font Menu ----------
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

    # ---------- API / Provider Menu ----------
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
        field = data.split("api_field_", 1)[1]  # api_url / api_key / model_name
        pretty = {"api_url": "Base URL", "api_key": "API Key", "model_name": "Model ID"}.get(field, field)
        awaiting_reply[user_id] = {"type": "api_field", "extra": {"field": field}}
        await safe_edit(query.message, f"✍️ **Reply to this message with the new {pretty}.**")
        return

    # ---------- Prompt Library Menu ----------
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
            # if deleted prompt was selected, fall back to whatever remains first
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
        # Quick in-place edit of whichever prompt is currently selected, without
        # having to delete it and re-add a new one under a different name.
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
            f"_Agar bada hai to multiple messages mein todkar reply karo, phir `/donedone` reply karo._"
        )
        return

    # ---------- Output Format Menu ----------
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

    # ---------- Upload Mode Selection (/translate flow) ----------
    if data.startswith("uploadmode_"):
        mode = data.split("_", 1)[1]
        pending_files[user_id] = {"mode": mode, "files": [], "collecting": True}
        mode_label = dict(UPLOAD_MODES).get(f"uploadmode_{mode}", mode)
        hint = {
            "raw": "Ab apni saari images bhejo. Jab complete ho jaye, `/end` bhejo.",
            "archive": "Ab apni ZIP ya CBZ file(s) bhejo. Multiple bhi bhej sakte ho. Jab complete ho jaye, `/end` bhejo.",
            "pdf": "Ab apni PDF file(s) bhejo. Multiple bhi bhej sakte ho. Jab complete ho jaye, `/end` bhejo.",
        }.get(mode, "Files bhejo, phir /end bhejo.")
        await safe_edit(query.message, f"📥 **Upload mode:** `{mode}`\n{hint}")
        return

    # ---------- Job Pipeline Controls ----------
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

    # Case 1: user is uploading a font (triggered via Font Track > Add Font)
    pending_reply = awaiting_reply.get(user_id)
    if pending_reply and pending_reply["type"] == "font_upload" and message.document:
        doc_name = message.document.file_name or ""
        if not doc_name.lower().endswith((".ttf", ".otf")):
            await message.reply_text("❌ Sirf .ttf ya .otf files allowed hain.")
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

    # Case 2: user is collecting files for a translation job
    queue = pending_files.get(user_id)
    if not queue or not queue.get("collecting"):
        await message.reply_text("ℹ️ Pehle `/translate` bhejo aur upload type select karo.")
        return

    mode = queue["mode"]
    if mode == "raw" and not message.photo and not (message.document and (message.document.mime_type or "").startswith("image/")):
        await message.reply_text("❌ Is mode mein sirf images allowed hain.")
        return
    if mode == "archive" and not (message.document and message.document.file_name and message.document.file_name.lower().endswith((".zip", ".cbz"))):
        await message.reply_text("❌ Is mode mein sirf ZIP/CBZ files allowed hain.")
        return
    if mode == "pdf" and not (message.document and message.document.file_name and message.document.file_name.lower().endswith(".pdf")):
        await message.reply_text("❌ Is mode mein sirf PDF files allowed hain.")
        return

    queue["files"].append(message)
    await message.reply_text(f"✅ Queued ({len(queue['files'])} total). Aur bhejo ya `/end` bhejo.")

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

    if kind == "prompt_name":
        prompt_kind = pending_reply["extra"]["kind"]
        if len(text) > PROMPT_NAME_MAX_LEN:
            await message.reply_text(f"❌ Naam {PROMPT_NAME_MAX_LEN} characters se zyada nahi ho sakta (`{len(text)}` diya). Dobara reply karo.")
            return
        awaiting_reply[user_id] = {"type": "prompt_body", "extra": {"kind": prompt_kind, "name": text, "parts": []}}
        await message.reply_text(
            f"✍️ **Reply to this message with the {prompt_kind} prompt text for** `{text}`.\n\n"
            f"_Agar prompt Telegram ki 4096 character limit se bada hai, usse multiple messages mein todkar "
            f"reply karo (har hissa isi message ko reply karke bhejo). Jab sab bhej do, `/donedone` reply karo._"
        )
        return

    if kind == "prompt_body":
        prompt_kind = pending_reply["extra"]["kind"]
        name = pending_reply["extra"]["name"]
        parts = pending_reply["extra"].setdefault("parts", [])

        if text.strip() == "/donedone":
            if not parts:
                await message.reply_text("❌ Koi prompt text abhi tak nahi mila. Pehle text reply karo, phir `/donedone`.")
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

        # Accumulate this chunk and keep waiting — this also transparently handles the
        # case where Telegram itself split one long paste into multiple messages.
        parts.append(text)
        total_len = sum(len(p) for p in parts)
        await message.reply_text(
            f"➕ Part {len(parts)} received ({len(text)} chars, total so far: {total_len}).\n"
            f"Aur bhejo, ya `/donedone` reply karke save karo."
        )
        return

# ================= Job State for Pause/Resume =================
paused_jobs = {}  # user_id -> {"translated_dir":..., "queue":[...], "current_index":..., "settings":..., "total_images":...}

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
    """
    Pure-Python PDF page extraction using PyMuPDF (fitz).
    Replaces the old `pdftoppm` shell-out, which used to hang or silently
    fail on headless runners where poppler-utils was missing/misconfigured
    or the PDF had complex/multi-layered content.
    Pages are rendered at ~200 DPI and saved as flat RGB .jpg frames
    (alpha=False) since the OCR/vision translation model doesn't need an
    alpha channel and .jpg keeps payload size down for the LLM.
    """
    import fitz  # PyMuPDF
    zoom = 200 / 72  # fitz default is 72 DPI; scale up to ~200 DPI
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
    """
    Stitches multiple narrow-strip slices (e.g. 001__001.jpg, 001__002.jpg) that
    together represent ONE logical manhwa page back into a single tall image.
    This fixes two problems caused by treating each slice as an independent page:
      1. Chaotic aspect ratios / extreme zoom-in on the compiled output.
      2. Text bubbles that were cut in half across two slices breaking OCR
         segmentation, since the model never saw the full bubble in one frame.
    Slices are sorted naturally (so 001__2 comes before 001__10) and pasted
    top-to-bottom using PIL with running y_offset accumulation.
    """
    from PIL import Image
    slice_files = sorted(slice_files, key=_natural_key)
    imgs = [Image.open(os.path.join(input_dir, f)).convert("RGB") for f in slice_files]
    total_width = max(im.width for im in imgs)
    total_height = sum(im.height for im in imgs)
    stitched = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    y_offset = 0
    for im in imgs:
        # Center narrower slices horizontally so stitched pages stay visually aligned.
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
    """
    Scans a flat directory for scraped-manhwa-style sliced filenames using the
    `<page>__<slice>.<ext>` convention (e.g. 001__001.jpg, 001__002.jpg) and
    stitches each group of slices back into one tall page before the images
    ever reach the translator core. Files that don't match the pattern (i.e.
    already-whole pages) are left untouched.
    """
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
        # A "group" of exactly one slice is just a normal page with an
        # incidental "__" in its name - leave it as-is, no stitching needed.

def flatten_and_order(input_dir, content_type="manhwa"):
    """Move nested images to root, stitch sliced strips, sort naturally, rename to 001,002... ordering.
    Returns (ordered_map, tile_manifest). tile_manifest is None unless tiling was applied."""
    for root, _, files in os.walk(input_dir, topdown=False):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                shutil.move(os.path.join(root, f), os.path.join(input_dir, f))
        if root != input_dir:
            try:
                os.rmdir(root)
            except Exception:
                pass

    # Stitch scraped narrow-strip Manhwa slices (e.g. 001__001.jpg, 001__002.jpg)
    # back into single full pages before ordering/renaming, so the translator
    # core sees one coherent page per file instead of fragments.
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
        tile_manifest = tile_tall_pages(input_dir, ordered_map)

    return ordered_map, tile_manifest

# ================= Long-Strip Tiling (Manhwa) =================
# Root cause of "no speech bubbles detected" on stitched Manhwa strips:
# YOLO detectors resize the whole image down to a fixed input resolution
# (commonly 640-1280px on the longest side) before running inference. A
# stitched webtoon page that's 10,000-16,000px tall gets crushed down so
# hard that speech bubbles shrink to a handful of pixels - well below what
# the detector can resolve, so it reports zero bubbles even though they're
# clearly visible to a human. The fix used by real webtoon translation
# tools: slice the tall page into overlapping windows sized close to the
# detector's native resolution, run detection/translation per window, then
# recompose the translated windows back into one tall page. The overlap
# (MANHWA_TILE_OVERLAP) ensures a bubble that straddles a cut line still
# appears whole in at least one tile.

def tile_tall_pages(input_dir, ordered_map):
    """
    For every page taller than MANHWA_TILE_TRIGGER_HEIGHT, slice it into
    overlapping tiles saved as `{page}_tile{n}.jpg` and remove the original
    tall page. Returns a manifest dict:
        { page_idx: {"tiles": [tile_filename, ...], "heights": [...],
                     "overlap": px, "width": px} }
    Pages at/under the trigger height are left completely untouched and are
    absent from the manifest.
    """
    from PIL import Image
    manifest = {}
    for idx, fname in list(ordered_map.items()):
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            continue
        with Image.open(path) as im:
            width, height = im.size
            if height <= MANHWA_TILE_TRIGGER_HEIGHT:
                continue  # short enough for the detector as-is, skip tiling

            im = im.convert("RGB")
            stride = MANHWA_TILE_HEIGHT - MANHWA_TILE_OVERLAP
            tile_files = []
            tile_heights = []
            y = 0
            tile_n = 0
            while y < height:
                tile_bottom = min(y + MANHWA_TILE_HEIGHT, height)
                tile = im.crop((0, y, width, tile_bottom))
                tile_name = f"{os.path.splitext(fname)[0]}_tile{tile_n:03d}.jpg"
                tile.save(os.path.join(input_dir, tile_name), quality=95)
                tile_files.append(tile_name)
                tile_heights.append(tile_bottom - y)
                tile_n += 1
                if tile_bottom >= height:
                    break
                y += stride

            manifest[idx] = {
                "tiles": tile_files,
                "heights": tile_heights,
                "overlap": MANHWA_TILE_OVERLAP,
                "width": width,
                "original_height": height,
                "original_name": fname,
            }
        os.remove(path)  # replaced by its tiles
    return manifest

def recompose_tiled_page(translated_dir, page_idx, manifest_entry):
    """
    Stitches translated tile outputs back into one tall page image, trimming
    the overlap region from every tile after the first so overlapping text
    doesn't appear duplicated in the final output. Returns the output path,
    or None if any translated tile is missing (caller should treat that page
    as a partial/failed translation rather than silently shipping gaps).
    """
    from PIL import Image
    tiles = manifest_entry["tiles"]
    heights = manifest_entry["heights"]
    overlap = manifest_entry["overlap"]
    width = manifest_entry["width"]
    total_height = manifest_entry["original_height"]

    translated_tile_paths = []
    for tile_name in tiles:
        stem = os.path.splitext(tile_name)[0]
        found = None
        for ext in IMAGE_EXTS:
            candidate = os.path.join(translated_dir, stem + ext)
            if os.path.exists(candidate):
                found = candidate
                break
        if found is None:
            return None  # a tile failed to translate - don't silently ship a gap
        translated_tile_paths.append(found)

    recomposed = Image.new("RGB", (width, total_height), (255, 255, 255))
    y_cursor = 0
    for i, tile_path in enumerate(translated_tile_paths):
        with Image.open(tile_path) as tile_im:
            tile_im = tile_im.convert("RGB")
            # Resize defensively in case the translator's render stage changed
            # tile dimensions slightly (e.g. rounding during upscale/cleanup).
            if tile_im.size != (width, heights[i]):
                tile_im = tile_im.resize((width, heights[i]))
            # Trim the top overlap from every tile after the first, since
            # that region was already painted by the previous tile's bottom.
            crop_top = overlap if i > 0 else 0
            visible = tile_im.crop((0, crop_top, width, heights[i]))
            recomposed.paste(visible, (0, y_cursor))
            y_cursor += visible.height

    out_name = manifest_entry["original_name"]
    out_path = os.path.join(translated_dir, out_name)
    recomposed.save(out_path, quality=95)
    for p in translated_tile_paths:
        try:
            os.remove(p)
        except Exception:
            pass
    return out_path

def build_dynamic_system_instruction(cfg):
    system_text = cfg["system_prompt_text"].replace("{target_lang}", cfg["target_lang"])
    return (
        f"{system_text}\n\n"
        f"User Specific Instructions Focus: {cfg['user_prompt_text']}\n"
        f"Produce strictly the final targeted localization stream mapping blocks directly."
    )

def resolve_font_path(cfg):
    if cfg.get("font_name"):
        p = FONTS_DIR / cfg["font_name"]
        if p.exists():
            return str(p.parent)
    return str(FONTS_DIR)

# ================= MangaTranslator CLI Capability Detection =================
# The upstream fork's flag names occasionally shift between versions (e.g.
# --target-language vs --output-language). Rather than hardcoding a guess and
# risking another "unrecognized arguments" crash, we introspect `main.py --help`
# once per process and cache which flags actually exist, then only pass flags
# the installed version supports. This also lets us safely opt in to OSB
# (Outside Speech Bubble) detection when available, which is what most
# scraped Manhwa needs - dialogue/narration/SFX sitting outside bubble shapes
# is invisible to the default bubble-only YOLO detector, which is why some
# pages come back as "no speech bubbles or outside text detected".
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

# ================= Main Pipeline Runner =================
async def execute_manga_pipeline(client, status_msg: Message, user_id: int):
    cfg = get_user_config(user_id)
    queue = pending_files.get(user_id)

    if not queue or not queue["files"]:
        await safe_edit(status_msg, "❌ Error: No files found in queue. Send `/translate` again.")
        active_jobs.pop(user_id, None)
        return

    # Novel content type is text-only prose with no bubbles/panels - running
    # it through the bubble-detection/inpainting/render pipeline would just
    # burn API calls trying to detect speech bubbles that don't exist, and
    # likely produce garbage output. Until a dedicated text-only OCR+translate
    # path is built, we stop here with a clear explanation rather than
    # silently shipping a broken or wasteful result.
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

    # copy selected font into the run's font dir so main.py picks it up
    if cfg.get("font_name"):
        src_font = FONTS_DIR / cfg["font_name"]
        if src_font.exists():
            shutil.copy(src_font, font_dir_for_run)

    total_files = len(files)
    all_translated_outputs = []  # list of (index, output_path) to send in order at the end
    failure_reasons = []  # list of (file_idx, reason_text) for files that produced no output
    resume_from = paused_jobs.pop(user_id, {}).get("stopped_at_file") or 1

    for file_idx, source_message in enumerate(files, start=1):
        if file_idx < resume_from:
            continue  # already completed & sent before pause

        job = active_jobs.get(user_id)
        if job and job["cancel"]:
            await handle_job_cancelled(client, status_msg, user_id, translated_dir)
            return

        input_dir = os.path.join(job_root, f"input_{file_idx:03d}")
        os.makedirs(input_dir, exist_ok=True)

        await safe_edit(status_msg, build_status_text(mode_label, "📥 Downloading payload", file_idx, total_files, 0, 0, 5))
        downloaded_path = await source_message.download(file_name=os.path.join(job_root, f"src_{file_idx:03d}"))

        # Renaming: pyrogram doesn't always preserve/add an extension, so fix it explicitly.
        # - message.document: use its original filename's extension.
        # - message.photo: Telegram compresses photos to JPEG, so force .jpg.
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

        # Extraction based on mode
        await safe_edit(status_msg, build_status_text(mode_label, "📂 Extracting", file_idx, total_files, 0, 0, 15))
        if mode == "archive" or downloaded_path.lower().endswith(('.zip', '.cbz')):
            extract_archive(downloaded_path, input_dir)
        elif mode == "pdf" or downloaded_path.lower().endswith('.pdf'):
            extract_pdf(downloaded_path, input_dir)
        else:
            shutil.move(downloaded_path, os.path.join(input_dir, os.path.basename(downloaded_path)))

        ordered_map, tile_manifest = flatten_and_order(input_dir, content_type=cfg.get("content_type", "manhwa"))
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
            # total_images now reflects tiles-on-disk in input_dir (original tall
            # pages were removed and replaced by their tiles), so re-count for
            # accurate progress reporting during the translation stage below.
            total_images = len([f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS)])

        dynamic_system_instruction = build_dynamic_system_instruction(cfg)

        # Build a per-subprocess environment instead of mutating the shared
        # os.environ of the bot process. With multiple users' jobs running
        # concurrently (asyncio tasks interleave while awaiting the subprocess),
        # writing to os.environ directly meant one user's API key/language
        # could leak into another user's in-flight translation job.
        subprocess_env = os.environ.copy()
        subprocess_env['INPUT_LANG'] = cfg['source_lang']
        subprocess_env['PROVIDER'] = cfg['provider']
        subprocess_env['API_URL'] = cfg['api_url']
        subprocess_env['API_KEY'] = cfg['api_key']
        subprocess_env['MODEL_NAME'] = cfg['model_name']
        subprocess_env['SPECIAL_INS'] = dynamic_system_instruction

        file_translated_dir = os.path.join(translated_dir, f"file_{file_idx:03d}")
        os.makedirs(file_translated_dir, exist_ok=True)

        cmd = [
            "python", "MangaTranslator/main.py",
            "--input", input_dir,
            "--output", file_translated_dir,
            "--batch",
            "--font-dir", font_dir_for_run,
            "--input-language", cfg['source_lang'],
            # Explicitly declared, mapped to the active user config. Without
            # this flag, MangaTranslator silently defaulted its internal
            # target-language tracking to English, which then conflicted with
            # the Roman Hindi/Urdu instruction baked into the prompt and made
            # the model emit BOTH strings ("English || Localized") in one
            # bubble. That bloated the token payload and triggered
            # "Text too large for bubble" render-overflow failures.
            # NOTE: MangaTranslator's actual CLI flag is `--output-language`,
            # not `--target-language` (confirmed against upstream --help).
            "--output-language", cfg['target_lang'],
            "--provider", cfg['provider'],
            "--openai-compatible-url", cfg['api_url'],
            "--openai-compatible-api-key", cfg['api_key'],
            "--model-name", cfg['model_name'],
            "--special-instructions", dynamic_system_instruction
        ]

        # OSB = "Outside Speech Bubble" text pipeline. Manhwa frequently has
        # narration boxes, SFX, and freeform text placed OUTSIDE the drawn
        # bubble shape - the default YOLO bubble detector only looks INSIDE
        # bubbles, so those pages come back as "no speech bubbles or outside
        # text detected" even though they clearly have translatable text.
        # Only added if the user has it enabled AND this installed fork
        # actually supports the flag, to avoid another
        # "unrecognized arguments" crash on forks that don't have it.
        if cfg.get("osb_enabled", True) and cli_supports_flag("--osb-enable"):
            cmd.append("--osb-enable")
            if cli_supports_flag("--osb-font-dir"):
                cmd += ["--osb-font-dir", font_dir_for_run]

        await safe_edit(status_msg, 
            build_status_text(mode_label, "🧠 OCR + Translation running", file_idx, total_files, 0, total_images, 40),
            reply_markup=kb_cancel_only()
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=subprocess_env
            )

            # Poll loop so cancel is responsive while subprocess runs
            while process.returncode is None:
                job = active_jobs.get(user_id)
                if job and job["cancel"]:
                    process.kill()
                    await handle_job_cancelled(client, status_msg, user_id, translated_dir, file_idx, total_files, ordered_map)
                    return
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    done_count = len([f for f in os.listdir(file_translated_dir) if f.lower().endswith(IMAGE_EXTS)]) if os.path.exists(file_translated_dir) else 0
                    pct = 40 + int((done_count / max(total_images, 1)) * 40)
                    await safe_edit(status_msg, 
                        build_status_text(mode_label, "🧠 OCR + Translation running", file_idx, total_files, done_count, total_images, min(pct, 80)),
                        reply_markup=kb_cancel_only()
                    )

            stdout_bytes, stderr_bytes = await process.communicate()
            stdout_text = (stdout_bytes or b"").decode(errors="replace")
            stderr_text = (stderr_bytes or b"").decode(errors="replace")

            # Always dump the engine's own logs to the runner console for full debugging.
            print("----- MangaTranslator stdout -----")
            print(stdout_text)
            print("----- MangaTranslator stderr -----")
            print(stderr_text)

            if process.returncode != 0:
                tail = (stderr_text.strip() or stdout_text.strip() or "no output captured")[-800:]
                await safe_edit(
                    status_msg,
                    f"❌ **Engine exited with error on file {file_idx}/{total_files}** (code {process.returncode}):\n```\n{tail}\n```"
                )
                active_jobs.pop(user_id, None)
                return
        except Exception as exec_err:
            await safe_edit(status_msg, f"❌ Engine error on file {file_idx}/{total_files}: {exec_err}")
            active_jobs.pop(user_id, None)
            return

        await safe_edit(status_msg, build_status_text(mode_label, "📦 Packaging output", file_idx, total_files, total_images, total_images, 90))

        # Verify the OCR/translation engine actually produced output before packaging.
        # Without this check, an empty output folder silently becomes an empty zip
        # that "successfully" sends, making it look like nothing happened.
        produced_files = []
        if os.path.exists(file_translated_dir):
            produced_files = [f for f in os.listdir(file_translated_dir) if f.lower().endswith(IMAGE_EXTS)]

        if not produced_files:
            debug_tail = (stderr_text.strip() or stdout_text.strip() or "no output captured")[-500:]
            failure_reasons.append((file_idx, debug_tail))
            await safe_edit(
                status_msg,
                f"❌ **File {file_idx}/{total_files} produced no output.**\n"
                f"Engine exited cleanly (code 0) but wrote no translated images.\n```\n{debug_tail}\n```\n"
                f"Skipping to next file."
            )
            continue

        # If this file had any tall pages that got tiled before translation,
        # stitch their translated tiles back into single tall pages now,
        # trimming the overlap so text doesn't appear duplicated at the seams.
        if tile_manifest:
            await safe_edit(status_msg, build_status_text(mode_label, "🧵 Recomposing tiled pages", file_idx, total_files, total_images, total_images, 85))
            recompose_failures = []
            for page_idx, manifest_entry in tile_manifest.items():
                result_path = recompose_tiled_page(file_translated_dir, page_idx, manifest_entry)
                if result_path is None:
                    recompose_failures.append(page_idx)
            if recompose_failures:
                await safe_edit(
                    status_msg,
                    f"⚠️ File {file_idx}/{total_files}: {len(recompose_failures)} tiled page(s) "
                    f"had a tile that failed to translate, so those pages may be incomplete. "
                    f"Continuing with the rest."
                )
            # Re-scan produced_files now that tiles have been merged back down
            # into full pages, so packaging sees the final page count, not
            # the intermediate per-tile count.
            produced_files = [f for f in os.listdir(file_translated_dir) if f.lower().endswith(IMAGE_EXTS)]

        try:
            if cfg['output_format'] == 'img':
                # Raw Images mode: send each translated image as its own document,
                # in order, instead of silently zipping them.
                image_files = sorted(
                    [f for f in os.listdir(file_translated_dir) if f.lower().endswith(IMAGE_EXTS)]
                )
                for img_idx, img_name in enumerate(image_files, start=1):
                    send_kwargs = {
                        "chat_id": source_message.chat.id,
                        "document": os.path.join(file_translated_dir, img_name),
                    }
                    if img_idx == 1:
                        send_kwargs["caption"] = f"💥 **File {file_idx}/{total_files} — image {img_idx}/{len(image_files)}**"
                    await client.send_document(**send_kwargs)
                all_translated_outputs.append((file_idx, file_translated_dir))
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
        # "img" (Raw Images) never reaches here - it's handled separately by sending
        # each translated image individually. This fallback only covers unknown formats.
        shutil.make_archive(archive_base, "zip", source_dir)
        return f"{archive_base}.zip"

# ================= Cancel / Pause Handling =================
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
    # Re-invoke the same pipeline; already-completed files were sent, so re-run from remaining queue.
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
