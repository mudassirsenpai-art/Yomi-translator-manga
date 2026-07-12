import os
import sys
import shutil
import zipfile
import asyncio
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery

# ================= Configuration & Secrets =================
# Ab aapki API keys aur token automatic GitHub environment se link ho jayenge
API_ID = int(os.environ.get("API_ID", 0))
API_HASH = os.environ.get("API_HASH", "")
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AUTH_USERS_STR = os.environ.get("AUTHORIZED_USERS", "")

# Validation check to prevent runtime loops
if not API_ID or not API_HASH or not BOT_TOKEN:
    print("❌ Error: API_ID, API_HASH, ya BOT_TOKEN settings repository secrets mein missing hain!")
    sys.exit(1)

AUTHORIZED_USERS = [int(u.strip()) for u in AUTH_USERS_STR.split(",") if u.strip().isdigit()]

app = Client("YomiTranslatorBot", bot_token=BOT_TOKEN, api_id=API_ID, api_hash=API_HASH)

# ================= In-Memory State & Templates =================
user_settings = {}
pending_files = {}  # Dynamic file queue management system

def get_user_config(user_id):
    if user_id not in user_settings:
        user_settings[user_id] = {
            "source_lang": "auto",
            "target_lang": "Roman hindi",
            "font_path": "/content/Cleanow.ttf",
            "provider": "OpenAI-Compatible",
            "api_url": "https://api.highwayapi.ai/openai",
            "api_key": "sk__cPTi97PRvIsMtBG9A7X36Pr59lPdw3-uunCT1QtyKk",
            "model_name": "gemini-3-flash-preview",
            "output_format": "zip",
            "custom_prompt": "Standard clean workflow processing baseline."
        }
    return user_settings[user_id]

# ================= Auth Gate Filter =================
async def auth_check(_, __, message):
    return message.from_user.id in AUTHORIZED_USERS
auth_filter = filters.create(auth_check)

# ================= Base Logic Commands =================
@app.on_message(filters.command("start") & auth_filter)
async def start_cmd(client, message: Message):
    await message.reply_text(
        "⚡ **Yomi Subs Core Engine Online** ⚡\n\n"
        "Welcome back, Senpai! Bot running in continuous automated lifecycle.\n"
        "👉 Send any Manga/Manhwa Image, ZIP, CBZ, or PDF to begin processing.\n"
        "⚙️ Use /settings to change backend system parameters dynamically."
    )

@app.on_message(filters.command("settings") & auth_filter)
async def settings_cmd(client, message: Message):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🌐 Input/Target Lang", callback_data="set_lang"), InlineKeyboardButton("🔡 Fonts Track", callback_data="set_font")],
        [InlineKeyboardButton("⚙️ Provider & LLM", callback_data="set_api"), InlineKeyboardButton("📝 System Prompt", callback_data="set_prompt")],
        [InlineKeyboardButton("📦 Output Extension", callback_data="set_output")]
    ])
    await message.reply_text("🛠 **Master Configuration Terminal:**\nTweak structural backend metrics below:", reply_markup=kb)

# ================= Interactive Callbacks Panel =================
@app.on_callback_query(auth_filter)
async def handle_callbacks(client, query: CallbackQuery):
    data = query.data
    user_id = query.from_user.id
    settings = get_user_config(user_id)

    if data == "set_lang":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Source: Auto", callback_data="src_auto"), InlineKeyboardButton("Source: Korean", callback_data="src_ko")],
            [InlineKeyboardButton("Target: Roman Hindi", callback_data="tgt_rh"), InlineKeyboardButton("Custom Target", callback_data="tgt_custom")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
        ])
        await query.message.edit_text(f"🌐 **Language Configuration Layer:**\nCurrent Source: `{settings['source_lang']}`\nCurrent Target: `{settings['target_lang']}`", reply_markup=kb)
    
    elif data.startswith("src_"):
        settings["source_lang"] = data.split("_")[1]
        await query.answer(f"Source Language set to {settings['source_lang']}!")
        await handle_callbacks(client, query) # Refresh view

    elif data == "set_output":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("ZIP Package", callback_data="out_zip"), InlineKeyboardButton("CBZ Archive", callback_data="out_cbz")],
            [InlineKeyboardButton("PDF Document", callback_data="out_pdf"), InlineKeyboardButton("Raw Images", callback_data="out_img")],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
        ])
        await query.message.edit_text(f"📦 **Output Extension Controller:**\nCurrent Format: `{settings['output_format']}`", reply_markup=kb)

    elif data.startswith("out_"):
        settings["output_format"] = data.split("_")[1]
        await query.answer(f"Output changed to .{settings['output_format']}!")
        await settings_cmd(client, query.message)

    elif data == "main_menu":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 Input/Target Lang", callback_data="set_lang"), InlineKeyboardButton("🔡 Fonts Track", callback_data="set_font")],
            [InlineKeyboardButton("⚙️ Provider & LLM", callback_data="set_api"), InlineKeyboardButton("📝 System Prompt", callback_data="set_prompt")],
            [InlineKeyboardButton("📦 Output Extension", callback_data="set_output")]
        ])
        await query.message.edit_text("🛠 **Master Configuration Terminal:**\nTweak structural backend metrics below:", reply_markup=kb)

    elif data == "start_pipeline":
        await query.message.edit_text("🔄 Connection pooled. Spawning background shell automation wrapper...")
        asyncio.create_task(execute_manga_pipeline(client, query.message, user_id))

# ================= Inbound Data Ingestion Guard =================
@app.on_message((filters.document | filters.photo) & auth_filter)
async def receive_files(client, message: Message):
    user_id = message.from_user.id
    pending_files[user_id] = message
    
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Continue & Execute", callback_data="start_pipeline")]])
    await message.reply_text("📋 File verified in transmission queue. Want to run translation?", reply_markup=kb)

# ================= Core Subprocess Pipeline Execution =================
async def execute_manga_pipeline(client, status_msg: Message, user_id: int):
    settings = get_user_config(user_id)
    source_message = pending_files.get(user_id)
    
    if not source_message:
        await status_msg.edit_text("❌ Error: Core package timeline detached. Transfer files again.")
        return

    # Phase 1: Localizing Environment Environment
    await status_msg.edit_text("📊 **Translation Status:**\nTotal Images: Syncing...\nStatus: ⏳ Setting up staging directories...")
    
    BASE_STAGING = "/content"
    input_dir = os.path.join(BASE_STAGING, "input_manga")
    translated_dir = os.path.join(BASE_STAGING, "translated_manga")
    font_dir = os.path.join(BASE_STAGING, "manga_fonts")

    for folder in [input_dir, translated_dir, font_dir]:
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # Downloading Payload
    await status_msg.edit_text("📊 **Translation Status:**\nStatus: 📥 Fetching input binary payloads from Telegram clouds...")
    downloaded_path = await source_message.download()

    # Asset Extraction Layout Controller (ZIP/CBZ handling)
    if downloaded_path.lower().endswith(('.zip', '.cbz')):
        with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
            zip_ref.extractall(input_dir)
        os.remove(downloaded_path)
    elif downloaded_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        shutil.move(downloaded_path, os.path.join(input_dir, os.path.basename(downloaded_path)))

    # Standardizing File Indices Sorting Hierarchy
    # Unflatten tree logic execution to bring raw subfolders items into primary base path
    for root, _, files in os.walk(input_dir, topdown=False):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                shutil.move(os.path.join(root, f), os.path.join(input_dir, f))
        if root != input_dir:
            try: os.rmdir(root)
            except: pass

    total_images = len([x for x in os.listdir(input_dir) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    
    if total_images == 0:
        await status_msg.edit_text("❌ Error: No valid raster frames isolated inside package stream.")
        return

    # Phase 2: Dynamic System Prompt Architecture (No hardcoded language examples)
    dynamic_system_instruction = (
        f"You are a professional multi-language manga and comic localization engine.\n"
        f"Your task is to adapt the source script lines into high-fidelity, smooth, natural spoken {settings['target_lang']} dialogue structure.\n\n"
        f"Operational Constraints:\n"
        f"1. Contextual Structural Alignment: Localize structural layout flows organically rather than serving a rigid word-for-word interpretation.\n"
        f"2. Preservation Matrix: Retain relational honorifics, names, and titles natively embedded in the source frames if they provide contextual narrative immersion.\n"
        f"3. Balanced Pacing: Limit overly aggressive, archaic or excessively modern slang variations unless explicitly demanded by scene gravity.\n"
        f"4. Volumetric Bounds: Maintain output textual block scale sizing matching original text frame bubbles to prevent clean clipping layout overlap errors.\n\n"
        f"User Specific Instructions Focus: {settings['custom_prompt']}\n"
        f"Produce strictly the final targeted localization stream mapping blocks directly."
    )

    # Phase 3: Trigger Shell Process Interface Execution
    await status_msg.edit_text(
        f"📊 **Translation Status:**\n"
        f"Total Images: `{total_images}`\n"
        f"Status: ⚙️ YOLO & Manga-OCR Engine parsing elements...\n"
        f"Progress: [▓▓▓░░░░░░░] 30%"
    )

    # Injection of system run environments parameters
    os.environ['INPUT_LANG'] = settings['source_lang']
    os.environ['PROVIDER'] = settings['provider']
    os.environ['API_URL'] = settings['api_url']
    os.environ['API_KEY'] = settings['api_key']
    os.environ['MODEL_NAME'] = settings['model_name']
    os.environ['SPECIAL_INS'] = dynamic_system_instruction

    cmd = [
        "python", "MangaTranslator/main.py",
        "--input", input_dir,
        "--output", translated_dir,
        "--batch",
        "--font-dir", font_dir,
        "--input-language", settings['source_lang'],
        "--provider", settings['provider'],
        "--openai-compatible-url", settings['api_url'],
        "--openai-compatible-api-key", settings['api_key'],
        "--model-name", settings['model_name'],
        "--special-instructions", dynamic_system_instruction
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await status_msg.edit_text(
            f"📊 **Translation Status:**\n"
            f"Total Images: `{total_images}`\n"
            f"Status: 🧠 LLM Translator processing tokens stream... \n"
            f"Progress: [▓▓▓▓▓▓▓░░░] 70%"
        )
        await process.communicate()
    except Exception as exec_err:
        await status_msg.edit_text(f"❌ Core Engine Runtime Panic Break: {exec_err}")
        return

    # Phase 4: Bundling Output Formats Execution
    await status_msg.edit_text(
        f"📊 **Translation Status:**\n"
        f"Total Images: `{total_images}`\n"
        f"Status: 📦 Rendering output `{settings['output_format']}` package framework container...\n"
        f"Progress: [▓▓▓▓▓▓▓▓▓░] 90%"
    )

    final_archive_path = f"{BASE_STAGING}/YomiSubs_Output"
    output_format = settings['output_format']

    if output_format in ["zip", "cbz"]:
        archive_ext = "zip"
        shutil.make_archive(final_archive_path, archive_ext, translated_dir)
        payload_file = f"{final_archive_path}.{archive_ext}"
        if output_format == "cbz":
            cbz_payload = payload_file.replace('.zip', '.cbz')
            os.rename(payload_file, cbz_payload)
            payload_file = cbz_payload
    else:
        # Default fallback transmission structure block if non-archived format
        shutil.make_archive(final_archive_path, "zip", translated_dir)
        payload_file = f"{final_archive_path}.zip"

    # Final Verification and transmission dispatch back to master chat thread
    if os.path.exists(payload_file):
        await client.send_document(
            source_message.chat.id, 
            document=payload_file, 
            caption=f"💥 **Localization Lifecycle Completed Successfully!**\n📦 Format: `.{output_format.upper()}`\n🖼 Processed Frames: `{total_images}`\n\nEnjoy Senpai! Built with precision."
        )
        os.remove(payload_file)
    else:
        await source_message.reply_text("❌ Output asset extraction pipeline compiled blank. Check layout tracking configurations.")
        
    await status_msg.delete()
    pending_files.pop(user_id, None)

if __name__ == "__main__":
    app.run()
