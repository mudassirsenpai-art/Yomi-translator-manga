import os
import shutil
import zipfile
import asyncio
import subprocess
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery

# ================= Configuration & Secrets =================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AUTH_USERS_STR = os.environ.get("AUTHORIZED_USERS", "")
AUTHORIZED_USERS = [int(u.strip()) for u in AUTH_USERS_STR.split(",") if u.strip().isdigit()]

app = Client("YomiTranslatorBot", bot_token=BOT_TOKEN, api_id=123456, api_hash="PUT_YOUR_API_HASH_HERE") # API ID/Hash add karein

# ================= In-Memory User Settings =================
user_settings = {}

def get_default_settings(user_id):
    if user_id not in user_settings:
        user_settings[user_id] = {
            "source_lang": "auto",
            "target_lang": "Roman hindi",
            "font_path": "/content/Cleanow.ttf",
            "provider": "OpenAI-Compatible",
            "api_url": "",
            "api_key": "",
            "model_name": "gemini-3-flash-preview",
            "output_format": "zip",
            "custom_prompt": "You are a localization expert. Translate text preserving structural flow without using generic examples. Maintain tone and format strictly based on the source context."
        }
    return user_settings[user_id]

# ================= Auth Filter =================
async def auth_check(_, __, message):
    return message.from_user.id in AUTHORIZED_USERS
auth_filter = filters.create(auth_check)

# ================= Commands =================
@app.on_message(filters.command("start") & auth_filter)
async def start_cmd(client, message: Message):
    await message.reply_text("✅ Bot is alive! Welcome Senpai.\nSend a ZIP, CBZ, PDF, or Image to begin. Use /settings to configure API and Languages.")

@app.on_message(filters.command("settings") & auth_filter)
async def settings_cmd(client, message: Message):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🌐 Target Language", callback_data="set_lang"), InlineKeyboardButton("🔡 Fonts Library", callback_data="set_font")],
        [InlineKeyboardButton("⚙️ API Config", callback_data="set_api"), InlineKeyboardButton("📝 Prompts", callback_data="set_prompt")],
        [InlineKeyboardButton("📦 Output Format", callback_data="set_output")]
    ])
    await message.reply_text("🛠 **Master Settings Menu:**\nSelect an option to configure:", reply_markup=keyboard)

# ================= Callback Handlers (Settings UI) =================
@app.on_callback_query(auth_filter)
async def handle_callbacks(client, query: CallbackQuery):
    data = query.data
    user_id = query.from_user.id
    settings = get_default_settings(user_id)

    if data == "set_lang":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("English", callback_data="lang_en"), InlineKeyboardButton("Roman Hindi", callback_data="lang_roman_hindi")],
            [InlineKeyboardButton("Custom Language", callback_data="lang_custom")]
        ])
        await query.message.edit_text("Select Target Language:", reply_markup=kb)
    
    elif data == "lang_custom":
        await query.message.reply_text("Reply to this message with your custom language name:")
        # Here you would typically use a conversation handler (like Pyromod) to catch the reply
        await query.answer()

    elif data == "start_translation":
        await query.message.edit_text("🔄 Initiating pipeline...")
        asyncio.create_task(run_translation_pipeline(client, query.message, user_id))

# ================= Processing Pipeline =================
@app.on_message(filters.document | filters.photo & auth_filter)
async def receive_files(client, message: Message):
    # Ask for confirmation first
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Continue Translation", callback_data="start_translation")]])
    await message.reply_text("File received! Want to translate? Click continue.", reply_markup=kb)
    # Note: Save the file_id/path in a temp dictionary to process upon button click

async def run_translation_pipeline(client, message: Message, user_id: int):
    settings = get_default_settings(user_id)
    
    # 1. Update UI Status
    status_msg = await message.edit_text(
        "📊 **Translation Status**\n"
        "Total Images: Calculating...\n"
        "Status: ⏳ Preparing directories..."
    )

    # Core Directories
    BASE_DIR = "/content" # Or standard temp dir in GitHub actions
    input_dir = os.path.join(BASE_DIR, "input_manga")
    translated_dir = os.path.join(BASE_DIR, "translated_manga")
    font_dir = os.path.join(BASE_DIR, "manga_fonts")
    
    # Clean folders
    for d in [input_dir, translated_dir, font_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # (Assume file is downloaded and extracted here in correct order)
    image_count = len(os.listdir(input_dir)) if os.path.exists(input_dir) else 20 # Dummy value for logic

    await status_msg.edit_text(
        f"📊 **Translation Status**\n"
        f"Total Images: {image_count}\n"
        f"Status: ⚙️ OCR Running & Translation in progress...\n"
        f"Progress: [▓▓▓░░░░░░░] 30%"
    )

    # 2. Dynamic System Instruction (No Examples)
    system_instruction = f"""You are a professional manga localization expert.
Your goal is to translate the dialogue into natural, casual, and street-style spoken {settings['target_lang']}.
Strict Rules:
1. Localize for structural flow.
2. Keep familiar relational terms and honorifics.
3. Restrict aggressive pronouns unless demanded by friction.
4. Ensure text lengths fit seamlessly.
User Prompt: {settings['custom_prompt']}"""

    # 3. Subprocess Execution Setup
    os.environ['INPUT_LANG'] = settings['source_lang']
    os.environ['PROVIDER'] = settings['provider']
    os.environ['API_URL'] = settings['api_url']
    os.environ['API_KEY'] = settings['api_key']
    os.environ['MODEL_NAME'] = settings['model_name']
    os.environ['SPECIAL_INS'] = system_instruction

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
        "--special-instructions", system_instruction
    ]

    # Run the background process
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate() # In a full version, read stdout line-by-line to update the status_msg dynamically
    except Exception as e:
        await status_msg.edit_text(f"❌ Error in execution: {e}")
        return

    # 4. Finalizing Output
    await status_msg.edit_text(
        f"📊 **Translation Status**\n"
        f"Total Images: {image_count}\n"
        f"Status: ✅ Completed!\n"
        f"Progress: [▓▓▓▓▓▓▓▓▓▓] 100%\n"
        f"Bundling {settings['output_format']}..."
    )
    
    # Create ZIP and send (Placeholder path)
    output_zip = f"{BASE_DIR}/Output.{settings['output_format']}"
    shutil.make_archive(output_zip.replace('.zip', ''), 'zip', translated_dir)
    await client.send_document(message.chat.id, document=output_zip, caption="💥 Here is your processed manga!")
    await status_msg.delete()

if __name__ == "__main__":
    app.run()
