"""
Standalone diagnostic tool for the Gemini chatbot layer. Run this
directly to find out exactly why the chatbot might be falling back to
rule-based answers instead of using Gemini:

    python check_gemini.py

Checks, in order: .env file present, GEMINI_API_KEY set (and not still
the placeholder), google-genai installed, and finally a real API call
-- reporting the exact point of failure in plain language rather than
a raw traceback.
"""
import os
import sys

from config_loader import PROJECT_ROOT  # importing this loads .env


def main():
    print("=== Gemini Chatbot Connection Check ===\n")

    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        print(f"FAIL: no .env file found at {env_path}")
        print("  -> Copy .env.example to .env, then edit it.")
        return 1
    print(f"OK: .env file found at {env_path}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("FAIL: GEMINI_API_KEY is not set.")
        print("  -> Open .env, find the line starting with '# GEMINI_API_KEY='")
        print("  -> Remove the '#' at the start of that line.")
        print("  -> Replace 'your-key-here' with your real key.")
        print("  -> Save the file and run this script again.")
        return 1
    if api_key.strip() in ("your-key-here", ""):
        print("FAIL: GEMINI_API_KEY is still the placeholder value.")
        print("  -> Get a real key at https://aistudio.google.com/apikey")
        print("  -> Paste it in .env in place of 'your-key-here'.")
        return 1
    masked = api_key[:6] + "..." if len(api_key) > 6 else "***"
    print(f"OK: GEMINI_API_KEY is set ({masked}, {len(api_key)} characters)")

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        print(f"FAIL: the google-genai package isn't installed ({e})")
        print("  -> Run: pip install -r requirements.txt")
        return 1
    print("OK: google-genai package is installed")

    print("\nSending a real test message to Gemini...")
    try:
        client = genai.Client(api_key=api_key)
        model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        response = client.models.generate_content(
            model=model,
            contents="Reply with the single word: OK",
        )
        text = (response.text or "").strip()
        print(f"SUCCESS: Gemini responded: {text!r}")
        print("\nYour chatbot should now use Gemini for general conversation.")
        print("If it's still giving rule-based answers, make sure you're")
        print("running this SAME app.py (not an older copy) and restart it.")
        return 0
    except Exception as e:
        print(f"FAIL: the API call itself failed.\n  {type(e).__name__}: {e}")
        print("\nMost common causes, in order of likelihood:")
        print("  1. The key was revoked or is invalid -> generate a new one")
        print("     at https://aistudio.google.com/apikey and update .env.")
        print("  2. Free tier quota exceeded for today -> wait, or check")
        print("     your quota at https://aistudio.google.com/")
        print("  3. No internet connection, or a firewall/proxy is blocking")
        print("     the request.")
        print("  4. The model name itself is outdated or deprecated --")
        print("     Google retires model names over time (this has")
        print("     already happened once in this project's lifetime).")
        print("     If the error message above mentions a model name,")
        print("     it often tells you the replacement to use directly.")
        print("     Update GEMINI_MODEL in .env (or the default in")
        print("     gemini_chat.py) to whatever it recommends.")
        print("  5. A brand-new model generation this code doesn't know")
        print("     about yet -- gemini_chat.py auto-detects Gemini 2.x")
        print("     vs 3.x to send the right 'thinking' parameter, but a")
        print("     future Gemini 4.x (or similar) may need its own case")
        print("     added to _build_thinking_config() in gemini_chat.py.")
        print("     An error mentioning 'thinking_budget' or")
        print("     'thinking_level' specifically points here.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
