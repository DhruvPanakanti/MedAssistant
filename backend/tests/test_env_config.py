"""
Regression tests for .env support (config_loader.py). Covers the real
bug caught during development: an uncommented-but-blank value (e.g.
ADMIN_USERNAME=) sets the environment variable to an empty string,
which silently overrides os.environ.get(key, default) with "" instead
of the intended default -- breaking the default admin login. The
shipped .env.example keeps every optional line commented out for
exactly this reason.
"""
import os
import subprocess
import sys

from config_loader import PROJECT_ROOT


def test_env_example_exists_and_is_fully_commented():
    """Every optional setting in the committed template must be
    commented out, so a fresh copy never silently breaks a default."""
    path = os.path.join(PROJECT_ROOT, ".env.example")
    assert os.path.exists(path), ".env.example should exist and be committed"
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    assert lines == [], f".env.example has uncommented lines that could override a default: {lines}"


def test_env_file_exists():
    path = os.path.join(PROJECT_ROOT, ".env")
    assert os.path.exists(path), ".env should exist locally so the user has something to edit"


def test_commented_env_does_not_override_defaults():
    """Loading the shipped (fully commented) .env must leave every
    default untouched -- run in a subprocess so we don't pollute this
    test session's already-loaded environment."""
    code = (
        "import os\n"
        "for k in ['GEMINI_API_KEY','ADMIN_USERNAME','ADMIN_PASSWORD','FLASK_SECRET_KEY','GEMINI_MODEL']:\n"
        "    os.environ.pop(k, None)\n"
        "from dotenv import load_dotenv\n"
        f"load_dotenv(r'{os.path.join(PROJECT_ROOT, '.env')}')\n"
        "assert os.environ.get('ADMIN_USERNAME', 'admin') == 'admin'\n"
        "assert os.environ.get('ADMIN_PASSWORD', 'admin123') == 'admin123'\n"
        "assert 'GEMINI_API_KEY' not in os.environ or os.environ['GEMINI_API_KEY'] == ''\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_uncommented_env_value_is_picked_up(tmp_path):
    """Sanity check the opposite case: a real, uncommented key in a
    temp .env file IS picked up correctly."""
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=a-real-looking-test-key\n")

    code = (
        "import os\n"
        "os.environ.pop('GEMINI_API_KEY', None)\n"
        "from dotenv import load_dotenv\n"
        f"load_dotenv(r'{env_file}')\n"
        "assert os.environ.get('GEMINI_API_KEY') == 'a-real-looking-test-key'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
