from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "scripts" / "prepare.sh").read_text(encoding="utf-8")


def test_prepare_checks_python_and_pip_before_downloading() -> None:
    python_check = SCRIPT.index("for candidate in python3 python")
    pip_check = SCRIPT.index('"${python_bin}" -m pip --version')
    dependency_download = SCRIPT.index("download_open_source_deps.sh")

    assert 'command -v "${candidate}"' in SCRIPT
    assert "Python with pip is required" in SCRIPT
    assert python_check < pip_check < dependency_download


def test_prepare_installs_pypi_mermaid_and_playwright_chromium() -> None:
    pip_install = SCRIPT.index('"${python_bin}" -m pip install mermaid-cli')
    browser_install = SCRIPT.index(
        '"${python_bin}" -m playwright install chromium'
    )

    assert pip_install < browser_install
    assert "pip install mermaid-cli failed" in SCRIPT
    assert "Playwright Chromium installation failed" in SCRIPT
    assert "pip install --user" not in SCRIPT


def test_prepare_validates_the_same_python_environment() -> None:
    assert 'sysconfig.get_path("scripts")' in SCRIPT
    assert 'mermaid_cli_bin="${mermaid_scripts_dir}/mmdc"' in SCRIPT
    assert '"${mermaid_cli_bin}" --help' in SCRIPT
    assert "--playwright-config-file" in SCRIPT
    assert "from playwright.sync_api import sync_playwright" in SCRIPT
    assert "playwright.chromium.launch" in SCRIPT
    assert '"--no-sandbox"' in SCRIPT
    assert '"--disable-setuid-sandbox"' in SCRIPT
