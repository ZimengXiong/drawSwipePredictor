#!/usr/bin/env python3
"""Build a macOS app that opens the memory recognizer without a terminal."""

from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("For other platforms, run cargo run --release --bin swipe-memory")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "swipe-desktop", "--no-default-features", "--bin", "swipe-memory"],
        cwd=ROOT, check=True,
    )
    contents = ROOT / "build/Draw from Memory.app/Contents"
    executable_dir = contents / "MacOS"
    executable_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "target/release/swipe-memory", executable_dir / "swipe-memory")
    with (contents / "Info.plist").open("wb") as target:
        plistlib.dump({
            "CFBundleExecutable": "swipe-memory",
            "CFBundleIdentifier": "com.zimeng.draw-swipe-memory",
            "CFBundleName": "Draw from Memory",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        }, target)
    print(contents.parent)


if __name__ == "__main__":
    main()
