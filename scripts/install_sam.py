#!/usr/bin/env python3
"""
SAMモデル自動セットアップスクリプト
依存関係インストール + モデルダウンロード
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(cmd: list[str]) -> bool:
    """コマンド実行"""
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies():
    """Python依存関係インストール"""
    print("📦 Installing Python dependencies...")
    
    packages = [
        "segment-anything",
        "opencv-python",
        "numpy",
        "pillow",
        "simple-lama-inpainting"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        if not run_command([sys.executable, "-m", "pip", "install", package]):
            print(f"❌ Failed to install {package}")
            return False
    
    print("✅ Dependencies installed")
    return True

def download_sam_model():
    """SAMモデルダウンロード"""
    print("\n🤖 Downloading SAM model (2.4GB)...")
    
    model_dir = Path.home() / ".sam-figma-layer" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "sam_vit_h_4b8939.pth"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return True
    
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    try:
        print(f"   Downloading from {url}...")
        urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
        print(f"\n✅ Model downloaded: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def download_progress(block_num, block_size, total_size):
    """ダウンロード進捗表示"""
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100)
    bar_length = 40
    filled = int(bar_length * downloaded / total_size)
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r   [{bar}] {percent:.1f}%', end='', flush=True)

def verify_installation():
    """インストール確認"""
    print("\n🔍 Verifying installation...")
    
    try:
        import segment_anything
        import cv2
        import numpy as np
        from PIL import Image
        from simple_lama_inpainting import SimpleLama

        print("✅ All Python packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def preload_lama():
    """LaMaモデル事前ダウンロード"""
    print("\n🎨 Preloading LaMa inpainting model...")
    try:
        from simple_lama_inpainting import SimpleLama
        SimpleLama()  # 初回呼び出しでモデル自動DL
        print("✅ LaMa model ready")
        return True
    except Exception as e:
        print(f"❌ LaMa preload failed: {e}")
        return False

def main():
    print("🚀 SAM-Figma-Layer Setup\n")
    
    # 依存関係インストール
    if not install_dependencies():
        print("\n❌ Setup failed at dependencies installation")
        sys.exit(1)
    
    # モデルダウンロード
    if not download_sam_model():
        print("\n❌ Setup failed at model download")
        sys.exit(1)
    
    # 確認
    if not verify_installation():
        print("\n❌ Setup failed at verification")
        sys.exit(1)

    # LaMaモデル事前ダウンロード
    if not preload_lama():
        print("\n❌ Setup failed at LaMa preload")
        sys.exit(1)

    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("  1. npm install")
    print("  2. npm run build")
    print("  3. Add to Claude Code MCP config")

if __name__ == "__main__":
    main()
