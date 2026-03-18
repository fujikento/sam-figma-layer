#!/usr/bin/env python3
"""
SAM自動セグメンテーションスクリプト
Node.jsから呼び出し可能
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class SAMSegmenter:
    def __init__(self, checkpoint_path: str):
        """SAMモデル初期化"""
        print(f"Loading SAM model from {checkpoint_path}...", file=sys.stderr)
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        print("SAM model loaded", file=sys.stderr)
    
    def segment_image(self, image_path: str, output_dir: str) -> dict:
        """
        画像を複数のセグメントに分離
        
        Returns:
            {
                "layers": [...],
                "classification": {...},
                "metadata": {...}
            }
        """
        # 画像読み込み
        print(f"Reading image: {image_path}", file=sys.stderr)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # セグメンテーション実行
        print("Running SAM segmentation...", file=sys.stderr)
        masks = self.mask_generator.generate(image_rgb)
        print(f"Found {len(masks)} segments", file=sys.stderr)
        
        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        layers = []
        
        # 各マスクを個別レイヤーとして保存
        for idx, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            bbox = mask_data["bbox"]  # [x, y, width, height]
            area = mask_data["area"]
            
            # マスク適用: 透過PNG作成
            result = image_rgb.copy()
            result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
            result[:, :, 3] = (mask * 255).astype(np.uint8)
            
            # バウンディングボックスでクロップ（効率化）
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            cropped = result[y:y+h, x:x+w]
            
            # 保存
            layer_filename = f"layer_{idx:03d}.png"
            layer_path = output_path / layer_filename
            cv2.imwrite(str(layer_path), cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA))
            
            layers.append({
                "id": idx,
                "filename": layer_filename,
                "path": str(layer_path),
                "bbox": bbox,
                "area": int(area),
                "center": [int(x + w/2), int(y + h/2)]
            })
            
            print(f"  Layer {idx}: {layer_filename} ({w}x{h}, area={area})", file=sys.stderr)
        
        # レイヤー分類
        classification = self.classify_layers(layers, image_rgb.shape)
        
        # メタデータ
        metadata = {
            "source_image": image_path,
            "image_size": {
                "width": image_rgb.shape[1],
                "height": image_rgb.shape[0]
            },
            "total_layers": len(layers),
            "layers": layers,
            "classification": classification
        }
        
        # metadata.json保存
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved: {metadata_path}", file=sys.stderr)
        
        return metadata
    
    def classify_layers(self, layers: list, image_shape: tuple) -> dict:
        """
        レイヤーを自動分類
        
        - background: 最大面積レイヤー
        - foreground_objects: 5%以上のレイヤー
        - small_elements: 5%未満のレイヤー
        """
        if not layers:
            return {
                "background": None,
                "foreground_objects": [],
                "small_elements": []
            }
        
        # 画像全体の面積
        total_image_area = image_shape[0] * image_shape[1]
        
        # 面積で降順ソート
        sorted_layers = sorted(layers, key=lambda x: x["area"], reverse=True)
        
        background = sorted_layers[0]["id"]
        foreground = []
        small_elements = []
        
        for layer in sorted_layers[1:]:
            area_ratio = layer["area"] / total_image_area
            
            if area_ratio > 0.05:  # 全体の5%以上 → 前景オブジェクト
                foreground.append(layer["id"])
            else:  # 5%未満 → 小要素（ボタン、アイコン等）
                small_elements.append(layer["id"])
        
        return {
            "background": background,
            "foreground_objects": foreground,
            "small_elements": small_elements
        }

def main():
    if len(sys.argv) != 4:
        print("Usage: python sam_segmenter.py <checkpoint_path> <image_path> <output_dir>", file=sys.stderr)
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    try:
        segmenter = SAMSegmenter(checkpoint_path)
        result = segmenter.segment_image(image_path, output_dir)
        
        # 結果をJSON出力（Node.jsが読み取る）
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
