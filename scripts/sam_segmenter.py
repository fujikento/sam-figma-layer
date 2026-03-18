#!/usr/bin/env python3
"""
SAM自動セグメンテーション + LaMa背景inpainting
Node.jsから呼び出し可能
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from simple_lama_inpainting import SimpleLama

class SAMSegmenter:
    def __init__(self, checkpoint_path: str):
        """SAM + LaMaモデル初期化"""
        print(f"Loading SAM model from {checkpoint_path}...", file=sys.stderr)
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        print("SAM model loaded", file=sys.stderr)

        print("Loading LaMa inpainting model...", file=sys.stderr)
        self.lama = SimpleLama()
        print("LaMa model loaded", file=sys.stderr)

    def segment_image(self, image_path: str, output_dir: str) -> dict:
        """画像を複数のセグメントに分離し、背景をinpaintする"""
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

            # バウンディングボックスでクロップ
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

        # 背景レイヤーをLaMaでinpaint（前景の穴を補完）
        bg_id = classification["background"]
        if bg_id is not None and (classification["foreground_objects"] or classification["small_elements"]):
            print("Inpainting background with LaMa...", file=sys.stderr)
            inpainted_bg = self.inpaint_background(image_rgb, masks, classification)

            # 穴なしフル画像で上書き保存
            bg_layer = layers[bg_id]
            bg_path = output_path / bg_layer["filename"]
            cv2.imwrite(str(bg_path), inpainted_bg)

            # bboxを画像全体に更新（Figmaで(0,0)フルサイズ配置）
            bg_layer["bbox"] = [0, 0, int(image_rgb.shape[1]), int(image_rgb.shape[0])]
            bg_layer["area"] = int(image_rgb.shape[0] * image_rgb.shape[1])
            bg_layer["center"] = [int(image_rgb.shape[1] // 2), int(image_rgb.shape[0] // 2)]
            bg_layer["inpainted"] = True

            print(f"  Background layer {bg_id} inpainted at full size", file=sys.stderr)

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

    def inpaint_background(self, image_rgb, masks, classification) -> np.ndarray:
        """前景マスクを合成し、LaMaで背景の穴を補完する"""
        # 前景+小要素のマスクを合成 → 穴マスク
        holes_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        non_bg = classification["foreground_objects"] + classification["small_elements"]
        for idx in non_bg:
            seg = masks[idx]["segmentation"]
            holes_mask = np.bitwise_or(holes_mask, (seg * 255).astype(np.uint8))

        # マスクを膨張してエッジの残像を消す
        kernel = np.ones((5, 5), np.uint8)
        holes_mask = cv2.dilate(holes_mask, kernel, iterations=2)

        # LaMa inpainting
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(holes_mask)
        result_pil = self.lama(image_pil, mask_pil)

        # numpy BGRA に変換（完全不透明）
        result_rgb = np.array(result_pil)
        result_bgra = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGRA)
        result_bgra[:, :, 3] = 255
        return result_bgra

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
