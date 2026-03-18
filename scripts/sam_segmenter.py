#!/usr/bin/env python3
"""
SAM自動セグメンテーション + LaMa背景inpainting
重複マスクの排除、エッジ品質の改善、適切なinpainting範囲の制御
"""

import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from simple_lama_inpainting.models.model import download_model, LAMA_MODEL_URL, prepare_img_and_mask
import pytoshop
from pytoshop import layers as psd_layers, enums as psd_enums


class SAMSegmenter:
    def __init__(self, checkpoint_path: str):
        """SAM + LaMaモデル初期化"""
        print(f"Loading SAM model from {checkpoint_path}...", file=sys.stderr)
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            # 品質重視の設定
            pred_iou_thresh=0.88,        # IoU閾値を上げてノイズマスク除去
            stability_score_thresh=0.95,  # 安定スコア閾値を上げる
            min_mask_region_area=500,     # 最小マスク面積
        )
        print("SAM model loaded", file=sys.stderr)

        print("Loading LaMa inpainting model...", file=sys.stderr)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model_path = download_model(LAMA_MODEL_URL)
        self.lama_model = torch.jit.load(model_path, map_location=device)
        self.lama_model.eval()
        self.lama_model.to(device)
        self.lama_device = device
        print(f"LaMa model loaded on {device}", file=sys.stderr)

    def segment_image(self, image_path: str, output_dir: str) -> dict:
        """画像をセグメント分離し、背景をinpaintする"""
        print(f"Reading image: {image_path}", file=sys.stderr)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # SAMセグメンテーション
        print("Running SAM segmentation...", file=sys.stderr)
        raw_masks = self.mask_generator.generate(image_rgb)
        print(f"Found {len(raw_masks)} raw segments", file=sys.stderr)

        # 重複排除: 各ピクセルを最も具体的な（最小の）マスクに割り当て
        exclusive_masks = self.resolve_overlaps(raw_masks, h, w)
        print(f"After de-overlap: {len(exclusive_masks)} non-overlapping layers", file=sys.stderr)

        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        layers = []
        # マスクIDからレイヤーIDへのマッピング（スパースIDを連番に変換）
        mask_by_layer_id = {}

        for mask_idx, mask_info in enumerate(exclusive_masks):
            mask = mask_info["mask"]
            bbox = mask_info["bbox"]
            area = int(mask.sum())

            if area < 100:
                continue

            layer_id = len(layers)  # 連番ID（リストインデックスと一致）

            refined_alpha = self.refine_mask_edges(mask, image_rgb)

            result = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
            result[:, :, 3] = refined_alpha

            x, y, bw, bh = bbox
            cropped = result[y:y+bh, x:x+bw]

            layer_filename = f"layer_{layer_id:03d}.png"
            layer_path = output_path / layer_filename
            cv2.imwrite(str(layer_path), cropped)

            layers.append({
                "id": layer_id,
                "filename": layer_filename,
                "path": str(layer_path),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": area,
                "center": [int(x + bw // 2), int(y + bh // 2)]
            })
            mask_by_layer_id[layer_id] = mask_info["mask"]

            print(f"  Layer {layer_id}: {layer_filename} ({bw}x{bh}, area={area})", file=sys.stderr)

        # レイヤー分類
        classification = self.classify_layers(layers, (h, w, 3))

        # 背景レイヤーをLaMaでinpaint
        bg_id = classification["background"]
        fg_ids = classification["foreground_objects"]
        if bg_id is not None and fg_ids:
            print("Inpainting background with LaMa...", file=sys.stderr)

            fg_masks = [mask_by_layer_id[fid] for fid in fg_ids if fid in mask_by_layer_id]

            if fg_masks:
                inpainted_bg = self.inpaint_background(image_rgb, fg_masks)

                bg_layer = layers[bg_id]
                bg_path = output_path / bg_layer["filename"]
                cv2.imwrite(str(bg_path), inpainted_bg)

                bg_layer["bbox"] = [0, 0, w, h]
                bg_layer["area"] = w * h
                bg_layer["center"] = [w // 2, h // 2]
                bg_layer["inpainted"] = True

                print(f"  Background layer {bg_id} inpainted at full size ({w}x{h})", file=sys.stderr)

        # メタデータ
        metadata = {
            "source_image": image_path,
            "image_size": {"width": w, "height": h},
            "total_layers": len(layers),
            "layers": layers,
            "classification": classification
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved: {metadata_path}", file=sys.stderr)
        return metadata

    def resolve_overlaps(self, raw_masks: list, h: int, w: int) -> list:
        """
        重複マスクを排除し、各ピクセルを1つのレイヤーにだけ割り当てる。
        小さいマスクを優先（より具体的なオブジェクトを保護）。
        """
        # 面積順にソート（小→大）
        sorted_masks = sorted(raw_masks, key=lambda m: m["area"])

        # ピクセル割り当てマップ: 各ピクセルがどのマスクに属するか
        assignment = np.full((h, w), -1, dtype=np.int32)

        exclusive = []
        for i, mask_data in enumerate(sorted_masks):
            seg = mask_data["segmentation"].astype(bool)

            # まだ未割り当てのピクセルだけこのマスクに割り当て
            available = (assignment == -1)
            exclusive_seg = seg & available

            area = int(exclusive_seg.sum())
            if area < 100:
                continue

            # 割り当て記録
            assignment[exclusive_seg] = len(exclusive)

            # bboxを再計算（排他マスクに基づいて）
            rows = np.any(exclusive_seg, axis=1)
            cols = np.any(exclusive_seg, axis=0)
            if not rows.any():
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox = [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]

            exclusive.append({
                "mask": exclusive_seg,
                "bbox": bbox,
                "area": area,
                "original_area": mask_data["area"],
            })

        # 未割り当てピクセルを「背景余白」として最大マスクに追加
        # （背景は後でinpaintされるのでここでは追加しない）

        return exclusive

    def refine_mask_edges(self, mask: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
        """
        マスク境界を滑らかにする。
        - 1px GaussianBlurでギザギザ除去（アンチエイリアス）
        - 閾値で再バイナリ化してシャープさを維持
        - エッジ付近のみ微妙なグラデーション
        """
        mask_u8 = (mask.astype(np.float32) * 255).astype(np.uint8)

        # 境界だけ検出（エッジから3px以内）
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        edge_band = dilated - eroded  # 境界帯のみ

        # 境界帯にだけ軽いブラーをかける
        blurred = cv2.GaussianBlur(mask_u8, (3, 3), 0.8)

        # 内部はシャープ(255)、外部は透明(0)、エッジだけブラー値
        alpha = mask_u8.copy()
        edge_pixels = edge_band > 0
        alpha[edge_pixels] = blurred[edge_pixels]

        return alpha

    def inpaint_background(self, image_rgb: np.ndarray, fg_masks: list) -> np.ndarray:
        """前景オブジェクトのマスクのみでLaMa inpainting"""
        h, w = image_rgb.shape[:2]

        # 前景マスクの合成
        holes_mask = np.zeros((h, w), dtype=np.uint8)
        for fm in fg_masks:
            holes_mask = np.bitwise_or(holes_mask, (fm.astype(np.uint8) * 255))

        # 軽い膨張でエッジの残像を消す（3x3カーネル、1回だけ）
        kernel = np.ones((3, 3), np.uint8)
        holes_mask = cv2.dilate(holes_mask, kernel, iterations=1)

        hole_pct = (holes_mask > 0).sum() / (h * w) * 100
        print(f"  Inpainting {hole_pct:.1f}% of image", file=sys.stderr)

        # LaMa inpainting
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(holes_mask)

        img_tensor, mask_tensor = prepare_img_and_mask(image_pil, mask_pil, self.lama_device)

        with torch.inference_mode():
            inpainted = self.lama_model(img_tensor, mask_tensor)
            result_np = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)

        # パディングで変わったサイズを元に戻す
        result_np = cv2.resize(result_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # BGRA（完全不透明）
        result_bgra = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGRA)
        result_bgra[:, :, 3] = 255
        return result_bgra

    def classify_layers(self, layers: list, image_shape: tuple) -> dict:
        """
        レイヤーを背景/前景/小要素に分類。

        前景オブジェクト判定条件:
        - 面積が画像の3%以上
        - 画像全幅/全高にまたがるストリップ状ではない（背景の一部と判定）
        """
        if not layers:
            return {"background": None, "foreground_objects": [], "small_elements": []}

        h, w = image_shape[0], image_shape[1]
        total_area = h * w
        sorted_layers = sorted(layers, key=lambda x: x["area"], reverse=True)

        background = sorted_layers[0]["id"]
        foreground = []
        small_elements = []

        for layer in sorted_layers[1:]:
            ratio = layer["area"] / total_area
            bx, by, bw, bh = layer["bbox"]
            width_ratio = bw / w
            height_ratio = bh / h

            # 画像全幅/全高の80%以上に広がるストリップは背景の一部
            is_strip = width_ratio > 0.8 or height_ratio > 0.8

            if ratio > 0.03 and not is_strip:
                foreground.append(layer["id"])
            else:
                small_elements.append(layer["id"])

        return {
            "background": background,
            "foreground_objects": foreground,
            "small_elements": small_elements
        }

    def export_psd(self, metadata: dict, psd_path: str) -> str:
        """セグメント結果をレイヤー付きPSDとして出力"""
        w = metadata["image_size"]["width"]
        h = metadata["image_size"]["height"]

        psd = pytoshop.PsdFile(num_channels=3, height=h, width=w)
        layer_records = []

        # レイヤー順序: 背景 → 小要素 → 前景（上が前面）
        ordered_ids = []
        bg_id = metadata["classification"]["background"]
        if bg_id is not None:
            ordered_ids.append(bg_id)
        ordered_ids.extend(metadata["classification"]["small_elements"])
        ordered_ids.extend(metadata["classification"]["foreground_objects"])

        for layer_id in ordered_ids:
            layer_data = metadata["layers"][layer_id]
            img = np.array(Image.open(layer_data["path"]).convert("RGBA"))
            bx, by, bw, bh = layer_data["bbox"]

            # 分類に基づいた名前
            if layer_id == bg_id:
                name = "Background"
            elif layer_id in metadata["classification"]["foreground_objects"]:
                name = f"Object_{layer_id}"
            else:
                name = f"Element_{layer_id}"

            # pytoshopはチャンネルごとに分離して渡す
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            a = img[:, :, 3]

            channels = {
                -1: psd_layers.ChannelImageData(image=a, compression=psd_enums.Compression.raw),
                0: psd_layers.ChannelImageData(image=r, compression=psd_enums.Compression.raw),
                1: psd_layers.ChannelImageData(image=g, compression=psd_enums.Compression.raw),
                2: psd_layers.ChannelImageData(image=b, compression=psd_enums.Compression.raw),
            }

            record = psd_layers.LayerRecord(
                top=by, left=bx, bottom=by + bh, right=bx + bw,
                name=name,
                channels=channels,
            )
            layer_records.append(record)

        layer_info = psd_layers.LayerInfo(layer_records=layer_records)
        psd.layer_and_mask_info = psd_layers.LayerAndMaskInfo(layer_info=layer_info)

        with open(psd_path, 'wb') as f:
            psd.write(f)

        print(f"PSD exported: {psd_path} ({len(layer_records)} layers)", file=sys.stderr)
        return psd_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAM image segmentation + LaMa inpainting")
    parser.add_argument("checkpoint_path", help="Path to SAM model checkpoint")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("output_dir", help="Output directory for layer PNGs")
    parser.add_argument("--psd", help="Also export as layered PSD file", metavar="PSD_PATH")
    args = parser.parse_args()

    try:
        segmenter = SAMSegmenter(args.checkpoint_path)
        result = segmenter.segment_image(args.image_path, args.output_dir)

        if args.psd:
            segmenter.export_psd(result, args.psd)

        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
