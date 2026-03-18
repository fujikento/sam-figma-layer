import { exec } from 'child_process';
import { promisify } from 'util';
import { readFile } from 'fs/promises';
import path from 'path';
import os from 'os';

const execAsync = promisify(exec);

export interface SAMLayer {
  id: number;
  filename: string;
  path: string;
  bbox: number[];
  area: number;
  center: number[];
}

export interface SAMResult {
  source_image: string;
  image_size: {
    width: number;
    height: number;
  };
  total_layers: number;
  layers: SAMLayer[];
  classification: {
    background: number | null;
    foreground_objects: number[];
    small_elements: number[];
  };
}

export class SAMProcessor {
  private checkpointPath: string;
  private scriptPath: string;

  constructor() {
    // Resolve script path relative to this package's root
    const packageRoot = path.resolve(new URL('.', import.meta.url).pathname, '..');
    this.scriptPath = path.join(packageRoot, 'scripts', 'sam_segmenter.py');

    // Model stored in user-agnostic location
    this.checkpointPath = path.join(os.homedir(), '.sam-figma-layer', 'models', 'sam_vit_h_4b8939.pth');
  }

  async segment(imagePath: string, outputDir: string): Promise<SAMResult> {
    // Pythonスクリプト実行
    const command = `python3 "${this.scriptPath}" "${this.checkpointPath}" "${imagePath}" "${outputDir}"`;

    try {
      const { stdout, stderr } = await execAsync(command, {
        maxBuffer: 10 * 1024 * 1024, // 10MB
        timeout: 300000 // 5分タイムアウト
      });

      // stdoutからJSON結果をパース
      const result: SAMResult = JSON.parse(stdout);
      return result;
    } catch (error: any) {
      throw new Error(`SAM segmentation failed: ${error.message}\n${error.stderr || ''}`);
    }
  }

  async isReady(): Promise<boolean> {
    try {
      // チェックポイントファイル存在確認
      await readFile(this.checkpointPath);
      return true;
    } catch {
      return false;
    }
  }
}
