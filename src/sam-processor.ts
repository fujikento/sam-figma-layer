import { execFile } from 'child_process';
import { promisify } from 'util';
import { access, constants } from 'fs/promises';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const execFileAsync = promisify(execFile);

export interface SAMLayer {
  id: number;
  filename: string;
  path: string;
  bbox: number[];
  area: number;
  center: number[];
  inpainted?: boolean;
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
    // Resolve script path relative to this package's root (Windows-safe)
    const currentDir = path.dirname(fileURLToPath(import.meta.url));
    const packageRoot = path.resolve(currentDir, '..');
    this.scriptPath = path.join(packageRoot, 'scripts', 'sam_segmenter.py');

    // Model stored in user-agnostic location
    this.checkpointPath = path.join(os.homedir(), '.sam-figma-layer', 'models', 'sam_vit_h_4b8939.pth');
  }

  async segment(imagePath: string, outputDir: string): Promise<SAMResult> {
    try {
      // Use execFile to avoid command injection
      const { stdout } = await execFileAsync(
        'python3',
        [this.scriptPath, this.checkpointPath, imagePath, outputDir],
        {
          maxBuffer: 10 * 1024 * 1024, // 10MB
          timeout: 300000, // 5 min
        }
      );

      const result: SAMResult = JSON.parse(stdout);
      return result;
    } catch (error: any) {
      throw new Error(`SAM segmentation failed: ${error.message}\n${error.stderr || ''}`);
    }
  }

  async isReady(): Promise<boolean> {
    try {
      // Check file existence without reading the entire 2.4GB model into memory
      await access(this.checkpointPath, constants.F_OK);
      return true;
    } catch {
      return false;
    }
  }
}
