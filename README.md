# SAM Figma Layer

Automatically segment any image into individual layers using Meta's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and place them in Figma.

**One image in, dozens of perfectly-masked layers out.**

## Features

- **Automatic segmentation** - SAM detects and separates every object in an image
- **Transparent PNG layers** - Each segment is exported as a cropped transparent PNG
- **Smart classification** - Layers are auto-classified as background, foreground objects, or small elements
- **AI background inpainting** - Background layer is automatically filled using [LaMa](https://github.com/advimman/lama) so moving foreground objects leaves no holes
- **Figma integration** - Import layers into Figma with correct positioning via the included Figma plugin
- **Claude Code / Desktop** - Use as an MCP server to segment images directly from Claude

## How It Works

```
Input Image ──> SAM Segmentation ──> Layer PNGs + metadata.json
                                          │
                          ┌───────────────┴───────────────┐
                          │                               │
                   Claude Code / Desktop           Figma Plugin
                   (via figma-console MCP)      (drag & drop import)
                          │                               │
                          └───────────────┬───────────────┘
                                          │
                                    Figma Layers
```

## Prerequisites

- **Python 3.9+** with pip
- **Node.js 18+** with npm
- ~3GB disk space for the SAM model

## Quick Start

### As MCP Server (Claude Code / Desktop)

```bash
# 1. Clone the repo
git clone https://github.com/fujikento/sam-figma-layer.git
cd sam-figma-layer

# 2. Install dependencies
npm install
pip install -r requirements.txt

# 3. Download SAM model (~2.4GB)
npm run setup

# 4. Build
npm run build

# 5. Add to Claude Code
claude mcp add sam-figma-layer -- node /path/to/sam-figma-layer/dist/index.js
```

Then in Claude Code:

```
You: "Segment ~/Downloads/photo.png"

Claude: [calls segment_image tool]
  -> 23 layers detected
  -> Output: /tmp/sam_output_xxx/
  -> metadata.json with layer positions
```

To also place layers in Figma, use with [figma-console MCP](https://github.com/anthropics/figma-console-mcp):

```
You: "Segment ~/Downloads/photo.png and place the layers in Figma"

Claude: 1. Calls segment_image -> gets layers
        2. Reads metadata.json
        3. Uses figma-console MCP to create frame + image fills in Figma
```

### As Figma Plugin

1. Open Figma Desktop
2. Go to **Plugins > Development > Import plugin from manifest**
3. Select `figma-plugin/manifest.json` from this repo
4. Run the plugin:
   - Paste the contents of `metadata.json`
   - Select the layer PNG files
   - Click **Import Layers**

## MCP Tools

### `segment_image`

Segments an image into multiple transparent PNG layers.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `imagePath` | string | Yes | Absolute path to the input image |
| `outputDir` | string | No | Output directory (defaults to temp dir) |

**Returns:** Layer count, classifications, bbox positions, and output paths.

### `check_sam_status`

Checks if the SAM model is downloaded and ready to use.

## Output Format

The `metadata.json` produced by segmentation:

```json
{
  "source_image": "/path/to/input.png",
  "image_size": { "width": 1920, "height": 1080 },
  "total_layers": 15,
  "layers": [
    {
      "id": 0,
      "filename": "layer_000.png",
      "bbox": [0, 0, 1920, 1080],
      "area": 1843200,
      "center": [960, 540]
    }
  ],
  "classification": {
    "background": 0,
    "foreground_objects": [1, 2, 3],
    "small_elements": [4, 5, 6, 7]
  }
}
```

## Project Structure

```
sam-figma-layer/
├── src/                    # MCP Server (TypeScript)
│   ├── index.ts            # Server entry point
│   └── sam-processor.ts    # SAM Python bridge
├── scripts/                # Python
│   ├── sam_segmenter.py    # SAM segmentation logic
│   └── install_sam.py      # Model download script
├── figma-plugin/           # Figma Plugin
│   ├── manifest.json
│   ├── code.ts             # Plugin logic
│   └── ui.html             # Plugin UI
├── package.json
├── requirements.txt
└── LICENSE
```

## Development

```bash
# MCP server development (auto-rebuild on change)
npm run dev

# Figma plugin development
cd figma-plugin
npm install
npm run watch
```

## License

MIT
