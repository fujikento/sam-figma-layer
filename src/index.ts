#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { SAMProcessor } from './sam-processor.js';
import os from 'os';
import path from 'path';

const server = new Server(
  {
    name: 'sam-figma-layer',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

const samProcessor = new SAMProcessor();

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'segment_image',
      description:
        'Segment an image into multiple layers using Segment Anything Model (SAM). Returns transparent PNG layers with metadata.',
      inputSchema: {
        type: 'object',
        properties: {
          imagePath: {
            type: 'string',
            description: 'Absolute path to the input image',
          },
          outputDir: {
            type: 'string',
            description:
              'Output directory for layer PNGs (defaults to a temp directory)',
          },
        },
        required: ['imagePath'],
      },
    },
    {
      name: 'segment_and_export_psd',
      description:
        'Segment an image into layers and export as a layered PSD file. Includes LaMa background inpainting. Output works in Photoshop, After Effects, Unity, etc.',
      inputSchema: {
        type: 'object',
        properties: {
          imagePath: {
            type: 'string',
            description: 'Absolute path to the input image',
          },
          psdPath: {
            type: 'string',
            description:
              'Output PSD file path (defaults to same directory as input image)',
          },
        },
        required: ['imagePath'],
      },
    },
    {
      name: 'check_sam_status',
      description:
        'Check if the SAM model is installed and ready. Run `npm run setup` if not ready.',
      inputSchema: {
        type: 'object',
        properties: {},
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'check_sam_status': {
        const isReady = await samProcessor.isReady();
        return {
          content: [
            {
              type: 'text',
              text: isReady
                ? 'SAM model is ready.'
                : 'SAM model not found. Run: npm run setup',
            },
          ],
        };
      }

      case 'segment_image': {
        const { imagePath, outputDir } = args as {
          imagePath: string;
          outputDir?: string;
        };

        if (!imagePath || !path.isAbsolute(imagePath)) {
          throw new Error('imagePath must be an absolute file path');
        }

        const outDir =
          outputDir || path.join(os.tmpdir(), `sam_output_${Date.now()}`);

        const result = await samProcessor.segment(imagePath, outDir);

        return {
          content: [
            {
              type: 'text',
              text: `Segmentation complete.

Results:
- Total layers: ${result.total_layers}
- Background: Layer ${result.classification.background}
- Foreground objects: ${result.classification.foreground_objects.length}
- Small elements: ${result.classification.small_elements.length}
- Output directory: ${outDir}
- Metadata file: ${outDir}/metadata.json

Layers:
${result.layers
  .map(
    (l) =>
      `- ${l.filename}: bbox=[${l.bbox}], area=${l.area}, center=[${l.center}]`
  )
  .join('\n')}

To place these layers in Figma via figma-console MCP:
1. Create a parent frame sized ${result.image_size.width}x${result.image_size.height}
2. For each layer, create a rectangle at the bbox position and set the layer PNG as an image fill`,
            },
          ],
        };
      }

      case 'segment_and_export_psd': {
        const { imagePath, psdPath } = args as {
          imagePath: string;
          psdPath?: string;
        };

        if (!imagePath || !path.isAbsolute(imagePath)) {
          throw new Error('imagePath must be an absolute file path');
        }

        const outDir = path.join(os.tmpdir(), `sam_output_${Date.now()}`);
        const outputPsd =
          psdPath ||
          path.join(
            path.dirname(imagePath),
            `${path.basename(imagePath, path.extname(imagePath))}_layers.psd`
          );

        const result = await samProcessor.segmentAndExportPsd(
          imagePath,
          outDir,
          outputPsd
        );

        return {
          content: [
            {
              type: 'text',
              text: `PSD export complete.

- PSD file: ${outputPsd}
- Total layers: ${result.total_layers}
- Background: Layer ${result.classification.background} (inpainted)
- Foreground objects: ${result.classification.foreground_objects.length}
- Small elements: ${result.classification.small_elements.length}
- Image size: ${result.image_size.width}x${result.image_size.height}

The PSD file contains all layers with correct positioning and can be opened in Photoshop, After Effects, Krita, or any PSD-compatible application.`,
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error: any) {
    return {
      content: [
        {
          type: 'text',
          text: `Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('SAM-Figma-Layer MCP server running');
}

main().catch(console.error);
