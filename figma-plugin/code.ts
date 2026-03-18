interface LayerData {
  id: number;
  filename: string;
  bbox: number[];
  area: number;
  center: number[];
  inpainted?: boolean;
}

interface SAMMetadata {
  source_image: string;
  image_size: { width: number; height: number };
  total_layers: number;
  layers: LayerData[];
  classification: {
    background: number | null;
    foreground_objects: number[];
    small_elements: number[];
  };
}

interface ImportMessage {
  type: 'import-layers';
  metadata: SAMMetadata;
  images: { [filename: string]: Uint8Array };
}

figma.showUI(__html__, { width: 480, height: 560 });

figma.ui.onmessage = async (msg: ImportMessage) => {
  if (msg.type !== 'import-layers') return;

  const { metadata, images } = msg;
  const { image_size, layers, classification } = metadata;

  // Create parent frame
  const parentFrame = figma.createFrame();
  parentFrame.name = 'SAM Layers';
  parentFrame.resize(image_size.width, image_size.height);
  parentFrame.clipsContent = true;

  // Sort layers: background first, then foreground, then small elements
  const layerOrder: number[] = [];
  if (classification.background !== null) {
    layerOrder.push(classification.background);
  }
  layerOrder.push(...classification.foreground_objects);
  layerOrder.push(...classification.small_elements);

  let placedCount = 0;

  for (const layerId of layerOrder) {
    const layer = layers.find((l) => l.id === layerId);
    if (!layer) continue;

    const imageBytes = images[layer.filename];
    if (!imageBytes) continue;

    const [x, y, w, h] = layer.bbox;

    // Create rectangle at bbox position
    const rect = figma.createRectangle();
    rect.x = x;
    rect.y = y;
    rect.resize(w, h);

    // Classify and name the layer
    let layerName: string;
    if (layerId === classification.background) {
      layerName = 'Background';
    } else if (classification.foreground_objects.includes(layerId)) {
      layerName = `Object_${layerId}`;
    } else {
      layerName = `Element_${layerId}`;
    }
    rect.name = layerName;

    // Set image fill
    const image = figma.createImage(new Uint8Array(imageBytes));
    rect.fills = [
      {
        type: 'IMAGE',
        imageHash: image.hash,
        scaleMode: 'FILL',
      },
    ];

    parentFrame.appendChild(rect);
    placedCount++;
  }

  // Center the frame in viewport
  figma.viewport.scrollAndZoomIntoView([parentFrame]);

  figma.notify(`Imported ${placedCount} layers from SAM segmentation.`);
  figma.ui.postMessage({ type: 'import-complete', count: placedCount });
};
