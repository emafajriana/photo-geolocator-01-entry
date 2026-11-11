import { pathToFileURL } from 'node:url';

const CONCURRENCY = 4;
const ANSWER_CHUNK_SIZE = 5;

export interface RgbRaster {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface ColourMean {
  r: number;
  g: number;
  b: number;
}

export interface SceneFeatures {
  sky: ColourMean;
  upper: ColourMean;
  ground: ColourMean;
  overall: ColourMean;
  brightness: number;
  saturation: number;
  groundGreenness: number;
  overallGreenness: number;
  skyBlueness: number;
  snowScore: number;
  aridScore: number;
  waterSkyBlueScore: number;
  greyUrbanScore: number;
  groundBrightness: number;
}

export type SceneClass = 'snow/boreal' | 'tropical lush' | 'arid/dry' | 'mediterranean'
  | 'temperate green' | 'urban grey/overcast' | 'fallback';

interface HuffmanTable {
  symbols: Map<number, number>;
}

interface Component {
  id: number;
  h: number;
  v: number;
  quantId: number;
  blockColumns: number;
  blockRows: number;
  dc: Int32Array;
  predictor: number;
}

interface Frame {
  width: number;
  height: number;
  maxH: number;
  maxV: number;
  mcuColumns: number;
  mcuRows: number;
  progressive: boolean;
  components: Component[];
}

interface ScanComponent {
  component: Component;
  dcTable: HuffmanTable | undefined;
  acTable: HuffmanTable | undefined;
}

const ANCHORS: Record<SceneClass, { lat: number; lon: number }> = {
  'snow/boreal': { lat: 61.5, lon: 14.0 },
  'tropical lush': { lat: 14.6, lon: 101.0 },
  'arid/dry': { lat: 32.0, lon: -6.5 },
  mediterranean: { lat: 39.5, lon: 2.5 },
  'temperate green': { lat: 47.5, lon: 9.0 },
  'urban grey/overcast': { lat: 50.5, lon: 4.5 },
  fallback: { lat: 41.0, lon: -30.0 },
};

function requireRange(value: number, min: number, max: number, what: string): void {
  if (!Number.isInteger(value) || value < min || value > max) throw new Error(`invalid ${what}`);
}

function readU16(bytes: Uint8Array, offset: number): number {
  if (offset < 0 || offset + 1 >= bytes.length) throw new Error('truncated JPEG');
  return (bytes[offset]! << 8) | bytes[offset + 1]!;
}

function buildHuffmanTable(counts: Uint8Array, values: Uint8Array): HuffmanTable {
  const symbols = new Map<number, number>();
  let code = 0;
  let valueIndex = 0;
  for (let length = 1; length <= 16; length += 1) {
    const count = counts[length - 1]!;
    if (valueIndex + count > values.length) throw new Error('truncated Huffman table');
    for (let i = 0; i < count; i += 1) {
      if (code >= (1 << length)) throw new Error('oversubscribed Huffman table');
      symbols.set((length << 16) | code, values[valueIndex++]!);
      code += 1;
    }
    code <<= 1;
  }
  if (valueIndex !== values.length) throw new Error('invalid Huffman symbol count');
  return { symbols };
}

class EntropyReader {
  private position: number;
  private bitBuffer = 0;
  private bitsRemaining = 0;
  private readonly bytes: Uint8Array;

  constructor(bytes: Uint8Array, position: number) {
    this.bytes = bytes;
    this.position = position;
  }

  get offset(): number {
    return this.position;
  }

  readBit(): number {
    if (this.bitsRemaining === 0) {
      if (this.position >= this.bytes.length) throw new Error('truncated entropy data');
      let value = this.bytes[this.position++]!;
      if (value === 0xff) {
        if (this.position >= this.bytes.length) throw new Error('truncated entropy marker');
        const next = this.bytes[this.position++]!;
        if (next !== 0x00) throw new Error('unexpected marker within entropy data');
        value = 0xff;
      }
      this.bitBuffer = value;
      this.bitsRemaining = 8;
    }
    this.bitsRemaining -= 1;
    return (this.bitBuffer >>> this.bitsRemaining) & 1;
  }

  readBits(count: number): number {
    if (count < 0 || count > 16) throw new Error('invalid entropy bit count');
    let value = 0;
    for (let i = 0; i < count; i += 1) value = (value << 1) | this.readBit();
    return value;
  }

  alignToByte(): void {
    this.bitBuffer = 0;
    this.bitsRemaining = 0;
  }

  consumeRestart(expectedIndex: number): void {
    this.alignToByte();
    if (this.position >= this.bytes.length || this.bytes[this.position] !== 0xff) {
      throw new Error('missing restart marker');
    }
    while (this.position < this.bytes.length && this.bytes[this.position] === 0xff) this.position += 1;
    if (this.position >= this.bytes.length || this.bytes[this.position] !== 0xd0 + expectedIndex) {
      throw new Error('unexpected restart marker');
    }
    this.position += 1;
  }
}

function decodeHuffman(reader: EntropyReader, table: HuffmanTable | undefined): number {
  if (table === undefined) throw new Error('missing Huffman table');
  let code = 0;
  for (let length = 1; length <= 16; length += 1) {
    code = (code << 1) | reader.readBit();
    const symbol = table.symbols.get((length << 16) | code);
    if (symbol !== undefined) return symbol;
  }
  throw new Error('invalid Huffman code');
}

function receiveExtend(reader: EntropyReader, size: number): number {
  if (size === 0) return 0;
  if (size > 16) throw new Error('unsupported coefficient magnitude');
  const bits = reader.readBits(size);
  const threshold = 1 << (size - 1);
  return bits < threshold ? bits - ((1 << size) - 1) : bits;
}

function decodeBaselineBlock(
  reader: EntropyReader,
  scanComponent: ScanComponent,
): number {
  const category = decodeHuffman(reader, scanComponent.dcTable);
  const difference = receiveExtend(reader, category);
  let k = 1;
  while (k <= 63) {
    const symbol = decodeHuffman(reader, scanComponent.acTable);
    const run = symbol >>> 4;
    const size = symbol & 15;
    if (size === 0) {
      if (run === 0) break;
      if (run !== 15) throw new Error('invalid AC run');
      k += 16;
      continue;
    }
    k += run;
    if (k > 63) throw new Error('AC run exceeds block');
    reader.readBits(size);
    k += 1;
  }
  return difference;
}

function scanEndSkippingAc(bytes: Uint8Array, offset: number): number {
  let position = offset;
  while (position < bytes.length) {
    if (bytes[position] !== 0xff) {
      position += 1;
      continue;
    }
    const markerStart = position;
    position += 1;
    while (position < bytes.length && bytes[position] === 0xff) position += 1;
    if (position >= bytes.length) throw new Error('truncated entropy marker');
    const marker = bytes[position++]!;
    if (marker === 0x00 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    return markerStart;
  }
  throw new Error('JPEG scan has no terminating marker');
}

function componentBlockIndex(component: Component, x: number, y: number): number {
  if (x < 0 || y < 0 || x >= component.blockColumns || y >= component.blockRows) {
    throw new Error('scan block outside component');
  }
  return y * component.blockColumns + x;
}

function decodeScan(
  bytes: Uint8Array,
  entropyStart: number,
  frame: Frame,
  scanComponents: ScanComponent[],
  spectralStart: number,
  spectralEnd: number,
  successiveHigh: number,
  successiveLow: number,
  restartInterval: number,
): number {
  if (!frame.progressive && (spectralStart !== 0 || spectralEnd !== 63 || successiveHigh !== 0 || successiveLow !== 0)) {
    throw new Error('unsupported sequential scan parameters');
  }

  const isDcScan = spectralStart === 0 && spectralEnd === 0;
  if (frame.progressive && !isDcScan) return scanEndSkippingAc(bytes, entropyStart);
  if (frame.progressive && (spectralStart !== 0 || spectralEnd !== 0 || successiveHigh > 13 || successiveLow > 13)) {
    throw new Error('unsupported progressive DC scan');
  }

  const interleaved = scanComponents.length > 1;
  const totalMcus = interleaved
    ? frame.mcuColumns * frame.mcuRows
    : Math.ceil((frame.width * scanComponents[0]!.component.h) / (frame.maxH * 8))
      * Math.ceil((frame.height * scanComponents[0]!.component.v) / (frame.maxV * 8));
  if (totalMcus < 1 || totalMcus > 2_000_000) throw new Error('invalid scan dimensions');

  const reader = new EntropyReader(bytes, entropyStart);
  for (const entry of scanComponents) entry.component.predictor = 0;

  const decodeBlock = (entry: ScanComponent, x: number, y: number): void => {
    const component = entry.component;
    const index = componentBlockIndex(component, x, y);
    if (!frame.progressive) {
      component.predictor += decodeBaselineBlock(reader, entry);
      component.dc[index] = component.predictor;
      return;
    }
    if (successiveHigh === 0) {
      component.predictor += receiveExtend(reader, decodeHuffman(reader, entry.dcTable));
      component.dc[index] = component.predictor * (1 << successiveLow);
    } else if (reader.readBit() !== 0) {
      const step = 1 << successiveLow;
      component.dc[index] += component.dc[index] < 0 ? -step : step;
    }
  };

  for (let mcu = 0; mcu < totalMcus; mcu += 1) {
    if (interleaved) {
      const mcuX = mcu % frame.mcuColumns;
      const mcuY = Math.floor(mcu / frame.mcuColumns);
      for (const entry of scanComponents) {
        for (let y = 0; y < entry.component.v; y += 1) {
          for (let x = 0; x < entry.component.h; x += 1) {
            decodeBlock(entry, mcuX * entry.component.h + x, mcuY * entry.component.v + y);
          }
        }
      }
    } else {
      const component = scanComponents[0]!.component;
      const columns = Math.ceil((frame.width * component.h) / (frame.maxH * 8));
      const x = mcu % columns;
      const y = Math.floor(mcu / columns);
      decodeBlock(scanComponents[0]!, x, y);
    }

    if (restartInterval > 0 && (mcu + 1) % restartInterval === 0 && mcu + 1 < totalMcus) {
      reader.consumeRestart(Math.floor(mcu / restartInterval) % 8);
      for (const entry of scanComponents) entry.component.predictor = 0;
    }
  }
  reader.alignToByte();
  return reader.offset;
}

function parseSegmentEnd(bytes: Uint8Array, offset: number): number {
  const length = readU16(bytes, offset);
  if (length < 2 || offset + length > bytes.length) throw new Error('invalid JPEG segment length');
  return offset + length;
}

function nextMarker(bytes: Uint8Array, offset: number): { marker: number; afterCode: number } {
  if (offset >= bytes.length || bytes[offset] !== 0xff) throw new Error('expected JPEG marker');
  let position = offset + 1;
  while (position < bytes.length && bytes[position] === 0xff) position += 1;
  if (position >= bytes.length || bytes[position] === 0x00) throw new Error('invalid JPEG marker');
  return { marker: bytes[position]!, afterCode: position + 1 };
}

function decodeJpegDcInternal(input: Uint8Array): RgbRaster {
  const bytes = input;
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) throw new Error('not a JPEG');

  const quantization = new Map<number, number>();
  const huffman = new Map<number, HuffmanTable>();
  let frame: Frame | null = null;
  let restartInterval = 0;
  let decodedDc = false;
  let sawEoi = false;
  let position = 2;

  while (position < bytes.length) {
    const { marker, afterCode } = nextMarker(bytes, position);
    position = afterCode;
    if (marker === 0xd9) {
      sawEoi = true;
      break;
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (marker === 0xcc || marker === 0xc8 || marker === 0xde || marker === 0xdf) {
      throw new Error('unsupported JPEG coding extension');
    }
    if (marker >= 0xc0 && marker <= 0xcf
      && marker !== 0xc0 && marker !== 0xc2 && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      throw new Error('unsupported JPEG frame coding');
    }

    const end = parseSegmentEnd(bytes, position);
    const body = position + 2;
    if (marker === 0xdb) {
      let cursor = body;
      while (cursor < end) {
        const info = bytes[cursor++]!;
        const precision = info >>> 4;
        const tableId = info & 15;
        if (precision > 1 || tableId > 3) throw new Error('invalid quantization table');
        const valueBytes = precision === 0 ? 1 : 2;
        if (cursor + 64 * valueBytes > end) throw new Error('truncated quantization table');
        let dcValue = 0;
        for (let i = 0; i < 64; i += 1) {
          const value = precision === 0 ? bytes[cursor++]! : readU16(bytes, cursor);
          if (precision === 1) cursor += 2;
          if (value === 0) throw new Error('zero quantization value');
          if (i === 0) dcValue = value;
        }
        quantization.set(tableId, dcValue);
      }
    } else if (marker === 0xc4) {
      let cursor = body;
      while (cursor < end) {
        if (cursor + 17 > end) throw new Error('truncated Huffman table');
        const info = bytes[cursor++]!;
        const tableClass = info >>> 4;
        const tableId = info & 15;
        if (tableClass > 1 || tableId > 3) throw new Error('invalid Huffman table selector');
        const counts = bytes.subarray(cursor, cursor + 16);
        cursor += 16;
        let count = 0;
        for (const item of counts) count += item;
        if (cursor + count > end) throw new Error('truncated Huffman values');
        huffman.set((tableClass << 4) | tableId, buildHuffmanTable(counts, bytes.subarray(cursor, cursor + count)));
        cursor += count;
      }
    } else if (marker === 0xdd) {
      if (end - body !== 2) throw new Error('invalid restart interval segment');
      restartInterval = readU16(bytes, body);
    } else if (marker === 0xc0 || marker === 0xc2) {
      if (frame !== null) throw new Error('multiple JPEG frames are unsupported');
      const precision = bytes[body]!;
      const height = readU16(bytes, body + 1);
      const width = readU16(bytes, body + 3);
      const count = bytes[body + 5]!;
      if (precision !== 8 || width < 1 || height < 1 || width > 1600 || height > 8192) {
        throw new Error('unsupported JPEG precision or dimensions');
      }
      if ((count !== 1 && count !== 3) || end - body !== 6 + count * 3) throw new Error('unsupported JPEG components');
      const components: Component[] = [];
      let maxH = 0;
      let maxV = 0;
      for (let i = 0; i < count; i += 1) {
        const cursor = body + 6 + i * 3;
        const id = bytes[cursor]!;
        const sampling = bytes[cursor + 1]!;
        const h = sampling >>> 4;
        const v = sampling & 15;
        const quantId = bytes[cursor + 2]!;
        requireRange(h, 1, 4, 'horizontal sampling factor');
        requireRange(v, 1, 4, 'vertical sampling factor');
        if (components.some((component) => component.id === id)) throw new Error('duplicate component id');
        maxH = Math.max(maxH, h);
        maxV = Math.max(maxV, v);
        components.push({ id, h, v, quantId, blockColumns: 0, blockRows: 0, dc: new Int32Array(0), predictor: 0 });
      }
      const mcuColumns = Math.ceil(width / (maxH * 8));
      const mcuRows = Math.ceil(height / (maxV * 8));
      let totalBlocks = 0;
      for (const component of components) {
        component.blockColumns = mcuColumns * component.h;
        component.blockRows = mcuRows * component.v;
        totalBlocks += component.blockColumns * component.blockRows;
        if (totalBlocks > 2_000_000) throw new Error('JPEG block count is too large');
        component.dc = new Int32Array(component.blockColumns * component.blockRows);
      }
      frame = {
        width,
        height,
        maxH,
        maxV,
        mcuColumns,
        mcuRows,
        progressive: marker === 0xc2,
        components,
      };
    } else if (marker === 0xda) {
      if (frame === null) throw new Error('scan before JPEG frame');
      const count = bytes[body]!;
      if (count < 1 || count > frame.components.length || end - body !== 1 + count * 2 + 3) {
        throw new Error('invalid scan component list');
      }
      const scanComponents: ScanComponent[] = [];
      for (let i = 0; i < count; i += 1) {
        const cursor = body + 1 + i * 2;
        const componentId = bytes[cursor]!;
        const selector = bytes[cursor + 1]!;
        const component = frame.components.find((candidate) => candidate.id === componentId);
        if (component === undefined || scanComponents.some((entry) => entry.component === component)) {
          throw new Error('unknown or repeated scan component');
        }
        scanComponents.push({
          component,
          dcTable: huffman.get(selector >>> 4),
          acTable: huffman.get(0x10 | (selector & 15)),
        });
      }
      const spectralOffset = body + 1 + count * 2;
      const spectralStart = bytes[spectralOffset]!;
      const spectralEnd = bytes[spectralOffset + 1]!;
      const successive = bytes[spectralOffset + 2]!;
      const successiveHigh = successive >>> 4;
      const successiveLow = successive & 15;
      position = decodeScan(
        bytes,
        end,
        frame,
        scanComponents,
        spectralStart,
        spectralEnd,
        successiveHigh,
        successiveLow,
        restartInterval,
      );
      if (!frame.progressive || (spectralStart === 0 && spectralEnd === 0)) decodedDc = true;
      continue;
    }
    position = end;
  }

  if (frame === null || !decodedDc || !sawEoi) throw new Error('JPEG has no complete decodable DC image');
  if (frame.components.some((component) => !quantization.has(component.quantId))) {
    throw new Error('JPEG is missing a quantization table');
  }

  const width = Math.ceil(frame.width / 8);
  const height = Math.ceil(frame.height / 8);
  if (width * height > 500_000) throw new Error('DC raster is too large');
  const raster = new Uint8Array(width * height * 3);
  const yComponent = frame.components[0]!;
  const cbComponent = frame.components[1];
  const crComponent = frame.components[2];
  const dcSample = (component: Component | undefined, x: number, y: number): number => {
    if (component === undefined) return 128;
    const blockX = Math.floor(x * component.h / frame!.maxH);
    const blockY = Math.floor(y * component.v / frame!.maxV);
    const coefficient = component.dc[componentBlockIndex(component, blockX, blockY)]!;
    return 128 + coefficient * quantization.get(component.quantId)! / 8;
  };
  const clampByte = (value: number): number => Math.max(0, Math.min(255, Math.round(value)));
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const luma = dcSample(yComponent, x, y);
      const cb = dcSample(cbComponent, x, y) - 128;
      const cr = dcSample(crComponent, x, y) - 128;
      const offset = (y * width + x) * 3;
      if (cbComponent === undefined || crComponent === undefined) {
        raster[offset] = clampByte(luma);
        raster[offset + 1] = clampByte(luma);
        raster[offset + 2] = clampByte(luma);
      } else {
        raster[offset] = clampByte(luma + 1.402 * cr);
        raster[offset + 1] = clampByte(luma - 0.344136 * cb - 0.714136 * cr);
        raster[offset + 2] = clampByte(luma + 1.772 * cb);
      }
    }
  }
  return { width, height, data: raster };
}

export function decodeJpegDc(input: Uint8Array): RgbRaster | null {
  try {
    return decodeJpegDcInternal(input);
  } catch {
    return null;
  }
}

function averageBand(raster: RgbRaster, startRow: number, endRow: number): ColourMean {
  const start = Math.max(0, Math.min(raster.height - 1, startRow));
  const end = Math.max(start + 1, Math.min(raster.height, endRow));
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  for (let y = start; y < end; y += 1) {
    for (let x = 0; x < raster.width; x += 1) {
      const offset = (y * raster.width + x) * 3;
      r += raster.data[offset]!;
      g += raster.data[offset + 1]!;
      b += raster.data[offset + 2]!;
      count += 1;
    }
  }
  return { r: r / count, g: g / count, b: b / count };
}

function greenness(mean: ColourMean): number {
  return mean.g - (mean.r + mean.b) / 2;
}

export function extractFeatures(raster: RgbRaster): SceneFeatures {
  const sky = averageBand(raster, 0, Math.ceil(raster.height * 0.18));
  const upper = averageBand(raster, 0, Math.ceil(raster.height * 0.4));
  const ground = averageBand(raster, Math.floor(raster.height * 0.7), raster.height);
  const overall = averageBand(raster, 0, raster.height);
  let brightnessSum = 0;
  let saturationSum = 0;
  let snowPixels = 0;
  let skyBluePixels = 0;
  let aridPixels = 0;
  let groundPixels = 0;
  for (let y = 0; y < raster.height; y += 1) {
    for (let x = 0; x < raster.width; x += 1) {
      const offset = (y * raster.width + x) * 3;
      const r = raster.data[offset]!;
      const g = raster.data[offset + 1]!;
      const b = raster.data[offset + 2]!;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const brightness = (r + g + b) / 3;
      const saturation = max === 0 ? 0 : (max - min) / max;
      brightnessSum += brightness;
      saturationSum += saturation;
      if (brightness > 200 && saturation < 0.18) snowPixels += 1;
      if (y < Math.ceil(raster.height * 0.4) && b > r * 1.08 && b > g * 1.02 && b > 90) {
        skyBluePixels += 1;
      }
      if (y >= Math.floor(raster.height * 0.7)) {
        groundPixels += 1;
        if (r > g && g > b && r - b > 18 && brightness > 70) aridPixels += 1;
      }
    }
  }
  const pixelCount = raster.width * raster.height;
  const brightness = brightnessSum / pixelCount;
  const saturation = saturationSum / pixelCount;
  const groundBrightness = (ground.r + ground.g + ground.b) / 3;
  const skyBlueFraction = skyBluePixels / Math.max(1, Math.ceil(raster.height * 0.4) * raster.width);
  const aridScore = aridPixels / Math.max(1, groundPixels);
  return {
    sky,
    upper,
    ground,
    overall,
    brightness,
    saturation,
    groundGreenness: greenness(ground),
    overallGreenness: greenness(overall),
    skyBlueness: sky.b - (sky.r + sky.g) / 2,
    snowScore: snowPixels / pixelCount,
    aridScore,
    waterSkyBlueScore: skyBlueFraction,
    greyUrbanScore: Math.max(0, Math.min(1, 1 - saturation / 0.35)),
    groundBrightness,
  };
}

export function classifyScene(features: SceneFeatures): SceneClass {
  if ((features.snowScore >= 0.22 && features.brightness > 135)
    || (features.brightness > 210 && features.saturation < 0.16)) {
    return 'snow/boreal';
  }
  if (features.groundGreenness >= 95
    && features.overallGreenness >= 45
    && features.brightness >= 135
    && features.saturation >= 0.32) {
    return 'tropical lush';
  }
  if (features.aridScore >= 0.48
    && features.groundBrightness >= 90
    && features.groundGreenness < 55
    && (features.waterSkyBlueScore >= 0.12 || features.aridScore >= 0.68)) {
    return 'arid/dry';
  }
  if (features.groundGreenness >= 3
    && features.groundGreenness < 40
    && features.groundBrightness >= 105
    && features.brightness >= 110
    && features.saturation >= 0.12) {
    return 'mediterranean';
  }
  if (features.groundGreenness >= 15 && features.brightness >= 55) return 'temperate green';
  if (features.saturation < 0.18 && features.overallGreenness < 20) return 'urban grey/overcast';
  return 'fallback';
}

function hashBytes(bytes: Uint8Array, seed: number): number {
  let hash = seed >>> 0;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
    hash ^= hash >>> 13;
  }
  return hash >>> 0;
}

export function locatePhoto(bytes: Uint8Array): { lat: number; lon: number; scene: SceneClass; features: SceneFeatures | null } {
  try {
    const raster = decodeJpegDc(bytes);
    const features = raster === null ? null : extractFeatures(raster);
    const scene = features === null ? 'fallback' : classifyScene(features);
    const anchor = ANCHORS[scene];
    const jitter = (seed: number): number => (seed / 0xffffffff) * 3 - 1.5;
    const lat = anchor.lat + jitter(hashBytes(bytes, 0x811c9dc5));
    const lon = anchor.lon + jitter(hashBytes(bytes, 0x9e3779b9));
    return {
      lat: Math.round(Math.max(-90, Math.min(90, lat)) * 1_000_000) / 1_000_000,
      lon: Math.round(Math.max(-180, Math.min(180, lon)) * 1_000_000) / 1_000_000,
      scene,
      features,
    };
  } catch {
    const anchor = ANCHORS.fallback;
    const jitter = (seed: number): number => (seed / 0xffffffff) * 3 - 1.5;
    return {
      lat: Math.round((anchor.lat + jitter(hashBytes(bytes, 0x811c9dc5))) * 1_000_000) / 1_000_000,
      lon: Math.round((anchor.lon + jitter(hashBytes(bytes, 0x9e3779b9))) * 1_000_000) / 1_000_000,
      scene: 'fallback',
      features: null,
    };
  }
}

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface PhotoList {
  photos: Photo[];
  answered?: number;
}

interface Answer {
  id: string;
  lat: number;
  lon: number;
}

async function handlePhoto(envUrl: string, photo: Photo): Promise<Answer> {
  let bytes = new Uint8Array(0);
  try {
    const response = await fetch(`${envUrl}${photo.url}`);
    if (!response.ok) throw new Error(`photo fetch returned HTTP ${response.status}`);
    bytes = new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    process.stderr.write(`photo ${photo.id}: ${String(error)}; using byte fallback\n`);
  }
  try {
    const position = locatePhoto(bytes);
    return { id: photo.id, lat: position.lat, lon: position.lon };
  } catch {
    const fallback = ANCHORS.fallback;
    return { id: photo.id, lat: fallback.lat, lon: fallback.lon };
  }
}

async function postAnswers(envUrl: string, answers: Answer[], alreadyAnswered: number): Promise<void> {
  for (let offset = 0; offset < answers.length; offset += ANSWER_CHUNK_SIZE) {
    const chunk = answers.slice(offset, offset + ANSWER_CHUNK_SIZE);
    const response = await fetch(`${envUrl}/answers`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ answers: chunk }),
    });
    const detail = await response.text();
    if (!response.ok) throw new Error(`answer post failed (HTTP ${response.status}): ${detail}`);
    const result = JSON.parse(detail) as { accepted?: unknown; answered?: unknown };
    const expectedIds = chunk.map((answer) => answer.id);
    if (!Array.isArray(result.accepted)
      || result.accepted.length !== expectedIds.length
      || !expectedIds.every((id, index) => result.accepted![index] === id)) {
      throw new Error(`answer post accepted an unexpected set: ${detail}`);
    }
    const expectedAnswered = alreadyAnswered + offset + chunk.length;
    if (result.answered !== undefined && result.answered !== expectedAnswered) {
      throw new Error(`answer post reported ${String(result.answered)} answered; expected ${expectedAnswered}`);
    }
    process.stdout.write(`posted ${chunk.length} answers; ${expectedAnswered} answered\n`);
  }
}

async function main(): Promise<void> {
  const configuredEnv = process.env.MAKEFUN_ENV_URL;
  if (configuredEnv === undefined || configuredEnv === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const envUrl = configuredEnv.replace(/\/+$/, '');
  const listResponse = await fetch(`${envUrl}/photos`);
  if (!listResponse.ok) throw new Error(`photo list failed: HTTP ${listResponse.status}`);
  const listing = await listResponse.json() as PhotoList;
  if (!Array.isArray(listing.photos)) throw new Error('photo list did not contain a photos array');
  const photos = listing.photos.filter((photo) => !photo.answered);
  const answers: (Answer | null)[] = new Array(photos.length).fill(null);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const index = next;
      next += 1;
      answers[index] = await handlePhoto(envUrl, photos[index]!);
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  const orderedAnswers = answers.filter((answer): answer is Answer => answer !== null);
  await postAnswers(envUrl, orderedAnswers, listing.answered ?? 0);
  process.stdout.write(`finished: ${orderedAnswers.length} photos processed\n`);
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await main();
  } catch (error) {
    process.stderr.write(`agent: ${(error as Error).stack ?? String(error)}\n`);
    process.exitCode = 1;
  }
}
