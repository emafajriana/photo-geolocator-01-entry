import { pathToFileURL } from 'node:url';

const CONCURRENCY = 4;
const ANSWER_CHUNK_SIZE = 5;

export interface RgbRaster {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface SceneFeatures {
  lush: number;
  snow: number;
  arid: number;
  red: number;
  pale: number;
  darkveg: number;
  yellow: number;
  whiteline: number;
  dry: number;
}

export type ZoneName = 'NA_temp' | 'CA_north' | 'MX' | 'BR' | 'SA_south' | 'EU_west'
  | 'EU_north' | 'RU_west' | 'RU_east' | 'CAsia' | 'ME' | 'AF_sub' | 'AF_north'
  | 'SAsia' | 'SEAsia' | 'EAsia' | 'AU' | 'NZ' | 'OCE';

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

const ZONES: { name: ZoneName; lat: number; lon: number; prior: number }[] = [
  { name: 'NA_temp', lat: 39, lon: -98, prior: 1.0 },
  { name: 'CA_north', lat: 56, lon: -106, prior: 0.7 },
  { name: 'MX', lat: 23, lon: -102, prior: 0.8 },
  { name: 'BR', lat: -12, lon: -53, prior: 1.1 },
  { name: 'SA_south', lat: -32, lon: -64, prior: 0.75 },
  { name: 'EU_west', lat: 48, lon: 10, prior: 0.9 },
  { name: 'EU_north', lat: 62, lon: 15, prior: 0.6 },
  { name: 'RU_west', lat: 55, lon: 45, prior: 1.1 },
  { name: 'RU_east', lat: 58, lon: 110, prior: 0.8 },
  { name: 'CAsia', lat: 48, lon: 68, prior: 0.6 },
  { name: 'ME', lat: 29, lon: 47, prior: 0.6 },
  { name: 'AF_sub', lat: 2, lon: 22, prior: 0.8 },
  { name: 'AF_north', lat: 23, lon: 8, prior: 0.3 },
  { name: 'SAsia', lat: 22, lon: 79, prior: 0.6 },
  { name: 'SEAsia', lat: 13, lon: 102, prior: 0.8 },
  { name: 'EAsia', lat: 35, lon: 118, prior: 0.6 },
  { name: 'AU', lat: -25, lon: 134, prior: 0.5 },
  { name: 'NZ', lat: -42, lon: 172, prior: 0.4 },
  { name: 'OCE', lat: -12, lon: 156, prior: 0.3 },
];

const ZONE_CUE_SCORES: Partial<Record<ZoneName, Partial<Record<keyof SceneFeatures, number>>>> = {
  NA_temp: { yellow: 3.0, snow: 1.5, arid: 1.2, pale: 1.5 },
  CA_north: { yellow: 2.2, snow: 5.0, darkveg: 3.0 },
  MX: { yellow: 2.6, red: 1.5, lush: 2.5 },
  BR: { yellow: 2.6, red: 2.5, lush: 3.5 },
  SA_south: { yellow: 1.6, red: 2.0, arid: 2.5, pale: 2.0, dry: 2.0 },
  EU_west: { whiteline: 1.2 },
  EU_north: { whiteline: 1.0, snow: 5.0, darkveg: 3.0 },
  RU_west: { yellow: 0.8, whiteline: 0.8, snow: 5.0, darkveg: 3.0 },
  RU_east: { yellow: 0.8, snow: 5.0, darkveg: 3.0, pale: 1.8 },
  CAsia: { arid: 3.5, pale: 2.5, dry: 2.0 },
  ME: { arid: 4.0, pale: 1.8, dry: 1.6 },
  AF_sub: { red: 3.0, lush: 3.5 },
  AF_north: { arid: 3.0 },
  SAsia: { lush: 3.0 },
  SEAsia: { lush: 4.0 },
  EAsia: { whiteline: 0.8 },
  AU: { red: 8.0, arid: 3.5, dry: 1.6 },
  NZ: { whiteline: 1.2, lush: 2.6 },
  OCE: { lush: 3.0, darkveg: 2.0 },
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

function skylike(raster: RgbRaster, start: number, end: number): number {
  let blueGreenPixels = 0;
  let brightPixels = 0;
  let sum = 0;
  let count = 0;
  for (let y = 0; y < raster.height; y += 1) {
    if (y < raster.height * start || y >= raster.height * end) continue;
    for (let x = 0; x < raster.width; x += 1) {
      const offset = (y * raster.width + x) * 3;
      const r = raster.data[offset]! / 255;
      const g = raster.data[offset + 1]! / 255;
      const b = raster.data[offset + 2]! / 255;
      const max = Math.max(r, g, b);
      if (b >= g && g >= r - 0.05) blueGreenPixels += 1;
      const mean = (r + g + b) / 3;
      if (mean > 0.75) brightPixels += 1;
      count += 1;
    }
  }
  if (count === 0) return 0;
  return 0.5 * (blueGreenPixels / count) + 0.5 * (brightPixels / count);
}

function verticallyFlipped(raster: RgbRaster): RgbRaster {
  const data = new Uint8Array(raster.data.length);
  const rowLength = raster.width * 3;
  for (let y = 0; y < raster.height; y += 1) {
    const source = (raster.height - 1 - y) * rowLength;
    data.set(raster.data.subarray(source, source + rowLength), y * rowLength);
  }
  return { width: raster.width, height: raster.height, data };
}

function bandFraction(
  raster: RgbRaster,
  band: (x: number, y: number) => boolean,
  cue: (r: number, g: number, b: number, max: number, min: number, sat: number, mean: number) => boolean,
): number {
  let count = 0;
  let matches = 0;
  for (let y = 0; y < raster.height; y += 1) {
    for (let x = 0; x < raster.width; x += 1) {
      if (!band(x, y)) continue;
      const offset = (y * raster.width + x) * 3;
      const r = raster.data[offset]! / 255;
      const g = raster.data[offset + 1]! / 255;
      const b = raster.data[offset + 2]! / 255;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const sat = max === 0 ? 0 : (max - min) / max;
      const mean = (r + g + b) / 3;
      count += 1;
      if (cue(r, g, b, max, min, sat, mean)) matches += 1;
    }
  }
  return count === 0 ? 0 : matches / count;
}

export function extractFeatures(input: RgbRaster): SceneFeatures {
  const raster = skylike(input, 0.875, 1) > skylike(input, 0, 0.125) + 0.08
    ? verticallyFlipped(input)
    : input;
  const whole = (): boolean => true;
  const top = (_x: number, y: number): boolean => y < raster.height * 0.125;
  const bottom = (_x: number, y: number): boolean => y >= raster.height * 0.875;
  const ground = (_x: number, y: number): boolean => y >= raster.height * 0.625;
  const road = (x: number, y: number): boolean =>
    y >= raster.height * 0.708 && x >= raster.width * 0.333 && x < raster.width * 0.667;
  const shoulder = (x: number, y: number): boolean =>
    y >= raster.height * 0.625 && (x < raster.width * 0.29 || x >= raster.width * 0.71);
  const green = (r: number, g: number, b: number): boolean => g > 1.04 * r && g > 1.04 * b;
  return {
    lush: bandFraction(raster, whole, (r, g, b, _max, _min, sat) => green(r, g, b) && sat > 0.18),
    snow: bandFraction(raster, whole, (_r, _g, _b, max, _min, sat) => max > 0.80 && sat < 0.20),
    arid: bandFraction(raster, whole, (r, g, b, _max, _min, sat) =>
      r > 0.42 && r > 1.22 * g && g > 1.12 * b && sat > 0.12),
    red: bandFraction(raster, shoulder, (r, g, b) =>
      r > 0.28 && r > 1.40 * g && r > 1.60 * b),
    pale: bandFraction(raster, whole, (_r, _g, _b, max, _min, sat) => max > 0.55 && sat < 0.25),
    darkveg: bandFraction(raster, whole, (r, g, b, max) => green(r, g, b) && max < 0.35),
    yellow: bandFraction(raster, road, (r, g, b) => r > 0.50 && g > 0.42 && b < r - 0.22),
    whiteline: bandFraction(raster, road, (_r, _g, _b, max, min, _sat, mean) =>
      mean > 0.62 && max - min < 0.22),
    dry: bandFraction(raster, ground, (r, g, b, _max, _min, sat) => r > g && g > b && sat > 0.10),
  };
}

function toVector(lat: number, lon: number): [number, number, number] {
  const latitude = lat * Math.PI / 180;
  const longitude = lon * Math.PI / 180;
  return [
    Math.cos(latitude) * Math.cos(longitude),
    Math.cos(latitude) * Math.sin(longitude),
    Math.sin(latitude),
  ];
}

function toCoordinates(vector: [number, number, number]): { lat: number; lon: number } {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  if (!Number.isFinite(length) || length < 1e-12) return { lat: 56.07, lon: 9.93 };
  const x = vector[0] / length;
  const y = vector[1] / length;
  const z = vector[2] / length;
  return {
    lat: Math.atan2(z, Math.hypot(x, y)) * 180 / Math.PI,
    lon: Math.atan2(y, x) * 180 / Math.PI,
  };
}

export function predictZones(features: SceneFeatures): { lat: number; lon: number } {
  const zoneWeights = ZONES.map((zone) => {
    const scoreTable = ZONE_CUE_SCORES[zone.name];
    let score = 0;
    if (scoreTable !== undefined) {
      for (const cue of Object.keys(scoreTable) as (keyof SceneFeatures)[]) {
        score += features[cue] * scoreTable[cue]!;
      }
    }
    return { ...zone, weight: zone.prior * Math.exp(score / 1.4) };
  });
  let x = 0;
  let y = 0;
  let z = 0;
  let totalWeight = 0;
  for (const zone of zoneWeights) {
    const vector = toVector(zone.lat, zone.lon);
    x += zone.weight * vector[0];
    y += zone.weight * vector[1];
    z += zone.weight * vector[2];
    totalWeight += zone.weight;
  }
  if (!Number.isFinite(totalWeight) || totalWeight <= 0) return { lat: 56.07, lon: 9.93 };
  const blended = toCoordinates([x / totalWeight, y / totalWeight, z / totalWeight]);
  const answerVector = toVector(blended.lat, blended.lon);
  const priorVector = toVector(56.07, 9.93);
  const result = toCoordinates([
    0.7 * answerVector[0] + 0.3 * priorVector[0],
    0.7 * answerVector[1] + 0.3 * priorVector[1],
    0.7 * answerVector[2] + 0.3 * priorVector[2],
  ]);
  return {
    lat: Math.round(result.lat * 1_000_000) / 1_000_000,
    lon: Math.round(result.lon * 1_000_000) / 1_000_000,
  };
}

export function locatePhoto(bytes: Uint8Array): { lat: number; lon: number; features: SceneFeatures | null } {
  const raster = decodeJpegDc(bytes);
  if (raster === null) {
    const fallback = predictZones({
      lush: 0, snow: 0, arid: 0, red: 0, pale: 0, darkveg: 0, yellow: 0, whiteline: 0, dry: 0,
    });
    return { ...fallback, features: null };
  }
  const features = extractFeatures(raster);
  return { ...predictZones(features), features };
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
    return { id: photo.id, lat: 56.07, lon: 9.93 };
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
