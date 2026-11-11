const MODEL = 'claude-sonnet-5-5';
const MAX_TOKENS = 700;
const CONCURRENCY = 4;
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const MAX_RETRIES = 2;
const PRIOR = { lat: 20, lon: 78 };

const FIRST_PROMPT = `Study this street-level photo for clues to its location: readable signs and their language or script, license plates, road markings and which side of the road traffic uses, architecture, vegetation and terrain, sun position, and any recognizable landmark. Weigh these clues together and give your best geographic estimate. Answer with ONLY one JSON object in exactly this shape: {"lat": <number>, "lon": <number>, "confidence": <number from 0 to 1>, "country": "<name or empty string>"}.`;

const SECOND_PROMPT = `Re-examine this same street-level photo, concentrating on text, signs, license plates, road markings, and road furniture. Check whether the first estimate's region fits those clues; correct it if the visual evidence points elsewhere. Answer with ONLY one JSON object in exactly this shape: {"lat": <number>, "lon": <number>, "confidence": <number from 0 to 1>, "country": "<name or empty string>"}.`;

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface PhotoList {
  photos: Photo[];
  answered: number;
}

interface Estimate {
  lat: number;
  lon: number;
  confidence: number;
  country: string;
}

interface Answer {
  id: string;
  lat: number;
  lon: number;
}

interface ModelResponse {
  content?: { type: string; text?: string }[];
}

function need(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === '') throw new Error(`${name} is not set`);
  return value;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = need('ANTHROPIC_BASE_URL').replace(/\/+$/, '');
const apiKey = need('ANTHROPIC_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));
const isLat = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= -90 && value <= 90;
const isLon = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= -180 && value <= 180;

function estimateFromObject(value: Record<string, unknown>): Estimate | null {
  const lat = value.lat ?? value.latitude;
  const lon = value.lon ?? value.lng ?? value.longitude;
  if (!isLat(lat) || !isLon(lon)) return null;
  const confidence = value.confidence;
  return {
    lat,
    lon,
    confidence: typeof confidence === 'number' && Number.isFinite(confidence) && confidence >= 0 && confidence <= 1
      ? confidence
      : 0,
    country: typeof value.country === 'string' ? value.country : '',
  };
}

/** Read the first valid coordinate object, otherwise the first in-range numeric pair. */
function parseEstimate(text: string): Estimate | null {
  for (const match of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const parsed = JSON.parse(match[0]) as Record<string, unknown>;
      const estimate = estimateFromObject(parsed);
      if (estimate !== null) return estimate;
    } catch {
      // Continue to another object or the numeric-pair fallback.
    }
  }

  const numbers = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((match) => Number(match[0]));
  for (let i = 0; i + 1 < numbers.length; i += 1) {
    const lat = numbers[i];
    const lon = numbers[i + 1];
    if (isLat(lat) && isLon(lon)) return { lat, lon, confidence: 0, country: '' };
  }
  return null;
}

/** Call the gateway; only transient HTTP statuses get up to two fixed-backoff retries. */
async function ask(jpeg: Buffer, prompt: string): Promise<string | null> {
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
        { type: 'text', text: prompt },
      ],
    }],
  });

  for (let attempt = 0; ; attempt += 1) {
    const response = await fetch(`${anthropicUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body,
    });

    if (response.ok) {
      const message = await response.json() as ModelResponse;
      return (message.content ?? [])
        .filter((block) => block.type === 'text')
        .map((block) => block.text ?? '')
        .join('\n');
    }

    const detail = await response.text();
    const refusal = response.status === 401
      || (response.status >= 400 && response.status < 500 && response.headers.get('x-should-retry') === 'false');
    let noModelAvailable = false;
    try {
      const error = JSON.parse(detail) as { error?: { type?: unknown; message?: unknown } };
      noModelAvailable = error.error?.type === 'upstream_unreachable'
        && typeof error.error.message === 'string'
        && error.error.message.includes('no model is behind the gateway');
    } catch {
      // Other error bodies are handled by their status and retry headers.
    }
    if (!refusal
      && !noModelAvailable
      && RETRY_STATUSES.includes(response.status)
      && response.headers.get('x-should-retry') !== 'false'
      && attempt < MAX_RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }

    process.stderr.write(`model call failed (HTTP ${response.status}): ${detail.slice(0, 300)}\n`);
    return null;
  }
}

async function fetchPhoto(photo: Photo): Promise<Buffer> {
  const response = await fetch(`${envUrl}${photo.url}`);
  if (!response.ok) throw new Error(`photo ${photo.id} fetch failed: HTTP ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}

function distanceKm(a: Estimate, b: Estimate): number {
  const radians = (degrees: number): number => degrees * Math.PI / 180;
  const lat1 = radians(a.lat);
  const lat2 = radians(b.lat);
  const dLat = lat2 - lat1;
  const dLon = radians(b.lon - a.lon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

function sphericalMean(a: Estimate, b: Estimate): Estimate {
  const radians = (degrees: number): number => degrees * Math.PI / 180;
  const toVector = (point: Estimate): [number, number, number] => {
    const lat = radians(point.lat);
    const lon = radians(point.lon);
    return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
  };
  const av = toVector(a);
  const bv = toVector(b);
  const x = av[0] + bv[0];
  const y = av[1] + bv[1];
  const z = av[2] + bv[2];
  return {
    lat: Math.atan2(z, Math.sqrt(x * x + y * y)) * 180 / Math.PI,
    lon: Math.atan2(y, x) * 180 / Math.PI,
    confidence: (a.confidence + b.confidence) / 2,
    country: a.country === b.country ? a.country : (b.country || a.country),
  };
}

function mergeEstimates(first: Estimate | null, second: Estimate | null): Estimate | null {
  if (first === null) return second;
  if (second === null) return first;
  if (distanceKm(first, second) <= 200) return sphericalMean(first, second);
  return first.confidence > second.confidence ? first : second;
}

function roundedCoordinate(value: number): number {
  const rounded = Math.round(value * 1_000_000) / 1_000_000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

async function postAnswers(answers: Answer[], alreadyAnswered: number): Promise<void> {
  for (let offset = 0; offset < answers.length; offset += 5) {
    const chunk = answers.slice(offset, offset + 5);
    const response = await fetch(`${envUrl}/answers`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ answers: chunk }),
    });
    const detail = await response.text();
    if (!response.ok) throw new Error(`answer post failed (HTTP ${response.status}): ${detail}`);

    let result: { accepted?: unknown; answered?: unknown };
    try {
      result = JSON.parse(detail) as { accepted?: unknown; answered?: unknown };
    } catch {
      throw new Error(`answer post returned invalid JSON: ${detail}`);
    }
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
  const listResponse = await fetch(`${envUrl}/photos`);
  if (!listResponse.ok) throw new Error(`photo list failed: HTTP ${listResponse.status}`);
  const listing = await listResponse.json() as PhotoList;
  if (!Array.isArray(listing.photos)) throw new Error('photo list did not contain a photos array');
  const photos = listing.photos.filter((photo) => !photo.answered);
  const jpegs: (Buffer | null)[] = new Array(photos.length).fill(null);
  const firstEstimates: (Estimate | null)[] = new Array(photos.length).fill(null);

  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const index = next;
      next += 1;
      const photo = photos[index]!;
      const jpeg = await fetchPhoto(photo);
      jpegs[index] = jpeg;
      const text = await ask(jpeg, FIRST_PROMPT);
      if (text !== null) firstEstimates[index] = parseEstimate(text);
      if (firstEstimates[index] === null) {
        process.stderr.write(`no first-pass coordinates parsed for ${photo.id}\n`);
      }
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

  const finalEstimates = [...firstEstimates];
  for (let index = 0; index < photos.length; index += 1) {
    const first = firstEstimates[index];
    if (first !== null && first.confidence >= 0.55) continue;

    const priorDescription = first === null
      ? 'No valid coordinates could be parsed from the first-pass answer.'
      : `First estimate: ${JSON.stringify({
        lat: first.lat,
        lon: first.lon,
        confidence: first.confidence,
        country: first.country,
      })}`;
    const text = await ask(jpegs[index]!, `${SECOND_PROMPT}\n${priorDescription}`);
    const second = text === null ? null : parseEstimate(text);
    if (text !== null && second === null) {
      process.stderr.write(`no refinement coordinates parsed for ${photos[index]!.id}\n`);
    }
    finalEstimates[index] = mergeEstimates(first, second);
  }

  const answers = photos.map((photo, index): Answer => {
    const estimate = finalEstimates[index] ?? PRIOR;
    return {
      id: photo.id,
      lat: roundedCoordinate(estimate.lat),
      lon: roundedCoordinate(estimate.lon),
    };
  });
  await postAnswers(answers, listing.answered ?? 0);
  process.stdout.write(`finished: ${answers.length} photos processed\n`);
}

try {
  await main();
} catch (error) {
  process.stderr.write(`agent: ${(error as Error).stack ?? String(error)}\n`);
  process.exitCode = 1;
}
