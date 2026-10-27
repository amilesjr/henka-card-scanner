// Henka Crop: the work for one photo, off the page's main thread.
// Decodes the photo with the browser's own decoder (upright, sRGB), finds
// the card with the same detector as the PC app, straightens it and lays it
// out as a square listing photo. Nothing on the card is redrawn or cleaned
// up: the warp only resamples the photo's own pixels.

importScripts("engine.js?v=202610270306");

// A 2x1 JPEG tagged "rotate 90° clockwise": decoded 1x2 if the browser
// applies EXIF orientation by itself.
const ORIENT_TEST =
  "/9j/4AAQSkZJRgABAQAAAQABAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYAAAAAAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8iviD/wAj9rn/AGGLn/0a1FFFf7pcCf8AJD5X/wBg1D/01E+Q4z/5LDMf+v8AW/8ATkj/2Q==";
let autoOrient = null;

// Settings from the PC app (config.DEFAULTS).
const S = {
  workLongSide: 1600,
  minConfidence: 0.75,
  marginPct: 0.03,
  reviewMarginPct: 0.06,
  listing: { canvas: "square", long_side: 2000, card_fill: 0.86, quality: 0.92 },
  // Phones limit image memory: 24 and 48 MP photos are decoded at this size.
  maxPixels: 16000000,
};

async function browserOrients() {
  if (autoOrient === null) {
    const bytes = Uint8Array.from(atob(ORIENT_TEST), (ch) => ch.charCodeAt(0));
    const bmp = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }));
    autoOrient = bmp.width === 1 && bmp.height === 2;
    bmp.close();
  }
  return autoOrient;
}

function canvas(w, h) {
  const cv = new OffscreenCanvas(w, h);
  return [cv, cv.getContext("2d", { willReadFrequently: true })];
}

/** The photo as an upright sRGB image {w, h, c: 3, data}. */
async function decode(blob, orientation) {
  let bmp;
  try {
    bmp = await createImageBitmap(blob);
  } catch {
    throw new Error("This browser can't open this file.");
  }
  const fix = !(await browserOrients()) && orientation > 1 && orientation <= 8 ? orientation : 1;
  const swap = fix >= 5;
  const scale = Math.min(1, Math.sqrt(S.maxPixels / (bmp.width * bmp.height)));
  const sw = Math.max(1, Math.round(bmp.width * scale));
  const sh = Math.max(1, Math.round(bmp.height * scale));
  const [w, h] = swap ? [sh, sw] : [sw, sh];
  const [, ctx] = canvas(w, h);
  ctx.imageSmoothingQuality = "high";
  const T = { 2: [-1, 0, 0, 1, sw, 0], 3: [-1, 0, 0, -1, sw, sh], 4: [1, 0, 0, -1, 0, sh], 5: [0, 1, 1, 0, 0, 0], 6: [0, 1, -1, 0, sh, 0], 7: [0, -1, -1, 0, sh, sw], 8: [0, -1, 1, 0, 0, sw] }[fix];
  if (T) ctx.setTransform(...T);
  ctx.drawImage(bmp, 0, 0, sw, sh);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const data = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    data[3 * i] = rgba[j];
    data[3 * i + 1] = rgba[j + 1];
    data[3 * i + 2] = rgba[j + 2];
  }
  return { w, h, c: 3, data };
}

/** JPEG of an RGB image, optionally scaled down to longSide. */
async function jpeg(img, quality, longSide = 0) {
  const { w, h } = img;
  let [cv, ctx] = canvas(w, h);
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    rgba[j] = img.data[3 * i];
    rgba[j + 1] = img.data[3 * i + 1];
    rgba[j + 2] = img.data[3 * i + 2];
    rgba[j + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
  if (longSide && Math.max(w, h) > longSide) {
    const s = longSide / Math.max(w, h);
    const [cv2, ctx2] = canvas(Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s)));
    ctx2.imageSmoothingQuality = "high";
    ctx2.drawImage(cv, 0, 0, cv2.width, cv2.height);
    cv = cv2;
  }
  return cv.convertToBlob({ type: "image/jpeg", quality });
}

/** EXIF (TIFF bytes) saying only when the original photo was taken, so
 * Photos files the listing photo next to it, in shooting order. No
 * location, no camera details. Null when the original doesn't say. */
function takenExif(iso) {
  const m = /^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?$/.exec(iso || "");
  if (!m) return null;
  const ascii = (s) => [...s].map((c) => c.charCodeAt(0)).concat(0);
  const tags = [
    [0x9000, 7, ascii("0232").slice(0, 4)], // ExifVersion
    [0x9003, 2, ascii(`${m[1]}:${m[2]}:${m[3]} ${m[4]}:${m[5]}:${m[6]}`)], // DateTimeOriginal
  ];
  if (m[7]) tags.push([0x9291, 2, ascii(m[7])]); // SubSecTimeOriginal
  // Big-endian TIFF: header, IFD0 holding only the Exif IFD's offset, the
  // Exif IFD, then the values too long to sit in their entries.
  const exifAt = 8 + 2 + 12 + 4;
  let at = exifAt + 2 + 12 * tags.length + 4;
  const b = new Uint8Array(at + tags.reduce((n, [, , v]) => n + (v.length > 4 ? v.length + (v.length & 1) : 0), 0));
  const dv = new DataView(b.buffer);
  b.set([0x4d, 0x4d, 0, 42, 0, 0, 0, 8]);
  dv.setUint16(8, 1);
  dv.setUint16(10, 0x8769);
  dv.setUint16(12, 4);
  dv.setUint32(14, 1);
  dv.setUint32(18, exifAt);
  dv.setUint16(exifAt, tags.length);
  tags.forEach(([tag, type, v], k) => {
    const e = exifAt + 2 + 12 * k;
    dv.setUint16(e, tag);
    dv.setUint16(e + 2, type);
    dv.setUint32(e + 4, v.length);
    if (v.length <= 4) b.set(v, e + 8);
    else {
      dv.setUint32(e + 8, at);
      b.set(v, at);
      at += v.length + (v.length & 1);
    }
  });
  return b;
}

/** The JPEG with an EXIF segment added after its JFIF header. */
async function withExif(blob, tiff) {
  if (!tiff) return blob;
  const head = new Uint8Array(await blob.slice(0, 6).arrayBuffer());
  const after = head[2] === 0xff && head[3] === 0xe0 ? 4 + ((head[4] << 8) | head[5]) : 2;
  const app1 = new Uint8Array(10);
  app1.set([0xff, 0xe1, (8 + tiff.length) >> 8, (8 + tiff.length) & 255, 0x45, 0x78, 0x69, 0x66, 0, 0]);
  return new Blob([blob.slice(0, after), app1, tiff, blob.slice(after)], { type: "image/jpeg" });
}

/** Straighten the card and lay it out on the square listing canvas. */
async function listing(rgb, quad, ok, rotation, fill, aspect, holder, taken) {
  let margin = ok ? S.marginPct : S.reviewMarginPct;
  if (holder) margin = Math.max(margin, HK.crop.holderExtent(quad, holder, aspect) + S.marginPct);
  const crop = HK.crop.rectify(rgb, quad, margin, rotation, fill, aspect);
  const holderPx = holder ? CV.pyround(HK.crop.holderExtent(quad, holder, aspect) * (Math.min(crop.image.w, crop.image.h) - 2 * crop.margin)) : 0;
  const L = S.listing;
  const square = HK.crop.renderListing(crop.image, crop.margin, fill, L.canvas, L.long_side, L.card_fill, holderPx);
  return { full: await withExif(await jpeg(square, L.quality), takenExif(taken)), preview: await jpeg(square, 0.82, 700) };
}

async function readPhoto(file) {
  const meta = HK.exif.read(await file.arrayBuffer());
  return { meta, rgb: await decode(file, meta.orientation) };
}

/** Card proportions from corners placed by hand, corrected for camera tilt
 * when the photo says which lens took it (review._human_aspect). */
function humanAspect(q, focal35, w, h) {
  const f = HK.detect.focalLengthPx(focal35, w, h);
  const a = f ? HK.detect.rectangleAspect(q, f, w / 2, h / 2) : null;
  if (!a) return null;
  return Math.abs(Math.min(a, 1 / a) - 63 / 88) > 0.04 ? null : a;
}

const ops = {
  /** Find the card (or use the corners given) and make the listing photo. */
  async crop(a) {
    const { meta, rgb } = await readPhoto(a.file);
    let det = a.det;
    if (!det) {
      const d = HK.detect.detectCard(rgb, meta.focal35, ["standard"], S.workLongSide, S.minConfidence);
      const blocking = d.flags.filter((f) => HK.detect.BLOCKING_FLAGS.has(HK.detect.flagBase(f)));
      det = {
        status: d.status,
        quad: d.quad,
        holder: d.holder_quad,
        aspect: d.aspect,
        fill: d.background_rgb,
        sideways: d.flags.includes("card_sideways"),
        problems: blocking.map(HK.detect.describeFlag),
      };
    }
    if (a.quad) {
      const q = HK.detect.orderQuad(a.quad);
      det = { ...det, quad: q, holder: null, aspect: humanAspect(q, meta.focal35, rgb.w, rgb.h), status: "fixed", problems: [] };
    }
    const out = { width: rgb.w, height: rgb.h, det };
    if (!det.quad) return out;
    Object.assign(out, await listing(rgb, det.quad, det.status !== "needs_review", a.rotation || 0, det.fill || [0, 0, 0], det.aspect, det.holder, meta.captured_at));
    return out;
  },

  /** The whole photo, upright and smaller, for placing corners by hand. */
  async original(a) {
    const { rgb } = await readPhoto(a.file);
    return { width: rgb.w, height: rgb.h, image: await jpeg(rgb, 0.85, 1600) };
  },
};

self.onmessage = async (e) => {
  const { id, op, args } = e.data;
  try {
    self.postMessage({ id, ok: true, out: await ops[op](args) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
