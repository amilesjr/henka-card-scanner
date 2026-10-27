// Photo metadata straight from the file bytes: capture time, camera, 35 mm
// focal length, orientation and pixel size. JPEG (APP1 EXIF), HEIC/HEIF
// (the "Exif" item) and PNG (size only). Mirrors what henka_photos/imaging.py
// reads with Pillow. Nothing here decodes or changes the image.

var HK = (globalThis.HK = globalThis.HK || {});

HK.exif = (() => {
  "use strict";

  const TAG = {
    make: 0x010f,
    model: 0x0110,
    orientation: 0x0112,
    dateTime: 0x0132,
    exifIfd: 0x8769,
    dateTimeOriginal: 0x9003,
    subSecOriginal: 0x9291,
    focal35: 0xa405,
  };

  function ascii(dv, off, n) {
    let s = "";
    for (let i = 0; i < n; i++) {
      const c = dv.getUint8(off + i);
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s;
  }

  /** Tags of one IFD as {tag: value}; strings, numbers or arrays. */
  function readIfd(dv, tiff, off, le) {
    const out = {};
    if (off < 0 || tiff + off + 2 > dv.byteLength) return out;
    const n = dv.getUint16(tiff + off, le);
    const SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };
    for (let i = 0; i < n; i++) {
      const e = tiff + off + 2 + 12 * i;
      if (e + 12 > dv.byteLength) break;
      const tag = dv.getUint16(e, le);
      const type = dv.getUint16(e + 2, le);
      const count = dv.getUint32(e + 4, le);
      const size = (SIZE[type] || 1) * count;
      const at = size <= 4 ? e + 8 : tiff + dv.getUint32(e + 8, le);
      if (at + size > dv.byteLength) continue;
      let v;
      if (type === 2) v = ascii(dv, at, count);
      else if (type === 3) v = count === 1 ? dv.getUint16(at, le) : Array.from({ length: count }, (_, k) => dv.getUint16(at + 2 * k, le));
      else if (type === 4) v = count === 1 ? dv.getUint32(at, le) : Array.from({ length: count }, (_, k) => dv.getUint32(at + 4 * k, le));
      else if (type === 5) v = dv.getUint32(at + 4, le) ? dv.getUint32(at, le) / dv.getUint32(at + 4, le) : 0;
      else if (type === 7 || type === 1) v = Array.from({ length: Math.min(count, 64) }, (_, k) => dv.getUint8(at + k));
      else continue;
      out[tag] = v;
    }
    return out;
  }

  function parseTiff(dv, tiff) {
    if (tiff + 8 > dv.byteLength) return null;
    const order = dv.getUint16(tiff);
    const le = order === 0x4949;
    if (!le && order !== 0x4d4d) return null;
    const ifd0 = readIfd(dv, tiff, dv.getUint32(tiff + 4, le), le);
    const exif = ifd0[TAG.exifIfd] ? readIfd(dv, tiff, ifd0[TAG.exifIfd], le) : {};
    return { ifd0, exif };
  }

  function jpeg(dv) {
    let tiff = null;
    let width = null;
    let height = null;
    let p = 2;
    while (p + 4 <= dv.byteLength) {
      if (dv.getUint8(p) !== 0xff) {
        p++;
        continue;
      }
      const marker = dv.getUint8(p + 1);
      if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01 || marker === 0xff) {
        p += marker === 0xff ? 1 : 2;
        continue;
      }
      if (marker === 0xd9 || marker === 0xda) break;
      const len = dv.getUint16(p + 2);
      if (marker === 0xe1 && !tiff && ascii(dv, p + 4, 4) === "Exif") tiff = parseTiff(dv, p + 10);
      const sof = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
      if (sof && width === null) {
        height = dv.getUint16(p + 5);
        width = dv.getUint16(p + 7);
      }
      p += 2 + len;
    }
    return { tiff, width, height };
  }

  // ISO base media file (HEIC/HEIF): find the "Exif" item via iinf + iloc,
  // and the image size from ispe.
  function boxes(dv, start, end) {
    const out = [];
    let p = start;
    while (p + 8 <= end) {
      let size = dv.getUint32(p);
      const type = ascii(dv, p + 4, 4);
      let head = 8;
      if (size === 1) {
        size = Number(dv.getBigUint64(p + 8));
        head = 16;
      } else if (size === 0) size = end - p;
      if (size < head || p + size > end) break;
      out.push({ type, start: p + head, end: p + size });
      p += size;
    }
    return out;
  }

  function heif(dv) {
    const top = boxes(dv, 0, dv.byteLength);
    const meta = top.find((b) => b.type === "meta");
    if (!meta) return { tiff: null, width: null, height: null };
    const inner = boxes(dv, meta.start + 4, meta.end);
    let exifId = null;
    const iinf = inner.find((b) => b.type === "iinf");
    if (iinf) {
      const ver = dv.getUint8(iinf.start);
      const first = iinf.start + 4 + (ver === 0 ? 2 : 4);
      for (const infe of boxes(dv, first, iinf.end).filter((b) => b.type === "infe")) {
        const v = dv.getUint8(infe.start);
        if (v < 2) continue;
        const id = v === 2 ? dv.getUint16(infe.start + 4) : dv.getUint32(infe.start + 4);
        const typeAt = infe.start + 4 + (v === 2 ? 2 : 4) + 2;
        if (ascii(dv, typeAt, 4) === "Exif") exifId = id;
      }
    }
    let tiff = null;
    const iloc = inner.find((b) => b.type === "iloc");
    if (iloc && exifId !== null) {
      const ver = dv.getUint8(iloc.start);
      let p = iloc.start + 4;
      const b1 = dv.getUint8(p);
      const b2 = dv.getUint8(p + 1);
      const offSize = b1 >> 4;
      const lenSize = b1 & 15;
      const baseSize = b2 >> 4;
      const idxSize = ver === 1 || ver === 2 ? b2 & 15 : 0;
      p += 2;
      const read = (n) => {
        let v = 0;
        for (let i = 0; i < n; i++) v = v * 256 + dv.getUint8(p + i);
        p += n;
        return v;
      };
      const count = ver < 2 ? read(2) : read(4);
      for (let i = 0; i < count && !tiff; i++) {
        const id = ver < 2 ? read(2) : read(4);
        if (ver === 1 || ver === 2) read(2); // construction method
        read(2); // data reference index
        const base = read(baseSize);
        const extents = read(2);
        for (let k = 0; k < extents; k++) {
          if (idxSize) read(idxSize);
          const off = base + read(offSize);
          const len = read(lenSize);
          if (id === exifId && k === 0 && off + 4 <= dv.byteLength) {
            const skip = dv.getUint32(off);
            tiff = parseTiff(dv, off + 4 + skip);
            void len;
          }
        }
      }
    }
    let width = null;
    let height = null;
    const iprp = inner.find((b) => b.type === "iprp");
    if (iprp) {
      const ipco = boxes(dv, iprp.start, iprp.end).find((b) => b.type === "ipco");
      if (ipco) {
        let turned = false;
        for (const b of boxes(dv, ipco.start, ipco.end)) {
          if (b.type === "irot") turned = (dv.getUint8(b.start) & 3) % 2 === 1;
          if (b.type !== "ispe") continue;
          const w = dv.getUint32(b.start + 4);
          const h = dv.getUint32(b.start + 8);
          if (!width || w * h > width * height) {
            width = w;
            height = h;
          }
        }
        // HEIF's own rotation is always applied by decoders; report the
        // upright size.
        if (turned && width) [width, height] = [height, width];
      }
    }
    return { tiff, width, height };
  }

  function png(dv) {
    if (dv.byteLength < 24) return { tiff: null, width: null, height: null };
    return { tiff: null, width: dv.getUint32(16), height: dv.getUint32(20) };
  }

  /** "2025:10:01 12:34:56" (+ sub-seconds) -> "2025-10-01T12:34:56.12",
   * like imaging._parse_exif_time. */
  function exifTime(value, subsec) {
    if (!value || typeof value !== "string") return null;
    const m = /^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})$/.exec(value.trim());
    if (!m) return null;
    const [, Y, Mo, D, h, mi, s] = m;
    const dt = new Date(Date.UTC(+Y, +Mo - 1, +D, +h, +mi, +s));
    if (isNaN(dt) || dt.getUTCMonth() !== +Mo - 1 || dt.getUTCDate() !== +D || +h > 23 || +mi > 59 || +s > 59) return null;
    let iso = `${Y}-${Mo}-${D}T${h}:${mi}:${s}`;
    if (subsec !== undefined && subsec !== null) {
      const digits = String(subsec).replace(/\D/g, "");
      if (digits) iso += "." + digits.slice(0, 6);
    }
    return iso;
  }

  /** Metadata of a photo file (ArrayBuffer). */
  function read(buffer) {
    const dv = new DataView(buffer);
    let kind = "unknown";
    let r = { tiff: null, width: null, height: null };
    if (dv.byteLength > 3 && dv.getUint16(0) === 0xffd8) {
      kind = "jpeg";
      r = jpeg(dv);
    } else if (dv.byteLength > 12 && ascii(dv, 4, 4) === "ftyp") {
      kind = "heif";
      r = heif(dv);
    } else if (dv.byteLength > 8 && dv.getUint32(0) === 0x89504e47) {
      kind = "png";
      r = png(dv);
    }
    const ifd0 = (r.tiff && r.tiff.ifd0) || {};
    const ex = (r.tiff && r.tiff.exif) || {};
    let captured = exifTime(ex[TAG.dateTimeOriginal], ex[TAG.subSecOriginal]);
    const source = captured ? "exif" : null;
    if (!captured) captured = exifTime(ifd0[TAG.dateTime], null);
    const make = String(ifd0[TAG.make] || "").replace(/[\0 ]+$/, "").replace(/^[\0 ]+/, "");
    const model = String(ifd0[TAG.model] || "").replace(/[\0 ]+$/, "").replace(/^[\0 ]+/, "");
    const camera = make && model.toLowerCase().startsWith(make.toLowerCase()) ? model : [make, model].filter(Boolean).join(" ");
    const f = ex[TAG.focal35];
    return {
      kind,
      width: r.width,
      height: r.height,
      // In HEIF the container's rotation is authoritative and decoders apply
      // it; the EXIF tag is informational only.
      orientation: kind === "heif" ? 1 : ifd0[TAG.orientation] || 1,
      captured_at: captured,
      captured_at_source: captured ? source || "exif" : "none",
      camera,
      focal35: f ? Number(f) : null,
    };
  }

  return { read, exifTime };
})();


