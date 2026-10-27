// Minimal ZIP files for exports and backups. Entries are stored without
// compression (photos are already compressed JPEGs), so writing is fast and
// needs no library. The reader opens ZIPs this app wrote.

var HK = (globalThis.HK = globalThis.HK || {});

HK.zip = (() => {
  "use strict";

  const CRC = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    CRC[n] = c >>> 0;
  }

  function crc32(bytes, crc = 0) {
    crc = ~crc >>> 0;
    for (let i = 0; i < bytes.length; i++) crc = CRC[(crc ^ bytes[i]) & 255] ^ (crc >>> 8);
    return ~crc >>> 0;
  }

  function dosTime(d) {
    return {
      time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
      date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
    };
  }

  /** Build a ZIP Blob from [{name, data: Blob|string|Uint8Array}]. */
  async function write(entries, onProgress = () => {}) {
    const enc = new TextEncoder();
    const parts = [];
    const central = [];
    let offset = 0;
    const { time, date } = dosTime(new Date());
    let i = 0;
    for (const e of entries) {
      const name = enc.encode(e.name);
      const blob = e.data instanceof Blob ? e.data : new Blob([typeof e.data === "string" ? enc.encode(e.data) : e.data]);
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const crc = crc32(bytes);
      const size = bytes.length;
      if (offset + size > 0xfffffff0) throw new Error("The export is larger than 4 GB. Export fewer cards at a time.");
      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true);
      local.setUint16(4, 20, true);
      local.setUint16(6, 0x0800, true); // UTF-8 names
      local.setUint16(8, 0, true); // stored
      local.setUint16(10, time, true);
      local.setUint16(12, date, true);
      local.setUint32(14, crc, true);
      local.setUint32(18, size, true);
      local.setUint32(22, size, true);
      local.setUint16(26, name.length, true);
      local.setUint16(28, 0, true);
      parts.push(local.buffer, name, blob);
      const cen = new DataView(new ArrayBuffer(46));
      cen.setUint32(0, 0x02014b50, true);
      cen.setUint16(4, 20, true);
      cen.setUint16(6, 20, true);
      cen.setUint16(8, 0x0800, true);
      cen.setUint16(10, 0, true);
      cen.setUint16(12, time, true);
      cen.setUint16(14, date, true);
      cen.setUint32(16, crc, true);
      cen.setUint32(20, size, true);
      cen.setUint32(24, size, true);
      cen.setUint16(28, name.length, true);
      cen.setUint32(42, offset, true);
      central.push(cen.buffer, name);
      offset += 30 + name.length + size;
      onProgress(++i, entries.length);
    }
    const cenSize = central.reduce((s, p) => s + p.byteLength, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, entries.length, true);
    end.setUint16(10, entries.length, true);
    end.setUint32(12, cenSize, true);
    end.setUint32(16, offset, true);
    return new Blob([...parts, ...central, end.buffer], { type: "application/zip" });
  }

  /** Entries of a stored (uncompressed) ZIP: [{name, blob}]. */
  async function read(file) {
    const tailSize = Math.min(file.size, 65557);
    const tail = new DataView(await file.slice(file.size - tailSize).arrayBuffer());
    let eocd = -1;
    for (let p = tail.byteLength - 22; p >= 0; p--) {
      if (tail.getUint32(p, true) === 0x06054b50) {
        eocd = p;
        break;
      }
    }
    if (eocd < 0) throw new Error("That file isn't a ZIP backup from this app.");
    const count = tail.getUint16(eocd + 10, true);
    const cenSize = tail.getUint32(eocd + 12, true);
    const cenOff = tail.getUint32(eocd + 16, true);
    const cen = new DataView(await file.slice(cenOff, cenOff + cenSize).arrayBuffer());
    const dec = new TextDecoder();
    const out = [];
    let p = 0;
    for (let i = 0; i < count; i++) {
      if (cen.getUint32(p, true) !== 0x02014b50) throw new Error("The backup file is damaged.");
      const method = cen.getUint16(p + 10, true);
      const size = cen.getUint32(p + 20, true);
      const nameLen = cen.getUint16(p + 28, true);
      const extraLen = cen.getUint16(p + 30, true);
      const commentLen = cen.getUint16(p + 32, true);
      const localOff = cen.getUint32(p + 42, true);
      const name = dec.decode(new Uint8Array(cen.buffer, cen.byteOffset + p + 46, nameLen));
      if (method !== 0) throw new Error("This ZIP was compressed by another program; use the backup file exactly as the app saved it.");
      const head = new DataView(await file.slice(localOff, localOff + 30).arrayBuffer());
      const start = localOff + 30 + head.getUint16(26, true) + head.getUint16(28, true);
      out.push({ name, blob: file.slice(start, start + size) });
      p += 46 + nameLen + extraLen + commentLen;
    }
    return out;
  }

  return { write, read, crc32 };
})();

if (typeof module !== "undefined") module.exports = HK.zip;
