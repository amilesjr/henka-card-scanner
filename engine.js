// Henka Crop engine (built 202610270306): card detection and straightening.
// The handful of OpenCV operations the Henka engine uses, in plain
// JavaScript so the phone app needs no native code or WebAssembly.
//
// Each function follows OpenCV's behavior closely (kernel sizes, border
// rules, the 8-bit Lab conversion, contour tracing) so results match the
// PC version, which is checked by the parity test in mobile/tests.
//
// Images are {w, h, c, data}: row-major, channels interleaved, data a typed
// array. Masks are single-channel Uint8Array images holding 0 or 1.

const CV = (() => {
  "use strict";

  function image(w, h, c = 1, Type = Float32Array) {
    return { w, h, c, data: new Type(w * h * c) };
  }

  // ------------------------------------------------------------ numbers

  function select(a, k) {
    // In-place quickselect: a[k] becomes the k-th smallest.
    let lo = 0;
    let hi = a.length - 1;
    while (hi > lo) {
      const mid = (lo + hi) >> 1;
      if (a[mid] < a[lo]) [a[mid], a[lo]] = [a[lo], a[mid]];
      if (a[hi] < a[lo]) [a[hi], a[lo]] = [a[lo], a[hi]];
      if (a[hi] < a[mid]) [a[hi], a[mid]] = [a[mid], a[hi]];
      const pivot = a[mid];
      let i = lo;
      let j = hi;
      while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) {
          const t = a[i];
          a[i] = a[j];
          a[j] = t;
          i++;
          j--;
        }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else break;
    }
    return a[k];
  }

  /** numpy.percentile with linear interpolation (the default). */
  function percentile(arr, p) {
    const n = arr.length;
    if (!n) return NaN;
    if (arr instanceof Uint8Array || arr instanceof Uint8ClampedArray) return percentile8(arr, p);
    const a = Float64Array.from(arr);
    const idx = (p / 100) * (n - 1);
    const lo = Math.floor(idx);
    const vlo = select(a, lo);
    if (lo + 1 >= n || idx === lo) return vlo;
    let vhi = Infinity;
    for (let i = lo + 1; i < n; i++) if (a[i] < vhi) vhi = a[i];
    return vlo + (vhi - vlo) * (idx - lo);
  }

  function percentile8(arr, p) {
    const hist = new Float64Array(256);
    for (let i = 0; i < arr.length; i++) hist[arr[i]]++;
    const idx = (p / 100) * (arr.length - 1);
    const kth = (k) => {
      let acc = 0;
      for (let v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc > k) return v;
      }
      return 255;
    };
    const lo = Math.floor(idx);
    const vlo = kth(lo);
    if (idx === lo || lo + 1 >= arr.length) return vlo;
    return vlo + (kth(lo + 1) - vlo) * (idx - lo);
  }

  function median(arr) {
    return percentile(arr, 50);
  }

  function mean(arr) {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return arr.length ? s / arr.length : NaN;
  }

  /** Python round(): halves go to the even neighbor. */
  function pyround(x) {
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  // ------------------------------------------------------------ color

  // OpenCV converts 8-bit RGB to Lab in fixed point with lookup tables
  // (color_lab.cpp, RGB2Lab_b); reproduced here so results match exactly.
  const LAB_SHIFT = 12;
  const GAMMA_SHIFT = 3;
  const LAB_SHIFT2 = LAB_SHIFT + GAMMA_SHIFT;
  const GAMMA_TAB = new Int32Array(256);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    const lin = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
    GAMMA_TAB[i] = Math.round(255 * (1 << GAMMA_SHIFT) * lin);
  }
  const CBRT_TAB = new Int32Array(256 * 3 / 2 * (1 << GAMMA_SHIFT));
  for (let i = 0; i < CBRT_TAB.length; i++) {
    const x = i * (1 / (255 * (1 << GAMMA_SHIFT)));
    CBRT_TAB[i] = Math.round((1 << LAB_SHIFT2) * (x < 0.008856 ? x * 7.787 + 0.13793103448275862 : Math.cbrt(x)));
  }
  // OpenCV builds this table in its own software float arithmetic, which
  // rounds these two entries the other way. With them, every one of the
  // 16.7 million 8-bit colors converts exactly as cv2.cvtColor does.
  CBRT_TAB[49] -= 1;
  CBRT_TAB[628] += 1;
  const XYZ = [0.412453, 0.35758, 0.180423, 0.212671, 0.71516, 0.072169, 0.019334, 0.119193, 0.950227];
  const WHITE = [0.950456, 1.0, 1.088754];
  const LAB_C = XYZ.map((v, k) => Math.round((v * (1 << LAB_SHIFT)) / WHITE[Math.floor(k / 3)]));
  const L_SCALE = Math.floor((116 * 255 + 50) / 100);
  const L_SHIFT = -Math.floor((16 * 255 * (1 << LAB_SHIFT2) + 50) / 100);
  const descale = (x, n) => (x + (1 << (n - 1))) >> n;
  const sat8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

  /** RGB(A) uint8 -> Lab in real units (L 0..100, a/b around 0), as
   * cv2.cvtColor(..., COLOR_RGB2LAB) computes it for 8-bit images. */
  function rgbToLab(src) {
    const { w, h, c } = src;
    const out = image(w, h, 3);
    const s = src.data;
    const o = out.data;
    const C = LAB_C;
    for (let i = 0, j = 0; i < w * h; i++, j += c) {
      const R = GAMMA_TAB[s[j]];
      const G = GAMMA_TAB[s[j + 1]];
      const B = GAMMA_TAB[s[j + 2]];
      const fX = CBRT_TAB[descale(R * C[0] + G * C[1] + B * C[2], LAB_SHIFT)];
      const fY = CBRT_TAB[descale(R * C[3] + G * C[4] + B * C[5], LAB_SHIFT)];
      const fZ = CBRT_TAB[descale(R * C[6] + G * C[7] + B * C[8], LAB_SHIFT)];
      const L = sat8(descale(L_SCALE * fY + L_SHIFT, LAB_SHIFT2));
      const A = sat8(descale(500 * (fX - fY) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2));
      const Bb = sat8(descale(200 * (fY - fZ) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2));
      o[3 * i] = (L * 100) / 255;
      o[3 * i + 1] = A - 128;
      o[3 * i + 2] = Bb - 128;
    }
    return out;
  }

  /** The same conversion with OpenCV's raw 8-bit Lab output (L*2.55, a+128, b+128). */
  function rgbToLab8(src) {
    const lab = rgbToLab(src);
    const out = image(src.w, src.h, 3, Uint8Array);
    for (let i = 0; i < src.w * src.h; i++) {
      out.data[3 * i] = Math.round((lab.data[3 * i] * 255) / 100);
      out.data[3 * i + 1] = lab.data[3 * i + 1] + 128;
      out.data[3 * i + 2] = lab.data[3 * i + 2] + 128;
    }
    return out;
  }

  function dropAlpha(src) {
    if (src.c === 3) return src;
    const out = image(src.w, src.h, 3, Uint8Array);
    for (let i = 0, j = 0; i < src.w * src.h; i++, j += src.c) {
      out.data[3 * i] = src.data[j];
      out.data[3 * i + 1] = src.data[j + 1];
      out.data[3 * i + 2] = src.data[j + 2];
    }
    return out;
  }

  function channel(src, k) {
    const out = image(src.w, src.h, 1, src.data.constructor);
    for (let i = 0; i < src.w * src.h; i++) out.data[i] = src.data[i * src.c + k];
    return out;
  }

  // ------------------------------------------------------------ resizing

  // cv2.resize(..., INTER_AREA), following resize.cpp: whole-number scales
  // average blocks (integer sums for 8-bit images), other scales use
  // OpenCV's weight tables with float32 arithmetic.
  function areaTab(ssize, dsize, scale) {
    const si = [];
    const di = [];
    const alpha = [];
    const f = Math.fround;
    for (let dx = 0; dx < dsize; dx++) {
      const fsx1 = dx * scale;
      const fsx2 = fsx1 + scale;
      const cellWidth = Math.min(scale, ssize - fsx1);
      let sx1 = Math.ceil(fsx1);
      let sx2 = Math.floor(fsx2);
      sx2 = Math.min(sx2, ssize - 1);
      sx1 = Math.min(sx1, sx2);
      if (sx1 - fsx1 > 1e-3) {
        di.push(dx);
        si.push(sx1 - 1);
        alpha.push(f((sx1 - fsx1) / cellWidth));
      }
      for (let sx = sx1; sx < sx2; sx++) {
        di.push(dx);
        si.push(sx);
        alpha.push(f(1.0 / cellWidth));
      }
      if (fsx2 - sx2 > 1e-3) {
        di.push(dx);
        si.push(sx2);
        alpha.push(f(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth));
      }
    }
    return { si: Int32Array.from(si), di: Int32Array.from(di), alpha: Float32Array.from(alpha) };
  }

  function resizeAreaFast(src, dw, dh, sx, sy, isInt) {
    const { w, h, c } = src;
    const sd = src.data;
    const f = Math.fround;
    const out = image(dw, dh, c, isInt ? Uint8Array : Float32Array);
    const od = out.data;
    const area = sx * sy;
    const scale = f(1 / area);
    const dwidth1 = Math.floor(w / sx) * c;
    const dwc = dw * c;
    const store = (v) => (isInt ? Math.min(255, Math.max(0, rint(v))) : v);
    const fast22 = isInt && sx === 2 && sy === 2 && (c === 1 || c === 3 || c === 4);
    for (let dy = 0; dy < dh; dy++) {
      const ob = dy * dwc;
      const sy0 = dy * sy;
      if (sy0 >= h) continue; // zeros
      const lim = sy0 + sy <= h ? dwidth1 : 0;
      let dx = 0;
      for (; dx < lim; dx++) {
        const ch = dx % c;
        const x0 = ((dx - ch) / c) * sx;
        if (fast22) {
          const a = (sy0 * w + x0) * c + ch;
          od[ob + dx] = (sd[a] + sd[a + c] + sd[a + w * c] + sd[a + w * c + c] + 2) >> 2;
          continue;
        }
        let sum = 0;
        let k = 0;
        const at = (q) => sd[((sy0 + Math.floor(q / sx)) * w + x0 + (q % sx)) * c + ch];
        if (isInt) {
          for (; k < area; k++) sum += at(k);
          od[ob + dx] = store(f(sum * scale));
        } else {
          for (; k <= area - 4; k += 4) sum = f(sum + f(f(f(at(k) + at(k + 1)) + at(k + 2)) + at(k + 3)));
          for (; k < area; k++) sum = f(sum + at(k));
          od[ob + dx] = f(sum * scale);
        }
      }
      for (; dx < dwc; dx++) {
        const ch = dx % c;
        const x0 = ((dx - ch) / c) * sx;
        let sum = 0;
        let count = 0;
        for (let yy = 0; yy < sy && sy0 + yy < h; yy++) {
          for (let xx = 0; xx < sx && x0 + xx < w; xx++) {
            const v = sd[((sy0 + yy) * w + x0 + xx) * c + ch];
            sum = isInt ? sum + v : f(sum + v);
            count++;
          }
        }
        od[ob + dx] = count ? store(f(f(sum) / count)) : 0;
      }
    }
    return out;
  }

  /** cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA) for shrinking. */
  function resizeArea(src, dw, dh) {
    const { w, h, c } = src;
    const isInt = !(src.data instanceof Float32Array || src.data instanceof Float64Array);
    const scaleX = 1 / (dw / w);
    const scaleY = 1 / (dh / h);
    if (scaleX < 1 || scaleY < 1) return resizeLinear(src, dw, dh);
    const isx = rint(scaleX);
    const isy = rint(scaleY);
    if (Math.abs(scaleX - isx) < 2.220446049250313e-16 && Math.abs(scaleY - isy) < 2.220446049250313e-16) {
      return resizeAreaFast(src, dw, dh, isx, isy, isInt);
    }
    const f = Math.fround;
    const xt = areaTab(w, dw, scaleX);
    const yt = areaTab(h, dh, scaleY);
    const sd = src.data;
    const dwc = dw * c;
    const out = image(dw, dh, c, isInt ? Uint8Array : Float32Array);
    const od = out.data;
    const buf = new Float32Array(dwc);
    const sum = new Float32Array(dwc);
    const store = (row) => {
      const ob = row * dwc;
      for (let i = 0; i < dwc; i++) od[ob + i] = isInt ? Math.min(255, Math.max(0, rint(sum[i]))) : sum[i];
    };
    let prev = yt.di[0];
    for (let j = 0; j < yt.si.length; j++) {
      const beta = yt.alpha[j];
      const dy = yt.di[j];
      const rowBase = yt.si[j] * w * c;
      buf.fill(0);
      for (let k = 0; k < xt.si.length; k++) {
        const s0 = rowBase + xt.si[k] * c;
        const d0 = xt.di[k] * c;
        const a = xt.alpha[k];
        for (let q = 0; q < c; q++) buf[d0 + q] = f(buf[d0 + q] + f(sd[s0 + q] * a));
      }
      if (dy !== prev) {
        store(prev);
        for (let i = 0; i < dwc; i++) sum[i] = f(beta * buf[i]);
        prev = dy;
      } else {
        for (let i = 0; i < dwc; i++) sum[i] = f(sum[i] + f(beta * buf[i]));
      }
    }
    store(prev);
    return out;
  }

  /** cv2.resize(..., INTER_LINEAR): pixel-center aligned bilinear. */
  function resizeLinear(src, dw, dh) {
    const { w, h, c } = src;
    const out = image(dw, dh, c, Float32Array);
    const sx = w / dw;
    const sy = h / dh;
    for (let y = 0; y < dh; y++) {
      let fy = (y + 0.5) * sy - 0.5;
      fy = Math.min(Math.max(fy, 0), h - 1);
      const y0 = Math.floor(fy);
      const y1 = Math.min(h - 1, y0 + 1);
      const wy = fy - y0;
      for (let x = 0; x < dw; x++) {
        let fx = (x + 0.5) * sx - 0.5;
        fx = Math.min(Math.max(fx, 0), w - 1);
        const x0 = Math.floor(fx);
        const x1 = Math.min(w - 1, x0 + 1);
        const wx = fx - x0;
        for (let k = 0; k < c; k++) {
          const a = src.data[(y0 * w + x0) * c + k];
          const b = src.data[(y0 * w + x1) * c + k];
          const d = src.data[(y1 * w + x0) * c + k];
          const e = src.data[(y1 * w + x1) * c + k];
          out.data[(y * dw + x) * c + k] = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
        }
      }
    }
    return out;
  }

  // ------------------------------------------------------------ filtering

  const reflect101 = (i, n) => {
    if (n === 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
  };

  /** Gaussian weights as OpenCV computes them (getGaussianKernelBitExact):
   * symmetric, normalized by the sum with the center weight exactly 1/sum. */
  function gaussianKernel(sigma, forUint8) {
    const n = Math.max(1, rint(sigma * (forUint8 ? 3 : 4) * 2 + 1) | 1);
    const scale2X = -0.125 / (sigma * sigma);
    const half = (n - 1) >> 1;
    const values = [];
    let sum = 0;
    for (let i = 0, x = 1 - n; i < half; i++, x += 2) {
      const t = Math.exp(x * x * scale2X);
      values.push(t);
      sum += t;
    }
    sum = sum * 2 + 1;
    const mul = 1 / sum;
    const k = new Float64Array(n);
    for (let i = 0; i < half; i++) k[i] = k[n - 1 - i] = values[i] * mul;
    k[half] = mul;
    return k;
  }

  /** The same weights in OpenCV's 8-bit fixed point (8 fraction bits),
   * rounded with error diffusion so they sum to exactly 256. */
  function gaussianKernelFixed(k) {
    const n = k.length;
    const half = n >> 1;
    const out = new Int32Array(n);
    let err = 0;
    let sum = 0;
    for (let i = 0; i < half; i++) {
      const adj = k[i] * 256 + err;
      const v = rint(adj);
      err = adj - v;
      out[i] = out[n - 1 - i] = v;
      sum += v;
    }
    out[half] = 256 - 2 * sum;
    return out;
  }

  /** cv2.GaussianBlur on 8-bit images: OpenCV's bit-exact fixed-point path
   * (rows to 8.8 fixed point, then columns to 16.16, rounded). */
  function gaussianBlurFixed(src, k) {
    const { w, h, c } = src;
    const n = k.length;
    const r = n >> 1;
    const sd = src.data;
    const xi = new Int32Array(w * n);
    for (let x = 0; x < w; x++) for (let t = 0; t < n; t++) xi[x * n + t] = reflect101(x + t - r, w) * c;
    const rowLen = w * c;
    const tmp = new Uint32Array(rowLen * h);
    for (let y = 0; y < h; y++) {
      const row = y * rowLen;
      for (let x = 0; x < w; x++) {
        for (let q = 0; q < c; q++) {
          let acc = 0;
          for (let t = 0; t < n; t++) acc += sd[row + xi[x * n + t] + q] * k[t];
          tmp[row + x * c + q] = acc;
        }
      }
    }
    const out = image(w, h, c, Uint8Array);
    const acc = new Float64Array(rowLen);
    for (let y = 0; y < h; y++) {
      acc.fill(0);
      for (let t = 0; t < n; t++) {
        const base = reflect101(y + t - r, h) * rowLen;
        for (let i = 0; i < rowLen; i++) acc[i] += tmp[base + i] * k[t];
      }
      for (let i = 0; i < rowLen; i++) out.data[y * rowLen + i] = Math.min(255, (acc[i] + 32768) >>> 16);
    }
    return out;
  }

  function convolveSeparable(src, kx, ky, roundToUint8) {
    const { w, h, c } = src;
    const nx = kx.length;
    const ny = ky.length;
    const rx = (nx - 1) / 2;
    const ry = (ny - 1) / 2;
    const sd = src.data;
    // Horizontal pass, row by row, with a precomputed (reflected) index table.
    const xi = new Int32Array(w * nx);
    for (let x = 0; x < w; x++) for (let t = 0; t < nx; t++) xi[x * nx + t] = reflect101(x + t - rx, w) * c;
    const tmp = new Float64Array(w * h * c);
    for (let y = 0; y < h; y++) {
      const row = y * w * c;
      for (let x = 0; x < w; x++) {
        const xb = x * nx;
        if (c === 3) {
          let a0 = 0;
          let a1 = 0;
          let a2 = 0;
          for (let t = 0; t < nx; t++) {
            const o = row + xi[xb + t];
            const kv = kx[t];
            a0 += sd[o] * kv;
            a1 += sd[o + 1] * kv;
            a2 += sd[o + 2] * kv;
          }
          const ob = row + x * 3;
          tmp[ob] = a0;
          tmp[ob + 1] = a1;
          tmp[ob + 2] = a2;
        } else {
          for (let k = 0; k < c; k++) {
            let acc = 0;
            for (let t = 0; t < nx; t++) acc += sd[row + xi[xb + t] + k] * kx[t];
            tmp[row + x * c + k] = acc;
          }
        }
      }
    }
    // Vertical pass: weighted sum of whole rows.
    const out = image(w, h, c, roundToUint8 ? Uint8Array : Float32Array);
    const od = out.data;
    const rowLen = w * c;
    const acc = new Float64Array(rowLen);
    for (let y = 0; y < h; y++) {
      acc.fill(0);
      for (let t = 0; t < ny; t++) {
        const base = reflect101(y + t - ry, h) * rowLen;
        const kv = ky[t];
        for (let i = 0; i < rowLen; i++) acc[i] += tmp[base + i] * kv;
      }
      const ob = y * rowLen;
      if (roundToUint8) for (let i = 0; i < rowLen; i++) od[ob + i] = Math.min(255, Math.max(0, Math.round(acc[i])));
      else for (let i = 0; i < rowLen; i++) od[ob + i] = acc[i];
    }
    return out;
  }

  /** cv2.GaussianBlur(src, (0, 0), sigma). */
  function gaussianBlur(src, sigma) {
    const isUint8 = src.data instanceof Uint8Array;
    const k = gaussianKernel(sigma, isUint8);
    if (isUint8) return gaussianBlurFixed(src, gaussianKernelFixed(k));
    // Float images use the weights rounded to float32, as OpenCV does.
    return convolveSeparable(src, k.map(Math.fround), k.map(Math.fround), false);
  }

  /** cv2.Sobel(src, CV_32F, dx, dy, ksize=3) on one channel. */
  function sobel(src, dx, dy) {
    const smooth = [1, 2, 1];
    const diff = [-1, 0, 1];
    return convolveSeparable(src, dx ? diff : smooth, dy ? diff : smooth, false);
  }

  /** Variance of cv2.Laplacian(gray, CV_64F) (ksize 1). */
  function laplacianVariance(gray) {
    const { w, h } = gray;
    const g = gray.data;
    let s = 0;
    let s2 = 0;
    for (let y = 0; y < h; y++) {
      const yu = reflect101(y - 1, h);
      const yd = reflect101(y + 1, h);
      for (let x = 0; x < w; x++) {
        const xl = reflect101(x - 1, w);
        const xr = reflect101(x + 1, w);
        const v = g[yu * w + x] + g[yd * w + x] + g[y * w + xl] + g[y * w + xr] - 4 * g[y * w + x];
        s += v;
        s2 += v * v;
      }
    }
    const n = w * h;
    return s2 / n - (s / n) ** 2;
  }

  // ------------------------------------------------------------ morphology

  /** cv2.getStructuringElement rows as [start, end) spans, plus anchor. */
  function kernel(shape, kw, kh = kw) {
    const rows = [];
    const r = Math.floor(kh / 2);
    const c = Math.floor(kw / 2);
    if (shape === "rect") {
      for (let i = 0; i < kh; i++) rows.push([0, kw]);
    } else {
      const invR2 = r ? 1 / (r * r) : 0;
      for (let i = 0; i < kh; i++) {
        const dy = i - r;
        let j1 = 0;
        let j2 = 0;
        if (Math.abs(dy) <= r) {
          const dx = Math.round(c * Math.sqrt((r * r - dy * dy) * invR2));
          j1 = Math.max(c - dx, 0);
          j2 = Math.min(c + dx + 1, kw);
        }
        rows.push([j1, j2]);
      }
    }
    return { rows, ax: c, ay: r, kw, kh, shape };
  }

  // Felzenszwalb & Huttenlocher's 1D distance transform of a sampled
  // function: d[q] = min over p of (q - p)^2 + f[p].
  function dt1d(f, n, d, v, z) {
    let k = 0;
    v[0] = 0;
    z[0] = -Infinity;
    z[1] = Infinity;
    for (let q = 1; q < n; q++) {
      let sv = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (sv <= z[k]) {
        k--;
        sv = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      }
      k++;
      v[k] = q;
      z[k] = sv;
      z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < n; q++) {
      while (z[k + 1] < q) k++;
      d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
  }

  /** Dilation by OpenCV's square elliptical kernel of radius r, exactly, in
   * time independent of r. An offset (dx, dy) with |dy| < r is in that
   * kernel exactly when (|dx| - 0.5)^2 + dy^2 <= r^2; the rows |dy| = r hold
   * only dx = 0. With dmin the distance to the nearest set pixel in a row,
   * (dmin - 0.5)^2 - 0.25 + dy^2 <= r^2 - 0.25 is a distance transform. */
  const TILE = 16;
  let scratch = { n: 0 };
  function scratchFor(n, h) {
    if (scratch.n < n || scratch.h < h) {
      const nn = Math.max(n, scratch.n);
      const hh = Math.max(h, scratch.h || 0);
      scratch = {
        n: nn,
        h: hh,
        F: new Float64Array(nn),
        block: new Float64Array(hh * TILE),
        d: new Float64Array(hh),
        v: new Int32Array(hh),
        z: new Float64Array(hh + 1),
      };
    }
    return scratch;
  }

  function ellipseField(src, w, h, setValue) {
    // Per pixel: dmin = distance to the nearest set pixel in its row, and D =
    // min over rows of (dmin - 0.5)^2 - 0.25 + dy^2 (see dilateEllipse).
    const BIG = 1e20;
    const n = w * h;
    const sc = scratchFor(n, h);
    const F = sc.F;
    const dmin = new Int32Array(n);
    for (let y = 0; y < h; y++) {
      const o = y * w;
      let last = -(1 << 29);
      for (let x = 0; x < w; x++) {
        if ((src[o + x] ? 1 : 0) === setValue) last = x;
        dmin[o + x] = x - last;
      }
      last = 1 << 29;
      for (let x = w - 1; x >= 0; x--) {
        if ((src[o + x] ? 1 : 0) === setValue) last = x;
        const dd = last - x;
        if (dd < dmin[o + x]) dmin[o + x] = dd;
        const dm = dmin[o + x];
        F[o + x] = dm < 1 << 28 ? (dm - 0.5) * (dm - 0.5) - 0.25 : BIG;
      }
    }
    // Column transforms, TILE columns at a time so memory is read in rows.
    const D = new Float64Array(n);
    const block = sc.block;
    const d = sc.d;
    const v = sc.v;
    const z = sc.z;
    for (let x0 = 0; x0 < w; x0 += TILE) {
      const tw = Math.min(TILE, w - x0);
      for (let y = 0; y < h; y++) {
        const o = y * w + x0;
        for (let t = 0; t < tw; t++) block[t * h + y] = F[o + t];
      }
      for (let t = 0; t < tw; t++) {
        dt1d(block.subarray(t * h, t * h + h), h, d, v, z);
        for (let y = 0; y < h; y++) block[t * h + y] = d[y];
      }
      for (let y = 0; y < h; y++) {
        const o = y * w + x0;
        for (let t = 0; t < tw; t++) D[o + t] = block[t * h + y];
      }
    }
    return { D, dmin, w, h };
  }

  /** Threshold an ellipseField at radius r: the dilation by OpenCV's square
   * elliptical kernel of that radius. */
  function ellipseHit(field, r) {
    const { D, dmin, w, h } = field;
    const out = new Uint8Array(w * h);
    const limit = r * r - 0.25;
    for (let y = 0; y < h; y++) {
      const o = y * w;
      const up = y - r >= 0 ? (y - r) * w : -1;
      const dn = y + r < h ? (y + r) * w : -1;
      for (let x = 0; x < w; x++) {
        out[o + x] = D[o + x] <= limit || (up >= 0 && dmin[up + x] === 0) || (dn >= 0 && dmin[dn + x] === 0) ? 1 : 0;
      }
    }
    return out;
  }

  /** Dilation by OpenCV's square elliptical kernel of radius r, exactly, in
   * time independent of r. An offset (dx, dy) with |dy| < r is in that
   * kernel exactly when (|dx| - 0.5)^2 + dy^2 <= r^2; the rows |dy| = r hold
   * only dx = 0. With dmin the distance to the nearest set pixel in a row,
   * (dmin - 0.5)^2 - 0.25 + dy^2 <= r^2 - 0.25 is a distance transform. */
  function dilateEllipse(src, w, h, r, setValue) {
    return ellipseHit(ellipseField(src, w, h, setValue), r);
  }

  function morph(mask, k, erode) {
    if (k.shape === "ellipse" && k.kw === k.kh && k.kw > 11) {
      const hit = dilateEllipse(mask.data, mask.w, mask.h, k.ay, erode ? 0 : 1);
      const out = image(mask.w, mask.h, 1, Uint8Array);
      for (let i = 0; i < hit.length; i++) out.data[i] = erode ? 1 - hit[i] : hit[i];
      return out;
    }
    // Kernels here are symmetric about their anchor (odd sizes), so a pixel
    // is hit by kernel row i when the nearest set pixel (dilate) or unset
    // pixel (erode) in the matching image row lies within that row's
    // half-width. Image borders never decide, as in OpenCV.
    const { w, h } = mask;
    const src = mask.data;
    const want = erode ? 0 : 1;
    const BIG = 1 << 29;
    const dist = new Int32Array(w * h);
    for (let y = 0; y < h; y++) {
      const o = y * w;
      let last = -BIG;
      for (let x = 0; x < w; x++) {
        if ((src[o + x] ? 1 : 0) === want) last = x;
        dist[o + x] = x - last;
      }
      last = BIG;
      for (let x = w - 1; x >= 0; x--) {
        if ((src[o + x] ? 1 : 0) === want) last = x;
        const d = last - x;
        if (d < dist[o + x]) dist[o + x] = d;
      }
    }
    const rows = [];
    for (let i = 0; i < k.kh; i++) {
      const [j1, j2] = k.rows[i];
      if (j2 > j1) rows.push(i - k.ay, (j2 - j1 - 1) >> 1);
    }
    const out = image(w, h, 1, Uint8Array);
    const od = out.data;
    const hit = new Uint8Array(w);
    for (let y = 0; y < h; y++) {
      hit.fill(0);
      for (let r = 0; r < rows.length; r += 2) {
        const yy = y + rows[r];
        if (yy < 0 || yy >= h) continue;
        const hw = rows[r + 1];
        const o = yy * w;
        for (let x = 0; x < w; x++) if (dist[o + x] <= hw) hit[x] = 1;
      }
      const ob = y * w;
      for (let x = 0; x < w; x++) od[ob + x] = erode ? 1 - hit[x] : hit[x];
    }
    return out;
  }

  const dilate = (mask, k) => morph(mask, k, false);
  const erode = (mask, k) => morph(mask, k, true);
  const open = (mask, k) => dilate(erode(mask, k), k);
  const close = (mask, k) => erode(dilate(mask, k), k);

  // ------------------------------------------------------------ thresholds, edges

  /** Otsu threshold of a uint8 single-channel image (cv2.THRESH_OTSU). */
  function otsu(gray) {
    const hist = new Float64Array(256);
    for (let i = 0; i < gray.data.length; i++) hist[gray.data[i]]++;
    const n = gray.data.length;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0;
    let wB = 0;
    let best = 0;
    let thr = 0;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = n - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) ** 2;
      if (between > best) {
        best = between;
        thr = t;
      }
    }
    return thr;
  }

  /** cv2.Canny(gray, lo, hi, L2gradient=True), with OpenCV's integer
   * arithmetic: squared magnitudes against floor(threshold^2), angle sectors
   * from tan(22.5°) in 15-bit fixed point, strict tests on diagonals, and
   * zero magnitude outside the image. */
  function canny(gray, loF, hiF) {
    const { w, h } = gray;
    const g = gray.data;
    const lo = Math.floor(Math.min(32767, loF) ** 2);
    const hi = Math.floor(Math.min(32767, hiF) ** 2);
    const mag = new Float64Array(w * h);
    const gxs = new Int32Array(w * h);
    const gys = new Int32Array(w * h);
    for (let y = 0; y < h; y++) {
      const up = Math.max(0, y - 1) * w;
      const mid = y * w;
      const dn = Math.min(h - 1, y + 1) * w;
      for (let x = 0; x < w; x++) {
        const l = x > 0 ? x - 1 : 0;
        const r = x < w - 1 ? x + 1 : w - 1;
        const gx = g[up + r] + 2 * g[mid + r] + g[dn + r] - g[up + l] - 2 * g[mid + l] - g[dn + l];
        const gy = g[dn + l] + 2 * g[dn + x] + g[dn + r] - g[up + l] - 2 * g[up + x] - g[up + r];
        gxs[mid + x] = gx;
        gys[mid + x] = gy;
        mag[mid + x] = gx * gx + gy * gy;
      }
    }
    const M = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : mag[y * w + x]);
    const TG22 = 13573; // tan(22.5°) * 2^15
    const state = new Uint8Array(w * h); // 0 none, 1 weak, 2 strong
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const m = mag[i];
        if (m <= lo) continue;
        const xs = gxs[i];
        const ys = gys[i];
        const ax = Math.abs(xs);
        const ay = Math.abs(ys) * 32768;
        const tg22x = ax * TG22;
        let keep;
        if (ay < tg22x) keep = m > M(x - 1, y) && m >= M(x + 1, y);
        else if (ay > tg22x + ax * 65536) keep = m > M(x, y - 1) && m >= M(x, y + 1);
        else {
          const s = (xs ^ ys) < 0 ? -1 : 1;
          keep = m > M(x - s, y - 1) && m > M(x + s, y + 1);
        }
        if (keep) state[i] = m > hi ? 2 : 1;
      }
    }
    const out = image(w, h, 1, Uint8Array);
    const stack = [];
    for (let i = 0; i < w * h; i++) {
      if (state[i] === 2 && !out.data[i]) {
        out.data[i] = 1;
        stack.push(i);
        while (stack.length) {
          const p = stack.pop();
          const px = p % w;
          const py = (p - px) / w;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const qx = px + dx;
              const qy = py + dy;
              if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
              const q = qy * w + qx;
              if (state[q] && !out.data[q]) {
                out.data[q] = 1;
                stack.push(q);
              }
            }
          }
        }
      }
    }
    return out;
  }

  // ------------------------------------------------------------ contours

  const DIRS = [
    [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1],
  ]; // counter-clockwise in image coordinates (y down), starting east

  /** cv2.findContours(mask, RETR_EXTERNAL | RETR_TREE, CHAIN_APPROX_NONE):
   * Suzuki & Abe border following. Returns an array of Int32Array point
   * lists [x0, y0, x1, y1, ...]. "external" keeps only outermost borders. */
  function findContours(mask, mode = "tree") {
    const W = mask.w + 2;
    const H = mask.h + 2;
    const f = new Int32Array(W * H); // one-pixel zero frame around the mask
    for (let y = 0; y < mask.h; y++) {
      for (let x = 0; x < mask.w; x++) f[(y + 1) * W + x + 1] = mask.data[y * mask.w + x] ? 1 : 0;
    }
    const contours = [];
    // Border 1 is the picture frame, which counts as a hole border.
    const isHole = [false, true];
    const parent = [0, 0];
    let nbd = 1;
    for (let y = 1; y < H - 1; y++) {
      let lnbd = 1;
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        const v = f[i];
        let startDir = -1;
        let hole = false;
        if (v === 1 && f[i - 1] === 0) {
          startDir = 4; // outer border: the zero pixel is to the west
        } else if (v >= 1 && f[i + 1] === 0) {
          startDir = 0; // hole border: the zero pixel is to the east
          hole = true;
          if (v > 1) lnbd = v;
        }
        if (startDir >= 0) {
          nbd++;
          isHole[nbd] = hole;
          const lb = Math.abs(lnbd);
          parent[nbd] = hole === isHole[lb] ? parent[lb] : lb;
          const pts = traceBorder(f, W, x, y, startDir, nbd);
          if (mode === "tree" || (!hole && parent[nbd] === 1)) {
            const out = new Int32Array(pts.length);
            for (let k = 0; k < pts.length; k++) out[k] = pts[k] - 1;
            contours.push(out);
          }
        }
        const after = f[i];
        if (after !== 0 && after !== 1) lnbd = Math.abs(after);
      }
    }
    return contours;
  }

  function traceBorder(f, W, x0, y0, startDir, nbd) {
    const pts = [];
    const at = (x, y, d) => f[(y + DIRS[d][1]) * W + x + DIRS[d][0]];
    // 3.1: look clockwise around the start, beginning after the zero pixel.
    let s = startDir;
    let first = -1;
    for (let k = 0; k < 8; k++) {
      s = (s + 7) % 8;
      if (at(x0, y0, s) !== 0) {
        first = s;
        break;
      }
    }
    if (first < 0) {
      f[y0 * W + x0] = -nbd; // a single isolated pixel
      pts.push(x0, y0);
      return pts;
    }
    const x1 = x0 + DIRS[first][0];
    const y1 = y0 + DIRS[first][1];
    let x3 = x0;
    let y3 = y0;
    s = first; // direction from the current pixel back to the previous one
    for (;;) {
      // 3.3: look counter-clockwise, starting just after the previous pixel.
      let next = -1;
      let eastZero = false;
      for (let k = 1; k <= 8; k++) {
        const d = (s + k) % 8;
        if (at(x3, y3, d) !== 0) {
          next = d;
          break;
        }
        if (d === 0) eastZero = true;
      }
      // 3.4: mark the pixel (negative when its east side is background).
      const i3 = y3 * W + x3;
      if (eastZero) f[i3] = -nbd;
      else if (f[i3] === 1) f[i3] = nbd;
      pts.push(x3, y3);
      const x4 = x3 + DIRS[next][0];
      const y4 = y3 + DIRS[next][1];
      // 3.5: done when the next step would leave the last pixel for the start.
      if (x4 === x0 && y4 === y0 && x3 === x1 && y3 === y1) break;
      s = (next + 4) % 8;
      x3 = x4;
      y3 = y4;
    }
    return pts;
  }

  // ------------------------------------------------------------ geometry

  function contourArea(pts, signed = false) {
    let a = 0;
    const n = pts.length / 2;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      a += pts[2 * i] * pts[2 * j + 1] - pts[2 * j] * pts[2 * i + 1];
    }
    a /= 2;
    return signed ? a : Math.abs(a);
  }

  function arcLength(pts, closed = true) {
    const n = pts.length / 2;
    let s = 0;
    for (let i = 0; i < n - (closed ? 0 : 1); i++) {
      const j = (i + 1) % n;
      s += Math.hypot(pts[2 * j] - pts[2 * i], pts[2 * j + 1] - pts[2 * i + 1]);
    }
    return s;
  }

  const sign = (v) => (v > 0) - (v < 0);

  // One chain of Sklansky's scan over x-sorted points (convhull.cpp).
  function sklansky(P, start, end, stack, so, nsign, sign2) {
    const incr = end > start ? 1 : -1;
    let pprev = start;
    let pcur = pprev + incr;
    let pnext = pcur + incr;
    let size = 3;
    const x = (k) => P[2 * k];
    const y = (k) => P[2 * k + 1];
    if (start === end || (x(start) === x(end) && y(start) === y(end))) {
      stack[so] = start;
      return 1;
    }
    stack[so] = pprev;
    stack[so + 1] = pcur;
    stack[so + 2] = pnext;
    end += incr;
    while (pnext !== end) {
      const cury = y(pcur);
      const by = y(pnext) - cury;
      if (sign(by) !== nsign) {
        const ax = x(pcur) - x(pprev);
        const ay = cury - y(pprev);
        const bx = x(pnext) - x(pcur);
        const convexity = ay * bx - ax * by;
        if (sign(convexity) === sign2 && (ax !== 0 || ay !== 0)) {
          pprev = pcur;
          pcur = pnext;
          pnext += incr;
          stack[so + size] = pnext;
          size++;
        } else if (pprev === start) {
          pcur = pnext;
          stack[so + 1] = pcur;
          pnext += incr;
          stack[so + 2] = pnext;
        } else {
          stack[so + size - 2] = pnext;
          pcur = pprev;
          pprev = stack[so + size - 4];
          size--;
        }
      } else {
        pnext += incr;
        stack[so + size - 1] = pnext;
      }
    }
    return size - 1;
  }

  /** cv2.convexHull(points) for integer points (default counter-clockwise
   * in OpenCV's terms), ported from convhull.cpp including the cyclic shift
   * that makes the indices run in contour order where possible. */
  function convexHull(pts) {
    const total = pts.length / 2;
    if (!total) return new Float64Array(0);
    const order = Array.from({ length: total }, (_, i) => i);
    order.sort((a, b) => pts[2 * a] - pts[2 * b] || pts[2 * a + 1] - pts[2 * b + 1] || a - b);
    const P = new Float64Array(total * 2);
    order.forEach((k, i) => {
      P[2 * i] = pts[2 * k];
      P[2 * i + 1] = pts[2 * k + 1];
    });
    let miny = 0;
    let maxy = 0;
    for (let i = 1; i < total; i++) {
      const yv = P[2 * i + 1];
      if (P[2 * miny + 1] > yv) miny = i;
      if (P[2 * maxy + 1] < yv) maxy = i;
    }
    const stack = new Int32Array(total + 2);
    let hull = [];
    if (P[0] === P[2 * (total - 1)] && P[1] === P[2 * (total - 1) + 1]) {
      hull.push(0);
    } else {
      // Upper half (clockwise = false swaps the two chains).
      const tlCount0 = sklansky(P, 0, maxy, stack, 0, -1, 1);
      const trCount0 = sklansky(P, total - 1, maxy, stack, tlCount0, -1, -1);
      const tl = { o: tlCount0, n: trCount0 };
      const tr = { o: 0, n: tlCount0 };
      for (let i = 0; i < tl.n - 1; i++) hull.push(stack[tl.o + i]);
      for (let i = tr.n - 1; i > 0; i--) hull.push(stack[tr.o + i]);
      const stopIdx = tr.n > 2 ? stack[tr.o + 1] : tl.n > 2 ? stack[tl.o + tl.n - 2] : -1;
      // Lower half.
      let blCount = sklansky(P, 0, miny, stack, 0, 1, -1);
      const brOff = blCount;
      let brCount = sklansky(P, total - 1, miny, stack, brOff, 1, 1);
      if (stopIdx >= 0) {
        const checkIdx = blCount > 2 ? stack[1] : blCount + brCount > 2 ? stack[brOff + 2 - blCount] : -1;
        if (
          checkIdx === stopIdx ||
          (checkIdx >= 0 && P[2 * checkIdx] === P[2 * stopIdx] && P[2 * checkIdx + 1] === P[2 * stopIdx + 1])
        ) {
          blCount = Math.min(blCount, 2);
          brCount = Math.min(brCount, 2);
        }
      }
      for (let i = 0; i < blCount - 1; i++) hull.push(stack[i]);
      for (let i = brCount - 1; i > 0; i--) hull.push(stack[brOff + i]);
      hull = hull.map((i) => order[i]);
      // Make the indices ascend or descend by a cyclic shift, if they can.
      const nout = hull.length;
      if (nout >= 3) {
        let minIdx = 0;
        let maxIdx = 0;
        let lt = 0;
        for (let i = 1; i < nout; i++) {
          const idx = hull[i];
          lt += hull[i - 1] < idx ? 1 : 0;
          if (lt > 1 && lt <= i - 2) break;
          if (idx < hull[minIdx]) minIdx = i;
          if (idx > hull[maxIdx]) maxIdx = i;
        }
        const mmdist = Math.abs(maxIdx - minIdx);
        if ((mmdist === 1 || mmdist === nout - 1) && (lt <= 1 || lt >= nout - 2)) {
          const ascending = (maxIdx + 1) % nout === minIdx;
          const i0 = ascending ? minIdx : maxIdx;
          if (i0 > 0) {
            const shifted = [];
            let j = i0;
            let i = 0;
            for (; i < nout; i++) {
              const curr = (shifted[i] = hull[j]);
              const nextJ = j + 1 < nout ? j + 1 : 0;
              if (i < nout - 1 && ascending !== curr < hull[nextJ]) break;
              j = nextJ;
            }
            if (i === nout) hull = shifted;
          }
        }
      }
    }
    const out = new Float64Array(hull.length * 2);
    hull.forEach((i, k) => {
      out[2 * k] = pts[2 * i];
      out[2 * k + 1] = pts[2 * i + 1];
    });
    return out;
  }

  /** cv2.approxPolyDP(curve, eps, closed=True), ported from OpenCV's
   * approxPolyDP_ including its final clean-up pass. */
  function approxPolyDP(pts, eps0) {
    const count = pts.length / 2;
    if (!count) return new Float64Array(0);
    const X = (i) => pts[2 * i];
    const Y = (i) => pts[2 * i + 1];
    const eps = eps0 * eps0;
    const dst = [];
    const stack = [];
    let rightStart = 0;
    let pos = 0;
    let leEps = false;
    let sx = 0;
    let sy = 0;
    for (let it = 0; it < 3; it++) {
      let maxDist = 0;
      pos = (pos + rightStart) % count;
      sx = X(pos);
      sy = Y(pos);
      pos = (pos + 1) % count;
      for (let j = 1; j < count; j++) {
        const dx = X(pos) - sx;
        const dy = Y(pos) - sy;
        pos = (pos + 1) % count;
        const dist = dx * dx + dy * dy;
        if (dist > maxDist) {
          maxDist = dist;
          rightStart = j;
        }
      }
      leEps = maxDist <= eps;
    }
    if (!leEps) {
      const sliceStart = pos % count;
      const rightEnd = sliceStart;
      const sliceEnd = (rightStart + sliceStart) % count;
      stack.push([sliceEnd, rightEnd]);
      stack.push([sliceStart, sliceEnd]);
    } else {
      dst.push([sx, sy]);
    }
    while (stack.length) {
      const [a, b] = stack.pop();
      const ex = X(b);
      const ey = Y(b);
      let p = a;
      const ax = X(p);
      const ay = Y(p);
      p = (p + 1) % count;
      let split = a;
      if (a !== b) {
        const dx = ex - ax;
        const dy = ey - ay;
        let maxDist = 0;
        while (p !== b) {
          const px = X(p);
          const py = Y(p);
          p = (p + 1) % count;
          const dist = Math.abs((py - ay) * dx - (px - ax) * dy);
          if (dist > maxDist) {
            maxDist = dist;
            split = (p + count - 1) % count;
          }
        }
        leEps = maxDist * maxDist <= eps * (dx * dx + dy * dy);
      } else {
        leEps = true;
      }
      if (leEps) dst.push([ax, ay]);
      else {
        stack.push([split, b]);
        stack.push([a, split]);
      }
    }
    // Clean-up: drop points lying on [almost] straight runs.
    let n = dst.length;
    const out = dst.slice();
    let newCount = n;
    let rp = n - 1;
    const readOut = () => {
      const v = out[rp];
      rp = (rp + 1) % n;
      return v;
    };
    let start = readOut();
    let wpos = rp;
    let pt = readOut();
    for (let i = 0; i < n && newCount > 2; i++) {
      const end = readOut();
      const dx = end[0] - start[0];
      const dy = end[1] - start[1];
      const dist = Math.abs((pt[0] - start[0]) * dy - (pt[1] - start[1]) * dx);
      const inner = (pt[0] - start[0]) * (end[0] - pt[0]) + (pt[1] - start[1]) * (end[1] - pt[1]);
      if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx !== 0 && dy !== 0 && inner >= 0) {
        newCount--;
        out[wpos] = start = end;
        wpos = (wpos + 1) % n;
        pt = readOut();
        i++;
        continue;
      }
      out[wpos] = start = pt;
      wpos = (wpos + 1) % n;
      pt = end;
    }
    const res = new Float64Array(newCount * 2);
    for (let i = 0; i < newCount; i++) {
      res[2 * i] = out[i][0];
      res[2 * i + 1] = out[i][1];
    }
    return res;
  }

  /** cv2.boxPoints(cv2.minAreaRect(hull)) via rotating calipers. */
  function minAreaRectPoints(hull) {
    const n = hull.length / 2;
    let best = null;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      let ux = hull[2 * j] - hull[2 * i];
      let uy = hull[2 * j + 1] - hull[2 * i + 1];
      const len = Math.hypot(ux, uy);
      if (len < 1e-9) continue;
      ux /= len;
      uy /= len;
      let minA = Infinity;
      let maxA = -Infinity;
      let minB = Infinity;
      let maxB = -Infinity;
      for (let k = 0; k < n; k++) {
        const a = hull[2 * k] * ux + hull[2 * k + 1] * uy;
        const b = -hull[2 * k] * uy + hull[2 * k + 1] * ux;
        minA = Math.min(minA, a);
        maxA = Math.max(maxA, a);
        minB = Math.min(minB, b);
        maxB = Math.max(maxB, b);
      }
      const area = (maxA - minA) * (maxB - minB);
      if (!best || area < best.area) best = { area, ux, uy, minA, maxA, minB, maxB };
    }
    if (!best) return new Float64Array(8);
    const { ux, uy, minA, maxA, minB, maxB } = best;
    const pt = (a, b) => [a * ux - b * uy, a * uy + b * ux];
    return Float64Array.from([...pt(minA, minB), ...pt(maxA, minB), ...pt(maxA, maxB), ...pt(minA, maxB)]);
  }

  // cv::RNG (multiply-with-carry), seeded as fitLine2D seeds it.
  function cvRng(seed) {
    let state = BigInt.asUintN(64, seed);
    return {
      next() {
        state = BigInt.asUintN(64, (state & 0xffffffffn) * 4164903690n + (state >> 32n));
        return Number(state & 0xffffffffn);
      },
      uniform(a, b) {
        return a === b ? a : (this.next() % (b - a)) + a;
      },
    };
  }

  const f32 = Math.fround;

  function fitLineWeighted(xs, ys, wts) {
    let x = 0;
    let y = 0;
    let x2 = 0;
    let y2 = 0;
    let xy = 0;
    let w = 0;
    for (let i = 0; i < xs.length; i++) {
      const wi = wts[i];
      x += wi * xs[i];
      y += wi * ys[i];
      x2 += wi * xs[i] * xs[i];
      y2 += wi * ys[i] * ys[i];
      xy += wi * xs[i] * ys[i];
      w += wi;
    }
    x /= w;
    y /= w;
    x2 /= w;
    y2 /= w;
    xy /= w;
    const t = f32(Math.atan2(2 * (xy - x * y), x2 - x * x - (y2 - y * y)) / 2);
    return [f32(Math.cos(t)), f32(Math.sin(t)), f32(x), f32(y)];
  }

  /** cv2.fitLine(points, DIST_HUBER, 0, 0.01, 0.01) -> [vx, vy, x0, y0],
   * following OpenCV's fitLine2D: 20 restarts from 10 random points, each
   * refined by Huber reweighting, keeping the smallest total distance. */
  function fitLineHuber(xsIn, ysIn) {
    const count = xsIn.length;
    const xs = Float32Array.from(xsIn);
    const ys = Float32Array.from(ysIn);
    const EPS = count * 1.1920929e-7;
    const rdelta = 0.01;
    const adelta = 0.01;
    const C = 1.345;
    const rng = cvRng(0xffffffffffffffffn);
    const w = new Float32Array(count);
    const r = new Float32Array(count);
    let best = [0, 0, 0, 0];
    let minErr = Infinity;
    let err = 0;
    const calcDist = (line) => {
      const [vx, vy, px, py] = line;
      let sum = 0;
      for (let j = 0; j < count; j++) {
        r[j] = f32(Math.abs(vy * (xs[j] - px) - vx * (ys[j] - py)));
        sum += r[j];
      }
      return f32(sum);
    };
    for (let k = 0; k < 20; k++) {
      w.fill(0);
      for (let i = 0; i < Math.min(count, 10); ) {
        const j = rng.uniform(0, count);
        if (w[j] < 1.1920929e-7) {
          w[j] = 1;
          i++;
        }
      }
      let line = fitLineWeighted(xs, ys, w);
      let prev = null;
      let stop = false;
      for (let i = 0; i < 30; i++) {
        if (prev) {
          const t = Math.min(1, Math.max(-1, line[0] * prev[0] + line[1] * prev[1]));
          if (Math.abs(Math.acos(t)) < adelta) {
            const d = Math.max(Math.abs(line[2] - prev[2]), Math.abs(line[3] - prev[3]));
            if (d < rdelta) break;
          }
        }
        err = calcDist(line);
        if (err < minErr) {
          minErr = err;
          best = line.slice();
          if (err < EPS) {
            stop = true;
            break;
          }
        }
        let sumW = 0;
        for (let j = 0; j < count; j++) {
          w[j] = r[j] < C ? 1 : C / r[j];
          sumW += w[j];
        }
        if (Math.abs(sumW) > 1.1920929e-7) for (let j = 0; j < count; j++) w[j] = w[j] / sumW;
        else w.fill(1);
        prev = line;
        line = fitLineWeighted(xs, ys, w);
      }
      if (stop) break;
      if (err < minErr) {
        minErr = err;
        best = line.slice();
        if (err < EPS) break;
      }
    }
    return best;
  }

  function isContourConvex(pts) {
    const n = pts.length / 2;
    if (n < 3) return false;
    let sign = 0;
    for (let i = 0; i < n; i++) {
      const a = i;
      const b = (i + 1) % n;
      const c = (i + 2) % n;
      const cr =
        (pts[2 * b] - pts[2 * a]) * (pts[2 * c + 1] - pts[2 * b + 1]) - (pts[2 * b + 1] - pts[2 * a + 1]) * (pts[2 * c] - pts[2 * b]);
      if (Math.abs(cr) < 1e-12) continue;
      const s = cr > 0 ? 1 : -1;
      if (!sign) sign = s;
      else if (s !== sign) return false;
    }
    return true;
  }

  /** Area of the intersection of two convex polygons (cv2.intersectConvexConvex). */
  function intersectConvexArea(p, q) {
    const ccw = (poly) => (contourArea(poly, true) < 0 ? reversePoly(poly) : poly);
    let subject = Array.from(ccw(p));
    const clip = ccw(q);
    const m = clip.length / 2;
    for (let i = 0; i < m && subject.length; i++) {
      const ax = clip[2 * i];
      const ay = clip[2 * i + 1];
      const bx = clip[2 * ((i + 1) % m)];
      const by = clip[2 * ((i + 1) % m) + 1];
      const inside = (x, y) => (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0;
      const out = [];
      const k = subject.length / 2;
      for (let j = 0; j < k; j++) {
        const cx = subject[2 * j];
        const cy = subject[2 * j + 1];
        const px = subject[2 * ((j + k - 1) % k)];
        const py = subject[2 * ((j + k - 1) % k) + 1];
        const cin = inside(cx, cy);
        const pin = inside(px, py);
        if (cin !== pin) {
          const dx1 = cx - px;
          const dy1 = cy - py;
          const den = (bx - ax) * dy1 - (by - ay) * dx1;
          if (Math.abs(den) > 1e-12) {
            const t = ((bx - ax) * (ay - py) - (by - ay) * (ax - px)) / den;
            out.push(px + t * dx1, py + t * dy1);
          }
        }
        if (cin) out.push(cx, cy);
      }
      subject = out;
    }
    return subject.length >= 6 ? contourArea(subject) : 0;
  }

  function reversePoly(poly) {
    const n = poly.length / 2;
    const out = new Float64Array(poly.length);
    for (let i = 0; i < n; i++) {
      out[2 * i] = poly[2 * (n - 1 - i)];
      out[2 * i + 1] = poly[2 * (n - 1 - i) + 1];
    }
    return out;
  }

  /** Fill a polygon (convex or not, even-odd) into a mask: pixel centers inside. */
  function fillPoly(mask, pts, value = 1, offX = 0, offY = 0) {
    const n = pts.length / 2;
    if (n < 3) return;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (let i = 0; i < n; i++) {
      ymin = Math.min(ymin, pts[2 * i + 1] - offY);
      ymax = Math.max(ymax, pts[2 * i + 1] - offY);
    }
    const y0 = Math.max(0, Math.ceil(ymin - 0.5));
    const y1 = Math.min(mask.h - 1, Math.floor(ymax + 0.5));
    const xs = [];
    for (let y = y0; y <= y1; y++) {
      xs.length = 0;
      const sy = y;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const ax = pts[2 * i] - offX;
        const ay = pts[2 * i + 1] - offY;
        const bx = pts[2 * j] - offX;
        const by = pts[2 * j + 1] - offY;
        if ((ay <= sy && by > sy) || (by <= sy && ay > sy)) xs.push(ax + ((sy - ay) / (by - ay)) * (bx - ax));
      }
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const xa = Math.max(0, Math.ceil(xs[k] - 0.5));
        const xb = Math.min(mask.w - 1, Math.floor(xs[k + 1] + 0.5));
        for (let x = xa; x <= xb; x++) mask.data[y * mask.w + x] = value;
      }
    }
  }

  /** numpy.rint: round half to even. */
  function rint(x) {
    const f = Math.floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  function drawLine(mask, x0, y0, x1, y1, value = 1) {
    let dx = Math.abs(x1 - x0);
    let dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    for (;;) {
      if (x0 >= 0 && y0 >= 0 && x0 < mask.w && y0 < mask.h) mask.data[y0 * mask.w + x0] = value;
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x0 += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y0 += sy;
      }
    }
  }

  /** cv2.fillConvexPoly / drawContours(FILLED) with integer vertices: the
   * interior plus the outline itself. */
  function fillConvexPoly(mask, pts, value = 1) {
    fillPoly(mask, pts, value);
    const n = pts.length / 2;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      drawLine(mask, Math.round(pts[2 * i]), Math.round(pts[2 * i + 1]), Math.round(pts[2 * j]), Math.round(pts[2 * j + 1]), value);
    }
  }

  /** cv2.cvtColor(rgb, COLOR_RGB2GRAY) for uint8 (OpenCV's fixed-point weights). */
  function rgbToGray(src) {
    const out = image(src.w, src.h, 1, Uint8Array);
    for (let i = 0, j = 0; i < src.w * src.h; i++, j += src.c) {
      out.data[i] = (src.data[j] * 4899 + src.data[j + 1] * 9617 + src.data[j + 2] * 1868 + 8192) >> 14;
    }
    return out;
  }

  /** One Lab color (real units) to sRGB uint8 [r, g, b]. */
  function labToRgb(L, a, b) {
    const fy = (L + 16) / 116;
    const fx = fy + a / 500;
    const fz = fy - b / 200;
    const inv = (t) => (t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787);
    const X = inv(fx) * 0.950456;
    const Y = inv(fy);
    const Z = inv(fz) * 1.088754;
    const lin = [
      3.240479 * X - 1.53715 * Y - 0.498535 * Z,
      -0.969256 * X + 1.875991 * Y + 0.041556 * Z,
      0.055648 * X - 0.204043 * Y + 1.057311 * Z,
    ];
    return lin.map((v) => {
      const c = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(Math.max(v, 0), 1 / 2.4) - 0.055;
      return Math.min(255, Math.max(0, Math.round(c * 255)));
    });
  }

  /** cv2.connectedComponents(mask, connectivity=8) -> {n, labels}. */
  function connectedComponents(mask) {
    const { w, h } = mask;
    const labels = new Int32Array(w * h);
    const parentArr = [0];
    const find = (x) => {
      while (parentArr[x] !== x) {
        parentArr[x] = parentArr[parentArr[x]];
        x = parentArr[x];
      }
      return x;
    };
    const unite = (a, b) => {
      const ra = find(a);
      const rb = find(b);
      if (ra !== rb) parentArr[Math.max(ra, rb)] = Math.min(ra, rb);
    };
    let next = 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask.data[i]) continue;
        const nbrs = [];
        if (x > 0 && labels[i - 1]) nbrs.push(labels[i - 1]);
        if (y > 0) {
          if (x > 0 && labels[i - w - 1]) nbrs.push(labels[i - w - 1]);
          if (labels[i - w]) nbrs.push(labels[i - w]);
          if (x < w - 1 && labels[i - w + 1]) nbrs.push(labels[i - w + 1]);
        }
        if (!nbrs.length) {
          parentArr.push(next);
          labels[i] = next++;
        } else {
          let m = nbrs[0];
          for (const v of nbrs) m = Math.min(m, v);
          labels[i] = m;
          for (const v of nbrs) unite(m, v);
        }
      }
    }
    const remap = new Int32Array(next);
    let n = 1;
    for (let l = 1; l < next; l++) {
      const r = find(l);
      if (!remap[r]) remap[r] = n++;
      remap[l] = remap[r];
    }
    for (let i = 0; i < w * h; i++) if (labels[i]) labels[i] = remap[labels[i]];
    return { n, labels };
  }

  // ------------------------------------------------------------ perspective

  // OpenCV's LU solver (matrix_decomp.cpp, LUImpl): partial pivoting,
  // elimination below the pivot only, then back substitution.
  function solve(A, b) {
    const m = b.length;
    A = A.map((row) => row.slice());
    b = b.slice();
    for (let i = 0; i < m; i++) {
      let k = i;
      for (let j = i + 1; j < m; j++) if (Math.abs(A[j][i]) > Math.abs(A[k][i])) k = j;
      if (Math.abs(A[k][i]) < 2.220446049250313e-15) throw new Error("singular matrix");
      if (k !== i) {
        for (let j = i; j < m; j++) [A[i][j], A[k][j]] = [A[k][j], A[i][j]];
        [b[i], b[k]] = [b[k], b[i]];
      }
      const d = -1 / A[i][i];
      for (let j = i + 1; j < m; j++) {
        const alpha = A[j][i] * d;
        for (let q = i + 1; q < m; q++) A[j][q] += alpha * A[i][q];
        b[j] += alpha * b[i];
      }
    }
    for (let i = m - 1; i >= 0; i--) {
      let s = b[i];
      for (let q = i + 1; q < m; q++) s -= A[i][q] * b[q];
      b[i] = s / A[i][i];
    }
    return b;
  }

  /** cv2.getPerspectiveTransform(src4, dst4) as OpenCV computes it: the
   * points are float32 (Point2f) and so are the products in the system.
   * Returns a 3x3 row-major array of 9. */
  function getPerspectiveTransform(src, dst) {
    const f = Math.fround;
    const A = [];
    const b = [];
    for (let i = 0; i < 4; i++) {
      const x = f(src[i][0]);
      const y = f(src[i][1]);
      const u = f(dst[i][0]);
      A.push([x, y, 1, 0, 0, 0, f(-x * u), f(-y * u)]);
      b.push(u);
    }
    for (let i = 0; i < 4; i++) {
      const x = f(src[i][0]);
      const y = f(src[i][1]);
      const v = f(dst[i][1]);
      A.push([0, 0, 0, x, y, 1, f(-x * v), f(-y * v)]);
      b.push(v);
    }
    const s = solve(A, b);
    return [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 1];
  }

  /** cv2.invert of a 3x3 double matrix (closed form, as OpenCV does for
   * matrices up to 3x3). */
  function invert3(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (det === 0) throw new Error("singular matrix");
    det = 1 / det;
    return [
      (e * i - f * h) * det, (c * h - b * i) * det, (b * f - c * e) * det,
      (f * g - d * i) * det, (a * i - c * g) * det, (c * d - a * f) * det,
      (d * h - e * g) * det, (b * g - a * h) * det, (a * e - b * d) * det,
    ];
  }

  function perspectiveTransform(points, m) {
    return points.map(([x, y]) => {
      const w = m[6] * x + m[7] * y + m[8];
      return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
    });
  }

  // Interpolation weights at 1/32-pixel steps, built as OpenCV builds them
  // (imgwarp.cpp initInterTab2D): float weights, and 15-bit fixed-point
  // ones for 8-bit images whose sum is corrected to exactly 1.
  const INTER_BITS = 5;
  const TAB = 1 << INTER_BITS;
  const COEF_SCALE = 1 << 15;
  const interTabs = {};

  function lanczos1d(x, out, o) {
    const f = Math.fround;
    if (x < 1.1920928955078125e-7) {
      for (let i = 0; i < 8; i++) out[o + i] = 0;
      out[o + 3] = 1;
      return;
    }
    const s45 = 0.7071067811865476;
    const cs = [[1, 0], [-s45, -s45], [0, 1], [s45, -s45], [-1, 0], [s45, s45], [0, -1], [-s45, s45]];
    const y0 = -(x + 3) * Math.PI * 0.25;
    const s0 = Math.sin(y0);
    const c0 = Math.cos(y0);
    let sum = 0;
    for (let i = 0; i < 8; i++) {
      const y = -(x + 3 - i) * Math.PI * 0.25;
      out[o + i] = f((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
      sum = f(sum + out[o + i]);
    }
    sum = f(1 / sum);
    for (let i = 0; i < 8; i++) out[o + i] = f(out[o + i] * sum);
  }

  function interTab(method) {
    if (interTabs[method]) return interTabs[method];
    const f = Math.fround;
    const k = method === "lanczos4" ? 8 : 2;
    const t1 = new Float32Array(TAB * k);
    for (let i = 0; i < TAB; i++) {
      const x = f(i * f(1 / TAB));
      if (k === 2) {
        t1[2 * i] = f(1 - x);
        t1[2 * i + 1] = x;
      } else lanczos1d(x, t1, 8 * i);
    }
    const kk = k * k;
    const ftab = new Float32Array(TAB * TAB * kk);
    const itab = new Int32Array(TAB * TAB * kk);
    for (let i = 0; i < TAB; i++) {
      for (let j = 0; j < TAB; j++) {
        const o = (i * TAB + j) * kk;
        let isum = 0;
        for (let k1 = 0; k1 < k; k1++) {
          const vy = t1[i * k + k1];
          for (let k2 = 0; k2 < k; k2++) {
            const v = f(vy * t1[j * k + k2]);
            ftab[o + k1 * k + k2] = v;
            const iv = Math.max(-32768, Math.min(32767, rint(v * COEF_SCALE)));
            itab[o + k1 * k + k2] = iv;
            isum += iv;
          }
        }
        if (isum !== COEF_SCALE) {
          const diff = isum - COEF_SCALE;
          const h = k >> 1;
          let Mk1 = h, Mk2 = h, mk1 = h, mk2 = h;
          for (let k1 = h; k1 < h + 2; k1++) {
            for (let k2 = h; k2 < h + 2; k2++) {
              if (itab[o + k1 * k + k2] < itab[o + mk1 * k + mk2]) {
                mk1 = k1;
                mk2 = k2;
              } else if (itab[o + k1 * k + k2] > itab[o + Mk1 * k + Mk2]) {
                Mk1 = k1;
                Mk2 = k2;
              }
            }
          }
          if (diff < 0) itab[o + Mk1 * k + Mk2] -= diff;
          else itab[o + mk1 * k + mk2] -= diff;
        }
      }
    }
    return (interTabs[method] = { k, ftab, itab });
  }

  /** cv2.warpPerspective(src, M, (dw, dh), flags, BORDER_CONSTANT, border),
   * following OpenCV step by step: source positions rounded to 1/32 pixel
   * per 64x16 block, then table-driven bilinear (float32 arithmetic for
   * float images, 15-bit fixed point for 8-bit ones) or Lanczos4.
   * interp: "linear" or "lanczos4". */
  function warpPerspective(src, m, dw, dh, interp = "linear", border = null) {
    const { w, h, c } = src;
    const M = invert3(m);
    const isInt = src.data instanceof Uint8Array || src.data instanceof Uint8ClampedArray;
    const out = image(dw, dh, c, isInt ? Uint8Array : Float32Array);
    const od = out.data;
    const sd = src.data;
    const bv = Array.from({ length: c }, (_, k) => {
      const v = border ? border[k] || 0 : 0;
      return isInt ? Math.min(255, Math.max(0, rint(v))) : Math.fround(v);
    });
    const { k: ks, ftab, itab } = interTab(interp === "lanczos4" ? "lanczos4" : "linear");
    const f = Math.fround;
    const sat = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
    const fix = (v) => sat((v + (1 << 14)) >> 15);
    const bw0 = Math.min(64, dw);
    const bh0 = Math.min(Math.floor(1024 / bw0), dh);
    const BIG = 2147483647;
    for (let y = 0; y < dh; y++) {
      for (let xb = 0; xb < dw; xb += bw0) {
        const X0 = M[0] * xb + M[1] * y + M[2];
        const Y0 = M[3] * xb + M[4] * y + M[5];
        const W0 = M[6] * xb + M[7] * y + M[8];
        const bw = Math.min(bw0, dw - xb);
        for (let x1 = 0; x1 < bw; x1++) {
          let W = W0 + M[6] * x1;
          W = W ? TAB / W : 0;
          const fX = Math.max(-BIG - 1, Math.min(BIG, (X0 + M[0] * x1) * W));
          const fY = Math.max(-BIG - 1, Math.min(BIG, (Y0 + M[3] * x1) * W));
          const X = rint(fX) | 0;
          const Y = rint(fY) | 0;
          const sx0 = Math.max(-32768, Math.min(32767, X >> INTER_BITS));
          const sy0 = Math.max(-32768, Math.min(32767, Y >> INTER_BITS));
          const alpha = (Y & (TAB - 1)) * TAB + (X & (TAB - 1));
          const o = (y * dw + xb + x1) * c;
          if (ks === 2) {
            const wo = alpha * 4;
            if (sx0 >= w || sx0 + 1 < 0 || sy0 >= h || sy0 + 1 < 0) {
              for (let q = 0; q < c; q++) od[o + q] = bv[q];
              continue;
            }
            const in0 = sx0 >= 0 && sy0 >= 0;
            const in1 = sx0 + 1 < w && sy0 >= 0;
            const in2 = sx0 >= 0 && sy0 + 1 < h;
            const in3 = sx0 + 1 < w && sy0 + 1 < h;
            const b0 = (sy0 * w + sx0) * c;
            for (let q = 0; q < c; q++) {
              const v0 = in0 ? sd[b0 + q] : bv[q];
              const v1 = in1 ? sd[b0 + c + q] : bv[q];
              const v2 = in2 ? sd[b0 + w * c + q] : bv[q];
              const v3 = in3 ? sd[b0 + w * c + c + q] : bv[q];
              if (isInt) {
                od[o + q] = fix(v0 * itab[wo] + v1 * itab[wo + 1] + v2 * itab[wo + 2] + v3 * itab[wo + 3]);
              } else {
                od[o + q] = f(f(f(f(v0 * ftab[wo]) + f(v1 * ftab[wo + 1])) + f(v2 * ftab[wo + 2])) + f(v3 * ftab[wo + 3]));
              }
            }
          } else {
            const sx = sx0 - 3;
            const sy = sy0 - 3;
            const wo = alpha * 64;
            const inside = sx >= 0 && sy >= 0 && sx < w - 7 && sy < h - 7;
            if (!inside && (sx >= w || sx + 8 <= 0 || sy >= h || sy + 8 <= 0)) {
              for (let q = 0; q < c; q++) od[o + q] = bv[q];
              continue;
            }
            for (let q = 0; q < c; q++) {
              if (inside) {
                let sum = 0;
                for (let r = 0; r < 8; r++) {
                  const base = ((sy + r) * w + sx) * c + q;
                  let row = 0;
                  if (isInt) {
                    for (let t = 0; t < 8; t++) row += sd[base + t * c] * itab[wo + r * 8 + t];
                    sum += row;
                  } else {
                    for (let t = 0; t < 8; t++) row = t ? f(row + f(sd[base + t * c] * ftab[wo + r * 8 + t])) : f(sd[base] * ftab[wo + r * 8]);
                    sum = f(sum + row);
                  }
                }
                od[o + q] = isInt ? fix(sum) : sum;
              } else {
                const cv = bv[q];
                let sum = isInt ? cv * COEF_SCALE : cv;
                for (let r = 0; r < 8; r++) {
                  const yy = sy + r;
                  if (yy < 0 || yy >= h) continue;
                  for (let t = 0; t < 8; t++) {
                    const xx = sx + t;
                    if (xx < 0 || xx >= w) continue;
                    const v = isInt ? sd[(yy * w + xx) * c + q] - cv : f(sd[(yy * w + xx) * c + q] - cv);
                    sum = isInt ? sum + v * itab[wo + r * 8 + t] : f(sum + f(v * ftab[wo + r * 8 + t]));
                  }
                }
                od[o + q] = isInt ? fix(sum) : sum;
              }
            }
          }
        }
      }
    }
    return out;
  }

  return {
    image,
    percentile,
    median,
    mean,
    pyround,
    rgbToLab,
    rgbToLab8,
    dropAlpha,
    channel,
    resizeArea,
    resizeLinear,
    gaussianBlur,
    sobel,
    laplacianVariance,
    kernel,
    ellipseField,
    ellipseHit,
    dilate,
    erode,
    open,
    close,
    otsu,
    canny,
    findContours,
    contourArea,
    arcLength,
    convexHull,
    approxPolyDP,
    minAreaRectPoints,
    fitLineHuber,
    isContourConvex,
    intersectConvexArea,
    fillPoly,
    fillConvexPoly,
    drawLine,
    rint,
    rgbToGray,
    labToRgb,
    connectedComponents,
    getPerspectiveTransform,
    invert3,
    perspectiveTransform,
    warpPerspective,
  };
})();



// Card detection: a line-for-line port of henka_photos/detect.py.
// See that file for the reasoning behind each check; comments here only
// note where JavaScript needs care to give the same numbers.

var HK = (globalThis.HK = globalThis.HK || {});

HK.detect = (() => {
  "use strict";

  const CARD_SIZES = { standard: [63.0, 88.0], small: [59.0, 86.0] };
  const SIDE_NAMES = ["top", "right", "bottom", "left"];
  const HOLDER_SHORT_LONG = 76.2 / 101.6;

  const FLAG_TEXT = {
    no_card_found: "No card-shaped outline was found.",
    touches_photo_edge: "The card touches or runs off the edge of the photo, so part of it may be missing.",
    not_card_shape: "The outline isn't a standard card shape (partial card, sleeve/top loader edge, or several cards).",
    shape_slightly_off: "The outline's proportions are slightly off a standard card: it may be missing a dark border or following a sleeve.",
    possible_outer_border: "A faint second edge sits just outside the outline: the card may have a dark border that blends into the table.",
    strong_perspective: "The card was photographed at a steep angle; straightening is less reliable.",
    weak_edge: "The {side} edge is faint against the background.",
    background_unclear: "Just outside the {side} edge doesn't look like the table: the outline may cut into the card, or there's clutter or glare there.",
    loose_outline: "Just inside the {side} edge looks like the table: the outline may be following a sleeve or top loader instead of the card.",
    content_outside_outline: "Card-colored pixels were found outside the outline.",
    low_contrast: "The card doesn't stand out clearly from the background.",
    card_too_small: "The card fills very little of the photo.",
    multiple_cards: "More than one card-like object is in the photo.",
    nested_outline: "Two nested card-shaped outlines were found (sleeve, top loader, or the card's inner frame); the right one is uncertain.",
    card_sideways: "The card is lying sideways in the photo; its correct rotation needs a human check.",
    busy_background: "The background isn't a plain, even surface.",
    auto_rotated: "Turned upright automatically: it's a back, and its design matched the other backs this way up.",
    glare_on_card: "Bright glare covers part of the card.",
    glare_at_edge: "Glare reaches the card's edges or corners (hides whitening).",
    possibly_blurry: "The photo may be out of focus.",
    low_resolution: "The card is small in the frame; fine details may not be visible.",
    in_toploader: "The card is in a top loader: the card was outlined inside it, and the crop keeps the whole top loader.",
    toploader_check:
      "Cards in top loaders are always checked by hand for now: make sure the outline follows the card's own edge, not its inner frame or the plastic.",
  };

  const BLOCKING_FLAGS = new Set([
    "no_card_found", "touches_photo_edge", "not_card_shape", "shape_slightly_off", "possible_outer_border",
    "strong_perspective", "weak_edge", "background_unclear", "loose_outline", "content_outside_outline",
    "low_contrast", "card_too_small", "multiple_cards", "nested_outline", "card_sideways", "toploader_check",
  ]);

  const flagBase = (f) => f.split(":")[0];
  function describeFlag(flag) {
    const [base, side] = flag.split(":");
    return (FLAG_TEXT[base] || base).replace("{side}", side || "");
  }

  const clamp01 = (x) => Math.min(1, Math.max(0, x));
  const norm = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)));
  const toQuad = (flat) => [0, 1, 2, 3].map((k) => [flat[2 * k], flat[2 * k + 1]]);
  const flat = (q) => Float64Array.from(q.flat());

  function focalLengthPx(focal35, width, height) {
    if (!focal35 || focal35 <= 0) return null;
    return (focal35 / 43.27) * Math.hypot(width, height);
  }

  function orderQuad(pts) {
    let p = pts.map((v) => [v[0], v[1]]);
    const c = [(p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4, (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4];
    p.sort((a, b) => Math.atan2(a[1] - c[1], a[0] - c[0]) - Math.atan2(b[1] - c[1], b[0] - c[0]));
    let best = null;
    let bestKey = null;
    for (let r = 0; r < 4; r++) {
      const q = p.slice(r).concat(p.slice(0, r));
      const topMidY = (q[0][1] + q[1][1]) / 2;
      const tilt = Math.abs((Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]) * 180) / Math.PI);
      const key = [topMidY >= c[1] ? 1 : 0, tilt];
      if (!bestKey || key[0] < bestKey[0] || (key[0] === bestKey[0] && key[1] < bestKey[1])) {
        best = q;
        bestKey = key;
      }
    }
    return best;
  }

  function cross3(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  /** Zhang & He: true width/height of the rectangle photographed as q. */
  function rectangleAspect(q, focal, cx, cy) {
    const m1 = [q[0][0], q[0][1], 1];
    const m2 = [q[1][0], q[1][1], 1];
    const m4 = [q[2][0], q[2][1], 1];
    const m3 = [q[3][0], q[3][1], 1];
    const d2 = dot3(cross3(m2, m4), m3);
    const d3 = dot3(cross3(m3, m4), m2);
    if (!d2 || !d3) return null;
    const k2 = dot3(cross3(m1, m4), m3) / d2;
    const k3 = dot3(cross3(m1, m4), m2) / d3;
    const n2 = [k2 * m2[0] - m1[0], k2 * m2[1] - m1[1], k2 * m2[2] - m1[2]];
    const n3 = [k3 * m3[0] - m1[0], k3 * m3[1] - m1[1], k3 * m3[2] - m1[2]];
    // B = inv(A)^T inv(A) for A = [[f,0,cx],[0,f,cy],[0,0,1]]
    const Bq = (n) => {
      const u = (n[0] - cx * n[2]) / focal;
      const v = (n[1] - cy * n[2]) / focal;
      return u * u + v * v + n[2] * n[2];
    };
    const ratioSq = Bq(n2) / Bq(n3);
    if (!isFinite(ratioSq) || ratioSq <= 0) return null;
    return Math.sqrt(ratioSq);
  }

  function lineIntersection(l1, l2) {
    const [vx1, vy1, x1, y1] = l1;
    const [vx2, vy2, x2, y2] = l2;
    const det = vx1 * -vy2 - vy1 * -vx2;
    if (Math.abs(det) < 1e-9) return null;
    const t = ((x2 - x1) * -vy2 - (y2 - y1) * -vx2) / det;
    return [x1 + t * vx1, y1 + t * vy1];
  }

  function refineQuad(points, quad0) {
    const n = points.length / 2;
    const sideLen = [0, 1, 2, 3].map((k) => norm(quad0[(k + 1) % 4], quad0[k]));
    const tol = Math.max(2.0, 0.015 * (sideLen.reduce((a, b) => a + b) / 4));
    const lines = [];
    for (let k = 0; k < 4; k++) {
      const a = quad0[k];
      const b = quad0[(k + 1) % 4];
      const len = sideLen[k];
      if (len < 1) return quad0;
      const ux = (b[0] - a[0]) / len;
      const uy = (b[1] - a[1]) / len;
      const xs = [];
      const ys = [];
      for (let i = 0; i < n; i++) {
        const rx = points[2 * i] - a[0];
        const ry = points[2 * i + 1] - a[1];
        const t = rx * ux + ry * uy;
        const dist = Math.abs(rx * -uy + ry * ux);
        if (t > 0.1 * len && t < 0.9 * len && dist < tol) {
          xs.push(points[2 * i]);
          ys.push(points[2 * i + 1]);
        }
      }
      if (xs.length >= 8) lines.push(CV.fitLineHuber(xs, ys));
      else lines.push([ux, uy, a[0], a[1]]);
    }
    const out = [];
    for (let k = 0; k < 4; k++) {
      const p = lineIntersection(lines[(k + 3) % 4], lines[k]);
      if (!p) return quad0;
      out.push(p);
    }
    const diag = norm(quad0[2], quad0[0]);
    let maxMove = 0;
    for (let k = 0; k < 4; k++) maxMove = Math.max(maxMove, norm(out[k], quad0[k]));
    if (maxMove > 0.06 * diag) return quad0;
    if (!CV.isContourConvex(flat(out))) return quad0;
    return orderQuad(out);
  }

  function quadFromContour(cnt) {
    const hull = CV.convexHull(cnt);
    const peri = CV.arcLength(hull, true);
    if (peri < 40) return null;
    let quad0 = null;
    for (const eps of [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]) {
      const approx = CV.approxPolyDP(hull, eps * peri);
      if (approx.length === 8) {
        quad0 = toQuad(approx);
        break;
      }
    }
    if (!quad0) quad0 = toQuad(CV.minAreaRectPoints(hull));
    quad0 = orderQuad(quad0);
    return refineQuad(cnt, quad0);
  }

  function quadGeometry(q, ctx, h, w) {
    const top = norm(q[1], q[0]);
    const right = norm(q[2], q[1]);
    const bottom = norm(q[3], q[2]);
    const left = norm(q[0], q[3]);
    const width = (top + bottom) / 2;
    const height = (left + right) / 2;
    let aspect = ctx.focal ? rectangleAspect(q, ctx.focal, w / 2, h / 2) : null;
    const method = aspect ? "focal" : "naive";
    if (!aspect) aspect = width / Math.max(height, 1e-6);
    const shortLong = Math.min(aspect, 1 / aspect);
    const aspectDev = Math.min(...ctx.aspects.map((r) => Math.abs(shortLong - r) / r));
    const angles = [];
    for (let k = 0; k < 4; k++) {
      const v1 = [q[(k + 3) % 4][0] - q[k][0], q[(k + 3) % 4][1] - q[k][1]];
      const v2 = [q[(k + 1) % 4][0] - q[k][0], q[(k + 1) % 4][1] - q[k][1]];
      const cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (Math.hypot(...v1) * Math.hypot(...v2) + 1e-9);
      angles.push((Math.acos(Math.min(1, Math.max(-1, cosv))) * 180) / Math.PI);
    }
    const persp = Math.max(
      Math.abs(top - bottom) / Math.max(top, bottom, 1e-6),
      Math.abs(left - right) / Math.max(left, right, 1e-6),
    );
    const rot = (Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]) * 180) / Math.PI;
    return {
      width,
      height,
      aspect,
      aspect_method: method,
      short_long: shortLong,
      aspect_dev: aspectDev,
      angle_dev: Math.max(...angles.map((a) => Math.abs(a - 90))),
      perspective: persp,
      rotation_deg: rot,
      area: CV.contourArea(flat(q)),
    };
  }

  /** Nearest-pixel sample of a 3-channel image at (x, y); valid = inside. */
  function sampleAt(img, x, y, out) {
    const xi = CV.rint(x);
    const yi = CV.rint(y);
    const valid = xi >= 0 && xi < img.w && yi >= 0 && yi < img.h;
    const cx = Math.min(img.w - 1, Math.max(0, xi));
    const cy = Math.min(img.h - 1, Math.max(0, yi));
    const o = (cy * img.w + cx) * img.c;
    out[0] = img.data[o];
    out[1] = img.data[o + 1];
    out[2] = img.data[o + 2];
    return valid;
  }

  function sideFrames(q) {
    const c = [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4];
    return SIDE_NAMES.map((name, k) => {
      const a = q[k];
      const b = q[(k + 1) % 4];
      const len = norm(b, a) + 1e-9;
      const u = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
      let n = [u[1], -u[0]];
      if (n[0] * ((a[0] + b[0]) / 2 - c[0]) + n[1] * ((a[1] + b[1]) / 2 - c[1]) < 0) n = [-n[0], -n[1]];
      return { name, a, b, n };
    });
  }

  const dist3 = (p, q) => Math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2);

  function sideChecks(q, ctx) {
    const short = Math.min(norm(q[1], q[0]), norm(q[3], q[0]));
    const offs = linspace(Math.max(2.0, 0.004 * short), Math.max(4.0, 0.013 * short), 6);
    const offsFar = linspace(Math.max(5.0, 0.016 * short), Math.max(8.0, 0.03 * short), 6);
    const ts = linspace(0.12, 0.88, 48);
    const result = {};
    const v = [0, 0, 0];
    const bgv = [0, 0, 0];
    for (const { name, a, b, n } of sideFrames(q)) {
      const contrasts = [];
      let nearHits = 0;
      let farHits = 0;
      let inHits = 0;
      let okOut = 0;
      let okFar = 0;
      const total = ts.length * offs.length;
      for (const t of ts) {
        const bx = a[0] + t * (b[0] - a[0]);
        const by = a[1] + t * (b[1] - a[1]);
        const meanIn = [0, 0, 0];
        const meanOut = [0, 0, 0];
        for (const o of offs) {
          const okI = sampleAt(ctx.lab, bx - o * n[0], by - o * n[1], v);
          meanIn[0] += v[0] / offs.length;
          meanIn[1] += v[1] / offs.length;
          meanIn[2] += v[2] / offs.length;
          sampleAt(ctx.bg, bx - o * n[0], by - o * n[1], bgv);
          if (dist3(v, bgv) >= ctx.tolIn && okI) inHits++;
          const okO = sampleAt(ctx.lab, bx + o * n[0], by + o * n[1], v);
          meanOut[0] += v[0] / offs.length;
          meanOut[1] += v[1] / offs.length;
          meanOut[2] += v[2] / offs.length;
          sampleAt(ctx.bg, bx + o * n[0], by + o * n[1], bgv);
          if (okO) okOut++;
          if (dist3(v, bgv) < ctx.tolOut && okO) nearHits++;
        }
        for (const o of offsFar) {
          const okF = sampleAt(ctx.lab, bx + o * n[0], by + o * n[1], v);
          sampleAt(ctx.bg, bx + o * n[0], by + o * n[1], bgv);
          if (okF) okFar++;
          if (dist3(v, bgv) < ctx.tolOut && okF) farHits++;
        }
        contrasts.push(dist3(meanIn, meanOut));
      }
      const nearBg = nearHits / total;
      const farBg = farHits / (ts.length * offsFar.length);
      result[name] = {
        contrast: CV.median(contrasts),
        outside_bg: Math.max(nearBg, farBg),
        outside_near: nearBg,
        outside_far: farBg,
        inside_fg: inHits / total,
        off_photo: 1 - Math.min(okOut / total, okFar / (ts.length * offsFar.length)),
      };
    }
    return result;
  }

  function outerEdgeSearch(q, lab) {
    const short = Math.min(norm(q[1], q[0]), norm(q[3], q[0]));
    const dists = [];
    for (let d = Math.max(2.0, 0.005 * short); d < 0.1 * short; d += 1.0) dists.push(d);
    if (dists.length < 8) return {};
    const ts = linspace(0.15, 0.85, 40);
    const found = {};
    const v = [0, 0, 0];
    for (const { name, a, b, n } of sideFrames(q)) {
      const prof = dists.map(() => [0, 0, 0]);
      let ok = 0;
      for (const t of ts) {
        const bx = a[0] + t * (b[0] - a[0]);
        const by = a[1] + t * (b[1] - a[1]);
        dists.forEach((d, j) => {
          if (sampleAt(lab, bx + d * n[0], by + d * n[1], v)) ok++;
          prof[j][0] += v[0] / ts.length;
          prof[j][1] += v[1] / ts.length;
          prof[j][2] += v[2] / ts.length;
        });
      }
      if (ok / (ts.length * dists.length) < 0.95) continue;
      const step = [];
      for (let j = 0; j + 2 < dists.length; j++) step.push(dist3(prof[j + 2], prof[j]));
      const mid = dists.slice(1, -1);
      const med = CV.median(step);
      const mad = CV.median(step.map((s) => Math.abs(s - med))) * 1.4826;
      let best = -1;
      let bestV = -Infinity;
      mid.forEach((m, j) => {
        const val = m >= 0.035 * short && m <= 0.075 * short ? step[j] : -1;
        if (val > bestV) {
          bestV = val;
          best = j;
        }
      });
      if (!mid.some((m) => m >= 0.035 * short && m <= 0.075 * short)) continue;
      if (step[best] > Math.max(2.5, med + 5 * mad)) found[name] = mid[best] / short;
    }
    return found;
  }

  function score(cand, ctx, h, w) {
    const q = cand.quad;
    const geo = cand.geo;
    const sides = sideChecks(q, ctx);
    const short = Math.min(geo.width, geo.height);
    const kIn = Math.max(3, Math.trunc(0.015 * short) | 1);
    const kOut = Math.max(kIn + 2, Math.trunc(0.06 * short) | 1);
    const pad = kOut + 2;
    const xsQ = q.map((p) => p[0]);
    const ysQ = q.map((p) => p[1]);
    const x0 = Math.max(0, Math.trunc(Math.min(...xsQ)) - pad);
    const y0 = Math.max(0, Math.trunc(Math.min(...ysQ)) - pad);
    const x1 = Math.min(w, Math.trunc(Math.max(...xsQ)) + pad + 1);
    const y1 = Math.min(h, Math.trunc(Math.max(...ysQ)) + pad + 1);
    const rw = x1 - x0;
    const rh = y1 - y0;
    const poly = CV.image(rw, rh, 1, Uint8Array);
    CV.fillConvexPoly(poly, Float64Array.from(q.flatMap((p) => [CV.rint(p[0] - x0), CV.rint(p[1] - y0)])), 1);
    let polyN = 0;
    let fillN = 0;
    for (let y = 0; y < rh; y++) {
      for (let x = 0; x < rw; x++) {
        if (poly.data[y * rw + x]) {
          polyN++;
          if (ctx.fgMask[(y + y0) * w + (x + x0)]) fillN++;
        }
      }
    }
    const fill = fillN / Math.max(1, polyN);
    // Python opens the strong silhouette inside this region of interest. Its
    // erosion here uses the whole image instead; that differs only within
    // kOpen / 2 of the region's border and the dilation spreads it at most
    // another kOpen / 2, while the ring measured below stays at least
    // kOut / 2 + 2 away from that border, so the ring pixels come out the same.
    const kOpen = Math.max(3, Math.trunc(0.03 * short) | 1);
    const fgSolid = solidForeground(ctx, kOpen, x0, y0, rw, rh, w);
    const field = CV.ellipseField(poly.data, rw, rh, 1);
    const near = kIn > 11 ? CV.ellipseHit(field, kIn >> 1) : CV.dilate(poly, CV.kernel("ellipse", kIn)).data;
    const far = CV.ellipseHit(field, kOut >> 1);
    let ringN = 0;
    let spillN = 0;
    for (let y = 0; y < rh; y++) {
      for (let x = 0; x < rw; x++) {
        const i = y * rw + x;
        if (far[i] && !near[i]) {
          ringN++;
          if (fgSolid[i]) spillN++;
        }
      }
    }
    const spill = spillN / Math.max(1, ringN);
    const edgeMargin = Math.min(Math.min(...xsQ), Math.min(...ysQ), w - 1 - Math.max(...xsQ), h - 1 - Math.max(...ysQ));
    const sAspect =
      geo.aspect_method === "focal"
        ? clamp01(1 - Math.max(0, geo.aspect_dev - 0.012) / 0.023)
        : clamp01(1 - Math.max(0, geo.aspect_dev - 0.02) / 0.03);
    const sideScores = {};
    for (const [name, s] of Object.entries(sides)) {
      sideScores[name] = {
        edge: clamp01((s.contrast - 6) / 12),
        outside: clamp01((s.outside_bg - 0.55) / 0.35),
        inside: clamp01((s.inside_fg - 0.3) / 0.5),
      };
    }
    let sSides = Infinity;
    for (const v of Object.values(sideScores)) sSides = Math.min(sSides, v.edge, v.outside, v.inside);
    const subs = {
      aspect: sAspect,
      angle: clamp01(1 - Math.max(0, geo.angle_dev - 8) / 12),
      perspective: clamp01(1 - Math.max(0, geo.perspective - 0.12) / 0.15),
      fill: clamp01((fill - 0.75) / 0.15),
      spill: clamp01(1 - Math.max(0, spill - 0.02) / 0.06),
      sides: sSides,
      area: clamp01((geo.area / (h * w) - 0.02) / 0.04),
      border: clamp01(edgeMargin / Math.max(2.0, 0.004 * Math.min(h, w))),
    };
    const vals = Object.values(subs);
    cand.score = Math.min(...vals);
    cand.soft = vals.reduce((a, b) => a + b) / vals.length;
    cand.sideScores = sideScores;
    cand.metrics = { ...geo, fill, spill, edge_margin_px: edgeMargin, area_frac: geo.area / (h * w), sides, subscores: subs };
  }

  function solidForeground(ctx, kOpen, x0, y0, rw, rh, w) {
    // Opening = erosion then dilation. The erosion of the whole strong
    // silhouette by any radius is a threshold of one distance field to its
    // background, computed once per photo; the dilation runs on the region.
    if (!ctx.bgField) ctx.bgField = CV.ellipseField(ctx.strongFg, w, ctx.lab.h, 0);
    const r = kOpen >> 1;
    const er = CV.image(rw, rh, 1, Uint8Array);
    const { D, dmin } = ctx.bgField;
    const h = ctx.lab.h;
    const limit = r * r - 0.25;
    for (let y = 0; y < rh; y++) {
      const gy = y + y0;
      const up = gy - r >= 0 ? (gy - r) * w : -1;
      const dn = gy + r < h ? (gy + r) * w : -1;
      for (let x = 0; x < rw; x++) {
        const gx = x + x0;
        const gi = gy * w + gx;
        const hitBg = D[gi] <= limit || (up >= 0 && dmin[up + gx] === 0) || (dn >= 0 && dmin[dn + gx] === 0);
        er.data[y * rw + x] = hitBg ? 0 : 1;
      }
    }
    return kOpen > 11 ? CV.ellipseHit(CV.ellipseField(er.data, rw, rh, 1), r) : CV.dilate(er, CV.kernel("ellipse", kOpen)).data;
  }

  const iou = (a, b) => {
    const inter = CV.intersectConvexArea(flat(a), flat(b));
    const union = CV.contourArea(flat(a)) + CV.contourArea(flat(b)) - inter;
    return union > 0 ? inter / union : 0;
  };
  const containment = (inner, outer) => CV.intersectConvexArea(flat(inner), flat(outer)) / Math.max(1e-6, CV.contourArea(flat(inner)));

  function tableModel(lab, cardMask) {
    const { w, h } = lab;
    const minSide = Math.min(h, w);
    const k = Math.max(3, Math.trunc(0.05 * minSide) | 1);
    const grown = CV.dilate(cardMask, CV.kernel("ellipse", k));
    const table = CV.image(w, h, 1, Float32Array);
    let tableN = 0;
    for (let i = 0; i < w * h; i++) {
      table.data[i] = grown.data[i] ? 0 : 1;
      tableN += table.data[i];
    }
    if (tableN / (w * h) < 0.15) return null;
    const f = 8;
    const sw = Math.max(8, Math.trunc(w / f));
    const sh = Math.max(8, Math.trunc(h / f));
    const labS = CV.resizeArea(lab, sw, sh);
    const wtArea = CV.resizeArea(table, sw, sh);
    const wt = CV.image(sw, sh, 1, Float32Array);
    for (let i = 0; i < sw * sh; i++) wt.data[i] = wtArea.data[i] > 0.99 ? 1 : 0;
    let model = null;
    let filled = null;
    for (const sigma of [0.05, 0.12, 0.3]) {
      const sg = sigma * Math.min(sw, sh);
      const weighted = CV.image(sw, sh, 3, Float32Array);
      for (let i = 0; i < sw * sh; i++) for (let c = 0; c < 3; c++) weighted.data[3 * i + c] = labS.data[3 * i + c] * wt.data[i];
      const num = CV.gaussianBlur(weighted, sg);
      const den = CV.gaussianBlur(wt, sg);
      if (!model) {
        model = CV.image(sw, sh, 3, Float32Array);
        filled = new Uint8Array(sw * sh);
        for (let i = 0; i < sw * sh; i++) {
          const d = Math.max(den.data[i], 1e-6);
          for (let c = 0; c < 3; c++) model.data[3 * i + c] = num.data[3 * i + c] / d;
          filled[i] = den.data[i] > 0.05 ? 1 : 0;
        }
      } else {
        for (let i = 0; i < sw * sh; i++) {
          if (!filled[i]) {
            const d = Math.max(den.data[i], 1e-6);
            for (let c = 0; c < 3; c++) model.data[3 * i + c] = num.data[3 * i + c] / d;
          }
          if (den.data[i] > 0.05) filled[i] = 1;
        }
      }
      if (filled.every((x) => x)) break;
    }
    const full = CV.resizeLinear(model, w, h);
    const resid = [];
    for (let i = 0; i < w * h; i++) {
      if (!table.data[i]) continue;
      const e0 = lab.data[3 * i] - full.data[3 * i];
      const e1 = lab.data[3 * i + 1] - full.data[3 * i + 1];
      const e2 = lab.data[3 * i + 2] - full.data[3 * i + 2];
      resid.push(Math.sqrt(e0 * e0 + e1 * e1 + e2 * e2));
    }
    return { model: full, spread: CV.percentile(resid, 75) };
  }

  function segmentation(lab, bg, bgSpread) {
    const { w, h } = lab;
    const d = new Float32Array(w * h);
    const d8 = CV.image(w, h, 1, Uint8Array);
    for (let i = 0; i < w * h; i++) {
      const b0 = bg.c === 3 ? bg.data[3 * i] : bg.flat[0];
      const b1 = bg.c === 3 ? bg.data[3 * i + 1] : bg.flat[1];
      const b2 = bg.c === 3 ? bg.data[3 * i + 2] : bg.flat[2];
      const e0 = lab.data[3 * i] - b0;
      const e1 = lab.data[3 * i + 1] - b1;
      const e2 = lab.data[3 * i + 2] - b2;
      d[i] = Math.sqrt(e0 * e0 + e1 * e1 + e2 * e2);
      d8.data[i] = Math.min(255, Math.max(0, Math.trunc(d[i] * 2)));
    }
    const otsu = CV.otsu(d8);
    const thr = Math.max(otsu / 2, bgSpread * 2.5 + 4, 10);
    let mask = CV.image(w, h, 1, Uint8Array);
    for (let i = 0; i < w * h; i++) mask.data[i] = d[i] > thr ? 1 : 0;
    const minSide = Math.min(w, h);
    mask = CV.open(mask, CV.kernel("ellipse", 3));
    const k = Math.max(3, Math.trunc(0.006 * minSide) | 1);
    mask = CV.close(mask, CV.kernel("ellipse", k));
    const contours = CV.findContours(mask, "external");
    const filled = CV.image(w, h, 1, Uint8Array);
    for (const c of contours) {
      CV.fillPoly(filled, c, 1);
      for (let p = 0; p < c.length; p += 2) filled.data[c[p + 1] * w + c[p]] = 1;
    }
    return { filled, contours, thr, d };
  }

  const L_TO_8 = Math.fround(2.55);

  function edgeContours(lab) {
    const { w, h } = lab;
    let edges = null;
    for (let c = 0; c < 3; c++) {
      const ch = CV.image(w, h, 1, Uint8Array);
      for (let i = 0; i < w * h; i++) {
        // float32 arithmetic, as NumPy does it: L * 2.55 can land just under an integer.
        const v = c === 0 ? Math.fround(lab.data[3 * i] * L_TO_8) : lab.data[3 * i + c] * 2 + 128;
        ch.data[i] = Math.trunc(Math.min(255, Math.max(0, v)));
      }
      const med = CV.median(ch.data);
      const lo = Math.max(10.0, 0.66 * med);
      let hi = Math.max(30.0, 1.33 * med);
      hi = Math.max(hi, lo + 20);
      const e = CV.canny(ch, lo, hi);
      if (!edges) edges = e;
      else for (let i = 0; i < w * h; i++) edges.data[i] |= e.data[i];
    }
    const k3 = CV.kernel("rect", 3);
    edges = CV.dilate(CV.dilate(edges, k3), k3);
    return CV.findContours(edges, "tree");
  }

  function linePeaks(grad, gw, gh, vertical, lo, hi, span, k = 4) {
    // grad is gw x gh; for horizontal lines work on the transpose.
    const nAlong = vertical ? gh : gw;
    const nAcross = vertical ? gw : gh;
    const at = vertical ? (along, across) => grad[along * gw + across] : (along, across) => grad[across * gw + along];
    const ys = [];
    for (let y = Math.trunc(span[0] * nAlong); y < Math.trunc(span[1] * nAlong); y++) ys.push(y);
    const mid = nAlong / 2;
    const best = new Float64Array(nAcross).fill(-1);
    const bestS = new Float64Array(nAcross);
    const a0 = Math.max(1, Math.trunc(lo * nAcross));
    const a1 = Math.min(nAcross - 1, Math.trunc(hi * nAcross));
    const vals = new Float64Array(ys.length);
    for (const s of linspace(-0.035, 0.035, 15)) {
      for (let x = a0; x < a1; x++) {
        for (let j = 0; j < ys.length; j++) {
          const xi = Math.min(nAcross - 1, Math.max(0, CV.rint(x + s * (ys[j] - mid))));
          vals[j] = at(ys[j], xi);
        }
        const strength = CV.median(vals);
        if (strength > best[x]) {
          best[x] = strength;
          bestS[x] = s;
        }
      }
    }
    const order = Array.from(best.keys()).sort((p, q) => best[q] - best[p] || q - p);
    const sep = Math.max(2, Math.trunc(0.012 * nAcross));
    const peaks = [];
    for (const i of order) {
      if (best[i] <= 0 || peaks.length >= k) break;
      if (peaks.every((p) => Math.abs(i - p.x0) > sep)) peaks.push({ x0: i, s: bestS[i], strength: best[i], mid });
    }
    return peaks;
  }

  const linePts = (p, vertical) => (vertical ? [p.s, 1.0, p.x0, p.mid] : [1.0, p.s, p.mid, p.x0]);

  function findCardInHolder(lab, holder, holderAspect, cardAspects) {
    const tw = 600;
    const th = holderAspect < 1 ? CV.pyround(tw / holderAspect) : CV.pyround(tw * holderAspect);
    const dst = [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]];
    const mat = CV.getPerspectiveTransform(holder, dst);
    const rect = CV.gaussianBlur(CV.warpPerspective(lab, mat, tw, th, "linear"), 1.0);
    const gx = new Float32Array(tw * th);
    const gy = new Float32Array(tw * th);
    for (let c = 0; c < 3; c++) {
      const ch = CV.channel(rect, c);
      const sx = CV.sobel(ch, 1, 0);
      const sy = CV.sobel(ch, 0, 1);
      for (let i = 0; i < tw * th; i++) {
        gx[i] += sx.data[i] * sx.data[i];
        gy[i] += sy.data[i] * sy.data[i];
      }
    }
    for (let i = 0; i < tw * th; i++) {
      gx[i] = Math.sqrt(gx[i]);
      gy[i] = Math.sqrt(gy[i]);
    }
    const all = new Float32Array(2 * tw * th);
    all.set(gx);
    all.set(gy, tw * th);
    const noise = CV.median(all) + 1e-3;
    const left = linePeaks(gx, tw, th, true, 0.015, 0.22, [0.25, 0.75]);
    const right = linePeaks(gx, tw, th, true, 0.78, 0.985, [0.25, 0.75]);
    const top = linePeaks(gy, tw, th, false, 0.01, 0.24, [0.25, 0.75]);
    const bottom = linePeaks(gy, tw, th, false, 0.76, 0.995, [0.25, 0.75]);
    const combos = [];
    for (const pl of left) {
      for (const pr of right) {
        for (const pt of top) {
          for (const pb of bottom) {
            const corners = [
              lineIntersection(linePts(pt, false), linePts(pl, true)),
              lineIntersection(linePts(pt, false), linePts(pr, true)),
              lineIntersection(linePts(pb, false), linePts(pr, true)),
              lineIntersection(linePts(pb, false), linePts(pl, true)),
            ];
            if (corners.some((c) => !c)) continue;
            const wq = (norm(corners[1], corners[0]) + norm(corners[2], corners[3])) / 2;
            const hq = (norm(corners[3], corners[0]) + norm(corners[2], corners[1])) / 2;
            const dev = Math.min(...cardAspects.map((r) => Math.abs(wq / hq - r) / r));
            if (dev > 0.03 || !(wq / tw >= 0.74 && wq / tw <= 0.92 && hq / th >= 0.78 && hq / th <= 0.96)) continue;
            const weakest = Math.min(pl.strength, pr.strength, pt.strength, pb.strength) / noise;
            combos.push({ q: corners, weakest, dev, frac: [wq / tw, hq / th] });
          }
        }
      }
    }
    if (!combos.length) return null;
    const f = 4;
    const smallRect = CV.resizeArea(rect, Math.trunc(tw / f), Math.trunc(th / f));
    const rim = CV.pyround((0.025 * tw) / f);
    const grow = CV.kernel("rect", 2 * Math.max(1, CV.pyround((0.01 * tw) / f)) + 1);
    const inner = CV.image(smallRect.w, smallRect.h, 1, Uint8Array);
    for (let y = rim; y < smallRect.h - rim; y++) for (let x = rim; x < smallRect.w - rim; x++) inner.data[y * smallRect.w + x] = 1;
    for (const cb of combos) {
      const card = CV.image(smallRect.w, smallRect.h, 1, Uint8Array);
      CV.fillConvexPoly(card, Float64Array.from(cb.q.flatMap((p) => [CV.rint(p[0] / f), CV.rint(p[1] / f)])), 1);
      const grown = CV.dilate(card, grow);
      const px = [];
      for (let i = 0; i < inner.data.length; i++) if (inner.data[i] && !grown.data[i]) px.push(i);
      if (px.length < 50) cb.even = 0;
      else {
        const med = [0, 1, 2].map((c) => CV.median(px.map((i) => smallRect.data[3 * i + c])));
        let close = 0;
        for (const i of px) {
          const o = 3 * i;
          if (Math.hypot(smallRect.data[o] - med[0], smallRect.data[o + 1] - med[1], smallRect.data[o + 2] - med[2]) < 10) close++;
        }
        cb.even = close / px.length;
      }
    }
    const strongest = Math.max(...combos.map((cb) => cb.weakest));
    let pick = combos[0];
    let pickKey = -Infinity;
    for (const cb of combos) {
      const key = cb.weakest / strongest + 0.5 * cb.even - 10 * cb.dev;
      if (key > pickKey) {
        pickKey = key;
        pick = cb;
      }
    }
    const pts = orderQuad(CV.perspectiveTransform(pick.q, CV.invert3(mat)));
    const short = Math.min(norm(pts[1], pts[0]), norm(pts[3], pts[0]));
    const offs = linspace(0.006, 0.02, 5).map((o) => o * short);
    const ts = linspace(0.15, 0.85, 40);
    const sideContrast = {};
    const v = [0, 0, 0];
    for (const { name, a, b, n } of sideFrames(pts)) {
      const diffs = [];
      for (const t of ts) {
        const bx = a[0] + t * (b[0] - a[0]);
        const by = a[1] + t * (b[1] - a[1]);
        const mi = [0, 0, 0];
        const mo = [0, 0, 0];
        for (const o of offs) {
          sampleAt(lab, bx - o * n[0], by - o * n[1], v);
          for (let c = 0; c < 3; c++) mi[c] += v[c] / offs.length;
          sampleAt(lab, bx + o * n[0], by + o * n[1], v);
          for (let c = 0; c < 3; c++) mo[c] += v[c] / offs.length;
        }
        diffs.push(dist3(mi, mo));
      }
      sideContrast[name] = CV.percentile(diffs, 20);
    }
    return {
      quad: pts,
      edge_strength: pick.weakest,
      ring_evenness: Math.round(pick.even * 1000) / 1000,
      side_contrast: sideContrast,
      aspect_dev: pick.dev,
      size_in_holder: pick.frac.map((x) => Math.round(x * 1000) / 1000),
    };
  }

  function glareAndBlur(small, q, portrait) {
    const [tw, th] = portrait ? [500, 700] : [700, 500];
    const m = CV.getPerspectiveTransform(q, [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]);
    const card = CV.warpPerspective(small, m, tw, th, "linear");
    let clipped = CV.image(tw, th, 1, Uint8Array);
    for (let i = 0; i < tw * th; i++) {
      const o = 3 * i;
      clipped.data[i] = Math.min(card.data[o], card.data[o + 1], card.data[o + 2]) >= 250 ? 1 : 0;
    }
    clipped = CV.open(clipped, CV.kernel("rect", 3));
    const bw = Math.trunc(0.05 * Math.min(tw, th));
    let glare = 0;
    let bandN = 0;
    let bandGlare = 0;
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const g = clipped.data[y * tw + x];
        glare += g;
        if (y < bw || y >= th - bw || x < bw || x >= tw - bw) {
          bandN++;
          bandGlare += g;
        }
      }
    }
    const gray = CV.rgbToGray(card);
    const inner = CV.image(tw - 2 * bw, th - 2 * bw, 1, Uint8Array);
    for (let y = 0; y < inner.h; y++) for (let x = 0; x < inner.w; x++) inner.data[y * inner.w + x] = gray.data[(y + bw) * tw + x + bw];
    return { glare_frac: glare / (tw * th), glare_edge_frac: bandGlare / bandN, sharpness: CV.laplacianVariance(inner) };
  }

  function cardInToploader(unique, ctx, small, scale, flags, baseMetrics, bgRgb, minConfidence) {
    const { w, h } = small;
    for (const c of unique.slice(0, 4)) {
      if (!(c.geo.short_long >= HOLDER_SHORT_LONG - 0.015 && c.geo.short_long <= HOLDER_SHORT_LONG + 0.025)) continue;
      if (CV.median(Object.values(c.sideScores).map((v) => v.edge)) < 0.5) continue;
      const res = findCardInHolder(ctx.lab, c.quad, c.geo.aspect, ctx.aspects);
      if (!res) continue;
      const weakestSide = Math.min(...Object.values(res.side_contrast));
      const confidence = Math.min(clamp01((res.edge_strength - 6) / 6), clamp01((weakestSide - 10) / 8));
      if (confidence < minConfidence) continue;
      const q = res.quad;
      const geo = quadGeometry(q, ctx, h, w);
      const kept = flags.filter((f) => ["busy_background", "multiple_cards", "touches_photo_edge"].includes(flagBase(f)));
      let held = ["in_toploader", "toploader_check", ...kept];
      if (geo.width > geo.height * 1.05) held.push("card_sideways");
      if (clamp01(1 - Math.max(0, geo.angle_dev - 8) / 12) < 0.6 || clamp01(1 - Math.max(0, geo.perspective - 0.12) / 0.15) < 0.6) {
        held.push("strong_perspective");
      }
      const gb = glareAndBlur(small, q, geo.width <= geo.height);
      if (gb.glare_frac > 0.01) held.push("glare_on_card");
      if (gb.glare_edge_frac > 0.01) held.push("glare_at_edge");
      if (gb.sharpness < 20) held.push("possibly_blurry");
      const cardShortFull = Math.min(geo.width, geo.height) / scale;
      if (cardShortFull < 900) held.push("low_resolution");
      held = [...new Set(held)];
      const blocking = held.filter((f) => BLOCKING_FLAGS.has(flagBase(f)));
      const { quad: _q, ...holderInfo } = res;
      return {
        status: blocking.length ? "needs_review" : "ok",
        confidence: Math.round(confidence * 1000) / 1000,
        quad: q.map((p) => [p[0] / scale, p[1] / scale]),
        flags: held,
        metrics: { ...baseMetrics, ...geo, ...gb, holder: { ...holderInfo, holder_short_long: c.geo.short_long }, card_short_side_px: cardShortFull },
        background_rgb: bgRgb,
        method: "toploader",
        aspect: geo.aspect_method === "focal" ? geo.aspect : null,
        holder_quad: c.quad.map((p) => [p[0] / scale, p[1] / scale]),
      };
    }
    return null;
  }

  /** Detect the card in an upright sRGB photo ({w, h, c: 3, data: Uint8Array}). */
  function detectCard(rgb, focal35 = null, cardSizes = ["standard"], workLongSide = 1600, minConfidence = 0.75) {
    const wFull = rgb.w;
    const hFull = rgb.h;
    const scale = Math.min(1, workLongSide / Math.max(hFull, wFull));
    const small = scale < 1 ? CV.resizeArea(rgb, CV.pyround(wFull * scale), CV.pyround(hFull * scale)) : rgb;
    const { w, h } = small;
    const lab = CV.gaussianBlur(CV.rgbToLab(small), 1.2);

    const ringW = Math.max(4, Math.trunc(0.04 * Math.min(h, w)));
    const ring = [[], [], []];
    const pushPx = (x, y) => {
      const o = (y * w + x) * 3;
      ring[0].push(lab.data[o]);
      ring[1].push(lab.data[o + 1]);
      ring[2].push(lab.data[o + 2]);
    };
    for (let y = 0; y < ringW; y++) for (let x = 0; x < w; x++) pushPx(x, y);
    for (let y = h - ringW; y < h; y++) for (let x = 0; x < w; x++) pushPx(x, y);
    for (let y = 0; y < h; y++) for (let x = 0; x < ringW; x++) pushPx(x, y);
    for (let y = 0; y < h; y++) for (let x = w - ringW; x < w; x++) pushPx(x, y);
    const bg = ring.map((c) => CV.median(c));
    let bgSpread = CV.percentile(ring[0].map((_, i) => Math.sqrt((ring[0][i] - bg[0]) ** 2 + (ring[1][i] - bg[1]) ** 2 + (ring[2][i] - bg[2]) ** 2)), 75);
    const L8 = Math.min(255, Math.max(0, Math.trunc((bg[0] * 255) / 100)));
    const a8 = Math.min(255, Math.max(0, Math.trunc(bg[1] + 128)));
    const b8 = Math.min(255, Math.max(0, Math.trunc(bg[2] + 128)));
    const bgRgb = CV.labToRgb((L8 * 100) / 255, a8 - 128, b8 - 128);

    let seg = segmentation(lab, { c: 1, flat: bg }, bgSpread);
    let bgMap = { w, h, c: 3, data: new Float32Array(w * h * 3) };
    for (let i = 0; i < w * h; i++) bgMap.data.set(bg, 3 * i);
    let tableKind = "flat";
    const modeled = tableModel(lab, seg.filled);
    if (modeled) {
      bgMap = modeled.model;
      bgSpread = modeled.spread;
      tableKind = "lighting-corrected";
      seg = segmentation(lab, bgMap, bgSpread);
    }
    const fFull = focalLengthPx(focal35, wFull, hFull);
    const tolIn = Math.max(14.0, 3.5 * bgSpread);
    const strongFg = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) strongFg[i] = seg.d[i] >= tolIn ? 1 : 0;
    const aspects = cardSizes.filter((s) => CARD_SIZES[s]).map((s) => Math.min(...CARD_SIZES[s]) / Math.max(...CARD_SIZES[s]));
    const ctx = {
      lab,
      bg: bgMap,
      tolIn,
      tolOut: Math.max(18.0, 4.5 * bgSpread),
      fgMask: seg.filled.data,
      strongFg,
      focal: fFull ? fFull * scale : null,
      aspects: aspects.length ? aspects : [63 / 88],
    };

    const minArea = 0.02 * h * w;
    const candidates = [];
    for (const [method, contours] of [["segment", seg.contours], ["edges", edgeContours(lab)]]) {
      for (const cnt of contours) {
        if (cnt.length / 2 < 20) continue;
        const hullArea = CV.contourArea(CV.convexHull(cnt));
        if (hullArea < minArea || hullArea > 0.995 * h * w) continue;
        const q = quadFromContour(cnt);
        if (!q) continue;
        candidates.push({ quad: q, method, geo: quadGeometry(q, ctx, h, w) });
      }
    }
    candidates.sort((a, b) => {
      const ka = clamp01(1 - a.geo.aspect_dev / 0.25);
      const kb = clamp01(1 - b.geo.aspect_dev / 0.25);
      return kb - ka || b.geo.area - a.geo.area;
    });
    const pool = [];
    for (const c of candidates) {
      if (pool.every((u) => iou(c.quad, u.quad) < 0.995)) pool.push(c);
      if (pool.length >= 24) break;
    }
    for (const c of pool) score(c, ctx, h, w);
    pool.sort((a, b) => b.score - a.score || b.soft - a.soft);
    const unique = [];
    for (const c of pool) if (unique.every((u) => iou(c.quad, u.quad) < 0.97)) unique.push(c);

    const baseMetrics = {
      work_scale: scale,
      background_spread: bgSpread,
      table_model: tableKind,
      tolerance_inside: ctx.tolIn,
      tolerance_outside: ctx.tolOut,
      segment_threshold: seg.thr,
      focal_px: fFull,
      candidates: unique.length,
    };
    let flags = [];
    if (bgSpread > 10) flags.push("busy_background");
    if (!unique.length) {
      return { status: "failed", confidence: 0, quad: null, flags: ["no_card_found", ...flags], metrics: baseMetrics, background_rgb: bgRgb, method: "", aspect: null, holder_quad: null };
    }
    const best = unique[0];
    const m = best.metrics;
    if (m.edge_margin_px < Math.max(2.0, 0.004 * Math.min(h, w)) || Object.values(m.sides).some((s) => s.off_photo > 0)) flags.push("touches_photo_edge");
    if (m.subscores.aspect < 0.5) flags.push("not_card_shape");
    else if (m.subscores.aspect < 0.9) flags.push("shape_slightly_off");
    if (m.subscores.angle < 0.6 || m.subscores.perspective < 0.6) flags.push("strong_perspective");
    for (const [name, ss] of Object.entries(best.sideScores)) {
      if (ss.edge < 0.5) flags.push(`weak_edge:${name}`);
      if (ss.outside < 0.6) flags.push(`background_unclear:${name}`);
      if (ss.inside < 0.5) flags.push(`loose_outline:${name}`);
    }
    if (m.subscores.spill < 0.5) flags.push("content_outside_outline");
    if (m.subscores.fill < 0.75) flags.push("low_contrast");
    if (m.subscores.area < 0.75) flags.push("card_too_small");
    const portrait = m.width <= m.height;
    if (m.width > m.height * 1.05) flags.push("card_sideways");

    const outer = outerEdgeSearch(best.quad, lab);
    const outerVals = Object.values(outer);
    if (outerVals.length >= 2) {
      const mn = Math.min(...outerVals);
      const mx = Math.max(...outerVals);
      const avg = outerVals.reduce((a, b) => a + b) / outerVals.length;
      if ((mx - mn) / avg < 0.35) flags.push("possible_outer_border");
    }
    for (const other of unique.slice(1)) {
      const areaRatio = other.metrics.area / Math.max(1, m.area);
      const [inner, outerC] = areaRatio < 1 ? [other, best] : [best, other];
      const contained = containment(inner.quad, outerC.quad) > 0.95;
      const linear = Math.sqrt(Math.min(areaRatio, 1 / Math.max(areaRatio, 1e-6)));
      if (contained && linear >= 0.78 && linear <= 0.975 && other.score >= 0.5 && other.score >= best.score - 0.15) {
        flags.push("nested_outline");
        continue;
      }
      const separate = containment(other.quad, best.quad) < 0.2 && containment(best.quad, other.quad) < 0.2 && areaRatio >= 0.3 && areaRatio <= 3.0;
      if (separate) {
        const sub = other.metrics.subscores;
        const edgesOk = CV.median(Object.values(other.sideScores).map((v) => v.edge)) > 0.5;
        if (edgesOk && (sub.aspect > 0.3 || sub.border < 1)) flags.push("multiple_cards");
      }
    }
    const gb = glareAndBlur(small, best.quad, portrait);
    if (gb.glare_frac > 0.01) flags.push("glare_on_card");
    if (gb.glare_edge_frac > 0.01) flags.push("glare_at_edge");
    if (gb.sharpness < 20) flags.push("possibly_blurry");
    const cardShortFull = Math.min(m.width, m.height) / scale;
    if (cardShortFull < 900) flags.push("low_resolution");

    flags = [...new Set(flags)];
    const blocking = flags.filter((f) => BLOCKING_FLAGS.has(flagBase(f)));
    const confidence = best.score;
    const status = confidence >= minConfidence && !blocking.length ? "ok" : "needs_review";
    const metrics = { ...baseMetrics, ...m, ...gb, outer_edges: outer, card_short_side_px: cardShortFull };
    if (status !== "ok") {
      const held = cardInToploader(unique, ctx, small, scale, flags, baseMetrics, bgRgb, minConfidence);
      if (held) return held;
    }
    return {
      status,
      confidence: Math.round(confidence * 1000) / 1000,
      quad: best.quad.map((p) => [p[0] / scale, p[1] / scale]),
      flags,
      metrics,
      background_rgb: bgRgb,
      method: best.method,
      aspect: m.aspect_method === "focal" ? m.aspect : null,
      holder_quad: null,
    };
  }

  return { detectCard, describeFlag, flagBase, BLOCKING_FLAGS, FLAG_TEXT, CARD_SIZES, orderQuad, rectangleAspect, focalLengthPx };
})();



// Straighten and crop a card, and lay it out for a listing: a port of
// henka_photos/crop.py. Only geometric operations: the perspective warp
// resamples existing pixels, and padding goes outside the card. Nothing on
// the card is redrawn or cleaned up.

var HK = (globalThis.HK = globalThis.HK || {});

HK.crop = (() => {
  "use strict";

  const norm = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  function cardSize(q, aspect) {
    const top = norm(q[1], q[0]);
    const bottom = norm(q[2], q[3]);
    const left = norm(q[3], q[0]);
    const right = norm(q[2], q[1]);
    let w = (top + bottom) / 2;
    let h = (left + right) / 2;
    if (aspect) {
      if (h >= w) w = h * aspect;
      else h = w / aspect;
    }
    return [w, h];
  }

  /** Warp the card upright at native resolution with a margin of real
   * photo around it. Returns {image, margin}. rotation: 90° clockwise steps. */
  function rectify(rgb, quad, marginPct = 0.03, rotation = 0, fillRgb = [0, 0, 0], aspect = null) {
    const [wf, hf] = cardSize(quad, aspect);
    const w = CV.pyround(wf);
    const h = CV.pyround(hf);
    const m = CV.pyround(marginPct * Math.min(w, h));
    const dst = [[m, m], [m + w, m], [m + w, m + h], [m, m + h]];
    const mat = CV.getPerspectiveTransform(quad, dst);
    let out = CV.warpPerspective(rgb, mat, w + 2 * m, h + 2 * m, "lanczos4", fillRgb);
    for (let s = 0; s < ((rotation % 4) + 4) % 4; s++) out = rotateCW(out);
    return { image: out, margin: m };
  }

  function rotateCW(img) {
    const { w, h, c } = img;
    const out = CV.image(h, w, c, img.data.constructor);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const src = (y * w + x) * c;
        const dst = (x * h + (h - 1 - y)) * c;
        for (let k = 0; k < c; k++) out.data[dst + k] = img.data[src + k];
      }
    }
    return out;
  }

  function holderExtent(quad, holderQuad, aspect = null) {
    const [w, h] = cardSize(quad, aspect);
    const mat = CV.getPerspectiveTransform(quad, [[0, 0], [w, 0], [w, h], [0, h]]);
    const pts = CV.perspectiveTransform(holderQuad, mat);
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const reach = Math.max(-Math.min(...xs), Math.max(...xs) - w, -Math.min(...ys), Math.max(...ys) - h, 0);
    return reach / Math.min(w, h);
  }

  function smooth1d(prof, n, sigma) {
    // prof: n x 3 rows; edge-padded Gaussian smoothing along the rows.
    sigma = Math.max(1, sigma);
    const r = Math.trunc(3 * sigma);
    const k = [];
    let sum = 0;
    for (let i = -r; i <= r; i++) {
      const v = Math.exp(-0.5 * (i / sigma) ** 2);
      k.push(v);
      sum += v;
    }
    const out = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let t = -r; t <= r; t++) {
          const j = Math.min(n - 1, Math.max(0, i + t));
          acc += prof[3 * j + c] * k[t + r];
        }
        out[3 * i + c] = acc / sum;
      }
    }
    return out;
  }

  function surroundField(img, band, cw, ch, x0, y0, step = 4) {
    const { w: iw, h: ih } = img;
    const d = img.data;
    const top = new Float64Array(iw * 3);
    const bottom = new Float64Array(iw * 3);
    const left = new Float64Array(ih * 3);
    const right = new Float64Array(ih * 3);
    for (let x = 0; x < iw; x++) {
      for (let b = 0; b < band; b++) {
        for (let c = 0; c < 3; c++) {
          top[3 * x + c] += d[(b * iw + x) * 3 + c] / band;
          bottom[3 * x + c] += d[((ih - 1 - b) * iw + x) * 3 + c] / band;
        }
      }
    }
    for (let y = 0; y < ih; y++) {
      for (let b = 0; b < band; b++) {
        for (let c = 0; c < 3; c++) {
          left[3 * y + c] += d[(y * iw + b) * 3 + c] / band;
          right[3 * y + c] += d[(y * iw + iw - 1 - b) * 3 + c] / band;
        }
      }
    }
    const T = smooth1d(top, iw, 0.08 * iw);
    const B = smooth1d(bottom, iw, 0.08 * iw);
    const L = smooth1d(left, ih, 0.08 * ih);
    const R = smooth1d(right, ih, 0.08 * ih);
    const fill = [0, 1, 2].map((c) => {
      const vals = [];
      for (const arr of [T, B, L, R]) for (let i = c; i < arr.length; i += 3) vals.push(arr[i]);
      return CV.median(vals);
    });
    const gh = Math.ceil(ch / step);
    const gw = Math.ceil(cw / step);
    const x1 = x0 + iw - 1;
    const y1 = y0 + ih - 1;
    const pad = Math.max(20.0, 0.5 * Math.min(x0, y0, cw - 1 - x1, ch - 1 - y1));
    const grid = CV.image(gw, gh, 3, Float32Array);
    for (let gy = 0; gy < gh; gy++) {
      const yy = gy * step;
      const v = Math.trunc(Math.min(Math.max(yy - y0, 0), ih - 1));
      for (let gx = 0; gx < gw; gx++) {
        const xx = gx * step;
        const u = Math.trunc(Math.min(Math.max(xx - x0, 0), iw - 1));
        const w0 = 1 / (Math.abs(yy - y0) + 1) ** 2;
        const w1 = 1 / (Math.abs(yy - y1) + 1) ** 2;
        const w2 = 1 / (Math.abs(xx - x0) + 1) ** 2;
        const w3 = 1 / (Math.abs(xx - x1) + 1) ** 2;
        const ws = w0 + w1 + w2 + w3;
        const ox = Math.max(x0 - xx, xx - x1, 0);
        const oy = Math.max(y0 - yy, yy - y1, 0);
        const fall = Math.exp(-Math.hypot(ox, oy) / pad);
        for (let c = 0; c < 3; c++) {
          const col = (w0 * T[3 * u + c] + w1 * B[3 * u + c] + w2 * L[3 * v + c] + w3 * R[3 * v + c]) / ws;
          grid.data[(gy * gw + gx) * 3 + c] = col * fall + fill[c] * (1 - fall);
        }
      }
    }
    const field = CV.resizeLinear(grid, cw, ch);
    const out = CV.image(cw, ch, 3, Uint8Array);
    for (let i = 0; i < out.data.length; i++) out.data[i] = Math.min(255, Math.max(0, Math.trunc(field.data[i] + 0.5)));
    return out;
  }

  /** Place the straightened card on a consistent canvas (see crop.py). */
  function renderListing(crop, marginPx, fillRgb, canvas = "square", longSide = 2000, cardFill = 0.86, holderPx = 0) {
    const ch = crop.h;
    const cw = crop.w;
    const cardH = ch - 2 * marginPx;
    const cardW = cw - 2 * marginPx;
    const portrait = cardH >= cardW;
    const aspect = canvas === "square" ? 1 : canvas === "4:5" ? (portrait ? 4 / 5 : 5 / 4) : cw / ch;
    holderPx = Math.min(Math.max(0, Math.trunc(holderPx)), marginPx);
    const cardLong = Math.max(cardH, cardW) + 2 * holderPx;
    const canvasLongNative = canvas !== "card" ? cardLong / cardFill : Math.max(ch, cw);
    const targetLong = Math.min(longSide, canvasLongNative);
    const scale = Math.min(1, targetLong / canvasLongNative);
    let canvasW;
    let canvasH;
    if (aspect >= 1) {
      canvasW = CV.pyround(targetLong);
      canvasH = CV.pyround(targetLong / aspect);
    } else {
      canvasH = CV.pyround(targetLong);
      canvasW = CV.pyround(targetLong * aspect);
    }
    let img = scale < 1 ? CV.resizeArea(crop, Math.max(1, CV.pyround(cw * scale)), Math.max(1, CV.pyround(ch * scale))) : crop;
    let m = (marginPx - holderPx) * (img.h / ch);
    if (img.w > canvasW || img.h > canvasH) {
      const ox = Math.max(0, Math.floor((img.w - canvasW) / 2));
      const oy = Math.max(0, Math.floor((img.h - canvasH) / 2));
      const nw = Math.min(img.w - ox, canvasW);
      const nh = Math.min(img.h - oy, canvasH);
      const cut = CV.image(nw, nh, 3, Uint8Array);
      for (let y = 0; y < nh; y++) cut.data.set(img.data.subarray(((y + oy) * img.w + ox) * 3, ((y + oy) * img.w + ox + nw) * 3), y * nw * 3);
      img = cut;
      m = Math.max(0, m - Math.max(ox, oy));
    }
    const { w: iw, h: ih } = img;
    const x0 = Math.floor((canvasW - iw) / 2);
    const y0 = Math.floor((canvasH - ih) / 2);
    const band = CV.pyround(0.6 * m);
    let base;
    if (band < 2) {
      base = CV.image(canvasW, canvasH, 3, Uint8Array);
      for (let i = 0; i < canvasW * canvasH; i++) base.data.set(fillRgb, 3 * i);
    } else {
      base = surroundField(img, band, canvasW, canvasH, x0, y0);
    }
    for (let y = 0; y < ih; y++) {
      for (let x = 0; x < iw; x++) {
        const s = (y * iw + x) * 3;
        const d = ((y + y0) * canvasW + x + x0) * 3;
        const edge = Math.min(y, ih - 1 - y, x, iw - 1 - x);
        if (band >= 2 && edge < band) {
          // The outer 60% of the margin fades into the surround: no seam.
          const a = Math.min(1, Math.max(0, edge / band));
          for (let c = 0; c < 3; c++) base.data[d + c] = Math.min(255, Math.max(0, Math.trunc(img.data[s + c] * a + base.data[d + c] * (1 - a) + 0.5)));
        } else {
          base.data[d] = img.data[s];
          base.data[d + 1] = img.data[s + 1];
          base.data[d + 2] = img.data[s + 2];
        }
      }
    }
    return base;
  }

  return { rectify, holderExtent, renderListing, rotateCW, cardSize };
})();



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


