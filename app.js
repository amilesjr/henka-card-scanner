// Henka Crop: choose photos, get square listing photos, save them.
// Photos are processed in background workers on this device and are never
// uploaded. Nothing is stored: closing the page clears the list.

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const OK_FILE = /\.(jpe?g|png|heic|heif)$/i;
  const SHARE_GROUP = 20; // photos per share sheet

  // ---------------------------------------------------------------- workers

  class Pool {
    constructor(size) {
      this.size = size;
      this.workers = [];
      this.queue = [];
    }
    run(op, args) {
      return new Promise((resolve, reject) => {
        this.queue.push({ op, args, resolve, reject });
        this.pump();
      });
    }
    pump() {
      while (this.queue.length) {
        let w = this.workers.find((x) => !x.job);
        if (!w && this.workers.length < this.size) w = this.spawn();
        if (!w) return;
        w.job = this.queue.shift();
        w.postMessage({ id: 0, op: w.job.op, args: w.job.args });
      }
    }
    spawn() {
      const w = new Worker("worker.js?v=202610270306");
      w.job = null;
      w.onmessage = (e) => {
        const job = w.job;
        w.job = null;
        if (e.data.ok) job.resolve(e.data.out);
        else job.reject(new Error(e.data.error));
        this.pump();
      };
      w.onerror = (e) => {
        e.preventDefault();
        const job = w.job;
        w.terminate();
        this.workers = this.workers.filter((x) => x !== w);
        if (job) job.reject(new Error("This photo was too big for the phone's memory."));
        this.pump();
      };
      this.workers.push(w);
      return w;
    }
  }

  const coarse = window.matchMedia("(pointer: coarse)").matches;
  const pool = new Pool(Math.max(1, Math.min(coarse ? 2 : 3, (navigator.hardwareConcurrency || 2) - 1)));

  // ---------------------------------------------------------------- state

  /** One chosen photo: {id, file, name, taken, state, det, rotation,
   * confirmed, full (Blob), preview (URL), saved, error}. */
  const items = [];
  let nextId = 1;
  let open = null; // item shown in the sheet

  const isCheck = (it) => it.state === "done" && it.det && it.det.status === "needs_review" && !it.confirmed;
  const isReady = (it) => it.state === "done" && it.full && !isCheck(it);
  const noCard = (it) => it.state === "done" && !it.full;

  function outName(it) {
    const stem = it.name.replace(/\.[^.]+$/, "") || "card";
    const same = items.filter((x) => x.name === it.name);
    return same.length > 1 ? `${stem}-${same.indexOf(it) + 1}_listing.jpg` : `${stem}_listing.jpg`;
  }

  // ---------------------------------------------------------------- processing

  function forget(it) {
    if (it.preview) URL.revokeObjectURL(it.preview);
    if (it.fullUrl) URL.revokeObjectURL(it.fullUrl);
    it.preview = it.fullUrl = null;
  }

  /** When the photo was taken ("2026-10-24T19:39:41.128"), from its EXIF;
   * null if it doesn't say. The EXIF block sits at the start of a JPEG. */
  async function takenAt(file) {
    try {
      let meta = HK.exif.read(await file.slice(0, 256 * 1024).arrayBuffer());
      if (!meta.captured_at && meta.kind === "heif") meta = HK.exif.read(await file.arrayBuffer());
      return meta.captured_at;
    } catch {
      return null;
    }
  }

  // The phone's photo picker doesn't always hand photos over in the order
  // they were taken: list them by capture time, or by file name (IMG_0828,
  // IMG_0829...) when some photos don't say when they were taken.
  const byName = new Intl.Collator(undefined, { numeric: true }).compare;
  function sortItems() {
    const timed = items.every((it) => it.taken);
    items.sort((a, b) => (timed && a.taken !== b.taken ? (a.taken < b.taken ? -1 : 1) : 0) || byName(a.name, b.name) || a.id - b.id);
  }

  async function addFiles(list) {
    const files = [...list].filter((f) => OK_FILE.test(f.name) || /^image\//.test(f.type));
    if (!files.length) return toast("Those files aren't photos.", true);
    const added = files.map((file) => ({ id: nextId++, file, name: file.name || "photo.jpg", state: "waiting", rotation: 0, confirmed: false, saved: false }));
    await Promise.all(added.map(async (it) => (it.taken = await takenAt(it.file))));
    items.push(...added);
    sortItems();
    renderGrid();
    update();
    for (const it of items) if (added.includes(it)) process(it);
  }

  async function process(it, extra = {}) {
    it.state = "working";
    renderTile(it);
    update();
    try {
      const out = await pool.run("crop", { file: it.file, det: extra.quad || extra.rotation !== undefined ? it.det : null, rotation: it.rotation, quad: extra.quad || null });
      if (!items.includes(it)) return;
      it.det = out.det;
      forget(it);
      it.full = out.full || null;
      it.fullUrl = it.full ? URL.createObjectURL(it.full) : null;
      it.preview = out.preview ? URL.createObjectURL(out.preview) : null;
      if (extra.quad) it.confirmed = true;
      it.saved = false;
      it.state = "done";
    } catch (e) {
      it.state = "error";
      it.error = e.message;
    }
    renderTile(it);
    update();
    if (open === it) renderSheet();
  }

  // ---------------------------------------------------------------- grid

  function tileInner(it) {
    let inner;
    if (it.state === "waiting") inner = `<div class="state">Waiting</div>`;
    else if (it.state === "working") inner = `<div class="state"><div class="spinner" aria-label="Cropping"></div></div>`;
    else if (it.state === "error") inner = `<div class="state">Can't open this file</div>`;
    else if (!it.full) inner = `<div class="state">No card found<br>Tap to place the corners</div>`;
    else {
      const badge = isCheck(it)
        ? `<span class="badge check">Check</span>`
        : it.saved
          ? `<span class="badge saved">Saved</span>`
          : it.det.status === "fixed"
            ? `<span class="badge fixed">Fixed</span>`
            : "";
      inner = `<img src="${esc(it.preview)}" alt="">${badge}`;
    }
    return `<div class="sq">${inner}</div><span class="name">${esc(it.name)}</span>`;
  }

  function renderGrid() {
    $("#grid").innerHTML = items.map((it) => `<button class="tile" type="button" data-id="${it.id}" aria-label="${esc(it.name)}">${tileInner(it)}</button>`).join("");
  }

  function renderTile(it) {
    const el = $(`.tile[data-id="${it.id}"]`);
    if (el) el.innerHTML = tileInner(it);
  }

  function update() {
    const any = items.length > 0;
    $("#intro").hidden = any;
    $("#results").hidden = !any;
    $("#savebar").hidden = !any;
    $("#add-more").hidden = !any;
    const working = items.filter((it) => it.state === "waiting" || it.state === "working").length;
    const ready = items.filter(isReady);
    const check = items.filter(isCheck).length;
    const none = items.filter(noCard).length + items.filter((it) => it.state === "error").length;
    const parts = [];
    if (working) parts.push(`Cropping ${Math.min(items.length, items.length - working + 1)} of ${items.length}…`);
    else parts.push(`${items.length} photo${items.length === 1 ? "" : "s"}`);
    if (check) parts.push(`${check} to check`);
    if (none) parts.push(`${none} without a crop`);
    $("#summary").textContent = parts.join(" · ") + (check ? ". Tap a photo marked Check to look at it." : "");
    const unsaved = ready.filter((it) => !it.saved);
    $("#save-count").textContent = ready.length ? `${ready.length} ready${unsaved.length < ready.length ? `, ${ready.length - unsaved.length} saved` : ""}` : working ? "Cropping…" : "Nothing ready yet";
    const btn = $("#save-all");
    const n = Math.min(unsaved.length, SHARE_GROUP);
    btn.disabled = !unsaved.length;
    btn.textContent = !ready.length ? "Save photos" : !unsaved.length ? "All saved" : unsaved.length > SHARE_GROUP ? `Save next ${n}` : `Save ${n} photo${n === 1 ? "" : "s"}`;
  }

  // ---------------------------------------------------------------- saving

  const canShareFiles = (() => {
    try {
      return !!(navigator.canShare && navigator.canShare({ files: [new File([new Blob(["x"], { type: "image/jpeg" })], "x.jpg", { type: "image/jpeg" })] }));
    } catch {
      return false;
    }
  })();

  /** Hand photos to the share sheet (Save Image puts them in Photos), or
   * download them on a computer. Returns true when they were handed over. */
  async function save(list) {
    const files = list.map((it) => new File([it.full], outName(it), { type: "image/jpeg" }));
    if (canShareFiles) {
      try {
        await navigator.share({ files });
        return true;
      } catch (e) {
        if (e.name === "AbortError") return false;
        toast("The share sheet didn't open. Try saving fewer photos at once.", true);
        return false;
      }
    }
    // Computers: one file, or a ZIP of several.
    const blob = files.length === 1 ? files[0] : await HK.zip.write(files.map((f) => ({ name: f.name, data: f })));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = files.length === 1 ? files[0].name : "henka-listing-photos.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    return true;
  }

  async function saveNext() {
    const batch = items.filter((it) => isReady(it) && !it.saved).slice(0, SHARE_GROUP);
    if (!batch.length) return;
    if (await save(batch)) {
      batch.forEach((it) => {
        it.saved = true;
        renderTile(it);
      });
      const left = items.filter((it) => isReady(it) && !it.saved).length;
      toast(left ? `${batch.length} handed over. ${left} more to save.` : canShareFiles ? "Done. Choose Save Image in the share sheet to put them in Photos." : "Downloaded.");
    }
    update();
  }

  // ---------------------------------------------------------------- photo sheet

  function openSheet(it) {
    open = it;
    renderSheet();
    $("#sheet").hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeSheet() {
    open = null;
    $("#sheet").hidden = true;
    document.body.style.overflow = "";
  }

  function renderSheet() {
    const it = open;
    if (!it) return;
    const i = items.indexOf(it);
    let body;
    if (it.state === "waiting" || it.state === "working") {
      body = `<div class="big"><div class="state"><div class="spinner"></div></div></div>`;
    } else if (it.state === "error") {
      body = `<div class="big"><div class="state">${esc(it.error)}</div></div>`;
    } else {
      const check = isCheck(it);
      const problems = (it.det.problems || []).filter(Boolean);
      body = `
        <div class="big">${it.full ? `<img src="${esc(it.fullUrl)}" alt="Listing photo">` : `<div class="state">No card was found in this photo.</div>`}</div>
        ${
          check
            ? `<div class="notice"><b>Check this crop</b>The card's outline wasn't certain${problems.length ? ":" : "."}${problems.length ? `<ul>${problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}</div>
               <div class="row"><button class="btn" data-act="fix">Fix corners</button><button class="btn btn-primary" data-act="confirm">Looks right</button></div>`
            : !it.full
              ? `<div class="row"><button class="btn btn-primary" data-act="fix">Place the corners</button></div>`
              : ""
        }
        ${
          it.full
            ? `<div class="tools ${check ? "two" : ""}"><button class="btn" data-act="rotl">⟲ Rotate left</button><button class="btn" data-act="rotr">⟳ Rotate right</button>${check ? "" : `<button class="btn" data-act="fix">Fix corners</button>`}</div>`
            : ""
        }
        <button class="link remove" data-act="remove">Remove from the list</button>`;
    }
    $("#sheet").innerHTML = `
      <div class="sheet-head">
        <button class="icon-btn" data-act="close" aria-label="Close">✕</button>
        <span class="title">${esc(it.name)} · ${i + 1} of ${items.length}</span>
        <button class="icon-btn" data-act="prev" aria-label="Previous photo" ${i > 0 ? "" : "disabled"}>‹</button>
        <button class="icon-btn" data-act="next" aria-label="Next photo" ${i < items.length - 1 ? "" : "disabled"}>›</button>
      </div>
      <div class="sheet-body">${body}</div>
      ${isReady(it) ? `<div class="sheet-foot"><div class="row"><button class="btn btn-primary" data-act="save">${it.saved ? "Save again" : "Save this photo"}</button></div></div>` : ""}`;
  }

  async function sheetAction(act) {
    const it = open;
    const i = items.indexOf(it);
    if (act === "close") return closeSheet();
    if (act === "prev" && i > 0) return openSheet(items[i - 1]);
    if (act === "next" && i < items.length - 1) return openSheet(items[i + 1]);
    if (act === "confirm") {
      it.confirmed = true;
      renderTile(it);
      update();
      renderSheet();
      return;
    }
    if (act === "rotl" || act === "rotr") {
      it.rotation = (it.rotation + (act === "rotr" ? 1 : 3)) % 4;
      renderSheet();
      return process(it, { rotation: it.rotation });
    }
    if (act === "fix") return openEditor(it);
    if (act === "save") {
      if (await save([it])) {
        it.saved = true;
        renderTile(it);
        update();
        renderSheet();
      }
      return;
    }
    if (act === "remove") {
      items.splice(i, 1);
      forget(it);
      renderGrid();
      update();
      if (items.length) openSheet(items[Math.min(i, items.length - 1)]);
      else closeSheet();
    }
  }

  // ---------------------------------------------------------------- corner editor

  async function openEditor(it) {
    const ed = $("#editor");
    ed.innerHTML = `
      <div class="editor-head">
        <button class="btn btn-quiet" data-ed="cancel">Cancel</button>
        <span class="title">Place the corners</span>
        <button class="btn btn-primary" data-ed="apply" disabled>Done</button>
      </div>
      <p class="editor-help">Drag each circle onto a corner of the card, where its straight edges would meet.</p>
      <div class="editor-stage"><div class="state"><div class="spinner"></div></div></div>`;
    ed.hidden = false;
    let orig;
    try {
      orig = await pool.run("original", { file: it.file });
    } catch (e) {
      ed.hidden = true;
      return toast(e.message, true);
    }
    const url = URL.createObjectURL(orig.image);
    const stage = ed.querySelector(".editor-stage");
    stage.innerHTML = `<img alt="Original photo"><svg></svg><div class="loupe"></div>`;
    const img = stage.querySelector("img");
    const svg = stage.querySelector("svg");
    const loupe = stage.querySelector(".loupe");
    await new Promise((r) => {
      img.onload = r;
      img.src = url;
    });
    const q0 =
      (it.det && it.det.quad) ||
      [[0.28, 0.22], [0.72, 0.22], [0.72, 0.78], [0.28, 0.78]].map(([x, y]) => [x * orig.width, y * orig.height]);
    let pts = []; // in displayed pixels
    const scale = () => img.clientWidth / orig.width;
    const layout = () => {
      svg.setAttribute("width", img.clientWidth);
      svg.setAttribute("height", img.clientHeight);
      svg.style.left = img.offsetLeft + "px";
      svg.style.top = img.offsetTop + "px";
    };
    const draw = () => {
      const r = 15;
      svg.innerHTML =
        `<polygon points="${pts.map((p) => p.join(",")).join(" ")}" fill="rgba(127,194,166,.12)" stroke="#7fc2a6" stroke-width="2"/>` +
        pts.map((p, k) => `<circle class="handle" data-k="${k}" cx="${p[0]}" cy="${p[1]}" r="${r}"/>`).join("");
    };
    const place = () => {
      const s = scale();
      pts = q0.map(([x, y]) => [x * s, y * s]);
      layout();
      draw();
    };
    place();
    ed.querySelector('[data-ed="apply"]').disabled = false;
    const onResize = () => {
      const old = +svg.getAttribute("width");
      if (old > 0) {
        const k = img.clientWidth / old;
        pts = pts.map(([x, y]) => [x * k, y * k]);
      }
      layout();
      draw();
    };
    window.addEventListener("resize", onResize);

    let drag = null;
    svg.onpointerdown = (e) => {
      const h = e.target.closest(".handle");
      if (!h) return;
      drag = +h.dataset.k;
      svg.setPointerCapture(e.pointerId);
      e.preventDefault();
    };
    svg.onpointermove = (e) => {
      if (drag === null) return;
      const rect = svg.getBoundingClientRect();
      const x = Math.min(img.clientWidth * 1.05, Math.max(-img.clientWidth * 0.05, e.clientX - rect.left));
      const y = Math.min(img.clientHeight * 1.05, Math.max(-img.clientHeight * 0.05, e.clientY - rect.top));
      pts[drag] = [x, y];
      draw();
      const zoom = 4;
      const size = 150;
      loupe.style.display = "block";
      loupe.style.backgroundImage = `url("${url}")`;
      loupe.style.backgroundSize = `${img.clientWidth * zoom}px ${img.clientHeight * zoom}px`;
      loupe.style.backgroundPosition = `${-(x * zoom - size / 2)}px ${-(y * zoom - size / 2)}px`;
      // Keep the loupe clear of the finger.
      loupe.style.left = img.offsetLeft + x + (x > img.clientWidth / 2 ? -size - 34 : 34) + "px";
      loupe.style.top = img.offsetTop + y + (y > img.clientHeight / 2 ? -size - 34 : 34) + "px";
    };
    svg.onpointerup = svg.onpointercancel = () => {
      drag = null;
      loupe.style.display = "none";
    };

    const finish = () => {
      window.removeEventListener("resize", onResize);
      ed.hidden = true;
      ed.innerHTML = "";
      URL.revokeObjectURL(url);
    };
    ed.onclick = (e) => {
      const b = e.target.closest("[data-ed]");
      if (!b) return;
      if (b.dataset.ed === "cancel") return finish();
      const s = scale();
      const quad = pts.map(([x, y]) => [x / s, y / s]);
      finish();
      it.det = it.det || { status: "needs_review", fill: [0, 0, 0], problems: [] };
      process(it, { quad });
    };
  }

  // ---------------------------------------------------------------- toast

  let toastTimer;
  function toast(msg, isError = false) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "show" + (isError ? " error" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.className = ""), isError ? 5000 : 3200);
  }

  // ---------------------------------------------------------------- events

  document.addEventListener("change", (e) => {
    if (e.target.matches("[data-pick]") && e.target.files.length) {
      addFiles(e.target.files);
      e.target.value = "";
    }
  });
  $("#grid").addEventListener("click", (e) => {
    const t = e.target.closest(".tile");
    if (t) openSheet(items.find((it) => it.id === +t.dataset.id));
  });
  $("#sheet").addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (b) sheetAction(b.dataset.act);
  });
  $("#save-all").addEventListener("click", saveNext);
  let clearArmed = false;
  $("#clear").addEventListener("click", (e) => {
    if (!clearArmed) {
      clearArmed = true;
      e.target.textContent = "Tap again to clear";
      setTimeout(() => {
        clearArmed = false;
        e.target.textContent = "Clear all";
      }, 3000);
      return;
    }
    clearArmed = false;
    e.target.textContent = "Clear all";
    for (const it of items) forget(it);
    items.length = 0;
    renderGrid();
    update();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#editor").hidden) return;
    if (!open) return;
    if (e.key === "Escape") closeSheet();
    if (e.key === "ArrowLeft") sheetAction("prev");
    if (e.key === "ArrowRight") sheetAction("next");
  });
  window.addEventListener("beforeunload", (e) => {
    if (items.some((it) => isReady(it) && !it.saved)) {
      e.preventDefault();
      e.returnValue = "";
    }
  });

  if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch(() => {});
  update();
  window.HenkaCrop = { items, isReady, isCheck }; // for tests
})();
