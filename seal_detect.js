// seal_detect.js
// Pre-marks likely engineer seals on each page using the trained seal model.
// The reviewer still checks every box before anything is redacted.
//
// Reads these files from the app folder the user picks (Load seal model button):
//   seal_v3.onnx, ort.wasm.min.js, ort-wasm-simd-threaded.wasm
// Uses these globals from SealRedact.html: DISPLAY_DPI, mkCanvas, msg, pages,
// entryFor, curPage, curIdx, boxes, sel, draw.

(function () {
  const MODEL_SIZE = 960;     // the model was trained at this input size
  const DETECT_CONF = 0.05;   // low on purpose: favor catching seals, reviewer removes false alarms
  const NMS_IOU = 0.7;        // overlap rule for merging duplicate detections (Ultralytics default)
  const PAD_VALUE = 114;      // grey padding, same as Ultralytics letterbox

  let ort = null;
  let session = null;
  let inputName = null;
  let outputName = null;

  function iou(a, b) {
    const ix0 = Math.max(a[0], b[0]), iy0 = Math.max(a[1], b[1]);
    const ix1 = Math.min(a[2], b[2]), iy1 = Math.min(a[3], b[3]);
    const inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    const union = areaA + areaB - inter;
    return union > 0 ? inter / union : 0;
  }

  function clampBox(b, wPt, hPt) {
    return [Math.max(0, b[0]), Math.max(0, b[1]), Math.min(wPt, b[2]), Math.min(hPt, b[3])];
  }

  // readBytes(name) must return a Promise<Uint8Array> for a file in the model folder.
  async function loadModelFrom(readBytes) {
    const ortText = new TextDecoder().decode(await readBytes('ort.wasm.min.js'));
    await new Promise((ok, bad) => {
      const s = document.createElement('script');
      s.src = URL.createObjectURL(new Blob([ortText], { type: 'text/javascript' }));
      s.onload = ok;
      s.onerror = () => bad(new Error('ONNX runtime failed to load'));
      document.head.appendChild(s);
    });
    ort = window.ort;
    if (!ort) throw new Error('ONNX runtime did not start');
    // The runtime fetches its companion module by relative path, which does not work
    // from a blob script, so both files are handed over as blob URLs instead.
    const mjsText = new TextDecoder().decode(await readBytes('ort-wasm-simd-threaded.mjs'));
    const wasmBytes = await readBytes('ort-wasm-simd-threaded.wasm');
    ort.env.wasm.wasmPaths = {
      mjs: URL.createObjectURL(new Blob([mjsText], { type: 'text/javascript' })),
      wasm: URL.createObjectURL(new Blob([wasmBytes], { type: 'application/wasm' })),
    };
    ort.env.wasm.wasmBinary = wasmBytes;
    ort.env.wasm.numThreads = 1;   // the page is not cross-origin isolated, so no threads
    ort.env.wasm.proxy = false;
    session = await ort.InferenceSession.create(await readBytes('seal_v3.onnx'), {
      executionProviders: ['wasm'],
    });
    inputName = session.inputNames[0];
    outputName = session.outputNames[0];
  }

  // Runs the model on a page rendered at DISPLAY_DPI (the same resolution as the training images).
  // Returns seal boxes in PDF points as [x0, y0, x1, y1].
  async function detectOnCanvas(canvas, pageWidthPt, pageHeightPt) {
    const W = canvas.width, H = canvas.height;
    const r = Math.min(MODEL_SIZE / W, MODEL_SIZE / H);
    const nw = Math.round(W * r), nh = Math.round(H * r);
    const padX = Math.round((MODEL_SIZE - nw) / 2);
    const padY = Math.round((MODEL_SIZE - nh) / 2);

    const cv = document.createElement('canvas');
    cv.width = MODEL_SIZE;
    cv.height = MODEL_SIZE;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
    cx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    cx.drawImage(canvas, 0, 0, W, H, padX, padY, nw, nh);

    const px = cx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    const N = MODEL_SIZE * MODEL_SIZE;
    const input = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) {
      input[i] = px[i * 4] / 255;
      input[N + i] = px[i * 4 + 1] / 255;
      input[2 * N + i] = px[i * 4 + 2] / 255;
    }
    const tensor = new ort.Tensor('float32', input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
    const result = await session.run({ [inputName]: tensor });
    const out = result[outputName];
    const data = out.data;          // layout: [1, 5, anchors]; rows are x, y, w, h, score
    const A = out.dims[2];
    const ptPerPx = 72 / DISPLAY_DPI;

    const cands = [];
    for (let a = 0; a < A; a++) {
      const score = data[4 * A + a];
      if (score < DETECT_CONF) continue;
      const xc = data[a], yc = data[A + a], w = data[2 * A + a], h = data[3 * A + a];
      // undo the letterbox: model pixels -> canvas pixels -> PDF points
      const x0 = (xc - w / 2 - padX) / r, y0 = (yc - h / 2 - padY) / r;
      const x1 = (xc + w / 2 - padX) / r, y1 = (yc + h / 2 - padY) / r;
      const box = clampBox([x0 * ptPerPx, y0 * ptPerPx, x1 * ptPerPx, y1 * ptPerPx], pageWidthPt, pageHeightPt);
      if (box[2] - box[0] > 2 && box[3] - box[1] > 2) cands.push({ box, score });
    }

    // Merge duplicate detections of the same seal, highest confidence first.
    cands.sort((p, q) => q.score - p.score);
    const kept = [];
    for (const c of cands) {
      if (kept.every(k => iou(k.box, c.box) < NMS_IOU)) kept.push(c);
    }
    return kept.map(k => k.box);
  }

  // Called by the app each time a page is shown. Skips pages already reviewed,
  // pages with boxes already drawn, and pages already checked.
  async function autoDetectPage(i) {
    if (!session) return;
    const { fi, pno } = pages[i];
    const e = entryFor(fi, pno);
    if (e.autoDone || e.reviewed || (e.boxes && e.boxes.length)) return;
    e.autoDone = true;

    const page = curPage;
    const pts = page.getViewport({ scale: 1 });
    try {
      msg('Looking for seals on this page...');
      const lv = page.getViewport({ scale: DISPLAY_DPI / 72 });
      const lc = mkCanvas(lv);
      await page.render({ canvasContext: lc.getContext('2d'), viewport: lv }).promise;
      const found = await detectOnCanvas(lc, pts.width, pts.height);
      lc.width = 0;
      lc.height = 0;

      if (curIdx === i) {
        if (boxes.length === 0) {
          boxes = found.map(b => b.slice());
          e.boxes = boxes.map(b => b.slice());
        }
        sel = -1;
        draw();
        msg(`${found.length} seal(s) marked automatically. Check each box before applying.`);
      } else {
        e.boxes = found.map(b => b.slice());
      }
    } catch (err) {
      console.error(err);
      msg('Automatic seal detection failed on this page. Mark seals by hand. ' + err.message);
    }
  }

  // Runs the model on one pdf.js page and returns its seal boxes in PDF points.
  async function detectOnPage(page) {
    const pts = page.getViewport({ scale: 1 });
    const lv = page.getViewport({ scale: DISPLAY_DPI / 72 });
    const lc = mkCanvas(lv);
    await page.render({ canvasContext: lc.getContext('2d'), viewport: lv }).promise;
    const found = await detectOnCanvas(lc, pts.width, pts.height);
    lc.width = 0;
    lc.height = 0;
    return found;
  }

  // Button handler: the user picks the app folder that holds the model files.
  async function loadSealModel() {
    try {
      const dir = await window.showDirectoryPicker({ mode: 'read' });
      msg('Loading seal model...');
      await loadModelFrom(async name => {
        const fh = await dir.getFileHandle(name);
        return new Uint8Array(await (await fh.getFile()).arrayBuffer());
      });
      msg('Seal model loaded. Each page will be pre-marked as you open it.');
    } catch (err) {
      if (err && err.name === 'AbortError') { msg('Seal model not loaded.'); return; }
      console.error(err);
      msg('Could not load the seal model: ' + err.message);
    }
  }

  window.autoDetectPage = autoDetectPage;
  window.sealDetect = { loadModelFrom, detectOnCanvas, detectOnPage, iou, isLoaded: () => !!session };
  const btn = document.getElementById('b-model');
  if (btn) btn.addEventListener('click', loadSealModel);
})();
