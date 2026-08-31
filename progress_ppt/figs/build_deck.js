const pptxgen = require("pptxgenjs");

const SCRATCH = "./_build";
const V2 = "/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2";
const OUT = "/Users/baioqile/Documents/roof_proj/progress_ppt/progress_report_instance_seg.pptx";

const IMG = {
  semOverview: { path: `${SCRATCH}/sem_overview.png`, ar: 2036 / 1619 },
  semZoom: { path: `${SCRATCH}/sem_zoom.png`, ar: 1704 / 1679 },
  instOverview: { path: `${V2}/qa_overview_rgb.png`, ar: 2240 / 1760 },
  instZoom: { path: `${V2}/qa_zoom.png`, ar: 1.0 },
  f1: { path: `${SCRATCH}/figs_en/toy_f1_scene.png`, ar: 2175 / 840 },
  f2: { path: `${SCRATCH}/figs_en/toy_f2_raster.png`, ar: 2025 / 810 },
  f3: { path: `${SCRATCH}/figs_en/toy_f3_components.png`, ar: 1290 / 930 },
  f4: { path: `${SCRATCH}/figs_en/toy_f4_closing.png`, ar: 2475 / 750 },
  f5: { path: `${SCRATCH}/figs_en/toy_f5_cores.png`, ar: 1290 / 930 },
  f6: { path: `${SCRATCH}/figs_en/toy_f6_watershed.png`, ar: 2475 / 750 },
  f7: { path: `${SCRATCH}/figs_en/toy_f7_final.png`, ar: 2070 / 915 },
  f8: { path: `${SCRATCH}/figs_en/toy_f8_gatea.png`, ar: 1980 / 780 },
};

const INK = "1A1A1A";
const TEXT = "333333";
const MUTED = "6B6B6B";
const BORDER = "8C8C8C";
const ACCENT = "B01E1E";
const ACCENT_TINT = "F7E9E9";
const FONT = "Calibri";

const W = 13.33, H = 7.5, M = 0.55;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

function slideBase() {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  return s;
}

function title(s, text) {
  s.addText(text, {
    x: M, y: 0.3, w: W - 2 * M, h: 0.65, isTextBox: true, margin: 0,
    fontFace: FONT, fontSize: 26, bold: true, color: INK, align: "left",
  });
}

// one accent "real data" line at the bottom of a slide
function factLine(s, text, y) {
  s.addText(text, {
    x: M, y, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
    fontFace: FONT, fontSize: 15, bold: true, color: ACCENT, align: "left",
  });
}

// big centered figure below the title
function centerImage(s, img, maxW, maxH, yTop) {
  let w = maxW, h = w / img.ar;
  if (h > maxH) { h = maxH; w = h * img.ar; }
  s.addImage({ path: img.path, x: (W - w) / 2, y: yTop, w, h });
  return { w, h, yBottom: yTop + h };
}

function shortLines(s, items, x, y, w, size) {
  const runs = items.map((it) => ({
    text: it,
    options: {
      bullet: { code: "2022", indent: 12 }, breakLine: true,
      paraSpaceAfter: size * 0.7, color: TEXT, fontSize: size, fontFace: FONT,
    },
  }));
  s.addText(runs, { x, y, w, h: 5.0, isTextBox: true, margin: 0, align: "left", valign: "top" });
}

function stepBox(s, x, y, w, h, name, param, opts = {}) {
  const fill = opts.accent ? ACCENT_TINT : "FFFFFF";
  const line = opts.accent ? ACCENT : BORDER;
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, line: { color: line, width: 1.25 }, rectRadius: 0.07,
  });
  const runs = [
    { text: name, options: { bold: true, fontSize: 14, color: opts.accent ? ACCENT : INK, breakLine: !!param } },
  ];
  if (param) runs.push({ text: param, options: { fontSize: 10.5, color: MUTED } });
  s.addText(runs, {
    x: x + 0.08, y, w: w - 0.16, h, isTextBox: true, margin: 0,
    fontFace: FONT, align: "center", valign: "middle",
  });
}

function arrowH(s, x1, x2, y) {
  s.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2), y, w: Math.abs(x2 - x1), h: 0,
    line: { color: BORDER, width: 1.5, endArrowType: "triangle" },
    flipH: x2 < x1,
  });
}

function arrowV(s, x, y1, y2) {
  s.addShape(pres.ShapeType.line, {
    x, y: Math.min(y1, y2), w: 0, h: Math.abs(y2 - y1),
    line: { color: BORDER, width: 1.5, endArrowType: "triangle" },
  });
}

// ---------------- 1: RandLA-Net recap ----------------
{
  const s = slideBase();
  title(s, "Last time: semantic segmentation with RandLA-Net");
  const imgH = 5.7, imgW = imgH * IMG.semOverview.ar; // ~7.17
  s.addImage({ path: IMG.semOverview.path, x: M, y: 1.25, w: imgW, h: imgH });
  shortLines(s, [
    "2.76 km² east Columbus, OH",
    "19 USGS 2019 tiles · 76.35 M points (~30 pts/m²)",
    "4 classes: ground / vegetation / building / other",
    "building IoU 0.978 (training tiles, checkpoint metric)",
    "→ 8.6 M building points (11.3 %) = input to this stage",
  ], M + imgW + 0.5, 1.6, W - 2 * M - imgW - 0.5, 15);
  s.addNotes("Quick recap: 19 tiles, 76 million points, RandLA-Net four-class semantic segmentation over the full study area. Building IoU 0.978 is a checkpoint-selection number on the labeled training tiles — a proper spatial hold-out is still future work. The 8.6 million building-labeled points are the input to this stage.");
}

// ---------------- 2: task ----------------
{
  const s = slideBase();
  title(s, "This stage: split the building points into individual buildings");
  const imgH = 5.55, imgW = imgH * IMG.semZoom.ar; // ~5.63
  s.addImage({ path: IMG.semZoom.path, x: M, y: 1.25, w: imgW, h: imgH });
  s.addText("Goal: one instance ID per building.", {
    x: M + imgW + 0.5, y: 1.6, w: W - 2 * M - imgW - 0.5, h: 0.5, isTextBox: true, margin: 0,
    fontFace: FONT, fontSize: 16, bold: true, color: ACCENT, align: "left",
  });
  shortLines(s, [
    "tree bridges weld neighboring houses together",
    "occlusion fragments split one house apart",
    "parking lots, bridges, power lines hide in the mask",
  ], M + imgW + 0.5, 2.35, W - 2 * M - imgW - 0.5, 15);
  s.addNotes("The semantic mask is noisy in both directions: mislabeled tree points bridge neighboring houses into one blob, tree-shadowed porches break off into fragments, and some objects in the mask were never buildings at all. The method has to handle all three.");
}

// ---------------- 3: flowchart ----------------
{
  const s = slideBase();
  title(s, "Method: a 2.5-D raster pipeline");
  const bw = 2.55, bh = 1.15, gap = 0.53;
  const xs = [0.6, 0.6 + bw + gap, 0.6 + 2 * (bw + gap), 0.6 + 3 * (bw + gap)];
  const y1 = 1.7, y2 = 4.35;
  const steps1 = [
    ["1 · Select building points", "class 6 from all 19 tiles · 8.6 M pts"],
    ["2 · Rasterize", "0.5 m grid · cell counts if ≥ 2 pts"],
    ["3 · Connected components", "8-connectivity → candidate blocks"],
    ["4 · Erosion → cores", "shrink 1.5 m · keep cores ≥ 6 m²"],
  ];
  const steps2 = [
    ["5 · Marker watershed", "flood from cores · cut at height valleys"],
    ["6 · Fragment absorption", "piece < 20 m² joins anchor within 3 m"],
    ["7 · Noise filter (Gate A)", "DTM · height above ground · shape rules"],
  ];
  steps1.forEach((t, i) => stepBox(s, xs[i], y1, bw, bh, t[0], t[1]));
  for (let i = 0; i < 3; i++) arrowH(s, xs[i] + bw + 0.06, xs[i + 1] - 0.06, y1 + bh / 2);
  arrowV(s, xs[3] + bw / 2, y1 + bh + 0.06, y2 - 0.06);
  steps2.forEach((t, i) => stepBox(s, xs[3 - i], y2, bw, bh, t[0], t[1]));
  stepBox(s, xs[0], y2, bw, bh, "2,283 building instances", "IDs written back to every point", { accent: true });
  for (let i = 0; i < 3; i++) arrowH(s, xs[3 - i] - 0.06, xs[2 - i] + bw + 0.06, y2 + bh / 2);
  s.addText("Each step is illustrated next on a 25 × 18 m synthetic scene built with the same operations and parameters.", {
    x: 0.6, y: 6.2, w: W - 1.2, h: 0.5, isTextBox: true, margin: 0,
    fontFace: FONT, fontSize: 13.5, italic: true, color: MUTED, align: "left",
  });
  s.addNotes("The whole method runs on a half-meter 2-D grid: find candidate blocks, split blocks that contain several buildings, absorb over-segmented fragments, and filter out non-buildings. The next slides walk through the steps on a small synthetic scene that uses exactly the same operations and parameters as the real pipeline.");
}

// ---------------- 4: toy scene ----------------
{
  const s = slideBase();
  title(s, "The toy scene");
  const r = centerImage(s, IMG.f1, 12.2, 5.0, 1.35);
  factLine(s, "Two houses joined by a mislabeled tree bridge, a porch fragment, a shed, a ground-level parking lot, stray points.", r.yBottom + 0.25);
  s.addNotes("Every failure mode from the real data has a representative here: house A and house B are connected by a strip of mislabeled tree points; a porch fragment sits 1 meter from A; the shed is a small but real building; the parking lot is labeled building but lies on the ground; five stray points and a sparse eave band round it out. The algorithm sees only 'building' points — the colors are ground truth it cannot access.");
}

// ---------------- 5: rasterize ----------------
{
  const s = slideBase();
  title(s, "Step 2 · Rasterize: 0.5 m grid, a cell needs ≥ 2 points");
  const r = centerImage(s, IMG.f2, 12.2, 5.0, 1.35);
  factLine(s, "Real data: 8.6 M points collapse into one occupancy image.", r.yBottom + 0.25);
  s.addNotes("Each point falls into a half-meter cell. A cell counts as occupied only with at least two points — that single rule kills all five stray points, so a lone misclassified point can never bridge two houses. On the real data this turns 8.6 million points into a manageable image.");
}

// ---------------- 6: connected components ----------------
{
  const s = slideBase();
  title(s, "Step 3 · Connected components");
  const r = centerImage(s, IMG.f3, 12.2, 5.1, 1.3);
  factLine(s, "Real data: thousands of blocks — some hide several buildings behind a tree bridge.", r.yBottom + 0.25);
  s.addNotes("Standard 8-connected labeling on the occupancy image. Most blocks are exactly one building, but whenever anything sits between two houses — here the tree bridge — they are welded into one block. Detecting and splitting those blocks is the job of the next two steps.");
}

// ---------------- 7: why no closing ----------------
{
  const s = slideBase();
  title(s, "Why not morphological closing (the v1 lesson)");
  const r = centerImage(s, IMG.f4, 12.2, 4.6, 1.5);
  factLine(s, "v1 used closing and welded whole rows of houses — v2 fills holes per block instead.", r.yBottom + 0.3);
  s.addNotes("Version 1 closed the whole occupancy image to fill roof holes — and in doing so it also welded the 1.5 to 2.5 meter side-yard gaps between houses; entire rows merged into single instances. That is what we caught in CloudCompare. Version 2 fills holes per block instead: holes get filled, gaps stay open.");
}

// ---------------- 8: erosion cores ----------------
{
  const s = slideBase();
  title(s, "Step 4 · Erosion: how many buildings hide in a block?");
  const r = centerImage(s, IMG.f5, 12.2, 5.1, 1.3);
  factLine(s, "Real data: 168 blocks kept ≥ 2 cores — flagged for splitting.", r.yBottom + 0.25);
  s.addNotes("Shrink every block inward by 1.5 meters. A normal house keeps a solid core; a connection thinner than about 3 meters — the tree bridge — is destroyed. If two or more cores of at least 6 square meters survive, the block must contain several buildings. On the real data 168 blocks were flagged this way.");
}

// ---------------- 9: watershed ----------------
{
  const s = slideBase();
  title(s, "Step 5 · Marker watershed: cut along the height valley");
  const r = centerImage(s, IMG.f6, 12.2, 4.6, 1.5);
  factLine(s, "Real data: 168 merged blocks split into 383 buildings.", r.yBottom + 0.3);
  s.addNotes("Where exactly to cut? Roofs are high; whatever connects them — tree branches, shrubs — is low. Flooding the per-cell max-height surface from the cores places the boundary in the height valley, right on the bridge. The profile on the right verifies it: the dam falls exactly in the low corridor between the two roofs. 168 real blocks became 383 buildings.");
}

// ---------------- 10: absorption + final ----------------
{
  const s = slideBase();
  title(s, "Step 6 · Fragment absorption, then final IDs");
  const r = centerImage(s, IMG.f7, 12.2, 5.0, 1.3);
  factLine(s, "Real data: 2,296 instances after this step.", r.yBottom + 0.25);
  s.addNotes("A piece smaller than 20 square meters joins the nearest instance of at least 50 square meters if it lies within 3 meters and the median heights differ by at most 3 meters — asymmetric, so two real houses can never merge, but the porch comes home. Tiny leftovers are dropped as noise, survivors get final IDs, and sparse orphan points are re-attached to the nearest instance. The real data ends this step with 2,296 instances.");
}

// ---------------- 11: Gate A ----------------
{
  const s = slideBase();
  title(s, "Step 7 · Gate A: height above ground kills fake buildings");
  const r = centerImage(s, IMG.f8, 12.2, 5.0, 1.3);
  factLine(s, "Real data: shape rules + manual review removed 13 (bridges, ramps, tree rows, power lines) → 2,283 buildings.", r.yBottom + 0.25);
  s.addNotes("A 1-meter DTM built from ground-class points gives every instance a height above ground; the parking lot sits at 8 centimeters and is removed. On the real data the survivors of this rule set were shortlisted by shape rules — solidity and inscribed width — and a manual review removed 13 non-buildings: bridge decks, highway ramps, tree rows, power lines. Every removal is logged with a reason code.");
}

// ---------------- 12: results ----------------
{
  const s = slideBase();
  title(s, "Result: 2,283 clean building instances");
  const stats = [
    ["2,296", "raw instances"],
    ["168 → 383", "merged blocks split"],
    ["13", "non-buildings removed"],
  ];
  let sy = 1.5;
  for (const [num, lab] of stats) {
    s.addText(num, {
      x: M, y: sy, w: 3.0, h: 0.62, isTextBox: true, margin: 0,
      fontFace: FONT, fontSize: 30, bold: true, color: TEXT, align: "left",
    });
    s.addText(lab, {
      x: M, y: sy + 0.6, w: 3.1, h: 0.4, isTextBox: true, margin: 0,
      fontFace: FONT, fontSize: 12.5, color: MUTED, align: "left", valign: "top",
    });
    sy += 1.25;
  }
  s.addText("2,283", {
    x: M, y: sy + 0.1, w: 3.0, h: 0.75, isTextBox: true, margin: 0,
    fontFace: FONT, fontSize: 40, bold: true, color: ACCENT, align: "left",
  });
  s.addText("clean building instances", {
    x: M, y: sy + 0.85, w: 3.1, h: 0.4, isTextBox: true, margin: 0,
    fontFace: FONT, fontSize: 13.5, bold: true, color: ACCENT, align: "left",
  });
  const oW = 4.9, oH = oW / IMG.instOverview.ar; // ~3.85
  const zH = oH, zW = zH * IMG.instZoom.ar;
  const ix = 3.85, iy = 1.7;
  s.addImage({ path: IMG.instOverview.path, x: ix, y: iy, w: oW, h: oH });
  s.addImage({ path: IMG.instZoom.path, x: ix + oW + 0.25, y: iy, w: zW, h: zH });
  s.addText("Same 400 × 400 m window as slide 2 — the tree-bridged houses are now separate. Collision-free palette: neighbors never share a color (verified in CloudCompare).", {
    x: ix, y: iy + oH + 0.2, w: 8.9, h: 0.75, isTextBox: true, margin: 0,
    fontFace: FONT, fontSize: 13, italic: true, color: MUTED, align: "left",
  });
  s.addNotes("2,296 instances come out of the raster pipeline; Gate A removes 13 non-buildings, leaving 2,283. On the right, the same window as the semantic view at the start — the tree-bridged houses are separated. I verified the result building by building in CloudCompare; the collision-free coloring makes any merge error visible at a glance.");
}

// ---------------- 13: next steps ----------------
{
  const s = slideBase();
  title(s, "Next: from instances to LOD2 building models");
  const stages = [
    ["Semantic segmentation", "done", false],
    ["Instance segmentation", "this report", true],
    ["Roof plane extraction", "region growing", false],
    ["Roof structure lines", "plane intersections", false],
    ["LOD2 assembly + evaluation", "CityJSON · county footprints", false],
  ];
  const bw = 2.2, bh = 1.1, gap = 0.32;
  let bx = 0.55;
  const by = 3.0;
  stages.forEach(([name, tag, isAccent], i) => {
    stepBox(s, bx, by, bw, bh, name, tag, { accent: isAccent });
    if (i < stages.length - 1) arrowH(s, bx + bw + 0.05, bx + bw + gap - 0.05, by + bh / 2);
    bx += bw + gap;
  });
  s.addNotes("The clean instances are the unit of work for everything downstream: planar roof faces per building via region growing, ridge, valley and step edges from plane intersections, then watertight LOD2 assembly with CityJSON export, and a quantitative evaluation against Franklin County footprints plus manually annotated reference buildings.");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("written:", OUT));
