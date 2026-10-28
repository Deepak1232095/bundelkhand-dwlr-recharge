const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.defineLayout({ name: "WRR", width: 7.5, height: 5.6 });   // 19 cm = AGU full page width
pres.layout = "WRR";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.title = "Methods workflow";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };
const ST = {
  input: { fill: "FFFFFF", line: "404040", dash: "solid", tc: "000000" },
  main:  { fill: "E6EEF5", line: "2B5C8A", dash: "solid", tc: "000000" },
  key:   { fill: "2B5C8A", line: "2B5C8A", dash: "solid", tc: "FFFFFF" },
  verif: { fill: "F2F2F2", line: "7F7F7F", dash: "solid", tc: "000000" },
  cons:  { fill: "FFFFFF", line: "A0522D", dash: "dash",  tc: "000000" },
};
let n = 0;
function box(x, y, w, h, t, st) {
  const S = ST[st];
  s.addText(t, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05, x, y, w, h,
    fill: { color: S.fill }, line: { color: S.line, width: st === "key" ? 1.5 : 1, dashType: S.dash },
    fontFace: "Arial", fontSize: 9, bold: true, color: S.tc, align: "center", valign: "middle",
    margin: [2, 4, 2, 4], objectName: "box" + (++n) });
}
function line(x1, y1, x2, y2, o = {}) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: o.color || "404040", width: 1, dashType: o.dash || "solid", endArrowType: o.arrow ? "triangle" : undefined },
    objectName: "arrow" + (++n) });
}
function label(x, y, w, h, t, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: "Arial", fontSize: o.size || 8.5, bold: !!o.bold, italic: !!o.italic,
    color: o.color || "404040", margin: 0, isTextBox: true, align: o.align || "left", valign: "middle" });
}
const DC = "A0522D", GC = "7F7F7F";

// inputs
const IW = 1.36, IG = 0.1, IY = 0.15, IH = 0.5, ix = i => 0.15 + i * (IW + IG);
["DWLR water levels", "IMD rainfall", "GEC Sy norms", "CGWB wells", "GEC-2024 district recharge"]
  .forEach((t, i) => box(ix(i), IY, IW, IH, t, "input"));
const inBus = 0.85;
for (let i = 0; i < 5; i++) line(ix(i) + IW / 2, IY + IH, ix(i) + IW / 2, inBus);
line(ix(0) + IW / 2, inBus, ix(4) + IW / 2, inBus);

// main chain
const MW = 2.6, MX = 3.75 - MW / 2, MC = MX + MW / 2, BH = 0.45;
line(MC, inBus, MC, 1.0, { arrow: true });
box(MX, 1.0, MW, BH, "Quality control and smoothing", "main");
line(MC, 1.0 + BH, MC, 1.7, { arrow: true });
box(MX, 1.7, MW, BH, "Master recession curve", "main");
line(MC, 1.7 + BH, MC, 2.4, { arrow: true });
box(MX, 2.4, MW, 0.55, "Monsoon rise per unit Sy (ΔH)", "key");

// verification
const VX = 5.35, VW = 2.0;
box(VX, 1.7, VW, BH, "Synthetic verification", "verif");
box(VX, 2.45, VW, BH, "MODFLOW 6 test", "verif");
line(VX, 1.7 + BH / 2, MX + MW, 1.7 + BH / 2, { arrow: true, color: GC, dash: "sysDash" });
line(VX, 2.675, MX + MW, 2.675, { arrow: true, color: GC, dash: "sysDash" });

// primary analyses
const PY = 3.3, PW = 3.45, B4X = 0.15, B5X = 3.9;
line(MC, 2.95, MC, 3.12);
line(B4X + PW / 2, 3.12, B5X + PW / 2, 3.12);
line(B4X + PW / 2, 3.12, B4X + PW / 2, PY, { arrow: true });
line(B5X + PW / 2, 3.12, B5X + PW / 2, PY, { arrow: true });
box(B4X, PY, PW, BH, "Recharge episodes and rainfall attribution", "main");
box(B5X, PY, PW, BH, "Recharge depth and Sy uncertainty", "main");

// consistency
const CY = 4.4, CW = 1.725, CG = 0.1, cx = i => 0.15 + i * (CW + CG);
const BUSX = (B4X + PW + B5X) / 2;
line(BUSX, 3.12, BUSX, 4.22, { color: DC, dash: "dash" });
line(cx(0) + CW / 2, 4.22, cx(3) + CW / 2, 4.22, { color: DC, dash: "dash" });
for (let i = 0; i < 4; i++) line(cx(i) + CW / 2, 4.22, cx(i) + CW / 2, CY, { arrow: true, color: DC, dash: "dash" });
label(0.15, 3.93, 3.4, 0.2, "Consistency checks and exploratory analyses", { bold: true, color: DC });
["Fixed-date rise", "Effective Sy (GEC)", "Spatial interpolation", "River proximity"]
  .forEach((t, i) => box(cx(i), CY, CW, 0.5, t, "cons"));

// legend
const LY = 5.18;
function key(x, st, t) {
  const S = ST[st];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: LY, w: 0.32, h: 0.2, rectRadius: 0.03, fill: { color: S.fill },
    line: { color: S.line, width: 1, dashType: S.dash }, objectName: "legend_" + st });
  label(x + 0.4, LY, 1.6, 0.2, t, { size: 8 });
}
key(0.15, "main", "Primary analysis");
key(2.05, "verif", "Verification");
key(3.75, "cons", "Consistency / exploratory");
key(6.0, "input", "Data input");
pres.writeFile({ fileName: "Fig2_workflow.pptx" }).then(f => console.log(f));
