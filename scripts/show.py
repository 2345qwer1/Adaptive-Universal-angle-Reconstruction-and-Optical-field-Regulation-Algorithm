from __future__ import annotations

"""
show.py

用途：
    将 dose profile CSV 转换为一个可交互 HTML 页面。
    HTML 中可以缩放、拖拽、悬停或点击曲线点，用于检查 dose distribution 细节。

输入：
    CSV 需要包含 sample_index、distance_px、row、col、dose、direction 等字段。
    这类 CSV 通常由 resolution-w study 的 profile 导出逻辑生成。

输出：
    与 CSV 同名的 HTML 文件，或命令行指定的输出 HTML 路径。
"""

import csv
import json
import sys
from pathlib import Path


# =========================
# 用户输入参数区
# =========================
# 默认 profile CSV 路径；命令行传入 CSV 时会覆盖该值。
DEFAULT_CSV_PATH = Path(
    r"F:\USTC\项目\VAM\5.9-3\resolution_w_study_2_0\m_iterations\optimized_m_049_plus_x_profile.csv"
)

# 默认 y 轴显示范围；命令行第 3、4 个参数可覆盖。
DEFAULT_Y_MIN = 0.0
DEFAULT_Y_MAX = 30.0


def read_profile_csv(csv_path: Path) -> list[dict[str, float | str]]:
    """
    读取 profile CSV 并执行字段校验。

    返回值中的数值字段会被转成 int/float，便于后续直接写入 JSON payload。
    """
    rows: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_index", "distance_px", "row", "col", "dose", "direction"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for raw in reader:
            rows.append(
                {
                    "sample_index": int(raw["sample_index"]),
                    "distance_px": float(raw["distance_px"]),
                    "row": float(raw["row"]),
                    "col": float(raw["col"]),
                    "dose": float(raw["dose"]),
                    "direction": str(raw["direction"]),
                }
            )

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows


def build_html(rows: list[dict[str, float | str]], csv_path: Path, y_min: float, y_max: float) -> str:
    """
    根据 profile 数据生成完整 HTML 字符串。

    生成的 HTML 不依赖外部 JS/CSS 文件，便于直接复制或在浏览器中打开。
    """
    x_values = [float(row["distance_px"]) for row in rows]
    dose_values = [float(row["dose"]) for row in rows]
    x_min = min(x_values)
    x_max = max(x_values)
    direction = str(rows[0]["direction"])

    payload = {
        "source": str(csv_path),
        "direction": direction,
        "xMin": x_min,
        "xMax": x_max,
        "yMin": float(y_min),
        "yMax": float(y_max),
        "rows": rows,
    }

    template = (
        r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #687078;
      --grid: #d9dee5;
      --accent: #0b5fff;
      --bg: #ffffff;
      --panel: #f6f8fb;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 14px 18px 8px;
      border-bottom: 1px solid #e5e8ef;
      background: var(--panel);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 18px;
      font-weight: 650;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
    }
    .controls button {
      min-width: 34px;
      height: 28px;
      border: 1px solid #cbd2dc;
      background: #fff;
      color: var(--ink);
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    .controls button:hover {
      background: #edf2ff;
      border-color: #9ab2ff;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
    }
    #wrap {
      height: calc(100vh - 116px);
      min-height: 460px;
      padding: 12px 18px 18px;
      box-sizing: border-box;
    }
    svg {
      width: 100%;
      height: 100%;
      display: block;
      background: #fff;
      border: 1px solid #dfe3ea;
      cursor: grab;
      touch-action: none;
      user-select: none;
    }
    svg.dragging {
      cursor: grabbing;
    }
    .grid {
      stroke: var(--grid);
      stroke-width: 1;
      vector-effect: non-scaling-stroke;
    }
    .axis {
      stroke: #333;
      stroke-width: 1.2;
      vector-effect: non-scaling-stroke;
    }
    .tick text {
      fill: var(--muted);
      font-size: 12px;
      user-select: none;
    }
    .line {
      fill: none;
      stroke: var(--accent);
      stroke-width: 1.4;
      vector-effect: non-scaling-stroke;
    }
    .point {
      fill: #fff;
      stroke: var(--accent);
      stroke-width: 1.2;
      vector-effect: non-scaling-stroke;
      cursor: pointer;
    }
    .point:hover, .point.active {
      fill: #ffb000;
      stroke: #111;
    }
    #readout {
      position: fixed;
      right: 20px;
      top: 86px;
      min-width: 480px;
      min-height: 170px;
      padding: 20px 24px;
      border: 1px solid #d8dde6;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 8px 26px rgba(0, 0, 0, 0.10);
      font-size: 26px;
      line-height: 1.55;
      pointer-events: none;
    }
    #readout strong {
      display: block;
      margin-bottom: 4px;
      font-size: 28px;
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="meta">
      <span>source: __SOURCE__</span>
      <span>direction: __DIRECTION__</span>
      <span>x: __X_RANGE__ px</span>
      <span>y: __Y_RANGE__ dose</span>
    </div>
    <div class="controls">
      <button id="zoomIn" type="button">+</button>
      <button id="zoomOut" type="button">-</button>
      <button id="resetZoom" type="button">Reset</button>
      <span class="hint">鼠标滚轮缩放，拖拽平移，双击复位；点击曲线点查看数值。</span>
    </div>
  </header>
  <div id="wrap">
    <svg id="chart" viewBox="0 0 1200 700" role="img" aria-label="dose profile chart"></svg>
  </div>
  <div id="readout"><strong>点击或悬停曲线点</strong>distance_px / dose / row / col 会显示在这里。</div>
  <script>
    const payload = __PAYLOAD__;
    const svg = document.getElementById("chart");
    const readout = document.getElementById("readout");
    const zoomInButton = document.getElementById("zoomIn");
    const zoomOutButton = document.getElementById("zoomOut");
    const resetZoomButton = document.getElementById("resetZoom");
    const W = 1200;
    const H = 700;
    const margin = { left: 78, right: 28, top: 28, bottom: 58 };
    const plotW = W - margin.left - margin.right;
    const plotH = H - margin.top - margin.bottom;

    function sx(x) {
      return margin.left + (x - payload.xMin) / (payload.xMax - payload.xMin || 1) * plotW;
    }
    function sy(y) {
      return margin.top + (payload.yMax - y) / (payload.yMax - payload.yMin || 1) * plotH;
    }
    function el(name, attrs = {}) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      return node;
    }
    function fmt(value, digits = 6) {
      const n = Number(value);
      if (!Number.isFinite(n)) return String(value);
      return n.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
    }
    function setReadout(row) {
      readout.innerHTML = `
        <strong>sample #${row.sample_index}</strong>
        distance_px: ${fmt(row.distance_px, 3)}<br>
        dose: ${fmt(row.dose, 8)}<br>
        row: ${fmt(row.row, 3)}<br>
        col: ${fmt(row.col, 3)}<br>
        direction: ${row.direction}
      `;
    }
    const initialViewBox = { x: 0, y: 0, w: W, h: H };
    let viewBox = { ...initialViewBox };
    let isDragging = false;
    let dragStart = null;

    function applyViewBox() {
      svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
    }
    function clientToSvgPoint(clientX, clientY) {
      const pt = svg.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      return pt.matrixTransform(svg.getScreenCTM().inverse());
    }
    function zoomAt(svgX, svgY, factor) {
      const minW = W / 80;
      const maxW = W * 6;
      const nextW = Math.max(minW, Math.min(maxW, viewBox.w * factor));
      const nextH = nextW * H / W;
      const rx = (svgX - viewBox.x) / viewBox.w;
      const ry = (svgY - viewBox.y) / viewBox.h;
      viewBox = {
        x: svgX - rx * nextW,
        y: svgY - ry * nextH,
        w: nextW,
        h: nextH,
      };
      applyViewBox();
    }
    function zoomAtCenter(factor) {
      zoomAt(viewBox.x + viewBox.w / 2, viewBox.y + viewBox.h / 2, factor);
    }
    function resetView() {
      viewBox = { ...initialViewBox };
      applyViewBox();
    }

    const bg = el("rect", { x: 0, y: 0, width: W, height: H, fill: "#fff" });
    svg.appendChild(bg);
    applyViewBox();

    const xTicks = 10;
    for (let i = 0; i <= xTicks; i++) {
      const value = payload.xMin + (payload.xMax - payload.xMin) * i / xTicks;
      const x = sx(value);
      svg.appendChild(el("line", { class: "grid", x1: x, y1: margin.top, x2: x, y2: margin.top + plotH }));
      const tick = el("g", { class: "tick" });
      tick.appendChild(el("line", { class: "axis", x1: x, y1: margin.top + plotH, x2: x, y2: margin.top + plotH + 5 }));
      const text = el("text", { x, y: margin.top + plotH + 22, "text-anchor": "middle" });
      text.textContent = fmt(value, 0);
      tick.appendChild(text);
      svg.appendChild(tick);
    }

    const yTicks = 6;
    for (let i = 0; i <= yTicks; i++) {
      const value = payload.yMin + (payload.yMax - payload.yMin) * i / yTicks;
      const y = sy(value);
      svg.appendChild(el("line", { class: "grid", x1: margin.left, y1: y, x2: margin.left + plotW, y2: y }));
      const tick = el("g", { class: "tick" });
      tick.appendChild(el("line", { class: "axis", x1: margin.left - 5, y1: y, x2: margin.left, y2: y }));
      const text = el("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end" });
      text.textContent = fmt(value, 1);
      tick.appendChild(text);
      svg.appendChild(tick);
    }

    svg.appendChild(el("line", { class: "axis", x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH }));
    svg.appendChild(el("line", { class: "axis", x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH }));

    const xLabel = el("text", { x: margin.left + plotW / 2, y: H - 16, "text-anchor": "middle", fill: "#202124", "font-size": 14 });
    xLabel.textContent = "distance / px";
    svg.appendChild(xLabel);
    const yLabel = el("text", { x: 18, y: margin.top + plotH / 2, transform: `rotate(-90 18 ${margin.top + plotH / 2})`, "text-anchor": "middle", fill: "#202124", "font-size": 14 });
    yLabel.textContent = "dose";
    svg.appendChild(yLabel);

    const d = payload.rows.map((row, i) => `${i === 0 ? "M" : "L"} ${sx(row.distance_px)} ${sy(row.dose)}`).join(" ");
    svg.appendChild(el("path", { class: "line", d }));

    const points = el("g");
    payload.rows.forEach((row) => {
      const point = el("circle", {
        class: "point",
        cx: sx(row.distance_px),
        cy: sy(row.dose),
        r: 3.2,
        tabindex: 0,
      });
      point.addEventListener("mouseenter", () => setReadout(row));
      point.addEventListener("click", () => {
        document.querySelectorAll(".point.active").forEach((node) => node.classList.remove("active"));
        point.classList.add("active");
        setReadout(row);
      });
      points.appendChild(point);
    });
    svg.appendChild(points);

    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const pt = clientToSvgPoint(event.clientX, event.clientY);
      zoomAt(pt.x, pt.y, event.deltaY < 0 ? 0.82 : 1.22);
    }, { passive: false });
    svg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.classList.contains("point")) return;
      isDragging = true;
      svg.classList.add("dragging");
      svg.setPointerCapture(event.pointerId);
      dragStart = {
        x: event.clientX,
        y: event.clientY,
        viewX: viewBox.x,
        viewY: viewBox.y,
      };
    });
    svg.addEventListener("pointermove", (event) => {
      if (!isDragging || dragStart === null) return;
      const dx = (event.clientX - dragStart.x) * viewBox.w / svg.clientWidth;
      const dy = (event.clientY - dragStart.y) * viewBox.h / svg.clientHeight;
      viewBox.x = dragStart.viewX - dx;
      viewBox.y = dragStart.viewY - dy;
      applyViewBox();
    });
    svg.addEventListener("pointerup", (event) => {
      if (!isDragging) return;
      isDragging = false;
      dragStart = null;
      svg.classList.remove("dragging");
      svg.releasePointerCapture(event.pointerId);
    });
    svg.addEventListener("pointercancel", () => {
      isDragging = false;
      dragStart = null;
      svg.classList.remove("dragging");
    });
    svg.addEventListener("dblclick", resetView);
    zoomInButton.addEventListener("click", () => zoomAtCenter(0.82));
    zoomOutButton.addEventListener("click", () => zoomAtCenter(1.22));
    resetZoomButton.addEventListener("click", resetView);

    setReadout(payload.rows[0]);
  </script>
</body>
</html>
"""
    )

    title = f"{csv_path.stem} dose profile"
    return (
        template.replace("__TITLE__", title)
        .replace("__SOURCE__", str(csv_path))
        .replace("__DIRECTION__", direction)
        .replace("__X_RANGE__", f"{x_min:g} .. {x_max:g}")
        .replace("__Y_RANGE__", f"{float(y_min):g} .. {float(y_max):g}")
        .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    )


def parse_args(argv: list[str]) -> tuple[Path, Path, float, float]:
    """
    解析命令行参数。

    用法：
        python scripts/show.py profile.csv [output.html] [y_min] [y_max]
    """
    csv_path = Path(argv[1]) if len(argv) >= 2 else DEFAULT_CSV_PATH
    output_path = Path(argv[2]) if len(argv) >= 3 else csv_path.with_suffix(".html")
    y_min = float(argv[3]) if len(argv) >= 4 else DEFAULT_Y_MIN
    y_max = float(argv[4]) if len(argv) >= 5 else DEFAULT_Y_MAX
    return csv_path, output_path, y_min, y_max


def main(argv: list[str]) -> int:
    """主流程：读取 CSV，生成 HTML，并写入指定输出路径。"""
    csv_path, output_path, y_min, y_max = parse_args(argv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print(
            'Usage: python "CAL 1.2\\scripts\\show.py" path\\to\\profile.csv [output.html] [y_min] [y_max]',
            file=sys.stderr,
        )
        return 2

    rows = read_profile_csv(csv_path)
    html = build_html(rows, csv_path, y_min, y_max)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
