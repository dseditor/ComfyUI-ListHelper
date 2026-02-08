import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Model Downloader progress bar extension for ComfyUI-ListHelper
 *
 * Design: The status panel is drawn BELOW the node body as an extension plate.
 * This prevents widgets (especially multiline text areas) from expanding into
 * and covering the panel area. The node's visual bounds include both the
 * standard node body and the panel extension below it.
 */

const PANEL_HEIGHT = 66;
const CORNER_RADIUS = 8;

// Border color per status
const BORDER_COLORS = {
	idle:        "#555570",
	checking:    "#c0a030",
	downloading: "#2979FF",
	complete:    "#43A047",
	error:       "#E53935",
};

app.registerExtension({
	name: "listhelper.ModelDownloader",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name !== "ModelDownloader") return;

		// NOTE: We intentionally do NOT override computeSize.
		// Adding PANEL_HEIGHT to computeSize causes widgets (especially
		// multiline STRING inputs) to auto-expand into the panel area,
		// completely covering the progress display.
		// Instead, the panel is drawn outside the node body in onDrawForeground.

		// ── onDrawForeground: draw extension panel below the node body ──
		const origOnDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx) {
			origOnDrawForeground?.apply(this, arguments);

			// Don't draw panel when node is collapsed
			if (this.flags?.collapsed) return;

			const w = this.size[0];
			const h = this.size[1];
			const d = this._downloadProgress || { status: "idle" };
			const status = d.status || "idle";
			const panelY = h; // Panel starts right at the bottom edge of the node body

			ctx.save();

			// ── 1. Bridge: fill the node's bottom rounded corners ──
			// The node body has rounded bottom corners, creating small gaps.
			// Fill them so the panel connects seamlessly to the node.
			const bgcolor = this.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR || "#353535";
			ctx.fillStyle = bgcolor;
			ctx.fillRect(0, h - CORNER_RADIUS, w, CORNER_RADIUS);

			// ── 2. Panel background plate (extends below the node) ──
			ctx.fillStyle = "rgba(0, 0, 0, 0.22)";
			ctx.beginPath();
			ctx.roundRect(0, panelY, w, PANEL_HEIGHT, [0, 0, CORNER_RADIUS, CORNER_RADIUS]);
			ctx.fill();

			// ── 3. Separator line at the joint ──
			ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(10, panelY + 1);
			ctx.lineTo(w - 10, panelY + 1);
			ctx.stroke();

			// ── 4. Status content ──
			if (status === "idle") {
				drawIdleState(ctx, w, panelY);
			} else if (status === "checking") {
				drawCheckingState(ctx, w, panelY);
			} else if (status === "downloading") {
				drawDownloadingState(ctx, d, w, panelY);
			} else if (status === "complete") {
				drawCompleteState(ctx, d, w, panelY);
			} else if (status === "error") {
				drawErrorState(ctx, d, w, panelY);
			}

			// ── 5. Unified accent border around node + panel ──
			const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;
			const totalH = h + PANEL_HEIGHT + titleH;
			const borderColor = BORDER_COLORS[status] || BORDER_COLORS.idle;
			ctx.strokeStyle = borderColor;
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.roundRect(-0.5, -titleH - 0.5, w + 1, totalH + 1, CORNER_RADIUS);
			ctx.stroke();

			ctx.restore();
		};

		// ── getBounding: extend bounding box to include the panel ──
		// Without this, LiteGraph may clip the panel during render culling.
		const origGetBounding = nodeType.prototype.getBounding;
		nodeType.prototype.getBounding = function (out) {
			const bounds = origGetBounding
				? origGetBounding.apply(this, arguments)
				: LiteGraph.LGraphNode.prototype.getBounding.apply(this, arguments);
			if (bounds && !this.flags?.collapsed) {
				bounds[3] += PANEL_HEIGHT;
			}
			return bounds;
		};

		// ── onNodeCreated: initialise progress state ──
		const origOnNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			origOnNodeCreated?.apply(this, arguments);
			this._downloadProgress = { status: "idle" };
		};

		// ── onConfigure: restore progress state for saved workflows ──
		const origOnConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			origOnConfigure?.apply(this, arguments);
			if (!this._downloadProgress) this._downloadProgress = { status: "idle" };
		};
	},

	// ── WebSocket listener ──
	async setup() {
		api.addEventListener("model_download_progress", (event) => {
			const data = event.detail;
			if (!data || !data.node_id) return;

			const nodeId = String(data.node_id);
			const node = app.graph._nodes_by_id?.[nodeId]
				|| app.graph._nodes?.find(n => String(n.id) === nodeId);
			if (!node) return;

			node._downloadProgress = {
				status:       data.status       || "idle",
				filename:     data.filename     || "",
				progress:     data.progress     || 0,
				downloaded:   data.downloaded   || 0,
				total:        data.total        || 0,
				current_file: data.current_file || 0,
				total_files:  data.total_files  || 0,
				message:      data.message      || "",
				has_downloads: data.has_downloads || false,
			};

			app.graph.setDirtyCanvas(true, false);

			if (data.status === "complete" && data.has_downloads) {
				showDownloadCompleteToast();
			}
		});
	},
});

// ===================== Drawing helpers =====================

function drawIdleState(ctx, w, panelY) {
	ctx.fillStyle = "#888899";
	ctx.font = "12px sans-serif";
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	const cy = panelY + PANEL_HEIGHT / 2;
	ctx.fillText("Enter URLs & folders, or select a", w / 2, cy - 8);
	ctx.fillText("template to start downloading", w / 2, cy + 10);
}

function drawCheckingState(ctx, w, panelY) {
	ctx.fillStyle = "#c0a030";
	ctx.font = "bold 13px sans-serif";
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	ctx.fillText("Checking existing files\u2026", w / 2, panelY + PANEL_HEIGHT / 2);
}

function drawDownloadingState(ctx, d, w, panelY) {
	const px = 12;
	const innerW = w - px * 2;

	// Row 1 — file label
	const fileLabel = d.total_files > 0
		? `[${d.current_file}/${d.total_files}]  ${d.filename || ""}`
		: (d.filename || "Preparing\u2026");
	ctx.fillStyle = "#D0D0D8";
	ctx.font = "12px sans-serif";
	ctx.textAlign = "left";
	ctx.textBaseline = "alphabetic";
	ctx.fillText(truncText(ctx, fileLabel, innerW), px, panelY + 16);

	// Row 2 — progress bar
	const barX = px;
	const barY = panelY + 23;
	const barW = innerW;
	const barH = 16;

	// track
	ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
	ctx.beginPath();
	ctx.roundRect(barX, barY, barW, barH, 3);
	ctx.fill();

	// fill
	const pct = clamp(d.progress || 0, 0, 100);
	if (pct > 0) {
		const fillW = Math.max(barW * pct / 100, 4);
		const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
		grad.addColorStop(0, "#1565C0");
		grad.addColorStop(1, "#42A5F5");
		ctx.fillStyle = grad;
		ctx.beginPath();
		ctx.roundRect(barX, barY, fillW, barH, 3);
		ctx.fill();
	}

	// percentage label
	ctx.fillStyle = "#FFFFFF";
	ctx.font = "bold 10px sans-serif";
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	ctx.fillText(`${pct.toFixed(1)}%`, barX + barW / 2, barY + barH / 2);

	// Row 3 — size
	ctx.textBaseline = "alphabetic";
	ctx.font = "11px sans-serif";
	ctx.textAlign = "right";
	if (d.total > 0) {
		ctx.fillStyle = "#909099";
		ctx.fillText(
			`${formatMB(d.downloaded)} / ${formatMB(d.total)} MB`,
			w - px, panelY + 56
		);
	} else {
		ctx.fillStyle = "#707078";
		ctx.fillText("Fetching file size\u2026", w - px, panelY + 56);
	}
}

function drawCompleteState(ctx, d, w, panelY) {
	const cy = panelY + PANEL_HEIGHT / 2;
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";

	if (d.has_downloads) {
		ctx.fillStyle = "#43A047";
		ctx.font = "bold 13px sans-serif";
		ctx.fillText("Download complete!", w / 2, cy - 9);
		ctx.fillStyle = "#FFD740";
		ctx.font = "12px sans-serif";
		ctx.fillText("Please refresh browser (F5)", w / 2, cy + 10);
	} else {
		ctx.fillStyle = "#43A047";
		ctx.font = "bold 13px sans-serif";
		ctx.fillText("All models verified \u2014 already downloaded", w / 2, cy);
	}
}

function drawErrorState(ctx, d, w, panelY) {
	ctx.fillStyle = "#E53935";
	ctx.font = "bold 12px sans-serif";
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	const msg = d.message
		? "Error: " + d.message
		: "Download failed";
	ctx.fillText(truncText(ctx, msg, w - 24), w / 2, panelY + PANEL_HEIGHT / 2);
}

// ===================== Utilities =====================

function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

function formatMB(bytes) { return (bytes / (1024 * 1024)).toFixed(1); }

function truncText(ctx, text, maxWidth) {
	if (ctx.measureText(text).width <= maxWidth) return text;
	let t = text;
	while (t.length > 0 && ctx.measureText(t + "\u2026").width > maxWidth) {
		t = t.slice(0, -1);
	}
	return t + "\u2026";
}

// ===================== Toast notification =====================

function showDownloadCompleteToast() {
	const existing = document.getElementById("md-download-toast");
	if (existing) existing.remove();

	const toast = document.createElement("div");
	toast.id = "md-download-toast";
	Object.assign(toast.style, {
		position: "fixed", top: "20px", left: "50%",
		transform: "translateX(-50%)",
		background: "linear-gradient(135deg, #1a5e1a, #2d8a2d)",
		color: "white", padding: "16px 28px", borderRadius: "10px",
		fontSize: "15px", fontFamily: "sans-serif", zIndex: "99999",
		boxShadow: "0 6px 24px rgba(0,0,0,0.4)",
		display: "flex", flexDirection: "column", alignItems: "center",
		gap: "10px", maxWidth: "480px",
		animation: "mdSlideDown 0.3s ease-out",
	});

	const titleEl = document.createElement("div");
	titleEl.style.fontWeight = "bold";
	titleEl.textContent = "Download complete!";

	const hintEl = document.createElement("div");
	Object.assign(hintEl.style, { fontSize: "13px", color: "#FFD740" });
	hintEl.textContent = "Please refresh browser (F5) to load new models";

	const btnRow = document.createElement("div");
	Object.assign(btnRow.style, { display: "flex", gap: "10px", marginTop: "4px" });

	const refreshBtn = document.createElement("button");
	refreshBtn.textContent = "Refresh now";
	Object.assign(refreshBtn.style, {
		background: "#FFD740", color: "#1a1a1a", border: "none",
		padding: "6px 18px", borderRadius: "5px", cursor: "pointer",
		fontSize: "13px", fontWeight: "bold",
	});
	refreshBtn.addEventListener("click", () => location.reload());

	const dismissBtn = document.createElement("button");
	dismissBtn.textContent = "Later";
	Object.assign(dismissBtn.style, {
		background: "rgba(255,255,255,0.2)", color: "white",
		border: "1px solid rgba(255,255,255,0.3)",
		padding: "6px 18px", borderRadius: "5px", cursor: "pointer",
		fontSize: "13px",
	});
	dismissBtn.addEventListener("click", () => toast.remove());

	btnRow.appendChild(refreshBtn);
	btnRow.appendChild(dismissBtn);
	toast.appendChild(titleEl);
	toast.appendChild(hintEl);
	toast.appendChild(btnRow);

	ensureToastStyle();
	document.body.appendChild(toast);

	setTimeout(() => {
		if (toast.parentNode) {
			toast.style.transition = "opacity 0.5s";
			toast.style.opacity = "0";
			setTimeout(() => toast.remove(), 500);
		}
	}, 30000);
}

function ensureToastStyle() {
	if (document.getElementById("md-toast-style")) return;
	const s = document.createElement("style");
	s.id = "md-toast-style";
	s.textContent = `@keyframes mdSlideDown {
		from { transform: translateX(-50%) translateY(-100%); opacity: 0; }
		to   { transform: translateX(-50%) translateY(0);     opacity: 1; }
	}`;
	document.head.appendChild(s);
}
