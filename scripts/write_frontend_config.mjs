import { copyFileSync, cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";

const apiBase = String(process.env.GRIDSENSE_API_BASE || "").replace(/\/$/, "");
const content = `window.GS_CONFIG = window.GS_CONFIG || { API_BASE: ${JSON.stringify(apiBase)} };`;
const processedDataDir = "data/processed";
const staticDataOutDir = "public/static/data";
const staticFallbackFiles = [
  "metrics.json",
  "forecasts.json",
  "zones.json",
  "anomalies.json",
  "anomaly_evidence.json",
  "pipeline_summary.json",
  "theft_validation.json",
  "inspection_feedback.json",
  "pipeline_status.json",
];

writeFileSync("static/config.js", `${content}\n`, "utf8");

const outDir = "public";
if (existsSync(outDir)) {
  rmSync(outDir, { recursive: true, force: true });
}
mkdirSync(outDir, { recursive: true });
cpSync("static", `${outDir}/static`, { recursive: true });
copyFileSync("static/index.html", `${outDir}/index.html`);
mkdirSync(staticDataOutDir, { recursive: true });

for (const filename of staticFallbackFiles) {
  const source = `${processedDataDir}/${filename}`;
  const target = `${staticDataOutDir}/${filename}`;
  if (existsSync(source)) {
    copyFileSync(source, target);
  }
}

console.log(`Wrote static/config.js with API base: ${apiBase || "<same-origin>"}`);
console.log(`Prepared ${outDir}/static for Vercel output`);
