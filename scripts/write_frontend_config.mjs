import { copyFileSync, cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";

const apiBase = String(process.env.GRIDSENSE_API_BASE || "").replace(/\/$/, "");
const content = `window.GS_CONFIG = window.GS_CONFIG || { API_BASE: ${JSON.stringify(apiBase)} };`;

writeFileSync("static/config.js", `${content}\n`, "utf8");

const outDir = "public";
if (existsSync(outDir)) {
  rmSync(outDir, { recursive: true, force: true });
}
mkdirSync(outDir, { recursive: true });
cpSync("static", `${outDir}/static`, { recursive: true });
copyFileSync("static/index.html", `${outDir}/index.html`);

console.log(`Wrote static/config.js with API base: ${apiBase || "<same-origin>"}`);
console.log(`Prepared ${outDir}/static for Vercel output`);
