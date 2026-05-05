import { writeFileSync } from "node:fs";

const apiBase = String(process.env.GRIDSENSE_API_BASE || "").replace(/\/$/, "");
const content = `window.GS_CONFIG = window.GS_CONFIG || { API_BASE: ${JSON.stringify(apiBase)} };`;

writeFileSync("static/config.js", `${content}\n`, "utf8");
console.log(`Wrote static/config.js with API base: ${apiBase || "<same-origin>"}`);
