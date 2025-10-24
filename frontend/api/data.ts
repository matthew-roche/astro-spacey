// api/[...path].ts (Node runtime)
import { Readable } from "node:stream";
import { getConfig } from "../lib/config.js";

export default async function handler(req: any, res: any) {
    if (req.method === "OPTIONS") { res.status(200).end(); return; }

    const { BACKEND_BASE } = await getConfig(); // from cache
    const base = BACKEND_BASE.replace(/\/$/, "");
    const SECRET = process.env.VITE_API_SEC!;

    // refactor
    // const CONFIG_URL = process.env.CONFIG_URL!;
    // const cfgResp = await fetch(CONFIG_URL, { cache: "no-store" });
    // if (!cfgResp.ok) {
    //     console.error("[Proxy] Failed to load config.json", cfgResp.status);
    //     res.status(500).json({ error: "failed_to_load_config" });
    //     return;
    // }
    // const { BACKEND_BASE } = await cfgResp.json();
    // const base = String(BACKEND_BASE || "").replace(/\/$/, "");
    const upstream = base + req.url;

    // Build headers (don’t force a content-type if none provided)
    const headers: Record<string, string> = { "X-Demo-Secret": SECRET };
    const ct = req.headers["content-type"];
    if (typeof ct === "string" && ct) headers["Content-Type"] = ct;
    if (req.headers["accept"]) headers["Accept"] = String(req.headers["accept"]);
    if (req.headers["authorization"]) headers["Authorization"] = String(req.headers["authorization"]);

    const isBodyless = ["GET", "HEAD"].includes((req.method || "GET").toUpperCase());
    const upstreamResp = await fetch(upstream, {
        method: req.method,
        headers,
        body: isBodyless ? undefined : req, // stream request body through
    });

    // Pass through status + headers (keep Arrow, etc.)
    res.status(upstreamResp.status);
    upstreamResp.headers.forEach((v, k) => {
        if (k.toLowerCase() === "content-length") return;
        res.setHeader(k, v);
    });

    // Stream body back (works for application/vnd.apache.arrow.stream)
    if (upstreamResp.body) {
        const fromWeb = (Readable as any).fromWeb;
        if (typeof fromWeb === "function") {
            fromWeb(upstreamResp.body as any).pipe(res);
        } else {
            const buf = Buffer.from(await upstreamResp.arrayBuffer());
            res.end(buf);
        }
    } else {
        res.end();
    }
}
