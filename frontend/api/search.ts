// api/search.ts
import { Readable } from "node:stream";
import { getConfig } from "../lib/config.js";

export default async function handler(req: any, res: any) {
    try {
        //refactor
        //const CONFIG_URL = process.env.CONFIG_URL!;
        //const SECRET = process.env.VITE_API_SEC!;
        // Load backend base
        // const cfgResp = await fetch(CONFIG_URL, { cache: "no-store" });
        // if (!cfgResp.ok) {
        //     res.status(500).json({ error: "config_fetch_failed", status: cfgResp.status });
        //     return;
        // }
        // const { BACKEND_BASE } = await cfgResp.json();
        // const base = String(BACKEND_BASE || "").replace(/\/$/, "");

        const { BACKEND_BASE } = await getConfig(); // from cache
        const base = BACKEND_BASE.replace(/\/$/, "");
        const SECRET = process.env.VITE_API_SEC!;

        // reconstruct
        const incoming = new URL(req.url, "http://dummy");
        incoming.searchParams.delete("...path");
        incoming.searchParams.delete("path");
        const upstream = base + incoming.pathname + incoming.search;

        // Headers: add secret, request identity
        const headers: Record<string, string> = {
            "X-Demo-Secret": SECRET,
            "Accept-Encoding": "identity",
        };
        const ct = req.headers["content-type"];
        if (typeof ct === "string" && ct) headers["Content-Type"] = ct;

        const isBodyless = ["GET", "HEAD"].includes((req.method || "GET").toUpperCase());
        const r = await fetch(upstream, { method: req.method, headers, body: isBodyless ? undefined : req });

        // Pass status; copy headers except encoding/length (we're re-streaming)
        res.status(r.status);
        r.headers.forEach((v, k) => {
            const kl = k.toLowerCase();
            if (kl === "content-length" || kl === "content-encoding" || kl === "transfer-encoding") return;
            res.setHeader(k, v);
        });

        // Stream JSON (or whatever) as-is
        if (r.body) {
            const fromWeb = (Readable as any).fromWeb;
            if (typeof fromWeb === "function") fromWeb(r.body as any).pipe(res);
            else res.end(Buffer.from(await r.arrayBuffer()));
        } else {
            res.end();
        }
    } catch (e) {
        console.error("[search] proxy error:", e);
        res.status(500).json({ error: "proxy_error", message: String(e) });
    }
}
