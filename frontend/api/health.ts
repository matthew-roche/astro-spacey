// api/health.ts
import { getConfig } from "../lib/config.js";

export default async function handler(req: any, res: any) {
    try {
        // const CONFIG_URL = process.env.CONFIG_URL!;
        // const { BACKEND_BASE } = await fetch(CONFIG_URL, { cache: "no-store" }).then(r => r.json());
        // const base = String(BACKEND_BASE || "").replace(/\/$/, "");

        const { BACKEND_BASE } = await getConfig(true); // force refresh
        const base = BACKEND_BASE.replace(/\/$/, "");

        const upstream = base + "/health"; // server health

        const r = await fetch(upstream);
        const text = await r.text();

        res.status(r.status);
        r.headers.forEach((v, k) => {
            if (k.toLowerCase() !== "content-length") res.setHeader(k, v);
        });
        res.send(text);
    } catch (e: any) {
        console.error("[Health] Error:", e);
        res.status(500).json({ error: "health_check_failed", message: String(e?.message || e) });
    }
}
