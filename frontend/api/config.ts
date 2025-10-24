// for s3 backend refresh
import { getConfig } from "../lib/config.js";

export default async function handler(_req: any, res: any) {
    try {
        await getConfig(true); // force refresh on first load
        res.status(204).end();
    } catch (e: any) {
        console.error("[Config] warm error:", e);
        res.status(500).json({ error: "config_warm_failed", message: String(e?.message || e) });
    }
}