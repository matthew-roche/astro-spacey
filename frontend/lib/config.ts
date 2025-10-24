// cache backend url from s3
let cached: { BACKEND_BASE: string } | null = null;
let lastFetched = 0;
const TTL = 1000 * 60 * 5; // 5 minutes refresh

export async function getConfig(force = false) {
  const CONFIG_URL = process.env.CONFIG_URL!;
  const now = Date.now();
  if (!cached || force || (now - lastFetched) > TTL) {
    const r = await fetch(CONFIG_URL, { cache: "no-store" });
    if (!r.ok) throw new Error(`config fetch failed: ${r.status}`);
    cached = await r.json();
    lastFetched = now;
  }
  return cached!;
}