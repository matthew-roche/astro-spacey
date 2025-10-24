import { tableFromIPC } from "apache-arrow";

export type Row = { id: string; title: string; content: string };

export async function loadIndex(url: string, api_sec: string): Promise<Map<string, Row>> {
  const res = await fetch(url, { headers: { 
    Accept: "application/vnd.apache.arrow.stream",
    
  } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  const table = tableFromIPC(buf);

  const idVec      = table.getChild("id");
  const titleVec   = table.getChild("title");
  const contentVec = table.getChild("content");

  if (!idVec || !titleVec || !contentVec) {
    throw new Error("One or more required columns are missing in the Arrow table");
  }

  const index = new Map<string, Row>();
  for (let i = 0; i < table.numRows; i++) {
    const id = String(idVec.get(i)); // normalize to string key
    index.set(id, {
      id,
      title:   titleVec.get(i) ?? "",
      content: contentVec.get(i) ?? "",
    });
  }
  return index;
}
export function rowsByIds(index: Map<string, Row>, ids: Array<string|number>): Row[] {

  // console.log(index)
  // console.log(ids)
  return ids.map(id => index.get(String(id))).filter(Boolean) as Row[];
}
