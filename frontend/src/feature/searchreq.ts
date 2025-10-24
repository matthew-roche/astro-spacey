export interface SearchResponse {
  response: { id: string}[];
  ai_answered: number;
  ai_text?: string;
  ctx_id?: { id: string}[];
}

export interface Record {
  id: string;
  title: string;
  content: string;
}
 
export async function search(url:string, question: string, mode: boolean, api_sec: string): Promise<SearchResponse> {

  const params = new URLSearchParams({
    question,
    ai_mode: String(mode),
  });

  const res = await fetch(`${url}/search?${params.toString()}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<SearchResponse>;
}