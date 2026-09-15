export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const res = await fetch("/api" + path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    throw Error(text || `Request failed with status ${res.status}`);
  }
  if (!text.trim()) return undefined as T;
  return JSON.parse(text) as T;
}
