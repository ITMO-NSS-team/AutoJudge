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
    // FastAPI reports the cause in "detail"; show it instead of the envelope.
    let message = text;
    try {
      const body = JSON.parse(text);
      const detail = body?.detail;
      if (typeof detail === "string" && detail.trim()) message = detail;
      else if (Array.isArray(detail) && detail.length)
        message = detail
          .map((item) => (typeof item?.msg === "string" ? item.msg : JSON.stringify(item)))
          .join("; ");
    } catch {
      /* not JSON — show the raw body */
    }
    throw Error(message || `Request failed with status ${res.status}`);
  }
  if (!text.trim()) return undefined as T;
  return JSON.parse(text) as T;
}
