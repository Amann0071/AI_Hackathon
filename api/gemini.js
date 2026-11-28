const ALLOWED_ORIGIN = "https://amann0071.github.io";
const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";

export default async function handler(req, res) {
  const origin = req.headers.origin;
  res.setHeader("Vary", "Origin");

  // Only allow the GitHub Pages site to call this function from a browser.
  if (origin !== ALLOWED_ORIGIN) {
    return res.status(403).json({ error: "Origin not allowed." });
  }
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "86400");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST, OPTIONS");
    return res.status(405).json({ error: "Method not allowed." });
  }

  // Decode and validate the request before contacting Gemini.
  let body = req.body;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return res.status(400).json({ error: "Request body must be valid JSON." });
    }
  }

  const prompt = body && body.prompt;
  if (typeof prompt !== "string" || prompt.trim().length === 0) {
    return res.status(400).json({ error: "Prompt must be a non-empty string." });
  }
  if (prompt.length > 2000) {
    return res.status(400).json({ error: "Prompt must be 2000 characters or fewer." });
  }

  // The key exists only in Vercel's server-side environment.
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: "Gemini is not configured on the server." });
  }

  // Stop waiting if the upstream request takes too long.
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(GEMINI_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt.trim() }] }],
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const status = response.status === 429 ? 503 : 502;
      return res.status(status).json({ error: "Gemini could not process the request." });
    }

    const data = await response.json();
    const text = (data.candidates?.[0]?.content?.parts || [])
      .map((part) => part.text || "")
      .join("")
      .trim();
    if (!text) {
      return res.status(502).json({ error: "Gemini returned an empty response." });
    }

    return res.status(200).json({ text });
  } catch (error) {
    if (error.name === "AbortError") {
      return res.status(504).json({ error: "Gemini request timed out." });
    }
    return res.status(502).json({ error: "Gemini is temporarily unavailable." });
  } finally {
    clearTimeout(timeout);
  }
}
