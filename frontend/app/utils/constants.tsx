export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8123";
export const LANGCHAIN_API_KEY = process.env.NEXT_PUBLIC_LANGCHAIN_API_KEY;

if (!LANGCHAIN_API_KEY) {
  console.error("LANGCHAIN_API_KEY is not defined");
}

export const RESPONSE_FEEDBACK_KEY = "user_score";
export const SOURCE_CLICK_KEY = "user_click";
