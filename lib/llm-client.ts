import { SYSTEM_PROMPT, UNIVERSAL_TOOLS } from "@/lib/benchmark";
import type { ModelConfig } from "@/lib/models";

export type ModelMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: ProviderToolCall[];
  tool_call_id?: string;
  reasoning?: string;
};

export type ProviderToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

export type AssistantResponse = {
  content: string;
  toolCalls: ProviderToolCall[];
  reasoning?: string;
};

const MODEL_REQUEST_TIMEOUT_MS = 30_000;

type ChatResponse = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type?: string; text?: string }>;
      reasoning_content?: string;
      reasoning?: string;
      tool_calls?: Array<{
        id?: string;
        type?: string;
        function?: {
          name?: string;
          arguments?: string | Record<string, unknown>;
        };
      }>;
    };
  }>;
  error?: {
    message?: string;
  };
};

type ProviderMessage = NonNullable<NonNullable<ChatResponse["choices"]>[number]["message"]>;
type ProviderContent = ProviderMessage["content"];

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
}

function normalizeContent(content: ProviderContent): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => (part?.type === "text" ? part.text ?? "" : ""))
      .join("")
      .trim();
  }

  return "";
}

function normalizeToolCalls(message: ProviderMessage): ProviderToolCall[] {
  return (
    message?.tool_calls?.map((call: NonNullable<ProviderMessage["tool_calls"]>[number], index: number) => ({
      id: call.id ?? `tool_call_${index + 1}`,
      type: "function",
      function: {
        name: call.function?.name ?? "unknown_tool",
        arguments:
          typeof call.function?.arguments === "string"
            ? call.function.arguments
            : JSON.stringify(call.function?.arguments ?? {})
      }
    })) ?? []
  );
}

function isTimeoutError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  return error.name === "TimeoutError" || error.name === "AbortError";
}

export async function callModel(model: ModelConfig, messages: ModelMessage[]): Promise<AssistantResponse> {
  const baseUrl = normalizeBaseUrl(model.baseUrl);
  const headers: Record<string, string> = {
    "Content-Type": "application/json"
  };

  if (model.apiKey) {
    headers.Authorization = `Bearer ${model.apiKey}`;
  }

  let response: Response;

  try {
    response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: model.model,
        temperature: 0,
        parallel_tool_calls: true,
        tool_choice: "auto",
        messages,
        tools: UNIVERSAL_TOOLS
      }),
      signal: AbortSignal.timeout(MODEL_REQUEST_TIMEOUT_MS)
    });
  } catch (error) {
    if (isTimeoutError(error)) {
      throw new Error(`Request timed out after ${MODEL_REQUEST_TIMEOUT_MS / 1000}s.`);
    }

    throw error;
  }

  const payload = (await response.json()) as ChatResponse;

  if (!response.ok) {
    throw new Error(payload.error?.message || `Provider request failed with ${response.status}.`);
  }

  const message = payload.choices?.[0]?.message;

  if (!message) {
    throw new Error("Provider returned no assistant message.");
  }

  return {
    content: normalizeContent(message.content),
    toolCalls: normalizeToolCalls(message),
    reasoning: message.reasoning_content ?? message.reasoning
  };
}

export function createInitialMessages(userMessage: string): ModelMessage[] {
  return [
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "system",
      content: "Benchmark context: today is 2026-03-20 (Friday). Use this date for any relative time request."
    },
    { role: "user", content: userMessage }
  ];
}
