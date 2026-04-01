import { SYSTEM_PROMPT, UNIVERSAL_TOOLS } from "@/lib/benchmark";
import type { ModelConfig } from "@/lib/models";

export type ModelMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: ProviderToolCall[];
  tool_call_id?: string;
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
};

export type GenerationParams = {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  tools_format?: "default" | "lfm";
};

const DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30;

type ChatResponse = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type?: string; text?: string }>;
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

function buildLfmMessages(messages: ModelMessage[]): ModelMessage[] {
  const toolList = UNIVERSAL_TOOLS.map((t) => t.function);
  const injection =
    `\n\nAvailable tools: ${JSON.stringify(toolList)}\n` +
    `When calling a tool output it between <|tool_call_start|> and <|tool_call_end|> tokens as a JSON array:\n` +
    `<|tool_call_start|>[{"name": "tool_name", "arguments": {"param": "value"}}]<|tool_call_end|>`;
  return messages.map((msg) => (msg.role === "system" ? { ...msg, content: msg.content + injection } : msg));
}

function parseLfmResponse(content: string): { content: string; toolCalls: ProviderToolCall[] } {
  const toolCalls: ProviderToolCall[] = [];
  let callIndex = 0;
  const blocks = content.match(/<\|tool_call_start\|>([\s\S]*?)<\|tool_call_end\|>/g) ?? [];

  for (const block of blocks) {
    const inner = block.replace(/<\|tool_call_start\|>/, "").replace(/<\|tool_call_end\|>/, "").trim();
    try {
      const parsed = JSON.parse(inner) as Array<{ name?: string; arguments?: unknown }>;
      for (const call of Array.isArray(parsed) ? parsed : [parsed]) {
        if (call?.name) {
          toolCalls.push({
            id: `tool_call_${++callIndex}`,
            type: "function",
            function: {
              name: call.name,
              arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {})
            }
          });
        }
      }
    } catch {
      // ignore malformed blocks
    }
  }

  return {
    content: content.replace(/<\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>/g, "").trim(),
    toolCalls
  };
}

function isTimeoutError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  return error.name === "TimeoutError" || error.name === "AbortError";
}

function resolveRequestTimeoutMs(): number {
  const rawTimeout = process.env.MODEL_REQUEST_TIMEOUT_SECONDS?.trim();

  if (!rawTimeout) {
    return DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS * 1000;
  }

  const parsed = Number.parseInt(rawTimeout, 10);

  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS * 1000;
  }

  return parsed * 1000;
}

export async function callModel(model: ModelConfig, messages: ModelMessage[], params?: GenerationParams): Promise<AssistantResponse> {
  const baseUrl = normalizeBaseUrl(model.baseUrl);
  const requestTimeoutMs = resolveRequestTimeoutMs();
  const headers: Record<string, string> = {
    "Content-Type": "application/json"
  };

  if (model.apiKey) {
    headers.Authorization = `Bearer ${model.apiKey}`;
  }

  const useLfmFormat = (params?.tools_format ?? "default") === "lfm";
  const body: Record<string, unknown> = {
    model: model.model,
    temperature: params?.temperature ?? 0,
    messages: useLfmFormat ? buildLfmMessages(messages) : messages,
    ...(useLfmFormat ? {} : { parallel_tool_calls: true, tool_choice: "auto", tools: UNIVERSAL_TOOLS })
  };

  if (params?.top_p !== undefined) {
    body.top_p = params.top_p;
  }

  if (params?.top_k !== undefined) {
    body.top_k = params.top_k;
  }

  if (params?.min_p !== undefined) {
    body.min_p = params.min_p;
  }

  if (params?.repetition_penalty !== undefined) {
    body.repetition_penalty = params.repetition_penalty;
  }

  let response: Response;

  try {
    response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(requestTimeoutMs)
    });
  } catch (error) {
    if (isTimeoutError(error)) {
      throw new Error(`Request timed out after ${requestTimeoutMs / 1000}s.`);
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

  if (useLfmFormat) {
    const parsed = parseLfmResponse(normalizeContent(message.content));
    return { content: parsed.content, toolCalls: parsed.toolCalls };
  }

  return {
    content: normalizeContent(message.content),
    toolCalls: normalizeToolCalls(message)
  };
}

export function createInitialMessages(userMessage: string): ModelMessage[] {
  return [
    { role: "system", content: `${SYSTEM_PROMPT}\n\nBenchmark context: today is 2026-03-20 (Friday). Use this date for any relative time request.` },
    { role: "user", content: userMessage }
  ];
}
