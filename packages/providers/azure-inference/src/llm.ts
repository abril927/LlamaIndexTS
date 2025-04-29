import { ToolCallLLM, type ChatResponse, type ChatResponseChunk, type LLMChatParamsNonStreaming, type LLMChatParamsStreaming, type LLMMetadata, type ToolCallLLMMessageOptions } from '@llamaindex/core/llms';
import ModelClient, { type ChatCompletionsToolDefinition, type GetChatCompletionsBodyParam, type ChatRequestMessage, isUnexpected, type GetChatCompletions200Response, type ChatCompletionsOutput, type ChatCompletionsToolCall } from '@azure-rest/ai-inference';
import type { TokenCredential, KeyCredential } from '@azure/core-auth';
import { wrapLLMEvent } from '@llamaindex/core/decorator';
import { extractText, streamConverter } from '@llamaindex/core/utils';
import { createSseStream, type EventMessage } from '@azure/core-sse';

export type AzureInferenceModelOptions = 
  Omit<GetChatCompletionsBodyParam['body'],
    // making them required
    'temperature' | 'top_p' | 'max_tokens' |
    // will be set by us/doesn't make sense to expose
    'model' | 'tools' | 'tool_choice' | 'response_format' | 'stream' | 'messages'
  >
  & Pick<Required<Required<GetChatCompletionsBodyParam>['body']>, 'temperature' | 'top_p' | 'max_tokens'>;

export type AzureInferenceParams = {
  model: string;
  endpoint: string;
  credential: TokenCredential | KeyCredential;
  options: Partial<AzureInferenceModelOptions>;
}

function messageAccessor(part: EventMessage): ChatResponseChunk<ToolCallLLMMessageOptions> | null {
  if (part.data == '[DONE]') return null;

  const data = JSON.parse(part.data) as ChatCompletionsOutput;

  const choice = data.choices[0] as any;
  if (!choice) return null;

  if (choice.delta?.tool_calls) {
    return {
      raw: part,
      delta: choice.delta.content,
      options: {
        toolCall: choice.delta.tool_calls.map((tc: ChatCompletionsToolCall) => ({
          name: tc.function.name,
          id: tc.id,
          input: JSON.parse(tc.function.arguments)
        }))
      }
    }
  } else if (choice.delta) {
    return {
      raw: part,
      delta: choice.delta.content
    }
  } else return null;
}

export class AzureInference extends ToolCallLLM {
  supportToolCall: boolean = true;
  public readonly client: ReturnType<typeof ModelClient>;

  model: string;
  options: AzureInferenceModelOptions = {
    top_p: 1,
    temperature: 1,
    max_tokens: 4096
  };

  constructor(params: AzureInferenceParams) {
    super();
    this.model = params.model;
    this.client = ModelClient(params.endpoint, params.credential);

    if (params.options) {
      this.options = {
        ...this.options,
        ...params.options
      }
    }
  }

  get metadata(): LLMMetadata {
    return {
      model: this.model,
      temperature: this.options.temperature,
      topP: this.options.top_p,
      maxTokens: this.options.max_tokens,
      contextWindow: this.options.max_tokens, // TODO: should this be different from maxTokens
      tokenizer: undefined
    }
  }

  chat(params: LLMChatParamsStreaming<ToolCallLLMMessageOptions>): Promise<AsyncIterable<ChatResponseChunk>>;
  chat(params: LLMChatParamsNonStreaming<ToolCallLLMMessageOptions>): Promise<ChatResponse<ToolCallLLMMessageOptions>>;
  @wrapLLMEvent
  async chat(
    params: 
      | LLMChatParamsNonStreaming<object, ToolCallLLMMessageOptions>
      | LLMChatParamsStreaming<object, ToolCallLLMMessageOptions>
  ): Promise<ChatResponse<ToolCallLLMMessageOptions> | AsyncIterable<ChatResponseChunk>> {
    // Prepare request
    const body: GetChatCompletionsBodyParam['body'] = {
      messages: params.messages.map((m): ChatRequestMessage => {
        // tool results are special
        if (m.options && 'toolResult' in m.options) {
          return {
            role: 'tool',
            content: m.options.toolResult.result,
            tool_call_id: m.options.toolResult.id
          }
        }

        // non-tool messages
        return {
          role: m.role, // TODO: what if this is 'memory'? when is this 'memory'?
          content: extractText(m.content)
        }
      }), // TODO
      tools: [], // TODO
      tool_choice: 'auto',
      response_format: { type: 'json' },

      model: this.model,
      stream: !!params.stream
    }

    if (params.tools) {
      body.tools = params.tools.map((t): ChatCompletionsToolDefinition => ({
        type: 'function',
        function: {
          name: t.metadata.name,
          description: t.metadata.description,
          parameters: t.metadata.parameters // is already JSON Schema, and Azure wants JSON Schema, so fine to just pass it I think
        }
      }));
    }

    // Make the request
    // TODO: error on extra parameters with header
    if (params.stream) {
      // Streaming
      const response = await this.client.path('/chat/completions').post({
        body,
        headers: {
          'extra-parameters': 'error'
        }
      }).asBrowserStream(); // FIXME: is this gonna cause problems? since it's browser and not node

      if (response.body == undefined) throw new Error('Response body is undefined');

      const stream = createSseStream(response.body);
      return streamConverter(stream, messageAccessor)
    } else {
      // Non-streaming
      const response = await this.client.path('/chat/completions').post({
        body,
        headers: {
          'extra-parameters': 'error'
        }
      });

      if ('error' in response.body) throw response.body.error;

      const choice = response.body.choices[0];
      if (choice == undefined) throw new Error('No choices received from chat completions endpoint');
      if (choice.message?.tool_calls) {
        return {
          message: {
            role: 'assistant',
            content: choice.message.content!,
            options: {
              toolCall: choice.message.tool_calls.map(call => ({
                name: call.function.name,
                input: call.function.arguments,
                id: call.id
              }))
            }
          },
          raw: response.body
        }
      }

      return {
        message: {
          role: 'assistant',
          content: choice.message.content!
        },
        raw: response.body
      }
    }
  }

  // TODO: complete()
}

/**
 * Convenience function to create a new Azure Inference instance.
 * @param init - Optional initialization parameters for the Azure Inference instance.
 * @returns A new Azure Inference instance.
 */
export const azureInference = (init: ConstructorParameters<typeof AzureInference>[0]) =>
  new AzureInference(init);
