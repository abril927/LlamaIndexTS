import { describe, expect, test } from "vitest";
// from unittest.mock import patch

import { OpenAI } from "llamaindex";
import { LLMSingleSelector } from "llamaindex/selectors/index";
import { mocStructuredkLlmGeneration } from "./utility/mockOpenAI.js";

describe("LLMSelector", () => {
  test("should be able to output a selection with a reason", async () => {
    const languageModel = new OpenAI({
      model: "gpt-3.5-turbo",
    });

    mocStructuredkLlmGeneration({
      languageModel,
    });

    const selector = new LLMSingleSelector({
      llm: languageModel,
    });

    const result = await selector.select(
      ["apple", "pear", "peach"],
      "what is the best fruit?",
    );

    expect(result.selections[0]!.reason).toBe("apple");
  });
});
