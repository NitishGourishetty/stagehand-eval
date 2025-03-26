import { generateObject, LanguageModel, streamText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { Stagehand } from "@browserbasehq/stagehand";
import StagehandConfig from "./stagehand.config.js";
import { getTools } from "./tools.js";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import readline from "readline";

/**
 * Helper function to count tokens using Anthropic's API
 */
async function countTokensFromMessage(message: string, role: "user" | "assistant" = "user"): Promise<number> {
  try {
    const response = await fetch("https://api.anthropic.com/v1/messages/count_tokens", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": process.env.ANTHROPIC_API_KEY || "",
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: "claude-3-sonnet-20240229",
        messages: [
          {
            role: role,
            content: message
          }
        ]
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Token counting failed (${response.status}): ${errorText}`);
      throw new Error(`Token counting failed: ${response.statusText}`);
    }

    const data = await response.json();
    // Debug log to standard error to avoid Python script parsing it
    console.error(`Token count response details: ${JSON.stringify(data, null, 2)}`);
    return data.input_tokens; // Return just the input_tokens because that's what we sent
  } catch (error) {
    console.error("Error counting tokens:", error);
    return 0;
  }
}

/**
 * Main function that executes an agent trajectory with Stagehand
 * @param query The prompt to be used for the agent trajectory
 * @param schema Optional Zod schema defining the structure of the output from the agent trajectory
 * @param config Configuration object for the models used
 * @param config.actionModel The model used to perform Stagehand actions (act/extract/observe)
 * @param config.structuredOutputModel The model used to generate structured output from the agent trajectory
 * @param config.cuaModel The model used to perform CUA actions (act/extract/observe)
 */
export async function main(
  query: string,
  schema?: z.ZodTypeAny,
  config: {
    trajectoryModel: LanguageModel;
    actionModel: LanguageModel;
    structuredOutputModel: LanguageModel;
    cuaModel:
      | "openai/computer-use-preview"
      | "anthropic/claude-3-7-sonnet-latest"
      | "anthropic/claude-3-5-sonnet-latest";
  } = {
    actionModel: openai("gpt-4o-mini"),
    structuredOutputModel: openai("gpt-4o-mini"),
    cuaModel: "openai/computer-use-preview",
    trajectoryModel: anthropic("claude-3-7-sonnet-latest"),
  }
) {
  const prompt = `You are a helpful assistant that can browse the web.
		You are given the following prompt:
		${query}
		${
      schema
        ? `Answer the prompt and be sure to contain a detailed response that covers at least the following requested data: ${JSON.stringify(
            schema
          )}`
        : ""
    }
		You may need to browse the web to find the answer.
		You may not need to browse the web at all; you may already know the answer.
		Do not ask follow up questions; I trust your judgement.
	  `;
  const stagehand = new Stagehand({
    ...StagehandConfig,
  });
  await stagehand.init();
  const page = stagehand.page;

  // Track accumulated output for token counting
  let accumulatedOutput = "";

  const result = streamText({
    model: config.trajectoryModel, // ONLY CLAUDE IS SUPPORTED FOR TRAJECTORY
    tools: getTools(page, stagehand, config.actionModel),
    toolCallStreaming: true,
    system:
      "You are a helpful assistant that can browse the web. You are given a prompt and you may need to browse the web to find the answer. You may not need to browse the web at all; you may already know the answer. Do not ask follow up questions; I trust your judgement.",
    prompt,
    maxSteps: 50,
    onStepFinish: async (step) => {      
      // Get actual usage from Anthropic response when available
      try {
        const responseObj = step.response as any;
        
        // Check if the response has usage information directly
        if (responseObj && responseObj.usage) {
          const usage = responseObj.usage;
          if (usage.input_tokens && usage.output_tokens) {
            const inputTokens = usage.input_tokens;
            const outputTokens = usage.output_tokens;
            const totalTokens = inputTokens + outputTokens;
            
            // Format token message consistently - this will be captured by the Python script
            // const tokenMessage = `TOKEN_COUNT: [stagehand:anthropic] Total tokens: ${totalTokens} (${inputTokens} input, ${outputTokens} output)`;
            // console.log(tokenMessage);
            return;
          }
        }
        
        // Fall back to explicit token counting for the current request if usage is not available
        if (step.request) {
          // Safely extract prompt text
          let promptText = "";
          try {
            const requestObj = step.request as any;
            promptText = requestObj.prompt || 
                         (requestObj.messages ? JSON.stringify(requestObj.messages) : "");
          } catch (error) {
            promptText = JSON.stringify(step.request);
          }
          
          const promptTokens = promptText ? 
            await countTokensFromMessage(promptText, "user") : 0;
          
          // Get response text if available
          let responseText = "";
          if (responseObj && responseObj.content) {
            responseText = typeof responseObj.content === 'string' ? 
              responseObj.content : 
              JSON.stringify(responseObj.content);
          }
          
          // Count tokens in the response
          const responseTokens = responseText ? 
            await countTokensFromMessage(responseText, "assistant") : 0;
          
          const totalTokens = promptTokens + responseTokens;
          
          // Format token message consistently
          // const tokenMessage = `TOKEN_COUNT: [stagehand:anthropic] Total tokens: ${totalTokens} (${promptTokens} input, ${responseTokens} output)`;
          // console.log(tokenMessage);
          return;
        }
        
        console.error("Cannot determine token usage for this request");
      } catch (error) {
        console.error("Error calculating token usage:", error);
      }
    },
    onFinish: async (result) => {
      console.log("\n\n\n---FINISHED---");
      
      // Use fixed values for token counts just to validate parsing
      const fixedTokenCount = 300;
      const fixedInputTokens = 100;
      const fixedOutputTokens = 200;
      
      // Send a consistent token format that will be captured by the Python script
      // console.log(`TOKEN_COUNT: [stagehand:anthropic] Total tokens: ${fixedTokenCount} (${fixedInputTokens} input, ${fixedOutputTokens} output)`);
      
      // Now try the actual token counting
      try {
        const promptTokens = await countTokensFromMessage(prompt, "user") || 0;
        const responseTokens = await countTokensFromMessage(accumulatedOutput, "assistant") || 0;
        const totalTokens = promptTokens + responseTokens;
        
        // Make sure this output gets captured by the Python script with a consistent format
        console.log(`FINAL_TOKEN_COUNT: [stagehand:anthropic] Total tokens: ${totalTokens} (${promptTokens} input, ${responseTokens} output)`);
      } catch (error) {
        console.error("Error counting final tokens:", error);
      }
      
      const cleanedResult = result.response.messages.map((m) => {
        if (m.role === "tool") {
          return {
            ...m,
            // Remove screenshot content
            content: m.content.map((c) => {
              if (c.toolName === "screenshot") {
                return {
                  ...c,
                  result: [],
                  experimental_content: [],
                };
              }
              return c;
            }),
          };
        }
        return m;
      });

      if (schema) {
        console.log("Generating structured output...");
        const structuredResult = await generateObject({
          model: config.structuredOutputModel,
          prompt: `You are given the following data of a web browsing session: 
			${JSON.stringify(cleanedResult)}
			Extract the requested data. If there is insufficient information, make it very clear that you are unable to adequately extract the requested data.
			If multiple pieces of information are requested, extract as much as you can without assuming or making up information.`,
          output: "no-schema",
        });
        console.log("Structured output:", structuredResult);
      }
    },
  });

  // Log the text stream and accumulate output
  for await (const textPart of result.textStream) {
    accumulatedOutput += textPart;
    process.stdout.write(textPart);
  }
}

// Add this at the end of the file
if (import.meta.url === `file://${process.argv[1]}`) {
  const query = process.argv[2] || "What can I help you with today?";
  console.log("Starting with query:", query);
  main(query).catch(console.error);
}
