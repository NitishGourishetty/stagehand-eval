// Set NODE_NO_WARNINGS to suppress deprecation warnings
process.env.NODE_NO_WARNINGS = "1";

import readline from "readline";
import { main } from "./index.js";
import dotenv from "dotenv";
import chalk from "chalk";

// Load environment variables
dotenv.config();

// Create readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Promisify readline question
const question = (query: string): Promise<string> => {
  return new Promise((resolve) => {
    rl.question(query, resolve);
  });
};

// Check and prompt for required API keys
async function checkAndPromptApiKeys() {
  let envUpdated = false;

  if (!process.env.ANTHROPIC_API_KEY) {
    const anthropicKey = await question(
      chalk.yellow("No Anthropic API key found. ") +
        chalk.gray(
          "We use Anthropic Claude 3.7 Sonnet to power our agent trajectory reasoning.\n\n"
        ) +
        chalk.cyan("Please enter your Anthropic API key: ")
    );
    process.env.ANTHROPIC_API_KEY = anthropicKey;
    envUpdated = true;
  }

  if (!process.env.OPENAI_API_KEY) {
    const openaiKey = await question(
      chalk.yellow("No OpenAI API key found. ") +
        chalk.gray(
          "We use OpenAI GPT-4o-mini to power our agent action execution.\n\n"
        ) +
        chalk.cyan("Please enter your OpenAI API key: ")
    );
    process.env.OPENAI_API_KEY = openaiKey;
    envUpdated = true;
  }

  if (envUpdated) {
    console.log("\nAPI keys have been set for this session.");
    console.log("To persist these keys, add them to your .env file.");
  }
}

// Get query from command line argument if provided, otherwise use readline
const queryFromArgs = process.argv[2];

async function run() {
  await checkAndPromptApiKeys();

  if (queryFromArgs) {
    main(queryFromArgs);
  } else {
    const query = await question("\n\nEnter your query: ");
    main(query);
    rl.close();
  }
}

run().catch(console.error);
