---
name: fanout-setup
description: Set up fanout environment — configure API keys and provider settings
user_invocable: true
---

# /setup — Configure Fanout Environment

Set up the `.env` file with the required API keys and provider configuration for fanout.

## Steps

1. Check if `.env` already exists in the project root. If it does, read it and show the user what's currently configured (mask key values, e.g. `sk-or-...****`).

2. Ask the user for their **OpenRouter API key** if `OPENROUTER_API_KEY` is not already set. They can get one at https://openrouter.ai/keys

3. Write/update the `.env` file with the key:
   ```
   OPENROUTER_API_KEY=<their key>
   ```

4. Verify the key works by running:
   ```bash
   uv run python -c "
   import httpx, os
   from dotenv import load_dotenv
   load_dotenv()
   r = httpx.get('https://openrouter.ai/api/v1/models', headers={'Authorization': f\"Bearer {os.environ['OPENROUTER_API_KEY']}\"})
   r.raise_for_status()
   models = r.json().get('data', [])
   print(f'OK — {len(models)} models available')
   "
   ```

5. Confirm `.env` is in `.gitignore` (it should be). Warn the user if it isn't.

6. Print a summary of what was configured and suggest a quick test:
   ```bash
   uv run fanout sample "Say hello" -m openai/gpt-4o-mini -n 1
   ```

## Important

- NEVER commit `.env` files or print full API keys
- If the user already has a key configured, ask before overwriting
- The `.env` file is loaded automatically by the fanout CLI via python-dotenv
