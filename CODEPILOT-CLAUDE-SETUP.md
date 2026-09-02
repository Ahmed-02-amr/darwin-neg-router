# CodePilot, Claude Code, and web search setup

The Darwin router serves both protocols from one local address:

- Anthropic Messages base URL: `http://127.0.0.1:11435`
- Anthropic Messages endpoint: `http://127.0.0.1:11435/v1/messages`
- OpenAI base URL: `http://127.0.0.1:11435/v1`
- OpenAI chat endpoint: `http://127.0.0.1:11435/v1/chat/completions`

Do not add `/v1` to the Anthropic base URL. Claude Code appends
`/v1/messages` itself.

The bundled native profile exposes a `163840`-token total context window and a
`43008`-token maximum output allowance. With the full output reserve, up to
`120832` prompt tokens remain available. The native runner uses Q8 K/V cache,
one slot, and Flash Attention while leaving the Q6_K model weights unchanged.

## Add Darwin as an Anthropic third-party provider

1. Start Darwin NEG Control and wait for **Running**.
2. In CodePilot, open **Settings -> Providers -> Add Provider**.
3. Select **Anthropic (Third-Party Compatible)**.
4. Enter:

   | Field | Value |
   |---|---|
   | Name | `Darwin NEG (Claude Code)` |
   | Authentication | `API Key` |
   | API key | `EMPTY` |
   | Base URL | `http://127.0.0.1:11435` |

5. Leave the optional Sonnet, Opus, and Haiku model mappings blank. Claude
   Code uses its recognized `sonnet`/`opus`/`haiku` slots, and this local
   router deliberately sends every ordinary slot to Darwin NEG. A raw custom
   model ID such as `darwin-neg-agent` can be rejected by Claude Code's local
   model allow-list before the request reaches the router.
6. Under extra environment variables, add this JSON if the form exposes the
   field:

   ```json
   {
     "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
     "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1"
   }
   ```

7. Save and run the provider diagnostic. The live request is the decisive
   check; model-list discovery alone is not.

Claude Code may display the recognized wire slot as Sonnet and calculate a
notional Anthropic cost from its local pricing table. The request still runs
entirely through Darwin on localhost; that display is not an external charge.

## Select the Claude Code runtime

1. Open CodePilot **Settings -> Runtime**.
2. Confirm Claude Code is detected. If it is not, install/re-detect the Claude
   Code CLI first.
3. In **Default engine**, select **Claude Code** rather than **CodePilot**.
4. Create a new task and choose the `Darwin NEG (Claude Code)` provider.

CodePilot describes this engine as the one that honors
`~/.claude/settings.json`, hooks, MCP servers, and Claude Code skills. Existing
tasks may remain pinned to their original runtime, so use a new task for the
first test.

Start with this protocol check:

```text
Call the Skill tool without arguments and tell me how many skills it returns.
```

## Add maintained web search

The previous `@modelcontextprotocol/server-puppeteer` entry is deprecated and
is browser automation rather than a search-results service. Tavily's maintained
MCP server supports keyless search and extraction; an API key unlocks its
additional features and higher limits.

In CodePilot's **MCP** page, remove or disable the broken `free-web-search`
entry, then add a **stdio** server with separate executable and argument fields:

| Field | Value |
|---|---|
| Name | `tavily-search` |
| Type | `stdio` |
| Command | `npx` |
| Argument 1 | `-y` |
| Argument 2 | `tavily-mcp@latest` |

For keyless mode, leave environment variables empty. For a Tavily account, add:

```json
{
  "TAVILY_API_KEY": "tvly-your-key"
}
```

Equivalent Claude settings JSON:

```json
{
  "mcpServers": {
    "tavily-search": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "tavily-mcp@latest"]
    }
  }
}
```

The important detail is that `command` is only `npx`; do not put the complete
shell command in that field. Enable/reconnect the server and verify that its
status is **Connected** before opening a new task.

Keep CodePilot's permission mode on **Request approval** for the first test and
approve the Tavily call when prompted. A deny-without-asking mode blocks MCP
execution even when the tool is connected. After verifying it, you can add the
specific Tavily search tool to your persistent allow-list if desired.

Test with:

```text
Use tavily-search to find the current gold spot price. Cite the result URLs and
state the quote time and currency. Do not use Bash or curl for the search.
```

If Darwin still uses Bash, inspect the current task's available-tool list. A
connected MCP should contribute named Tavily tools; their absence means the
server did not attach to that task, regardless of what the model says.

## API smoke checks

PowerShell non-streaming request:

```powershell
$body = @{
    model = 'sonnet'
    max_tokens = 1024
    messages = @(@{ role = 'user'; content = 'Reply with: ok' })
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Method Post `
    -Uri 'http://127.0.0.1:11435/v1/messages' `
    -Headers @{ 'x-api-key' = 'EMPTY'; 'anthropic-version' = '2023-06-01' } `
    -ContentType 'application/json' `
    -Body $body
```

The router also implements `POST /v1/messages/count_tokens` for Claude SDK
context checks. Its value is a conservative local estimate because exact token
counting occurs after llama.cpp renders the Qwen chat template.
