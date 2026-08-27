# Chibi with VS Code: user guide

This guide covers the current `chibi-vscode` MVP. The extension is a thin local client: Chibi remains the
backend, owns the IDE JSONL protocol, and retains its existing model, tool, permission, moderation, history,
and command behavior.

## Requirements

- VS Code 1.85 or newer (the extension declares `engines.vscode: ^1.85.0`).
- Node.js 20 or newer is required to develop/package the extension; end users only need VS Code and a packaged
  extension.
- A Chibi installation with at least one configured AI-provider key. The backend repository currently publishes
  the `chibi-bot` package at version `1.13.0` and exposes the `chibi` executable.
- A VS Code workspace. Workspace trust is recommended before allowing Chibi's existing file and terminal tools.

The extension is published separately from the Chibi backend. Installing the extension does not install Chibi or
provider credentials.

## Install

Install the published Chibi extension from the VS Code Marketplace when available. For a locally built package,
install the VSIX from a terminal:

```sh
code --install-extension chibi-vscode-0.1.0.vsix --force
```

The filename is version-dependent. The extension package currently identifies itself as `chibi.chibi-vscode`,
displays as **Chibi**, and is MIT-licensed.

Install/configure Chibi independently. The simplest backend installation is:

```sh
pip install chibi-bot
chibi config
```

Configure the required provider credentials using Chibi's normal configuration process. Do not put credentials in
VS Code settings or in the extension repository.

## Configure the backend executable

Open **Settings**, search for `Chibi`, and configure:

| Setting | Type | Current default | Meaning |
|---|---|---|---|
| `chibi.executable` | string | `chibi` | Executable used to spawn the local backend. |
| `chibi.args` | array of strings | `["ide", "--stdio"]` | Arguments passed to that executable. |

The equivalent JSON settings are:

```json
{
  "chibi.executable": "chibi",
  "chibi.args": ["ide", "--stdio"]
}
```

If Chibi is installed in a virtual environment, use its absolute executable path, for example:

```json
{
  "chibi.executable": "/absolute/path/to/venv/bin/chibi",
  "chibi.args": ["ide", "--stdio"]
}
```

Keep the arguments as separate array elements; do not make one shell command string. The extension launches the
process directly and expects newline-delimited JSON on stdout. Chibi logs and diagnostics belong on stderr.

Before troubleshooting VS Code, verify the configured executable and arguments independently:

```sh
chibi ide --stdio
```

This is a protocol process, not an interactive terminal. Stop it with EOF/Ctrl-D rather than typing ordinary chat
text into it.

## Start a chat

1. Open a trusted VS Code workspace.
2. Run **Chibi: Open Chibi Chat** from the Command Palette (`chibi.openChat`).
3. Type a prompt and choose **Send**.
4. The extension starts the backend on demand, performs the protocol-v1 handshake, and shows the answer in the
   chat panel.

The backend is a child process of the extension. Reloading the VS Code window shuts down the owned process; opening
the chat again starts a fresh process.

The extension currently supports Chibi backend `1.13.0` and IDE protocol **v1**. A protocol mismatch is rejected
during startup; it is not silently downgraded.

## What context is sent

Each request includes the current editor snapshot:

- absolute workspace root;
- absolute active-file path, or `null` when no file is focused;
- selected text and zero-based selection start/end lines, or `null`;
- zero-based cursor line and character, or `null`;
- VS Code `languageId`, or `null`.

The extension does not pre-expand the workspace or send the whole file. Chibi receives this context and uses its
existing tools and permissions for any further work. Review prompts and tool requests accordingly; an active
selection is useful context, not a sandbox or an automatic file-edit proposal.

## Multiple chats and thread IDs

Use **＋ Chat** in the chat panel to create another chat. Each chat gets its own stable positive `thread_id` and
therefore its own Chibi history, lock, and task identity. Switching chats changes the active thread; it does not
merge histories. Requests in different threads can run concurrently. Requests in the same thread are serialized by
Chibi's existing per-thread lock.

The extension has no separate login/session identity layer. All IDE chats use Chibi's reserved IDE storage identity,
with `thread_id` providing per-chat isolation. The backend's persistent storage configuration determines whether
history survives a process restart.

## Commands, models, tools, and cancellation

Slash commands are ordinary request text sent through the same protocol. Current IDE commands are:

- `/help` — show the available command list.
- `/info` — show the IDE user identity and active model.
- `/model` — list models; send a model number or model name to select one for the current thread.
- `/reset` — clear the current thread and apply Chibi's existing reset/task behavior.
- `/imagine <description>` — request image generation; the description is required.
- `/quit` or `/exit` — request backend shutdown. Reload the window or reopen the chat to start it again.

Unknown slash commands are rejected; Telegram-only thread commands such as `/new_thread`, `/clone_thread`, and
`/drop_thread` are not part of this MVP. `/stop` is not the IDE cancellation mechanism.

Normal prompts use the same autonomous Chibi core as the other interfaces. Existing configured provider selection,
file/terminal tools, MCP integrations, permission checks, and moderation remain authoritative. The extension adds no
IDE tool allowlist or alternative policy. Confirm tool actions in the Chibi output/VS Code UI and do not assume the
extension provides workspace sandboxing.

The extension does not stream tokens. It shows coarse queued/running status and renders the complete result when
Chibi finishes. Use the request's **Cancel** button to cancel one in-flight request. Cancellation is targeted by
request ID: it does not cancel other requests, including another request in the same thread. Chibi reports a
cancelled request as an error and does not return a result for it.

## Troubleshooting

### The backend does not start

Run the exact executable and arguments from a terminal:

```sh
/path/to/chibi ide --stdio
```

Check that the executable is on PATH (or use an absolute path), that Chibi is configured, and that at least one
provider is ready. Ensure `chibi.args` is an array containing `"ide"` and `"--stdio"`.

### The chat reports a protocol error

This extension is pinned to protocol v1 and Chibi `1.13.0` by `chibi-vscode/package.json` (`chibiCompatibility`).
Install a matching backend release. The handshake rejects another protocol version rather than attempting an
unsafe compatibility mode.

### There is no answer or a provider error appears

Check Chibi's provider/API-key configuration and normal backend logs. The extension only transports the request;
provider availability, model selection, tool failures, and moderation are backend concerns. Use `/info` and `/model`
to inspect/select the current model.

### I cannot see diagnostics

Open **View → Output**, select the **Chibi** output channel, and inspect backend stderr. Protocol stdout must contain
JSONL only, so do not redirect logs into stdout.

### Context or tools behave unexpectedly

Confirm the intended workspace is open and the correct file/selection is active. Remember that the extension sends
the snapshot described above; it does not send a complete workspace and does not provide a sandbox, diff workflow,
`WorkspaceEdit` proposals, or a tool allowlist.

### Stop and restart

Reload the VS Code window. This shuts down the extension-owned backend process and permits a clean restart.

## Current limitations

The MVP deliberately does not provide token streaming, a dedicated model-picker protocol, IDE-specific tool policy,
workspace sandboxing, diff/`WorkspaceEdit` workflows, a generated shared SDK, a session identity layer, or other IDE
products. These are not hidden capabilities and should not be promised to users.

The backend and extension are released independently. A backend release is not a Marketplace extension release;
check both compatibility declarations before upgrading either one.
