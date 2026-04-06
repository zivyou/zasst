import asyncio
import sys
import time

from rich.console import Console
from rich.panel import Panel

from rag.library import LibraryAgent


class TuiApp:

    def __init__(self):
        self._tool_count = 0
        self._session_start = time.time()
        self._agent = None
        self._console = Console()
        self._default_slash_cmds = {
            "/help": self._handle_help
        }

    def run(self) -> None:
        self._console.clear()
        self._print_welcome()

        # Initialize agent
        self._console.print("[dim]Initializing agent...[/dim]")
        library = LibraryAgent()

        while True:
            try:
                user_input = self._get_input()
                if (not user_input) or (not user_input.strip()):
                    continue

                if user_input.startswith("/"):
                    self._handle_slash_command(user_input)
                    continue
                loop = asyncio.new_event_loop()
                answer = loop.run_until_complete(library.ask(user_input))
                self._console.print(
                    Panel(
                        title="library answer",
                        border_style="blue",
                        renderable=answer,
                    )
                )
                loop.close()
            except (EOFError, KeyboardInterrupt):
                print("error")
                raise

    def _print_welcome(self):
        self._console.print(
            Panel(
                "[bold]Zasst[/bold] - zivyou's assistant\n"
                "[dim]Type a message, /help for commands, Ctrl+C twice to exit[/dim]",
                border_style="blue",
            )
        )


    def _get_input(self) -> str | None:
        """Get user input with prompt.

        Layout (matches Claude Code style):
          ────────────────────────────────────────
          ❯ <user types here>
          ────────────────────────────────────────
            branch:master
            claude-opus-4-6 | session:24s | ... | 🔧3
        """
        sep = "─" * 40
        status_lines = self._build_status_lines()
        # Pre-print: top sep, prompt, bottom sep, status bar
        # Then move cursor back up to the prompt line
        lines_below = 1 + len(status_lines)  # bottom sep + status lines
        self._console.print(f"  [dim]{sep}[/dim]")
        sys.stdout.write("  ❯ \n")
        sys.stdout.write(f"  {sep}\n")
        for sl in status_lines:
            sys.stdout.write(f"{sl}\n")
        # Move cursor back up to prompt line, column 5
        sys.stdout.write(f"\x1b[{lines_below + 1}A\x1b[5C")
        sys.stdout.flush()
        try:
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            # Move cursor down to clean position
            sys.stdout.write(f"\x1b[{lines_below}B\n")
            sys.stdout.flush()
            raise
        # Move past pre-printed lines below
        sys.stdout.write(f"\x1b[{lines_below}E")
        sys.stdout.flush()
        return user_input

    def _build_status_lines(self) -> list[str]:
        """Build the status bar text lines (plain strings for pre-printing)."""
        branch = "no-git"
        elapsed = int(time.time() - self._session_start)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"

        usage = "type anything you want"
        # total_tokens = usage["total_input_tokens"] + usage["total_output_tokens"]
        total_tokens = 1001
        if total_tokens > 1000:
            token_str = f"{total_tokens/1000:.1f}k"
        else:
            token_str = str(total_tokens)

        # cost = self.agent.cost_tracker.total_cost
        cost = 1000

        line1 = f"    branch:{branch}"
        line2 = f"    {"glm"} | session:{time_str} | tokens:{token_str} | ${cost:.4f} | 🔧{self._tool_count}"
        return [line1, line2]

    def _handle_slash_command(self, user_input: str) -> None:
        args: list[str] = user_input.split()[1:]
        if self._default_slash_cmds.get(user_input) is not None:
            self._default_slash_cmds[user_input](args)
            return



    def _handle_help(self, args: list) -> None:
        self._console.print(
            Panel(
                "[bold]Commands:[/bold]\n"
                "  /help     - Show this help\n"
                "  /model    - Show or switch model (e.g. /model openai:gpt-4o)\n"
                "  /cost     - Show token usage and cost\n"
                "  /skills   - List available skills\n"
                "  /mcp      - Show MCP server status\n"
                "  /clear    - Clear screen\n"
                "  /compact  - Reset conversation\n"
                "  /exit     - Exit",
                title="Help",
                border_style="blue",
            )
        )

