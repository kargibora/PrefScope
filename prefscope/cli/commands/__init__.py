"""Domain-specific command registration for the PrefScope CLI.

Each module owns the arguments for one coherent command family. Command handlers
remain in the matching ``prefscope.cli`` module so argument declarations and
implementations can be navigated by the same domain name.
"""

from prefscope.cli.commands.analysis import register_analysis_commands
from prefscope.cli.commands.bank import register_bank_commands
from prefscope.cli.commands.data import register_data_commands
from prefscope.cli.commands.interpret import register_interpret_commands
from prefscope.cli.commands.lens import register_lens_commands
from prefscope.cli.commands.token import register_token_commands
from prefscope.cli.commands.workflow import register_workflow_commands

__all__ = [
    "register_analysis_commands",
    "register_bank_commands",
    "register_data_commands",
    "register_interpret_commands",
    "register_lens_commands",
    "register_token_commands",
    "register_workflow_commands",
]
