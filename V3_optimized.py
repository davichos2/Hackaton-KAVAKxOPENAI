"""
V3_optimized.py - Modified version with parameter support

This version allows the evaluator to control LLM parameters.
Your original V3.py can stay as-is for interactive use.
The evaluator will use this version instead.
"""

from agents import Agent, Runner
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    print("⚠️  Warning: .env file not found")


def create_agent_with_params(params: dict = None):
    """
    Create an agent with specified parameters.

    Note: The 'agents' library may not support all OpenAI parameters directly.
    We'll set what we can through environment variables and agent configuration.
    """

    if params is None:
        params = {}

    # Set parameters as environment variables (if agents library reads them)
    if "temperature" in params:
        os.environ["OPENAI_TEMPERATURE"] = str(params["temperature"])
    if "max_tokens" in params:
        os.environ["OPENAI_MAX_TOKENS"] = str(params["max_tokens"])

    # Create agent with instructions
    instructions = """You are a helpful AI assistant. Provide clear, concise, and accurate responses.

Focus on:
- Relevance: Stay on topic and answer what was asked
- Factuality: Provide accurate information
- Clarity: Be clear and easy to understand
- Appropriate tone: Match the context of the question
- No hallucinations: Only state what you know to be true
"""

    agent = Agent(
        name="Assistant",
        instructions=instructions,
        # Add any other agent configuration here
    )

    return agent


def generate_response_with_params(prompt: str, params: dict = None):
    """
    Generate a response with specified parameters.
    This is the function the evaluator will call.

    Args:
        prompt: The user's input prompt
        params: Dictionary of LLM parameters to use

    Returns:
        str: The agent's response
    """
    agent = create_agent_with_params(params)
    result = Runner.run_sync(agent, prompt)
    return result.final_output


def print_banner():
    """Display welcome banner"""
    print("\n" + "="*60)
    print("🤖  CHAT WITH OPENAI - Interactive Terminal")
    print("="*60)
    print("Available commands:")
    print("  - Type your question and press Enter")
    print("  - 'salir', 'exit', 'quit' to exit")
    print("  - 'limpiar', 'clear' to clear screen")
    print("  - 'ayuda', 'help' to see commands")
    print("="*60 + "\n")


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def show_help():
    """Show help for commands"""
    print("\n📚 Available commands:")
    print("  • salir/exit/quit - Close chat")
    print("  • limpiar/clear - Clear screen")
    print("  • ayuda/help - Show this help")
    print("  • Any other input will be sent to AI\n")


def chat_loop():
    """Main chat loop for interactive use"""
    print_banner()

    # Create default agent for interactive use
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful AI assistant. Provide clear, concise, and accurate responses."
    )

    while True:
        try:
            # Read user input
            user_input = input("👤 You: ").strip()

            # Check if empty
            if not user_input:
                continue

            # Special commands
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("\n👋 Goodbye! Closing chat...\n")
                break

            elif user_input.lower() in ['limpiar', 'clear']:
                clear_screen()
                print_banner()
                continue

            elif user_input.lower() in ['ayuda', 'help']:
                show_help()
                continue

            # Send query to agent
            print("\n🤖 Assistant: ", end="", flush=True)
            result = Runner.run_sync(agent, user_input)
            print(result.final_output)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Interruption detected. Goodbye!\n")
            break

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            continue


# For testing the parameter-controlled version
def test_with_params():
    """Test function to verify parameter control works"""
    print("\n🧪 Testing parameter
