#!/usr/bin/env python
"""
CLI tool to interact with the CSV-to-Python-Code Generator.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from agent.graph import graph
from agent.state import State
from agent.configuration import Configuration


def main():
    """Run the CLI application."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Python code to analyze CSV data based on natural language questions."
    )
    parser.add_argument(
        "--csv", "-c", type=str, required=True, help="Path to the CSV file"
    )
    parser.add_argument(
        "--question", "-q", type=str, help="Question to ask about the data"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="anthropic/claude-3.7-sonnet", 
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.0,
        help="Temperature for the LLM (0.0-1.0)"
    )
    parser.add_argument(
        "--preview-rows", "-p", type=int, default=5,
        help="Number of rows to include in the preview"
    )
    parser.add_argument(
        "--show-code", "-s", action="store_true",
        help="Show the generated Python code in the output"
    )
    
    args = parser.parse_args()
    
    # Validate CSV file
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Get user question if not provided
    user_input = args.question
    if not user_input:
        user_input = input("Enter your question about the data: ")
    
    # Prepare configuration
    config = {
        "configurable": {
            "model_name": args.model,
            "model_temperature": args.temperature,
            "csv_preview_rows": args.preview_rows
        }
    }
    
    # Initialize state
    initial_state = State(
        user_input=user_input,
        csv_path=csv_path
    )
    
    # Run the graph
    print(f"Processing CSV file: {csv_path}")
    print(f"Question: {user_input}")
    print("Analyzing data...\n")
    
    result = graph.invoke(initial_state, config)
    
    # Display result - result is now an AddableValuesDict not a State object
    error_message = result.get("error_message")
    generated_code = result.get("generated_code")
    execution_output = result.get("execution_output")
    execution_successful = result.get("execution_successful", False)
    
    if error_message:
        print(f"Error: {error_message}")
    elif not generated_code:
        print("No code was generated.")
    else:
        # Show code if requested
        if args.show_code:
            print("\nGenerated Python Code:")
            print("-" * 80)
            print(generated_code)
            print("-" * 80)
        
        # Show execution results
        if execution_successful:
            print("\n" + execution_output)
        else:
            print("\nExecution failed:")
            print(execution_output)
            
            # Offer to save the code for debugging
            save = input("\nDo you want to save the code for debugging? (y/n): ")
            if save.lower() == 'y':
                filename = input("Enter filename (default: debug_code.py): ") or "debug_code.py"
                with open(filename, 'w') as f:
                    f.write(generated_code)
                print(f"Code saved to {filename}")


if __name__ == "__main__":
    main() 