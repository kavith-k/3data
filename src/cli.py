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
    print("Generating code...\n")
    
    result = graph.invoke(initial_state, config)
    
    # Display result - result is now an AddableValuesDict not a State object
    error_message = result.get("error_message")
    generated_code = result.get("generated_code")
    
    if error_message:
        print(f"Error: {error_message}")
    elif generated_code:
        print("Generated Python Code:")
        print("-" * 80)
        print(generated_code)
        print("-" * 80)
        
        # Ask if user wants to save code to file
        save = input("Do you want to save this code to a file? (y/n): ")
        if save.lower() == 'y':
            filename = input("Enter filename (default: analysis.py): ") or "analysis.py"
            with open(filename, 'w') as f:
                f.write(generated_code)
            print(f"Code saved to {filename}")
            
            # Ask if user wants to run the code
            run_code = input("Do you want to run the code? (y/n): ")
            if run_code.lower() == 'y':
                print("\nRunning code...")
                try:
                    exec(generated_code)
                    print("\nCode execution completed.")
                except Exception as e:
                    print(f"\nError executing code: {str(e)}")
    else:
        print("No code was generated.")


if __name__ == "__main__":
    main() 