"""Define a LangGraph application for processing CSV data and generating Python code.

This agent takes a CSV file and a user question, and returns Python code that can answer
the user's question by analyzing the CSV data.
"""

import os
import pandas as pd
import re
import io
import sys
from typing import Any, Dict, Tuple, Optional, Callable
from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agent.configuration import Configuration
from agent.state import State


def setup_llm(config: Configuration) -> ChatOpenAI:
    """Create and configure the LLM with environment variables."""
    return ChatOpenAI(
        model=config.model_name,
        temperature=config.model_temperature,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )


def validate_input(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Validate the input state."""
    # Extract the Configuration object from the RunnableConfig
    configuration = Configuration.from_runnable_config(config)
    
    if not state.csv_path or not state.csv_path.exists():
        return {
            "error_message": f"CSV file not found: {state.csv_path}"
        }
    
    # Check file size
    file_size_mb = state.csv_path.stat().st_size / (1024 * 1024)
    if file_size_mb > configuration.max_csv_size_mb:
        return {
            "error_message": f"CSV file is too large: {file_size_mb:.2f}MB (max: {configuration.max_csv_size_mb}MB)"
        }
    
    return {}  # No errors


def load_csv_data(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load and analyze the CSV data."""
    # Extract the Configuration object from the RunnableConfig
    configuration = Configuration.from_runnable_config(config)
    
    # Skip if there's an error or if data is already loaded
    if state.error_message or state.csv_metadata:
        return {}
    
    try:
        # Read the CSV file
        df = pd.read_csv(state.csv_path)
        
        # Get preview
        preview = df.head(configuration.csv_preview_rows).to_string()
        
        # Get metadata
        metadata = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_values": {col: df[col].iloc[:3].tolist() for col in df.columns},
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
        }
        
        return {
            "csv_preview": preview,
            "csv_metadata": metadata
        }
    except Exception as e:
        return {
            "error_message": f"Error loading CSV data: {str(e)}"
        }


def clean_code(code: str) -> str:
    """Clean up the generated code to remove any syntax issues."""
    # Remove markdown code blocks if present
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    
    # Remove any HTML-like tags
    code = re.sub(r'<.*?>', '', code)
    
    # Fix path definitions
    code = code.replace("csv_path = 'path_to_your_csv_file.csv'", "")
    code = code.replace("csv_path = \"path_to_your_csv_file.csv\"", "")
    
    # Make sure print statements don't contain HTML or markdown
    code = re.sub(r'print\("(.*?)"\)', r'print("\1")', code)
    
    # Replace plt.show() with code to save the figure
    code = code.replace("plt.show()", "# plt.show() - Figures will be saved instead")
    
    return code


def generate_code(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate Python code to answer the user's question."""
    # Extract the Configuration object from the RunnableConfig
    configuration = Configuration.from_runnable_config(config)
    
    # Skip if there's an error or if we already have code
    if state.error_message or state.generated_code:
        return {}
    
    # Skip if we don't have metadata yet
    if not state.csv_metadata:
        return {}
    
    llm = setup_llm(configuration)
    
    # System prompt
    system_prompt = """You are a Python programming assistant specialized in data analysis.
Your task is to generate Python code that answers the user's question about a CSV dataset.

The CSV data has the following properties:
- Columns: {columns}
- Shape: {shape}
- Data types: {dtypes}
- Sample values: {sample_values}

Here's a preview of the data:
{csv_preview}

Generate complete, functioning Python code that:
1. Loads the CSV file from the 'csv_path' variable that is already defined
2. Uses pandas to process the data
3. Directly answers the user's question with appropriate data manipulation
4. Includes helpful comments explaining the code
5. MUST include print statements to display the results to the user
6. If appropriate, includes data visualization using matplotlib/seaborn

IMPORTANT GUIDELINES:
- The code must fully answer the user's question, not just load the data
- DO NOT omit any part of the solution; include all necessary data processing steps
- DO NOT include placeholders like '# Code to answer question here'
- The 'csv_path' variable is already defined; do not redefine it
- Always include necessary imports at the top of the file
- Make sure the final result is clearly printed or visualized
- Do NOT wrap your code in markdown code blocks (no ```python or ``` tags)
- Double-check your code against the available column names in the CSV file
- Make sure to use the correct column names in the code
- For visualizations, it's better to use plt.savefig() instead of plt.show() as this will be run in a non-interactive environment

EXAMPLE SOLUTION:
For the question "What are the top 3 products by total sales?" with a sales dataset containing 'product_name' and 'sales' columns:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file 
df = pd.read_csv(csv_path)

# Group by product_name and sum the sales
product_sales = df.groupby('product_name')['sales'].sum().reset_index()

# Sort products by total sales in descending order
top_products = product_sales.sort_values('sales', ascending=False).head(3)

# Print the top 3 products by total sales
print("Top 3 Products by Total Sales:")
print(top_products)

# Create a bar chart to visualize the top products
plt.figure(figsize=(10, 6))
plt.bar(top_products['product_name'], top_products['sales'])
plt.title('Top 3 Products by Total Sales')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
# Instead of plt.show() which doesn't work in non-interactive environments
# Save the figure to a file instead
plt.savefig('sales_chart.png')
print("Visualization saved to 'sales_chart.png'")
```

DO NOT include any commentary, explanations, or markdown. ONLY output valid Python code."""
    
    try:
        # Format the system prompt with the CSV metadata
        formatted_system_prompt = system_prompt.format(
            columns=str(state.csv_metadata.get("columns", [])),
            shape=str(state.csv_metadata.get("shape", (0, 0))),
            dtypes=str(state.csv_metadata.get("dtypes", {})),
            sample_values=str(state.csv_metadata.get("sample_values", {})),
            csv_preview=str(state.csv_preview or "CSV preview not available")
        )

        # Create the prompt with formatted system prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=state.user_input)
        ])
        
        # Create the chain
        chain = prompt | llm | StrOutputParser()
        
        # Generate the code
        code = chain.invoke({})
        
        # Clean the code
        code = clean_code(code)
        
        # Add file path handling if not present
        if not code.startswith("import"):
            code = "# Generated by CSV-to-Python-Code Generator\n" + code
        
        # Ensure the code has the correct CSV path
        abs_path = str(state.csv_path.absolute()).replace("\\", "/")
        code = f"# Define the path to the CSV file\ncsv_path = '{abs_path}'\n\n{code}"
        
        return {"generated_code": code}
    except Exception as e:
        return {"error_message": f"Error generating code: {str(e)}"}


def execute_code(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the generated Python code and capture the output."""
    # Skip if there's an error or no code to execute
    if state.error_message or not state.generated_code:
        return {}
    
    # Prepare to capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Create a namespace for execution
    namespace = {
        'pd': pd, 
        'plt': plt, 
        'csv_path': state.csv_path,
        'np': __import__('numpy')
    }
    
    # Add additional common imports that might be used
    try:
        namespace['sns'] = __import__('seaborn')
    except ImportError:
        pass  # seaborn not available
        
    try:
        # Configure matplotlib to use Agg backend (non-interactive)
        plt.switch_backend('Agg')
        
        # Clear any existing figures
        plt.close('all')
        
        # Temporarily redirect stdout and stderr
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code
            exec(state.generated_code, namespace)
            
            # Save any generated figures
            figure_paths = []
            for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
                figure_path = f"figure_{i+1}.png"
                fig.savefig(figure_path)
                figure_paths.append(figure_path)
            
            # Close all figures to free memory
            plt.close('all')
        
        # Get the captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # Filter out matplotlib warnings related to non-interactive backend
        filtered_stderr = []
        for line in stderr_output.splitlines():
            if "is non-interactive, and thus cannot be shown" not in line:
                filtered_stderr.append(line)
        
        stderr_output = "\n".join(filtered_stderr)
        
        # Handle errors in execution
        if stderr_output.strip():
            return {
                "execution_output": f"Code execution encountered errors:\n{stderr_output}",
                "execution_successful": False
            }
        
        # Format the result nicely
        result = "=== EXECUTION RESULTS ===\n\n"
        if stdout_output.strip():
            result += stdout_output.strip() + "\n\n"
        
        # Mention any saved figures
        if figure_paths:
            result += f"Visualization(s) saved to:\n"
            for path in figure_paths:
                result += f"- {path}\n"
        
        return {
            "execution_output": result,
            "execution_successful": True
        }
    
    except Exception as e:
        return {
            "execution_output": f"Error executing code: {str(e)}",
            "execution_successful": False
        }


# Define the workflow
workflow = StateGraph(State, config_schema=Configuration)

# Add nodes to the graph
workflow.add_node("validate_input", validate_input)
workflow.add_node("load_csv_data", load_csv_data)
workflow.add_node("generate_code", generate_code)
workflow.add_node("execute_code", execute_code)

# Define a simple linear flow
workflow.add_edge("__start__", "validate_input")
workflow.add_edge("validate_input", "load_csv_data")
workflow.add_edge("load_csv_data", "generate_code")
workflow.add_edge("generate_code", "execute_code")
workflow.add_edge("execute_code", END)

# Compile the workflow
graph = workflow.compile()
graph.name = "CSV-to-Python-Code Generator"

