# Code Quality Guidelines for Triathlon ML Pipeline

## Indentation Standards

### 1. **Use 4 Spaces (Never Tabs)**
```python
# GOOD - 4 spaces per level
def my_function():
    if condition:
        do_something()
        return result

# BAD - mixing tabs and spaces
def my_function():
	if condition:  # tab
        do_something()  # 4 spaces
```

### 2. **VS Code Settings for Consistency**
Add to your VS Code settings.json:
```json
{
    "python.formatting.provider": "black",
    "editor.insertSpaces": true,
    "editor.tabSize": 4,
    "editor.detectIndentation": false,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true
}
```

### 3. **Docstring Templates**
Use consistent docstring formatting:
```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    Brief description of function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
    """
    # Implementation here
```

### 4. **Multi-line String Handling**
```python
# GOOD - properly indented
logger.info(
    "This is a long message that spans "
    "multiple lines with proper indentation"
)

# GOOD - using f-strings with proper breaks
logger.info(
    f"Processed {len(data)} rows with "
    f"{success_count} successful operations"
)
```

## Prevention Strategies

### 1. **Use Black Formatter**
Install and configure Black for automatic formatting:
```bash
pip install black
black ml/ train.py
```

### 2. **Pre-commit Hooks**
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

### 3. **Linting Configuration**
Create `.flake8` file:
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = __pycache__, .git, venv
```

### 4. **Template for New Files**
Use this template for new Python files:
```python
"""
Module description.

Detailed module documentation here.
"""

import logging
from typing import List, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)


class ExampleClass:
    """Class description."""
    
    def __init__(self, param: str):
        """Initialize class."""
        self.param = param
    
    def example_method(self, data: List[str]) -> Dict[str, int]:
        """
        Method description.
        
        Args:
            data: List of strings to process
            
        Returns:
            Dictionary with results
        """
        # Implementation
        pass


if __name__ == "__main__":
    # Test code here
    pass
```

## Common Pitfalls to Avoid

### 1. **Broken Docstrings**
```python
# BAD - missing closing quotes
def function():
    """
    Docstring without closing
    
# GOOD - properly closed
def function():
    """
    Docstring with proper closing.
    """
```

### 2. **Mixed Indentation in Same Block**
```python
# BAD
if condition:
    do_something()
        do_something_else()  # Wrong indentation

# GOOD
if condition:
    do_something()
    do_something_else()
```

### 3. **Long Lines Breaking Indentation**
```python
# BAD
very_long_function_call_that_exceeds_line_limit(argument1, argument2, argument3, argument4)

# GOOD
very_long_function_call_that_exceeds_line_limit(
    argument1, 
    argument2, 
    argument3, 
    argument4
)
```

## Tools for Checking

### 1. **Python AST Checker**
```python
import ast

def check_syntax(filename):
    try:
        with open(filename, 'r') as f:
            ast.parse(f.read())
        print(f"✓ {filename} syntax is valid")
    except SyntaxError as e:
        print(f"✗ {filename} syntax error: {e}")
```

### 2. **Indentation Checker Script**
```python
def check_indentation(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        if line.strip() and line.startswith('\t'):
            print(f"Tab found at line {i}: {filename}")
```

Would you like me to:
1. Set up Black formatter for the project?
2. Create a syntax checking script?
3. Fix any remaining indentation issues in the current files?
