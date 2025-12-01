import re
import sys

def add_logging_to_file(filepath, log_filename):
    """Add logging imports and replace print statements in a Python file."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if logging already added
    if 'from src.logging.logging_config import setup_logger' in content:
        print(f"Logging already added to {filepath}")
        return
    
    # Imports to add
    imports_to_add = f"""import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logging.logging_config import setup_logger

# Set up logger for this module
logger = setup_logger(__name__, '{log_filename}')

"""
    
    # Find first import statement
    first_import = re.search(r'^(import |from )', content, re.MULTILINE)
    if first_import:
        content = content[:first_import.start()] + imports_to_add + content[first_import.start():]
    else:
        # No imports found, add at start after shebang and docstring
        content = imports_to_add + content
    
    # Replace print statements with logger.info
    content = re.sub(r'\bprint\(', 'logger.info(', content)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated {filepath}")

if __name__ == "__main__":
    files_to_update = [
        ('src/embeding/train_cbow.py', 'train_cbow.log'),
        ('src/embeding/test_cbow.py', 'test_cbow.log'),
    ]
    
    for filepath, logfile in files_to_update:
        add_logging_to_file(filepath, logfile)
