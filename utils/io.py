"""IO utilities for the morpho tokenizer library."""

import json
import os
from typing import Dict, Any, List, Union


def load_json(path: str) -> Union[Dict[str, Any], List[Any]]:
    """Load a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Union[Dict[str, Any], List[Any]], path: str, indent: int = 2):
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Path to save to
        indent: Indentation level for JSON
        
    Raises:
        TypeError: If data is not JSON serializable
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_text(path: str) -> str:
    """Load a text file.
    
    Args:
        path: Path to text file
        
    Returns:
        Text content
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save_text(text: str, path: str):
    """Save text to a file.
    
    Args:
        text: Text to save
        path: Path to save to
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def load_lines(path: str, max_lines: int = None) -> List[str]:
    """Load lines from a text file.
    
    Args:
        path: Path to text file
        max_lines: Maximum number of lines to read (None for all)
        
    Returns:
        List of lines
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    with open(path, 'r', encoding='utf-8') as f:
        if max_lines is None:
            return [line.rstrip('\n') for line in f]
        else:
            return [line.rstrip('\n') for line in f][:max_lines]


def save_lines(lines: List[str], path: str):
    """Save lines to a text file.
    
    Args:
        lines: Lines to save
        path: Path to save to
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n') 