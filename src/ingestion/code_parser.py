"""
src/ingestion/code_parser.py

Multi-language code parser using regex patterns and AST where available
Supports: Python, Java, TypeScript, JavaScript
"""
import ast
import re
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PythonCodeParser:
    """Extract structured information from Python code using AST"""
    
    def parse_file(self, content: str, file_path: str) -> Dict:
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return {'file_path': file_path, 'functions': [], 'classes': [], 'imports': []}
        
        result = {'file_path': file_path, 'functions': [], 'classes': [], 'imports': []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                result['functions'].append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'docstring': ast.get_docstring(node),
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'type': 'function'
                })
            elif isinstance(node, ast.ClassDef):
                methods = [{'name': item.name, 'line_start': item.lineno, 'line_end': item.end_lineno,
                           'docstring': ast.get_docstring(item), 'args': [a.arg for a in item.args.args],
                           'decorators': [], 'is_async': False, 'type': 'method'}
                          for item in node.body if isinstance(item, ast.FunctionDef)]
                result['classes'].append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'docstring': ast.get_docstring(node),
                    'methods': methods,
                    'bases': [self._get_name(base) for base in node.bases],
                    'type': 'class'
                })
        return result
    
    def _get_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def create_searchable_chunks(self, parsed_data: Dict, content: str) -> List[Dict]:
        chunks = []
        lines = content.split('\n')
        file_path = parsed_data['file_path']
        
        for func in parsed_data['functions']:
            code = '\n'.join(lines[func['line_start']-1:func['line_end']])
            text = f"Python Function: {func['name']}\nFile: {file_path}\n"
            if func['docstring']:
                text += f"Description: {func['docstring']}\n"
            text += f"Arguments: {', '.join(func['args'])}\nCode:\n{code}"
            
            chunks.append({
                'type': 'function', 'name': func['name'], 'file': file_path,
                'line_start': func['line_start'], 'line_end': func['line_end'],
                'text': text, 'code': code, 'docstring': func['docstring'],
                'metadata': {'args': func['args'], 'decorators': func['decorators'],
                            'is_async': func['is_async'], 'language': 'python'}
            })
        
        for cls in parsed_data['classes']:
            code = '\n'.join(lines[cls['line_start']-1:cls['line_end']])
            text = f"Python Class: {cls['name']}\nFile: {file_path}\n"
            if cls['docstring']:
                text += f"Description: {cls['docstring']}\n"
            text += f"Methods: {', '.join([m['name'] for m in cls['methods']])}\nCode:\n{code}"
            
            chunks.append({
                'type': 'class', 'name': cls['name'], 'file': file_path,
                'line_start': cls['line_start'], 'line_end': cls['line_end'],
                'text': text, 'code': code, 'docstring': cls['docstring'],
                'metadata': {'methods': [m['name'] for m in cls['methods']],
                            'bases': cls['bases'], 'language': 'python'}
            })
        return chunks


class JavaCodeParser:
    """Extract structured information from Java code using regex patterns"""
    
    CLASS_PATTERN = re.compile(
        r'(?:@\w+(?:\([^)]*\))?\s*)*(?:public|private|protected)?\s*(?:abstract|final)?\s*'
        r'(?:class|interface)\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?\s*\{',
        re.MULTILINE
    )
    METHOD_PATTERN = re.compile(
        r'(?:@\w+(?:\([^)]*\))?\s*)*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?'
        r'(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
        re.MULTILINE
    )
    ANNOTATION_PATTERN = re.compile(r'@(\w+)(?:\([^)]*\))?')
    
    def parse_file(self, content: str, file_path: str) -> Dict:
        result = {'file_path': file_path, 'classes': [], 'methods': [], 'imports': []}
        lines = content.split('\n')
        
        for line in lines:
            if line.strip().startswith('import '):
                result['imports'].append(line.strip())
        
        for match in self.CLASS_PATTERN.finditer(content):
            start_pos = match.start()
            prefix = content[max(0, start_pos-500):start_pos]
            annotations = self.ANNOTATION_PATTERN.findall(prefix[-200:])
            line_start = content[:start_pos].count('\n') + 1
            line_end = self._find_block_end(content, match.end())
            
            result['classes'].append({
                'name': match.group(1),
                'extends': match.group(2),
                'implements': match.group(3).split(',') if match.group(3) else [],
                'annotations': annotations,
                'line_start': line_start,
                'line_end': line_end,
                'type': 'class'
            })
        
        for match in self.METHOD_PATTERN.finditer(content):
            return_type, method_name, params = match.group(1), match.group(2), match.group(3)
            if return_type == method_name:  # Skip constructors
                continue
            start_pos = match.start()
            prefix = content[max(0, start_pos-300):start_pos]
            annotations = self.ANNOTATION_PATTERN.findall(prefix[-150:])
            line_start = content[:start_pos].count('\n') + 1
            line_end = self._find_block_end(content, match.end())
            
            result['methods'].append({
                'name': method_name, 'return_type': return_type, 'params': params,
                'annotations': annotations, 'line_start': line_start,
                'line_end': line_end, 'type': 'method'
            })
        return result
    
    def _find_block_end(self, content: str, start: int) -> int:
        brace_count, pos = 1, start
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        return content[:pos].count('\n') + 1
    
    def create_searchable_chunks(self, parsed_data: Dict, content: str) -> List[Dict]:
        chunks = []
        lines = content.split('\n')
        file_path = parsed_data['file_path']
        
        for cls in parsed_data['classes']:
            code = '\n'.join(lines[max(0, cls['line_start']-1):cls['line_end']])[:2000]
            text = f"Java Class: {cls['name']}\nFile: {file_path}\n"
            if cls['extends']:
                text += f"Extends: {cls['extends']}\n"
            if cls['implements']:
                text += f"Implements: {', '.join(cls['implements'])}\n"
            if cls['annotations']:
                text += f"Annotations: @{', @'.join(cls['annotations'])}\n"
            text += f"Code:\n{code}"
            
            chunks.append({
                'type': 'class', 'name': cls['name'], 'file': file_path,
                'line_start': cls['line_start'], 'line_end': cls['line_end'],
                'text': text, 'code': code, 'docstring': None,
                'metadata': {'extends': cls['extends'], 'implements': cls['implements'],
                            'annotations': cls['annotations'], 'language': 'java'}
            })
        
        for method in parsed_data['methods']:
            code = '\n'.join(lines[max(0, method['line_start']-1):method['line_end']])
            text = f"Java Method: {method['name']}\nFile: {file_path}\n"
            text += f"Returns: {method['return_type']}\nParameters: {method['params']}\n"
            if method['annotations']:
                text += f"Annotations: @{', @'.join(method['annotations'])}\n"
            text += f"Code:\n{code}"
            
            chunks.append({
                'type': 'method', 'name': method['name'], 'file': file_path,
                'line_start': method['line_start'], 'line_end': method['line_end'],
                'text': text, 'code': code, 'docstring': None,
                'metadata': {'return_type': method['return_type'], 'params': method['params'],
                            'annotations': method['annotations'], 'language': 'java'}
            })
        return chunks


class TypeScriptCodeParser:
    """Extract structured information from TypeScript/JavaScript code"""
    
    CLASS_PATTERN = re.compile(
        r'(?:@\w+(?:\([^)]*\))?\s*)*(?:export\s+)?class\s+(\w+)'
        r'(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?\s*\{',
        re.MULTILINE
    )
    FUNCTION_PATTERN = re.compile(
        r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*(\w+(?:<[^>]+>)?))?\s*\{',
        re.MULTILINE
    )
    DECORATOR_PATTERN = re.compile(r'@(\w+)\s*\(([^)]*)\)')
    
    def parse_file(self, content: str, file_path: str) -> Dict:
        result = {'file_path': file_path, 'classes': [], 'functions': [], 'imports': []}
        lines = content.split('\n')
        
        for line in lines:
            if 'import ' in line:
                result['imports'].append(line.strip())
        
        for match in self.CLASS_PATTERN.finditer(content):
            start_pos = match.start()
            prefix = content[max(0, start_pos-500):start_pos]
            decorators = [d[0] for d in self.DECORATOR_PATTERN.findall(prefix[-200:])]
            line_start = content[:start_pos].count('\n') + 1
            line_end = self._find_block_end(content, match.end())
            
            result['classes'].append({
                'name': match.group(1), 'extends': match.group(2),
                'implements': match.group(3), 'decorators': decorators,
                'line_start': line_start, 'line_end': line_end, 'type': 'class'
            })
        
        for match in self.FUNCTION_PATTERN.finditer(content):
            line_start = content[:match.start()].count('\n') + 1
            line_end = self._find_block_end(content, match.end())
            result['functions'].append({
                'name': match.group(1), 'params': match.group(2),
                'return_type': match.group(3), 'line_start': line_start,
                'line_end': line_end, 'type': 'function'
            })
        return result
    
    def _find_block_end(self, content: str, start: int) -> int:
        brace_count, pos = 1, start
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        return content[:pos].count('\n') + 1
    
    def create_searchable_chunks(self, parsed_data: Dict, content: str) -> List[Dict]:
        chunks = []
        lines = content.split('\n')
        file_path = parsed_data['file_path']
        
        for cls in parsed_data['classes']:
            code = '\n'.join(lines[max(0, cls['line_start']-1):cls['line_end']])[:2000]
            
            # Determine component type from file name
            component_type = "Angular Component" if "component" in file_path.lower() else \
                           "Angular Service" if "service" in file_path.lower() else \
                           "Angular Module" if "module" in file_path.lower() else "TypeScript Class"
            
            text = f"{component_type}: {cls['name']}\nFile: {file_path}\n"
            if cls['decorators']:
                text += f"Decorators: @{', @'.join(cls['decorators'])}\n"
            if cls['extends']:
                text += f"Extends: {cls['extends']}\n"
            if cls['implements']:
                text += f"Implements: {cls['implements']}\n"
            text += f"Code:\n{code}"
            
            chunks.append({
                'type': 'class', 'name': cls['name'], 'file': file_path,
                'line_start': cls['line_start'], 'line_end': cls['line_end'],
                'text': text, 'code': code, 'docstring': None,
                'metadata': {'decorators': cls['decorators'], 'extends': cls['extends'],
                            'implements': cls['implements'], 'language': 'typescript'}
            })
        
        for func in parsed_data['functions']:
            code = '\n'.join(lines[max(0, func['line_start']-1):func['line_end']])
            text = f"TypeScript Function: {func['name']}\nFile: {file_path}\n"
            text += f"Parameters: {func['params']}\n"
            if func['return_type']:
                text += f"Returns: {func['return_type']}\n"
            text += f"Code:\n{code}"
            
            chunks.append({
                'type': 'function', 'name': func['name'], 'file': file_path,
                'line_start': func['line_start'], 'line_end': func['line_end'],
                'text': text, 'code': code, 'docstring': None,
                'metadata': {'params': func['params'], 'return_type': func['return_type'],
                            'language': 'typescript'}
            })
        return chunks


class MultiLanguageParser:
    """Unified parser that handles multiple languages"""
    
    def __init__(self):
        self.python_parser = PythonCodeParser()
        self.java_parser = JavaCodeParser()
        self.ts_parser = TypeScriptCodeParser()
    
    def parse_file(self, content: str, file_path: str) -> Dict:
        ext = file_path.split('.')[-1].lower()
        if ext == 'py':
            return self.python_parser.parse_file(content, file_path)
        elif ext == 'java':
            return self.java_parser.parse_file(content, file_path)
        elif ext in ('ts', 'tsx', 'js', 'jsx'):
            return self.ts_parser.parse_file(content, file_path)
        else:
            logger.warning(f"No parser for {ext} files, using text chunking")
            return self._text_chunk(content, file_path)
    
    def create_searchable_chunks(self, parsed_data: Dict, content: str) -> List[Dict]:
        file_path = parsed_data['file_path']
        ext = file_path.split('.')[-1].lower()
        
        if ext == 'py':
            chunks = self.python_parser.create_searchable_chunks(parsed_data, content)
        elif ext == 'java':
            chunks = self.java_parser.create_searchable_chunks(parsed_data, content)
        elif ext in ('ts', 'tsx', 'js', 'jsx'):
            chunks = self.ts_parser.create_searchable_chunks(parsed_data, content)
        else:
            chunks = parsed_data.get('chunks', [])
        
        # Fallback: if no chunks extracted, use whole file
        if not chunks and content.strip():
            chunks.append({
                'type': 'file', 'name': file_path.split('/')[-1], 'file': file_path,
                'line_start': 1, 'line_end': len(content.split('\n')),
                'text': f"File: {file_path}\n{content[:2000]}",
                'code': content[:2000], 'docstring': None,
                'metadata': {'language': ext}
            })
        return chunks
    
    def _text_chunk(self, content: str, file_path: str, chunk_size: int = 1000) -> Dict:
        chunks = []
        lines = content.split('\n')
        current_chunk, current_start, current_size = [], 1, 0
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size >= chunk_size:
                code = '\n'.join(current_chunk)
                chunks.append({
                    'type': 'text_chunk', 'name': f"chunk_{current_start}_{i}",
                    'file': file_path, 'line_start': current_start, 'line_end': i,
                    'text': f"File: {file_path}\nLines: {current_start}-{i}\n{code}",
                    'code': code, 'docstring': None, 'metadata': {'language': 'unknown'}
                })
                current_chunk, current_start, current_size = [], i + 1, 0
        
        if current_chunk:
            code = '\n'.join(current_chunk)
            chunks.append({
                'type': 'text_chunk', 'name': f"chunk_{current_start}_{len(lines)}",
                'file': file_path, 'line_start': current_start, 'line_end': len(lines),
                'text': f"File: {file_path}\nLines: {current_start}-{len(lines)}\n{code}",
                'code': code, 'docstring': None, 'metadata': {'language': 'unknown'}
            })
        
        return {'file_path': file_path, 'chunks': chunks, 'functions': [], 'classes': []}


if __name__ == "__main__":
    parser = MultiLanguageParser()
    
    # Test Java
    java_code = """
@RestController
@RequestMapping("/api/employees")
public class EmployeeController {
    @GetMapping
    public List<Employee> getAllEmployees() {
        return repository.findAll();
    }
}
"""
    parsed = parser.parse_file(java_code, "EmployeeController.java")
    chunks = parser.create_searchable_chunks(parsed, java_code)
    print(f"Java: {len(chunks)} chunks - {[c['name'] for c in chunks]}")
    
    # Test TypeScript
    ts_code = """
@Component({selector: 'app-employee-list'})
export class EmployeeListComponent implements OnInit {
    ngOnInit(): void { this.getEmployees(); }
}
"""
    parsed = parser.parse_file(ts_code, "employee-list.component.ts")
    chunks = parser.create_searchable_chunks(parsed, ts_code)
    print(f"TypeScript: {len(chunks)} chunks - {[c['name'] for c in chunks]}")