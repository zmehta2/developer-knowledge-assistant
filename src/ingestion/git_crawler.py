"""
src/ingestion/git_crawler.py

Clone and process Git repositories
"""
import os
import git
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitCrawler:
    """Clone and extract files from Git repositories"""
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def clone_repo(self, repo_url: str, repo_name: str = None) -> Path:
        """
        Clone a repository
        
        Args:
            repo_url: GitHub repository URL
            repo_name: Optional name for local folder
            
        Returns:
            Path to cloned repository
        """
        if repo_name is None:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            
        repo_path = self.data_dir / repo_name
        
        if repo_path.exists():
            logger.info(f"Repository {repo_name} already exists, pulling latest")
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            logger.info(f"Cloning {repo_url} to {repo_path}")
            git.Repo.clone_from(repo_url, repo_path)
            
        return repo_path
    
    def get_code_files(self, repo_path: Path, 
                       extensions: List[str] = None) -> List[Dict]:
        """
        Get all code files from repository
        
        Args:
            repo_path: Path to repository
            extensions: File extensions to include (e.g., ['.py', '.js'])
            
        Returns:
            List of dicts with file info
        """
        if extensions is None:
            extensions = ['.py', '.js', '.java', '.ts', '.go']
            
        code_files = []
        
        # Directories to ignore
        ignore_dirs = {
            '.git', 'node_modules', '__pycache__', 
            'venv', 'env', 'dist', 'build', 
            '.pytest_cache', 'coverage'
        }
        
        for root, dirs, files in os.walk(repo_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        code_files.append({
                            'path': str(file_path.relative_to(repo_path)),
                            'full_path': str(file_path),
                            'extension': file_path.suffix,
                            'content': content,
                            'size': len(content),
                            'repo': repo_path.name
                        })
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")
                        
        logger.info(f"Found {len(code_files)} code files in {repo_path.name}")
        return code_files


if __name__ == "__main__":
    crawler = GitCrawler()
    
    test_repos = [
        "https://github.com/zmehta2/Spring-Boot-Angular-8-CRUD-Example.git",
    ]
    
    for repo_url in test_repos:
        repo_path = crawler.clone_repo(repo_url)
        files = crawler.get_code_files(repo_path, extensions=['.py'])
        
        print(f"\n{repo_path.name}:")
        print(f"  Total Python files: {len(files)}")
        print(f"  Total lines: {sum(f['size'] for f in files)}")
        print(f"  Sample files:")
        for f in files[:5]:
            print(f"    - {f['path']}")