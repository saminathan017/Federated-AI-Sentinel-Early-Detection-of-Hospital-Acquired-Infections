"""
Automatic Git commit and push on file changes.

Watches the repository for changes and automatically commits and pushes
to the configured remote branch. Useful for continuous deployment during development.

SECURITY WARNING: Only use in development environments with proper access controls.
Never use with credentials stored in the repository.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from git import Repo
from git.exc import GitCommandError
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class GitAutoPusher(FileSystemEventHandler):
    """Watch for file changes and auto-commit to Git."""

    def __init__(
        self,
        repo_path: Path,
        branch: str = "main",
        remote: str = "origin",
        debounce_seconds: int = 10,
        run_tests: bool = True,
    ):
        """
        Initialize autopusher.
        
        Args:
            repo_path: Path to Git repository
            branch: Branch to push to
            remote: Remote name
            debounce_seconds: Wait this long after last change before committing
            run_tests: Run lint and tests before committing
        """
        self.repo_path = repo_path
        self.branch = branch
        self.remote = remote
        self.debounce_seconds = debounce_seconds
        self.run_tests = run_tests
        
        self.repo = Repo(repo_path)
        self.last_change_time = None
        self.pending_changes = False

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification event."""
        if event.is_directory:
            return
        
        # Ignore certain files
        ignored_patterns = [
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            "node_modules/",
            ".DS_Store",
            "*.pyc",
            "*.log",
        ]
        
        path_str = str(event.src_path)
        
        if any(pattern in path_str for pattern in ignored_patterns):
            return
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] File changed: {event.src_path}")
        
        self.last_change_time = time.time()
        self.pending_changes = True

    def run_lint_and_tests(self) -> bool:
        """
        Run linters and tests before committing.
        
        Returns:
            True if all checks pass, False otherwise
        """
        print("\nRunning pre-commit checks...")
        
        try:
            # Run ruff
            print("  Checking with Ruff...")
            subprocess.run(
                ["ruff", "check", "src/", "tests/"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
            )
            
            # Run black
            print("  Checking formatting with Black...")
            subprocess.run(
                ["black", "--check", "src/", "tests/"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
            )
            
            # Run unit tests (quick ones only)
            print("  Running unit tests...")
            result = subprocess.run(
                ["pytest", "tests/unit/", "-x", "-q"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                print("  ✗ Tests failed:")
                print(result.stdout)
                return False
            
            print("  ✓ All checks passed")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Check failed: {e}")
            return False

    def generate_commit_message(self) -> str:
        """
        Generate commit message based on changed files.
        
        Returns:
            Commit message string
        """
        # Get list of changed files
        changed_files = [item.a_path for item in self.repo.index.diff(None)]
        changed_files.extend(self.repo.untracked_files)
        
        if not changed_files:
            return "chore: minor updates"
        
        # Categorize changes
        categories = {
            "feat": [],
            "fix": [],
            "docs": [],
            "test": [],
            "infra": [],
            "ui": [],
            "chore": [],
        }
        
        for file_path in changed_files:
            if file_path.startswith("src/modeling/") or file_path.startswith("src/federated/"):
                categories["feat"].append(file_path)
            elif file_path.startswith("docs/"):
                categories["docs"].append(file_path)
            elif file_path.startswith("tests/"):
                categories["test"].append(file_path)
            elif file_path.startswith("src/ui/"):
                categories["ui"].append(file_path)
            elif file_path.startswith(".github/") or file_path in ["Makefile", "Dockerfile"]:
                categories["infra"].append(file_path)
            else:
                categories["chore"].append(file_path)
        
        # Find primary category
        primary_category = max(categories.items(), key=lambda x: len(x[1]))
        
        if primary_category[1]:
            category = primary_category[0]
            num_files = len(changed_files)
            
            # Generate concise message
            if num_files == 1:
                file_name = Path(changed_files[0]).name
                return f"{category}: update {file_name}"
            else:
                return f"{category}: update {num_files} files"
        
        return "chore: automated update"

    def commit_and_push(self) -> None:
        """Commit staged changes and push to remote."""
        try:
            # Check if there are changes
            if not self.repo.is_dirty() and not self.repo.untracked_files:
                print("No changes to commit")
                return
            
            # Run checks if enabled
            if self.run_tests:
                if not self.run_lint_and_tests():
                    print("\n⚠ Skipping commit due to failed checks")
                    print("Fix the issues and changes will be committed on next save\n")
                    return
            
            # Stage all changes
            self.repo.git.add(A=True)
            
            # Generate commit message
            commit_msg = self.generate_commit_message()
            
            # Commit
            self.repo.index.commit(commit_msg)
            print(f"\n✓ Committed: {commit_msg}")
            
            # Push
            print(f"Pushing to {self.remote}/{self.branch}...")
            origin = self.repo.remote(name=self.remote)
            origin.push(refspec=f"{self.branch}:{self.branch}")
            
            print(f"✓ Pushed to {self.remote}/{self.branch}\n")
        
        except GitCommandError as e:
            print(f"\n✗ Git error: {e}\n")
        except Exception as e:
            print(f"\n✗ Error: {e}\n")

    def check_and_commit(self) -> None:
        """Check if debounce period has passed and commit if needed."""
        if not self.pending_changes:
            return
        
        if self.last_change_time is None:
            return
        
        time_since_change = time.time() - self.last_change_time
        
        if time_since_change >= self.debounce_seconds:
            self.commit_and_push()
            self.pending_changes = False
            self.last_change_time = None


def main() -> None:
    """Run the autopusher."""
    parser = argparse.ArgumentParser(description="Auto-commit and push Git changes")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to Git repository",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to push to",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Remote name",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=10,
        help="Seconds to wait after last change before committing",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip running tests before committing",
    )
    
    args = parser.parse_args()
    
    # Verify it's a Git repo
    try:
        repo = Repo(args.repo)
    except Exception as e:
        print(f"ERROR: Not a valid Git repository: {e}")
        sys.exit(1)
    
    # Check if remote is configured
    try:
        repo.remote(name=args.remote)
    except Exception:
        print(f"ERROR: Remote '{args.remote}' not configured")
        print(f"\nTo add remote: git remote add {args.remote} <url>")
        sys.exit(1)
    
    print("=" * 70)
    print("FEDERATED AI SENTINEL - AUTOPUSH")
    print("=" * 70)
    print(f"\nRepository: {args.repo}")
    print(f"Remote: {args.remote}")
    print(f"Branch: {args.branch}")
    print(f"Debounce: {args.debounce} seconds")
    print(f"Run tests: {not args.no_tests}")
    print("\nWatching for changes... (Ctrl+C to stop)\n")
    
    # Create event handler
    handler = GitAutoPusher(
        repo_path=args.repo,
        branch=args.branch,
        remote=args.remote,
        debounce_seconds=args.debounce,
        run_tests=not args.no_tests,
    )
    
    # Create observer
    observer = Observer()
    observer.schedule(handler, str(args.repo), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
            handler.check_and_commit()
    except KeyboardInterrupt:
        print("\n\nStopping autopush...")
        observer.stop()
    
    observer.join()
    print("✓ Autopush stopped\n")


if __name__ == "__main__":
    main()

