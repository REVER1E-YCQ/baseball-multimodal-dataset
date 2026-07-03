"""
GitHub auto-commit and push for new dataset samples.

Safety rules:
  - Never force push
  - Only push to configured personal branch (zichen-yang)
  - Verify branch before commit
"""

import subprocess
import logging
from pathlib import Path

from src.config import AppConfig

logger = logging.getLogger(__name__)


class GitHubSync:
    """Handles git add, commit, push for new samples."""

    def __init__(self, config: AppConfig):
        self.cfg = config.git
        self.repo_root = Path(__file__).resolve().parent.parent

    def sync_sample(self, sample_dir: Path, sample_id: str,
                    label: str) -> bool:
        """Stage, commit, push a single sample directory.

        Returns True on success.
        """
        collector = "Zichen_Yang"  # From pipeline config

        # Verify branch
        if not self._on_correct_branch():
            logger.error("Not on branch '%s'. Skipping sync.", self.cfg.branch)
            return False

        # Stage the sample directory + any source code changes
        self._run_git(["add", str(sample_dir)])

        # Commit
        commit_msg = self.cfg.commit_message_template.format(
            collector=collector, label=label, sample_id=sample_id
        )
        if not self._commit(commit_msg):
            return False

        # Push
        if self.cfg.auto_push:
            if not self._push():
                return False

        logger.info("Synced %s to %s/%s", sample_id, self.cfg.remote, self.cfg.branch)
        return True

    def _on_correct_branch(self) -> bool:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=self.repo_root
        )
        current = result.stdout.strip()
        if current != self.cfg.branch:
            logger.warning(
                "Current branch '%s' != configured '%s'. Run: git checkout %s",
                current, self.cfg.branch, self.cfg.branch
            )
            return False
        return True

    def _run_git(self, args: list) -> bool:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, cwd=self.repo_root
        )
        return result.returncode == 0

    def _commit(self, message: str) -> bool:
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True, cwd=self.repo_root
        )
        if result.returncode == 0:
            return True
        if "nothing to commit" in (result.stdout + result.stderr):
            logger.info("Nothing to commit (already synced?)")
            return True
        logger.error("Commit failed: %s", result.stderr.strip())
        return False

    def _push(self) -> bool:
        result = subprocess.run(
            ["git", "push", self.cfg.remote, self.cfg.branch],
            capture_output=True, text=True, cwd=self.repo_root
        )
        if result.returncode != 0:
            logger.error("Push failed: %s", result.stderr.strip())
            return False
        return True
