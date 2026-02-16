import os, subprocess, shutil, json, logging, re

class LabEngine:
    def __init__(self, root_dir="bot_lab"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.git_token = os.getenv("GIT_TOKEN")
        self.repo_url = f"https://{self.git_token}@github.com/SBKofficial/rpgbot.git" if self.git_token else None

    def get_user_base(self, uid):
        path = os.path.join(self.root_dir, str(uid))
        os.makedirs(path, exist_ok=True)
        return path

    def setup_venv(self, uid):
        """Creates an isolated virtual environment."""
        user_path = self.get_user_base(uid)
        venv_path = os.path.join(user_path, "venv")
        if not os.path.exists(venv_path):
            subprocess.run(["python3", "-m", "venv", venv_path], check=True)
        return venv_path

    def get_venv_exe(self, uid):
        """Returns the path to the python executable inside the venv."""
        return os.path.join(self.get_user_base(uid), "venv", "bin", "python3")

    def read_config(self, uid):
        path = os.path.join(self.get_user_base(uid), "bot.json")
        try:
            if os.path.exists(path):
                with open(path, "r") as f: return json.load(f)
        except: return None
        return None

    def get_formatted_logs(self, uid, pid):
        path = os.path.join(self.get_user_base(uid), f"{pid}.log")
        if not os.path.exists(path): return r"⚠️ _No logs available yet\._"
        try:
            with open(path, "r") as f:
                lines = f.readlines()[-15:]
                # Using your original escape_md logic in the main file
                return lines
        except: return []

    def git_poll_update(self, uid):
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        if not os.path.exists(os.path.join(path, ".git")): return False
        try:
            subprocess.run(["git", "fetch"], cwd=path, capture_output=True)
            local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
            remote = subprocess.check_output(["git", "rev-parse", f"origin/{branch}"], cwd=path).decode().strip()
            return local != remote
        except: return False

    def deploy_pull(self, uid):
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        old_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        res = subprocess.run(["git", "pull", "origin", branch], cwd=path, capture_output=True)
        return res.returncode == 0, old_hash

    def rollback(self, uid, target_hash):
        subprocess.run(["git", "reset", "--hard", target_hash], cwd=self.get_user_base(uid))
