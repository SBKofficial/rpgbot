import os, subprocess, shutil, json, logging, re

class LabEngine:
    def __init__(self, root_dir="bot_lab"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.repo_url = f"https://{os.getenv('GIT_TOKEN')}@github.com/SBKofficial/rpgbot.git"

    def get_user_path(self, uid):
        path = os.path.join(self.root_dir, str(uid))
        os.makedirs(path, exist_ok=True)
        return path

    def setup_venv(self, uid):
        user_path = self.get_user_path(uid)
        venv_path = os.path.join(user_path, "venv")
        if not os.path.exists(venv_path):
            subprocess.run(["python3", "-m", "venv", venv_path], check=True)
        return venv_path

    def get_venv_exe(self, uid):
        return os.path.join(self.get_user_path(uid), "venv", "bin", "python3")

    def read_config(self, uid):
        path = os.path.join(self.get_user_path(uid), "bot.json")
        try:
            if os.path.exists(path):
                with open(path, "r") as f: return json.load(f)
        except: return None
        return None

    def get_logs(self, uid, slug, lines=15):
        path = os.path.join(self.get_user_path(uid), f"{slug}.log")
        if not os.path.exists(path): return "No logs available."
        with open(path, "r") as f:
            content = f.readlines()[-lines:]
            return "".join(content) if content else "Log is empty."

    def git_poll_update(self, uid):
        path = self.get_user_path(uid)
        branch = f"user_{uid}"
        if not os.path.exists(os.path.join(path, ".git")): return False
        subprocess.run(["git", "fetch"], cwd=path, capture_output=True)
        local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        remote = subprocess.check_output(["git", "rev-parse", f"origin/{branch}"], cwd=path).decode().strip()
        return local != remote

    def deploy_pull(self, uid):
        path = self.get_user_path(uid)
        old_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        res = subprocess.run(["git", "pull", "origin", f"user_{uid}"], cwd=path, capture_output=True)
        return res.returncode == 0, old_hash

    def rollback(self, uid, target_hash):
        subprocess.run(["git", "reset", "--hard", target_hash], cwd=self.get_user_path(uid))
