import os, subprocess, shutil, json, logging, re

class LabEngine:
    def __init__(self, root_dir="bot_lab"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.git_token = os.getenv("GIT_TOKEN")

    def get_user_base(self, uid):
        path = os.path.join(self.root_dir, str(uid))
        os.makedirs(path, exist_ok=True)
        return path

    def setup_venv(self, uid):
        user_path = self.get_user_base(uid)
        venv_path = os.path.join(user_path, "venv")
        if not os.path.exists(venv_path):
            subprocess.run(["python3", "-m", "venv", venv_path], check=True)
        return venv_path

    def get_venv_exe(self, uid):
        return os.path.join(self.get_user_base(uid), "venv", "bin", "python3")

    def read_config(self, uid):
        path = os.path.join(self.get_user_base(uid), "bot.json")
        try:
            if os.path.exists(path):
                with open(path, "r") as f: return json.load(f)
        except: return None
        return None

    def save_config(self, uid, config):
        path = os.path.join(self.get_user_base(uid), "bot.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def get_config_template(self):
        return {
            "name": "my-bot",
            "start_cmd": "python3 main.py",
            "auto_deploy": True
        }

    def connect_repo(self, uid, repo_url):
        user_path = self.get_user_base(uid)
        branch_name = f"user_{uid}"
        if self.git_token and "github.com" in repo_url:
            repo_url = repo_url.replace("https://", f"https://{self.git_token}@")
        try:
            for item in os.listdir(user_path):
                if item != "venv":
                    p = os.path.join(user_path, item)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            subprocess.run(["git", "clone", repo_url, "."], cwd=user_path, check=True)
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=user_path, capture_output=True)
            return True, branch_name
        except Exception as e:
            return False, str(e)

    def git_push(self, uid, msg):
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        try:
            subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=path)
            subprocess.run('git config user.name "BotLabManager"', shell=True, cwd=path)
            subprocess.run("git add .", shell=True, cwd=path)
            # Check if there are changes to commit
            check = subprocess.run("git status --porcelain", shell=True, cwd=path, capture_output=True, text=True)
            if not check.stdout.strip(): return True
            subprocess.run(f'git commit -m "{msg}"', shell=True, cwd=path)
            res = subprocess.run(f"git push origin {branch}", shell=True, cwd=path, capture_output=True)
            return res.returncode == 0
        except: return False

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
        try:
            old_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
            subprocess.run(["git", "pull", "origin", branch], cwd=path, capture_output=True)
            return True, old_hash
        except: return False, None
