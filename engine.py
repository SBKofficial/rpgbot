import os, subprocess, shutil, json, logging

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

    def git_push_file(self, uid, filename, repo_url=None):
        """Saves file locally and pushes to the user's branch on GitHub."""
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        
        try:
            # Init git if folder is fresh
            if not os.path.exists(os.path.join(path, ".git")):
                subprocess.run(["git", "init"], cwd=path)
                if repo_url:
                    if self.git_token and "github.com" in repo_url:
                        repo_url = repo_url.replace("https://", f"https://{self.git_token}@")
                    subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=path)

            subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=path)
            subprocess.run('git config user.name "LabManager"', shell=True, cwd=path)
            
            # Ensure we are on the user's branch
            subprocess.run(["git", "checkout", "-b", branch], cwd=path, capture_output=True)
            
            # Add, Commit, Push
            subprocess.run(["git", "add", filename], cwd=path)
            subprocess.run(["git", "commit", "-m", f"Auto-upload: {filename}"], cwd=path)
            res = subprocess.run(["git", "push", "origin", branch], cwd=path, capture_output=True, text=True)
            return res.returncode == 0, res.stderr
        except Exception as e: return False, str(e)
