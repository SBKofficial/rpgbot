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

    def get_config_template(self):
        """The specific template we discussed for copy-pasting."""
        return {
            "name": "my-bot",
            "entry_point": "main.py",
            "install_cmd": "pip install -r requirements.txt",
            "start_cmd": "python3 -u main.py",
            "auto_deploy": True
        }

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

    def connect_repo(self, uid, repo_url):
        """Initializes a user's isolated workspace with a specific repo."""
        user_path = self.get_user_base(uid)
        branch_name = f"user_{uid}"
        
        # Security: Use the GIT_TOKEN for the clone if provided
        if self.git_token and "github.com" in repo_url:
            repo_url = repo_url.replace("https://", f"https://{self.git_token}@")

        try:
            # 1. Clean old files if they exist (except venv)
            for item in os.listdir(user_path):
                if item != "venv":
                    path = os.path.join(user_path, item)
                    shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            
            # 2. Clone into the directory
            subprocess.run(["git", "clone", repo_url, "."], cwd=user_path, check=True)
            
            # 3. Create or switch to the private user branch
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=user_path, capture_output=True)
            subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=user_path, capture_output=True)
            
            return True, branch_name
        except Exception as e:
            return False, str(e)

