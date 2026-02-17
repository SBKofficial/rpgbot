import os, subprocess, shutil, json, logging

class LabEngine:
    def __init__(self, root_dir="bot_lab"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        # Ensure your GIT_TOKEN is set in your environment variables
        self.git_token = os.getenv("GIT_TOKEN")
        
        # --- CONFIGURATION: YOUR MAIN REPO ---
        # Replace this with your actual repository URL
        self.main_repo_url = "https://github.com/SBKofficial/rpgbot.git"

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

    def git_push_file(self, uid, filename):
        """
        Connects the local user folder to the main repo and pushes 
        the file to a user-specific branch.
        """
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        
        try:
            # 1. Initialize local git if it doesn't exist
            if not os.path.exists(os.path.join(path, ".git")):
                subprocess.run(["git", "init"], cwd=path)

            # 2. Add 'origin' pointing to your main repo with PAT authentication
            remotes = subprocess.run(["git", "remote"], cwd=path, capture_output=True, text=True).stdout
            if "origin" not in remotes:
                if self.git_token:
                    # Authenticate the URL so it doesn't ask for a password
                    auth_url = self.main_repo_url.replace("https://", f"https://{self.git_token}@")
                else:
                    auth_url = self.main_repo_url
                
                subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=path)

            # 3. Identity setup (required for commits)
            subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=path)
            subprocess.run('git config user.name "LabManager"', shell=True, cwd=path)
            
            # 4. Create and switch to the user's unique branch
            # We use checkout -B to force create/reset to a clean branch state
            subprocess.run(["git", "checkout", "-B", branch], cwd=path, capture_output=True)
            
            # 5. Stage, Commit, and Push
            subprocess.run(["git", "add", filename], cwd=path)
            subprocess.run(["git", "commit", "-m", f"Upload from user {uid}: {filename}"], cwd=path)
            
            # -u origin {branch} sets the upstream so future pushes are easier
            res = subprocess.run(["git", "push", "-u", "origin", branch, "--force"], cwd=path, capture_output=True, text=True)
            
            if res.returncode == 0:
                return True, "Successfully pushed to branch"
            else:
                return False, res.stderr

        except Exception as e:
            return False, str(e)
