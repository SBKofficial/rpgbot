import os, subprocess, shutil, json, logging

class LabEngine:
    def __init__(self, root_dir="bot_lab"):
        # We use absolute paths to ensure consistency across restarts
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        # Ensure your GIT_TOKEN is set as an Environment Variable
        self.git_token = os.getenv("GIT_TOKEN")
        # Your Master Repository
        self.main_repo_url = "https://github.com/SBKofficial/rpgbot.git"

    def get_user_base(self, uid):
        """Returns the absolute path to a user's workspace."""
        path = os.path.join(self.root_dir, str(uid))
        os.makedirs(path, exist_ok=True)
        return path

    def sync_from_github(self, uid):
        """
        CRITICAL: Recovers files from GitHub. 
        Checks if the user's branch exists and clones/pulls it.
        """
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        
        # Build Authenticated URL
        if self.git_token:
            auth_url = self.main_repo_url.replace("https://", f"https://{self.git_token}@")
        else:
            auth_url = self.main_repo_url

        try:
            # If .git doesn't exist, we need to clone or initialize
            if not os.path.exists(os.path.join(path, ".git")):
                # Attempt to clone the specific user branch
                res = subprocess.run(
                    ["git", "clone", "-b", branch, auth_url, "."], 
                    cwd=path, capture_output=True, text=True
                )
                
                if res.returncode != 0:
                    # Branch doesn't exist yet (New User), so initialize locally
                    subprocess.run(["git", "init"], cwd=path)
                    subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=path)
                    subprocess.run(["git", "checkout", "-b", branch], cwd=path)
                    return True, "🆕 New workspace initialized."
                return True, "✅ Files recovered from GitHub."
            
            else:
                # Local .git exists, just pull latest updates
                subprocess.run(["git", "pull", "origin", branch], cwd=path)
                return True, "🔄 Synced with latest cloud changes."
                
        except Exception as e:
            return False, f"Sync Error: {str(e)}"

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
        if os.path.exists(path):
            try:
                with open(path, "r") as f: return json.load(f)
            except: return None
        return None

    def git_push_file(self, uid, filename):
        """Saves a file locally, then pushes it to the user's branch."""
        path = self.get_user_base(uid)
        branch = f"user_{uid}"

        try:
            # 1. Identity setup (required after server restart)
            subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=path)
            subprocess.run('git config user.name "LabManager"', shell=True, cwd=path)

            # 2. Commit logic
            subprocess.run(["git", "add", filename], cwd=path)
            # --allow-empty prevents errors if nothing changed
            subprocess.run(["git", "commit", "-m", f"Update {filename}", "--allow-empty"], cwd=path)

            # 3. Push to GitHub
            res = subprocess.run(["git", "push", "origin", branch], cwd=path, capture_output=True, text=True)

            if res.returncode == 0:
                return True, "Cloud Sync Successful"
            else:
                return False, res.stderr

        except Exception as e:
            return False, str(e)
