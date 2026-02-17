import os, subprocess, shutil, json, logging, signal

class LabEngine:
    def __init__(self, root_dir="bot_lab"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.git_token = os.getenv("GIT_TOKEN")
        self.main_repo_url = "https://github.com/SBKofficial/rpgbot.git"

    def get_user_base(self, uid):
        path = os.path.join(self.root_dir, str(uid))
        os.makedirs(path, exist_ok=True)
        return path

    def sync_from_github(self, uid):
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        auth_url = self.main_repo_url.replace("https://", f"https://{self.git_token}@") if self.git_token else self.main_repo_url
        try:
            if not os.path.exists(os.path.join(path, ".git")):
                res = subprocess.run(["git", "clone", "-b", branch, auth_url, "."], cwd=path, capture_output=True, text=True)
                if res.returncode != 0:
                    subprocess.run(["git", "init"], cwd=path)
                    subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=path)
                    subprocess.run(["git", "checkout", "-b", branch], cwd=path)
                    return True, "🆕 New workspace initialized."
                return True, "✅ Files recovered from GitHub."
            else:
                subprocess.run(["git", "pull", "origin", branch], cwd=path)
                return True, "🔄 Synced with cloud."
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
        path = self.get_user_base(uid)
        branch = f"user_{uid}"
        try:
            subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=path)
            subprocess.run('git config user.name "LabManager"', shell=True, cwd=path)
            subprocess.run(["git", "add", filename], cwd=path)
            subprocess.run(["git", "commit", "-m", f"Update {filename}", "--allow-empty"], cwd=path)
            res = subprocess.run(["git", "push", "origin", branch], cwd=path, capture_output=True, text=True)
            return (True, "Synced.") if res.returncode == 0 else (False, res.stderr)
        except Exception as e:
            return False, str(e)

    # --- NEW: PERSISTENT PROCESS EXECUTION ---
    def start_subprocess(self, cmd, user_path, log_path):
        """Starts a process in a new session to allow clean termination of children."""
        return subprocess.Popen(
            cmd, 
            shell=True, 
            cwd=user_path, 
            stdout=open(log_path, "w"), 
            stderr=subprocess.STDOUT, 
            stdin=subprocess.PIPE, 
            text=True, 
            bufsize=1, # Line buffering for real-time logs
            preexec_fn=os.setsid # Create process group
        )

    def kill_subprocess(self, proc):
        """Kills the entire process group (parent + all children)."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return True
        except:
            return False
