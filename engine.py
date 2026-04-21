import os, subprocess, shutil, json, logging, signal, resource

class LabEngine:
    def __init__(self, root_dir="users_files"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.git_token = os.getenv("GIT_TOKEN")
        self.main_repo_url = "https://github.com/SBKofficial/rpgbot.git"

    def get_user_root(self, uid):
        """The base folder for the user. Git and Venv live HERE."""
        path = os.path.join(self.root_dir, str(uid))
        os.makedirs(path, exist_ok=True)
        return path

    def get_project_path(self, uid, project_name="default"):
        """The actual folder where the specific bot files live."""
        path = os.path.join(self.get_user_root(uid), project_name)
        os.makedirs(path, exist_ok=True)
        return path

    def sync_from_github(self, uid):
        user_root = self.get_user_root(uid)
        branch = f"user_{uid}"
        auth_url = self.main_repo_url.replace("https://", f"https://{self.git_token}@") if self.git_token else self.main_repo_url
        try:
            if not os.path.exists(os.path.join(user_root, ".git")):
                res = subprocess.run(["git", "clone", "-b", branch, auth_url, "."], cwd=user_root, capture_output=True, text=True)
                if res.returncode != 0:
                    subprocess.run(["git", "init"], cwd=user_root)
                    subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=user_root)
                    subprocess.run(["git", "checkout", "-b", branch], cwd=user_root)
                    return True, "🆕 New workspace initialized."
                return True, "✅ Files recovered from GitHub."
            else:
                subprocess.run(["git", "pull", "origin", branch], cwd=user_root)
                return True, "🔄 Synced with cloud."
        except Exception as e:
            return False, f"Sync Error: {str(e)}"

    def setup_venv(self, uid):
        user_root = self.get_user_root(uid)
        venv_path = os.path.join(user_root, "venv")
        if not os.path.exists(venv_path):
            subprocess.run(["python3", "-m", "venv", venv_path], check=True)
        return venv_path

    def get_venv_exe(self, uid):
        return os.path.join(self.get_user_root(uid), "venv", "bin", "python3")

    def read_config(self, uid, project_name):
        path = os.path.join(self.get_project_path(uid, project_name), "bot.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f: return json.load(f)
            except: return None
        return None

    def git_push_file(self, uid, project_name, filename):
        user_root = self.get_user_root(uid)
        branch = f"user_{uid}"
        git_target = os.path.join(project_name, filename) # The relative path Git needs
        try:
            subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=user_root)
            subprocess.run('git config user.name "LabManager"', shell=True, cwd=user_root)
            subprocess.run(["git", "add", git_target], cwd=user_root)
            subprocess.run(["git", "commit", "-m", f"Update {git_target}", "--allow-empty"], cwd=user_root)
            res = subprocess.run(["git", "push", "origin", branch], cwd=user_root, capture_output=True, text=True)
            return (True, "Synced.") if res.returncode == 0 else (False, res.stderr)
        except Exception as e:
            return False, str(e)

    def _apply_limits(self):
        mem_limit = 512 * 1024 * 1024 
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        file_limit = 50 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        os.nice(19)
        os.setsid()

    def start_subprocess(self, cmd, cwd_path, log_path, user_root):
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": user_root,
            "LANG": "en_US.UTF-8",
            "PYTHONUNBUFFERED": "1"
        }
        return subprocess.Popen(
            cmd, shell=True, cwd=cwd_path, stdout=open(log_path, "w"), 
            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, 
            env=safe_env, preexec_fn=self._apply_limits, text=True, bufsize=1
        )

    def kill_subprocess(self, proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return True
        except:
            return False

    def git_delete_file(self, uid, project_name, filename):
        user_root = self.get_user_root(uid)
        branch = f"user_{uid}"
        git_target = os.path.join(project_name, filename)
        try:
            subprocess.run(["git", "rm", git_target], cwd=user_root)
            subprocess.run(["git", "commit", "-m", f"Delete {git_target}"], cwd=user_root)
            res = subprocess.run(["git", "push", "origin", branch], cwd=user_root, capture_output=True, text=True)
            return True, "Permanently removed from cloud."
        except Exception as e:
            return False, str(e)
