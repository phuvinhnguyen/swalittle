"""
Clean, extensible reinforcement-style environment for SWE-bench.
"""

from __future__ import annotations
import subprocess, atexit, random, os, shutil, re, textwrap
from typing import Dict, List, Tuple, Any
from pathlib import Path
from difflib import SequenceMatcher

from datasets import load_dataset
from torch.utils.data import Dataset

# ------------------------------------------------------------------
# 1. Helpers
# ------------------------------------------------------------------
def count_step(path_a, path_list):
    total_steps = 0
    
    path_a_str = str(path_a) if isinstance(path_a, Path) else path_a
    
    for path_b in path_list:
        path_b_str = str(path_b) if isinstance(path_b, Path) else path_b
        
        a = path_a_str.rstrip('/') + '/'
        b = path_b_str.rstrip('/') + '/'
        
        common_prefix = os.path.commonprefix([a, b])
        common_depth = common_prefix.strip('/').count('/') + 1 if common_prefix else 0
        
        steps = (a.count('/') - common_depth) + (b.count('/') - common_depth)
        total_steps += steps
    
    return total_steps

def extract_edited_files(diff_text: str, project_root: str | Path = ".") -> set[Path]:
    """
    Trả về tập các Path tuyệt đối của các file được edit trong diff_text.
    project_root: thư mục gốc của repo (để nối đường dẫn cho đầy đủ).
    """
    if not diff_text:
        return set()

    # Regex khớp dòng: diff --git a/path b/path
    pattern = re.compile(r'^diff --git a/(.+?) b/\1', re.MULTILINE)
    matches = pattern.findall(diff_text)

    root = Path(project_root).resolve()
    files = {root / m for m in matches}
    return files

def is_valid_patch(diff_text: str) -> bool:
    if not diff_text:
        return False
    # Look for the unified diff header pattern
    header_pattern = r'^diff --git a/.* b/.*\nindex .*'
    return bool(re.search(header_pattern, diff_text, flags=re.MULTILINE))

def extract_answer(text: str) -> Tuple[str, str]:
    """
    Extract content from the last <plan> and <execute> tags in the text.
    Returns tuple with (plan_content, execute_content) with specific error messages.
    Handles cases where only execute tag exists.
    """
    # Handle empty/non-string input
    if not isinstance(text, str) or not text.strip():
        return ("your answer is empty or not a string", "your answer is empty or not a string")

    # Initialize default error messages
    plan_msg = "missing <plan> tag"
    execute_msg = "missing <execute> tag"

    # Check for execute tag independently
    execute_matches = re.findall(r"<execute>(.*?)</execute>", text, re.DOTALL)
    has_execute = bool(execute_matches)
    
    # Check for plan tag independently
    plan_matches = re.findall(r"<plan>(.*?)</plan>", text, re.DOTALL)
    has_plan = bool(plan_matches)

    # Case 1: Only execute exists
    if has_execute and not has_plan:
        execute_content = execute_matches[-1].strip()
        return (
            "missing <plan> tag",
            execute_content if execute_content else "empty <execute> command"
        )

    # Case 2: Find complete plan-execute pairs
    try:
        matches = re.findall(
            r"<plan>(.*?)</plan>\s*<execute>(.*?)</execute>",
            text,
            re.DOTALL
        )
        if matches:
            last_plan, last_execute = matches[-1][0].strip(), matches[-1][1].strip()
            return (
                last_plan if last_plan else "empty <plan> content",
                last_execute if last_execute else "empty <execute> command"
            )
    except Exception:
        pass

    # Case 3: Handle malformed/missing tags
    return (
        "malformed <plan> tag" if "<plan>" in text or "</plan>" in text else plan_msg,
        "malformed <execute> tag" if "<execute>" in text or "</execute>" in text else execute_msg
    )

def remove_comments(patch: str) -> str:
    """Naive strip of single-line comments in Python/C/Java/JS/Go."""
    # Python / C / Java / JS / Go single-line comments
    lines = patch.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        if "/*" in line and "*/" in line:      # single-line /* … */
            line = re.sub(r"/\*.*?\*/", "", line)
        cleaned.append(line)
    return "\n".join(cleaned)

def patch_similarity(patch_a: str, patch_b: str) -> float:
    a, b = remove_comments(patch_a), remove_comments(patch_b)
    return SequenceMatcher(None, a, b).ratio()

# ------------------------------------------------------------------
# 2. Dataset wrapper
# ------------------------------------------------------------------
class SweDataset(Dataset):
    def __init__(
        self,
        data_name_or_path: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        filter_fn=None,
    ):
        self.data = load_dataset(data_name_or_path, split=split)
        if data_name_or_path.startswith("princeton-nlp/"):
            self.data = self.data.map(
                lambda x: {
                    "name": x["instance_id"].split("/")[-1],
                    "problem_statement": x["problem_statement"],
                    "github_url": f"https://github.com/{x['repo'].strip()}.git",
                    "true_patch": x["patch"],
                    "hint": x.get("hints_text", ""),
                    "base_commit": x["base_commit"],
                    "setup_commit": x.get("environment_setup_commit", ""),
                }
            )
        if filter_fn is None:
            filter_fn = lambda x: bool(x.get("problem_statement", "").strip())
        self.data = self.data.filter(filter_fn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ------------------------------------------------------------------
# 3. Container environment
# ------------------------------------------------------------------
class SweEnv:
    """
    One container instance = one SWE-bench task.
    step() returns (message, reward, terminated?, extra_info)
    """

    def __init__(
        self,
        data_name_or_path: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        sif_folder: str = "./tmp",
        use_plan: bool = True,
        base_tools_path: str | None = None,
        tool_list: list[str] | None = None,
    ):
        self.dataset = SweDataset(data_name_or_path, split)
        self.sif_folder = sif_folder
        self.base_tools_path = base_tools_path
        self.tool_list = tool_list or []
        self.use_plan = use_plan

        self.current_env: Dict[str, Any] | None = None
        self.project_dir: str | None = None
        self.history: List[Tuple[str, str]] = []
        self.proc: subprocess.Popen | None = None

        os.makedirs(self.sif_folder, exist_ok=True)
        self.reset()
        atexit.register(self.close)

    # ----------------------------------------------------------
    # Low-level container helpers
    # ----------------------------------------------------------
    def _run_command(self, cmd: str) -> str:
        """Send raw command to container and collect output."""
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError("Container is not running")
        if cmd.strip() == "exit":
            self.proc.terminate()
            return "Container terminated"
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
            output_lines = []
            while True:
                line = self.proc.stdout.readline()
                if line.strip() == "__END__":
                    break
                output_lines.append(line)
            return "".join(output_lines).strip()
        except Exception as e:
            return f"[ERROR] {e}"

    # ----------------------------------------------------------
    # Environment API
    # ----------------------------------------------------------
    def reset(self) -> str:
        """Pick a new task, spin up a fresh container, clone repo, install tools."""
        # 1. Pick task
        self.current_env = random.choice(self.dataset)
        task = self.current_env

        # 2. Build / reuse SIF
        docker_link = task.get("docker_link", "docker://ubuntu:22.04")
        sif_name = re.sub(r"[^\w]", "_", docker_link) + ".sif"
        sif_path = os.path.join(self.sif_folder, sif_name)
        if not os.path.exists(sif_path):
            subprocess.run(["apptainer", "build", sif_path, docker_link], check=True)

        # 3. Clean home mount
        if os.path.exists("/tmp/home_mount"):
            shutil.rmtree("/tmp/home_mount")
        os.makedirs("/tmp/home_mount", exist_ok=True)

        # 4. Start container
        self.close()
        binds = [f"/tmp/home_mount:/home"]
        if self.base_tools_path:
            binds.append(f"{self.base_tools_path}:/mnt/tools")
        bind_arg = ",".join(binds)

        self.proc = subprocess.Popen(
            [
                "apptainer", "exec",
                "--containall", "--cleanenv", "--no-home",
                "--pwd", "/home",
                "--bind", bind_arg,
                sif_path,
                "/bin/bash", "-c",
                'while IFS= read -r line; do eval "$line"; echo __END__; done'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 5. Clone repo and install tools
        self.project_dir = f"/home/{task['name']}"
        self.history.clear()

        # 6. Setup variables and environment
        self.edited_files = extract_edited_files(task['true_patch'], self.project_dir)
        self.diff = ''

        # self._run_command(f"git clone {task['github_url']} {self.project_dir}")
        # if os.path.exists(f"{self.project_dir}/requirements.txt"):
        #     self._run_command(f"cd {self.project_dir} && python3 -m pip install -r requirements.txt")

        for tool in self.tool_list:
            self._run_command(f"bash /mnt/tools/{tool}/install.sh")
            self._run_command(f"export PATH=$PATH:/mnt/tools/{tool}/bin")

        # 6. Return initial prompt
        return self._build_message("", "")

    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------
    def step(self, raw_cmd: str) -> Tuple[str, float, bool, Dict]:
        """
        Executes the agent command inside the project directory.
        Returns:
            message   : prompt for next step
            reward    : scalar
            done      : True if diff was produced
            extra     : dict with details
        """
        # 1. Execute
        plan, cmd = extract_answer(raw_cmd)
        output = self._run_command(cmd)
        done = cmd == 'exit'

        # 2. Check if the task is completed
        if done:
            message = "Patch submitted"
            reward = 50.0 * patch_similarity(self.diff, self.current_env["true_patch"])
        else:
            prev_location = self.location
            self.location = self._run_command("pwd")

            # 3. Diff
            self.diff = self._run_command(f"cd {self.project_dir} && git diff && cd -").strip()

            # 4. History
            self.history.append((cmd, output))

            # 5. Reward
            reward = count_step(prev_location, self.edited_files) - count_step(self.location, self.edited_files)

            # 6. Build message
            message = self._build_message(self.location, plan if self.use_plan else "")

        # 7. Extra
        extra = {"diff": self.diff, "command": cmd, "output": output}
        return message, reward, done, extra

    # ----------------------------------------------------------
    # Prompt builder
    # ----------------------------------------------------------
    def _build_message(self, location: str = "", plan: str = "") -> str:
        history = "\n".join(
            f"$ {c}\n{o}" for c, o in self.history[-4:]
        )  # last 4 rounds
        history = "\nHistory (last 4 commands and outputs): " + history if history!='' else ""
        plan = f"\nYour previous plan:\n{plan}" if plan!='' else ""
        prompt = textwrap.dedent(
            f"""\
Problem statement:
{self.current_env['problem_statement']}

Repository location: {self.project_dir}
Current shell location: {location}
{history}
{plan}

Your answer must contain your plan in the <plan> tag and the command in the <execute> tag in the end of the answer.
Example answer:
I believe that I should go to folder /example/folder in the next step.
<plan>
1. go to folder /example/folder
2. investigate the main problem in this folder and form solution
3. fix and test if the patch works
</plan>
<execute>cd /example/folder</execute>

Your answer:
"""
        ).strip()
        return prompt

    # ----------------------------------------------------------
    # Teardown
    # ----------------------------------------------------------
    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("exit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

# ------------------------------------------------------------------
# 4. Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    with SweEnv(
        data_name_or_path="princeton-nlp/SWE-bench_Lite",
        split="test",
        sif_folder="/home/kat/Desktop/FPTAI/swalittle/ext/env",
        base_tools_path="/home/kat/Desktop/FPTAI/swalittle/ext/tools",
        tool_list=["search"],
    ) as env:
        env.reset()
        for i, command in enumerate(['pwd', 'touch a.txt', 'ls', 'exit']):
            msg, reward, done, extra = env.step(f"<execute>{command}</execute>")
            print(f"Step {i+1}: command={command}, reward={reward}, done={done}\n{extra['output']}")
            if done:
                break