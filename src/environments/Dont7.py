from __future__ import annotations
import re
import random

class Dont7Env:
    """
    Don't-7 game with LLM-friendly I/O.
    The LLM must place its move (1 or 2) inside <step>...</step>.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_move(raw: str) -> str:
        """
        Pull the content between <step> and </step>; fall back to the whole
        stripped string if tags are missing.
        """
        m = raw.split('Answer:\n<step>')[1].split('</step>')[0]
        print(m)
        return m if m else raw.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> tuple[str, dict]:
        self.cur = 0
        self.last_agent_step = None
        self.last_env_step = None
        obs = self._build_obs()
        return obs

    def step(self, raw_llm: str) -> tuple[str, float, bool, dict]:
        move_str = self._extract_move(raw_llm)

        # Validate move
        try:
            agent_step = int(move_str)
            if agent_step not in (1, 2):
                raise ValueError
        except ValueError:
            obs = "Invalid move. Please answer with <step>1</step> or <step>2</step>."
            return obs, -1.0, True, self._build_info()

        self.last_agent_step = agent_step
        self.cur += agent_step
        if self.cur >= 7:
            return self._lose("You made the total 7. You lose!")

        # Environment turn
        env_step = self.rng.choice((1, 2))
        self.last_env_step = env_step
        self.cur += env_step
        if self.cur >= 7:
            return self._win("Environment made the total 7. You win!")

        obs = self._build_obs()
        info = self._build_info()
        return obs, 1.0, False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_obs(self) -> str:
        lines = [
            "Don't-7 Game",
            f"Current total: {self.cur}",
            "You need to choose to take 1 or 2 steps from this momment, if you reach 7, you lose.",
            "Reply with your move inside <step> and </step>, e.g. <step>1</step>",
            "You will get minus point if your answer is not in this format.",
            "Answer:",
            "<step>"
        ]
        if self.last_env_step is not None:
            lines.insert(2, f"Environment last added: {self.last_env_step}")
        return "\n".join(lines)

    def _build_info(self) -> dict:
        return {
            "cur": self.cur,
            "last_agent_step": self.last_agent_step,
            "last_env_step": self.last_env_step,
        }

    def _win(self, msg: str) -> tuple[str, float, bool, dict]:
        obs = msg + f"\nFinal total: {self.cur}"
        return obs, 10.0, True, self._build_info()

    def _lose(self, msg: str) -> tuple[str, float, bool, dict]:
        obs = msg + f"\nFinal total: {self.cur}"
        return obs, -1.0, True, self._build_info()


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = Dont7Env(seed=42)
    obs = env.reset()
    print(obs, "\n")

    for llm_reply in (
        "<step>1</step>",
        "<step>1</step>",
        "<step>2</step>",
    ):
        obs, reward, done, info = env.step(llm_reply)
        print(obs)
        if done:
            print("reward =", reward)
            break