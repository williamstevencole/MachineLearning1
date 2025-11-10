import numpy as np
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.NUM_POS = 6           # positions
        self.TREASURE_POS = 5      # position of the treasure
        self.START_POS = 0         # starting position

        # Q-learning
        self.ALPHA = 0.10          # learning rate
        self.GAMMA = 0.90          # discount factor

        # Exploration
        self.EPS_START = 0.60      # initial epsilon
        self.EPS_MIN = 0.05        # floor for epsilon
        self.EPS_DECAY = 0.99      # multiplicative decay per episode

        self.EPISODES = 500        # training episodes
        self.MAX_STEPS = 20        # max steps per episode

        # Rewards
        self.PENALTY_RATE = -0.1   # per-step penalty
        self.SUCCESS_REWARD = 1.0  # reward for reaching treasure

        self.SEED = 42
        self.SAVE_PNG = True
        self.PNG_NAME = "TreasureHuntResults.png"

class LinearWorld:
    LEFT, RIGHT = 0, 1

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.SEED)
        # Q-table: states x actions
        self.q = np.zeros((cfg.NUM_POS, 2), dtype=np.float64)

    def step(self, pos, action):
        if action == self.LEFT:
            new_pos = max(0, pos - 1)
        else:
            new_pos = min(self.cfg.NUM_POS - 1, pos + 1)

        if new_pos == self.cfg.TREASURE_POS:
            return new_pos, self.cfg.SUCCESS_REWARD, True
        return new_pos, self.cfg.PENALTY_RATE, False

    # Epsilon-greedy policy
    def choose_action(self, pos, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.choice([self.LEFT, self.RIGHT])
        values = self.q[pos]
        m = np.max(values)
        ties = np.flatnonzero(values == m)
        return int(self.rng.choice(ties))

    # Q-learning update 
    def update_q(self, pos, action, reward, new_pos):
        target = reward + self.cfg.GAMMA * np.max(self.q[new_pos])
        self.q[pos, action] += self.cfg.ALPHA * (target - self.q[pos, action])

    # Train
    def train(self):
        eps = self.cfg.EPS_START
        rewards = []

        for ep in range(1, self.cfg.EPISODES + 1):
            pos = self.cfg.START_POS
            total = 0.0

            for _ in range(self.cfg.MAX_STEPS):
                a = self.choose_action(pos, eps)
                new_pos, r, done = self.step(pos, a)
                self.update_q(pos, a, r, new_pos)

                pos = new_pos
                total += r
                if done:
                    break

            rewards.append(total)
            eps = max(self.cfg.EPS_MIN, eps * self.cfg.EPS_DECAY)

            if ep % 100 == 0:
                avg = np.mean(rewards[-100:])
                print(f"Episode {ep:>3}: avg reward (last 100) = {avg:.2f} | epsilon={eps:.3f}")

        return rewards

    # Greedy evaluation
    def test_policy(self, max_steps=1):
        pos = self.cfg.START_POS
        path = [pos]
        step_rewards = []

        for _ in range(max_steps):
            a = int(np.argmax(self.q[pos])) 
            new_pos, r, done = self.step(pos, a)
            path.append(new_pos)
            step_rewards.append(r)
            pos = new_pos
            if done:
                break
        return path, step_rewards

    # ---------- Pretty print ----------
    def print_q_table(self):
        print("\nQ-TABLE")
        print("State |     Left     |     Right    | Best")
        print("-" * 50)
        for s in range(self.cfg.NUM_POS):
            best = "LEFT" if self.q[s, self.LEFT] > self.q[s, self.RIGHT] else "RIGHT"
            print(f"{s:>5} | {self.q[s,self.LEFT]:10.4f} | {self.q[s,self.RIGHT]:10.4f} | {best}")

    # ---------- Plots ----------
    def plot_results(self, rewards, path):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # 1) Reward per episode + moving average
        axes[0].plot(rewards, alpha=0.35, label="Reward per episode")
        window = 50
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[0].plot(range(window - 1, len(rewards)), ma, linewidth=2, label=f"Moving average ({window})")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total reward")
        axes[0].set_title("Learning progress")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 2) Q-values by state/action
        x = np.arange(self.cfg.NUM_POS)
        left_vals = self.q[:, self.LEFT]
        right_vals = self.q[:, self.RIGHT]
        width = 0.35
        axes[1].bar(x - width / 2, left_vals, width, label="Left")
        axes[1].bar(x + width / 2, right_vals, width, label="Right")
        axes[1].axvline(self.cfg.TREASURE_POS, linestyle="--", linewidth=2, label="Treasure")
        axes[1].set_xticks(x)
        axes[1].set_xlabel("State (position)")
        axes[1].set_ylabel("Q-value")
        axes[1].set_title("Learned Q-values")
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.3)

        # 3) Greedy path visualization (simple)
        for p in range(self.cfg.NUM_POS):
            color = "gold" if p == self.cfg.TREASURE_POS else ("lightblue" if p == self.cfg.START_POS else "lightgray")
            circle = plt.Circle((p, 0), 0.4, color=color, ec="black", linewidth=2)
            axes[2].add_patch(circle)
            axes[2].text(p, 0, f"{p}", ha="center", va="center", fontsize=12, fontweight="bold")

        for i in range(len(path) - 1):
            axes[2].arrow(path[i], -0.5, path[i + 1] - path[i], 0, head_width=0.2, head_length=0.15, linewidth=2)

        axes[2].set_xlim(-1, self.cfg.NUM_POS)
        axes[2].set_ylim(-1, 1)
        axes[2].set_aspect("equal")
        axes[2].axis("off")
        axes[2].set_title("Greedy policy path")

        plt.tight_layout()
        if self.cfg.SAVE_PNG:
            plt.savefig(self.cfg.PNG_NAME, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {self.cfg.PNG_NAME}")
        plt.show()


def main():
    cfg = Config()
    env = LinearWorld(cfg)

    print(env.q)  # misma impresión inicial que tenías
    print("Initial Q-table")
    print("State |   Left   |   Right")
    print("-" * 30)
    for s in range(cfg.NUM_POS):
        print(f"{s:>5} | {env.q[s,env.LEFT]:7.2f} | {env.q[s,env.RIGHT]:7.2f}")

    print("\nTraining...")
    rewards = env.train()
    print("Training complete.")
    env.print_q_table()

    print("\nEvaluation (greedy, no exploration):")
    path, step_rs = env.test_policy(max_steps=10)
    print(f"Path: {path}")
    print(f"Step rewards: {np.round(step_rs, 2).tolist()}")
    if path[-1] == cfg.TREASURE_POS:
        print(f"Treasure reached in {len(path) - 1} steps.")
    else:
        print("Treasure not reached within the given steps.")

    env.plot_results(rewards, path)


if __name__ == "__main__":
    main()
