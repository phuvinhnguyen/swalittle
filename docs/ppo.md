# Bài giảng: Hiểu sâu về REINFORCE và PPO trong Học Tăng Cường

## Phần 1: Giới thiệu tổng quan

### 1.1. Học Tăng Cường (Reinforcement Learning - RL) là gì?
Hãy tưởng tượng bạn đang dạy một chú chó thực hiện mệnh lệnh:
- **Môi trường (Environment)**: Sân chơi nơi chú chó hoạt động
- **Agent**: Chú chó (AI của chúng ta)
- **Hành động (Action)**: Ngồi, nằm, chạy,...
- **Phần thưởng (Reward)**: Miếng thịt khi làm đúng

**Mục tiêu**: Agent học cách thực hiện hành động để nhận được nhiều phần thưởng nhất có thể.

### 1.2. Hai cách tiếp cận chính trong RL
- **Value-based methods** (VD: Q-learning): Học giá trị của các hành động
- **Policy-based methods** (VD: REINFORCE, PPO): Học chính sách hành động trực tiếp

> **Chú ý**: Bài giảng này tập trung vào policy-based methods, cụ thể là REINFORCE và PPO.

## Phần 2: REINFORCE - Thuật toán nền tảng

### 2.1. Ý tưởng cốt lõi
REINFORCE là thuật toán policy gradient đơn giản nhất:
- **Chính sách (Policy)**: Hàm π(a|s) cho biết xác suất chọn hành động a tại trạng thái s
- **Mục tiêu**: Tối đa hóa tổng phần thưởng kỳ vọng J(θ) = E[Σr]

### 2.2. Công thức toán học
Gradient của hàm mục tiêu:
```math
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
```
Trong đó:
- θ: Tham số của mạng neural
- $π(a_t|s_t)$: Xác suất chọn $a_t$ tại $s_t$
- $G_t = ∑_{k=t}^T γ^{k-t} r_k$: Discounted return từ thời điểm t

### 2.3. Chứng minh công thức
Bắt đầu từ hàm mục tiêu:
```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
```
Với $τ = (s_0,a_0,r_0,s_1,a_1,r_1,...)$ là một trajectory.

Áp dụng log trick:
```math
\nabla_\theta J(\theta) = \nabla_\theta \int p(\tau|\theta) R(\tau) d\tau = \int \nabla_\theta p(\tau|\theta) R(\tau) d\tau
```
```math
= \int p(\tau|\theta) \nabla_\theta \log p(\tau|\theta) R(\tau) d\tau = \mathbb{E}[\nabla_\theta \log p(\tau|\theta) R(\tau)]
```

Phân tích p(τ|θ):
```math
p(\tau|\theta) = p(s_0) \prod_{t=0}^T \pi_\theta(a_t|s_t) p(s_{t+1}|s_t,a_t)
```
```math
\log p(\tau|\theta) = \log p(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t|s_t) + \sum_{t=0}^T \log p(s_{t+1}|s_t,a_t)
```

Lấy gradient:
```math
\nabla_\theta \log p(\tau|\theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)
```

Vậy:
```math
\nabla_\theta J(\theta) = \mathbb{E}\left[\left(\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)\right) R(\tau)\right]
```

Để giảm variance, ta thay R(τ) bằng $G_t$ (return từ thời điểm t):
```math
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
```

### 2.4. Ví dụ minh họa bằng tay
**Bài toán**: GridWorld 2x2
- Trạng thái: (0,0), (0,1), (1,0), (1,1)
- Hành động: Lên, xuống, trái, phải
- Phần thưởng: +10 khi đến (1,1), -1 cho mỗi bước

**Trajectory**: 
1. s0=(0,0), a=phải → s1=(0,1), r=-1
2. s1=(0,1), a=xuống → s2=(1,1), r=10

Return:
- G0 = (-1) + 10 = 9
- G1 = 10

Giả sử chính sách ban đầu:
- Tại (0,0): [Lên=0.2, Xuống=0.2, Trái=0.2, Phải=0.4]
- Tại (0,1): [Lên=0.2, Xuống=0.4, Trái=0.2, Phải=0.2]

Tính gradient cho hành động "phải" tại (0,0):
```math
\nabla_\theta \log \pi(\text{phải}|(0,0)) = 1 - \pi(\text{phải}|(0,0)) = 1 - 0.4 = 0.6
```
Gradient cập nhật:
```math
\Delta\theta = \alpha \cdot G_0 \cdot 0.6 = \alpha \cdot 9 \cdot 0.6 = 5.4\alpha
```

Sau cập nhật, xác suất chọn "phải" tăng lên.

### 2.5. Implement REINFORCE
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        return self.fc(state)
    
    def act(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def train(self, trajectories):
        policy_loss = []
        
        for trajectory in trajectories:
            states, actions, rewards, log_probs = zip(*trajectory)
            G = 0
            returns = []
            
            # Tính discounted return từ cuối về đầu
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
                
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Chuẩn hóa
            
            # Tính loss
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
        
        # Cập nhật policy
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
```

### 2.6. Ưu điểm và nhược điểm của REINFORCE
**Ưu điểm**:
- Đơn giản, dễ hiểu và cài đặt
- Hoạt động trực tiếp trên policy

**Nhược điểm**:
- Variance cao (biến động lớn)
- Hiệu quả mẫu thấp (cần nhiều trajectory)
- Cập nhật không ổn định

## Phần 3: PPO - Proximal Policy Optimization

### 3.1. Động lực phát triển PPO
REINFORCE có hai vấn đề lớn:
1. **High variance**: Gradient có biến động lớn do sử dụng toàn bộ return
2. **Destructive updates**: Một cập nhật quá lớn có thể phá hủy policy hiện có

PPO giải quyết bằng:
- **Clipped Surrogate Objective**: Giới hạn sự thay đổi policy
- **Advantage Estimation**: Giảm variance

### 3.2. Công thức toán học của PPO
**Hàm mục tiêu**:
```math
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
```

Trong đó:
- $r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_{old}}(a_t|s_t)}$ (tỉ lệ policy mới/cũ)
- $A_t = Q(s_t,a_t) - V(s_t)$ (advantage function)
- `ϵ` là siêu tham số (thường 0.1-0.3)

### 3.3. Giải thích công thức PPO

**a. Tỉ lệ policy (r_t(θ))**:
- Nếu r_t(θ) > 1: Policy mới có xác suất cao hơn cho hành động này
- Nếu r_t(θ) < 1: Policy mới có xác suất thấp hơn

**b. Advantage function (A_t)**:
- Đo lường hành động này tốt hơn trung bình bao nhiêu
- A_t > 0: Hành động tốt → Khuyến khích
- A_t < 0: Hành động xấu → Phạt

**c. Clip function**:
- Giới hạn r_t(θ) trong khoảng [1-ϵ, 1+ϵ]
- Ngăn chặn thay đổi quá lớn trong một bước cập nhật

**d. Min function**:
- Đảm bảo cập nhật không quá tích cực
- Chọn giá trị bảo thủ hơn giữa hai phương án

### 3.4. Ví dụ minh họa bằng tay
**Tình huống 1: Hành động tốt (A_t = +2)**
- Policy cũ: π_old(a|s) = 0.6
- Policy mới: π_new(a|s) = 0.9 → r_t = 0.9/0.6 = 1.5
- ϵ = 0.2 → clip(1.5, 0.8, 1.2) = 1.2
- min(1.5*2, 1.2*2) = min(3.0, 2.4) = 2.4
→ Loss = -2.4 (giảm loss = tăng reward)

**Tình huống 2: Hành động xấu (A_t = -1)**
- Policy cũ: π_old(a|s) = 0.4
- Policy mới: π_new(a|s) = 0.1 → r_t = 0.1/0.4 = 0.25
- clip(0.25, 0.8, 1.2) = 0.8
- min(0.25*(-1), 0.8*(-1)) = min(-0.25, -0.8) = -0.8
→ Loss = -(-0.8) = +0.8 (tăng loss = giảm xác suất)

### 3.5. Cài đặt PPO
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)
    
    def act(self, state):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
    
    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def compute_advantages(self, returns, values):
        return returns - values
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.epochs):
            # Lấy giá trị mới
            logits, values = self.actor_critic(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Tính ratio
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0-self.epsilon, 1.0+self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Tổng loss
            loss = actor_loss + critic_loss - 0.01 * entropy
            
            # Cập nhật
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
```

### 3.6. Quy trình huấn luyện PPO
1. **Thu thập dữ liệu**:
   - Dùng policy hiện tại chơi N bước
   - Lưu (state, action, reward, done, log_prob, value)
   
2. **Tính returns và advantages**:
   ```math
   R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t} V(s_T)
   ```
   ```math
   A_t = R_t - V(s_t)
   ```

3. **Tối ưu hóa policy**:
   - Lặp lại K epoch:
     - Lấy minibatch từ dữ liệu
     - Tính loss PPO
     - Cập nhật mạng Actor-Critic

4. **Lặp lại** từ bước 1

## Phần 4: So sánh REINFORCE và PPO

### 4.1. Bảng so sánh chi tiết
| Tiêu chí          | REINFORCE                     | PPO                           |
|-------------------|-------------------------------|-------------------------------|
| **Cơ chế**        | Policy gradient cơ bản        | Clipped surrogate objective   |
| **Variance**      | Cao (không baseline)          | Thấp (dùng critic baseline)   |
| **Sample eff.**   | Kém (dùng mỗi trajectory 1 lần)| Tốt (dùng nhiều epoch)        |
| **Ổn định**       | Thấp (cập nhật lớn gây phá vỡ)| Cao (giới hạn thay đổi policy)|
| **Phức tạp**      | Đơn giản                      | Phức tạp hơn                  |
| **Hyperparameter**| Ít (learning rate, gamma)     | Nhiều (ϵ, epoch, clip norm)   |
| **Ứng dụng**      | Bài toán đơn giản             | Bài toán phức tạp (Dota 2, Robotics)|

### 4.2. Biểu đồ minh họa
```
Reward
  ^
  |    PPO: Ổn định, hội tụ nhanh
  |     /\
  |    /  \       REINFORCE: Biến động lớn
  |   /    \      /
  |  /      \    /
  | /        \/\/
  |/__________\______> Episode
 0
```

### 4.3. Khi nào dùng cái nào?
- **REINFORCE**: Bài toán đơn giản, không gian hành động nhỏ, tài nguyên tính toán hạn chế
- **PPO**: Bài toán phức tạp, cần ổn định, tái sử dụng dữ liệu hiệu quả

## Phần 5: Kết luận và bài tập thực hành

### 5.1. Tóm tắt bài học
- **REINFORCE**: Thuật toán policy gradient cơ bản, dễ hiểu nhưng kém ổn định
- **PPO**: Cải tiến vượt trội bằng cách giới hạn sự thay đổi policy, giảm variance
- **Ưu điểm PPO**: Ổn định, hiệu quả mẫu cao, phù hợp cho bài toán phức tạp

### 5.2. Bài tập thực hành
1. **Cài đặt REINFORCE** cho bài toán CartPole:
   - Môi trường: Cột cân bằng trên xe đẩy
   - Phần thưởng: +1 mỗi bước giữ thăng bằng
   - Thành công: Reward > 195 trong 100 episode liên tiếp

2. **Nâng cấp lên PPO**:
   - Thêm mạng Critic
   - Triển khai clipped objective
   - So sánh hiệu suất với REINFORCE

3. **Thử nghiệm hyperparameter**:
   - Với PPO: Thử các giá trị ϵ khác nhau (0.1, 0.2, 0.3)
   - Quan sát ảnh hưởng đến sự ổn định huấn luyện

### 5.3. Tài nguyên mở rộng
- Sách: "Reinforcement Learning: An Introduction" (Sutton & Barto)
- Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- Môi trường: OpenAI Gym, Unity ML-Agents

> "PPO không phải là thuật toán hoàn hảo, nhưng là sự cân bằng tốt giữa độ phức tạp, tính ổn định và hiệu suất." - John Schulman, tác giả PPO

Chúc các bạn thành công trên hành trình khám phá Học Tăng Cường!