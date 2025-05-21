#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_quadratic_solver_vectorized.py
---------------------------------
用纯张量操作实现并行 N_ENVS 个环境，避免 Python 循环带来的瓶颈。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# ------------------------------------------------------------------
# 0. 全局设置（超参数）
# ------------------------------------------------------------------
MAX_STEPS     = 1        # 每个 episode 最多步数
TOL           = 0      # 收敛误差阈值
LR            = 3e-4      # 学习率
CLIP_EPS      = 0.2       # PPO 截断范围
VF_COEF       = 0.5       # 值函数 loss 权重
ENT_COEF      = 0.01      # 熵正则系数
GAMMA         = 0.99      # 折扣因子
LAMBDA        = 0.95      # GAE 衰减
HIDDEN_DIM    = 128       # MLP 隐藏层大小
ROLLOUT_STEPS = 1024      # total steps to collect before each update
BATCH_SIZE    = 1024     # PPO 内部 mini-batch 大小
EPOCHS        = 4         # PPO 每次 update 重复轮数
TEST_INTERVAL = 5        # 每隔多少次迭代做一次测试打印
MAX_ITER      = 1000      # 最大训练迭代次数
N_ENVS        = 1024      # 并行环境数量
N_TEST = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(32)  # CPU 上使用多线程
torch.autograd.set_detect_anomaly(True)
# ------------------------------------------------------------------
# 1. Vectorized 环境：所有 env 状态存为 (N_ENVS, …) 的张量
# ------------------------------------------------------------------
class VectorizedQuadraticEnv:
    def __init__(self, n_envs, max_steps, tol, device):
        self.n_envs    = n_envs
        self.max_steps = max_steps
        self.tol       = tol
        self.device    = device
        # 分配存储空间
        self.a        = torch.empty(n_envs, device=device)
        self.b        = torch.empty(n_envs, device=device)
        self.c        = torch.empty(n_envs, device=device)
        self.x        = torch.empty(n_envs, device=device)
        self.err      = torch.empty(n_envs, device=device)
        self.step_cnt = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.reset()

    def _sample_coeffs(self, n):
        x1 = torch.rand(n, device=self.device) * 2 - 1
        x2 = torch.rand(n, device=self.device) * 2 - 1
        a  = torch.rand(n, device=self.device) * 2 - 1
        a[a == 0] = 1.0
        b  = -a * (x1 + x2)
        c  = a * x1 * x2
        return a, b, c, x2

    def reset(self):
        # 重置整个 batch
        self.a, self.b, self.c, x2 = self._sample_coeffs(self.n_envs)
        self.x   = torch.zeros(self.n_envs, device=self.device)
        self.root = x2
        self.err = (self.x - self.root).abs()
        self.step_cnt.zero_()
        return torch.stack([self.a, self.b, self.c, self.x], dim=1)

    @torch.no_grad()
    def step(self, dx):
        # dx: (N_ENVS,1) 或 (N_ENVS,)
        dx       = dx.squeeze(-1)
        x_new    = self.x + dx
        err_prev = self.err
        err_curr =  err_curr = (x_new - self.root).abs()
        reward = torch.log(err_prev)-torch.log(err_curr)
        reward = torch.clamp(reward, min=-10.0, max=10.0)

        self.x    = x_new
        self.err  = err_curr
        self.step_cnt += 1

        done = (err_curr < self.tol) | (self.step_cnt >= self.max_steps)
        states = torch.stack([self.a, self.b, self.c, self.x], dim=1)

        if done.any():
            idx = done.nonzero(as_tuple=True)[0]
            na, nb, nc, x2 = self._sample_coeffs(idx.size(0))
            self.a[idx], self.b[idx], self.c[idx],self.root[idx] = na, nb, nc, x2
            nx = torch.zeros(idx.size(0), device=self.device)
            ne = (nx-x2).abs()
            self.x[idx], self.err[idx] = nx, ne
            self.step_cnt[idx] = 0
            states[idx] = torch.stack([
                self.a[idx], self.b[idx],
                self.c[idx], self.x[idx]
            ], dim=1)

        return states, reward, done.float()

# ------------------------------------------------------------------
# 2. ActorCritic 网络
# ------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, hidden=HIDDEN_DIM):
        super().__init__()
        self.net    = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.logstd  = nn.Parameter(torch.zeros(1) - 1.0)
        self.v_head  = nn.Linear(hidden, 1)

    def forward(self, s):
        h   = self.net(s)
        mu  = self.mu_head(h)
        std = self.logstd.clamp(-30,2).exp().expand_as(mu)
        v   = self.v_head(h).squeeze(-1)
        return mu, std, v

    @torch.no_grad()
    def act(self, s):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        mu, std, v = self.forward(s)
        dist       = Normal(mu, std)
        a          = dist.sample()
        logp       = dist.log_prob(a).sum(-1)
        return a.detach(), logp.detach(), v.detach()

# ------------------------------------------------------------------
# 3. RolloutBuffer：批量 add_batch，GAE 与 returns 计算
# ------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, size, state_dim, device):
        self.size   = size
        self.ptr    = 0
        self.device = device
        # 预分配
        self.s    = torch.zeros(size, state_dim, device=device)
        self.a    = torch.zeros(size, 1, device=device)
        self.r    = torch.zeros(size, device=device)
        self.logp = torch.zeros(size, device=device)
        self.v    = torch.zeros(size, device=device)
        self.done = torch.zeros(size, device=device)

    def add_batch(self, states, actions, rewards, logps, vals, dones):
        batch = states.shape[0]
        idx   = self.ptr
        self.s[   idx:idx+batch] = states
        self.a[   idx:idx+batch] = actions
        self.r[   idx:idx+batch] = rewards
        self.logp[idx:idx+batch] = logps
        self.v[   idx:idx+batch] = vals
        self.done[idx:idx+batch] = dones
        self.ptr += batch

    def ready(self):
        return self.ptr >= self.size

    def clear(self):
        self.ptr = 0

    def compute_adv_ret(self, gamma=GAMMA, lam=LAMBDA):
        adv      = torch.zeros_like(self.r)
        next_val = 0.0
        gae      = 0.0
        for t in reversed(range(self.size)):
            mask  = 1 - self.done[t]
            delta = self.r[t] + gamma * next_val * mask - self.v[t]
            gae   = delta + gamma * lam * mask * gae
            adv[t] = gae
            next_val = self.v[t]
        ret        = adv + self.v
        adv        = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.adv   = adv
        self.ret   = ret

# ------------------------------------------------------------------
# 4. PPOAgent：与原版相同
# ------------------------------------------------------------------
class PPOAgent:
    def __init__(self, net):
        self.net = net
        self.opt = optim.Adam(net.parameters(), lr=LR)

    def update(self, buf):
        for _ in range(EPOCHS):
            idx = torch.randperm(buf.size, device=device)
            for start in range(0, buf.size, BATCH_SIZE):
                sl        = idx[start:start+BATCH_SIZE]
                s, a      = buf.s[sl], buf.a[sl]
                old_lp    = buf.logp[sl]
                adv, ret  = buf.adv[sl], buf.ret[sl]
                mu, std, v= self.net(s)
                dist      = Normal(mu, std)
                logp      = dist.log_prob(a).sum(-1)
                ratio     = (logp - old_lp).exp()
                surr1     = ratio * adv
                surr2     = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
                pg_loss   = -torch.min(surr1, surr2).mean()
                v_loss    = F.mse_loss(v, ret)
                entropy   = dist.entropy().mean()
                loss      = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

class TestEnv(VectorizedQuadraticEnv):
    @torch.no_grad()
    def step(self, dx):
        # 和原版一模一样，除了**不在 done 时自动重置**
        dx       = dx.squeeze(-1)
        x_new    = self.x + dx
        err_prev = self.err
        err_curr = (x_new - self.root).abs()
        reward = torch.log(err_prev)-torch.log(err_curr)
        reward = torch.clamp(reward, min=-10.0, max=10.0)
        self.x        = x_new
        self.err      = err_curr
        self.step_cnt += 1

        done = (err_curr < self.tol) | (self.step_cnt >= self.max_steps)
        states = torch.stack([self.a, self.b, self.c, self.x], dim=1)

        # **直接返回，不做 reset**
        return states, reward, done.float()

# ------------------------------------------------------------------
# 5. 主训练循环
# ------------------------------------------------------------------
def train():
    env    = VectorizedQuadraticEnv(N_ENVS, MAX_STEPS, TOL, device)
    states = env.reset()

    net   = ActorCritic().to(device)
    agent = PPOAgent(net)
    buf   = RolloutBuffer(ROLLOUT_STEPS, state_dim=4, device=device)

    for it in range(1, MAX_ITER+1):
        # 收集数据
        while not buf.ready():
            actions, logps, vals = net.act(states)
            next_states, rewards, dones = env.step(actions)
            buf.add_batch(states, actions, rewards, logps, vals, dones)
            states = next_states

        # 更新策略
        buf.compute_adv_ret()
        agent.update(buf)
        buf.clear()

        # 周期性测试
        if it % TEST_INTERVAL == 0:
            with torch.no_grad():
                test_env = TestEnv(N_TEST, max_steps=MAX_STEPS, tol=TOL, device=device)
                s = test_env.reset()
                total_reward = torch.zeros(N_TEST, device=device)
                init_err = test_env.err.clone()
                for t in range(MAX_STEPS):
                    a, _, _ = net.act(s)
                    s, r, d = test_env.step(a)
                    total_reward += r 
                    final_err = test_env.err 
                avg_init_err=init_err.mean().item()
                avg_reward = total_reward.mean().item()
                avg_err    = final_err.mean().item()
                print(f"[Iter {it:4d}] last_steps={t+1:2d}, avg_init_err={avg_init_err:.3f},avg_err={avg_err:.3e}, avg_reward={avg_reward:.6f}")

if __name__ == "__main__":
    train()
