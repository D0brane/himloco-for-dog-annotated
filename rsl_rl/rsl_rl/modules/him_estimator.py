"""HIM 估计器：从观测历史估计线速度与隐向量。

相对原始 PPO，本模块是 HIM 多出来的核心：
- encoder：历史帧 → 速度(3) + 隐向量 z_s
- target：下一帧策略可见观测 → 隐向量 z_t（仅训练用）
- proto：原型嵌入，配合 Sinkhorn 做对比对齐（swap_loss）
- estimation_loss：速度 MSE，标签来自特权观测中的真实线速度

部署时只保留 encoder；target / proto 可丢弃。
"""

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class HIMEstimator(nn.Module):
    """历史编码器 + 目标编码器 + 原型，自带 Adam。

    forward / get_latent 在出动作时调用，返回 detach 后的速度与隐向量，
    不把策略梯度传进来。真正训练在 update() 里单独完成。
    """

    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps          # 历史帧数，如 dog 为 6
        self.num_one_step_obs = num_one_step_obs      # 单步策略可见维数，如 57
        self.num_latent = enc_hidden_dims[-1]         # 隐向量维数，默认 16
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature                # swap 分类的 softmax 温度

        # ---- encoder：真机也会带走的「参谋」----
        # 输入：过去若干帧策略可见观测拼成一条（dog：6×57=342）。
        # 输出：速度 3 维 + 隐向量 16 维；给 Actor 拼成
        #       [当前一帧 57 | 估速 3 | 隐向量 16] → 出动作。
        # 真机没有可靠线速度，所以部署时仍每拍跑 encoder；
        # PolicyExporterHIM 只拷贝本网络，不拷贝下面的 target / proto。
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        # 最后一层多输出 3 维速度，故 +3
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # ---- target：只用在训练场的「另一只眼睛」----
        # 输入：已经发生的下一帧策略可见观测（从特权观测里切出来）。
        # 输出：隐向量 z_t（无速度头）。
        # 和 encoder 从「历史去猜」不同，target 是「事后看下一拍身体反应」。
        # update() 里让 z_s 与 z_t 对齐（再经 proto），逼 encoder 的隐向量
        # 真正带上环境/扰动信息。导出 policy.pt 时丢掉，真机不用。
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # ---- proto：只用在训练场的 32 个「类别筐」----
        # 可学习的 32 个向量，当作隐空间的共用分类词典（坑/滑/被推等是我们脑补的
        # 名字，网络自己分筐）。z_s、z_t 各自和这 32 筐打分，Sinkhorn 成软分类后
        # 交叉预测（swap_loss）：不是把隐向量解码成下一帧观测。
        # 教完 encoder 就下车，导出与真机都不带 proto。
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # 估计器独立优化器，不与 Actor/Critic 的 Adam 混用
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        """编码并 detach，供外部取速度/隐向量用。"""
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        """推理前向：速度与 L2 归一化后的隐向量，均 detach。

        HIMActorCritic.update_distribution 在 no_grad 下调用本函数。
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        """与 forward 类似，但速度/隐向量可带回梯度（供 update 用）。"""
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        """HIM 相对 PPO 多出的两路损失，不改策略损失本身。

        estimation_loss：历史编码器预测的速度，对齐特权观测里的真实线速度。
        swap_loss：历史隐向量与下一帧目标编码器的隐向量互相分类对齐。
        不是把隐向量解码成下一帧观测。

        特权观测布局约定（与 LeggedRobot.compute_observations 一致）::
            [策略可见 num_one_step_obs | lin_vel(3) | ...]
        下一帧策略可见段取 next_critic_obs[:, 3 : num_one_step_obs+3]，
        即去掉开头 3 维角速度后的一段（与训练时拼接顺序对应）。
        """
        if lr is not None:
            # 与 PPO 的 adaptive LR 同步，便于估计器和策略同节奏
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        # 速度监督标签：特权观测中紧跟策略可见段之后的 3 维线速度
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        # 目标编码器输入：下一帧「近似策略可见」观测（去掉角速度头 3 维）
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]

        z_s = self.encoder(obs_history)   # 含速度头，下面再切开
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        # 原型权重始终保持单位长度，便于余弦相似度式打分
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T  # (B, num_prototype)
        score_t = z_t @ self.proto.weight.T

        # Sinkhorn：得到软分配目标 q，不反传进原型打分的归一化过程
        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        # 交叉预测：用对方的 soft 标签监督自己的分类 → 两边隐空间对齐
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    """Sinkhorn-Knopp：把打分矩阵变成近似双随机的软分配。

    用于对比学习里得到均衡的原型占用（避免所有样本塌到同一类）。
    """
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    """按名字返回激活模块实例。"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
