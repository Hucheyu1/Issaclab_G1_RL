import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_triple_ae import ActorCritic_Triple_AE, ActorCritic_Triple_AE_Single_Finetune

class Triple_AE_PPO(PPO):
    """
    Triple_AE_PPO扩展了PPO算法, 以集成三重自编码器(Triple Autoencoder)进行三模态策略学习。

    该实现在PPO中添加了多个损失项:
    1. 重建损失(Reconstruction loss)：从各自的隐层表征中重建机器人、人类和关键点的状态
    2. 对齐损失(Alignment loss)：通过均方误差(MSE)将机器人、人类和关键点的隐层表征进行三向对齐
    3. 一致性损失(Consistency loss)：保证不同模态之间的编码与解码保持跨模态一致性
    
    与Dual_AE_PPO的主要区别:
    - 拥有三个独立的编码器而不是两个（机器人、人类、关键点）
    - 三向隐空间对齐而不是双向
    - 三个重建损失而不是两个
    - 将一致性损失扩展到三个模态
    
    属性：
        reconstruction_loss_coef_sg (float): 机器人状态重建损失的系数。
        reconstruction_loss_coef_sh (float): 人类状态重建损失的系数。
        reconstruction_loss_coef_sk (float): 关键点状态重建损失的系数。
        alignment_loss_coef (float): 三向隐空间对齐损失的系数。
        consistency_loss_coef (float): 跨模态一致性损失的系数。
        finetune_human_keypoints (bool): 是否仅微调(finetune)人类和关键点的编码器/解码器。
    """
    policy: ActorCritic_Triple_AE
    
    def __init__(
        self,
        policy: ActorCritic_Triple_AE,
        reconstruction_loss_coef_sg: float = 0.0,
        reconstruction_loss_coef_sh: float = 0.0,
        reconstruction_loss_coef_sk: float = 0.0,
        alignment_loss_coef: float = 0.0,
        consistency_loss_coef: float = 0.0,
        finetune_human_encoder: bool = False,
        finetune_robot_encoder: bool = False,
        finetune_keypoints_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.reconstruction_loss_coef_sg = reconstruction_loss_coef_sg
        self.reconstruction_loss_coef_sh = reconstruction_loss_coef_sh
        self.reconstruction_loss_coef_sk = reconstruction_loss_coef_sk
        self.alignment_loss_coef = alignment_loss_coef
        self.consistency_loss_coef = consistency_loss_coef
        
        # 如果指定了任何网络，则应用微调(finetuning)
        if finetune_human_encoder or finetune_robot_encoder or finetune_keypoints_encoder:
            # 确定需要微调的网络
            finetune_networks = []
            if finetune_human_encoder:
                finetune_networks.extend(['human_encoder', 'human_decoder'])
            if finetune_robot_encoder:
                finetune_networks.extend(['robot_encoder', 'robot_decoder'])
            if finetune_keypoints_encoder:
                finetune_networks.extend(['keypoints_encoder', 'keypoints_decoder'])
            self.policy.actor.freeze_for_finetune(finetune_networks)


    def update(self):  # noqa: C901
        """
        执行带有Triple_AE辅助损失的PPO更新，同时遵循rollout_storage标准。

        该方法在计算标准的PPO损失（替代损失、值函数损失、熵损失）的基础上，
        添加了Triple_AE特征损失：
        - 重建损失(Reconstruction loss)：三个解码器重建各自对应的状态（机器人、人类、关键点）
        - 对齐损失(Alignment loss)：隐层表征（z_robot, z_human, z_keypoints）之间的三向MSE
        - 一致性损失(Consistency loss)：跨模态重建，确保所有三个模态可以相互完美重建
        """
        # 初始化平均损失以便用于日志记录
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_recon_sg_loss = 0
        mean_recon_sh_loss = 0
        mean_recon_sk_loss = 0
        mean_alignment_loss = 0
        mean_consistency_loss = 0

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None

        # 使用非循环的小批量生成器 (丢弃了 is_recurrent 分支)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 遍历多个批次
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            
            # 原始批量大小
            original_batch_size = obs_batch.shape[0]

            # 检查是否应该针对每个小批量(mini batch)对优势函数(advantages)进行归一化
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)


            # 为当前的转换批次重新计算动作对数概率(log prob)和熵(entropy)
            # 注意：我们需要这样做，因为我们使用新参数更新了策略
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # 我们只保留第一个数据增强（即原始状态）的熵
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 将所有GPU上的KL散度进行Reduce聚合(跨卡求和并取均值)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # 更新学习率
                    # 仅在主进程中执行此调整操作
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 更新学习率 for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # 更新学习率 for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 替代损失 (Surrogate loss)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 值函数损失 (Value function loss)
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean())

            ############################################################
            #          TRIPLE_AE AUXILIARY LOSSES                      #
            ############################################################         
            loss_dict = self.policy.get_auxiliary_loss(obs_batch)
            
            recon_sg_loss = loss_dict['reconstruction_sg']
            recon_sh_loss = loss_dict['reconstruction_sh']
            recon_sk_loss = loss_dict['reconstruction_sk']
            alignment_loss = loss_dict['alignment']
            consistency_loss = loss_dict['consistency']
            
            # 总损失：PPO损失 + Triple_AE辅助损失
            loss += (
                self.reconstruction_loss_coef_sg * recon_sg_loss
                + self.reconstruction_loss_coef_sh * recon_sh_loss
                + self.reconstruction_loss_coef_sk * recon_sk_loss
                + self.alignment_loss_coef * alignment_loss
                + self.consistency_loss_coef * consistency_loss
            )

            ############################################################
            #          END TRIPLE_AE AUXILIARY LOSSES                  #
            ############################################################

            # 随机网络蒸馏(RND)损失
            if self.rnd:
                # 预测嵌入(embedding)与目标(target)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # 用均方误差(MSE)计算损失
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # 计算梯度
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # 从所有GPU中收集并同步梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 应用梯度更新
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # 累积并存储损失
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_recon_sg_loss += recon_sg_loss.item()
            mean_recon_sh_loss += recon_sh_loss.item()
            mean_recon_sk_loss += recon_sk_loss.item()
            mean_alignment_loss += alignment_loss.item()
            mean_consistency_loss += consistency_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_recon_sg_loss /= num_updates
        mean_recon_sh_loss /= num_updates
        mean_recon_sk_loss /= num_updates
        mean_alignment_loss /= num_updates
        mean_consistency_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- 清空数据存储
        self.storage.clear()

        # 构造损失字典
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "reconstruction_sg": mean_recon_sg_loss,
            "reconstruction_sh": mean_recon_sh_loss,
            "reconstruction_sk": mean_recon_sk_loss,
            "alignment": mean_alignment_loss,
            "consistency": mean_consistency_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss

        return loss_dict

class Triple_AE_PPO_Single_Finetune(PPO):
    """
    Triple_AE_PPO_Single_Finetune 扩展了 PPO 算法，适用于冻结了编码器/解码器的单模态微调。

    该实现旨在于单一模态上微调预先训练好的 Triple_AE 模型。
    指令(cmd)状态的编码器和解码器会被冻结，模型仅训练动作(action)解码器。

    与 Triple_AE_PPO 的主要区别：
    - 仅有单一模态(cmd)，而不是三个模态（机器人、人类、关键点）
    - 对于cmd状态仅具有重建损失（没有对齐或者一致性损失）
    - 默认情况下冻结编码器/解码器，只有动作解码器是可训练的
    
    属性：
        reconstruction_loss_coef_cmd (float): 指令(cmd)状态重建损失的系数。
    """
    
    def __init__(
        self,
        policy,  # ActorCritic_Triple_AE_Single_Finetune
        reconstruction_loss_coef_cmd: float = 0.0,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.reconstruction_loss_coef_cmd = reconstruction_loss_coef_cmd

    def update(self):  # noqa: C901
        """
        执行带有单模态重建损失的PPO更新。

        该方法利用标准PPO损失（替代损失、值函数损失、熵损失）进行计算，
        并为指令(cmd)状态添加了单个重建损失：
        - 重建损失(Reconstruction loss)：解码器基于隐变向量还原并重建出cmd状态
        """
        # 初始化平均损失以便用于日志记录
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_recon_cmd_loss = 0

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None

        # 使用非循环的小批量生成器 (丢弃了 is_recurrent 分支)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 遍历多个批次
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            
            # 原始批量大小
            original_batch_size = obs_batch.shape[0]

            # 检查是否应该针对每个小批量(mini batch)对优势函数(advantages)进行归一化
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)


            # 为当前的转换批次重新计算动作对数概率(log prob)和熵(entropy)
            # 注意：我们需要这样做，因为我们使用新参数更新了策略
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # 我们只保留第一个数据增强（即原始状态）的熵
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 将所有GPU上的KL散度进行Reduce聚合(跨卡求和并取均值)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # 更新学习率
                    # 仅在主进程中执行此调整操作
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 更新学习率 for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # 更新学习率 for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 替代损失 (Surrogate loss)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 值函数损失 (Value function loss)
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean())

            ############################################################
            #          SINGLE FINETUNE AUXILIARY LOSSES                #
            ############################################################         
            # 如果编码器/解码器被冻结且损失系数为0，则跳过辅助损失的计算
            # 这可以避免通过已冻结编码器进行不必要的前向计算
            loss_dict = self.policy.get_auxiliary_loss(obs_batch)
            recon_cmd_loss = loss_dict['reconstruction_cmd']
            
            # 总损失：PPO损失 + cmd状态重建损失
            if recon_cmd_loss is not None and not self.policy.freeze:
                loss += self.reconstruction_loss_coef_cmd * recon_cmd_loss

            ############################################################
            #          END SINGLE FINETUNE AUXILIARY LOSSES            #
            ############################################################

            # 随机网络蒸馏(RND)损失
            if self.rnd:
                # 预测嵌入(embedding)与目标(target)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # 用均方误差(MSE)计算损失
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # 计算梯度
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # 从所有GPU中收集并同步梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 应用梯度更新
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # 累积并存储损失
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if recon_cmd_loss is not None:
                mean_recon_cmd_loss += recon_cmd_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_recon_cmd_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- 清空数据存储
        self.storage.clear()

        # 构造损失字典
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "reconstruction_cmd": mean_recon_cmd_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss

        return loss_dict