#!/usr/bin/env python3
"""Show an offline A-group execution plan; does not launch anything."""
print('A组离线计划（不启动任务）')
print('数据：官方 train / test；不读取自建训练集或 2570 初筛。')
print('目标：Tutor Qwen3.5-4B；学生 Llama-3.1-8B；门槛 Qwen3.6-27B。')
print('显卡目标：2 张 Tutor / 2 张学生 / 2 张门槛；最多 6 张。')
print('rollout 16；门槛通过后每轨迹 16 次学生后测。')
print('待准备：独立训练环境、单机启动适配、验证集、训练配置及 GPU 归属。')
print('试点：先真实单步，再完成 10 步；阶段结束汇报并等待确认。')
