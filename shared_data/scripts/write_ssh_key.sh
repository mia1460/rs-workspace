#!/bin/bash
set -e

# 请把以下内容替换为你自己的密钥内容
PRIVATE_KEY_CONTENT="-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACCF1I7FRqY+TUAsLr6YWY4IpetXU1Y8PEakstFrh0bgXwAAAJjz16EX89eh
FwAAAAtzc2gtZWQyNTUxOQAAACCF1I7FRqY+TUAsLr6YWY4IpetXU1Y8PEakstFrh0bgXw
AAAEAcey0Hp5wH9tudxAX6r5YyorKmwi+uHb/uTeTDm6yZEYXUjsVGpj5NQCwuvphZjgil
61dTVjw8RqSy0WuHRuBfAAAAEXJvb3RAMjczNWU0MTcyOTc4AQIDBA==
-----END OPENSSH PRIVATE KEY-----"

PUBLIC_KEY_CONTENT="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIXUjsVGpj5NQCwuvphZjgil61dTVjw8RqSy0WuHRuBf root@2735e4172978"

# 创建 ~/.ssh 目录
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 写入 Ed25519 私钥
echo "$PRIVATE_KEY_CONTENT" > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519

# 写入 Ed25519 公钥
echo "$PUBLIC_KEY_CONTENT" > ~/.ssh/id_ed25519.pub
chmod 644 ~/.ssh/id_ed25519.pub

# 启动 ssh-agent 并添加密钥（可选）
if command -v ssh-agent &> /dev/null; then
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
fi

echo "[INFO] Ed25519 SSH key pair written to ~/.ssh/"
