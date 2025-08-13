#!/bin/bash
set -e  # 一旦出错就退出
echo "[Step 1] Configuring Git safe directory"
git config --global --add safe.directory /workspace/recsys-examples
git submodule update --init --recursive

# 安装 dynamicemb
echo "[Step 2] Installing dynamicemb..."
cd /workspace/recsys-examples/corelib/dynamicemb
python setup.py clean --all
# python setup.py install
pip install -e .

# 安装 hstu（主模块）
echo "[Step 3] Installing hstu..."

# rm -rf third_party/cutlass
# cd /workspace/recsys-examples
# git submodule update --init --recursive
cd /workspace/recsys-examples/corelib/hstu
python setup.py clean --all
HSTU_DISABLE_LOCAL=TRUE \
HSTU_DISABLE_RAB=TRUE \
HSTU_DISABLE_DRAB=TRUE \
pip install -e .

echo "[Step 4] Installing hstu examples..."
# 安装 hstu 示例
cd /workspace/recsys-examples/examples/hstu
python setup.py clean --all
# python setup.py install
pip install -e .

# 安装 flashinfer-python（假设它已在环境变量中或 requirements 里）
# pip install ujson
# pip install --upgrade 'tensordict[dev]'

echo "Clean build and installation completed successfully."
