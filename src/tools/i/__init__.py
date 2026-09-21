"""
========================================
tools/i/__init__.py — I 工具入口
========================================

I 是「我写下关于我自己的认识」。条目不参与普通 breath/dream（dont_surface=True），
SessionStart 时先覆盖每个已有维度，再按新到旧补入，I 独享最多 2500 tokens。

对外暴露：dispatch(...) → str（参数与 server.py 中的 I tool 同名）
========================================
"""

from .core import i_core as dispatch  # noqa: F401
