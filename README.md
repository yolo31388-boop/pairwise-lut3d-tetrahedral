# pairwise-lut3d-tetrahedral

## 领域

游戏、图形与多媒体

## 主题

色域 LUT 四面体插值

## 说明

域外钳制、统一四面体划分、输入色彩空间。

初始状态下 `imaging/lut3d_grade.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
