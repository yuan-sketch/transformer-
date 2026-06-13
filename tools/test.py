# 袁超
# 开发时间：2026/6/12 14:36
import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(
    img_size=224,
    num_classes=400,
    num_frames=8,
    attention_type='divided_space_time'
)

x = torch.randn(2, 8, 3, 224, 224)

y = model(x)

print(y.shape)