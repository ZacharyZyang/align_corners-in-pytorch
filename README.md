# align_corners-in-pytorch
在torch.nn.functional中，interpolate和grid_sample函数中都有align_corners参数，此处以grid_sample函数来介绍该参数的具体含义。

可参见该函数官方文档 https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample

# 参考图
![image](https://github.com/user-attachments/assets/47d103e3-fb4c-4149-a22b-a64ac471128e)

所以在align_corners=False时，grid在进行归一化时应除以H和W，而在align_corners=True时，grid在进行归一化时应除以H-1和W-1.

# 示例代码
在align_corners=False的情况下，如下代码可以取到a的四个顶点：
```
a = torch.arange(12, dtype=torch.float).reshape(3,4).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 4)
grid = torch.tensor([[[-0.75, -2/3], [0.75, -2/3]], 
                     [[-0.75, 2/3], [0.75, 2/3]]]).unsqueeze(0)                 # (1, 2, 2, 2)
out = F.grid_sample(a, grid=grid, align_corners=False, padding_mode='zeros')    # (1, 1, 2, 2)
print(out)
```
    
可见align_corners=False时，角点并未对齐，即(-1,-1)并不是真正的图像左上角，而是图像左上角像素方块的左上角。

可通过如下代码测试:
```
grid = torch.tensor([[[-1., -1.], [1., -1.]],
                     [[-1., 1,], [1., 1.]]]).unsqueeze(0)                       # (1, 2, 2, 2)                   
out = F.grid_sample(a, grid=grid, align_corners=False, padding_mode='zeros')    # (1, 1, 2, 2)
print(out)
```

# align_corners不同取值时的行为
1. align_corners=True时，归一化坐标空间[-1, 1]对应的是像素中心点，此时(-1, -1)代表图像像素坐标(0, 0)，(1, 1)代表图像像素坐标(W-1, H-1)；
2. align_corners=False时，归一化坐标空间[-1, 1]对应的是像素的边界框。此时归一化坐标(-1, -1)代表图像像素(0, 0)的左上角顶点，坐标是(-0.5, -0.5)；归一化坐标(1, 1)代表的是图像像素 (W-1, H-1) 的右下角顶点，坐标是 (W-0.5, H-0.5)；
所以align_corners=True时，将像素视为点，上/下采样时，是将四个角点的位置保持对齐；而align_corners=False时，将像素视为小方块，整个图像则是一个框，上/下采样时，是将这个图像框的边界对齐，此时角点是方格的中心点，不一定对齐；


# 归一化坐标与图像像素坐标的映射： 
1. align_corners=False, unnormalize coord from [-1, 1] to [0, size - 1]
映射公式：x_pixel = (x_norm + 1) / 2 * W - 0.5; y_pixel = (y_norm + 1) / 2 * H - 0.5
2. align_corners=True, unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
映射公式：x_pixel = (x_norm + 1) / 2 * (W - 1); y_pixel = (y_norm + 1) / 2 * (H - 1)

# 输入输出尺寸的坐标映射：
1. align_corners=False:
  映射公式：x_in = (x_out + 0.5) * (W_in / W_out) - 0.5; y_in = (y_out + 0.5) * (H_in / H_out) - 0.5
  所以此时 [-0.5, -0.5] 与 [-0.5, -0.5] 对齐，[W_out-0.5, H_out-0.5] 与 [W_in-0.5, H_in-0.5] 对齐，即输入输出边界框是对齐的； 但此时采样会超出原始图像的范围（比如-0.5也为预测值，但不为原像素值）；
  当此时x_out=0 或 y_out=0 时，可得到 x_in != 0 或 y_in != 0， 所以角点并未对齐；
2. align_corners=True:
  映射公式：x_in = x_out * (W_in - 1) / (W_out - 1); y_in = y_out * (H_in - 1) / (H_out - 1)
  所以此时 [0, 0] 与 [0, 0] 对齐，[W_out-1, H_out-1] 与 [W_in-1, H_in-1] 对齐，即角点是对齐的；

# 插值以及验证-0.5的偏移
如果不在方格的中心处位置，则需要应用插值方法，pytorch中的默认插值方法为bilinear方法，即根据点位置与周围的四个点的距离来进行插值，
可以参见链接 https://blog.csdn.net/suiyuemeng/article/details/103293671 中具体双线性插值的计算方法。
同时如果在边缘处，则需要根据padding_mode来得到边缘处的pad值，然后再进行插值计算。
grid_sample的padding_mode默认是'zeros'，即边缘处补0，同时还有'border', 'reflection'取值。
