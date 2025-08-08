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
通过手工重写双线性插值，来验证-0.5的偏移，见如下示例代码：
```
def manual_bilinear_interpolation(input_tensor, norm_x, norm_y, align_corners=False):
    """手动实现双线性插值来验证 PyTorch 的行为"""
    _, _, H, W = input_tensor.shape
    
    # 坐标转换
    if align_corners:
        # align_corners=True: [-1,1] -> [0, H-1] or [0, W-1]
        x_pixel = (norm_x + 1) * (W - 1) / 2
        y_pixel = (norm_y + 1) * (H - 1) / 2
    else:
        # align_corners=False: [-1,1] -> [-0.5, W-0.5] or [-0.5, H-0.5]
        x_pixel = (norm_x + 1) * W / 2 - 0.5
        y_pixel = (norm_y + 1) * H / 2 - 0.5
    
    print(f"归一化坐标 ({norm_x}, {norm_y}) 映射到像素坐标 ({x_pixel}, {y_pixel})")
    
    # 获取四个邻近点
    x0 = int(np.floor(x_pixel))
    y0 = int(np.floor(y_pixel))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # 计算权重
    alpha = x_pixel - x0
    beta = y_pixel - y0
    
    print(f"四个邻近点: ({x0},{y0}), ({x1},{y0}), ({x0},{y1}), ({x1},{y1})")
    print(f"权重: alpha={alpha}, beta={beta}")
    
    # 边界检查和插值计算
    def get_pixel_value(x, y):
        if 0 <= x < W and 0 <= y < H:
            return input_tensor[0, 0, y, x].item()
        else:
            return 0.0  # padding with zeros
    
    # 双线性插值公式
    val_00 = get_pixel_value(x0, y0)
    val_10 = get_pixel_value(x1, y0)
    val_01 = get_pixel_value(x0, y1)
    val_11 = get_pixel_value(x1, y1)
    
    result = (1-alpha)*(1-beta)*val_00 + alpha*(1-beta)*val_10 + (1-alpha)*beta*val_01 + alpha*beta*val_11
    return result

# 测试 align_corners=False 时，是否存在 -0.5 的偏移
H, W = 3, 3
input_tensor = torch.arange(9, dtype=torch.float32).reshape(1, 1, H, W)
print("输入图像:")
print(input_tensor.squeeze())

# 测试不同坐标点
test_points = [(-1, -1), (0, 0), (1, 1)]

for norm_x, norm_y in test_points:
    print(f"\n--- 测试点 ({norm_x}, {norm_y}) ---")
    
    # 手动计算
    manual_result = manual_bilinear_interpolation(input_tensor, norm_x, norm_y, align_corners=False)
    print(f"手动计算结果: {manual_result}")
    
    # PyTorch 计算
    grid = torch.tensor([[[norm_x, norm_y]]], dtype=torch.float32).unsqueeze(0)
    pytorch_result = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=False)
    print(f"PyTorch 结果: {pytorch_result.item()}")
    
    print(f"差异: {abs(manual_result - pytorch_result.item())}")
```

pytorch中的默认插值方法为bilinear方法，即根据点位置与周围的四个点的距离来进行插值，
可以参见链接 https://blog.csdn.net/suiyuemeng/article/details/103293671 中具体双线性插值的计算方法。
同时如果在边缘处，则需要根据padding_mode来得到边缘处的pad值，然后再进行插值计算。
grid_sample的padding_mode默认是'zeros'，即边缘处补0，同时还有'border', 'reflection'取值。

但是我们用F.interpolate插值时会发现align_corners=False时的四个角点的值也和原始图像的四个顶点相同，这是由于clamp截断机制。
如下代码验证：
```
    # 边界clamp测试
    inp = torch.arange(12).view(1, 1, 3, 4).float()   # 1×1×3×4
    print("input:\n", inp[0, 0])

    # 1. align_corners=True
    out1 = F.interpolate(inp, size=(6, 8), mode='bilinear', align_corners=True)
    print("\nalign_corners=True, mapping:\n", out1[0, 0])

    # 2. align_corners=False
    # 会发现此时的四个角点的值也和原始图像的四个顶点相同，因为F.interpolate时使用了clamp，当计算出的采样点超出输入图像边界时，会被截断为边界值；
    out2 = F.interpolate(inp, size=(6, 8), mode='bilinear', align_corners=False)
    print("\nalign_corners=False, mapping:\n", out2[0, 0])
```
