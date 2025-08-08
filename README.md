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

如果不在方格的中心处位置，则需要应用插值方法，pytorch中的默认插值方法为bilinear方法，即根据点位置与周围的四个点的距离来进行插值，
可以参见链接 https://blog.csdn.net/suiyuemeng/article/details/103293671 中具体双线性插值的计算方法。
同时如果在边缘处，则需要根据padding_mode来得到边缘处的pad值，然后再进行插值计算。
grid_sample的padding_mode默认是'zeros'，即边缘处补0，同时还有'border', 'reflection'取值。
