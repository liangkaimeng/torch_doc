# torch

Torch包含用于多维张量的数据结构，并定义了对这些张量进行数学运算的方法。此外，它还提供了许多用于高效序列化张量和任意类型以及其他有用实用工具。

它还有一个CUDA版本，允许您在具有计算能力 >= 3.0 的NVIDIA GPU上运行张量计算。

## :collision:Tensors

| 方法                    | 介绍                                                         |
| ----------------------- | ------------------------------------------------------------ |
| is_tensor               | 如果 obj 是一个 PyTorch 张量，则返回 True。                  |
| is_storage              | 如果 obj 是一个 PyTorch 存储对象，则返回 True。              |
| is_complex              | 如果输入的数据类型是复数数据类型，即 torch.complex64 或 torch.complex128 中的一种，则返回 True。 |
| is_conj                 | 如果输入是一个共轭张量，即其共轭位（conjugate bit）设置为 True，则返回 True。 |
| is_floating_point       | 如果输入的数据类型是浮点数数据类型，即 torch.float64、torch.float32、torch.float16 或 torch.bfloat16 中的一种，则返回 True。 |
| is_nonzero              | 如果输入是一个经过类型转换后不等于零的单元素张量，则返回 True。 |
| set_default_dtype       | 将默认的浮点数数据类型设置为 d。                             |
| get_default_dtype       | 获取当前默认的浮点数 torch.dtype。                           |
| set_default_device      | 设置默认情况下分配在设备上的 torch.Tensor。                  |
| set_default_tensor_type | 将默认的 torch.Tensor 类型设置为浮点数张量类型 t。           |
| numel                   | 返回输入张量中的总元素数。                                   |
| set_printoptions        | 设置打印选项。                                               |
| set_flush_denormal      | 在 CPU 上禁用非正规化浮点数。                                |























