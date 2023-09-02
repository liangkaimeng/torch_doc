# torch

Torch包含用于多维张量的数据结构，并定义了对这些张量进行数学运算的方法。此外，它还提供了许多用于高效序列化张量和任意类型以及其他有用实用工具。

它还有一个CUDA版本，允许您在具有计算能力 >= 3.0 的NVIDIA GPU上运行张量计算。

## Tensors

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

## Creation Ops

随机抽样创建操作列在 "随机抽样" 下，包括以下操作：

- `torch.rand()`
- `torch.rand_like()`
- `torch.randn()`
- `torch.randn_like()`
- `torch.randint()`
- `torch.randint_like()`
- `torch.randperm()`

此外，您还可以使用 "就地" 随机抽样方法结合 `torch.empty()` 来创建具有从更广泛分布中抽样的值的 `torch.Tensor`。

| 方法                 | 介绍                                                         |
| -------------------- | ------------------------------------------------------------ |
| tensor               | 通过复制数据来构建一个没有自动求导历史（也称为 "叶子张量"）的张量，可以使用以下方法来实现：`torch.Tensor.detach()` |
| sparse_coo_tensor    | 使用给定的索引在 COO（坐标）格式中构建稀疏张量，该张量具有指定的值。 |
| sparse_csr_tensor    | 构建一个 CSR（压缩稀疏行）格式的稀疏张量，该张量在给定的行索引（`row_indices`）和列索引（`col_indices`）处具有指定的值。 |
| sparse_csc_tensor    | 构建一个 CSC（压缩稀疏列）格式的稀疏张量，该张量在给定的列索引（`col_indices`）和行索引（`row_indices`）处具有指定的值。 |
| sparse_bsr_tensor    | 构建一个 BSR（块压缩稀疏行）格式的稀疏张量，该张量在给定的行索引（`row_indices`）和列索引（`col_indices`）处具有指定的二维块。 |
| sparse_bsc_tensor    | 构建一个 BSC（块压缩稀疏列）格式的稀疏张量，该张量在给定的列索引（`col_indices`）和行索引（`row_indices`）处具有指定的二维块。 |
| asarray              | 将对象（`obj`）转换为张量（tensor）。                        |
| as_tensor            | 将数据转换为张量，并在可能的情况下共享数据并保留自动求导历史。 |
| as_strided           | 使用指定的大小（size）、步幅（stride）和存储偏移（storage_offset）创建一个现有的 torch.Tensor 输入的视图。 |
| from_numpy           | 从一个 numpy.ndarray 创建一个张量（Tensor）。                |
| from_dlpack          | 将来自外部库的张量转换为 torch.Tensor。                      |
| frombuffer           | 从实现了 Python 缓冲协议（Python buffer protocol）的对象创建一个一维张量（1-dimensional Tensor）。 |
| zeros                | 返回一个用标量值 0 填充的张量，其形状由变量参数 size 定义。  |
| zeros_like           | 返回一个与输入具有相同大小的张量，其中所有元素都填充为标量值 0。 |
| ones                 | 返回一个用标量值 1 填充的张量，其形状由变量参数 size 定义。  |
| ones_like            | 返回一个与输入具有相同大小的张量，其中所有元素都填充为标量值 1。 |
| arange               | 返回一个 1 维张量，其大小为⌈(end - start) / step⌉，其中包含从 start 开始，以 step 为公差的取值范围为 [start, end) 的数值。 |
| range                | 返回一个 1 维张量，其大小为⌊(end - start) / step⌋ + 1，其中包含从 start 到 end，以 step 为步长的数值。 |
| linspace             | 创建一个大小为 `steps` 的一维张量，其值从 `start` 到 `end`（包括 `start` 和 `end`）均匀间隔。 |
| logspace             | 在以 `base` 为底的对数尺度上创建一个大小为 `steps` 的一维张量，其值从 `base^start` 到 `base^end`（包括 `base^start` 和 `base^end`）均匀间隔。 |
| eye                  | 返回一个二维张量，在对角线上的元素为1，其他位置的元素为0。这通常被称为单位矩阵（Identity Matrix）或对角矩阵（Diagonal Matrix）。 |
| empty                | 返回一个填充有未初始化数据的张量。                           |
| empty_like           | 返回一个未初始化的张量，其大小与输入相同。                   |
| empty_strided        | 创建一个具有指定大小和步幅的张量，并用未定义的数据填充它。   |
| full                 | 创建一个大小为 `size`，并填充有值 `fill_value` 的张量。      |
| full_like            | 返回一个与输入大小相同的张量，填充有值 `fill_value`。        |
| quantize_per_tensor  | 将一个浮点数张量转换为具有给定比例（scale）和零点（zero point）的量化张量。 |
| quantize_per_channel | 将一个浮点数张量转换为具有给定通道比例（scales）和通道零点（zero points）的分通道量化张量。 |
| dequantize           | 通过将一个量化的张量进行反量化，返回一个 fp32 张量。         |
| complex              | 构建一个复数张量，其中实部等于 `real`，虚部等于 `imag`。     |
| polar                | 构建一个复数张量，其元素是与极坐标中的绝对值 `abs` 和角度 `angle` 对应的笛卡尔坐标。 |
| heaviside            | 计算输入中每个元素的 Heaviside 阶跃函数。                    |





















































