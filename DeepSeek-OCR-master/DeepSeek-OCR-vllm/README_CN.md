# DeepSeek-OCR Web 界面

一个用户友好的 Streamlit Web 界面，用于 DeepSeek-OCR，可轻松将文档和图像转换为 Markdown 格式。

![DeepSeek-OCR Web 界面](./show.png "DeepSeek-OCR Web 界面")

## 在线体验

[仙宫云镜像](https://www.xiangongyun.com/image/detail/12bc7fec-c31a-401e-8034-38992c71fc1b)

## 功能特性

- **多文件支持**：处理图像文件（JPG、PNG）和 PDF 文件
- **批量处理**：一次上传和处理多个文件
- **模型模式**：可选择 Tiny、Small、Base、Large 或 Gundam 模式
- **自定义提示词**：从预设提示词中选择或创建自定义提示词
- **高级设置**：配置最大裁剪数、并发数和工作线程数
- **下载结果**：下载单个结果或将所有文件打包为 ZIP

## 前置要求

在运行 Web 界面之前，请确保您已：

1. **DeepSeek-OCR 环境**：根据原始 DeepSeek-OCR 说明完成设置
2. **Conda 环境**：激活 `deepseek-ocr` conda 环境
3. **模型文件**：下载并可访问 DeepSeek-OCR 模型

## 安装

1. 导航到项目目录：
```bash
cd /path/to/dpsk-ocr
```

2. 激活 DeepSeek-OCR conda 环境：
```bash
conda activate deepseek-ocr
```

3. 安装额外的依赖项：
```bash
pip install streamlit
```

## 使用方法

### 启动 Web 界面

运行以下命令启动 Streamlit 应用：

```bash
streamlit run app.py
```

Web 界面将在您的默认浏览器中打开（通常是 http://localhost:8501）。

### 使用界面

1. **配置设置**（侧边栏）：
   - 选择**模型模式**：Gundam（推荐用于大型文档）、Tiny、Small、Base 或 Large
   - 根据需要调整**高级设置**（最大裁剪数、并发数、工作线程数）
   - 选择**提示词类型**或创建自定义提示词
   - 验证**模型路径**（默认：`deepseek-ai/DeepSeek-OCR`）

2. **上传文件**：
   - 点击"浏览文件"或拖放文件
   - 支持的格式：JPG、JPEG、PNG、PDF
   - 可以一次上传多个文件

3. **处理**：
   - 点击"开始 OCR 处理"
   - 等待处理完成（进度条将显示状态）
   - 首次运行时将加载模型（这可能需要一些时间）

4. **查看结果**：
   - 结果将在下方显示预览
   - 下载单个结果或将所有文件下载为 ZIP

## 模型模式

- **Tiny**：快速处理简单文档（512x512）
- **Small**：平衡速度和准确性（640x640）
- **Base**：大多数文档的标准模式（1024x1024）
- **Large**：复杂文档的高精度模式（1280x1280）
- **Gundam**：针对大型文档优化的裁剪模式（1024 基础，640 图像）

## 提示词类型

- **文档转 Markdown**：转换文档并保留布局
- **OCR 图像**：从图像中提取带结构的文本
- **自由 OCR（无布局）**：提取文本但不保留布局信息
- **解析图表**：从图表和图形中提取信息
- **描述图像**：获取详细的图像描述
- **自定义**：创建您自己的提示词

## 使用技巧

- 对于**大型 PDF**：如果您的 GPU 内存有限，可考虑使用 Gundam 模式并降低并发数
- 为获得**最佳质量**：使用 Base 或 Large 模式
- 为获得**最快处理速度**：使用 Tiny 或 Small 模式
- **GPU 内存**：如果遇到内存不足错误，请减少最大裁剪数（至 4-6）和并发数

## 输出文件

结果保存为 Markdown 文件（.mmd）：
- `filename.mmd`：处理后的输出，包含清晰的 Markdown
- `filename_ori.mmd`：原始输出，包含所有引用标签
- 对于 PDF：`filename_det.mmd` 包含检测信息

## 故障排除

### 内存不足错误
- 将**最大裁剪数**减少到 4-6
- 将**最大并发数**降低到 50 或更少
- 使用较小的模型模式（Tiny 或 Small）

### 模型加载问题
- 确保 DeepSeek-OCR 模型已正确安装
- 检查模型路径是否正确
- 验证您是否在正确的 conda 环境中

### 处理速度慢
- 对于大型文档或高质量模式，这是正常现象
- 考虑对多个文件使用批处理
- 使用较小的模型模式以获得更快的结果

## 文件结构

```
dpsk-ocr/
├── app.py                      # Streamlit Web 界面
├── config.py                   # 配置文件
├── run_dpsk_ocr_image.py      # 原始图像处理脚本
├── run_dpsk_ocr_pdf.py        # 原始 PDF 处理脚本
├── requirements.txt            # Python 依赖项
└── README.md                   # 英文说明文件
└── README_CN.md                # 中文说明文件（本文件）
```

## 高级配置

`config.py` 文件包含可修改的其他设置：
- `MIN_CROPS`：最小裁剪数（默认：2）
- `PRINT_NUM_VIS_TOKENS`：打印视觉令牌计数（默认：False）
- `SKIP_REPEAT`：跳过没有正确 EOS 令牌的页面（默认：True）

## 许可证

本项目使用 DeepSeek-OCR。有关使用条款，请参阅原始 DeepSeek-OCR 许可证。

## 致谢

使用以下技术构建：
- [Streamlit](https://streamlit.io/)
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM](https://github.com/vllm-project/vllm)
