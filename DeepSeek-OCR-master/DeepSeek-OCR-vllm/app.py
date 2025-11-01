import streamlit as st
import os
import tempfile
import asyncio
import re
from pathlib import Path
import shutil
import torch
from PIL import Image, ImageOps
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import fitz  # PyMuPDF

# Page configuration
st.set_page_config(
    page_title="DeepSeek-OCR 网页界面",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📄 DeepSeek-OCR 网页界面")
st.markdown("使用 DeepSeek-OCR 将文档和图片转换为 Markdown 格式")

# Helper functions
def pdf_to_images_high_quality(pdf_path, dpi=144):
    """Convert PDF to high-quality images"""
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        images.append(img)

    pdf_document.close()
    return images

def re_match(text):
    """Extract reference patterns from OCR output"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other

@st.cache_resource
def get_model(model_path, max_concurrency):
    """Initialize and cache the VLLM model"""
    from vllm import LLM
    from vllm.model_executor.models.registry import ModelRegistry
    from deepseek_ocr import DeepseekOCRForCausalLM

    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    llm = LLM(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=max_concurrency,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True
    )
    return llm

def process_single_image_prep(image, prompt, crop_mode):
    """Prepare a single image for batch processing"""
    from process.image_process import DeepseekOCRProcessor

    cache_item = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=crop_mode
            )
        },
    }
    return cache_item

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None

# Sidebar for configuration
st.sidebar.header("⚙️ 配置")

# Model mode selection
mode = st.sidebar.selectbox(
    "模型模式",
    ["Gundam", "Tiny", "Small", "Base", "Large"],
    index=0,
    help="选择 OCR 模型模式"
)

# Set mode parameters
mode_configs = {
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True}
}

config = mode_configs[mode]

# Advanced settings
with st.sidebar.expander("高级设置"):
    max_crops = st.slider("最大裁剪数", 2, 9, 6, help="最大裁剪数量（GPU 显存有限时可降低此值）")
    max_concurrency = st.number_input("最大并发数", 1, 200, 100, help="最大并发请求数")
    num_workers = st.number_input("工作进程数", 1, 128, 64, help="图像预处理工作进程数")
    skip_repeat = st.checkbox("跳过重复页面", value=True, help="跳过没有正确结束标记的页面")

# Prompt selection
st.sidebar.header("📝 提示词设置")
prompt_type = st.sidebar.selectbox(
    "提示词类型",
    [
        "文档转 Markdown",
        "OCR 图片",
        "自由 OCR（无布局）",
        "解析图表",
        "描述图片",
        "自定义"
    ],
    index=0
)

# Define prompts
prompts = {
    "文档转 Markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "OCR 图片": "<image>\n<|grounding|>OCR this image.",
    "自由 OCR（无布局）": "<image>\nFree OCR.",
    "解析图表": "<image>\nParse the figure.",
    "描述图片": "<image>\nDescribe this image in detail.",
}

if prompt_type == "自定义":
    prompt = st.sidebar.text_area("自定义提示词", value="<image>\n<|grounding|>Convert the document to markdown.")
else:
    prompt = prompts[prompt_type]
    st.sidebar.info(f"提示词: {prompt}")

# Model path
model_path = st.sidebar.text_input(
    "模型路径",
    value="deepseek-ai/DeepSeek-OCR",
    help="HuggingFace 模型路径或本地路径"
)

# Main content area
st.header("📤 上传文件")

# File uploader
uploaded_files = st.file_uploader(
    "选择图片或 PDF 文件",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
    help="上传一个或多个图片（JPG、PNG）或 PDF 文件"
)

# Process button
if uploaded_files and not st.session_state.processing:
    if st.button("🚀 开始 OCR 处理", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.session_state.results = None

        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        st.session_state.output_dir = output_dir

        try:
            # Set environment variables
            if torch.version.cuda == '11.8':
                os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
            os.environ['VLLM_USE_V1'] = '0'
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'

            # Update config values dynamically
            import config as cfg
            cfg.BASE_SIZE = config["base_size"]
            cfg.IMAGE_SIZE = config["image_size"]
            cfg.CROP_MODE = config["crop_mode"]
            cfg.MAX_CROPS = max_crops
            cfg.MAX_CONCURRENCY = max_concurrency
            cfg.NUM_WORKERS = num_workers
            cfg.SKIP_REPEAT = skip_repeat
            cfg.MODEL_PATH = model_path
            cfg.PROMPT = prompt
            cfg.OUTPUT_PATH = output_dir

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize model
            status_text.text("正在加载 DeepSeek-OCR 模型... 这可能需要一些时间...")
            try:
                llm = get_model(model_path, max_concurrency)

                # Import necessary modules
                from vllm import SamplingParams
                from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

                # Setup sampling parameters
                logits_processors = [NoRepeatNGramLogitsProcessor(
                    ngram_size=20,
                    window_size=50,
                    whitelist_token_ids={128821, 128822}
                )]

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=8192,
                    logits_processors=logits_processors,
                    skip_special_tokens=False,
                    include_stop_str_in_output=True,
                )
            except Exception as e:
                st.error(f"模型加载错误: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.processing = False
                st.stop()

            results = []

            # Group files by type
            pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
            image_files = [f for f in uploaded_files if f.name.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Process image files
            if image_files:
                status_text.text(f"正在处理 {len(image_files)} 张图片...")

                all_images = []
                file_names = []

                for uploaded_file in image_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    image = Image.open(file_path).convert('RGB')
                    all_images.append(image)
                    file_names.append(uploaded_file.name)

                # Prepare batch inputs
                status_text.text("正在准备图片进行批处理...")
                batch_inputs = []
                for image in all_images:
                    batch_input = process_single_image_prep(image, prompt, config["crop_mode"])
                    batch_inputs.append(batch_input)

                # Run batch inference
                status_text.text(f"正在对 {len(batch_inputs)} 张图片进行 OCR 处理... 这可能需要一些时间...")
                progress_bar.progress(0.3)

                outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

                progress_bar.progress(0.7)

                # Process outputs
                for idx, (output, image, filename) in enumerate(zip(outputs_list, all_images, file_names)):
                    status_text.text(f"正在处理输出结果: {filename}...")

                    content = output.outputs[0].text

                    # Clean output
                    if '<｜end▁of▁sentence｜>' in content:
                        content = content.replace('<｜end▁of▁sentence｜>', '')

                    # Save original output
                    result_ori_path = os.path.join(output_dir, f"{filename}_ori.mmd")
                    with open(result_ori_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    # Extract and process matches
                    matches_ref, matches_images, mathes_other = re_match(content)

                    # Replace image references
                    for img_idx, a_match_image in enumerate(matches_images):
                        content = content.replace(a_match_image, f'![](images/{idx}_{img_idx}.jpg)\n')

                    # Remove other references
                    for a_match_other in mathes_other:
                        content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

                    # Save processed output
                    result_path = os.path.join(output_dir, f"{filename}.mmd")
                    with open(result_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    results.append({
                        'filename': filename,
                        'type': '图片',
                        'output_file': result_path,
                        'status': '成功'
                    })

                progress_bar.progress(0.9)

            # Process PDF files
            if pdf_files:
                for pdf_idx, uploaded_file in enumerate(pdf_files):
                    status_text.text(f"正在处理 PDF {uploaded_file.name}... ({pdf_idx+1}/{len(pdf_files)})")

                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Convert PDF to images
                    status_text.text(f"正在将 PDF 转换为图片: {uploaded_file.name}...")
                    images = pdf_to_images_high_quality(file_path)

                    # Prepare batch inputs
                    status_text.text(f"正在准备 {len(images)} 页进行批处理...")
                    batch_inputs = []
                    for image in images:
                        batch_input = process_single_image_prep(image, prompt, config["crop_mode"])
                        batch_inputs.append(batch_input)

                    # Run batch inference
                    status_text.text(f"正在对 {len(images)} 页进行 OCR 处理... 这可能需要一些时间...")
                    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

                    # Process outputs
                    mmd_det_path = os.path.join(output_dir, uploaded_file.name.replace('.pdf', '_det.mmd'))
                    mmd_path = os.path.join(output_dir, uploaded_file.name.replace('.pdf', '.mmd'))

                    contents_det = ''
                    contents = ''
                    jdx = 0

                    for output, img in zip(outputs_list, images):
                        content = output.outputs[0].text

                        if '<｜end▁of▁sentence｜>' in content:
                            content = content.replace('<｜end▁of▁sentence｜>', '')
                        else:
                            if skip_repeat:
                                continue

                        page_num = f'\n<--- Page {jdx+1} --->'

                        contents_det += content + f'\n{page_num}\n'

                        # Extract and process matches
                        matches_ref, matches_images, mathes_other = re_match(content)

                        # Replace image references
                        for img_idx, a_match_image in enumerate(matches_images):
                            content = content.replace(a_match_image, f'![](images/{jdx}_{img_idx}.jpg)\n')

                        # Remove other references
                        for a_match_other in mathes_other:
                            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

                        contents += content + f'\n{page_num}\n'
                        jdx += 1

                    # Save outputs
                    with open(mmd_det_path, 'w', encoding='utf-8') as f:
                        f.write(contents_det)

                    with open(mmd_path, 'w', encoding='utf-8') as f:
                        f.write(contents)

                    results.append({
                        'filename': uploaded_file.name,
                        'type': 'PDF',
                        'output_file': mmd_path,
                        'status': f'成功（已处理 {jdx} 页）'
                    })

            st.session_state.results = results
            progress_bar.progress(1.0)
            status_text.text("✅ 处理完成！")

        except Exception as e:
            st.error(f"处理过程中出错: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

        finally:
            st.session_state.processing = False

# Display results
if st.session_state.results:
    st.header("📊 结果")

    for result in st.session_state.results:
        with st.expander(f"📄 {result['filename']}", expanded=True):
            st.write(f"**类型:** {result['type']}")
            st.write(f"**状态:** {result['status']}")

            if 'output_file' in result and os.path.exists(result['output_file']):
                # Display markdown result
                with open(result['output_file'], 'r', encoding='utf-8') as f:
                    content = f.read()

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown("**预览:**")
                    st.text_area("OCR 结果预览", content, height=300, key=f"preview_{result['filename']}", label_visibility="collapsed")

                with col2:
                    st.markdown("**下载:**")
                    st.download_button(
                        label="下载结果",
                        data=content,
                        file_name=os.path.basename(result['output_file']),
                        mime="text/markdown",
                        key=f"download_{result['filename']}"
                    )

    # Download all results
    if st.session_state.output_dir and os.path.exists(st.session_state.output_dir):
        st.markdown("---")

        # Create zip file with all results
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(st.session_state.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, st.session_state.output_dir)
                    zip_file.write(file_path, arcname)

        st.download_button(
            label="📦 下载所有结果（ZIP）",
            data=zip_buffer.getvalue(),
            file_name="ocr_results.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )

# Information section
with st.expander("ℹ️ 关于 DeepSeek-OCR"):
    st.markdown("""
    ### DeepSeek-OCR 网页界面

    这个网页界面提供了一个简单易用的方式来使用 DeepSeek-OCR 进行文档和图片处理。

    **功能特性:**
    - 支持图片（JPG、PNG）和 PDF 文件
    - 多种模型模式（Tiny、Small、Base、Large、Gundam）
    - 可自定义提示词，适用于不同的 OCR 任务
    - 支持批处理
    - 可将结果下载为 Markdown 文件

    **模型模式说明:**
    - **Tiny**: 快速处理简单文档
    - **Small**: 速度和精度平衡
    - **Base**: 适用于大多数文档的标准模式
    - **Large**: 复杂文档的高精度处理
    - **Gundam**: 针对大型文档优化，采用裁剪技术
    """)

# Footer
st.markdown("---")
st.markdown("基于 Streamlit 构建 | 由 DeepSeek-OCR 驱动")
