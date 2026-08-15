"""
本地图片描述（vit-gpt2）最小封装。

依赖：
  transformers, torch, pillow

接口：
  describe_image_local(path_or_bytes) -> str

用法示例 (见 README):
  from chatbot.vision_local import describe_image_local
  caption = describe_image_local("path/to/image.jpg")
"""
from typing import Union
from PIL import Image
import io

# 延迟导入以免在未安装依赖时导入模块就报错
_model = None
_processor = None
_tokenizer = None


def _ensure_model_loaded():
    global _model, _processor, _tokenizer
    if _model is not None:
        return
    try:
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        import torch
    except Exception as e:
        raise RuntimeError("需要安装依赖：transformers, torch, pillow。参考 README 中的安装提示。") from e

    model_name = "nlpconnect/vit-gpt2-image-captioning"
    # 在 CPU 环境下确保使用 cpu device
    device = "cpu"
    _model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    _processor = ViTImageProcessor.from_pretrained(model_name)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 推荐的 generate 参数可以在调用处调整
    # 这里不启用 fp16 等 GPU 优化，确保 CPU 可运行


def _load_image(input_data: Union[str, bytes, io.BytesIO]) -> Image.Image:
    if isinstance(input_data, bytes):
        return Image.open(io.BytesIO(input_data)).convert("RGB")
    if isinstance(input_data, io.BytesIO):
        return Image.open(input_data).convert("RGB")
    # assume path-like
    return Image.open(str(input_data)).convert("RGB")


def describe_image_local(path_or_bytes: Union[str, bytes, io.BytesIO]) -> str:
    """
    生成圖片描述（簡短 caption）。
    path_or_bytes: 本地文件路徑 (str) 或 圖片字節 (bytes) / BytesIO。
    返回：模型生成的文本（若失败返回空字符串）。
    """
    try:
        _ensure_model_loaded()
    except Exception as e:
        # 未安装依赖或加载失败，调用方应根据返回值处理
        return ""

    try:
        image = _load_image(path_or_bytes)
    except Exception:
        return ""

    try:
        import torch
        pixel_values = _processor(images=image, return_tensors="pt").pixel_values  # shape (1, C, H, W)
        # 确保在 cpu
        pixel_values = pixel_values.to("cpu")
        # 生成
        with torch.no_grad():
            gen_ids = _model.generate(pixel_values, max_length=64, num_beams=4)
        caption = _tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return caption
    except Exception:
        return ""
