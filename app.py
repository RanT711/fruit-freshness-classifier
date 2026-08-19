"""Streamlit interface for offline fruit freshness classification."""

from __future__ import annotations

from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st

from fruit_grader.inference import (
    image_array_from_bytes,
    prediction_from_result,
    resolve_trusted_model_path,
    validate_image_bytes,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load and cache a local YOLO weight file only when prediction is requested."""

    from ultralytics import YOLO

    return YOLO(model_path)


def main() -> None:
    st.set_page_config(page_title="水果新鲜度判别", page_icon="🍎")
    st.title("🍎 水果新鲜度图片判别")
    st.caption("上传一张本地水果图片，使用训练后的 YOLO 分类模型判别其外观新鲜度。")
    st.info("结果仅反映图片可见的外观状态，不构成食品安全或内部品质结论。")

    models_root = Path(__file__).resolve().parent / "models"
    model_path = st.sidebar.text_input("模型文件路径", value="models/best.pt")
    uploaded = st.file_uploader("选择 JPG 或 PNG 图片", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    image_bytes = uploaded.getvalue()
    try:
        validate_image_bytes(image_bytes)
    except ValueError as error:
        st.error(str(error))
        return

    st.image(image_bytes, caption="待判别图片", use_container_width=True)
    try:
        model_file = resolve_trusted_model_path(model_path, models_root)
    except ValueError as error:
        st.error(str(error))
        return

    if not model_file.is_file():
        st.error("找不到模型文件。请先完成训练，或在侧边栏填写正确的 .pt 模型路径。")
        return

    try:
        with st.spinner("正在进行 YOLO 判别..."):
            image_array = image_array_from_bytes(image_bytes)
            results = load_model(str(model_file)).predict(source=image_array, verbose=False)
            prediction = prediction_from_result(results[0])
    except Exception as error:  # Streamlit must present inference errors to the user.
        st.error(f"预测失败：{error}")
        return

    st.success(prediction.message)
    left, right = st.columns(2)
    left.metric("原始类别", prediction.label)
    right.metric("置信度", f"{prediction.confidence:.2%}")


if __name__ == "__main__":
    main()
