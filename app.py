import os
from typing import List, Tuple

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd



st.set_page_config(page_title="Dog Breed Classification — ResNet18", page_icon=None, layout="wide")

CUSTOM_CSS = """
<style>
html, body, [class*="css"]  {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell,
                 "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif;
    color: #111827;
}
.card { background:#fff; border:1px solid #EAECF0; border-radius:12px; padding:18px 20px;
        box-shadow:0 1px 2px rgba(16,24,40,.06); margin-bottom:16px; }
.card-title { font-weight:600; font-size:1.05rem; margin-bottom:8px; color:#111827; }
.subtle { color:#6B7280; } .small { font-size:.9rem; }
.badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:.9rem;
         border:1px solid #D1D5DB; background:#F9FAFB; }
.confbar { height:10px; background:#F3F4F6; border-radius:999px; overflow:hidden; margin-top:8px; }
.confbar-fill { height:100%; background:#2563EB; }
.dataframe td, .dataframe th { font-size:.9rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


DEFAULT_WEIGHTS_PATH = "./Models/resnet18_best_baseline_enhanced.pth"
NUM_CLASSES = 15



DEFAULT_CLASS_NAMES = [
    "Blenheim_spaniel","Border_terrier","Chihuahua","Japanese_spaniel","Lakeland_terrier",
    "Maltese_dog","Norfolk_terrier","Pekinese","Sealyham_terrier","Shih-Tzu","Yorkshire_terrier",
    "cairn","papillon","toy_terrier","wire-haired_fox_terrier",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])




class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def generate(self) -> torch.Tensor:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Run forward + backward before generating CAM.")
        A = self.activations        # (1, C, H, W)
        dY = self.gradients         # (1, C, H, W)
        weights = dY.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * A).sum(dim=1, keepdim=False)  # (1, H, W) -> (H, W)
        cam = torch.relu(cam.squeeze(0))
        cam = self._normalize_cam(cam)
        return cam  # (H, W) in [0,1]

def cam_to_pil_overlay(cam: torch.Tensor, pil_img: Image.Image, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm
    w, h = pil_img.size
    cam_up = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()
    colormap = cm.get_cmap("viridis")
    heatmap = (colormap(cam_up)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).convert("RGBA")
    base = pil_img.convert("RGBA")
    return Image.blend(base, heatmap_img, alpha=alpha)




with st.sidebar:
    st.markdown("### Settings")
    weights_path = st.text_input(
        "Model weights path", value=DEFAULT_WEIGHTS_PATH,
        help="Path to your .pth file (state_dict) saved after training."
    )
    class_source = st.selectbox(
        "Class names source",
        options=["Use default (hardcoded)", "Upload JSON mapping"],
        index=0,
        help="Upload the class_to_idx JSON from training to guarantee correct mapping."
    )
    uploaded_class_json = None
    if class_source == "Upload JSON mapping":
        uploaded_class_json = st.file_uploader("Upload class_to_idx JSON", type=["json"])
    topk = st.slider("Top-k predictions", 1, 5, 3, step=1)
    show_probs = st.checkbox("Show class probabilities", value=True)
    explain = st.checkbox("Explain prediction (Grad-CAM)", value=True)
    alpha = st.slider("Grad-CAM overlay opacity", 0.10, 0.90, 0.45, step=0.05)
    st.markdown("---")
    st.markdown("ResNet18 fine-tuned with enhanced augmentation + cosine LR.")




@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_class_names(uploaded_json, default_names: List[str]) -> List[str]:
    if uploaded_json is not None:
        import json
        try:
            mapping = json.load(uploaded_json)  # {'class_name': idx, ...}
            idx_to_name = {idx: name for name, idx in mapping.items()}
            names = [idx_to_name[i] for i in range(len(idx_to_name))]
            return names
        except Exception as e:
            st.warning(f"Failed to parse JSON mapping. Using default classes. Error: {e}")
    if len(default_names) != NUM_CLASSES:
        st.warning("Default class list length does not match NUM_CLASSES; check your configuration.")
    return default_names

@st.cache_resource
def load_model(weights_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    try:
        model = models.resnet18(weights="IMAGENET1K_V1")
    except Exception:
        model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.to(device)
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found at: {weights_path}")
        st.stop()
    state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict) and all(k.split('.')[0] in {"fc","layer1","layer2","layer3","layer4","conv1","bn1"} for k in state.keys()):
        model.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return EVAL_TRANSFORM(img).unsqueeze(0)

@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    x = x.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_idx = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    return top_idx[0].cpu().numpy(), top_probs[0].cpu().numpy()

def pretty_class(name: str) -> str:
    return name.replace("_", " ").title()

def probs_table(indices: np.ndarray, probs: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    rows = []
    for i, p in zip(indices, probs):
        raw = class_names[int(i)] if int(i) < len(class_names) else f"idx_{int(i)}"
        rows.append({"Class": pretty_class(raw), "Probability": float(p)})
    df = pd.DataFrame(rows)
    df["Probability"] = df["Probability"].map(lambda x: f"{x:.2%}")
    return df

def render_conf_bar(conf: float):
    pct = max(0.0, min(conf, 1.0)) * 100
    st.markdown(f'<div class="confbar"><div class="confbar-fill" style="width:{pct:.1f}%"></div></div>', unsafe_allow_html=True)




device = get_device()
class_names = load_class_names(uploaded_class_json, DEFAULT_CLASS_NAMES)
model = load_model(weights_path, NUM_CLASSES, device)




st.title("Dog Breed Classification")
st.markdown("Image classification into **15 dog breeds** using **ResNet18** fine-tuned with enhanced augmentation and a cosine learning-rate schedule.")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.markdown('<div class="card"><div class="card-title">Model</div>', unsafe_allow_html=True)
    st.write("ResNet18 (transfer learning)")
    st.write(f"Device: **{device.type.upper()}**")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card"><div class="card-title">Reported Test Metrics</div>', unsafe_allow_html=True)
    st.metric(label="Accuracy", value="0.91")
    st.metric(label="Macro F1", value="0.91")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><div class="card-title">Notes</div>', unsafe_allow_html=True)
    st.write("Selected after a comparative study against a baseline. Enhanced augmentation and cosine LR improved generalization.")
    st.markdown('</div>', unsafe_allow_html=True)

def pretty_class(name: str) -> str:
    return " ".join([w.capitalize() for w in name.split("_")])

SUPPORTED_BREEDS = [pretty_class(n) for n in class_names]

st.markdown(
    '<div class="card"><div class="card-title">Supported Dog Breeds</div>'
    '<div class="subtle small">'
    'This model has been trained to recognize only the dog breeds listed below. '
    'It guarantees reliable predictions only if you upload an image containing a single dog '
    'belonging to one of these breeds. '
    'For mixed breeds or complex images, '
    'the predictions may be inaccurate.'
    '</div><br>',
    unsafe_allow_html=True
)

cols = st.columns(3)
for i, breed in enumerate(SUPPORTED_BREEDS):
    with cols[i % 3]:
        st.markdown(f"- {breed}")

st.markdown("</div>", unsafe_allow_html=True)




st.markdown('<div class="card"><div class="card-title">Inference</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload image(s) (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for up in uploaded_files:
       
        try:
            img = Image.open(up)
        except Exception as e:
            st.warning(f"Could not open image: {e}")
            continue

        x = preprocess_image(img)
        idx, p = predict(model, x, device, k=topk)
        top1_idx = int(idx[0])
        top1_prob = float(p[0])
        raw_label = class_names[top1_idx] if top1_idx < len(class_names) else f"idx_{top1_idx}"
        disp_label = pretty_class(raw_label)

        overlay = None
        if explain:
            cam_obj = GradCAM(model, model.layer4[-1])
            model.zero_grad(set_to_none=True)
            x_grad = preprocess_image(img).to(device)
            logits = model(x_grad)
            logits[0, top1_idx].backward(retain_graph=True)
            cam = cam_obj.generate()
            cam_obj.remove()
            overlay = cam_to_pil_overlay(cam, img, alpha=alpha)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.image(img, caption=f"Input: {up.name}", use_container_width=True)
        with col_right:
            if overlay is not None:
                st.image(overlay, caption=f"Grad-CAM — {disp_label}", use_container_width=True)
            else:
                st.image(img, caption="Enable Grad-CAM in the sidebar for explanation", use_container_width=True)

        st.markdown("")  
        left_sp, mid, right_sp = st.columns([1, 2, 1])
        with mid:
            st.markdown('<div class="card"><div class="card-title">Prediction</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="badge">Prediction: {disp_label}</span>', unsafe_allow_html=True)
            if show_probs:
                st.markdown(f'<div class="subtle small">Confidence: {top1_prob:.2%}</div>', unsafe_allow_html=True)
                render_conf_bar(top1_prob)
            st.table(probs_table(idx, p, class_names))
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)



st.markdown("---")
st.markdown(
    "Author: **Nasser Chaouchi**  &nbsp;&nbsp;|&nbsp;&nbsp;  "
    "Model: **resnet18_best_baseline_enhanced.pth**  &nbsp;&nbsp;|&nbsp;&nbsp;  "
    "Framework: **PyTorch + Streamlit** &nbsp;&nbsp;|&nbsp;&nbsp;"
    "[LinkedIn](https://www.linkedin.com/in/nasser-chaouchi/) &nbsp;&nbsp;|&nbsp;&nbsp; "
    "[Github](https://github.com/nasser-chaouchi)"
)

    