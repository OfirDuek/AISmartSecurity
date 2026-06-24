import os
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model", "violence_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViolenceDetectorMViT(nn.Module):
    def __init__(self, dropout=0.5):
        super(ViolenceDetectorMViT, self).__init__()
        weights = MViT_V2_S_Weights.DEFAULT
        self.mvit = mvit_v2_s(weights=weights)

        in_features = self.mvit.head[1].in_features
        self.mvit.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        return self.mvit(x).squeeze(1)


def load_model():
    model = ViolenceDetectorMViT().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[+] Model loaded on {DEVICE}")
    return model


def predict(model, frames: torch.Tensor, threshold: float = 0.5) -> bool:
    """
    frames: torch.Tensor of shape (C, T, H, W) - output of preprocess.extract_clips()
    Returns True if violence detected.
    """
    input_tensor = frames.unsqueeze(0).to(DEVICE)  # [1, C, T, H, W]
    with torch.no_grad():
        prob = model(input_tensor).item()
    return prob > threshold
