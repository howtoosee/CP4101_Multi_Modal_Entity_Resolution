import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from functools import cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from torchvision.transforms import functional as F, transforms
from tqdm import tqdm

from datasets.coco_org import Coco2017Dataset
from models.image_encoder import get_image_embedding_module
from models.text_encoder import get_text_embedding_module


tqdm.pandas()
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
image_model = get_image_embedding_module("clip", device=device)
text_model = get_text_embedding_module("clip", device=device)
print(f"Using device {device}")
# %%
BASE_DIR = os.path.abspath("/home/x/xchen/CP4101_BComp_Dissertation")

PROJECT_DIR = os.path.join(BASE_DIR, "projects", "main-project")
SAVE_DIR = os.path.join(PROJECT_DIR, "checkpoints")
COCO_DIR = os.path.join(BASE_DIR, "data", "coco-org", "coco2017")
# %%
coco = Coco2017Dataset(COCO_DIR)
dataset = coco.dataframe
target_classes = coco.topics
dataset = dataset.explode("labels")
print(f"Dataset size: {len(dataset)}")

# %%
lb = LabelBinarizer()
lb.fit(dataset["labels"].tolist())


# %%
def get_k_samples_per_class(df: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
    print(f"Sampling {k} points per concept")
    _samples = []
    for _concept in target_classes:
        _data = df[df["labels"] == _concept]
        _samples.append(_data.sample(n=k, replace=len(_data) < k))

    _returned_df = pd.concat(_samples, axis=0)
    _returned_df.reset_index(inplace=True)
    return _returned_df


img_resize = transforms.Resize((224, 224))


@cache
def process_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = F.pil_to_tensor(image)
    image = img_resize(image).float()
    encoded = image_model(image).cpu().numpy()
    return encoded


def convert_image_to_X_y(data: pd.DataFrame, show_progress=False) -> tuple[np.ndarray, np.ndarray]:
    all_images = list()
    all_labels = data["labels"].tolist()

    iter = data.iterrows()
    if show_progress:
        iter = tqdm(iter, total=len(data))

    with torch.no_grad():
        for i, row in iter:
            encoded = process_image(row["filepath"])
            all_images.append(encoded)

    all_images = np.vstack(all_images)
    return all_images, all_labels


@cache
def process_text(text) -> np.ndarray:
    return text_model(text).cpu().numpy()


def convert_text_to_X_y(data: pd.DataFrame, show_progress=False) -> tuple[np.ndarray, np.ndarray]:
    all_text = list()
    all_labels = data["labels"].tolist()

    iter = data.iterrows()
    if show_progress:
        iter = tqdm(iter, total=len(data))

    with torch.no_grad():
        for i, row in iter:
            encoded = process_text(row["text"])
            all_text.append(encoded)

    all_text = np.vstack(all_text)
    return all_text, all_labels


# %%
train_df, test_df = train_test_split(dataset, train_size=0.6, random_state=42)
test_df = get_k_samples_per_class(test_df, 50, seed=42)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# %%
test_image_features, test_image_labels = convert_image_to_X_y(test_df, show_progress=True)
test_text_features, test_text_labels = convert_text_to_X_y(test_df, show_progress=True)
assert np.array_equal(test_image_labels, test_text_labels), "Labels are not the same"
test_labels = test_image_labels


# %%
def few_shot(ks, all_train_df, print_results=True, show_sample_progress=False):
    _best_params = {}
    _reports_str = {}
    _reports_dict = {}

    _image_classifier = LogisticRegression(verbose=False, max_iter=10_000, multi_class="ovr", warm_start=True)
    _text_classifier = LogisticRegression(verbose=False, max_iter=10_000, multi_class="ovr", warm_start=True)

    print(f"ks:{ks.tolist()}")
    for k in tqdm(ks, desc="k"):
        _best_params[k] = {}
        _reports_str[k] = {}
        _reports_dict[k] = {}

        _samples = get_k_samples_per_class(all_train_df, k)
        _train_image_features, _train_image_labels = convert_image_to_X_y(_samples, show_progress=show_sample_progress)
        _train_text_features, _train_text_labels = convert_text_to_X_y(_samples, show_progress=show_sample_progress)
        print("Encoded all images and texts")

        """TRAIN IMAGE CLASSIFIER"""
        _image_scalar = StandardScaler()
        _train_image_features_scaled = _image_scalar.fit_transform(_train_image_features)
        _test_image_features_scaled = _image_scalar.transform(test_image_features)

        print("Training image classifier")
        _image_classifier.fit(_train_image_features_scaled, _train_image_labels)
        print("Done training image classifier")

        _best_params[k]["image"] = _image_classifier.get_params(deep=True)

        _image_pred = _image_classifier.predict(_test_image_features_scaled)
        _reports_str[k]["image"] = classification_report(test_image_labels, _image_pred, zero_division=1)
        _reports_dict[k]["image"] = classification_report(test_image_labels, _image_pred, output_dict=True, zero_division=1)

        """TRAIN TEXT CLASSIFIER"""
        _text_scalar = StandardScaler()
        _train_text_features_scaled = _text_scalar.fit_transform(_train_text_features)
        _test_text_features_scaled = _text_scalar.transform(test_text_features)

        print("Training text classifier")
        _text_classifier.fit(_train_text_features_scaled, _train_text_labels)
        print("Done training text classifier")

        _best_params[k]["text"] = _text_classifier.get_params(deep=True)

        _text_pred = _text_classifier.predict(_test_text_features_scaled)
        _reports_str[k]["text"] = classification_report(test_text_labels, _text_pred, zero_division=1)
        _reports_dict[k]["text"] = classification_report(test_text_labels, _text_pred, output_dict=True, zero_division=1)

        """GET CROSS MODAL PREDICTION"""
        _image_pred = lb.transform(_image_pred)
        _text_pred = lb.transform(_text_pred)
        _cross_pred = np.multiply(np.array(_image_pred), np.array(_text_pred))
        _cross_pred = lb.inverse_transform(_cross_pred)
        _reports_str[k]["cross"] = classification_report(test_labels, _cross_pred, zero_division=1)
        _reports_dict[k]["cross"] = classification_report(test_labels, _cross_pred, output_dict=True, zero_division=0)

        if print_results:
            print(f"\nk={k}")
            for key in ("image", "text", "cross"):
                print(f"{key.upper()}")
                print(_reports_str[k][key])

        torch.save({
            "best_params": best_params,
            "reports_str": reports_str,
            "reports_dict": reports_dict,
        }, os.path.join(SAVE_DIR, f"few_shot_results-{k}.pt"))

    return _best_params, _reports_str, _reports_dict


# %%
# best_params, reports_str, reports_dict = few_shot(2 ** np.arange(0, 4), train_df)  # test
best_params, reports_str, reports_dict = few_shot(2 ** np.arange(0, 8), train_df)  # run
# %%
for k, reports in sorted(reports_str.items()):
    print("=" * 20)
    print(f"k={k}")
    for task, report in reports.items():
        print(f"{task.upper()}")
        print(report)
    print("=" * 20)
