import logging
import typing

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import functional as nn_func
from torch.utils.data import Dataset
from torchvision.transforms import functional as vision_func, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


class LaionCocoDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, is_multilabel: bool = True):
        super().__init__()
        self.name = f"laion_coco-{'multilabel' if is_multilabel else 'single_label'}"

        self.is_multilabel = is_multilabel

        df = dataframe
        df = df[df["DOWNLOAD_SUCCESS"] == True]
        df = df[df["TOP_TOPICS"] > 0]

        if is_multilabel:
            df["CAPTION"] = df["all_captions"]
            df["TOPICS"] = df["ALL_TOPICS"] - 1
        else:
            df["CAPTION"] = df["top_caption"]
            df["TOPICS"] = df["TOP_TOPICS"]

        self.num_topics = int(dataframe["TOP_TOPICS"].max()) + 1
        self.topics_oh = self.__topics_one_hot(df, self.num_topics)

        self.dataframe = df

        self.__img_resize = transforms.Resize((224, 224))


    def __len__(self) -> int:
        return len(self.dataframe)


    def __getitem__(self, idx: int) -> typing.Tuple[
        typing.Union[torch.Tensor, Image.Image], typing.List[str], torch.Tensor, bool]:

        row = self.dataframe.iloc[idx]
        image_path = row["FILENAME"]

        try:
            image = Image.open(image_path).convert("RGB")
            # image = image.transform((244, 244))
            image = vision_func.pil_to_tensor(image)
            image = self.__img_resize(image).float()
            # image = vision_func.to_pil_image(image, mode="RGB")

            text = row["CAPTION"]

            topics = self.topics_oh[idx]

            return image, text, topics, True

        except FileNotFoundError:
            logging.error(f"Laion-Coco getitem: Image not found at row {idx:<8d}, filename {row['FILENAME']}")

        except Exception as e:
            logging.error(f"Laion-Coco getitem: Error getting row {idx:<8d}, error type {e.__class__.__name__}")

            return self.__get_dummies()


    def __topics_one_hot(self, df: pd.DataFrame, num_classes: int) -> typing.Tuple[torch.Tensor, np.ndarray]:
        topic_idx = df["TOPICS"].tolist()

        if self.is_multilabel:
            mlb = MultiLabelBinarizer(classes=[i for i in range(1, num_classes + 1)])
            index_oh = mlb.fit_transform(topic_idx)
            return torch.tensor(index_oh), mlb.classes_

        index_oh = nn_func.one_hot(torch.tensor(topic_idx), num_classes=num_classes)
        classes = np.arange(0, num_classes, 1)
        return index_oh, classes


    def __get_dummies(self) -> typing.Tuple[typing.Union[torch.Tensor, Image.Image], typing.List[str], torch.Tensor, bool]:
        return torch.zeros((3, 224, 224)), [""] if self.is_multilabel else "", torch.zeros((self.num_topics,)), False
