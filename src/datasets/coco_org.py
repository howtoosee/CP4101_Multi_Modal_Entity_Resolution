import os.path

import json
import logging
import numpy as np
import pandas as pd
import torch
import typing
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torchvision.transforms import functional as vision_func, transforms


# def to_dataframe(dataset):
#     logging.info("Converting dataset to dataframe")
#
#     dataset_dict = dataset.to_dict()
#     topics_list = [r["name"] for r in dataset_dict["info"]["categories"]]
#
#     df = pd.DataFrame(dataset_dict["samples"])
#     logging.debug(f"Dataset has {df.isna().sum()} NA rows")
#     df = df.dropna()
#
#     df["labels"] = df["ground_truth"].apply(
#         lambda row: list(set([det["label"] for det in row["detections"]])) if row else list()
#     )
#     df["num_labels"] = df["labels"].apply(len)
#     df["text"] = df["labels"].apply(
#         lambda labels: f"An image with {' and '.join(labels)}." if labels else "This image shows nothing.")
#
#     df.drop(columns=["tags", "metadata", "ground_truth"], inplace=True)
#
#     return df, topics_list


class Coco2017Dataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", max_samples: int = None):
        super().__init__()
        self.name = "coco-2017"

        data_dir = os.path.expanduser(data_dir)
        if not os.path.exists(data_dir):
            logging.error("Dataset directory does not exist")

        # dataset = foz.load_zoo_dataset(name="coco-2017", split=split, dataset_dir=data_dir)
        # self.dataframe, topics_list = to_dataframe(dataset)

        df = pd.read_csv(os.path.join(data_dir, "coco2017.csv"))
        self.dataframe = df
        # self.dataframe = df[df["labels"].apply(lambda x: "person" not in x)]

        # convert string rep of list of strings to actual list of strings
        self.dataframe["labels"] = self.dataframe["labels"].map(eval)

        topics_list = self.__get_topics_list(data_dir)

        if max_samples:
            self.dataframe = self.dataframe.sample(max_samples, random_state=42)

        self.num_topics = len(topics_list)

        self.topics_oh, self.topics = self.__topics_one_hot(self.dataframe, topics_list, )

        self.__img_resize = transforms.Resize((224, 224))


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx: int) -> typing.Tuple[typing.Union[torch.Tensor, Image.Image], typing.List[str], torch.Tensor, bool]:

        row = self.dataframe.iloc[idx]
        image_path = row["filepath"]

        try:
            image = Image.open(image_path).convert("RGB")
            image = vision_func.pil_to_tensor(image)
            image = self.__img_resize(image).float()

            text = row["text"]

            topics = self.topics_oh[idx]

            return image, text, topics, True

        except FileNotFoundError:
            logging.error(f"Coco2017 getitem: Image not found at row {idx:<8d}, filename {row['FILENAME']}")

        except Exception as e:
            logging.error(f"Coco2017 getitem: Error getting row {idx:<8d}, error type {e.__class__.__name__}")

            return self.__get_dummies()


    def __get_dummies(self) -> typing.Tuple[typing.Union[torch.Tensor, Image.Image], typing.List[str], torch.Tensor, bool]:
        return torch.zeros((3, 224, 224)), [""], torch.zeros((self.num_topics,)), False


    @staticmethod
    def __topics_one_hot(df: pd.DataFrame, topics_list: typing.List[str]) -> typing.Tuple[torch.Tensor, np.ndarray]:
        sample_topics = df["labels"].tolist()
        mlb = MultiLabelBinarizer(classes=topics_list)
        index_oh = mlb.fit_transform(sample_topics)
        return torch.tensor(index_oh), mlb.classes_


    @staticmethod
    def __get_topics_list(data_dir) -> typing.List[str]:
        dataset_info = json.load(open(os.path.join(data_dir, "info.json")))
        all_classes = dataset_info["classes"]
        target_classes = list(filter(lambda s: not s.isnumeric(), all_classes))
        # target_classes = list(filter(lambda s: s != "person", target_classes))
        return target_classes
