import logging
import numpy as np
import os
import torch
import typing
import warnings
from sklearn.metrics import average_precision_score, classification_report
from torch import cuda, nn, optim
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .image_encoder import get_image_embedding_module
from .modal_discriminator import get_modal_discriminator
from .text_encoder import get_text_embedding_module
from .topic_predictor import get_topic_predictor


def losses_to_string(losses: typing.List[str]) -> str:
    return ", ".join([f"{loss:.3f}" for loss in losses])


def get_map5_scores(y_true: np.ndarray, y_logits: np.ndarray) -> np.float:
    if y_true.shape != y_logits.shape:
        raise ValueError(f"y_true and y_logits must have the same shape, but got y_true: {y_true.shape} and y_pred: {y_logits.shape}")

    top_5_true = list()
    top_5_preds = list()
    for yt, yl in zip(y_true, y_logits):
        sorted_pairs = sorted(zip(yt, yl), key=lambda x: x[0], reverse=True)[:5]
        sorted_pairs = np.vstack(sorted_pairs)
        yt = sorted_pairs[:, 0].astype(int)
        yl = sorted_pairs[:, 1]
        yp = 1 * (yl > 0.5)
        top_5_true.append(yt)
        top_5_preds.append(yp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ap_scores: np.ndarray = average_precision_score(top_5_true, top_5_preds, average=None)
    return ap_scores.mean()


def get_map_scores(y_true: np.ndarray, y_logits: np.ndarray) -> typing.Tuple[np.ndarray, np.float, np.float]:
    if y_true.shape != y_logits.shape:
        raise ValueError(f"y_true and y_logits must have the same shape, but got y_true: {y_true.shape} and y_pred: {y_logits.shape}")

    y_pred = 1 * (y_logits > 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ap_scores: np.ndarray = average_precision_score(y_true, y_pred, average=None)
    map_all = ap_scores.mean()

    map_5 = get_map5_scores(y_true, y_logits)

    return ap_scores, map_all, map_5


def get_cross_map_5(targets: np.ndarray, from_logits: np.ndarray, to_logits: np.ndarray) -> np.float:
    if targets.shape != from_logits.shape or targets.shape != to_logits.shape:
        raise ValueError(
            f"targets, from_logits and to_logits must have the same shape, but got targets: {targets.shape}, from_logits: {from_logits.shape} and to_logits: {to_logits.shape}")

    top_5_true = list()
    top_5_preds = list()
    for ytrue, yfrom, yto in zip(targets, from_logits, to_logits):
        sorted_pairs = sorted(zip(ytrue, yfrom, yto), key=lambda x: x[0], reverse=True)[:5]
        sorted_pairs = np.vstack(sorted_pairs)
        ytrue = sorted_pairs[:, 0].astype(int)
        yfrom = 1 * (sorted_pairs[:, 0] > 0.5)
        yto = 1 * (sorted_pairs[:, 2] > 0.5)

        top_5_true.append(ytrue)
        top_5_preds.append(np.multiply(yfrom, yto))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ap_scores: np.ndarray = average_precision_score(top_5_true, top_5_preds, average=None)
    return ap_scores.mean()


class Solver:
    DIS_IMAGE_TARGET = torch.Tensor([1, 0]).unsqueeze(0)
    DIS_TEXT_TARGET = torch.Tensor([0, 1]).unsqueeze(0)


    def __init__(self, name: str, is_multilabel: bool, topics: typing.Union[typing.List[str], np.ndarray], test_data: DataLoader,
                 image_encoder: nn.Module, text_encoder: nn.Module, image_predictor: nn.Module, text_predictor: nn.Module, discriminator: nn.Module,
                 learn_rate=0.0001, d_coef=0.1, prev_epoch=None, checkpoint_dir=None, **_):
        self.name = name
        self.is_multilabel = is_multilabel
        self.topics = np.asarray(topics)
        self.num_topics = len(topics)
        self.test_dataloader = test_data

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_predictor = image_predictor
        self.text_predictor = text_predictor
        self.discriminator = discriminator

        logging.info(
            f"Devices: Image: {self.image_encoder.device}, Text: {self.text_encoder.device}, Image Predictor: {self.image_predictor.device}, Text Predictor: {self.text_predictor.device}, Discriminator: {self.discriminator.device}")

        # self.optimizer_ie = optim.Adam(self.image_encoder.parameters(), lr=learn_rate)
        # self.optimizer_te = optim.Adam(self.text_encoder.parameters(), lr=learn_rate)
        self.optimizer_ip = optim.Adam(self.image_predictor.parameters(), lr=learn_rate)
        self.optimizer_tp = optim.Adam(self.text_predictor.parameters(), lr=learn_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=learn_rate)

        self.d_coef = d_coef

        if is_multilabel:
            # multi-label classification
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # multi-class classification
            self.criterion = nn.CrossEntropyLoss()

        self.discriminator_criterion = nn.BCEWithLogitsLoss()

        if cuda.is_available():
            self.criterion = self.criterion.cuda()

        self.d_losses = list()
        self.g_losses = list()

        # Top 5 MAP scores
        self.map_5_history_i2i = list()
        self.map_5_history_t2t = list()
        self.map_5_history_i2t = list()
        self.map_5_history_t2i = list()

        # All MAP scores
        self.map_all_history_i2i = list()
        self.map_all_history_t2t = list()
        self.map_all_history_cross = list()

        # Classification reports
        self.classification_reports_i2i = list()
        self.classification_reports_t2t = list()
        self.classification_reports_cross = list()

        self.curr_epoch = prev_epoch + 1 if prev_epoch else 1
        if prev_epoch:
            logging.info(f"Model already trained for {prev_epoch} epochs, continuing training...")
        logging.info(f"Starting training at epoch {self.curr_epoch}")
        self.checkpoint_dir = checkpoint_dir


        # if cuda.is_available():
        #     self.DIS_IMAGE_TARGET = self.DIS_IMAGE_TARGET.cuda()
        #     self.DIS_TEXT_TARGET = self.DIS_TEXT_TARGET.cuda()
        #     logging.info(f"Disc target shape: {self.DIS_IMAGE_TARGET.shape}, Device: {self.DIS_IMAGE_TARGET.device}")


    def train(self, num_epochs: int, dataloader: data.DataLoader) -> None:
        """losses from each sample"""
        all_g_losses = list()
        all_d_losses = list()

        """mean of all batches in each epoch"""
        mean_epoch_g_losses = list()
        mean_epoch_d_losses = list()

        # Iterate over epochs
        for epoch in trange(self.curr_epoch, self.curr_epoch + num_epochs, desc="Epochs"):
            self.curr_epoch = epoch

            curr_epoch_g_losses = list()
            curr_epoch_d_losses = list()

            self.set_train()

            # Iterate over batches
            for i, (images, texts, topicss, validities) in tqdm(enumerate(dataloader),
                                                                total=len(dataloader), desc=f"Epoch {epoch} Batches"):
                # logging.info(f"Training Epoch {epoch}, Batch {i} of {len(dataloader)}")

                d_loss = torch.tensor(0).float()
                g_loss = torch.tensor(0).float()

                topicss = topicss.float()

                batch_image_features = list()
                batch_text_features = list()

                with torch.no_grad():
                    for image, text, topics, is_valid in zip(images, texts, topicss, validities):
                        if not is_valid or None in (image, text, topics):
                            # ignore invalid samples
                            continue

                        image_features = self.image_encoder(image)
                        text_features = self.text_encoder(text)

                        batch_image_features.append(image_features)
                        batch_text_features.append(text_features)

                batch_image_features = torch.vstack(batch_image_features)
                batch_text_features = torch.vstack(batch_text_features)

                batch_image_discriminator_prediction = self.discriminator(batch_image_features.to(self.discriminator.device)).cpu()
                batch_text_discriminator_prediction = self.discriminator(batch_text_features.to(self.discriminator.device)).cpu()

                batch_image_topic_prediction = self.image_predictor(batch_image_features.to(self.image_predictor.device)).cpu()
                batch_text_topic_prediction = self.text_predictor(batch_text_features.to(self.text_predictor.device)).cpu()

                d_loss += self.discriminator_criterion(batch_image_discriminator_prediction,
                                                       self.DIS_IMAGE_TARGET.repeat(batch_image_discriminator_prediction.shape[0], 1).float())
                d_loss += self.discriminator_criterion(batch_text_discriminator_prediction,
                                                       self.DIS_TEXT_TARGET.repeat(batch_text_discriminator_prediction.shape[0], 1).float())

                g_loss += self.criterion(batch_image_topic_prediction, topicss)
                g_loss += self.criterion(batch_text_topic_prediction, topicss)
                total_loss = g_loss + self.d_coef * d_loss

                if cuda.is_available():
                    d_loss = d_loss.cuda()
                    g_loss = g_loss.cuda()

                self.zero_grad()

                d_loss.backward(retain_graph=True)
                total_loss.backward()

                # self.optimizer_ie.step()
                # self.optimizer_te.step()
                self.optimizer_ip.step()
                self.optimizer_tp.step()
                self.optimizer_d.step()

                curr_epoch_g_losses.append(g_loss.item())
                curr_epoch_d_losses.append(d_loss.item())

            mean_g_loss = np.mean(curr_epoch_g_losses).item()
            mean_d_loss = np.mean(curr_epoch_d_losses).item()
            logging.info(f"Epoch {epoch}/{num_epochs}, Mean G_loss: {mean_g_loss:.3f}, Mean D_loss: {mean_d_loss:.3f}")
            # logging.info(f"G loss per batch: {losses_to_string(curr_epoch_g_losses)}")
            # logging.info(f"D loss per batch: {losses_to_string(curr_epoch_d_losses)}")

            i2i_map_all, i2i_map_5, t2t_map_all, t2t_map_5, cross_map_all, i2t_map_5, t2i_map_5, i2i_report, t2t_report, cross_report = self.evaluate()

            all_g_losses.extend(curr_epoch_g_losses)
            all_d_losses.extend(curr_epoch_d_losses)
            mean_epoch_g_losses.append(mean_g_loss)
            mean_epoch_d_losses.append(mean_d_loss)

            self.map_5_history_i2i.append(i2i_map_5)
            self.map_5_history_t2t.append(t2t_map_5)
            self.map_5_history_i2t.append(i2t_map_5)
            self.map_5_history_t2i.append(t2i_map_5)

            self.map_all_history_i2i.append(i2i_map_all)
            self.map_all_history_t2t.append(t2t_map_all)
            self.map_all_history_cross.append(cross_map_all)

            self.classification_reports_i2i.append(i2i_report)
            self.classification_reports_t2t.append(t2t_report)
            self.classification_reports_cross.append(cross_report)

            if epoch % 5 == 0 or epoch == self.curr_epoch + num_epochs - 1:
                self.save_checkpoint()

        logging.info(f"Mean G_loss per epoch: {losses_to_string(mean_epoch_g_losses)}")
        logging.info(f"Mean D_loss per epoch: {losses_to_string(mean_epoch_d_losses)}")


    def evaluate(self) -> typing.Tuple[float, float, float, float, float, float, float, dict, dict, dict]:
        self.set_eval()
        exp_shape = (len(self.test_dataloader.dataset), self.num_topics)

        targets = list()

        batch_image_features = list()
        batch_text_features = list()

        for i, (image, text, topics, is_valid) in enumerate(self.test_dataloader.dataset):
            if not is_valid:  # ignore invalid samples
                continue

            with torch.no_grad():
                image_features = self.image_encoder(image)
                text_features = self.text_encoder(text)

                batch_image_features.append(image_features)
                batch_text_features.append(text_features)

            targets.append(topics)

        batch_image_features = torch.vstack(batch_image_features).to(self.image_predictor.device)
        batch_text_features = torch.vstack(batch_text_features).to(self.text_predictor.device)

        with torch.no_grad():
            batch_image_topic_logits = self.image_predictor.predict(batch_image_features)
            batch_text_topic_logits = self.text_predictor.predict(batch_text_features)

        del batch_image_features, batch_text_features
        batch_image_topic_logits = batch_image_topic_logits.cpu().numpy()
        batch_text_topic_logits = batch_text_topic_logits.cpu().numpy()

        batch_image_topic_prediction = 1 * (batch_image_topic_logits > 0.5)
        batch_text_topic_prediction = 1 * (batch_text_topic_logits > 0.5)

        targets = np.vstack(targets)

        assert batch_image_topic_prediction.shape == exp_shape, f"Image prediction shape: {batch_image_topic_prediction.shape} != {exp_shape}"
        assert batch_text_topic_prediction.shape == exp_shape, f"Text prediction shape: {batch_image_topic_prediction.shape} != {exp_shape}"
        assert targets.shape == exp_shape, f"Targets shape: {targets.shape} != {exp_shape}"

        image_ap_scores, image_map_all, image_map_5 = get_map_scores(targets, batch_image_topic_logits)
        text_ap_scores, text_map_all, text_map_5 = get_map_scores(targets, batch_text_topic_logits)

        cross_topic_prediction = np.multiply(batch_image_topic_prediction, batch_text_topic_prediction)
        cross_ap_scores, cross_map_all, _ = get_map_scores(targets, cross_topic_prediction)
        i2t_map_5 = get_cross_map_5(targets, batch_image_topic_logits, batch_text_topic_logits)
        t2i_map_5 = get_cross_map_5(targets, batch_text_topic_logits, batch_image_topic_logits)

        logging.info(f"MAP@all: I2I: {image_map_all:.3f}, T2T: {text_map_all:.3f}, I2T: {cross_map_all:.3f}, T2I: {cross_map_all:.3f}")
        logging.info(f"MAP@5  : I2I: {image_map_5:.3f}, T2T: {text_map_5:.3f}, I2T: {i2t_map_5:.3f}, T2I: {t2i_map_5:.3f}")

        i2i_report = classification_report(targets, batch_image_topic_prediction, target_names=self.topics, output_dict=True)
        t2t_report = classification_report(targets, batch_text_topic_prediction, target_names=self.topics, output_dict=True)
        i2t_report = classification_report(targets, cross_topic_prediction, target_names=self.topics, output_dict=True)

        return image_map_all, image_map_5, text_map_all, text_map_5, cross_map_all, i2t_map_5, t2i_map_5, i2i_report, t2t_report, i2t_report


    def zero_grad(self) -> None:
        # self.optimizer_ie.zero_grad()
        # self.optimizer_te.zero_grad()

        self.optimizer_ip.zero_grad()
        self.optimizer_tp.zero_grad()

        self.optimizer_d.zero_grad()


    def set_train(self) -> None:
        # self.image_encoder.train()
        # self.text_encoder.train()

        self.image_predictor.train()
        self.text_predictor.train()

        self.discriminator.train()


    def set_eval(self) -> None:
        # self.image_encoder.eval()
        # self.text_encoder.eval()

        self.image_predictor.eval()
        self.text_predictor.eval()

        self.discriminator.eval()


    def save_checkpoint(self) -> None:
        if self.checkpoint_dir is None:
            return

        save_path = os.path.expanduser(os.path.join(
            self.checkpoint_dir,
            f"{self.name}-{self.image_encoder.model_name}_{self.text_encoder.model_name}-epoch_{self.curr_epoch}.pt"
        ))

        logging.info(f"Saving checkpoint at epoch {self.curr_epoch}")
        torch.save({
            "name": self.name,
            "is_multilabel": self.is_multilabel,
            "topics": self.topics,
            "image_model_name": self.image_encoder.model_name,
            "text_model_name": self.text_encoder.model_name,
            "model_image_encoder": self.image_encoder.state_dict(),
            "model_text_encoder": self.text_encoder.state_dict(),
            "model_image_predictor": self.image_predictor.state_dict(),
            "model_text_predictor": self.text_predictor.state_dict(),
            "model_discriminator": self.discriminator.state_dict(),
            # "optimizer_ie": self.optimizer_ie.state_dict(),
            # "optimizer_te": self.optimizer_te.state_dict(),
            "optimizer_ip": self.optimizer_ip.state_dict(),
            "optimizer_tp": self.optimizer_tp.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "d_losses": self.d_losses,
            "g_losses": self.g_losses,
            "map_5_image": self.map_5_history_i2i,
            "map_5_text": self.map_5_history_t2t,
            "map_5_i2t": self.map_5_history_i2t,
            "map_5_t2i": self.map_5_history_t2i,
            "map_all_image": self.map_all_history_i2i,
            "map_all_text": self.map_all_history_t2t,
            "map_all_cross": self.map_all_history_cross,
            "report_i2i": self.classification_reports_i2i,
            "report_t2t": self.classification_reports_t2t,
            "report_cross": self.classification_reports_cross,
            "epochs": self.curr_epoch,
        }, save_path)
        logging.info(f"Done saving checkpoint to {save_path}")


def get_solver(name: str, is_multilabel: bool, topics: typing.Union[typing.List[str], np.ndarray],
               test_data: DataLoader, learn_rate: float, d_coef: float,
               image_model: str = "clip", text_model: str = "clip", checkpoint_dir: str = None) -> Solver:

    if cuda.is_available():
        num_cuda_devices = cuda.device_count()
        image_device = f"cuda:{0 & num_cuda_devices}"
        text_device = f"cuda:{1 % num_cuda_devices}"
    else:
        image_device = text_device = None

    num_topics = len(topics)
    image_encoder = get_image_embedding_module(image_model, image_device)
    text_encoder = get_text_embedding_module(text_model, text_device)
    image_topic_predictor = get_topic_predictor(is_multilabel, num_topics)
    text_topic_predictor = get_topic_predictor(is_multilabel, num_topics)
    modal_discriminator = get_modal_discriminator()

    return Solver(
        name=name,
        is_multilabel=is_multilabel,
        topics=topics,
        test_data=test_data,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_predictor=image_topic_predictor,
        text_predictor=text_topic_predictor,
        discriminator=modal_discriminator,
        learn_rate=learn_rate,
        d_coef=d_coef,
        checkpoint_dir=checkpoint_dir
    )


def get_solver_from_checkpoint(test_data: DataLoader, topics: typing.Union[typing.List[str], np.ndarray],
                               checkpoint_path: str, new_checkpoint_dir: str = None) -> Solver:
    if cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        num_cuda_devices = cuda.device_count()
        image_device = f"cuda:{0 & num_cuda_devices}"
        text_device = f"cuda:{1 % num_cuda_devices}"
    else:
        checkpoint = torch.load(checkpoint_path)
        image_device = text_device = None

    num_topics = len(topics)
    image_encoder = get_image_embedding_module(checkpoint["image_model_name"], image_device)
    text_encoder = get_text_embedding_module(checkpoint["text_model_name"], text_device)
    image_topic_predictor = get_topic_predictor(checkpoint["is_multilabel"], num_topics)
    text_topic_predictor = get_topic_predictor(checkpoint["is_multilabel"], num_topics)
    modal_discriminator = get_modal_discriminator()

    image_encoder.load_state_dict(checkpoint["model_image_encoder"])
    text_encoder.load_state_dict(checkpoint["model_text_encoder"])
    image_topic_predictor.load_state_dict(checkpoint["model_image_predictor"])
    text_topic_predictor.load_state_dict(checkpoint["model_text_predictor"])
    modal_discriminator.load_state_dict(checkpoint["model_discriminator"])

    solver = Solver(
        name=checkpoint["name"],
        is_multilabel=checkpoint["is_multilabel"],
        topics=topics,
        test_data=test_data,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_predictor=image_topic_predictor,
        text_predictor=text_topic_predictor,
        discriminator=modal_discriminator,
        prev_epoch=checkpoint["epochs"],
        checkpoint_dir=new_checkpoint_dir,
    )
    # solver.optimizer_ie.load_state_dict(checkpoint["optimizer_ie"])
    # solver.optimizer_te.load_state_dict(checkpoint["optimizer_te"])
    solver.optimizer_ip.load_state_dict(checkpoint["optimizer_ip"])
    solver.optimizer_tp.load_state_dict(checkpoint["optimizer_tp"])
    solver.optimizer_d.load_state_dict(checkpoint["optimizer_d"])

    solver.d_losses = checkpoint["d_losses"]
    solver.g_losses = checkpoint["g_losses"]

    solver.map_5_history_i2i = checkpoint["map_5_image"]
    solver.map_5_history_t2t = checkpoint["map_5_text"]
    solver.map_5_history_i2t = checkpoint["map_5_i2t"]
    solver.map_5_history_t2i = checkpoint["map_5_t2i"]

    solver.map_all_history_i2i = checkpoint["map_all_image"]
    solver.map_all_history_t2t = checkpoint["map_all_text"]
    solver.map_all_history_cross = checkpoint["map_all_cross"]

    solver.classification_reports_i2i = checkpoint["report_i2i"]
    solver.classification_reports_t2t = checkpoint["report_t2t"]
    solver.classification_reports_cross = checkpoint["report_cross"]

    return solver
