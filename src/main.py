import argparse
import logging
import numpy as np
import pytz
import torch.cuda
import typing
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split

from datasets.coco_org import Coco2017Dataset
from models.solver import get_solver, get_solver_from_checkpoint


def run_solver(args: argparse.Namespace) -> None:
    logging.info("Loading dataset")
    train_dataset, test_dataset, topics_list = load_dataset(args.data, args.data_dir, args.is_multilabel, args.max_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    if args.load_checkpoint:
        logging.info(f"Loading checkpoint from {args.load_checkpoint}")
        solver = get_solver_from_checkpoint(test_data=test_dataloader, topics=topics_list, checkpoint_path=args.load_checkpoint,
                                            new_checkpoint_dir=args.save_checkpoint)
    else:
        logging.info("Initialising solver")
        solver = get_solver(name=args.data, is_multilabel=args.is_multilabel, topics=topics_list, test_data=test_dataloader, learn_rate=args.learn_rate,
                            d_coef=args.d_loss_coef, image_model=args.image_model, text_model=args.text_model, checkpoint_dir=args.save_checkpoint)

    logging.info(f"Training: epochs: {args.num_epochs}, batch size: {args.batch_size}, no. of batches: {len(train_dataloader)}")
    solver.train(args.num_epochs, train_dataloader)


def load_dataset(dataset_name: str, data_dir: str, is_multilabel: bool = True, max_samples: int = None) -> typing.Tuple[
    Dataset, Dataset, typing.Union[typing.List[str], np.ndarray]]:
    # if dataset_name == "laion_coco":
    #     df = pd.read_csv(data_dir)
    #     dataset = LaionCocoDataset(df, is_multilabel)
    if dataset_name == "coco2017":
        dataset = Coco2017Dataset(data_dir, max_samples=max_samples)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    topics_list = dataset.topics
    train, test = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(117))

    logging.info(f"Loaded {len(dataset)} samples, {dataset.num_topics} topics, train size: {len(train)}, test size: {len(test)}")

    return train, test, topics_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Dataset")
    parser.add_argument("--data_dir", type=str, help="Path to dataframe")
    parser.add_argument("--save_checkpoint", type=str, help="Dir to save checkpoint to", required=False)
    parser.add_argument("--load_checkpoint", type=str, help="Path to load checkpoint from", required=False)
    parser.add_argument("--image_model", type=str, help="Image model to use", required=False, default="clip", choices=["clip", "beit"])
    parser.add_argument("--text_model", type=str, help="Text model to use", required=False, default="clip", choices=["clip", "roberta"])
    parser.add_argument("--is_multilabel", action=argparse.BooleanOptionalAction, help="Uses multilabel topics")
    parser.add_argument("--learn_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--d_loss_coef", default=1, type=float, help="Coefficient for discriminator loss")
    parser.add_argument("--max_samples", default=None, type=int, required=False, help="Max number of samples to use")

    args = parser.parse_args()

    start = datetime.now(tz=pytz.timezone("Asia/Singapore"))
    log_path = f"./logs/log-{start.strftime('%Y-%m-%d %H:%M:%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] [%(levelname)-6.6s]  %(message)s",
        datefmt="%Y-%m-%d-%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ]
    )

    logging.info(f"Log saved to: {log_path}")
    logging.info(f"Configuration: {args}")
    logging.info(f"CUDA available ? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"Number of devices: {torch.cuda.device_count()}")

    logging.info("Starting training")
    run_solver(args)

    end = datetime.now(tz=pytz.timezone("Asia/Singapore"))
    duration = end - start
    logging.info(f"Completed in: {str(duration)}")
