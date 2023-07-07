
import os

from lightning import Trainer
from torch.utils.data import DataLoader

from datasets import load_dataset
from model import load_model, load_series_model
from utils.losses import load_criteria


def test_worker(args):
    kfold = 5 if args.kfold else 1
    save_path = os.path.join(args.save_name, str(args.seed))

    for fold in range(kfold):
        args.fold = fold
        test_dataset = load_dataset(args, fold=fold, train=False)
        model = load_model(criteria=[None], args=args)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        # load corresponding model
        best_model_path = "{}/model_{}_segmentation_best_{}.ckpt".format(
            save_path, args.dataset_name, fold
        )
        model.load_weights(best_model_path)

        trainer = Trainer(devices=[args.gpu[0]], logger=args.logger)
        trainer.test(model, dataloaders=test_loader)


def test_aug_worker(args, aug_k=40, aug_n=1):
    eval_metric = args.eval_metric
    if aug_n > aug_k / 2:
        print('Exceed situation')
        return 10

    kfold = 5 if args.kfold else 1

    for fold in range(kfold):
        test_dataset = load_dataset(
            args, fold=fold, train=False, aug_k=aug_k, aug_n=aug_n
        )