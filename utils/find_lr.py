import logging
import math
import os
import re
from typing import List, Tuple
import itertools
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)

def find_learning_rate_model(model,
                data_loader,
                optimizer,
                criterion,
                serialization_dir=None,
                start_lr: float = 1e-5,
                end_lr: float = 1,
                num_batches: int = 100,
                linear_steps: bool = False,
                stopping_factor: float = None):
    if os.path.exists(serialization_dir):
        os.makedirs(serialization_dir)

    logger.info(
            f"Starting learning rate search from {start_lr} to {end_lr} in {num_batches} iterations."
        )
    learning_rates, losses = search_learning_rate(
        model,
        data_loader,
        optimizer,
        start_lr=start_lr,
        end_lr=end_lr,
        num_batches=num_batches,
        linear_steps=linear_steps,
        stopping_factor=stopping_factor,
    )
    logger.info("Finished learning rate search.")
    losses = _smooth(losses, 0.98)

    _save_plot(learning_rates, losses, os.path.join(serialization_dir, "lr-losses.png"))


def search_learning_rate(
        model,
        data_loader,
        optimizer,
        criterion,
        start_lr: float = 1e-5,
        end_lr: float = 10,
        num_batches: int = 100,
        linear_steps: bool = False,
        stopping_factor: float = None,
    ) -> Tuple[List[float], List[float]]:
    """
    # Parameters
    start_lr : `float`
        The learning rate to start the search.
    end_lr : `float`
        The learning rate upto which search is done.
    num_batches : `int`
        Number of batches to run the learning rate finder.
    linear_steps : `bool`
        Increase learning rate linearly if False exponentially.
    stopping_factor : `float`
        Stop the search when the current loss exceeds the best loss recorded by
        multiple of stopping factor. If `None` search proceeds till the `end_lr`
    # Returns
    (learning_rates, losses) : `Tuple[List[float], List[float]]`
        Returns list of learning rates and corresponding losses.
        Note: The losses are recorded before applying the corresponding learning rate
    """
    if num_batches <= 10:
        raise ValueError(
            "The number of iterations for learning rate finder should be greater than 10."
        )

    model.train()

    infinite_generator = itertools.cycle(data_loader)
    train_generator_tqdm = Tqdm.tqdm(infinite_generator, total=num_batches)

    learning_rates = []
    losses = []
    best = 1e9
    if linear_steps:
        lr_update_factor = (end_lr - start_lr) / num_batches
    else:
        lr_update_factor = (end_lr / start_lr) ** (1.0 / num_batches)

    device = torch.device("cuda:0")
    for i, (images, training_mask, distance_map) in enumerate(train_generator_tqdm):

        if linear_steps:
            current_lr = start_lr + (lr_update_factor * i)
        else:
            current_lr = start_lr * (lr_update_factor ** i)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self.optimizer.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in param_group["params"]:
                p.grad = None

        cur_batch = images.size()[0]
        images = images.to(device)
        # Forward
        outputs = net(images)  # B1HW
        training_mask = training_mask.to(device, non_blocking=non_blocking)
        distance_map = distance_map.to(device, non_blocking=non_blocking)
        distance_map = distance_map.to(torch.float)

        #
        dice_center, dice_region, weighted_mse_region, loss, dice_bi_region = criterion(outputs, distance_map,
                                                                                        training_mask)

        loss = trainer.batch_outputs(batch, for_training=True)["loss"]
        loss.backward()
        loss = loss.detach().cpu().item()

        if stopping_factor is not None and (math.isnan(loss) or loss > stopping_factor * best):
            logger.info(f"Loss ({loss}) exceeds stopping_factor * lowest recorded loss.")
            break

        norm_gradients(optimizer)
        optimizer.step()

        learning_rates.append(current_lr)
        losses.append(loss)

        if loss < best and i > 10:
            best = loss

        if i == num_batches:
            break

    return learning_rates, losses


def _smooth(values: List[float], beta: float) -> List[float]:
    """ Exponential smoothing of values """
    avg_value = 0.0
    smoothed = []
    for i, value in enumerate(values):
        avg_value = beta * avg_value + (1 - beta) * value
        smoothed.append(avg_value / (1 - beta ** (i + 1)))
    return smoothed


def _save_plot(learning_rates: List[float], losses: List[float], save_path: str):

    try:
        import matplotlib

        matplotlib.use("Agg")  # noqa
        import matplotlib.pyplot as plt

    except ModuleNotFoundError as error:

        logger.warn(
            "To use find-learning-rate, please install matplotlib: pip install matplotlib>=2.2.3 ."
        )
        raise error

    plt.ylabel("loss")
    plt.xlabel("learning rate (log10 scale)")
    plt.xscale("log")
    plt.plot(learning_rates, losses)
    logger.info(f"Saving learning_rate vs loss plot to {save_path}.")
    plt.savefig(save_path)

def norm_gradients(optimizer) -> float:
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    Returns the norm of the gradients.
    """
    parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]
    return torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip])
    )

if __name__ == "__main__":
    from models.craft_test import CRAFT_test
    from models.loss import Loss
    from utils.ranger import Ranger

    model = CRAFT_test(num_out=2, pretrained=True)
    optimizer = Ranger([{'params': model.parameters(), 'initial_lr': 1e-3}], lr=1e-3,
                       weight_decay=5e-4)

    criterion = Loss(OHEM_ratio=3, reduction='mean')


    train_data = IC15Dataset('../../data/IC15/train', data_shape=640, transform=transforms.ToTensor())
    # train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
    #                                num_workers=int(config.workers))

    train_loader = DataLoaderX(dataset=train_data, batch_size=6, shuffle=True,
                               num_workers=10)



    find_learning_rate_model(model, train_loader, optimizer, criterion, './aoto_find_lr',
                             start_lr=1e-5, end_lr=1, num_batches= 100, linear_steps = False, stopping_factor = None)