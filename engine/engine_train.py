#
import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import numpy as np

#
import gc
from typing import Optional

#
from utils import logger, save_checkpoint, Mix
from optim import enable_running_stats, disable_running_stats


class Trainer(object):
    def __init__(
        self,
        opts,
        model: torch.nn.Module,
        model_ema,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        mix: Mix,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gradient_scaler: torch.cuda.amp.grad_scaler,
        save_dir: str,
        start_epoch: Optional[int] = 0,
        max_epoch: Optional[int] = 50,
        device_type: Optional[str] = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super(Trainer, self).__init__()

        #
        self.opts = opts
        self.model = model
        self.model_ema = model_ema
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.mix = mix
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_scaler = gradient_scaler
        self.save_dir = save_dir
        self.max_epoch = max_epoch
        self.device_type = device_type
        self.start_epoch = start_epoch

        #
        local_rank = getattr(opts, "local_rank", -1)
        self.device = torch.device(self.device_type, local_rank)
        if self.device_type != "cpu":
            # Manually move optimizer to GPU memory, seems like a bug in pytorch
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        print(self.device)
        self.model.to(self.device)
        self.model_ema.to(self.device)

        #
        self.profile_first = getattr(opts, "common.profile_first", False)
        self.enable_mix_precision = getattr(opts, "common.mixed_precision", True)

        # Tensorboard writer
        # Accuracy
        self.best_model_metric = (
            0 if "best_model_metric" not in kwargs else kwargs["best_model_metric"]
        )
        self.best_model_ema_metric = (
            0
            if "best_model_ema_metric" not in kwargs
            else kwargs["best_model_ema_metric"]
        )
        self.log_iter = 0
        self.log_freq = getattr(opts, "common.log_freq", 100)
        self.writer = SummaryWriter(f"{self.save_dir}/tb_logs")

    def train_one_epoch(self, epoch: int):

        #
        logger.info("********************* Training *********************")
        self.model.train()

        #
        if isinstance(self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        #
        loss_history, acc_history = [], []
        for batch_idx, batch in enumerate(self.train_loader):

            # Learning rate
            lr = self.optimizer.param_groups[0]["lr"]

            #
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Mix-based augmentation
            # 1. mode == "none" if we did not use any mix augmentation
            mode, inputs, target_a, target_b, lam = self.mix.forward(inputs, targets)

            # closure function for SAM (Sharpness-Aware Minimization)
            def closure():
                disable_running_stats(self.model)
                loss = self.mix.mix_criterion(
                    mode, self.criterion, self.model(inputs), target_a, target_b, lam
                )
                loss.backward()
                return loss

            #
            enable_running_stats(self.model)
            logits = self.model(inputs)

            # Compute loss and then start back-propagation
            # loss = self.criterion(logits, targets)
            loss = self.mix.mix_criterion(
                mode, self.criterion, logits, target_a, target_b, lam
            )
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            #
            disable_running_stats(self.model)
            self.mix.mix_criterion(
                mode, self.criterion, self.model(inputs), target_a, target_b, lam
            ).backward()
            self.optimizer.second_step(zero_grad=True)

            # EMA
            # update the moving average with the new parameters from the last optimizer step
            self.model_ema.update()

            #
            loss_history.append(loss.detach().item())
            acc_history.append(
                (logits.argmax(dim=-1) == targets).float().mean().detach().item()
            )

            # logger
            if self.log_iter % self.log_freq == 0:
                #
                self.writer.add_scalar(
                    f"Train/Accuracy_iter", acc_history[-1], self.log_iter
                )
                self.writer.add_scalar(
                    f"Train/Loss_iter", loss_history[-1], self.log_iter
                )
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.3f}\tLR: {:.5f}".format(
                        epoch,
                        (batch_idx),
                        len(self.train_loader),
                        100.0 * (batch_idx) / len(self.train_loader),
                        np.mean(loss_history),
                        np.mean(acc_history),
                        lr,
                    )
                )
            self.log_iter += 1

        # Return training metrics
        metric_acc = np.mean(acc_history)
        metric_loss = np.mean(loss_history)
        logger.info("Finish, Accuracy: {}, Loss: {}".format(metric_acc, metric_loss))
        metrics = {"Accuracy": metric_acc, "Loss": metric_loss}
        return metrics

    def valid_one_epoch(self, epoch: int, isEMA: Optional[bool] = False):

        #
        model = self.model if isEMA else self.model_ema
        logger.info("********************* Validation *********************")
        model.eval()

        #
        predictions = []
        loss_history, acc_history = [], []
        with torch.no_grad():
            for batch in self.valid_loader:
                #
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Start model forwarding and then calculate loss
                logits = model(inputs)
                loss = self.criterion(logits, targets)

                #
                loss_history.append(loss.detach().item())
                acc_history.append(
                    (logits.argmax(dim=-1) == targets).float().mean().detach().item()
                )
                predictions.extend((logits.argmax(dim=-1).detach().tolist()))

        # Return training metrics and logs
        metric_acc = np.mean(acc_history)
        metric_loss = np.mean(loss_history)
        logger.info("Finish, Accuracy: {}, Loss: {}".format(metric_acc, metric_loss))
        metrics = {
            "Accuracy": metric_acc,
            "Loss": metric_loss,
            "Predictions": predictions,
        }
        return metrics

    def save_predictions(self, predictions, fileName="predictions.csv"):
        #
        with open(fileName, "w") as fw:
            fw.write(f"ID,Category\n")
            for i in range(len(predictions)):
                fw.write(f"{i},{predictions[i]}\n")

    def write_log(
        self,
        epoch: int,
        lr,
        train_metric,
        valid_metric,
        valid_ema_metric,
        *args,
        **kwargs,
    ) -> None:

        # Training Metric
        self.writer.add_scalar(f"Learing rate", lr, epoch)
        self.writer.add_scalar(f"Train/Accuracy", train_metric["Accuracy"], epoch)
        self.writer.add_scalar(f"Train/Loss", train_metric["Loss"], epoch)

        # Validating Metric
        self.writer.add_scalar(f"Valid/Accuracy", valid_metric["Accuracy"], epoch)
        self.writer.add_scalar(f"Valid/Loss", valid_metric["Loss"], epoch)
        self.writer.add_scalar(
            f"Valid/EMA_Accuracy", valid_ema_metric["Accuracy"], epoch
        )
        self.writer.add_scalar(f"Valid/EMA_Loss", valid_ema_metric["Loss"], epoch)

    def run(self):

        #
        # this zero gradient update is needed to avoid a warning message, issue #8.
        self.optimizer.zero_grad()
        # self.optimizer.step()

        # TODO: Provide model profilier
        # # Do some iterations to profile the whole model
        # if self.profile_first:
        #     self.train_one_epoch_with_profiler()

        #
        for epoch in range(self.start_epoch, self.max_epoch):

            # Forward scheduler one step to alter learning rate
            self.scheduler.step(epoch + 1)

            # Model forward
            # 1. train_one_epoch
            # 2. valid_one_epoch
            # 3. valid_one_epoch (EMA)
            logger.info("=" * 80)
            train_metrics = self.train_one_epoch(epoch)
            valid_metrics = self.valid_one_epoch(epoch)
            valid_ema_metrics = self.valid_one_epoch(epoch, isEMA=True)
            gc.collect()

            # Check whether current model is the best one, which we will store it latter        
            is_best = False
            if valid_metrics["Accuracy"] > self.best_model_metric:
                self.best_model_metric = valid_metrics["Accuracy"]
                self.save_predictions(
                    valid_metrics["Predictions"],
                    fileName=f"{self.save_dir}/predictions.csv",
                )
                is_best = True

            is_ema_best = False
            if valid_ema_metrics["Accuracy"] > self.best_model_ema_metric:
                self.best_model_ema_metric = valid_ema_metrics["Accuracy"]
                self.save_predictions(
                    valid_metrics["Predictions"],
                    fileName=f"{self.save_dir}/predictions_ema.csv",
                )
                is_ema_best = True

            # Model save checkpoint
            # 1. save the current best model, model_ema
            # 2. save the last model, model_ema
            # 3. Do forget to save the info of optimizer, and so on
            if dist.get_rank() == 0:
                save_checkpoint(
                    epoch,
                    self.model,
                    is_best,
                    self.model_ema,
                    is_ema_best,
                    self.optimizer,
                    self.gradient_scaler,
                    self.save_dir,
                    best_model_metric=self.best_model_metric,
                    best_model_ema_metric=self.best_model_ema_metric,
                )

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            self.write_log(epoch, lr, train_metrics, valid_metrics, valid_ema_metrics)

            logger.info("=" * 80)


#
if __name__ == "__main__":
    pass
