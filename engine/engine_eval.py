#
import torch
import torch.cuda.amp as amp
import numpy as np

#
import gc
from tqdm import tqdm
from typing import Optional

#
from utils import logger


class Evaluator(object):
    def __init__(
        self,
        opts,
        model: torch.nn.Module,
        model_ema,
        eval_loader: torch.utils.data.DataLoader,
        img_names: Optional[list[str]] = None,
        label2class: Optional[dict] = None,
        device_type: Optional[str] = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super(Evaluator, self).__init__()

        #
        self.opts = opts
        self.model = model
        self.model_ema = model_ema
        self.eval_loader = eval_loader
        self.img_names = img_names
        self.label2class = label2class
        self.device_type = device_type

        #
        self.device = torch.device(self.device_type)
        self.model = self.model.to(self.device)
        self.model_ema = self.model_ema.to(self.device)

        # #
        # self.enable_mix_precision = getattr(opts, "common.mixed_precision", True)

        #
        if self.label2class:
            logger.info(f"{self.label2class}")
        if self.img_names:
            logger.info(f"{self.imgs}")

    def eval_one_epoch(self, isEMA: Optional[bool] = False):

        #
        model = self.model if isEMA else self.model_ema
        logger.info("********************* Evaluating *********************")
        model.eval()

        #
        predictions = []
        with torch.no_grad():
            for batch in tqdm(self.eval_loader):
                #
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                #
                logits = model(inputs)

                #
                predictions_batch = logits.argmax(dim=-1).detach().tolist()
                predictions.extend(predictions_batch)

        return predictions

    def export_prediction(self, predictions, fileName: str) -> None:
        """ """

        logger.info(f"Start store the predictions into file {fileName} ...")
        with open(fileName, "w") as file:
            # Title of prediction csv file
            file.write("filename,label\n")
            #
            for i in range(len(predictions)):
                # Replace with the row name
                query = f"{self.img_names[i]}" if self.img_names != None else str(i)
                value = (
                    predictions[i]
                    if self.label2class == None
                    else self.label2class[predictions[i]]
                )
                file.write("{},{}\n".format(query, value))

        logger.info(f"Finish store the predictions into file {fileName}")

    def run(self):
        """ """

        logger.info("=" * 80)

        # Generate predictions with model
        predictions = self.eval_one_epoch()
        gc.collect()
        self.export_prediction(predictions, "prediction.csv")

        # Generate predictions with model_ema
        predictions = self.eval_one_epoch(isEMA=True)
        gc.collect()
        self.export_prediction(predictions, "prediction_ema.csv")

        logger.info("=" * 80)


#
if __name__ == "__main__":
    pass
