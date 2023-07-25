import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.common.log import LOGGER
from PIL import Image
from torchvision import transforms

from RFNet.train import Trainer
from RFNet.eval import Validator, load_my_state_dict
from RFNet.dataloaders import custom_transforms as tr
from RFNet.dataloaders import make_data_loader
from RFNet.utils.args import TrainArgs, ValArgs

# set backend
os.environ['BACKEND_TYPE'] = 'PYTORCH'


@ClassFactory.register(ClassType.GENERAL, alias="BaseModel")
class BaseModel:
    def __init__(self, **kwargs):
        self.train_args = TrainArgs(**kwargs)
        self.trainer = None

        self.val_args = ValArgs(**kwargs)
        label_save_dir = Context.get_parameters("INFERENCE_RESULT_DIR", "./inference_results")
        self.val_args.color_label_save_path = os.path.join(label_save_dir, "color")
        self.val_args.merge_label_save_path = os.path.join(label_save_dir, "merge")
        self.val_args.label_save_path = os.path.join(label_save_dir, "label")
        self.validator = Validator(self.val_args)

    def get_weights(self):
        return self.trainer.get_weight()
    
    def set_weights(self, weights):
        self.trainer.set_weight(weights)
        
        epoch_num = 0
        print("Total epoch: ", epoch_num)
        loss_all = []
        for epoch in range(epoch_num):
            train_loss = self.trainer.my_training(epoch)
            #train_loss = self.trainer.training(epoch)
            loss_all.append(train_loss)
        with open('/home/shijing.hu/ianvs/project/ianvs/train_loss_2.txt', 'a+') as file:
            np.savetxt(file, loss_all)
        file.close

    def train(self, train_data, valid_data=None, **kwargs):
        self.trainer = Trainer(self.train_args, train_data=train_data)
        print("Total epoches:", self.trainer.args.epochs)
        loss_all = []
        for epoch in range(
                self.trainer.args.start_epoch,
                self.trainer.args.epochs):
            if epoch == 0 and self.trainer.val_loader:
                self.trainer.validation(epoch)
            if self.train_args.resume:
                print("my_training:", self.train_args.resume)
                loss = self.trainer.my_training(epoch)
                #loss = self.trainer.training(epoch)
            else:
                loss = self.trainer.training(epoch)
            loss_all.append(loss)
            if self.trainer.args.no_val and (
                    epoch %
                    self.trainer.args.eval_interval == (
                            self.trainer.args.eval_interval -
                            1) or epoch == self.trainer.args.epochs -
                    1):
                # save checkpoint when it meets eval_interval or the training
                # finished
                is_best = False
                self.train_model_url = self.trainer.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.trainer.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'best_pred': self.trainer.best_pred,
                }, is_best)

        self.trainer.writer.close()
        with open('/home/shijing.hu/ianvs/project/ianvs/train_loss.txt', 'a+') as file:
            np.savetxt(file, loss_all)
        file.close
        return self.train_model_url

    def predict(self, data, **kwargs):
        if len(data) > 10:
            print("predict start for big data")
            my_kwargs = {'num_workers': self.val_args.workers, 'pin_memory': True}
            _, _, self.validator.test_loader, _ = make_data_loader(self.val_args, test_data=data, **my_kwargs)
        else:
            print("predict start for small data")
            if not isinstance(data[0][0], dict):
                data = self._preprocess(data)
                #print("predict starting 69")
            if type(data) is np.ndarray:
                data = data.tolist()
                #print("predict starting 72")
            #print("predict starting 73")
            self.validator.test_loader = DataLoader(data, batch_size=self.val_args.test_batch_size, shuffle=False,
                                                pin_memory=True)
        
        #print("predict starting 75")
        return self.validator.validate()

    def evaluate(self, data, **kwargs):
        print("evaluate starting 77")
        self.val_args.save_predicted_image = kwargs.get("save_predicted_image", True)
        samples = self._preprocess(data.x)
        predictions = self.predict(samples)
        metric_name, metric_func = kwargs.get("metric")
        if callable(metric_func):
            return metric_func(data.y, predictions)
        else:
            raise Exception(f"not found model metric func(name={metric_name}) in model eval phase")

    def load(self, model_url, **kwargs):
        if model_url:
            print("load model url: ",model_url)
            self.validator.new_state_dict = torch.load(model_url, map_location=torch.device("cpu"))
            self.train_args.resume = model_url
        else:
            raise Exception("model url does not exist.")
        self.validator.model = load_my_state_dict(self.validator.model, self.validator.new_state_dict['state_dict'])

    def save(self, model_path=None):
        # TODO: save unstructured data model
        if not model_path:
            LOGGER.warning(f"Not specify model path.")
            return self.train_model_url

        return FileOps.upload(self.train_model_url, model_path)

    def _preprocess(self, image_urls):
        transformed_images = []
        for paths in image_urls:
            if len(paths) == 2:
                img_path, depth_path = paths
                _img = Image.open(img_path).convert('RGB')
                _depth = Image.open(depth_path)
                #print(img_path, depth_path)
            else:
                img_path = paths[0]
                #print(img_path)
                _img = Image.open(img_path).convert('RGB')
                _depth = _img

            sample = {'image': _img, 'depth': _depth, 'label': _img}
            del _img
            gc.collect()
            composed_transforms = transforms.Compose([
                # tr.CropBlackArea(),
                # tr.FixedResize(size=self.args.crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])

            transformed_images.append((composed_transforms(sample), img_path))

        return transformed_images
