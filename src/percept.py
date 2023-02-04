import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

from slot_attention.model import SlotAttention_model
import sys
import config

sys.path.insert(0, 'src/yolov5')


class YOLOPerceptionModule(nn.Module):
    """A perception module using YOLO.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
    """

    def __init__(self, e, d, device, train=False):
        super().__init__()
        self.e = e  # num of entities
        self.d = d  # num of dimension
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        self.model = self.load_model(
            path=str(config.root) + '/src/weights/yolov5/best.pt', device=device)
        # function to transform e * d shape, YOLO returns class labels,
        # it should be decomposed into attributes and the probabilities.
        self.preprocess = YOLOPreprocess(device)

    def load_model(self, path, device):
        print("Loading YOLO model...")
        yolo_net = attempt_load(weights=path)
        yolo_net.to(device)
        if not self.train_:
            for param in yolo_net.parameters():
                param.requires_grad = False
        return yolo_net

    def forward(self, imgs):
        pred = self.model(imgs)[0]  # yolo model returns tuple
        # yolov5.utils.general.non_max_supression returns List[tensors]
        # with lengh of batch size
        # the number of objects can vary image to iamge
        yolo_output = self.pad_result(
            non_max_suppression(pred, max_det=self.e))
        return self.preprocess(yolo_output)

    def pad_result(self, output):
        """Padding the result by zeros.
            (batch, n_obj, 6) -> (batch, n_max_obj, 6)
        """
        padded_list = []
        for objs in output:
            if objs.size(0) < self.e:
                diff = self.e - objs.size(0)
                zero_tensor = torch.zeros((diff, 6)).to(self.device)
                padded = torch.cat([objs, zero_tensor], dim=0)
                padded_list.append(padded)
            else:
                padded_list.append(objs)
        return torch.stack(padded_list)


class SlotAttentionPerceptionModule(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        e (int): The maximum number of entities.
        d (int): The dimension of the object-centric vector.
        device (device): The device where the model and tensors are loaded.
        train (bool): The flag if the parameters are trained.
        preprocess (tensor->tensor): Reshape the yolo output into the unified format of the perceptiom module.
        model: The slot attention model.
    """

    def __init__(self, e, d, device, train=False):
        super().__init__()
        self.e = e  # num of entities -> n_slots=10
        self.d = d  # num of dimension -> encoder_hidden_channels=64
        self.device = device
        self.train_ = train  # the parameters should be trained or not
        self.model = self.load_model()

    def load_model(self):
        """Load slot attention network.
        """
        if self.device == torch.device('cpu'):
            sa_net = SlotAttention_model(n_slots=10, n_iters=3, n_attr=18,
                                         encoder_hidden_channels=64,
                                         attention_hidden_channels=128, device=self.device)
            log = torch.load(
                "src/weights/slot_attention/best.pt", map_location=torch.device(self.device))
            sa_net.load_state_dict(log['weights'], strict=True)
            sa_net.to(self.device)
            if not self.train_:
                for param in sa_net.parameters():
                    param.requires_grad = False
            return sa_net
        else:
            sa_net = SlotAttention_model(n_slots=10, n_iters=3, n_attr=18,
                                         encoder_hidden_channels=64,
                                         attention_hidden_channels=128, device=self.device)
            log = torch.load("src/weights/slot_attention/best.pt")
            sa_net.load_state_dict(log['weights'], strict=True)
            sa_net.to(self.device)
            if not self.train_:
                for param in sa_net.parameters():
                    param.requires_grad = False
            return sa_net

    def forward(self, imgs):
        return self.model(imgs)


class YOLOPreprocess(nn.Module):
    """A perception module using Slot Attention.

    Attrs:
        device (device): The device where the model to be loaded.
        img_size (int): The size of the (resized) image to normalize the xy-coordinates.
        classes (list(str)): The classes of objects.
        colors (tensor(int)): The one-hot encodings of the colors (repeated 3 times).
        shapes (tensor(int)): The one-hot encodings of the shapes (repeated 3 times).
    """

    def __init__(self, device, img_size=128):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.classes = ['red square', 'red circle', 'red triangle',
                        'yellow square', 'yellow circle', 'yellow triangle',
                        'blue square', 'blue circle', 'blue triangle']
        self.colors = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([0, 0, 1]).to(device)
        ])
        self.shapes = torch.stack([
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device),
            torch.tensor([1, 0, 0]).to(device),
            torch.tensor([0, 1, 0]).to(device),
            torch.tensor([0, 0, 1]).to(device)
        ])

    def forward(self, x):
        """A preprocess funciton for the YOLO model. The format is: [x1, y1, x2, y2, prob, class].

        Args:
            x (tensor): The output of the YOLO model. The format is:

        Returns:
            Z (tensor): The preprocessed object-centric representation Z. The format is: [x1, y1, x2, y2, color1, color2, color3, shape1, shape2, shape3, objectness].
            x1,x2,y1,y2 are normalized to [0-1].
            The probability for each attribute is obtained by copying the probability of the classification of the YOLO model.
        """
        batch_size = x.size(0)
        obj_num = x.size(1)
        object_list = []
        for i in range(obj_num):
            zi = x[:, i]
            class_id = zi[:, -1].to(torch.int64)
            color = self.colors[class_id] * zi[:, -2].unsqueeze(-1)
            shape = self.shapes[class_id] * zi[:, -2].unsqueeze(-1)
            xyxy = zi[:, 0:4] / self.img_size
            prob = zi[:, -2].unsqueeze(-1)
            obj = torch.cat([xyxy, color, shape, prob], dim=-1)
            object_list.append(obj)
        return torch.stack(object_list, dim=1).to(self.device)


def eval_images(args, model_file, device, pos_loader, neg_loader):
    if os.path.exists(model_file):
        pm_res = torch.load(model_file)
        pos_pred = pm_res['pos_res']
        neg_pred = pm_res['neg_res']

    else:
        prop_dim = 11
        # perception model
        pm = YOLOPerceptionModule(e=args.e, d=prop_dim, device=device)

        # positive image evaluation
        N_data = 0
        pos_pred = torch.zeros((pos_loader.dataset.__len__(), args.e, prop_dim)).to(device)
        for i, sample in tqdm(enumerate(pos_loader, start=0)):
            imgs, target_set = map(lambda x: x.to(device), sample)
            # print(NSFR.clauses)
            img_array = imgs.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
            img_array_int8 = np.uint8(img_array * 255)
            img_pil = Image.fromarray(img_array_int8)
            # img_pil.show()
            N_data += imgs.size(0)
            B = imgs.size(0)
            # C * B * G
            # when evaluate a clause which its body contains invented predicates,
            # the invented predicates shall be evaluated with all the clauses which head contains the predicate.
            res = pm(imgs)
            pos_pred[i, :] = res

            # negative image evaluation
        N_data = 0
        neg_pred = torch.zeros((neg_loader.dataset.__len__(), args.e, prop_dim)).to(device)
        for i, sample in tqdm(enumerate(neg_loader, start=0)):
            imgs, target_set = map(lambda x: x.to(device), sample)
            # print(NSFR.clauses)
            img_array = imgs.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
            img_array_int8 = np.uint8(img_array * 255)
            img_pil = Image.fromarray(img_array_int8)
            # img_pil.show()
            N_data += imgs.size(0)
            B = imgs.size(0)
            # C * B * G
            # when evaluate a clause which its body contains invented predicates,
            # the invented predicates shall be evaluated with all the clauses which head contains the predicate.
            res = pm(imgs)
            neg_pred[i, :] = res

        # save tensors
        pm_res = {'pos_res': pos_pred.detach(),
                  'neg_res': neg_pred.detach()}
        torch.save(pm_res, str(model_file))

    return pos_pred, neg_pred
