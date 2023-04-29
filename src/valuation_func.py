import torch
import torch.nn as nn
from neural_utils import LogisticRegression

import config


################################
# Valuation functions for YOLO #
################################


class FCNNColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self):
        super(FCNNColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [x,y,z, (0:3)
                color1, color2, color3, (3:6)
                sphere, 6
                cube, 7
                ]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        # z_color = torch.zeros(size=z[:, 3:6].shape).to(z.device)
        # colors = z[:, 3:6]
        # for c in range(colors.shape[0]):
        #     c_index = torch.argmax(colors[c])
        #     z_color[c, c_index] = 1
        return (a * z[:, 3:6]).sum(dim=1)


class FCNNShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self):
        super(FCNNShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x,y,z, (0:3)
                color1, color2, color3, (3:6)
                sphere, 6
                cube, 7
                ]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 6:10]
        # a_batch = a.repeat((z.size(0), 1))  # one-hot encoding for batch
        return (a * z_shape).sum(dim=1)


class FCNNInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self):
        super(FCNNInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        prob, _ = z[:, 6:10].max(dim=-1)
        return prob


class FCNNRhoValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNRhoValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, dist_grade):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)

        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        dist_id = torch.zeros(rho.shape)

        dist_grade_num = dist_grade.shape[1]
        grade_weight = 1 / dist_grade_num
        for i in range(1, dist_grade_num):
            threshold = grade_weight * i
            dist_id[rho >= threshold] = i

        dist_pred = torch.zeros(dist_grade.shape).to(dist_grade.device)
        for i in range(dist_pred.shape[0]):
            dist_pred[i, int(dist_id[i])] = 1

        return (dist_grade * dist_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def to_center(self, z):
        return torch.stack((z[:, 0], z[:, 2]))


class FCNNPhiValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNPhiValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, dir):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)

        round_divide = dir.shape[1]
        area_angle = int(360 / round_divide)
        area_angle_half = area_angle * 0.5
        # area_angle_half = 0
        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        phi_clock_shift = (90 - phi.long()) % 360
        zone_id = (phi_clock_shift + area_angle_half) // area_angle % round_divide

        # This is a threshold, but it can be decided automatically.
        # zone_id[rho >= 0.12] = zone_id[rho >= 0.12] + round_divide

        dir_pred = torch.zeros(dir.shape).to(dir.device)
        for i in range(dir_pred.shape[0]):
            dir_pred[i, int(zone_id[i])] = 1

        return (dir * dir_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def to_center(self, z):
        return torch.stack((z[:, 0], z[:, 2]))


class FCNNSlopeValuationFunction(nn.Module):
    """The function v_area.
    """

    def __init__(self, device):
        super(FCNNSlopeValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, dir):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_left(z_1)
        c_2 = self.to_right(z_1)

        round_divide = dir.shape[1]
        area_angle = int(180 / round_divide)
        area_angle_half = area_angle * 0.5
        # area_angle_half = 0
        dir_vec = c_2 - c_1
        dir_vec[1] = -dir_vec[1]
        rho, phi = self.cart2pol(dir_vec[0], dir_vec[1])
        phi_clock_shift = (90 - phi.long()) % 180
        zone_id = (phi_clock_shift + area_angle_half) // area_angle % round_divide

        # This is a threshold, but it can be decided automatically.
        # zone_id[rho >= 0.12] = zone_id[rho >= 0.12] + round_divide

        dir_pred = torch.zeros(dir.shape).to(dir.device)
        for i in range(dir_pred.shape[0]):
            dir_pred[i, int(zone_id[i])] = 1

        return (dir * dir_pred).sum(dim=1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    def to_left(self, z):
        return torch.stack(
            (z[:, config.group_tensor_index["screen_left_x"]], z[:, config.group_tensor_index["screen_left_y"]]))

    def to_right(self, z):
        return torch.stack(
            (z[:, config.group_tensor_index["screen_right_x"]], z[:, config.group_tensor_index["screen_right_y"]]))


#####################################################
# Valuation functions for invented predicates       #
#####################################################

class Inv1ValuationFunction(nn.Module):
    """The function v_online.
    """

    def __init__(self, device):
        super(Inv1ValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """The function to compute the probability of the online predicate.

        The closed form of the linear regression is computed.
        The error value is fed into the 1-d logistic regression function.

        Args:
            z_i (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """

        return self.logi(0).squeeze()

    def to_center_x(self, z):
        x = (z[:, 0] + z[:, 2]) / 2
        return x

    def to_center_y(self, z):
        y = (z[:, 1] + z[:, 3]) / 2
        return y


##########################################
# Valuation functions for slot attention #
##########################################


class SlotAttentionInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self, device):
        super(SlotAttentionInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        # return the objectness
        return z[:, 0]


class SlotAttentionShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self, device):
        super(SlotAttentionShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 4:7]
        return (a * z_shape).sum(dim=1)


class SlotAttentionSizeValuationFunction(nn.Module):
    """The function v_size.
    """

    def __init__(self, device):
        super(SlotAttentionSizeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_size = z[:, 7:9]
        return (a * z_size).sum(dim=1)


class SlotAttentionMaterialValuationFunction(nn.Module):
    """The function v_material.
    """

    def __init__(self, device):
        super(SlotAttentionMaterialValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_material = z[:, 9:11]
        return (a * z_material).sum(dim=1)


class SlotAttentionColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(SlotAttentionColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 11:19]
        return (a * z_color).sum(dim=1)


class SlotAttentionRightSideValuationFunction(nn.Module):
    """The function v_rightside.
    """

    def __init__(self, device):
        super(SlotAttentionRightSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness


class SlotAttentionLeftSideValuationFunction(nn.Module):
    """The function v_leftside.
    """

    def __init__(self, device):
        super(SlotAttentionLeftSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness


class SlotAttentionFrontValuationFunction(nn.Module):
    """The function v_infront.
    """

    def __init__(self, device):
        super(SlotAttentionFrontValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=6, output_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        xyz_1 = z_1[:, 1:4]
        xyz_2 = z_2[:, 1:4]
        xyzxyz = torch.cat([xyz_1, xyz_2], dim=1)
        prob = self.logi(xyzxyz).squeeze()  # (B,)
        objectness = z_1[:, 0] * z_2[:, 0]  # (B,)
        return prob * objectness
