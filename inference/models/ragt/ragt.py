import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock
from inference.models.attention import SpatialTransformer
from .mobile_vit import get_model


class RAGT(GraspModel):

    def __init__(
        self,
        input_channels=4,
        output_channels=1,
        channel_size=18,
        dropout=False,
        prob=0.0,
    ):
        super(RAGT, self).__init__()
        self.mobile_vit = get_model()

        # Upsampling layers to increase spatial dimensions
        self.upsample_layers = nn.Sequential(
            nn.Upsample(scale_factor=33, mode="bilinear", align_corners=False),
            nn.ReLU(),
        )

        self.pos_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )
        self.cos_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )
        self.sin_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )
        self.width_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = self.mobile_vit(x_in)
        x = self.upsample_layers(x)
        x = x[:, :, :225, :225]

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output


class TransRAGT(GraspModel):

    def __init__(
        self,
        input_channels=4,
        output_channels=1,
        channel_size=18,
        dropout=False,
        prob=0.0,
    ):
        super(TransRAGT, self).__init__()
        self.mobile_vit = get_model()
        for param in self.mobile_vit.parameters():
            param.requires_grad = False

        self.up_conv = nn.Conv2d(
            in_channels=18, out_channels=channel_size, kernel_size=1
        )

        self.res1 = ResidualBlock(channel_size, channel_size)
        self.st1 = SpatialTransformer(
            channel_size, 1, 1, 512
        )  # Add SpatialTransformer layer after res1
        self.res2 = ResidualBlock(channel_size, channel_size)
        self.st2 = SpatialTransformer(
            channel_size, 1, 1, 512
        )  # Add SpatialTransformer layer after res2
        self.res3 = ResidualBlock(channel_size, channel_size)
        self.st3 = SpatialTransformer(channel_size, 1, 1, 512)

        # Upsampling layers to increase spatial dimensions
        self.upsample_layers = nn.Sequential(
            nn.Upsample(scale_factor=33, mode="bilinear", align_corners=False),
            nn.ReLU(),
        )

        self.pos_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )
        self.cos_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )
        self.sin_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )
        self.width_output = nn.Conv2d(
            in_channels=channel_size, out_channels=output_channels, kernel_size=2
        )

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):

        x_in, cond = x_in[0], x_in[1]

        x = self.mobile_vit(x_in)
        x = self.up_conv(x)
        x = self.res1(x)
        x = self.st1(x, cond)  # Apply SpatialTransformer layer after res1
        x = self.res2(x)
        x = self.st2(x, cond)  # Apply SpatialTransformer layer after res2
        x = self.res3(x)
        x = self.st3(x, cond)

        x = self.upsample_layers(x)
        x = x[:, :, :225, :225]


        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
