import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModule


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


@torch.no_grad()
def get_locations(features, stride, pad_h, pad_w):
    """Position embedding for image pixels.

    Args:
        features (Tensor): (N, C, H, W)
    Returns:
        Tensor: (H, W, 2)
    """
    h, w = features.size()[-2:]
    device = features.device

    shifts_x = (torch.arange(
        0, stride * w, step=stride,
        dtype=torch.float32, device=device
    ) + stride // 2) / pad_w
    shifts_y = (torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    ) + stride // 2) / pad_h
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    locations = locations.reshape(h, w, 2)

    return locations


class WorldModel(BaseModule):
    def __init__(
        self,
        hidden_channel=256,
        dim_feedforward=1024,
        num_heads=8,
        dropout=0.0,
        # pos embedding
        depth_step=0.8,
        depth_num=64,
        depth_start=0,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        stride=32,
        num_views=6,
        num_proposals=6,
        num_tf_layers=2,
        action_dim=12,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.hidden_channel = hidden_channel
        self.num_views = num_views
        self.num_proposals = num_proposals

        self.view_query_feat = nn.Parameter(
            torch.randn(1, self.num_views, hidden_channel, self.num_proposals)
        )

        spatial_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._spatial_decoder = nn.ModuleList(
            [
                nn.TransformerDecoder(spatial_decoder_layer, 1)
                for _ in range(self.num_views)
            ]
        )

        wm_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, num_tf_layers)

        self.action_dim = action_dim
        self.action_aware_encoder = nn.Sequential(
            nn.Linear(hidden_channel + self.action_dim, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.depth_start = depth_start
        self.stride = stride

        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, hidden_channel * 4),
            nn.ReLU(),
            nn.Linear(hidden_channel * 4, hidden_channel),
        )

        self.pc_range = nn.Parameter(torch.tensor(point_cloud_range), requires_grad=False)
        self.position_range = nn.Parameter(
            torch.tensor(position_range), requires_grad=False
        )

        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (
            self.depth_num * (1 + self.depth_num)
        )
        coords_d = self.depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.loss_rec = nn.MSELoss()

    def prepare_location(self, img_metas, img_feats):
        bs, n, c, h, w = img_feats.shape
        pad_h = h * self.stride
        pad_w = w * self.stride

        x = img_feats.flatten(0, 1)
        location = get_locations(x, self.stride, pad_h, pad_w)[None].repeat(
            bs * n, 1, 1, 1
        )
        return location

    def img_position_embeding(self, img_feats, img_metas, projection_mat=None):
        eps = 1e-5
        B, num_views, C, H, W = img_feats.shape
        assert num_views == self.num_views

        if isinstance(img_metas, dict):
            meta = img_metas
        else:
            meta = img_metas[0]

        num_sample_tokens = num_views * H * W
        LEN = num_sample_tokens
        img_pixel_locations = self.prepare_location(img_metas, img_feats)

        pad_h = H * self.stride
        pad_w = W * self.stride

        img_pixel_locations[..., 0] = img_pixel_locations[..., 0] * pad_w
        img_pixel_locations[..., 1] = img_pixel_locations[..., 1] * pad_h

        D = self.coords_d.shape[0]
        pixel_centers = img_pixel_locations.detach().view(B, LEN, 1, 2).repeat(
            1, 1, D, 1
        )
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1, 1)
        coords = torch.cat([pixel_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps
        )

        coords = coords.unsqueeze(-1)

        if projection_mat is not None:
            lidar2img = projection_mat
            if lidar2img.ndim == 3:
                lidar2img = lidar2img.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            if "lidar2img" in meta:
                lidar2img = torch.tensor(
                    np.stack(meta["lidar2img"]), device=img_feats.device
                ).float()
                lidar2img = lidar2img.unsqueeze(0).repeat(B, 1, 1, 1)
            else:
                raise ValueError(
                    "projection_mat must be provided when metadata lacks 'lidar2img'"
                )

        lidar2img = lidar2img[:, :num_views]
        img2lidars = lidar2img.inverse()

        img2lidars = (
            img2lidars.view(B, num_views, 1, 1, 4, 4)
            .repeat(1, 1, H * W, D, 1, 1)
            .view(B, LEN, D, 4, 4)
        )

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (
            coords3d[..., 0:3] - self.position_range[0:3]
        ) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D * 3)

        pos_embed = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        return coords_position_embeding

    def extract_scene_state(self, img_feat, img_metas, projection_mat=None):
        Bz, num_views, num_channels, height, width = img_feat.shape

        init_view_query_feat = (
            self.view_query_feat.clone()
            .repeat(Bz, 1, 1, 1)
            .permute(0, 1, 3, 2)
        )

        img_pos = self.img_position_embeding(img_feat, img_metas, projection_mat)
        img_pos = img_pos.reshape(Bz, num_views, height, width, num_channels)
        img_pos = img_pos.permute(0, 1, 4, 2, 3)

        img_feat_emb = img_feat + img_pos

        img_feat_emb = img_feat_emb.reshape(
            Bz, num_views, num_channels, height * width
        ).permute(0, 1, 3, 2)
        spatial_view_feat = torch.zeros_like(init_view_query_feat)

        for i in range(self.num_views):
            spatial_view_feat[:, i] = self._spatial_decoder[i](
                init_view_query_feat[:, i], img_feat_emb[:, i]
            )

        batch_size, num_view, num_tokens, num_channel = spatial_view_feat.shape
        spatial_view_feat = spatial_view_feat.reshape(batch_size, -1, num_channel)

        return spatial_view_feat

    def forward_prediction(self, current_scene_state, action):
        batch_size, num_tokens, num_channel = current_scene_state.shape
        action_repeated = action.reshape(batch_size, 1, -1).repeat(1, num_tokens, 1)
        cur_view_query_feat_with_ego = torch.cat(
            [current_scene_state, action_repeated], dim=-1
        )
        action_aware_latent = self.action_aware_encoder(cur_view_query_feat_with_ego)
        wm_next_latent = self._wm_decoder(action_aware_latent, action_aware_latent)
        return wm_next_latent

    def loss(self, pred_next_state, target_next_state):
        return self.loss_rec(pred_next_state, target_next_state)
