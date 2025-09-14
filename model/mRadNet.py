import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.metaformer import MetaFormerBlock, StarReLU


@torch.compile
class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: int = 2,
        act1_layer: type[nn.Module] = StarReLU,
        act2_layer: type[nn.Module] = nn.Identity,
        bias: bool = False,
        kernel_size: int | tuple[int, int, int] = (3, 7, 7),
        padding: int | tuple[int, int, int] = (1, 3, 3),
        **kwargs
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv3d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, T, H, W, C] -> [B, C, T, H, W]
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] -> [B, T, H, W, C]
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


@torch.compile
class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        num_heads: int | None = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        proj_bias: bool = False,
        **kwargs
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        N = T * H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, T, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TokenEmbed(nn.Module):

    def __init__(
        self,
        patch_size: list[int],
        in_channels: int,
        embed_dim: int,
        norm_layer: type[nn.Module],
    ):
        super().__init__()
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.conv = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 4, 1, 2, 3)  # B T H W C -> B C T H W
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)  # B C T H W -> B T H W C
        x = self.norm(x)
        return x


class TokenMerging(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: type[nn.Module]
    ):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.linear(x)
        return x


class mRadNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: list[int],
        depths: list[int],
        dims: list[int]
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.num_stages = len(depths)

        self.input = TokenEmbed(
            patch_size, in_channels=in_channels,
            embed_dim=dims[0], norm_layer=nn.LayerNorm
        )

        token_mixers = [SepConv] * 2 + [Attention] * 2

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages.append(
                nn.Sequential(*[
                    MetaFormerBlock(
                        dims[i],
                        token_mixer=token_mixers[i],
                        norm_layer=nn.LayerNorm,
                        proj_drop=0.1,
                        use_nchw=False)
                    for _ in range(depths[i])
                ])
            )

        self.downsamples = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.downsamples.append(
                TokenMerging(dims[i], norm_layer=nn.LayerNorm)
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: B T H W C
        x = self.input(x)

        features: list[torch.Tensor] = []
        for i in range(self.num_stages):
            x = self.stages[i](x)
            features.append(x)  # B T H W C

            if i == self.num_stages - 1:
                break

            x = self.downsamples[i](x)
        return features


class mRadNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        depths: list[int],
        embed_dim: int,
        out_channels: int,
        patch_size: list[int]
    ):
        super().__init__()
        num_stages = len(depths)

        self.expands = nn.ModuleList()
        for i in range(num_stages):
            self.expands.append(nn.Sequential(
                nn.GroupNorm(embed_dim // 2, embed_dim * 2),
                nn.ConvTranspose3d(
                    embed_dim*2, embed_dim, (1, 2, 2),
                    stride=(1, 2, 2)
                ),
                nn.GELU()
            ))

        token_mixers = [Attention] * 1 + [SepConv] * 2

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                nn.Sequential(*[
                    MetaFormerBlock(
                        embed_dim*2,
                        token_mixer=token_mixers[i],
                        norm_layer=nn.LayerNorm,
                        proj_drop=0.1,
                        use_nchw=False)
                    for _ in range(depths[i])
                ])
            )

        self.linears = nn.ModuleList()
        for i in range(num_stages + 1):
            self.linears.append(nn.Linear(
                in_channels[-i-1], embed_dim*2 if i == 0 else embed_dim
            ))

        self.expand_final = nn.Sequential(
            nn.Upsample(scale_factor=tuple(patch_size), mode="trilinear"),
            nn.GroupNorm(embed_dim // 2, embed_dim * 2),
            nn.Conv3d(embed_dim*2, embed_dim, 1),
            nn.GELU()
        )
        self.out = nn.Conv3d(
            embed_dim, out_channels, 3, padding=1
        )  # DO NOT REMOVE

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = self.linears[0](features[-1])
        for i in range(0, len(features)-1):
            x = x.permute(0, 4, 1, 2, 3)  # B T H W C -> B C T H W
            x = self.expands[i](x)
            x = x.permute(0, 2, 3, 4, 1)  # B C T H W -> B T H W C
            x = torch.cat((x, self.linears[i+1](features[-i-2])), dim=-1)
            x = self.stages[i](x)
        x = x.permute(0, 4, 1, 2, 3)  # B T H W C -> B C T H W
        x = self.expand_final(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 4, 1)  # B C T H W -> B T H W C
        return x


class mRadNet(nn.Module):
    def __init__(
        self,
        model_cfg: dict,
        dataset_cfg: dict
    ):
        super().__init__()
        encoder_cfg = model_cfg['encoder']
        decoder_cfg = model_cfg['decoder']
        num_chirps = len(dataset_cfg['chirps'])  # 4

        self.stem = nn.Conv3d(
            2, 2 * num_chirps, (num_chirps, 1, 1),
            stride=(num_chirps, 1, 1)
        )

        self.encoder = mRadNetEncoder(
            in_channels=2 * num_chirps,
            patch_size=encoder_cfg['patch_size'],
            depths=encoder_cfg['depths'],
            dims=encoder_cfg['dims']
        )
        self.decoder = mRadNetDecoder(
            in_channels=encoder_cfg['dims'],
            depths=decoder_cfg['depths'],
            embed_dim=decoder_cfg['embed_dim'],
            out_channels=len(dataset_cfg['class_names']),
            patch_size=encoder_cfg['patch_size']
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, chirp, H, W, C = x.shape
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B T C chirp H W
        x_in = torch.zeros((B, T, 2 * chirp, H, W), device=x.device)
        for i in range(T):
            # B T C chirp H W -> B T 8 H W
            x_in[:, i] = self.stem(x[:, i]).squeeze(2)
        x_in = x_in.permute(0, 1, 3, 4, 2)  # B T 8 H W -> B T H W 8

        features = self.encoder(x_in)
        output = self.decoder(features)
        return {'output': output}
