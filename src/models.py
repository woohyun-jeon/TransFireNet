import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TransFireNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False):
        super(TransFireNet, self).__init__()
        self.pvt = timm.create_model('pvt_v2_b2', pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.channels = self.pvt.feature_info.channels()

        decoder_channels = [256, 128, 64, 32]
        self.decoder = nn.ModuleList()

        for idx in range(len(decoder_channels)):
            in_ch = self.channels[-1] if idx == 0 else decoder_channels[idx - 1]
            skip_ch = self.channels[-(idx + 2)] if idx < len(self.channels) - 1 else 0
            out_ch = decoder_channels[idx]

            total_ch = in_ch + skip_ch * 2 if skip_ch > 0 else in_ch

            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(total_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )

        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, 1)
        )

    def forward(self, x1, x2):
        feats_x1 = self.pvt(x1)
        feats_x2 = self.pvt(x2)

        diff_feats = []
        for f1, f2 in zip(feats_x1, feats_x2):
            diff = torch.abs(f1 - f2)
            diff_feats.append(diff)

        x = diff_feats[-1]

        for idx, decoder_block in enumerate(self.decoder):
            if idx < len(diff_feats) - 1:
                skip_diff = diff_feats[-(idx + 2)]
                skip_orig = feats_x2[-(idx + 2)]

                x = F.interpolate(x, size=skip_diff.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_diff, skip_orig], dim=1)

            x = decoder_block(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final(x)

        return x


if __name__ == "__main__":
    in_channels = 3
    num_classes = 4
    batch_size = 2
    input_size = 256

    x1 = torch.randn(batch_size, in_channels, input_size, input_size)
    x2 = torch.randn(batch_size, in_channels, input_size, input_size)

    model = TransFireNet(in_channels, num_classes, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(x1, x2)
    print(f"TransFireNet Output shape: {out.shape}")
