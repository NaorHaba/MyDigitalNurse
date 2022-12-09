import torch
import torch.nn as nn


class MS_TCN_PP(nn.Module):  # MS_TCN++
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, **kw):
        super(MS_TCN_PP, self).__init__()
        stages = []
        dilations_inc = [2 ** i for i in range(num_layers + 1)]
        dilations_dec = [2 ** i for i in range(num_layers - 1 + 1, -1, -1)]
        dilations = list(zip(dilations_inc, dilations_dec))
        dilation_layer = DualDilatedResidualLayer
        in_dim = dim
        self.softmax = nn.Softmax(dim=1)
        for i in range(num_stages):
            if i != 0:
                in_dim = num_classes
                dilations = dilations_inc[:-1]
                dilation_layer = DilatedResidualLayer
            stages.append(SS_TCN(dilations, dilation_layer, num_f_maps, in_dim, num_classes, **kw))
        self.stages = nn.ModuleList(stages)

    def forward(self, x, mask):
        out = self.stages[0](x, mask) * mask
        outputs = [out]
        for s in self.stages[1:]:
            # passing the output of the previous stage through a softmax layer by task (hand-tool)
            out = out.permute(0, 2, 1)
            out = out.view(out.shape[0], out.shape[1], 4, 5)
            out = self.softmax(out)
            out = out.view(out.shape[0], out.shape[1], 20)
            out = out.permute(0, 2, 1)

            out = out * mask
            out = s(out, mask) * mask
            outputs.append(out)

        return outputs


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, **kw):
        super(MS_TCN, self).__init__()
        stages = []
        in_dim = dim
        dilations = [2 ** i for i in range(num_layers)]
        self.softmax = nn.Softmax(dim=1)
        for i in range(num_stages):
            if i != 0:
                in_dim = num_classes
            stages.append(SS_TCN(dilations, DilatedResidualLayer, num_f_maps, in_dim, num_classes, **kw))
        self.stages = nn.ModuleList(stages)

    def forward(self, x, mask):
        out = self.stages[0](x, mask) * mask
        outputs = [out]
        for s in self.stages[1:]:
            # passing the output of the previous stage through a softmax layer by task (hand-tool)
            out = out.permute(0, 2, 1)
            out = out.view(out.shape[0], out.shape[1], 4, 5)
            out = self.softmax(out)
            out = out.view(out.shape[0], out.shape[1], 20)
            out = out.permute(0, 2, 1)

            out = out * mask
            out = s(out, mask) * mask
            outputs.append(out)

        return outputs


class SS_TCN(nn.Module):
    def __init__(self, dilations, dilated_layer, num_f_maps, dim, num_classes, **kw):
        super(SS_TCN, self).__init__()
        self.gate_in = nn.Conv1d(dim, num_f_maps, 1)
        self.stage = nn.Sequential(
            *[
                dilated_layer(dilation, num_f_maps, num_f_maps, **kw) for dilation in dilations
            ],
        )

        self.gate_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.gate_in(x) * mask
        for sub_stage in self.stage:
            out = sub_stage(out, mask)
        out = self.gate_out(out) * mask
        # returning the results from all the heads
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, activation=nn.ReLU, dropout=0.1):
        super(DilatedResidualLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            activation(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        out = self.layer(x)
        return (x + out) * mask


class DualDilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, activation=nn.ReLU, dropout=0.1):
        super(DualDilatedResidualLayer, self).__init__()
        dilation_inc, dilation_dec = dilation
        self.conv_inc = nn.Conv1d(in_channels, out_channels, 3, padding=dilation_inc, dilation=dilation_inc)
        self.conv_dec = nn.Conv1d(in_channels, out_channels, 3, padding=dilation_dec, dilation=dilation_dec)
        self.conv_to_out = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, 1),
            activation(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        inc = self.conv_inc(x)
        dec = self.conv_dec(x)
        out = self.conv_to_out(torch.cat([inc, dec], dim=1))
        return (x + out) * mask


# defining the final model which comprises a feature extractor and a time series model
class SurgeryModel(nn.Module):
    def __init__(self, feature_extractor, time_series_model):
        super().__init__()
        self.fe = feature_extractor
        self.ts = time_series_model

    def forward(self, x, lengths, mask):
        features = self.fe(x)
        features = torch.cat(features, dim=-1)
        features = features.permute(0, 2, 1)
        features *= mask
        result = self.ts(features, mask)
        return result


# an object that utilizes given feature extractions for each input to extract and concat the features from all inputs
class SeparateFeatureExtractor(nn.Module):
    def __init__(self, top_fe=None, side_fe=None):  # accepts 'nn.Identity'
        super().__init__()
        self.fe = {}
        if top_fe:
            self.top_fe = top_fe
            self.fe['top'] = self.top_fe
        if side_fe:
            self.side_fe = side_fe
            self.fe['side'] = self.side_fe
        assert self.fe, 'must receive at least one feature extractor'

    def forward(self, x):
        features = []
        for key in self.fe:
            try:
                features.append(self.fe[key](x[key]))
            except KeyError:
                raise RuntimeError(f"received feature extractor for {key} but it is not present in x")
        return features
