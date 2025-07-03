from models.classifier import Classifier
from models.convnextv2 import convnextv2_tiny, remap_checkpoint_keys, load_state_dict
from util.lars import LARS
import torch
import os
from util.convnext_optim import get_parameter_groups, LayerDecayValueAssigner

def build_model(args, device):
    if args.model == "profound_conv":
        convnext = convnextv2_tiny(in_chans=3, drop_path_rate=0.1)
        if args.pretrain is None:
            raise NotImplementedError(f"No pretrained weight")
        if not os.path.exists(args.pretrain):
            raise FileExistsError(f"{args.pretrain} Not exists")
        ckpt = torch.load(args.pretrain, map_location="cpu")
        ckpt = remap_checkpoint_keys(ckpt)
        load_state_dict(convnext, ckpt)
        model = Classifier(convnext, args.num_classes)
        model = model.to(device)
        if args.train == "freeze":
            for key, value in model.encoder.named_parameters():
                value.requires_grad = False
            optimizer = LARS(model.head.parameters(), weight_decay=0, lr=args.lr)
        else:
            num_layers = sum(convnext.depths)
            assigner = LayerDecayValueAssigner(
                list(
                    args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
                ),
                depths=convnext.depths,
                layer_decay_type=args.layer_decay_type,
            )

            skip = {}
            if hasattr(model.encoder, "no_weight_decay"):
                skip = model.encoder.no_weight_decay()

            backbone_param_groups = get_parameter_groups(
                model.encoder,
                args.weight_decay,
                skip,
                assigner.get_layer_id,
                assigner.get_scale,
            )
            decoder_param_groups = [
                {"params": model.head.parameters(), "weight_decay": 0.0, "lr": args.lr}
            ]

            optimizer = torch.optim.AdamW(
                backbone_param_groups + decoder_param_groups, lr=args.lr
            )

    else:
        raise NotImplementedError(f"unknown model: {args.model}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    return model, optimizer


def vit_backbone_parameters(
    model: torch.nn.Module, weight_decay=1e-5, no_weight_decay_list=(), lr=1e-3
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
    ]
