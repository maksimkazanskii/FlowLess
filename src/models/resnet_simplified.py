import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block used by ResNet-18."""

    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out, inplace=True)

        return out


class ResNet18(nn.Module):
    """
    Configurable ResNet-18 for CIFAR-10 and TinyImageNet.

    Examples
    --------
    Standard-width model:
        ResNet18(num_classes=10, width=1.0)

    Half-width model:
        ResNet18(num_classes=10, width=0.5)

    Half-width TinyImageNet model:
        ResNet18(num_classes=200, width=0.5)
    """

    def __init__(
            self,
            num_classes: int = 10,
            width: float = 1.0,
            use_maxpool: bool = False,
            zero_init_residual: bool = True,
    ):
        super().__init__()

        if width <= 0:
            raise ValueError(
                f"width must be positive, received {width}"
            )

        self.num_classes = num_classes
        self.width = width
        self.use_maxpool = use_maxpool

        c1 = max(8, round(64 * width))
        c2 = max(8, round(128 * width))
        c3 = max(8, round(256 * width))
        c4 = max(8, round(512 * width))

        self.feature_dims = {
            "layer1": c1,
            "layer2": c2,
            "layer3": c3,
            "layer4": c4,
        }

        self.in_channels = c1

        # Small-image stem suitable for CIFAR-10 and TinyImageNet.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=c1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(c1)

        if use_maxpool:
            self.maxpool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            self.maxpool = nn.Identity()

        # ResNet-18 uses a 2-2-2-2 block configuration.
        self.layer1 = self._make_layer(
            out_channels=c1,
            num_blocks=2,
            stride=1,
        )
        self.layer2 = self._make_layer(
            out_channels=c2,
            num_blocks=2,
            stride=2,
        )
        self.layer3 = self._make_layer(
            out_channels=c3,
            num_blocks=2,
            stride=2,
        )
        self.layer4 = self._make_layer(
            out_channels=c4,
            num_blocks=2,
            stride=2,
        )

        self.fc = nn.Linear(c4, num_classes)

        self._initialize_weights(
            zero_init_residual=zero_init_residual
        )

    def _make_layer(
            self,
            out_channels: int,
            num_blocks: int,
            stride: int,
    ) -> nn.Sequential:
        layers = [
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(
            self,
            zero_init_residual: bool,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _forward_stages(
            self,
            x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)

        return z1, z2, z3, z4

    @staticmethod
    def _pool_features(x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return torch.flatten(x, start_dim=1)

    def features(
            self,
            x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Return globally pooled representations from every ResNet stage.

        This preserves the interface expected by the FlowLess code:
            model.features(x)["layer4"]
        """
        z1, z2, z3, z4 = self._forward_stages(x)

        return {
            "layer1": self._pool_features(z1),
            "layer2": self._pool_features(z2),
            "layer3": self._pool_features(z3),
            "layer4": self._pool_features(z4),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, z4 = self._forward_stages(x)

        pooled = self._pool_features(z4)
        logits = self.fc(pooled)

        return logits


if __name__ == "__main__":
    # Simple shape and parameter-count checks.
    cifar_model = ResNet18(
        num_classes=10,
        width=0.5,
        use_maxpool=False,
    )

    tiny_model = ResNet18(
        num_classes=200,
        width=0.5,
        use_maxpool=False,
    )

    cifar_input = torch.randn(4, 3, 32, 32)
    tiny_input = torch.randn(4, 3, 64, 64)

    cifar_logits = cifar_model(cifar_input)
    tiny_logits = tiny_model(tiny_input)

    cifar_features = cifar_model.features(cifar_input)
    tiny_features = tiny_model.features(tiny_input)

    cifar_params = sum(
        parameter.numel()
        for parameter in cifar_model.parameters()
    )

    tiny_params = sum(
        parameter.numel()
        for parameter in tiny_model.parameters()
    )

    print("CIFAR logits:", cifar_logits.shape)
    print("Tiny logits:", tiny_logits.shape)

    print(
        "CIFAR layer4 features:",
        cifar_features["layer4"].shape,
    )
    print(
        "Tiny layer4 features:",
        tiny_features["layer4"].shape,
    )

    print(f"CIFAR parameters: {cifar_params:,}")
    print(f"Tiny parameters: {tiny_params:,}")