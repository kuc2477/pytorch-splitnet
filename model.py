import itertools
import copy
import operator
from functools import reduce, partial
from torch import nn
import splits


# =====================================
# Building Blocks (Split Regularizable)
# =====================================

class WeightRegularized(nn.Module):
    def reg_losses(self):
        """
        Should return an iterable of (OVERLAP_LOSS, UNIFORM_LOSS, SPLIT_LOSS)
        triples.
        """
        raise NotImplementedError

    @property
    def is_cuda(self):
        # Simple hack to see if the module is built with CUDA parameters.
        if hasattr(self, '__cuda_flag_cache'):
            return self.__cuda_flag_cache
        self.__cuda_flag_cache = next(self.parameters()).is_cuda
        return self.__cuda_flag_cache


class RegularizedLinear(WeightRegularized):
    def __init__(self, in_channels, out_channels,
                 split_size=None, split_qa=None, dropout_prob=.5):
        super().__init__()
        self.split_size = split_size
        self.splitted = self.split_size is not None
        self.use_merged_q = split_qa is not None

        # Layers.
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_prob)

        # Split indicator alphas.
        if split_size:
            self.pa = splits.alpha(split_size, in_channels)
            self.qa = (
                split_qa or
                splits.alpha(split_size, out_channels)
            )
        else:
            self.pa = None
            self.qa = None

    def p(self):
        return splits.q(self.pa)

    def q(self):
        return (
            splits.merge_q(splits.q(self.qa), self.split_size)
            if self.use_merged_q else splits.q(self.qa)
        )

    def forward(self, x):
        return self.dropout(self.linear(x))

    def reg_losses(self):
        return [splits.reg_loss(
            self.linear.weight, self.p(), self.q(), cuda=self.is_cuda
        )]


class ResidualBlock(WeightRegularized):
    def __init__(self, in_channels, out_channels, stride,
                 split_size=None, split_qa=None, dropout_prob=.5):
        super().__init__()
        self.split_size = split_size
        self.splitted = self.split_size is not None
        self.use_merged_q = split_qa is not None

        # 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_prob)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        # 2
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout_prob)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # transformation
        self.need_transform = in_channels != out_channels
        self.conv_transform = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

        # weight
        self.w1 = self.conv1.weight
        self.w2 = self.conv2.weight
        self.w3 = self.conv_transform.weight if self.need_transform else None

        # split indicators
        if split_size:
            self.pa = splits.alpha(split_size, in_channels)
            self.ra = splits.alpha(split_size, out_channels)
            self.qa = (
                split_qa if split_qa is not None else
                splits.alpha(split_size, out_channels)
            )
        else:
            self.pa = None
            self.ra = None
            self.qa = None

    def p(self):
        return splits.q(self.pa)

    def r(self):
        return splits.q(self.ra)

    def q(self):
        return (
            splits.merge_q(splits.q(self.qa), self.split_size)
            if self.use_merged_q else splits.q(self.qa)
        )

    def forward(self, x):
        # conv1
        x_nonlinear = self.relu1(self.bn1(x))
        x_nonlinear = (
            self.dropout1(x_nonlinear) if
            self.splitted else x_nonlinear
        )
        y = self.conv1(x_nonlinear)
        # conv2
        y = self.relu2(self.bn2(y))
        y = self.dropout2(y) if self.splitted else y
        y = self.conv2(y)
        # conv2 + residual
        return y.add_(self.conv_transform(x) if self.need_transform else x)

    def reg_losses(self):
        weights_and_split_indicators = filter(partial(operator.is_not, None), [
            (self.w1, self.p(), self.r()),
            (self.w2, self.r(), self.q()),
            (self.w3, self.p(), self.q()) if self.need_transform else None
        ])

        return [
            splits.reg_loss(w, p, q, cuda=self.is_cuda) for w, p, q in
            weights_and_split_indicators if (p is not None and q is not None)
        ]


class ResidualBlockSplitBottleneck(ResidualBlock):
    """ResidualBlock with split bottleneck"""
    def __init__(self, in_channels, out_channels, stride,
                 split_size=None, split_qa=None, dropout_prob=.5):
        super().__init__(
            in_channels, out_channels, stride,
            dropout_prob=dropout_prob
        )

        # split indicators
        if split_size:
            self.pa = split_qa
            self.ra = splits.alpha(split_size, out_channels)
            self.qa = split_qa
        else:
            self.pa = None
            self.ra = None
            self.qa = None


class ResidualBlockGroup(WeightRegularized):
    def __init__(self, block_number, in_channels, out_channels, stride,
                 split_size=None, split_qa_last=None, dropout_prob=.5, ):
        super().__init__()

        # Residual block group's hyperparameters.
        self.block_number = block_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.split_size = split_size
        self.split_qa_last = split_qa_last
        self.splitted = split_size is not None

        # Define residual blocks in a reversed order. This is to define
        # feature groups in hierarchical manner - from subgroups to
        # supergroups.
        residual_blocks = []
        for i in reversed(range(self.block_number)):
            is_first = (i == 0)
            is_last = (i == self.block_number - 1)
            is_split_bottleneck = (i % 2 == 1)

            if self.splitted:
                qa = self.split_qa_last if is_last else residual_blocks[0].pa
            else:
                qa = None

            block_class = (
                ResidualBlockSplitBottleneck if is_split_bottleneck else
                ResidualBlock
            )
            block = block_class(
                self.in_channels if is_first else self.out_channels,
                self.out_channels,
                self.stride if is_first else 1,
                split_size=split_size,
                split_qa=qa,
                dropout_prob=dropout_prob,
            )
            residual_blocks.insert(0, block)
        # Register the residual block modules.
        self.residual_blocks = nn.ModuleList(residual_blocks)

    def forward(self, x):
        return reduce(lambda x, f: f(x), self.residual_blocks, x)

    def reg_losses(self):
        return itertools.chain(*[
            b.reg_losses() for b in self.residual_blocks
        ])


# =======================================
# Wide Residual Net (Split Regularizable)
# =======================================

class WideResNet(WeightRegularized):
    def __init__(self, label, input_size, input_channels, classes,
                 total_block_number, widen_factor=1, dropout_prob=.5,
                 baseline_strides=None,
                 baseline_channels=None,
                 split_sizes=None):
        super().__init__()

        # Model name label.
        self.label = label

        # Data specific hyperparameters.
        self.input_size = input_size
        self.input_channels = input_channels
        self.classes = classes

        # Model hyperparameters.
        self.total_block_number = total_block_number
        self.widen_factor = widen_factor
        self.dropout_prob = dropout_prob
        self.split_sizes = split_sizes or [2, 2, 2]
        self.baseline_strides = baseline_strides or [1, 1, 2, 2]
        self.baseline_channels = baseline_channels or [16, 16, 32, 64]
        self.widened_channels = [
            w*widen_factor if i != 0 else w for i, w in
            enumerate(self.baseline_channels)
        ]
        self.group_number = len(self.widened_channels) - 1

        # Validate hyperparameters.
        if self.split_sizes is not None:
            assert len(self.split_sizes) <= len(self.baseline_channels)
            assert len(self.split_sizes) <= len(self.baseline_strides)
        assert len(self.baseline_channels) == len(self.baseline_strides)
        assert (
            self.total_block_number % (2*self.group_number) == 0 and
            self.total_block_number // (2*self.group_number) >= 1
        ), 'Total number of residual blocks should be multiples of 2 x N.'

        # Residual block group configurations.
        split_sizes_stack = copy.deepcopy(self.split_sizes)
        blocks_per_group = self.total_block_number // self.group_number
        zipped_channels_and_strides = list(zip(
            self.widened_channels[:-1],
            self.widened_channels[1:],
            self.baseline_strides[1:]
        ))

        # 4. Affine layer.
        self.fc = RegularizedLinear(
            self.widened_channels[self.group_number], self.classes,
            split_size=split_sizes_stack.pop(), dropout_prob=dropout_prob
        )

        # 3. Batchnorm & nonlinearity & pooling.
        self.bn = nn.BatchNorm2d(self.widened_channels[self.group_number])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(
            self.input_size //
            reduce(operator.mul, self.baseline_strides)
        )

        # 2. Residual block groups.
        residual_block_groups = []
        for k, (i, o, s) in reversed(list(
                enumerate(zipped_channels_and_strides)
        )):
            is_last = (k == len(zipped_channels_and_strides) - 1)
            try:
                # Case of splitting a residual block group.
                split_size = split_sizes_stack.pop()
                split_qa_last = (
                    self.fc.pa if is_last else
                    residual_block_groups[0].residual_blocks[0].pa
                )
            except IndexError:
                # Case of not splitting a residual block group.
                split_size = None
                split_qa_last = None

            # Push the residual block groups from upside down.
            residual_block_groups.insert(0, ResidualBlockGroup(
                blocks_per_group, i, o, s,
                split_size=split_size,
                split_qa_last=split_qa_last,
                dropout_prob=dropout_prob,
            ))
        # Register the residual block group modules.
        self.residual_block_groups = nn.ModuleList(residual_block_groups)

        # 1. Convolution layer.
        self.conv = nn.Conv2d(
            self.input_channels, self.widened_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        return reduce(lambda x, f: f(x), [
            self.conv,
            *self.residual_block_groups,
            self.bn,
            self.relu,
            self.pool,
            (lambda x: x.view(-1, self.widened_channels[-1])),
            self.fc
        ], x)

    def reg_losses(self):
        return itertools.chain(self.fc.reg_losses(), *[
            g.reg_losses() for
            g in self.residual_block_groups if g.splitted
        ])

    def reg_loss(self):
        reg_losses = self.reg_losses()
        overlap_losses, uniform_losses, split_losses = tuple(zip(*reg_losses))
        split_loss_weights = [l.detach() for l in split_losses]
        split_losses_weighted = [l.detach() * l for l in split_losses]
        return (
            sum(overlap_losses) / len(overlap_losses),
            sum(uniform_losses) / len(uniform_losses),
            sum(split_losses_weighted) / sum(split_loss_weights),
        )

    @property
    def name(self):
        # Label for the split group configurations.
        if self.split_sizes:
            split_label = 'split[{}]-'.format('-'.join(
                str(s) for s in self.split_sizes
            ))
        else:
            split_label = ''

        # First block of a residual group contains 3 conv layers and rest
        # blocks of the group contains 2 conv layers.
        depth = self.group_number*3 + (
            self.total_block_number -
            self.group_number
        )*2 + 1

        # Name of the model.
        return (
            'WRN-{depth}-{widen_factor}-{split_label}dropout{dropout_prob}-'
            '{label}-{size}x{size}x{channels}'
        ).format(
            depth=depth,
            widen_factor=self.widen_factor,
            split_label=split_label,
            dropout_prob=self.dropout_prob,
            label=self.label,
            size=self.input_size,
            channels=self.input_channels,
        )
