import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global

class Model(base.Model):
    def __init__(self, bn, nmp, initializer, outputs=10):
        super(Model, self).__init__()

        # TODO

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, outputs),
        )

        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # TODO
    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name.startswith('lenet5_mnist') and len(model_name.split('_')) >= 2

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """The name of a model is mnist_lenet_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'mnist_lenet_300_100'.
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        bn = True if "bn" in model_name else False
        nmp = True if "nmp" in model_name else False
        return Model(bn, nmp, initializer, outputs)


    @property
    def loss_criterion(self):
        return self.criterion


    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='lenet5_mnist',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            lr=0.001,
            training_steps='40ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='bias',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)