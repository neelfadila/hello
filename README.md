[ex2.ipynb](https://github.com/user-attachments/files/24597959/ex2.ipynb)
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "Libraries (don't change)"
      ],
      "metadata": {
        "id": "Q3S5loqBCqgM"
      },
      "id": "Q3S5loqBCqgM"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "5f1ba5ec",
      "metadata": {
        "id": "5f1ba5ec"
      },
      "outputs": [],
      "source": [
        "!pip -q install torchinfo\n",
        "\n",
        "from dataclasses import dataclass\n",
        "from typing import List, Callable, Optional, Tuple\n",
        "\n",
        "import numpy as np\n",
        "import time\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import DataLoader, random_split\n",
        "from torchvision import datasets, transforms\n",
        "from torchinfo import summary\n",
        "\n",
        "from sklearn.metrics import confusion_matrix, classification_report\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Device (don't change)"
      ],
      "metadata": {
        "id": "ykrfYnc6g1Bs"
      },
      "id": "ykrfYnc6g1Bs"
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "use_amp = (DEVICE == \"cuda\")\n",
        "\n",
        "print(f\"Using device: {DEVICE}\")\n",
        "print(f\"Mixed precision (AMP): {use_amp}\")"
      ],
      "metadata": {
        "id": "XJIYaWIggSCp"
      },
      "id": "XJIYaWIggSCp",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "id": "d71ffe61",
      "metadata": {
        "id": "d71ffe61"
      },
      "source": [
        "Data (don't change)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "2cb0bf0c",
      "metadata": {
        "id": "2cb0bf0c"
      },
      "outputs": [],
      "source": [
        "\n",
        "class DataManager:\n",
        "    def __init__(self, dataset_class, root: str = \"./data\", val_fraction: float = 0.1,\n",
        "                 batch_size: int = 32, seed: int = 42):\n",
        "        self.dataset_class = dataset_class\n",
        "        self.root = root\n",
        "        self.val_fraction = val_fraction\n",
        "        self.batch_size = batch_size\n",
        "        self.seed = seed\n",
        "\n",
        "        self.transform = transforms.Compose([\n",
        "            transforms.ToTensor(),\n",
        "            transforms.Normalize((0.1918,), (0.3483,))\n",
        "        ])\n",
        "\n",
        "    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:\n",
        "        full_train = self.dataset_class(root=self.root, train=True,\n",
        "                                        download=True, transform=self.transform)\n",
        "        test_ds = self.dataset_class(root=self.root, train=False,\n",
        "                                     download=True, transform=self.transform)\n",
        "\n",
        "        val_size = int(len(full_train) * self.val_fraction)\n",
        "        train_size = len(full_train) - val_size\n",
        "\n",
        "        generator = torch.Generator().manual_seed(self.seed)\n",
        "        train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)\n",
        "\n",
        "        train_loader = DataLoader(train_ds, batch_size=self.batch_size,\n",
        "                                  shuffle=True, num_workers=2, pin_memory=True)\n",
        "        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,\n",
        "                                  shuffle=False, num_workers=2, pin_memory=True)\n",
        "        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size,\n",
        "                                  shuffle=False, num_workers=2, pin_memory=True)\n",
        "\n",
        "        print(f\"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\")\n",
        "        return train_loader, val_loader, test_loader"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Configurations (don't change)"
      ],
      "metadata": {
        "id": "O0SlvxE36HG0"
      },
      "id": "O0SlvxE36HG0"
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "@dataclass\n",
        "class LayerSpec:\n",
        "    out_dim: int\n",
        "    activation: Callable[[torch.Tensor], torch.Tensor] = F.relu\n",
        "    dropout: float = 0.0\n",
        "    batch_norm: bool = True\n",
        "    weight_decay: float = 0.0\n",
        "\n",
        "@dataclass\n",
        "class ModelConfig:\n",
        "    input_dim: Tuple[int, int, int] = (1, 28, 28)\n",
        "    num_classes: int = 10\n",
        "    layers: List[LayerSpec] = None\n",
        "\n",
        "@dataclass\n",
        "class TrainConfig:\n",
        "    batch_size: int = 64\n",
        "    epochs: int = 100\n",
        "    lr: float = 1e-4\n",
        "    patience: int = 15\n",
        "    min_delta: float = 1e-4\n",
        "    val_fraction: float = 0.1\n",
        "    seed: int = 42\n"
      ],
      "metadata": {
        "id": "FwDGf9J66bXL"
      },
      "id": "FwDGf9J66bXL",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "id": "53221445",
      "metadata": {
        "id": "53221445"
      },
      "source": [
        "Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "b5843387",
      "metadata": {
        "id": "b5843387"
      },
      "outputs": [],
      "source": [
        "\n",
        "class MLPFromConfig(nn.Module):\n",
        "    def __init__(self, config: ModelConfig):\n",
        "        super().__init__()\n",
        "        flat_dim = config.input_dim[0] * config.input_dim[1] * config.input_dim[2]\n",
        "        self.layers_specs = config.layers\n",
        "        layers = []\n",
        "        prev_dim = flat_dim\n",
        "\n",
        "        for i, spec in enumerate(config.layers):\n",
        "            linear = nn.Linear(prev_dim, spec.out_dim)\n",
        "\n",
        "            layers.append(linear)\n",
        "            if spec.batch_norm:\n",
        "                layers.append(nn.BatchNorm1d(spec.out_dim))\n",
        "            if spec.dropout > 0:\n",
        "                layers.append(nn.Dropout(spec.dropout))\n",
        "            layers.append(spec.activation())\n",
        "            prev_dim = spec.out_dim\n",
        "\n",
        "        # Final classifier layer\n",
        "        self.final_linear = nn.Linear(prev_dim, config.num_classes)\n",
        "        layers.append(self.final_linear)\n",
        "\n",
        "        self.net = nn.Sequential(*layers)\n",
        "\n",
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
        "        x = x.view(x.size(0), -1)\n",
        "        return self.net(x)\n",
        "\n",
        "    def get_layer_params(self):\n",
        "        param_groups = []\n",
        "        for i, spec in enumerate(self.layers_specs):\n",
        "            linear_layer = self.net[i * (4 if spec.batch_norm or spec.dropout > 0 else 3)]\n",
        "            pass\n",
        "        return self.layers_specs"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Early Stopping (don't change)"
      ],
      "metadata": {
        "id": "eND74vML5XYh"
      },
      "id": "eND74vML5XYh"
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "class EarlyStopping:\n",
        "    def __init__(self, patience: int = 10, min_delta: float = 1e-4):\n",
        "        self.patience = patience\n",
        "        self.min_delta = min_delta\n",
        "        self.counter = 0\n",
        "        self.best_loss = float('inf')\n",
        "        self.should_stop = False\n",
        "\n",
        "    def __call__(self, val_loss: float) -> bool:\n",
        "        if val_loss < self.best_loss - self.min_delta:\n",
        "            self.best_loss = val_loss\n",
        "            self.counter = 0\n",
        "        else:\n",
        "            self.counter += 1\n",
        "            if self.counter >= self.patience:\n",
        "                self.should_stop = True\n",
        "        return self.should_stop"
      ],
      "metadata": {
        "id": "KgImYRwI5hAr"
      },
      "id": "KgImYRwI5hAr",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "id": "9b11a04f",
      "metadata": {
        "id": "9b11a04f"
      },
      "source": [
        "Trainer (don't change)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "e474713f",
      "metadata": {
        "id": "e474713f"
      },
      "outputs": [],
      "source": [
        "\n",
        "class Trainer:\n",
        "    def __init__(self, model: nn.Module, config: TrainConfig):\n",
        "        self.model = model.to(DEVICE)\n",
        "        self.config = config\n",
        "        self.criterion = nn.CrossEntropyLoss()\n",
        "        self.optimizer = self._build_optimizer()\n",
        "        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)\n",
        "        self.early_stopping = EarlyStopping(patience=config.patience,\n",
        "                                            min_delta=config.min_delta)\n",
        "\n",
        "        self.history = {\"train_loss\": [], \"train_acc\": [],\n",
        "                        \"val_loss\": [], \"val_acc\": []}\n",
        "\n",
        "    def _build_optimizer(self):\n",
        "\n",
        "        # Collect all Linear layers in the order they appear\n",
        "        linear_layers = []\n",
        "        for name, module in self.model.named_modules():\n",
        "            if isinstance(module, nn.Linear):\n",
        "                linear_layers.append((name, module))\n",
        "\n",
        "        param_groups = []\n",
        "\n",
        "        for i, spec in enumerate(self.model.layers_specs):\n",
        "            name, layer = linear_layers[i]\n",
        "            param_groups.append({\n",
        "                'params': layer.parameters(),\n",
        "                'weight_decay': spec.weight_decay\n",
        "            })\n",
        "\n",
        "        final_name, final_layer = linear_layers[-1]\n",
        "        param_groups.append({\n",
        "            'params': final_layer.parameters(),\n",
        "            'weight_decay': 0.0\n",
        "        })\n",
        "\n",
        "        return torch.optim.SGD(param_groups, momentum=0.9, nesterov=True, lr=self.config.lr)\n",
        "\n",
        "    def _train_epoch(self, loader: DataLoader):\n",
        "        self.model.train()\n",
        "        total_loss = 0.0\n",
        "        correct = 0\n",
        "        total = 0\n",
        "\n",
        "        for data, target in loader:\n",
        "            data, target = data.to(DEVICE), target.to(DEVICE)\n",
        "\n",
        "            self.optimizer.zero_grad()\n",
        "            with torch.cuda.amp.autocast(enabled=use_amp):\n",
        "                output = self.model(data)\n",
        "                loss = self.criterion(output, target)\n",
        "\n",
        "            self.scaler.scale(loss).backward()\n",
        "            self.scaler.step(self.optimizer)\n",
        "            self.scaler.update()\n",
        "\n",
        "            total_loss += loss.item() * data.size(0)\n",
        "            correct += (output.argmax(1) == target).sum().item()\n",
        "            total += data.size(0)\n",
        "\n",
        "        return total_loss / total, correct / total\n",
        "\n",
        "    @torch.no_grad()\n",
        "    def _eval_epoch(self, loader: DataLoader):\n",
        "        self.model.eval()\n",
        "        total_loss = 0.0\n",
        "        correct = 0\n",
        "        total = 0\n",
        "\n",
        "        for data, target in loader:\n",
        "            data, target = data.to(DEVICE), target.to(DEVICE)\n",
        "            with torch.cuda.amp.autocast(enabled=use_amp):\n",
        "                output = self.model(data)\n",
        "                loss = self.criterion(output, target)\n",
        "\n",
        "            total_loss += loss.item() * data.size(0)\n",
        "            correct += (output.argmax(1) == target).sum().item()\n",
        "            total += data.size(0)\n",
        "\n",
        "        return total_loss / total, correct / total\n",
        "\n",
        "    def fit(self, train_loader: DataLoader, val_loader: DataLoader):\n",
        "        print(\"🚀 Starting training...\\n\")\n",
        "        for epoch in range(1, self.config.epochs + 1):\n",
        "            train_loss, train_acc = self._train_epoch(train_loader)\n",
        "            val_loss, val_acc     = self._eval_epoch(val_loader)\n",
        "\n",
        "            self.history[\"train_loss\"].append(train_loss)\n",
        "            self.history[\"train_acc\"].append(train_acc)\n",
        "            self.history[\"val_loss\"].append(val_loss)\n",
        "            self.history[\"val_acc\"].append(val_acc)\n",
        "\n",
        "            print(f\"Epoch {epoch:3d} | \"\n",
        "                  f\"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | \"\n",
        "                  f\"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\")\n",
        "\n",
        "            if self.early_stopping(val_loss):\n",
        "                print(f\"\\n🛑 Early stopping triggered at epoch {epoch}\")\n",
        "                break\n",
        "\n",
        "        print(\"\\n✅ Training complete!\")\n",
        "\n",
        "    @torch.no_grad()\n",
        "    def evaluate(self, loader: DataLoader):\n",
        "        return self._eval_epoch(loader)\n",
        "\n",
        "    @torch.no_grad()\n",
        "    def predict_all(self, loader: DataLoader):\n",
        "        self.model.eval()\n",
        "        all_preds, all_targets = [], []\n",
        "        for x, y in loader:\n",
        "            x = x.to(DEVICE, non_blocking=True)\n",
        "            logits = self.model(x)\n",
        "            preds = logits.argmax(dim=1).cpu().numpy()\n",
        "            all_preds.append(preds)\n",
        "            all_targets.append(y.numpy())\n",
        "        return np.concatenate(all_preds), np.concatenate(all_targets)\n",
        "\n",
        "\n",
        "    def save(self, path: str = \"mlp_best.pt\"):\n",
        "        torch.save(self.model.state_dict(), path)\n",
        "        print(f\"💾 Model saved to {path}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "55c7d02d",
      "metadata": {
        "id": "55c7d02d"
      },
      "source": [
        "\n",
        "Run (do change)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "d133bb1a",
      "metadata": {
        "id": "d133bb1a"
      },
      "outputs": [],
      "source": [
        "train_cfg = TrainConfig(\n",
        "    batch_size=128,\n",
        "    epochs=100,\n",
        "    lr=1e-4,\n",
        "    patience=5,\n",
        "    val_fraction=0.1\n",
        ")\n",
        "\n",
        "data_mgr = DataManager(\n",
        "    dataset_class=datasets.KMNIST,\n",
        "    val_fraction=train_cfg.val_fraction,\n",
        "    batch_size=train_cfg.batch_size,\n",
        "    seed=train_cfg.seed\n",
        ")\n",
        "train_loader, val_loader, test_loader = data_mgr.get_loaders()\n",
        "\n",
        "model_cfg = ModelConfig(\n",
        "    layers=[\n",
        "        # Add layers here in the format of LayerSpec. For example\n",
        "        LayerSpec(out_dim=10,  dropout=0.1, batch_norm=False, activation = nn.ReLU, weight_decay=5e-1),\n",
        "    ]\n",
        ")\n",
        "\n",
        "\n",
        "# Build model\n",
        "model = MLPFromConfig(model_cfg)\n",
        "print(summary(model, input_size=(1, 28, 28)))\n",
        "\n",
        "trainer = Trainer(model, train_cfg)\n",
        "trainer.fit(train_loader, val_loader)\n",
        "trainer.save(\"mlp_colab_best.pt\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "3e0c6f58",
      "metadata": {
        "id": "3e0c6f58"
      },
      "source": [
        "Visuazize the train\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "740b2e48",
      "metadata": {
        "id": "740b2e48"
      },
      "outputs": [],
      "source": [
        "\n",
        "history = trainer.history\n",
        "\n",
        "plt.figure(figsize=(12, 4))\n",
        "\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(history[\"train_loss\"], label=\"Train Loss\")\n",
        "plt.plot(history[\"val_loss\"],   label=\"Val Loss\")\n",
        "plt.title(\"Loss\")\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.plot(history[\"train_acc\"], label=\"Train Acc\")\n",
        "plt.plot(history[\"val_acc\"],   label=\"Val Acc\")\n",
        "plt.title(\"Accuracy\")\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Visualized the test (do change)"
      ],
      "metadata": {
        "id": "5kjoDILMt2Zw"
      },
      "id": "5kjoDILMt2Zw"
    },
    {
      "cell_type": "code",
      "source": [
        "# entire test result\n",
        "test_loss, test_acc = trainer.evaluate(test_loader)\n",
        "print(f\"🏆 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}\")\n",
        "\n",
        "# predictions\n",
        "preds, targets = trainer.predict_all(test_loader)\n",
        "\n",
        "# confusion matrix\n",
        "cm = confusion_matrix(targets, preds)\n",
        "plt.figure()\n",
        "plt.imshow(cm)\n",
        "plt.title(\"Confusion Matrix (Test)\")\n",
        "plt.xlabel(\"Predicted\")\n",
        "plt.ylabel(\"True\")\n",
        "plt.colorbar()\n",
        "plt.show()\n",
        "\n",
        "\n",
        "# per-class report\n",
        "print(\"Classification report (Test):\")\n",
        "print(classification_report(targets, preds, digits=4))\n",
        "\n"
      ],
      "metadata": {
        "id": "A-FR-NYgt6Hv"
      },
      "id": "A-FR-NYgt6Hv",
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "language_info": {
      "name": "python"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
