# Building model
from functools import partial
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model.base_model import DragonNetBase
from model.loss_functions import default_loss, tarreg_loss
from model.early_stopper import EarlyStopper


class Dragonnet:
    """
    Main class for the Dragonnet model

    Parameters
    ----------
    input_dim: int
        Input demension for convariates (X - features)
    shared_hidden: int, default=200
        The number of hidden layers in the dragon body
    outcome_hidden: int, default=100
        The number of hidden layers in the dragon accuracy head
    alpha: float, default=1.0
        loss component weighting hyperparameter between 0 and 1
    beta: float, default=1.0
        targeted regularization hyperparameter between 0 and 1
    epochs: int, default=200
        Number training epochs
    batch_size: int, default=64
        Training batch size
    learning_rate: float, default=1e-3
        Learning rate
    data_loader_num_workers: int, default=4
        Number of workers for data loader
    loss_type: str, {'tarreg', 'default'}, default='tarreg'
        Loss function to use
    device=None
        Whether we use the CPU or GPU to train
    """
    def __init__(
            self,
            input_dim, # Input demension for convariates (X - features)
            shared_hidden=200, # The number of hidden layers in the dragon body
            outcome_hidden=100, # The number of hidden layers in the dragon accuracy head
            alpha=1.0, #
            beta=1.0,
            epochs=30,
            batch_size=32,
            learning_rate=0.0005,
            data_loader_num_workers=2,
            loss_type="tarreg",
            device=None,
            seed=42
    ):
        # 1. Thiết lập Device
        if device:
            self.model_device = device
        else:
            self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Khởi tạo Model và đưa lên Device ngay lập tức
        self.model = DragonNetBase(input_dim=input_dim, shared_hidden=shared_hidden, outcome_hidden=outcome_hidden)
        self.model.to(self.model_device) # Move model to GPU
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = data_loader_num_workers
        self.seed = seed

        # Optimizer phải được khởi tạo SAU KHI model đã được move lên GPU
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_dataloader = None
        self.valid_dataloader = None

        if loss_type == "tarreg":
            self.loss_f = partial(tarreg_loss, alpha=alpha, beta=beta)
        elif loss_type == "default":
            self.loss_f = partial(default_loss, alpha=alpha)

    def create_dataloaders(self, X, y, T, valid_perc=None):
        """
        Utility function to create train and validation data loader:

        Parameters
        ----------
        X: np.array
            covariates
        y: np.array
            target variable
        T: np.array
            treatment
        """
        if valid_perc:
            X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
                X, y, T, test_size=valid_perc, random_state=self.seed
            )
            # Không cần .to(device) ở đây để tiết kiệm VRAM, sẽ move theo batch
            X_train = torch.Tensor(X_train)
            X_test = torch.Tensor(X_test)
            y_train = torch.Tensor(y_train).reshape(-1, 1)
            y_test = torch.Tensor(y_test).reshape(-1, 1)
            T_train = torch.Tensor(T_train).reshape(-1, 1)
            T_test = torch.Tensor(T_test).reshape(-1, 1)
            train_dataset = TensorDataset(X_train, T_train, y_train)
            valid_dataset = TensorDataset(X_test, T_test, y_test)
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            X = torch.Tensor(X)
            T = torch.Tensor(T).reshape(-1, 1)
            y = torch.Tensor(y).reshape(-1, 1)
            train_dataset = TensorDataset(X, T, y)
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )

    def fit(self, X, y, T, valid_perc=None):
        """
        Function used to train the dragonnet model

        Parameters
        ----------
        x: np.array
            covariates
        y: np.array
            target variable
        t: np.array
            treatment
        valid_perc: float
            Percentage of data to allocate to validation set
        """
        self.train_losses, self.valid_losses = [], []
        self.create_dataloaders(X, y, T, valid_perc)
        early_stopper = EarlyStopper(patience=10, min_delta=0)
        for epoch in range(self.epochs):
            running_loss_train = 0.0
            for batch, (X, tr, y1) in enumerate(self.train_dataloader):
                # <--- QUAN TRỌNG: Move batch data lên GPU
                X = X.to(self.model_device)
                tr = tr.to(self.model_device)
                y1 = y1.to(self.model_device)

                self.optim.zero_grad()

                y0_pred, y1_pred, t_pred, eps = self.model(X)
                loss = self.loss_f(y1, tr, t_pred, y0_pred, y1_pred, eps)

                loss.backward()
                self.optim.step()
                running_loss_train += loss.item()

            train_loss = running_loss_train/len(self.train_dataloader)
            self.train_losses.append(train_loss)
            if self.valid_dataloader:
                self.model.eval()
                valid_loss = self.validate_step()
                self.valid_losses.append(valid_loss)
                print(
                    f"epoch: {epoch}--------- train_loss: {train_loss:.4f} ----- valid_loss: {valid_loss}"
                )
                self.model.train()
                if early_stopper.early_stop(valid_loss):
                    print("Early stopping activated")
                    break
            else:
                print(f"epoch: {epoch}--------- train_loss: {train_loss:.4f}")

    def validate_step(self):
        """
        Calculates validation loss

        Returns
        -------
        valid_loss: torch.Tensor
            validation loss
        """
        self.model.eval()
        valid_loss = []
        with torch.no_grad():
            for batch, (X, tr, y1) in enumerate(self.valid_dataloader):
                # <--- QUAN TRỌNG: Move batch data lên GPU
                X = X.to(self.model_device)
                tr = tr.to(self.model_device)
                y1 = y1.to(self.model_device)

                y0_pred, y1_pred, t_pred, eps = self.model(X)
                loss = self.loss_f(y1, tr, t_pred, y0_pred, y1_pred, eps)
                valid_loss.append(loss)
        return torch.Tensor(valid_loss).mean()


    def predict(self, X):
        """
        Function used to predict on covariates.

        Parameters
        ----------
        X: torch.Tensor or numpy.array
            covariates

        Returns
        -------
        y0_pred: torch.Tensor
            outcome under control
        y1_pred: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        self.model.eval()
        X = torch.Tensor(X).to(self.model_device) # <--- Move input lên GPU
        with torch.no_grad():
            y0_pred, y1_pred, t_pred, eps = self.model(X)
        return (
            y0_pred.cpu().numpy(),
            y1_pred.cpu().numpy(),
            t_pred.cpu().numpy(),
            eps.cpu().numpy()
        )