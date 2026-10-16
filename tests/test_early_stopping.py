import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_mist.estimators import club, mine
from torch_mist.utils.data.utils import make_dataset
from torch_mist.utils.evaluation import evaluate
from torch_mist.utils.train import train_mi_estimator
from torch_mist.utils.train.utils import RunTerminationManager


def _make_manager(maximize: bool, patience: int = 2) -> RunTerminationManager:
    return RunTerminationManager(
        early_stopping=True,
        tolerance=0.001,
        patience=patience,
        warmup_iterations=0,
        max_iterations=None,
        maximize=maximize,
        minimize=not maximize,
    )


def _set_weight(model: nn.Linear, value: float):
    with torch.no_grad():
        model.weight.fill_(value)


def test_minimize_positive_scores_restores_best():
    # Upper bounds (e.g. CLUB) produce positive values that should be minimized
    manager = _make_manager(maximize=False)
    model = nn.Linear(1, 1, bias=False)

    for iteration, (score, weight) in enumerate(
        [(1.0, 1.0), (0.5, 2.0), (0.8, 3.0), (0.9, 4.0)], start=1
    ):
        _set_weight(model, weight)
        stop = manager.should_stop(iteration=iteration, score=score, model=model)
        assert stop == (iteration == 4)

    assert manager.best_value == 0.5
    manager.load_best_weights(model)
    assert model.weight.item() == 2.0


def test_maximize_negative_scores_saves_weights():
    # Lower bounds can be negative when the true MI is close to zero
    manager = _make_manager(maximize=True)
    model = nn.Linear(1, 1, bias=False)

    _set_weight(model, 1.0)
    assert not manager.should_stop(iteration=1, score=-0.2, model=model)
    _set_weight(model, 2.0)
    assert not manager.should_stop(iteration=2, score=-0.5, model=model)

    assert manager.best_value == -0.2
    assert manager.current_patience == 1
    manager.load_best_weights(model)
    assert model.weight.item() == 1.0


def test_concurrent_managers_do_not_share_checkpoints():
    managers = [_make_manager(maximize=True) for _ in range(2)]
    models = [nn.Linear(1, 1, bias=False) for _ in range(2)]

    # Same iteration for both runs, as in parallel sweeps on a single node
    for i, (manager, model) in enumerate(zip(managers, models)):
        _set_weight(model, float(i))
        manager.should_stop(iteration=1, score=1.0, model=model)

    # Deleting one run's checkpoint must not affect the other
    managers[0].delete_best_weights()
    _set_weight(models[1], 10.0)
    managers[1].load_best_weights(models[1])
    assert models[1].weight.item() == 1.0


def _correlated_data(n_samples: int = 1000, n_dim: int = 2):
    torch.manual_seed(0)
    x = torch.randn(n_samples, n_dim)
    y = x + 0.5 * torch.randn(n_samples, n_dim)
    n_train = int(n_samples * 0.8)
    train = {"x": x[:n_train], "y": y[:n_train]}
    valid = {"x": x[n_train:], "y": y[n_train:]}
    return train, valid


def _valid_losses(log):
    return log[(log["split"] == "valid") & (log["name"] == "loss")]["value"]


def test_club_early_stopping_monitors_valid_loss():
    # CLUB is neither flagged as a lower nor an upper bound, and its MI value grows while q(y|x) is trained,
    # so early stopping must monitor the validation loss instead
    train, valid = _correlated_data()
    estimator = club(x_dim=2, y_dim=2, hidden_dims=[16])
    valid_loader = DataLoader(make_dataset(valid), batch_size=100)

    log = train_mi_estimator(
        estimator=estimator,
        train_data=DataLoader(make_dataset(train), batch_size=100, shuffle=True),
        valid_data=valid_loader,
        max_epochs=30,
        early_stopping=True,
        patience=2,
        verbose=False,
    )

    valid_losses = _valid_losses(log)
    assert len(valid_losses) > 0

    # The weights with the lowest validation loss are restored at the end of training
    final_loss = evaluate(estimator, method="loss", data=valid_loader)
    assert abs(final_loss - valid_losses.min()) < 1e-5


def test_mine_early_stopping_enabled():
    train, valid = _correlated_data()
    estimator = mine(x_dim=2, y_dim=2, hidden_dims=[16])

    log = train_mi_estimator(
        estimator=estimator,
        train_data=DataLoader(make_dataset(train), batch_size=100, shuffle=True),
        valid_data=DataLoader(make_dataset(valid), batch_size=100),
        max_epochs=5,
        early_stopping=True,
        verbose=False,
    )

    assert len(_valid_losses(log)) > 0
