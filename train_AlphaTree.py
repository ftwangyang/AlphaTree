import os
import yaml
import argparse
import torch
from datetime import datetime
from pathlib import Path

from AlphaTree_model.agent import QRQCMAgent1, IQCMAgent, FQCMAgent
from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.data.expression import StockData
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv


def load_instruments_from_file(instruments_dir: Path, filename: str) -> list[str]:
    file_path = instruments_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Instrument file not found: {file_path}")

    stocks: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            stock_code = parts[0]
            if stock_code.startswith(("SH", "SZ")):
                stocks.append(stock_code)

    return sorted(set(stocks))


def align_stock_data(data_train: StockData, data_valid: StockData, data_test: StockData) -> tuple[StockData, StockData, StockData]:
    n_train = data_train.n_stocks
    n_valid = data_valid.n_stocks
    n_test = data_test.n_stocks

    if n_train == n_valid == n_test:
        return data_train, data_valid, data_test

    min_stocks = min(n_train, n_valid, n_test)

    data_train.data = data_train.data[:, :, :min_stocks]
    data_valid.data = data_valid.data[:, :, :min_stocks]
    data_test.data = data_test.data[:, :, :min_stocks]

    return data_train, data_valid, data_test


def build_log_dir(args, config: dict, time_str: str) -> str:
    name = args.model
    lstm_type = "TreeLSTM" if args.use_tree_lstm else "LSTM"
    dense_type = "FFDense" if args.use_dense_reward else "Sparse"

    base = os.path.join(
        f"data/{args.instruments}_logs",
        f"pool_{args.pool}_AlphaTree_{args.std_lam}_{lstm_type}_{dense_type}",
    )

    if name in ["qrdqn", "iqn"]:
        return os.path.join(
            base,
            f"{name}-seed{args.seed}-{time_str}-N{config['N']}-lr{config['lr']}-"
            f"per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}",
        )

    if name == "fqf":
        return os.path.join(
            base,
            f"{name}-seed{args.seed}-{time_str}-N{config['N']}-lr{config['quantile_lr']}-"
            f"per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}",
        )

    raise ValueError(f"Unknown model: {name}")


def run(args) -> None:
    config_path = os.path.join("config/qcm_config", f"{args.model}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if torch.cuda.is_available() and args.cuda_device is not None:
        torch.cuda.set_device(args.cuda_device)
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    instruments_dir = Path(args.instruments_dir)
    index_name = args.instruments.lower()
    instruments_all = load_instruments_from_file(instruments_dir, f"{index_name}.txt")

    data_train = StockData(
        instrument=instruments_all,
        start_time="2010-01-01",
        end_time="2020-12-31",
        qlib_path=args.qlib_path,
        filter_stocks=False,
        stock_quality_threshold=0.7,
        smart_fill=True,
        verbose=args.verbose,
    )

    data_valid = StockData(
        instrument=instruments_all,
        start_time="2021-01-01",
        end_time="2021-12-31",
        qlib_path=args.qlib_path,
        filter_stocks=False,
        stock_quality_threshold=0.7,
        smart_fill=True,
        verbose=args.verbose,
    )

    data_test = StockData(
        instrument=instruments_all,
        start_time="2022-01-01",
        end_time="2024-12-31",
        qlib_path=args.qlib_path,
        filter_stocks=False,
        stock_quality_threshold=0.7,
        smart_fill=True,
        verbose=args.verbose,
    )

    data_train, data_valid, data_test = align_stock_data(data_train, data_valid, data_test)

    train_pool = AlphaPool(
        capacity=args.pool,
        stock_data=data_train,
        target=target,
        ic_lower_bound=None,
    )

    train_env = AlphaEnv(
        pool=train_pool,
        device=device,
        print_expr=args.print_expr,
        use_dense_reward=args.use_dense_reward,
        subtree_reward=args.subtree_reward,
        complexity_reward=args.complexity_reward,
        syntax_reward=args.syntax_reward,
        diversity_reward=args.diversity_reward,
        dense_reward_scale=args.dense_reward_scale,
    )

    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = build_log_dir(args, config, time_str)

    if args.model == "qrdqn":
        agent = QRQCMAgent1(
            env=train_env,
            data_valid=data_valid,
            data_test=data_test,
            target=target,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=torch.cuda.is_available(),
            use_tree_lstm=args.use_tree_lstm,
            **config,
        )
    elif args.model == "iqn":
        agent = IQCMAgent(
            env=train_env,
            data_valid=data_valid,
            data_test=data_test,
            target=target,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=torch.cuda.is_available(),
            use_tree_lstm=args.use_tree_lstm,
            **config,
        )
    elif args.model == "fqf":
        agent = FQCMAgent(
            env=train_env,
            data_valid=data_valid,
            data_test=data_test,
            target=target,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=torch.cuda.is_available(),
            use_tree_lstm=args.use_tree_lstm,
            **config,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    agent.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaTree model")

    parser.add_argument("--model", type=str, default="qrdqn", choices=["qrdqn", "iqn", "fqf"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--std-lam", type=float, default=1.0)
    parser.add_argument("--instruments", type=str, default="csi300", choices=["csi300", "csi500", "csi800", "csi1000"])

    parser.add_argument("--use-tree-lstm", action="store_true", default=True)
    parser.add_argument("--no-tree-lstm", action="store_false", dest="use_tree_lstm")

    parser.add_argument("--use-dense-reward", action="store_true", default=False)
    parser.add_argument("--no-dense-reward", action="store_false", dest="use_dense_reward")
    parser.add_argument("--subtree-reward", type=float, default=0.00)
    parser.add_argument("--complexity-reward", type=float, default=0.000)
    parser.add_argument("--syntax-reward", type=float, default=0.000)
    parser.add_argument("--diversity-reward", type=float, default=0.000)
    parser.add_argument("--dense-reward-scale", type=float, default=1.0)

    # Paths (edit these defaults to your machine, or pass via CLI)
    parser.add_argument(
        "--qlib-path",
        type=str,
        default="/path/to/qlib_data/cn_data",
        help="Qlib data root directory",
    )
    parser.add_argument(
        "--instruments-dir",
        type=str,
        default="/path/to/qlib_data/cn_data/instruments",
        help="Directory containing instrument files like csi300.txt",
    )

    # Runtime toggles
    parser.add_argument("--cuda-device", type=int, default=1, help="CUDA device index (ignored if CUDA unavailable)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output in data loader")
    parser.add_argument("--print-expr", action="store_true", default=False, help="Enable expression printing in env")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
