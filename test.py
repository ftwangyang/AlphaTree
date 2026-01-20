import torch
import os
from gan.utils import load_pickle
from alphagen_generic.features import *
from alphagen.data.expression import *
from typing import Tuple, List, Union
import json
import argparse
from alphagen.data.expression import StockData
from alphagen.data.expression import Feature, FeatureType, Ref
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, \
    batch_max_drawdown
from gan.utils.builder import exprs2tensor


def load_instruments_from_file(filename, instruments_dir):
    """
    Load stock list from instruments file.

    Args:
        filename: File name (e.g., 'csi300.txt' or 'csi500.txt')
        instruments_dir: Directory containing instruments files

    Returns:
        list: List of stock codes
    """
    file_path = Path(instruments_dir) / filename

    print(f"Loading instruments from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stocks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts:
                stock_code = parts[0]
                if stock_code.startswith('SH') or stock_code.startswith('SZ'):
                    stocks.append(stock_code)

    stocks = list(set(stocks))
    stocks.sort()

    print(f"Loaded {len(stocks)} stocks")
    print(f"Sample: {stocks[:10]}")

    return stocks


def remove_linearly_dependent_rows(x, y, to_pred, tol=1e-10):

    if x.shape[0] <= x.shape[1]:
        return x, y, to_pred, list(range(x.shape[0]))

    sample_ratio = x.shape[0] / x.shape[1]

    if sample_ratio < 5:
        return x, y, to_pred, list(range(x.shape[0]))

    try:
        U, S, Vh = torch.linalg.svd(x.T, full_matrices=False)

        rank = torch.sum(S > tol * S[0]).item()

        if rank >= min(x.shape[0], x.shape[1]):
            return x, y, to_pred, list(range(x.shape[0]))

        Q, R = torch.linalg.qr(x.T, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol

        if not torch.any(pivot_mask):
            selected_rows = [0]
        else:
            selected_rows = torch.where(pivot_mask)[0].tolist()
            if len(selected_rows) == 0:
                selected_rows = [0]

    except:
        return x, y, to_pred, list(range(x.shape[0]))

    x_filtered = x[selected_rows]
    y_filtered = y[selected_rows] if y is not None else None

    return x_filtered, y_filtered, to_pred, selected_rows


def remove_linearly_dependent_cols(x, to_pred, tol=1e-10):

    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))

    try:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        rank = torch.sum(S > tol * S[0]).item()

        if rank == 0:
            selected_factors = [0]
        else:
            selected_factors = list(range(min(rank, x.shape[1])))

    except:
        Q, R = torch.linalg.qr(x, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol

        if not torch.any(pivot_mask):
            selected_factors = [0]
        else:
            selected_factors = torch.where(pivot_mask)[0].tolist()
            if len(selected_factors) == 0:
                selected_factors = [0]

    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]

    return x_filtered, to_pred_filtered, selected_factors


def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    """Load alpha pool from various JSON formats."""
    if 'exprs' in raw and 'weights' in raw:
        exprs_raw = raw['exprs']
        weights = raw['weights']
        exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
        return exprs, weights

    elif 'cache' in raw:
        cache = raw['cache']
        exprs = []
        weights = []

        print(f"Loading from cache format, total {len(cache)} factors")

        for expr_str, score in cache.items():
            if score == -1.0 or score == 0.0:
                continue

            try:
                expr = eval(expr_str.replace('open', 'open_').replace('$', ''))
                exprs.append(expr)
                weights.append(float(score))
            except Exception as e:
                print(f"Skipping invalid expression: {expr_str[:60]}... - {e}")
                continue

        print(f"Successfully loaded {len(exprs)} valid factors (filtered {len(cache) - len(exprs)} invalid)")

        if len(exprs) == 0:
            raise ValueError("No valid factors found!")

        return exprs, weights

    else:
        raise ValueError(f"Unsupported JSON format. Expected 'exprs'+'weights' or 'cache', got: {list(raw.keys())}")


def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    """Load alpha pool from file."""
    print(f"\n{'=' * 70}")
    print(f"Loading factor file: {path}")
    print(f"{'=' * 70}")

    if path.endswith('.json'):
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
            return load_alpha_pool(raw)

    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        print(f"CSV loaded successfully: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")

        exprs_raw = df['exprs'].tolist()
        exprs = []
        valid_indices = []

        for i, expr_raw in enumerate(exprs_raw):
            if "Ensemble" in str(expr_raw):
                continue
            try:
                expr = eval(str(expr_raw).replace('open', 'open_').replace('$', ''))
                exprs.append(expr)
                valid_indices.append(i)
            except Exception as e:
                print(f"Skipping invalid expression [row {i}]: {str(e)[:60]}")

        print(f"Successfully parsed {len(exprs)}/{len(exprs_raw)} expressions")

        weights = None
        weight_columns = ['scores', 'score', 'weights', 'weight', 'ic', 'ric']

        for col in weight_columns:
            if col in df.columns:
                all_weights = df[col].tolist()
                weights = [all_weights[i] for i in valid_indices]
                weights = [float(w) if (w is not None and pd.notna(w)) else 0.0 for w in weights]
                print(f"Loaded {len(weights)} weights from column '{col}'")
                print(f"Weight range: [{min(weights):.6f}, {max(weights):.6f}]")
                break

        if weights is None:
            weights = [1.0 / len(exprs)] * len(exprs)
            print(f"No weight column found, using equal weights")

        print(f"{'=' * 70}\n")
        return exprs, weights

    else:
        raise ValueError(f"Unsupported file type: {path}")


def chunk_batch_spearmanr(x, y, chunk_size=100):
    """Calculate Spearman correlation in chunks to save memory."""
    n_days = len(x)
    spearmanr_list = []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i + chunk_size], y[i:i + chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list


def get_tensor_metrics(x, y, risk_free_rate=0.0, label_days=20):
    """Calculate performance metrics for predictions."""
    if x.dim() > 2:
        x = x.squeeze(-1)
    if y.dim() > 2:
        y = y.squeeze(-1)

    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)
    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0.)
    ric_s = torch.nan_to_num(ric_s, nan=0.)
    ret_s = torch.nan_to_num(ret_s, nan=0.) / label_days

    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item() if ic_s.std().item() > 1e-6 else 1.0
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item() if ric_s.std().item() > 1e-6 else 1.0
    ret_s_mean = ret_s.mean().item()
    ret_s_std = ret_s.std().item() if ret_s.std().item() > 1e-6 else 1.0

    ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item()
    ret_mdd = batch_max_drawdown(ret_s).item()

    result = dict(
        ic=ic_s_mean,
        ic_std=ic_s_std,
        icir=ic_s_mean / ic_s_std,
        ric=ric_s_mean,
        ric_std=ric_s_std,
        ricir=ric_s_mean / ric_s_std,
        ret=ret_s_mean * len(ret_s) / 3,
        ret_std=ret_s_std,
        retir=ret_s_mean / ret_s_std,
        ret_sharpe=ret_sharpe,
        ret_mdd=ret_mdd,
    )
    return result, ret_s


def run(args):
    """
    Main function to run adaptive factor combination and evaluation.
    """
    window = args.window
    if isinstance(window, str):
        assert window == 'inf'
        window = float('inf')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    print(f"\n{'=' * 70}")
    print(f"Loading {args.instruments.upper()} stock list")
    print(f"{'=' * 70}")

    index_name = args.instruments.lower()
    instruments_dir = Path(args.qlib_path) / "instruments"
    instruments = load_instruments_from_file(f'{index_name}.txt', instruments_dir)

    print(f"Using {len(instruments)} stocks for testing")
    print(f"{'=' * 70}\n")

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -args.label_days) / close - 1

    train_end_time = f'{args.train_end_year}-12-31'
    valid_start_time = f'{args.train_end_year + 1}-01-01'
    valid_end_time = f'{args.train_end_year + 1}-12-31'
    test_start_time = f'{args.train_end_year + 2}-01-01'
    test_end_time = f'{args.train_end_year + 3}-12-31'

    data_all = StockData(instrument=instruments,
                         start_time='2010-01-01',
                         end_time=test_end_time,
                         qlib_path=args.qlib_path)
    data_valid = StockData(instrument=instruments,
                           start_time=valid_start_time,
                           end_time=valid_end_time,
                           qlib_path=args.qlib_path)
    data_test = StockData(instrument=instruments,
                          start_time=test_start_time,
                          end_time=test_end_time,
                          qlib_path=args.qlib_path)

    print(f"Loading expressions from {args.expressions_file}...")
    expressions, weights = load_alpha_pool_by_path(args.expressions_file)
    print(f"Loaded {len(expressions)} expressions.")

    if args.use_weights:
        fct_tensor = exprs2tensor(expressions, data_test, normalize=True)
        weights = torch.tensor(weights).cuda()
        fct_tensor = fct_tensor @ weights
        tgt_tensor = exprs2tensor([target], data_test, normalize=False)
        test_results, ret_s = get_tensor_metrics(fct_tensor.cuda(), tgt_tensor.cuda(), label_days=args.label_days)
        ret_s = ret_s.cpu().numpy()
        save_path = os.path.join(os.path.dirname(args.expressions_file), 'ret_s.npy')
        np.save(save_path, ret_s)

        results_df = pd.DataFrame([test_results], index=['Test'])
        print("\n--- Final Performance Metrics ---")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.round(4))

        print("\n--- Parseable Format ---")
        print(
            f"{'Dataset':<12} {'IC':>8} {'IC_STD':>8} {'ICIR':>8} {'RIC':>8} {'RIC_STD':>8} {'RICIR':>8} {'RET':>8} {'RET_STD':>8} {'RETIR':>8} {'RET_SR':>8} {'RET_MDD':>8}")
        for index, row in results_df.iterrows():
            print(
                f"{index:<12} {row['ic']:>8.4f} {row['ic_std']:>8.4f} {row['icir']:>8.4f} {row['ric']:>8.4f} {row['ric_std']:>8.4f} {row['ricir']:>8.4f} {row['ret']:>8.4f} {row['ret_std']:>8.4f} {row['retir']:>8.4f} {row['ret_sharpe']:>8.4f} {row['ret_mdd']:>8.4f}")

        print("=" * 50)

    else:
        fct_tensor = exprs2tensor(expressions, data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)

        ic_list, ric_list = [], []
        print("Pre-calculating daily metrics for each factor...")
        for i in tqdm(range(fct_tensor.shape[-1])):
            factor_slice = fct_tensor[..., i]
            target_slice = tgt_tensor[..., 0]
            ic_s = batch_pearsonr(factor_slice, target_slice)
            ric_s = chunk_batch_spearmanr(factor_slice, target_slice, chunk_size=args.chunk_size)
            ic_list.append(torch.nan_to_num(ic_s, nan=0.))
            ric_list.append(torch.nan_to_num(ric_s, nan=0.))

        ic_s = torch.stack(ic_list, dim=-1)
        ric_s = torch.stack(ric_list, dim=-1)
        torch.cuda.empty_cache()

        pred_list = []
        shift = args.label_days + 1

        valid_test_days = data_valid.n_days + data_test.n_days
        start_day = len(fct_tensor) - valid_test_days

        print("Starting adaptive combination process...")
        pbar = tqdm(range(start_day, len(fct_tensor)))
        for cur in pbar:
            begin = 0 if not np.isfinite(window) else max(0, cur - window - shift)

            cur_ic = ic_s[begin:cur - shift]
            cur_ric = ric_s[begin:cur - shift]

            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0)
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0)

            icir = ic_mean / ic_std
            ricir = ric_mean / ric_std

            metrics_df = pd.DataFrame({
                'ric': ric_mean.cpu().numpy(),
                'ricir': ricir.cpu().numpy()
            })
            good_factors = metrics_df[
                (metrics_df['ric'].abs() > args.threshold_ric) & (metrics_df['ricir'].abs() > args.threshold_ricir)]
            if len(good_factors) < 1:
                good_factors = metrics_df.reindex(metrics_df.ricir.abs().sort_values(ascending=False).index).iloc[:1]

            good_idx = good_factors.iloc[:args.n_factors].index.to_list()

            x = fct_tensor[begin:cur - shift, :, good_idx]
            y = tgt_tensor[begin:cur - shift, :, :]
            to_pred = fct_tensor[cur, :, good_idx]
            y = y.reshape(-1, y.shape[-1])
            x = x.reshape(-1, x.shape[-1])

            valid_mask = torch.isfinite(y)[:, 0]
            y = y[valid_mask]
            x = x[valid_mask]

            to_pred = torch.nan_to_num(to_pred, nan=0.)

            x, to_pred, selected_factors = remove_linearly_dependent_cols(x, to_pred, tol=args.linear_dep_tol)
            x, y, to_pred, selected_rows = remove_linearly_dependent_rows(x, y, to_pred, tol=args.linear_dep_tol)

            ones = torch.ones_like(x[..., 0:1])
            x = torch.cat([x, ones], dim=-1)
            ones_pred = torch.ones_like(to_pred[..., 0:1])
            to_pred = torch.cat([to_pred, ones_pred], dim=-1)

            try:
                coef = torch.linalg.lstsq(x, y).solution
                pred = to_pred @ coef
            except Exception as e:
                print(f"Warning: Regression failed with error {e}, using zero prediction")
                pred = torch.zeros_like(to_pred[:, 0:1])

            pred_list.append(pred[:, 0])

            if len(pred_list) > 1:
                running_preds = torch.stack(pred_list, dim=0)
                running_targets = tgt_tensor[start_day:cur + 1, :, 0]
                running_ic = batch_pearsonr(running_preds, running_targets).mean().item()
                pbar.set_description(f"Running IC: {running_ic:.4f}, Factors selected: {len(good_idx)}")

        print("\n" + "=" * 50)
        print("Adaptive combination finished. Calculating final metrics...")

        all_pred = torch.stack(pred_list, dim=0)

        pred_valid = all_pred[:data_valid.n_days]
        pred_test = all_pred[data_valid.n_days:]

        tgt_valid = tgt_tensor[start_day: start_day + data_valid.n_days, :, 0]
        tgt_test = tgt_tensor[start_day + data_valid.n_days:, :, 0]

        valid_results, _ = get_tensor_metrics(pred_valid.cuda(), tgt_valid.cuda(), label_days=args.label_days)
        test_results, ret_s = get_tensor_metrics(pred_test.cuda(), tgt_test.cuda(), label_days=args.label_days)
        ret_s = ret_s.cpu().numpy()
        save_path = os.path.join(os.path.dirname(args.expressions_file), 'ret_s.npy')
        np.save(save_path, ret_s)

        results_df = pd.DataFrame([valid_results, test_results], index=['Validation', 'Test'])
        print("\n--- Final Performance Metrics ---")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.round(4))

        print("\n--- Parseable Format ---")
        print(
            f"{'Dataset':<12} {'IC':>8} {'IC_STD':>8} {'ICIR':>8} {'RIC':>8} {'RIC_STD':>8} {'RICIR':>8} {'RET':>8} {'RET_STD':>8} {'RETIR':>8} {'RET_SR':>8} {'RET_MDD':>8}")
        for index, row in results_df.iterrows():
            print(
                f"{index:<12} {row['ic']:>8.4f} {row['ic_std']:>8.4f} {row['icir']:>8.4f} {row['ric']:>8.4f} {row['ric_std']:>8.4f} {row['ricir']:>8.4f} {row['ret']:>8.4f} {row['ret_std']:>8.4f} {row['retir']:>8.4f} {row['ret_sharpe']:>8.4f} {row['ret_mdd']:>8.4f}")

        print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expressions_file', type=str, required=True,
                        help='Path to a JSON file containing a list of alpha expressions.')
    parser.add_argument('--qlib_path', type=str, required=True,
                        help='Path to qlib data directory')
    parser.add_argument('--instruments', type=str, default='csi300',
                        choices=['csi300', 'csi500', 'csi800', 'csi1000'])
    parser.add_argument('--train_end_year', type=int, default=2020)
    parser.add_argument('--threshold_ric', type=float, default=0.015)
    parser.add_argument('--threshold_ricir', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--n_factors', type=int, default=10,
                        help='Maximum number of factors to select at each step.')
    parser.add_argument('--chunk_size', type=int, default=400,
                        help='Chunk size for calculating Spearman correlation.')
    parser.add_argument('--window', type=str, default='inf',
                        help="Rolling window size for factor evaluation. 'inf' for expanding window.")
    parser.add_argument('--label_days', type=int, default=20,
                        help="Number of days to label the target.")
    parser.add_argument('--use_weights', type=bool, default=False,
                        help="Whether to use weights for the factors.")
    parser.add_argument('--linear_dep_tol', type=float, default=1e-10,
                        help="Tolerance for linear dependence detection.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)

