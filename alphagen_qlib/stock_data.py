from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch
import os


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


def change_to_raw_min(features):
    result = []
    for feature in features:
        if feature in ['$vwap']:
            result.append("$money/$volume")
        elif feature in ['$volume']:
            result.append(f"{feature}/100000")
        else:
            result.append(feature)
    return result


def change_to_raw(features):
    result = []
    for feature in features:
        if feature in ['$open', '$close', '$high', '$low', '$vwap']:
            result.append(f"{feature}*$factor")
        elif feature in ['$volume']:
            result.append(f"{feature}/$factor/1000000")
        else:
            raise ValueError(f"feature {feature} not supported")
    return result


class StockData:

    _qlib_initialized: bool = False
    _qlib_current_path: str = ""

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0'),
                 raw: bool = False,
                 qlib_path: Union[str, Dict] = "",
                 freq: str = 'day',
                 filter_stocks: bool = True,
                 stock_quality_threshold: float = 0.5,
                 smart_fill: bool = True,
                 verbose: bool = True,
                 ) -> None:

        self.verbose = verbose
        self.filter_stocks = filter_stocks
        self.stock_quality_threshold = stock_quality_threshold
        self.smart_fill = smart_fill

        FORCE_CORRECT_PATH = "/root/wy/AlphaSAGE-main/src/qlib_data/cn_data"

        if isinstance(qlib_path, str):
            if 'your_path' in qlib_path or not qlib_path or qlib_path == "":
                if self.verbose:
                    print(f"[Fix] Invalid path detected, using default path: {FORCE_CORRECT_PATH}")
                qlib_path = FORCE_CORRECT_PATH

        if not os.path.exists(qlib_path):
            raise ValueError(
                f"Qlib data path does not exist: {qlib_path}\n"
                f"Please ensure Qlib data has been downloaded to this path"
            )

        self._ensure_metadata(qlib_path)

        self._init_qlib(qlib_path, force_reinit=True)

        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Loading dataset: {start_time} to {end_time}")
            print(f"Stock pool: {instrument}")
            print(f"Backtrack days: {max_backtrack_days}, Future days: {max_future_days}")
            print(f"Filter stocks: {'Yes' if filter_stocks else 'No'}")
            if filter_stocks:
                print(f"Quality threshold: {stock_quality_threshold * 100:.0f}%")
            print(f"Smart fill: {'Yes' if smart_fill else 'No'}")
            print(f"{'=' * 60}")

        self.data, self._dates, self._stock_ids = self._get_data()

        if self.verbose:
            print(f"Data loading complete")
            print(f"  - Data shape: {self.data.shape}")
            print(f"  - Trading days: {len(self._dates)}")
            print(f"  - Valid trading days: {self.n_days}")
            print(f"  - Number of stocks: {self.n_stocks}")
            print(f"{'=' * 60}\n")

    @staticmethod
    def _ensure_metadata(qlib_path: str) -> None:
        meta_dir = os.path.join(qlib_path, 'features', '.meta')
        os.makedirs(meta_dir, exist_ok=True)

        meta_file = os.path.join(meta_dir, 'dir.txt')
        if not os.path.exists(meta_file):
            with open(meta_file, 'w') as f:
                f.write('day\n')

    @classmethod
    def _init_qlib(cls, qlib_path: str, force_reinit: bool = False) -> None:
        path_changed = (qlib_path != cls._qlib_current_path)

        if cls._qlib_initialized and not force_reinit and not path_changed:
            return

        import qlib
        from qlib.config import REG_CN, REG_US

        if 'us' in qlib_path.lower():
            region = REG_US
        else:
            region = REG_CN

        if cls._qlib_current_path:
            qlib.init(provider_uri=qlib_path, region=region, override=True)
        else:
            qlib.init(provider_uri=qlib_path, region=region)

        cls._qlib_initialized = True
        cls._qlib_current_path = qlib_path

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D

        if not isinstance(exprs, list):
            exprs = [exprs]

        try:
            cal: np.ndarray = D.calendar(freq=self.freq)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load calendar file\n"
                f"Qlib path: {self._qlib_current_path}\n"
                f"Error: {e}"
            )

        start_index = cal.searchsorted(pd.Timestamp(self._start_time))
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))

        if start_index >= len(cal):
            raise ValueError(
                f"Start time {self._start_time} exceeds calendar range\n"
                f"Calendar range: {cal[0]} to {cal[-1]}"
            )

        if end_index >= len(cal):
            if self.verbose:
                print(f"[Warning] End time {self._end_time} exceeds calendar range, adjusting to last trading day")
            end_index = len(cal) - 1

        if cal[start_index] != pd.Timestamp(self._start_time):
            if self.verbose:
                print(f"[Info] Start time {self._start_time} is not a trading day")
                print(f"       Using next trading day: {cal[start_index]}")

        if cal[end_index] != pd.Timestamp(self._end_time):
            if self.verbose:
                print(f"[Info] End time {self._end_time} is not a trading day")
                print(f"       Using previous trading day: {cal[end_index]}")

        real_start_index = max(0, start_index - self.max_backtrack_days)
        real_end_index = min(len(cal) - 1, end_index + self.max_future_days)

        real_start_time = cal[real_start_index]
        real_end_time = cal[real_end_index]

        if self.verbose:
            print(f"  Actual loading range: {real_start_time} to {real_end_time}")
            print(f"  (Including {self.max_backtrack_days} backtrack days, {self.max_future_days} future days)")

        try:
            result = (QlibDataLoader(config=exprs, freq=self.freq)
                      .load(self._instrument, real_start_time, real_end_time))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load data\n"
                f"Stock pool: {self._instrument}\n"
                f"Time range: {real_start_time} to {real_end_time}\n"
                f"Error: {e}"
            )

        return result

    def _smart_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smart fill missing values (fixed deprecation warning)

        Strategy:
        1. Forward fill (max 5 days) - for suspension
        2. Backward fill (max 2 days) - for new listings
        3. Linear interpolation (max 3 days) - for data gaps
        4. Fill remaining with 0
        """
        if self.verbose:
            original_missing = df.isnull().sum().sum()
            print(f"  [Smart Fill] Original missing values: {original_missing:,}")

        df_filled = df.ffill(limit=5)

        if self.verbose:
            after_ffill = df_filled.isnull().sum().sum()
            filled_by_ffill = original_missing - after_ffill
            print(f"  [Smart Fill] Forward fill: {filled_by_ffill:,}")

        df_filled = df_filled.bfill(limit=2)

        if self.verbose:
            after_bfill = df_filled.isnull().sum().sum()
            filled_by_bfill = after_ffill - after_bfill
            print(f"  [Smart Fill] Backward fill: {filled_by_bfill:,}")

        df_filled = df_filled.interpolate(method='linear', limit=3, limit_area='inside')

        if self.verbose:
            after_interp = df_filled.isnull().sum().sum()
            filled_by_interp = after_bfill - after_interp
            print(f"  [Smart Fill] Linear interpolation: {filled_by_interp:,}")

        df_filled = df_filled.fillna(0)

        if self.verbose:
            filled_by_zero = after_interp
            print(f"  [Smart Fill] Fill with 0: {filled_by_zero:,}")
            print(f"  [Smart Fill] Complete")

        return df_filled

    def _filter_low_quality_stocks_before_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Key fix: Filter stocks based on original NaN rate before filling

        Args:
            df: Original DataFrame (MultiIndex: datetime x instrument)

        Returns:
            Filtered DataFrame
        """
        if not self.filter_stocks:
            return df

        if self.verbose:
            print(f"  [First Filter] Filtering based on original missing rate...")

        if isinstance(df.index, pd.MultiIndex):
            stocks = df.index.get_level_values(1).unique()
        else:
            stocks = df.columns if hasattr(df, 'columns') else []

        original_stock_count = len(stocks)

        if self.verbose:
            print(f"  [First Filter] Original stock count: {original_stock_count}")

        stock_quality = {}

        for stock in stocks:
            try:
                if isinstance(df.index, pd.MultiIndex):
                    stock_data = df.xs(stock, level=1)
                else:
                    stock_data = df[stock] if stock in df.columns else df

                total_values = stock_data.size
                valid_values = stock_data.notna().sum().sum() if hasattr(stock_data,
                                                                         'sum') else stock_data.notna().sum()

                quality = valid_values / total_values if total_values > 0 else 0
                stock_quality[stock] = quality
            except Exception as e:
                if self.verbose:
                    print(f"  [First Filter] Warning: Cannot evaluate stock {stock}: {e}")
                stock_quality[stock] = 0

        valid_stocks = [stock for stock, quality in stock_quality.items()
                        if quality >= self.stock_quality_threshold]

        filtered_count = len(valid_stocks)
        removed_count = original_stock_count - filtered_count

        if self.verbose:
            print(f"  [First Filter] Retained stocks: {filtered_count}")
            print(f"  [First Filter] Removed stocks: {removed_count}")

            if len(stock_quality) > 0:
                avg_quality_before = np.mean(list(stock_quality.values()))
                if len(valid_stocks) > 0:
                    avg_quality_after = np.mean([stock_quality[s] for s in valid_stocks])
                    print(
                        f"  [First Filter] Average data quality: {avg_quality_before * 100:.2f}% -> {avg_quality_after * 100:.2f}%")

        if len(valid_stocks) == 0:
            print(f"  [Warning] All stocks filtered! Lower threshold or check data")
            return df

        if isinstance(df.index, pd.MultiIndex):
            df_filtered = df.loc[(slice(None), valid_stocks), :]
        else:
            df_filtered = df[valid_stocks] if hasattr(df, '__getitem__') else df

        return df_filtered

    def _filter_low_quality_stocks_after_tensor(self,
                                                tensor: torch.Tensor,
                                                stock_ids: pd.Index) -> Tuple[torch.Tensor, pd.Index]:
        """
        Tensor-level secondary filtering (based on final zero rate)

        Args:
            tensor: Data tensor [n_days, n_features, n_stocks]
            stock_ids: Stock ID list

        Returns:
            (filtered tensor, filtered stock_ids)
        """
        if not self.filter_stocks:
            return tensor, stock_ids

        if self.verbose:
            print(f"  [Second Filter] Filtering based on tensor zero rate...")

        n_stocks = tensor.shape[2]
        stock_quality = []

        for i in range(n_stocks):
            stock_data = tensor[:, :, i]
            valid_ratio = ((stock_data != 0) & ~torch.isnan(stock_data)).float().mean().item()
            stock_quality.append(valid_ratio)

        stock_quality_tensor = torch.tensor(stock_quality)
        valid_mask = stock_quality_tensor >= self.stock_quality_threshold

        n_before = tensor.shape[2]
        n_after = valid_mask.sum().item()

        if self.verbose:
            print(f"  [Second Filter] Before filter: {n_before} stocks")
            print(f"  [Second Filter] After filter: {n_after} stocks")
            print(f"  [Second Filter] Removed: {n_before - n_after}")

            if n_after > 0:
                avg_quality_before = stock_quality_tensor.mean().item()
                avg_quality_after = stock_quality_tensor[valid_mask].mean().item()
                print(f"  [Second Filter] Average quality: {avg_quality_before * 100:.2f}% -> {avg_quality_after * 100:.2f}%")

        if n_after == 0:
            print(f"  [Warning] All stocks filtered!")
            print(f"  [Suggestion] stock_quality_threshold={self.stock_quality_threshold} is too high")
            print(f"  [Keeping] Original data retained")
            return tensor, stock_ids

        tensor_filtered = tensor[:, :, valid_mask]
        stock_ids_filtered = stock_ids[valid_mask.cpu().numpy()]

        if self.verbose:
            final_zero_rate = (tensor_filtered == 0).float().mean().item()
            print(f"  [Second Filter] Final zero rate: {final_zero_rate * 100:.2f}%")

        return tensor_filtered, stock_ids_filtered

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        if self.raw and self.freq == 'day':
            features = change_to_raw(features)
        elif self.raw:
            features = change_to_raw_min(features)

        if self.verbose:
            print(f"  Loading features: {features}")

        df = self._load_exprs(features)
        self.df_bak = df.copy()

        # Phase 1: Original data quality assessment and first filter

        if self.verbose:
            original_missing = df.isnull().sum().sum()
            total_values = df.size
            missing_rate = original_missing / total_values
            print(f"  [Warning] Detected {original_missing:,} missing values ({missing_rate * 100:.2f}%)")

        if self.verbose:
            print(f"  Original data shape: {df.shape}")

        df = self._filter_low_quality_stocks_before_fill(df)

        # Phase 2: Data filling

        if self.smart_fill:
            if self.verbose:
                print(f"  [Processing] Using smart fill strategy...")
            df = self._smart_fill_missing(df)
        else:
            if self.verbose:
                print(f"  [Processing] Using simple fill (0)...")
            df = df.fillna(0)

        remaining_na = df.isnull().sum().sum()
        if remaining_na > 0:
            print(f"  [Warning] DataFrame still has {remaining_na:,} NaN after filling")
        else:
            if self.verbose:
                print(f"  [Success] All missing values filled at DataFrame level")

        # Phase 3: Data restructuring

        df_stacked = df.stack()
        df_unstacked = df_stacked.unstack(level=1)

        new_na = df_unstacked.isnull().sum().sum()
        if new_na > 0:
            if self.verbose:
                print(f"  [Warning] stack/unstack produced {new_na:,} new NaN")
                print(f"  [Processing] Filling again...")

            if self.smart_fill:
                df_unstacked = self._smart_fill_missing(df_unstacked)
            else:
                df_unstacked = df_unstacked.fillna(0)

        if isinstance(df_unstacked.index, pd.MultiIndex):
            dates = df_unstacked.index.levels[0]
        else:
            dates = df_unstacked.index.unique()

        stock_ids = df_unstacked.columns
        values = df_unstacked.values

        if self.verbose:
            print(f"  Restructured shape: {values.shape}")
            print(f"  Trading days: {len(dates)}")
            print(f"  Number of stocks: {len(stock_ids)}")

        # Phase 4: Reshape to 3D tensor

        n_stocks = len(stock_ids)
        n_features = len(features)

        if values.shape[0] % n_features != 0:
            raise ValueError(
                f"Data shape mismatch:\n"
                f"  values.shape[0] = {values.shape[0]}\n"
                f"  n_features = {n_features}\n"
                f"  Not divisible"
            )

        n_dates = values.shape[0] // n_features

        total_expected = n_dates * n_features * n_stocks
        if values.size != total_expected:
            raise ValueError(
                f"Data size mismatch:\n"
                f"  Actual size: {values.size}\n"
                f"  Expected size: {n_dates} x {n_features} x {n_stocks} = {total_expected}"
            )

        values = values.reshape((n_dates, n_features, n_stocks))

        if len(dates) != n_dates:
            if self.verbose:
                print(f"  [Adjusting] dates length from {len(dates)} to {n_dates}")
            dates = dates[:n_dates]

        if self.verbose:
            print(f"  Final data shape: {values.shape}")
            print(f"  Final date range: {dates[0]} to {dates[-1]}")

        # Phase 5: Convert to Tensor and validate

        tensor = torch.tensor(values, dtype=torch.float, device=self.device)

        nan_count = torch.isnan(tensor).sum().item()
        if nan_count > 0:
            print(f"  [Critical Warning] Tensor still has {nan_count:,} NaN values!")
            print(f"  [Processing] Automatically replacing NaN with 0...")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        else:
            if self.verbose:
                print(f"  [Validation] Tensor has no NaN values")

        inf_count = torch.isinf(tensor).sum().item()
        if inf_count > 0:
            print(f"  [Warning] Tensor has {inf_count:,} Inf values")
            print(f"  [Processing] Replacing Inf with 0...")
            tensor = torch.nan_to_num(tensor, posinf=0.0, neginf=0.0)
        else:
            if self.verbose:
                print(f"  [Validation] Tensor has no Inf values")

        # Phase 6: Tensor-level secondary filtering

        if self.verbose:
            zero_count = (tensor == 0).sum().item()
            zero_rate = zero_count / tensor.numel()
            print(f"  [Stats] Zero rate before filtering: {zero_rate * 100:.2f}%")

        tensor, stock_ids = self._filter_low_quality_stocks_after_tensor(tensor, stock_ids)

        if self.verbose:
            zero_count_after = (tensor == 0).sum().item()
            zero_rate_after = zero_count_after / tensor.numel()
            print(f"  [Stats] Zero rate after filtering: {zero_rate_after * 100:.2f}%")

            if zero_rate_after > 0.2:
                print(f"  [Warning] Zero rate still relatively high")
            else:
                print(f"  [Success] Data quality significantly improved")

        return tensor, dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def add_data(self, data: torch.Tensor, dates: pd.Index):
        data = data.to(self.device)
        self.data = torch.cat([self.data, data], dim=0)
        self._dates = pd.Index(list(self._dates) + list(dates))

    def make_dataframe(
            self,
            data: Union[torch.Tensor, List[torch.Tensor]],
            columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)

        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]

        n_days, n_stocks, n_columns = data.shape

        if self.n_days != n_days:
            raise ValueError(
                f"Days mismatch: provided data has {n_days} days, StockData has {self.n_days} days"
            )

        if self.n_stocks != n_stocks:
            raise ValueError(
                f"Stocks mismatch: provided data has {n_stocks} stocks, StockData has {self.n_stocks} stocks"
            )

        if len(columns) != n_columns:
            raise ValueError(
                f"Columns mismatch: provided {len(columns)} columns, data has {n_columns} columns"
            )

        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]

        index = pd.MultiIndex.from_product(
            [date_index, self._stock_ids],
            names=['datetime', 'instrument']
        )

        data_reshaped = data.reshape(-1, n_columns)
        df = pd.DataFrame(
            data_reshaped.detach().cpu().numpy(),
            index=index,
            columns=columns
        )

        return df


class Feature:

    def __init__(self, feature_type: FeatureType):
        self.feature_type = feature_type

    def __repr__(self):
        return f"Feature({self.feature_type.name})"


class Ref:

    def __init__(self, feature, offset: int):
        self.feature = feature
        self.offset = offset

    def __truediv__(self, other):
        return Division(self, other)

    def __sub__(self, other):
        return Subtraction(self, other)

    def __repr__(self):
        return f"Ref({self.feature}, {self.offset})"


class Division:

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __sub__(self, other):
        return Subtraction(self, other)

    def __repr__(self):
        return f"({self.numerator} / {self.denominator})"


class Subtraction:

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} - {self.right})"

