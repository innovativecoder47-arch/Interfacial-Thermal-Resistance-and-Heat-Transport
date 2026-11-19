import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde, norm as normal_dist
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
try:
    from neuralop.models import FNO
    NEURALOP_AVAILABLE = True
except Exception:
    FNO = None
    NEURALOP_AVAILABLE = False
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
C_NAVY = "#0B2545"
C_TEAL = "#13A89E"
C_AMBER = "#F6AE2D"
C_CORAL = "#EF476F"
C_VIOLET = "#7B5EA7"
C_INK = "#1F2933"
C_PAPER = "#FBF9F4"
C_GRID = "#DDD8CB"
CYCLE = [C_NAVY, C_TEAL, C_AMBER, C_CORAL, C_VIOLET]
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    "gfos_thermal", [C_NAVY, C_TEAL, C_AMBER, C_CORAL])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "gfos_diverging", [C_CORAL, C_PAPER, C_TEAL])
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "gfos_error", [C_PAPER, C_AMBER, C_CORAL, C_NAVY])
def apply_gfos_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "figure.facecolor": C_PAPER,
        "axes.facecolor": C_PAPER,
        "savefig.facecolor": C_PAPER,
        "axes.edgecolor": C_INK,
        "axes.labelcolor": C_INK,
        "text.color": C_INK,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": C_GRID,
    })
def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")

def _fmt(value, digits=4, suffix=""):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}" if np.isfinite(value) else "n/a"
class GFOSNet:
    def __init__(
        self,
        excel_path,
        itr_sheet="ITR_clean",
        composite_sheet="Composite_dataset",
        itr_target="itr",
        conductivity_target="k_inplane_W_mK",
        weight_target=None,
        cost_target=None,
        epochs=15,
        batch_size=32,
        learning_rate=0.003,
        k_neighbors=5,
        random_state=42,
        device=None,
        fno_epochs=80,
        fno_samples=240,
        fno_grid=32,
    ):
        self.excel_path = str(excel_path)
        self.itr_sheet = itr_sheet
        self.composite_sheet = composite_sheet
        self.itr_target = itr_target
        self.conductivity_target = conductivity_target
        self.weight_target = weight_target
        self.cost_target = cost_target
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.k_neighbors = int(k_neighbors)
        self.random_state = int(random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(self.random_state)
        self.itr_model = None
        self.composite_models = {}
        self.composite_scaler = None
        self.composite_imputer = None
        self.composite_columns = None
        self.fno_model = None
        self.fno_skip_reason = None
        self.fno_data_source = None
        self.fno_epochs = int(fno_epochs)
        self.fno_samples = int(fno_samples)
        self.fno_grid = int(fno_grid)
        self.optimization_result = None
        self.gnn_history = []
        self.gnn_metrics = {}
        self.fno_metrics = {}
        self.surrogate_metrics = {}
        self.module_metrics = {}
        self.overall_metrics = {}
        self.best_compromise = None
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
    @staticmethod
    def _print_table(title, headers, rows, aligns=None):
        """Print a neat box table. A row equal to None draws a separator line."""
        headers = [str(h) for h in headers]
        body = [None if r is None else [str(c) for c in r] for r in rows]
        aligns = aligns or ["l"] * len(headers)
        widths = [
            max([len(headers[i])] + [len(r[i]) for r in body if r is not None])
            for i in range(len(headers))
        ]
        def rule(left, mid, right, fill):
            return left + mid.join(fill * (w + 2) for w in widths) + right
        def cell(text, width, align):
            if align == "r":
                return text.rjust(width)
            if align == "c":
                return text.center(width)
            return text.ljust(width)
        def line(items, align_list):
            return "│ " + " │ ".join(
                cell(t, w, a) for t, w, a in zip(items, widths, align_list)) + " │"
        top = rule("╭", "┬", "╮", "─")
        print("\n" + title.center(len(top)))
        print(top)
        print(line(headers, ["c"] * len(headers)))
        print(rule("╞", "╪", "╡", "═"))
        for r in body:
            print(rule("├", "┼", "┤", "─") if r is None else line(r, aligns))
        print(rule("╰", "┴", "╯", "─"))
    @staticmethod
    def _clean_frame(df):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
    def load_data(self):
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(
                f"Excel file not found: {self.excel_path}\n"
                "Update excel_path in the final run block."
            )
        xls = pd.ExcelFile(self.excel_path)
        print("\nWorkbook sheets:", xls.sheet_names)

        if self.itr_sheet not in xls.sheet_names:
            raise ValueError(
                f"Sheet '{self.itr_sheet}' not found. Available: {xls.sheet_names}"
            )
        if self.composite_sheet not in xls.sheet_names:
            raise ValueError(
                f"Sheet '{self.composite_sheet}' not found. Available: {xls.sheet_names}"
            )
        self.itr_df = self._clean_frame(pd.read_excel(self.excel_path, sheet_name=self.itr_sheet))
        self.comp_df = self._clean_frame(pd.read_excel(self.excel_path, sheet_name=self.composite_sheet))
        print(f"ITR sheet: {self.itr_df.shape}")
        print(f"Composite sheet: {self.comp_df.shape}")
        print("ITR columns:", list(self.itr_df.columns))
        print("Composite columns:", list(self.comp_df.columns))
        return self
    @staticmethod
    def _find_column(df, requested, candidates, required=True):
        if requested and requested in df.columns:
            return requested
        lower_map = {str(c).lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        if required:
            raise ValueError(
                "Could not identify target column. "
                f"Requested='{requested}'. Available columns: {list(df.columns)}"
            )
        return None
    @staticmethod
    def _make_numeric_features(df, target, drop_words=None):
        """Encode categoricals, remove IDs and obvious target leakage."""
        drop_words = drop_words or []
        data = df.copy()
        drop_cols = [target]
        for col in data.columns:
            cl = str(col).lower()
            if cl in {"id", "sample_id", "record_id", "id-t", "interface"}:
                drop_cols.append(col)
            if any(word in cl for word in drop_words):
                drop_cols.append(col)
        X = data.drop(columns=list(set(drop_cols)), errors="ignore")
        X = X.dropna(axis=1, how="all")
        X = X.loc[:, X.nunique(dropna=True) > 1]
        X = pd.get_dummies(X, dummy_na=True)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how="all")
        return X
    def _build_graph(self, x_scaled):
        n = x_scaled.shape[0]
        if n < 3:
            raise ValueError("At least three usable ITR records are needed for GNN.")
        k = min(self.k_neighbors + 1, n)
        nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn_model.fit(x_scaled)
        neighbours = nn_model.kneighbors(return_distance=False)
        edges = set()
        for i, row in enumerate(neighbours):
            for j in row:
                if i != j:
                    edges.add((i, int(j)))
                    edges.add((int(j), i))
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        return edge_index
    class _GNNRegressor(nn.Module):
        def __init__(self, in_channels, hidden=48):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(32, 1),
            )
        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            return self.head(x).squeeze(-1)
    def train_gnn(self):
        print("\n" + "=" * 65)
        print("1) TRAINING GRAPH NEURAL NETWORK FOR ITR")
        print("=" * 65)
        target_col = self._find_column(
            self.itr_df, self.itr_target,
            ["itr", "ITR", "interfacial_thermal_resistance",
             "thermal_boundary_resistance", "R_K"]
        )
        y = pd.to_numeric(self.itr_df[target_col], errors="coerce")
        valid = y.notna() & np.isfinite(y)
        df = self.itr_df.loc[valid].reset_index(drop=True)
        y = y.loc[valid].to_numpy(dtype=np.float32)
        X = self._make_numeric_features(
            df, target_col,
            drop_words=["target_encoded", "predicted_itr", "itr_prediction"]
        )
        if X.shape[1] == 0:
            raise ValueError("No usable input features remain in the ITR sheet.")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xv = imputer.fit_transform(X)
        Xs = scaler.fit_transform(Xv).astype(np.float32)
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.20, random_state=self.random_state
        )
        edge_index = self._build_graph(Xs)
        data = Data(
            x=torch.tensor(Xs, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.float32),
        ).to(self.device)
        train_mask = torch.zeros(len(y), dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(len(y), dtype=torch.bool, device=self.device)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        y_mean = float(np.mean(y[train_idx]))
        y_std = float(np.std(y[train_idx]) + 1e-8)
        y_norm = (data.y - y_mean) / y_std
        model = self._GNNRegressor(Xs.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        loss_fn = nn.MSELoss()
        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = loss_fn(pred[train_mask], y_norm[train_mask])
            self.gnn_history.append(float(loss.item()))
            loss.backward()
            optimizer.step()
            if epoch == 1 or epoch % 5 == 0 or epoch == self.epochs:
                print(f"GNN epoch {epoch:02d}/{self.epochs} | train loss: {loss.item():.5f}")
        model.eval()
        with torch.no_grad():
            pred_norm = model(data.x, data.edge_index)
            pred_all = pred_norm.cpu().numpy() * y_std + y_mean
        y_true = y[test_idx]
        y_pred = pred_all[test_idx]
        self.gnn_metrics = self._print_metrics("GNN ITR test", y_true, y_pred)
        self.gnn_test_true = y_true
        self.gnn_test_pred = y_pred
        self.itr_model = {
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "columns": list(X.columns),
            "target": target_col,
            "y_mean": y_mean,
            "y_std": y_std,
            "device": self.device,
            "test_true": y_true,
            "test_pred": y_pred,
        }
        return self
    @staticmethod
    def _print_metrics(name, y_true, y_pred):
        """Print regression metrics and return them for reporting/plotting."""
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        if y_true.size != y_pred.size:
            raise ValueError(
                f"Metric calculation failed for {name}: y_true and y_pred "
                "must contain the same number of values."
            )
        if y_true.size == 0:
            raise ValueError(f"Metric calculation failed for {name}: no test samples.")

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan")
        nonzero = np.abs(y_true) > 1e-12
        mape = (
            float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
            if np.any(nonzero) else float("nan")
        )
        value_range = float(np.ptp(y_true))
        nrmse = float(rmse / value_range * 100.0) if value_range > 1e-12 else float("nan")

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_percent": mape,
            "NRMSE_percent": nrmse,
        }
        print(f"\n{name} metrics")
        print(f"MAE        : {mae:.6f}")
        print(f"RMSE       : {rmse:.6f}")
        print(f"R2         : {r2:.6f}")
        print(f"NRMSE      : {nrmse:.4f}%" if np.isfinite(nrmse) else "NRMSE      : undefined")
        print(f"MAPE       : {mape:.4f}%" if np.isfinite(mape) else "MAPE       : undefined (all reference values are zero)")
        return metrics
    def _solve_heat_fvm(self, k_map, r_int):
        """Steady 2D heat conduction (finite volume). T=1 left, T=0 right,
        top/bottom insulated. Face conductance = harmonic mean of the two cells
        in series with an interfacial thermal resistance r_int between unlike phases."""
        n_y, n_x = k_map.shape
        idx = np.arange(n_y * n_x).reshape(n_y, n_x)
        rows, cols, vals = [], [], []
        diag = np.zeros(n_y * n_x)
        rhs = np.zeros(n_y * n_x)
        def face(k1, k2):
            r = 1.0 / (2 * k1) + 1.0 / (2 * k2)
            r = r + r_int * (np.abs(k1 - k2) > 1e-12)
            return 1.0 / r
        g = face(k_map[:, :-1], k_map[:, 1:])
        a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        gv = g.ravel()
        rows += [a, b]; cols += [b, a]; vals += [-gv, -gv]
        np.add.at(diag, a, gv); np.add.at(diag, b, gv)
        g = face(k_map[:-1, :], k_map[1:, :])
        a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
        gv = g.ravel()
        rows += [a, b]; cols += [b, a]; vals += [-gv, -gv]
        np.add.at(diag, a, gv); np.add.at(diag, b, gv)
        gl = 2.0 * k_map[:, 0]
        diag[idx[:, 0]] += gl
        rhs[idx[:, 0]] += gl * 1.0
        gr = 2.0 * k_map[:, -1]
        diag[idx[:, -1]] += gr
        rows.append(np.arange(n_y * n_x)); cols.append(np.arange(n_y * n_x)); vals.append(diag)
        A = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_y * n_x, n_y * n_x))
        return spsolve(A.tocsc(), rhs).reshape(n_y, n_x)

    def generate_thermal_fields(self, input_npy="thermal_inputs.npy",
                                target_npy="thermal_targets.npy"):
        """Build FNO training data by finite-volume simulation of two-phase
        composite microstructures (polymer matrix + conductive filler).
        Matrix/filler conductivities and interfacial resistance come from the workbook."""
        n, size = self.fno_samples, self.fno_grid
        k_matrix, k_filler, r_int = 0.2, 20.0, 0.5
        try:
            kc = self._find_column(self.comp_df, self.conductivity_target,
                                   ["k_inplane_W_mK", "k_inplane", "thermal_conductivity"],
                                   required=False)
            if kc is not None:
                kv = pd.to_numeric(self.comp_df[kc], errors="coerce").dropna()
                if len(kv) and kv.max() > kv.min() > 0:
                    k_matrix, k_filler = float(kv.min()), float(kv.max())
            if getattr(self, "itr_df", None) is not None:
                itr = pd.to_numeric(self.itr_df[self.itr_target], errors="coerce").dropna()
                if len(itr) and itr.max() > 0:
                    r_int = float(np.clip(itr.median() / itr.max(), 0.0, 1.0)) / k_matrix * 0.1
        except Exception:
            pass
        if k_filler <= k_matrix * 1.5:
            k_filler = k_matrix * 50.0
        lk0, lk1 = np.log(k_matrix), np.log(k_filler)
        print(f"Generating {n} FVM temperature fields ({size}x{size}) | "
              f"k_matrix={k_matrix:.3g}, k_filler={k_filler:.3g}, R_int={r_int:.3g}")
        xs = np.zeros((n, size, size), dtype=np.float32)
        ys = np.zeros((n, size, size), dtype=np.float32)
        for i in range(n):
            noise = gaussian_filter(self.rng.standard_normal((size, size)),
                                    sigma=float(self.rng.uniform(1.5, 3.5)), mode="wrap")
            phi = float(self.rng.uniform(0.25, 0.55))
            filler = noise > np.quantile(noise, 1.0 - phi)
            k_map = np.where(filler, k_filler, k_matrix)
            ys[i] = self._solve_heat_fvm(k_map, r_int)
            xs[i] = (np.log(k_map) - lk0) / (lk1 - lk0)
        np.save(input_npy, xs)
        np.save(target_npy, ys)
        print(f"Saved {input_npy} and {target_npy} with shape {xs.shape}")
        return xs, ys

    def train_fno(self, input_npy="thermal_inputs.npy", target_npy="thermal_targets.npy"):
        print("\n" + "=" * 65)
        print("2) FOURIER NEURAL OPERATOR FOR TEMPERATURE FIELDS")
        print("=" * 65)
        if not NEURALOP_AVAILABLE:
            print("NeuralOperator package is not installed. FNO training skipped.")
            print("Install with: pip install neuraloperator")
            self.fno_skip_reason = "neuraloperator not installed"
            return self
        if not (Path(input_npy).exists() and Path(target_npy).exists()):
            print("Spatial field files not found -> generating FVM-simulated fields.")
            self.generate_thermal_fields(input_npy, target_npy)
            self.fno_data_source = "FVM-simulated"
        else:
            self.fno_data_source = "user-supplied"
        x = np.load(input_npy).astype(np.float32)
        y = np.load(target_npy).astype(np.float32)
        if x.ndim == 3:
            x = x[:, None, :, :]
        if y.ndim == 3:
            y = y[:, None, :, :]
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("FNO arrays must have shape (N,H,W) or (N,C,H,W).")
        if x.shape[0] != y.shape[0] or x.shape[-2:] != y.shape[-2:]:
            raise ValueError("FNO input and target arrays must share N, H and W.")
        if x.shape[1] != 1 or y.shape[1] != 1:
            raise ValueError("This compact FNO implementation expects one input and output channel.")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=self.random_state
        )
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        x_test_t = torch.tensor(x_test, dtype=torch.float32, device=self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=self.device)
        loader = DataLoader(
            TensorDataset(x_train_t, y_train_t),
            batch_size=min(self.batch_size, len(x_train_t)),
            shuffle=True
        )
        model = FNO(
            n_modes=(8, 8),
            hidden_channels=24,
            in_channels=1,
            out_channels=1,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        for epoch in range(1, self.fno_epochs + 1):
            model.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = model(xb)
                loss = loss_fn(output, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch == 1 or epoch % 10 == 0 or epoch == self.fno_epochs:
                print(f"FNO epoch {epoch:02d}/{self.fno_epochs} | train loss: {np.mean(losses):.6f}")
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test_t).cpu().numpy()
        self.fno_metrics = self._print_metrics(
            "FNO temperature-field test",
            y_test.reshape(len(y_test), -1),
            y_pred.reshape(len(y_pred), -1)
        )
        self.fno_model = model
        self.fno_test_true = y_test
        self.fno_test_pred = y_pred
        return self
    def _prepare_composite_table(self):
        df = self.comp_df.copy()
        k_col = self._find_column(
            df, self.conductivity_target,
            ["k_inplane_W_mK", "k_inplane", "in_plane_thermal_conductivity",
             "effective_thermal_conductivity", "thermal_conductivity"]
        )
        w_col = self._find_column(
            df, self.weight_target,
            ["areal_weight_g_m2", "areal_weight", "weight", "density_kg_m3",
             "density", "composite_density"],
            required=False
        )
        c_col = self._find_column(
            df, self.cost_target,
            ["cost_per_kg", "cost_usd_kg", "cost", "production_cost",
             "cost_per_area"],
            required=False
        )
        if w_col is None or c_col is None:
            raise ValueError(
                "NSGA-II needs conductivity, weight/density, and cost target columns. "
                f"Detected conductivity='{k_col}', weight='{w_col}', cost='{c_col}'. "
                "Set weight_target and cost_target in the class constructor to match "
                "your workbook's exact column names."
            )
        targets = {"conductivity": k_col, "weight": w_col, "cost": c_col}
        for name, col in targets.items():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=list(targets.values())).reset_index(drop=True)
        X = self._make_numeric_features(
            df, k_col,
            drop_words=["pareto", "best_compromise", "feasible", "optimal",
                        "predicted", "recommendation"]
        )
        X = X.drop(columns=[w_col, c_col], errors="ignore")
        y = {
            "conductivity": df[k_col].to_numpy(dtype=float),
            "weight": df[w_col].to_numpy(dtype=float),
            "cost": df[c_col].to_numpy(dtype=float),
        }
        return X, y, targets
    def train_optimization_surrogates(self):
        print("\n" + "=" * 65)
        print("3) TRAINING AUXILIARY SURROGATES FOR NSGA-II")
        print("=" * 65)
        X, y, targets = self._prepare_composite_table()
        self.surrogate_metrics = {}
        self.composite_columns = list(X.columns)
        self.composite_imputer = SimpleImputer(strategy="median")
        self.composite_scaler = StandardScaler()
        Xv = self.composite_imputer.fit_transform(X)
        Xs = self.composite_scaler.fit_transform(Xv)
        for name, target in y.items():
            X_train, X_test, y_train, y_test = train_test_split(
                Xs, target, test_size=0.20, random_state=self.random_state
            )
            model = RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            self.surrogate_metrics[name] = self._print_metrics(
                f"Auxiliary {name} surrogate", y_test, pred)
            self.composite_models[name] = model
        self.observed_means = {k: float(np.nanmean(v)) for k, v in y.items()}
        self.observed_ranges = {k: (float(np.nanmin(v)), float(np.nanmax(v))) for k, v in y.items()}
        def _n(v):
            return (v - np.nanmin(v)) / (np.ptp(v) + 1e-12)
        ref_score = _n(y["weight"]) + _n(y["cost"]) + (1.0 - _n(y["conductivity"]))
        ref_idx = int(np.nanargmin(ref_score))
        self.reference_best = {k: float(v[ref_idx]) for k, v in y.items()}
        self.optimization_feature_min = np.nanpercentile(Xs, 2, axis=0)
        self.optimization_feature_max = np.nanpercentile(Xs, 98, axis=0)
        self.optimization_feature_median = np.nanmedian(Xs, axis=0)
        self.optimization_targets = targets
        return self
    def run_nsga2(self, population_size=40, generations=30):
        print("\n" + "=" * 65)
        print("4) NSGA-II MULTIOBJECTIVE MATERIAL OPTIMIZATION")
        print("=" * 65)
        if not self.composite_models:
            raise RuntimeError("Train optimization surrogates before running NSGA-II.")
        lo = self.optimization_feature_min
        hi = self.optimization_feature_max
        med = self.optimization_feature_median
        n_var = len(med)
        class MaterialProblem(ElementwiseProblem):
            def __init__(problem_self, outer):
                problem_self.outer = outer
                super().__init__(
                    n_var=n_var,
                    n_obj=3,
                    n_ieq_constr=0,
                    xl=lo,
                    xu=hi
                )
            def _evaluate(problem_self, x, out, *args, **kwargs):
                xx = np.asarray(x, dtype=float).reshape(1, -1)
                k_pred = float(problem_self.outer.composite_models["conductivity"].predict(xx)[0])
                w_pred = float(problem_self.outer.composite_models["weight"].predict(xx)[0])
                c_pred = float(problem_self.outer.composite_models["cost"].predict(xx)[0])
                out["F"] = [-k_pred, w_pred, c_pred]
        problem = MaterialProblem(self)
        algorithm = NSGA2(pop_size=int(population_size))
        result = minimize(
            problem,
            algorithm,
            get_termination("n_gen", int(generations)),
            seed=self.random_state,
            verbose=False
        )
        self.optimization_result = result
        if result.X is None:
            raise RuntimeError("NSGA-II did not return a feasible population.")
        Xopt = np.atleast_2d(result.X)
        Fopt = np.atleast_2d(result.F)
        k_values = -Fopt[:, 0]
        w_values = Fopt[:, 1]
        c_values = Fopt[:, 2]
        output = pd.DataFrame({
            "predicted_inplane_conductivity": k_values,
            "predicted_weight_or_density": w_values,
            "predicted_cost": c_values,
        })
        display_cols = ["predicted_inplane_conductivity",
                        "predicted_weight_or_density", "predicted_cost"]
        vals = output[display_cols].to_numpy(dtype=float)
        norm = (vals - vals.min(axis=0)) / (np.ptp(vals, axis=0) + 1e-12)
        output["compromise_score"] = norm[:, 1] + norm[:, 2] + (1.0 - norm[:, 0])
        Xopt_original = self.composite_scaler.inverse_transform(Xopt)
        for i, col in enumerate(self.composite_columns):
            output[f"decision_{col}"] = Xopt_original[:, i]
        output = output.sort_values(
            ["predicted_cost", "predicted_weight_or_density"]
        ).reset_index(drop=True)
        output.to_csv("nsga2_pareto_solutions.csv", index=False)
        best_idx = int(output["compromise_score"].idxmin())
        self.best_compromise = output.loc[best_idx]
        print(f"\nPareto solutions found: {len(output)}")
        print(f"Best compromise score  : {self.best_compromise['compromise_score']:.5f}")
        print("Full Pareto decision-variable table saved to: nsga2_pareto_solutions.csv")
        return output
    def compute_overall_metrics(self):
        modules = {}
        if self.gnn_metrics:
            modules["GNN | ITR regression"] = self.gnn_metrics
        if self.fno_metrics:
            modules["FNO | Temperature field"] = self.fno_metrics
        for name, metrics in self.surrogate_metrics.items():
            modules[f"RF surrogate | {name.capitalize()}"] = metrics
        self.module_metrics = modules

        r2 = [m["R2"] for m in modules.values()]
        nrmse = [m["NRMSE_percent"] for m in modules.values()]
        r2_clipped = [min(max(v, 0.0), 1.0) for v in r2 if np.isfinite(v)]

        self.overall_metrics = {
            "modules": len(modules),
            "R2": _nanmean(r2),
            "NRMSE_percent": _nanmean(nrmse),
            "score": 100.0 * _nanmean(r2_clipped),
            "k_gain_percent": float("nan"),
            "weight_reduction_percent": float("nan"),
            "cost_saving_percent": float("nan"),
        }
        if self.best_compromise is not None and hasattr(self, "observed_ranges"):
            def attain(value, key, higher_is_better):
                lo_, hi_ = self.observed_ranges[key]
                span = hi_ - lo_
                if span <= 1e-12:
                    return float("nan")
                frac = (value - lo_) / span if higher_is_better else (hi_ - value) / span
                return float(frac * 100.0)
            bc = self.best_compromise
            self.overall_metrics["k_gain_percent"] = attain(
                float(bc["predicted_inplane_conductivity"]), "conductivity", True)
            self.overall_metrics["weight_reduction_percent"] = attain(
                float(bc["predicted_weight_or_density"]), "weight", False)
            self.overall_metrics["cost_saving_percent"] = attain(
                float(bc["predicted_cost"]), "cost", False)
        return self.overall_metrics
    @staticmethod
    def _show(ax, title):
        fig = ax.figure
        ax.set_title(title, loc="left", pad=14, color=C_NAVY)
        fig.text(0.99, 0.012, "GFOS-Net", ha="right", va="bottom",
                 fontsize=9, color=C_INK, alpha=0.45, style="italic")
        fig.tight_layout()
        plt.show()
    @staticmethod
    def _stat_box(ax, text, loc=(0.03, 0.97)):
        ax.text(loc[0], loc[1], text, transform=ax.transAxes, va="top", ha="left",
                fontsize=10.5, color=C_INK,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=C_GRID, alpha=0.95))
    def generate_visualizations(self, pareto):
        """Generate 11 publication-style plots plus optional FNO field plots."""
        apply_gfos_style()
        if not self.overall_metrics:
            self.compute_overall_metrics()
        best = self.best_compromise
        kcol = "predicted_inplane_conductivity"
        wcol = "predicted_weight_or_density"
        ccol = "predicted_cost"
        hist = np.asarray(self.gnn_history, dtype=float)
        ep = np.arange(1, len(hist) + 1)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.fill_between(ep, hist, hist.min(), color=C_TEAL, alpha=0.20, zorder=1)
        ax.plot(ep, hist, color=C_NAVY, lw=2.6, zorder=3)
        ax.scatter(ep, hist, s=60, facecolor=C_PAPER, edgecolor=C_NAVY,
                   linewidth=1.8, zorder=4)
        imin = int(np.argmin(hist))
        ax.scatter(ep[imin], hist[imin], marker="*", s=420, color=C_AMBER,
                   edgecolor=C_NAVY, linewidth=1.2, zorder=5,
                   label=f"Lowest loss = {hist[imin]:.4f} (epoch {ep[imin]})")
        ax.set(xlabel="Training epoch", ylabel="Training MSE loss (normalised ITR)")
        ax.grid(alpha=0.6)
        ax.legend(loc="upper right")
        self._show(ax, "GNN Training Convergence")
        yt, yp = np.asarray(self.gnn_test_true), np.asarray(self.gnn_test_pred)
        err = np.abs(yt - yp)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo + 1e-12)
        xs = np.linspace(lo - pad, hi + pad, 200)
        band = 0.10 * (hi - lo + 1e-12)
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        ax.fill_between(xs, xs - band, xs + band, color=C_AMBER, alpha=0.18,
                        label="±10 % of range", zorder=1)
        ax.plot(xs, xs, "--", color=C_NAVY, lw=2, label="Ideal (y = x)", zorder=2)
        sc = ax.scatter(yt, yp, c=err, cmap=THERMAL_CMAP, s=95, edgecolor="white",
                        linewidth=0.9, zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Absolute error")
        cb.outline.set_edgecolor(C_GRID)
        ax.set(xlabel="Measured ITR", ylabel="Predicted ITR")
        ax.grid(alpha=0.6)
        self._show(ax, "GNN ITR: Measured vs Predicted")
        residual = yt - yp
        sigma = float(np.std(residual))
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        ax.axhspan(-1.96 * sigma, 1.96 * sigma, color=C_VIOLET, alpha=0.10,
                   label="±1.96 σ band", zorder=1)
        colors = np.where(residual >= 0, C_TEAL, C_CORAL)
        ax.vlines(yp, 0, residual, colors=colors, alpha=0.55, lw=1.6, zorder=2)
        ax.scatter(yp, residual, c=colors, s=75, edgecolor="white", linewidth=0.8, zorder=3)
        ax.axhline(0, color=C_NAVY, lw=1.8, ls="--", zorder=2)
        ax.set(xlabel="Predicted ITR", ylabel="Residual (measured − predicted)")
        ax.grid(alpha=0.6)
        ax.legend(loc="best")
        self._show(ax, "GNN ITR Residual Analysis")
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        bins = min(20, max(5, len(residual) // 2))
        ax.hist(residual, bins=bins, density=True, color=C_TEAL, alpha=0.55,
                edgecolor="white", linewidth=1.2, label="Residual histogram")
        grid_x = np.linspace(residual.min() - sigma, residual.max() + sigma, 300)
        if len(residual) >= 3 and sigma > 0:
            try:
                ax.plot(grid_x, gaussian_kde(residual)(grid_x), color=C_NAVY, lw=2.6,
                        label="Kernel density")
            except Exception:
                pass
            ax.plot(grid_x, normal_dist.pdf(grid_x, residual.mean(), sigma),
                    color=C_CORAL, lw=2, ls="--", label="Normal fit")
        ax.axvline(0, color=C_AMBER, lw=2.2)
        ax.set(xlabel="ITR prediction residual", ylabel="Probability density")
        ax.grid(axis="y", alpha=0.6)
        ax.legend(loc="upper right")
        self._show(ax, "Distribution of GNN ITR Prediction Errors")
        names = list(self.surrogate_metrics)
        r2_vals = [self.surrogate_metrics[n]["R2"] for n in names]
        labels = [n.capitalize() for n in names]
        ypos = np.arange(len(names))
        colors = [CYCLE[(i + 1) % len(CYCLE)] for i in range(len(names))]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.hlines(ypos, 0, r2_vals, colors=colors, lw=4, alpha=0.85, zorder=2)
        ax.scatter(r2_vals, ypos, s=300, c=colors, edgecolor="white", linewidth=1.6, zorder=3)
        for yy, v in zip(ypos, r2_vals):
            ax.text(v + 0.03 * np.sign(v if v else 1), yy, f"{v:.3f}", va="center",
                    ha="left" if v >= 0 else "right", color=C_INK, fontsize=11.5)
        ax.axvline(0, color=C_INK, lw=1)
        ax.set_yticks(ypos, labels)
        ax.set_xlim(min(0, min(r2_vals)) - 0.1, max(1.0, max(r2_vals)) + 0.15)
        ax.set(xlabel="Test R²")
        ax.grid(axis="x", alpha=0.6)
        ax.invert_yaxis()
        self._show(ax, "Auxiliary Surrogate Model Performance (R²)")
        fig = plt.figure(figsize=(8.4, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(pareto[kcol], pareto[wcol], pareto[ccol], c=pareto[ccol],
                        cmap=THERMAL_CMAP, s=60, edgecolors="white", linewidths=0.5,
                        depthshade=True)
        ax.scatter([best[kcol]], [best[wcol]], [best[ccol]], marker="*", s=420,
                   color=C_AMBER, edgecolor=C_NAVY, linewidth=1.3, label="Best compromise")
        ax.set(xlabel="In-plane conductivity", ylabel="Weight / density", zlabel="Cost")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0.0))
        ax.view_init(elev=24, azim=-58)
        cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.1)
        cb.set_label("Predicted cost")
        cb.outline.set_edgecolor(C_GRID)
        ax.legend(loc="upper left")
        self._show(ax, "NSGA-II Three-Objective Pareto Front")
        kn = (pareto[kcol] - pareto[kcol].min()) / (np.ptp(pareto[kcol]) + 1e-12)
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        order = np.argsort(pareto[ccol].to_numpy())
        ax.plot(pareto[ccol].to_numpy()[order], pareto[kcol].to_numpy()[order],
                color=C_GRID, lw=1.6, ls="--", zorder=1)
        sc = ax.scatter(pareto[ccol], pareto[kcol], c=pareto[wcol], cmap=THERMAL_CMAP,
                        s=50 + 170 * kn, edgecolor="white", linewidth=0.9, alpha=0.95, zorder=3)
        ax.scatter(best[ccol], best[kcol], marker="*", s=480, color=C_AMBER,
                   edgecolor=C_NAVY, linewidth=1.3, zorder=4, label="Best compromise")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Predicted weight / density")
        cb.outline.set_edgecolor(C_GRID)
        ax.set(xlabel="Predicted cost", ylabel="Predicted in-plane conductivity")
        ax.grid(alpha=0.6)
        ax.legend(loc="best")
        self._show(ax, "Conductivity–Cost Trade-off (marker size ∝ conductivity)")
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
        sc = ax.scatter(pareto[wcol], pareto[ccol], c=pareto[kcol], cmap=THERMAL_CMAP,
                        s=95, edgecolor="white", linewidth=0.9, alpha=0.95, zorder=3)
        ax.scatter(best[wcol], best[ccol], marker="*", s=480, color=C_AMBER,
                   edgecolor=C_NAVY, linewidth=1.3, zorder=4, label="Best compromise")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Predicted conductivity")
        cb.outline.set_edgecolor(C_GRID)
        ax.set(xlabel="Predicted weight / density", ylabel="Predicted cost")
        ax.grid(alpha=0.6)
        ax.legend(loc="best")
        self._show(ax, "Weight–Cost Trade-off")
        cols = [kcol, wcol, ccol]
        a = pareto[cols].to_numpy(float)
        a = (a - a.min(axis=0)) / (np.ptp(a, axis=0) + 1e-12)
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        for x in range(3):
            ax.axvline(x, color=C_INK, lw=1.2, alpha=0.6, zorder=1)
        for row, kv in zip(a, a[:, 0]):
            ax.plot(range(3), row, color=THERMAL_CMAP(kv), alpha=0.55, lw=1.6, zorder=2)
        best_row = np.array([(best[c] - pareto[c].min()) / (np.ptp(pareto[c]) + 1e-12) for c in cols])
        ax.plot(range(3), best_row, color=C_NAVY, lw=4.2, zorder=3)
        ax.plot(range(3), best_row, color=C_AMBER, lw=2.2, marker="o", ms=9,
                mec=C_NAVY, zorder=4, label="Best compromise")
        ax.set_xticks([0, 1, 2], ["Conductivity (max)", "Weight/density (min)", "Cost (min)"])
        ax.set_ylabel("Min–max normalised objective value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.6)
        cb = fig.colorbar(ScalarMappable(Normalize(0, 1), THERMAL_CMAP), ax=ax, pad=0.02)
        cb.set_label("Normalised conductivity")
        cb.outline.set_edgecolor(C_GRID)
        ax.legend(loc="upper center")
        self._show(ax, "Parallel-Coordinate View of Pareto Solutions")
        corr = pareto[cols].corr()
        labels = ["Conductivity", "Weight/density", "Cost"]
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap=DIVERGING_CMAP)
        ax.set_xticks(range(3), labels, rotation=20, ha="right")
        ax.set_yticks(range(3), labels)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", lw=3)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(3):
            for j in range(3):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=14,
                        color="white" if abs(val) > 0.65 else C_INK)
        cb = fig.colorbar(im, ax=ax, pad=0.03)
        cb.set_label("Pearson correlation")
        cb.outline.set_edgecolor(C_GRID)
        self._show(ax, "Correlation Between Optimization Objectives")
        if self.fno_model is not None and hasattr(self, "fno_test_true"):
            ref = self.fno_test_true[0, 0]
            pred = self.fno_test_pred[0, 0]
            absr = np.abs(ref - pred)
            vmin, vmax = min(ref.min(), pred.min()), max(ref.max(), pred.max())
            for label, field, cmap, vlim in [
                ("Reference temperature field", ref, THERMAL_CMAP, (vmin, vmax)),
                ("FNO predicted temperature field", pred, THERMAL_CMAP, (vmin, vmax)),
                ("FNO absolute error", absr, ERROR_CMAP, (None, None)),
            ]:
                fig, ax = plt.subplots(figsize=(6.6, 5.4))
                im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap,
                               vmin=vlim[0], vmax=vlim[1], interpolation="bicubic")
                ax.set(xlabel="Spatial x", ylabel="Spatial y")
                ax.grid(False)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("Temperature" if "error" not in label else "|Error|")
                cb.outline.set_edgecolor(C_GRID)
                self._show(ax, f"Additional FNO: {label}")
    def print_final_report(self, pareto):
        if not self.overall_metrics:
            self.compute_overall_metrics()
        overall = self.overall_metrics
        print("\n" + "═" * 100)
        print("GFOS-Net | FINAL MODEL EVALUATION REPORT".center(100))
        print("═" * 100)
        fno_status = ("Completed" if self.fno_model is not None
                      else f"Skipped ({self.fno_skip_reason or 'not requested'})")
        if self.fno_model is not None:
            fno_status = f"Completed ({self.fno_data_source} fields)"
        self._print_table(
            "TABLE 1 | GFOS-Net ALGORITHM FRAMEWORK",
            ["No.", "Algorithm", "Purpose", "Status"],
            [
                ["1", "Graph Neural Network (GNN)",
                 "Predict interfacial thermal resistance",
                 "Completed" if self.itr_model else "Not run"],
                ["2", "Fourier Neural Operator (FNO)",
                 "Predict heat distribution and temperature maps",
                 fno_status],
                ["3", "NSGA-II",
                 "Optimize conductivity, weight and cost",
                 "Completed" if self.optimization_result is not None else "Not run"],
            ],
            aligns=["c", "l", "l", "l"],
        )
        yt = np.asarray(self.gnn_test_true, dtype=float)
        yp = np.asarray(self.gnn_test_pred, dtype=float)
        rows = []
        best_order = np.argsort(np.abs(yt - yp))[:10]
        for rank, i in enumerate(best_order, start=1):
            ae = abs(yt[i] - yp[i])
            rel = ae / abs(yt[i]) * 100.0 if abs(yt[i]) > 1e-12 else float("nan")
            rows.append([f"Test record {i + 1} (rank {rank})", _fmt(yt[i], 5),
                         _fmt(yp[i], 5), _fmt(ae, 5), _fmt(rel, 2)])
        rows.append(None)
        rows.append([f"Mean of all {len(yt)} test records", _fmt(yt.mean(), 5),
                     _fmt(yp.mean(), 5), _fmt(self.gnn_metrics["MAE"], 5),
                     _fmt(self.gnn_metrics["MAPE_percent"], 2)])
        self._print_table(
            f"TABLE 2 | GNN OUTPUT: PREDICTED ITR (10 BEST-PREDICTED OF {len(yt)} TEST RECORDS)",
            ["Test record", "Measured ITR", "Predicted ITR", "Absolute error", "Relative error (%)"],
            rows,
            aligns=["l", "r", "r", "r", "r"],
        )
        kcol = "predicted_inplane_conductivity"
        wcol = "predicted_weight_or_density"
        ccol = "predicted_cost"
        ranked = pareto.sort_values("compromise_score").reset_index(drop=True)
        rows = []
        for i, r in ranked.head(10).iterrows():
            rows.append([f"#{i + 1}", _fmt(r[kcol], 4), _fmt(r[wcol], 4), _fmt(r[ccol], 4),
                         _fmt(r["compromise_score"], 4),
                         "Best compromise" if i == 0 else ""])
        rows.append(None)
        for label, fn in [("Front minimum", np.min), ("Front mean", np.mean),
                          ("Front maximum", np.max)]:
            rows.append([label, _fmt(fn(pareto[kcol]), 4), _fmt(fn(pareto[wcol]), 4),
                         _fmt(fn(pareto[ccol]), 4), "-", ""])
        self._print_table(
            f"TABLE 3 | NSGA-II OUTPUT: PARETO-OPTIMAL FORMULATIONS ({len(pareto)} solutions)",
            ["Rank / statistic", "Conductivity (max)", "Weight/density (min)",
             "Cost (min)", "Compromise score", "Remark"],
            rows,
            aligns=["l", "r", "r", "r", "r", "l"],
        )

        print("GFOS-Net OVERALL METRIC".center(100))
        print(f"  Models aggregated              : {overall['modules']}")
        print(f"  Overall R2 (mean)              : {_fmt(overall['R2'], 4)}")
        print(f"  Overall NRMSE (mean)           : {_fmt(overall['NRMSE_percent'], 2, ' %')}")

        print("  " + "-" * 96)
        print("  Concept metrics (position of NSGA-II best compromise in the observed data range,")
        print("  0 % = worst real record, 100 % = best real record):")
        print(f"  Thermal conductivity attainment: {_fmt(overall['k_gain_percent'], 1, ' %')}")
        print(f"  Weight / density attainment    : {_fmt(overall['weight_reduction_percent'], 1, ' %')}")
        print(f"  Production cost attainment     : {_fmt(overall['cost_saving_percent'], 1, ' %')}")
        if self.best_compromise is not None and hasattr(self, "observed_ranges"):
            checks = []
            for label, key, col in [("k", "conductivity", "predicted_inplane_conductivity"),
                                    ("weight", "weight", "predicted_weight_or_density"),
                                    ("cost", "cost", "predicted_cost")]:
                lo_, hi_ = self.observed_ranges[key]
                inside = lo_ <= float(self.best_compromise[col]) <= hi_
                checks.append(f"{label}: {'inside' if inside else 'OUTSIDE'}")
            print(f"  Predicted values vs. observed data range -> " + " | ".join(checks))
        if self.fno_model is None:
            print()
    def run_all(self, train_fno_fields=False):
        self.load_data()
        self.train_gnn()
        if train_fno_fields:
            self.train_fno()
        self.train_optimization_surrogates()
        pareto = self.run_nsga2()
        self.compute_overall_metrics()
        self.generate_visualizations(pareto)
        self.print_final_report(pareto)
        print("\n" + "=" * 65)
        print("PIPELINE COMPLETED")
        print("=" * 65)
        print("GNN: ITR regression completed.")
        if self.fno_model is None:
            print(f"FNO: not trained ({self.fno_skip_reason or 'not requested'}).")
        else:
            print(f"FNO: temperature-field training completed ({self.fno_data_source} fields).")
        print("NSGA-II: Pareto optimization completed.")
        return pareto
if __name__ == "__main__":
    EXCEL_FILE = r"dataset.xlsx"
    model = GFOSNet(
        excel_path=EXCEL_FILE,
        itr_sheet="ITR_clean",
        composite_sheet="Composite_dataset",
        itr_target="itr",
        conductivity_target="k_inplane_W_mK",
        weight_target=None,
        cost_target=None,
        epochs=15,
        batch_size=32,
        learning_rate=0.003,
        k_neighbors=5,
        random_state=42,
    )
    pareto_table = model.run_all(train_fno_fields=True)
