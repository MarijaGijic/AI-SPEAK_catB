import os
import json
import csv
import shutil
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.Visualization.visualization import (
    plot_loss_curves,
    plot_per_blendshape_mse,
    plot_prediction_vs_target,
    plot_blendshape_heatmap,
    plot_mfcc,
    plot_phoneme_alignment,
    plot_scatter_pred_vs_target,
    plot_velocity_profile,
    plot_error_correlation,
    plot_sample_report,
)

RESULTS_ROOT = "/content/results"


class ResultsManager:

    def __init__(
        self,
        model_name: str = "model",
        results_root: str = RESULTS_ROOT,
        session_id: Optional[str] = None,
        display_inline: bool = False,
    ) -> None:

        ts = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name    = f"{model_name}_{ts}"
        self.session_dir     = os.path.join(results_root, self.session_name)
        self.display_inline  = display_inline

        self.plots_dir   = os.path.join(self.session_dir, "plots")
        self.samples_dir = os.path.join(self.plots_dir,   "samples")
        self.ckpt_dir    = os.path.join(self.session_dir, "checkpoints")

        for d in (self.session_dir, self.plots_dir, self.samples_dir, self.ckpt_dir):
            os.makedirs(d, exist_ok=True)

        self._history: Dict[str, List[float]] = {}
        self._csv_path   = os.path.join(self.session_dir, "metrics.csv")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file   = None

        print(f"[ResultsManager] Sesija: {self.session_dir}")


    def log_epoch(
        self,
        epoch: int,
        train: Dict[str, float],
        val: Dict[str, float],
    ) -> None:

        row: Dict[str, Any] = {"epoch": epoch}

        for k, v in train.items():
            key = f"train_{k}"
            row[key] = v
            self._history.setdefault(key, []).append(float(v))
        for k, v in val.items():
            key = f"val_{k}"
            row[key] = v
            self._history.setdefault(key, []).append(float(v))

        if self._csv_writer is None:
            self._csv_file   = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def get_history(self) -> Dict[str, List[float]]:
        return dict(self._history)


    def save_config(self, config: Dict[str, Any]) -> str:
        path = os.path.join(self.session_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        return path

    def save_summary(self, metrics: Dict[str, Any]) -> str:
        path = os.path.join(self.session_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        return path

    def register_checkpoint(self, ckpt_path: str) -> None:
        log_path = os.path.join(self.ckpt_dir, "log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}  {ckpt_path}\n")

    def copy_checkpoint(self, ckpt_path: str) -> str:
        dest = os.path.join(self.ckpt_dir, os.path.basename(ckpt_path))
        shutil.copy2(ckpt_path, dest)
        return dest


    def save_loss_curves(
        self,
        title: str = "Loss krive",
        display_inline: Optional[bool] = None,
    ) -> str:
        
        if not self._history:
            print("[ResultsManager] Nema logovane istorije - preskacem loss krive")
            return ""
        path = os.path.join(self.plots_dir, "loss_curves.png")
        show = display_inline if display_inline is not None else self.display_inline
        plot_loss_curves(self._history, save_path=path, title=title, display_inline=show)
        return path

    def save_per_blendshape_mse(
        self,
        mse_per_bs: np.ndarray,
        title: str = "MSE po blendshape-u",
        suffix: str = "",
        display_inline: Optional[bool] = None,
    ) -> str:
        fname = f"per_blendshape_mse{'_' + suffix if suffix else ''}.png"
        path  = os.path.join(self.plots_dir, fname)
        show  = display_inline if display_inline is not None else self.display_inline
        plot_per_blendshape_mse(mse_per_bs, save_path=path, title=title, display_inline=show)
        return path

    def save_error_correlation(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        suffix: str = "",
        display_inline: Optional[bool] = None,
    ) -> str:
        fname = f"error_correlation{'_' + suffix if suffix else ''}.png"
        path  = os.path.join(self.plots_dir, fname)
        show  = display_inline if display_inline is not None else self.display_inline
        plot_error_correlation(pred, target, save_path=path, display_inline=show)
        return path

    def save_prediction(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mfcc_feats: Optional[np.ndarray] = None,
        phoneme_ids: Optional[np.ndarray] = None,
        phoneme_trel: Optional[np.ndarray] = None,
        name: str = "sample",
        save_all: bool = True,
        display_inline: Optional[bool] = None,
    ) -> Dict[str, str]:

        saved: Dict[str, str] = {}
        safe_name = name.replace(os.sep, "_").replace(" ", "_")
        show = display_inline if display_inline is not None else self.display_inline

        def p(suffix: str) -> str:
            return os.path.join(self.samples_dir, f"{safe_name}_{suffix}.png")

        path = p("pred_vs_target")
        plot_prediction_vs_target(pred, target, save_path=path,
                                  sample_name=name, display_inline=show)
        saved["pred_vs_target"] = path

        if save_all:
            path = p("heatmap")
            plot_blendshape_heatmap(pred, target, save_path=path,
                                    sample_name=name, display_inline=show)
            saved["heatmap"] = path

            path = p("scatter")
            plot_scatter_pred_vs_target(pred, target, save_path=path,
                                        sample_name=name, display_inline=show)
            saved["scatter"] = path

            path = p("velocity")
            plot_velocity_profile(pred, target, save_path=path,
                                  sample_name=name, display_inline=show)
            saved["velocity"] = path

            if mfcc_feats is not None:
                path = p("mfcc")
                plot_mfcc(mfcc_feats, save_path=path,
                          sample_name=name, display_inline=show)
                saved["mfcc"] = path

            if phoneme_ids is not None and phoneme_trel is not None:
                path = p("phonemes")
                plot_phoneme_alignment(phoneme_ids, phoneme_trel,
                                       save_path=path, sample_name=name,
                                       display_inline=show)
                saved["phonemes"] = path

            if mfcc_feats is not None and phoneme_ids is not None and phoneme_trel is not None:
                path = p("report")
                plot_sample_report(
                    pred, target, mfcc_feats, phoneme_ids, phoneme_trel,
                    save_path=path, sample_name=name, display_inline=show,
                )
                saved["report"] = path

        return saved


    def finalize(self) -> str:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

        print(f"\n[ResultsManager] Sesija zavrsena.")
        print(f"[ResultsManager]   Rezultati: {os.path.abspath(self.session_dir)}\n")
        return self.session_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False