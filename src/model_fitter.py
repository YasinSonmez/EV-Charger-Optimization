import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import ast
import json
import os

def model(x, a, b, cap, fft):
    """Model function used for generating and fitting data."""
    return fft * (1 + a * (x / cap)**b)

class TrafficModelFitter:
    def __init__(self, n_links=50, n_samples=500, pandas_df=None):
        if pandas_df is None:
            self.df = pd.DataFrame()
            self.n_links = n_links
            self.n_samples = n_samples
            self.generate_data_for_links()
        else:
            self.df = pandas_df
            self.n_links = len(self.df)
            self.n_samples = len(self.df.at[0,'x_vector'])
            print(self.n_links, self.n_samples)

    def generate_data_for_links(self):
        """Generates synthetic data for each link."""
        params = np.random.uniform([0.5, 30, 500, 0.1, 2, 0.05], [10.0, 70, 2000, 0.2, 20, 0.5], (self.n_links, 6))
        link_data = []
        for link_id in range(self.n_links):
            link_params = params[link_id]
            x_vector = np.random.uniform(0, 2500, self.n_samples)
            noise = np.random.normal(scale=link_params[5], size=self.n_samples)
            y_vector = model(x_vector, link_params[3], link_params[4], link_params[0], link_params[1]) * (1 + noise)

            link_data.append({
                'link_id': link_id,
                'link_length': link_params[0],
                'free_flow_speed': link_params[1],
                'capacity': link_params[2],
                'a_true': link_params[3],
                'b_true': link_params[4],
                'noise_scale': link_params[5],
                'x_vector': x_vector,
                'y_vector': y_vector,
                'a_fit': None,
                'b_fit': None,
                'cap_fit': None,
                'fft_fit': None,
                'R^2': None
            })

        self.df = pd.DataFrame(link_data)

    def fit_and_evaluate(self, link_row, r2_threshold=0.5, variation_ratio_threshold=0.03, correlation_threshold=0.3):
        """Fits the full model first; if data doesn't show clear increasing trend or fit is poor, falls back to mean model."""

        if getattr(self, 'bpr_mode', None) == 'historical_artifact_compatible':
            return self._fit_historical(link_row, r2_threshold, variation_ratio_threshold,
                                         correlation_threshold)

        if bool(getattr(self, 'fixed_references', False)):
            return self._fit_fixed_references(link_row)

        # Check if `x_vector` is all zeros
        if np.all(np.asarray(link_row['x_vector']) == 0):
            return link_row['link_id'], np.nan, np.nan, np.nan, np.nan, np.nan

        x = np.asarray(link_row['x_vector'], dtype=float)
        y = np.asarray(link_row['y_vector'], dtype=float)
        strict_full = bool(getattr(self, 'require_full_fit', False))
        r2_threshold = float(getattr(self, 'r2_threshold', r2_threshold))
        
        # Calculate correlation coefficient to check for increasing trend
        correlation = np.corrcoef(x, y)[0, 1] if np.std(y) > 0 and np.std(x) > 0 else 0.0
        if not np.isfinite(correlation):
            correlation = 0.0
        
        # Calculate a dimensionless variation metric.  Dividing y_std by
        # x_std is invalid here because x is vehicles/hour and y is seconds;
        # with realistic flow magnitudes it makes every changing curve look
        # constant.  Relative travel-time variation is independent of flow
        # units and is what the non-strict screening threshold represents.
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        variation_ratio = y_std / max(abs(y_mean), np.finfo(float).eps)

        # Use mean model if:
        # 1. Data shows weak positive correlation (not clearly increasing)
        # 2. Or if variation ratio is too low (y doesn't vary much with x)
        if not strict_full and (correlation < correlation_threshold or variation_ratio < variation_ratio_threshold):
            fft_constant = np.mean(y)
            r2 = r2_score(y, np.full_like(y, fft_constant)) if np.std(y) > 0 else 0.0
            return link_row['link_id'], 0, 0, 1, fft_constant, float(r2)

        # Try full model fit for data with clear increasing trend
        try:
            capacity_hint = float(link_row.get('calibration_capacity', np.nan))
            if not np.isfinite(capacity_hint) or capacity_hint <= 0:
                capacity_hint = max(float(np.max(x)), 1.0)
            fft_hint = max(float(np.min(y)), 1e-6)
            cap_lower = max(1.0, capacity_hint * 0.1)
            cap_upper = max(capacity_hint * 10.0, float(np.max(x)) * 2.0, cap_lower * 10.0)
            popt, _ = curve_fit(
                model,
                x,
                y,
                p0=[0.15, 4.0, capacity_hint, fft_hint],
                bounds=([0, 0.8, cap_lower, 0], [np.inf, 5, cap_upper, np.inf]),
                maxfev=100000,
            )
            y_pred = model(x, *popt)
            r2 = r2_score(y, y_pred)
        except (RuntimeError, ValueError, TypeError, FloatingPointError):
            # If fit fails, use mean model
            fft_constant = np.mean(y)
            r2 = r2_score(y, np.full_like(y, fft_constant)) if np.std(y) > 0 else 0.0
            return link_row['link_id'], 0, 0, 1, fft_constant, float(r2)

        # If R² is too low, use mean model
        if r2 < r2_threshold and not strict_full:
            fft_constant = np.mean(y)
            r2_mean = r2_score(y, np.full_like(y, fft_constant)) if np.std(y) > 0 else 0.0
            return link_row['link_id'], 0, 0, 1, fft_constant, float(r2_mean)

        return link_row['link_id'], *popt, r2

    def _fit_historical(self, link_row, r2_threshold=0.5,
                        variation_ratio_threshold=0.03,
                        correlation_threshold=0.3):
        """Reproduce the committed pre-canonical BPR fitter numerics.

        This deliberately keeps the historical four-free-parameter model and
        its sentinel constant model (a=0, b=0, cap=1).  Diagnostics are
        computed by ``parallel_fit_and_evaluate``; this method returns the
        original six-value contract so existing callers remain compatible.
        """
        link_id = link_row['link_id']
        x = np.asarray(link_row['x_vector'], dtype=float)
        y = np.asarray(link_row['y_vector'], dtype=float)
        if np.all(x == 0):
            return link_id, np.nan, np.nan, np.nan, np.nan, np.nan

        # Keep the historical defaults for regression compatibility, but make
        # the two rejection gates configurable.  In particular, the old
        # y_std/x_std screen is sensitive to the simulator's flow units and
        # can reject curves that are still useful for exploratory fitting.
        screening = getattr(self, 'fit_screening', 'legacy')
        correlation_threshold = float(getattr(
            self, 'correlation_threshold', correlation_threshold
        ))
        variation_ratio_threshold = float(getattr(
            self, 'variation_ratio_threshold', variation_ratio_threshold
        ))
        r2_threshold = float(getattr(self, 'r2_threshold', r2_threshold))
        accept_low_r2 = bool(getattr(self, 'accept_low_r2', False))

        correlation = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0.0
        x_std = np.std(x)
        y_std = np.std(y)
        variation_ratio = y_std / x_std if x_std != 0 else float('inf')
        if screening == 'legacy' and (
                not np.isfinite(correlation)
                or correlation < correlation_threshold
                or variation_ratio < variation_ratio_threshold
        ):
            return link_id, 0, 0, 1, float(np.mean(y)), 1.0

        try:
            popt, _ = curve_fit(
                model,
                x,
                y,
                p0=[1, 1, 1, 1],
                bounds=([0, 0.8, 1, 0], [np.inf, 5, 1000, np.inf]),
                maxfev=100000,
            )
            y_pred = model(x, *popt)
            r2 = float(r2_score(y, y_pred))
        except (RuntimeError, ValueError, TypeError, FloatingPointError):
            return link_id, 0, 0, 1, float(np.mean(y)), 1.0

        # ``accept_low_r2`` retains any finite nonlinear solution for
        # diagnostics/downstream exploratory runs.  The honest R² remains in
        # the output, so accepting a fit never turns poor data into a good fit.
        if r2 < r2_threshold and not accept_low_r2:
            return link_id, 0, 0, 1, float(np.mean(y)), 1.0
        return link_id, *popt, r2

    def _fit_fixed_references(self, link_row):
        """Fit only ``a`` and ``b`` with canonical FFT and capacity fixed.

        FFT and capacity are measured network properties in the queue-based
        calibration.  Refitting them simultaneously with ``a`` and ``b`` is
        unnecessarily underidentified and caused many otherwise changing
        curves to fail.  Strict mode returns NaNs on failure; it never emits
        a constant or proxy model.
        """
        link_id = link_row['link_id']
        x = np.asarray(link_row['x_vector'], dtype=float)
        y = np.asarray(link_row['y_vector'], dtype=float)
        if x.size < 3 or y.size != x.size or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return link_id, np.nan, np.nan, np.nan, np.nan, np.nan
        capacity = float(link_row.get('calibration_capacity', np.nan))
        fft = float(link_row.get('calibration_fft', np.nan))
        if not np.isfinite(capacity) or capacity <= 0 or not np.isfinite(fft) or fft <= 0:
            return link_id, np.nan, np.nan, np.nan, np.nan, np.nan
        # A nearly horizontal simulated response is a valid free-flow curve,
        # not a reason to invent congestion parameters or reject the link.
        if np.ptp(y) <= 0.01 * max(abs(fft), np.finfo(float).eps):
            prediction = np.full_like(y, fft)
            r2 = float(r2_score(y, prediction)) if np.std(y) > 0 else 1.0
            return link_id, 0.0, 1.0, capacity, fft, r2

        def fixed_model(flow, a, b):
            return model(flow, a, b, capacity, fft)

        try:
            popt, _ = curve_fit(
                fixed_model,
                x,
                y,
                p0=[0.15, 4.0],
                bounds=([1e-8, 0.2], [100.0, 12.0]),
                maxfev=100000,
            )
            y_pred = fixed_model(x, *popt)
            r2 = float(r2_score(y, y_pred))
            return link_id, float(popt[0]), float(popt[1]), capacity, fft, r2
        except (RuntimeError, ValueError, TypeError, FloatingPointError):
            return link_id, np.nan, np.nan, capacity, fft, np.nan

    def parallel_fit_and_evaluate(self, workers=None, output_dir=None, save_plots=True,
                                  require_full_fit=False, r2_threshold=0.5,
                                  expected_link_ids=None, fixed_references=False,
                                  fit_mode=None, validation_mode=None,
                                  fit_screening='legacy',
                                  correlation_threshold=0.3,
                                  variation_ratio_threshold=0.03,
                                  accept_low_r2=False):
        """Fit all links, using processes when available and a safe fallback.

        Some managed/macOS environments prohibit process semaphores.  That
        should not turn a valid BPR table into an end-to-end failure, so the
        method falls back to deterministic in-process fitting and records the
        execution mode.  ``workers=1`` is an explicit bounded/CI mode.
        """
        self.require_full_fit = bool(require_full_fit)
        self.r2_threshold = float(r2_threshold)
        self.fixed_references = bool(fixed_references)
        self.fit_screening = str(fit_screening)
        self.correlation_threshold = float(correlation_threshold)
        self.variation_ratio_threshold = float(variation_ratio_threshold)
        self.accept_low_r2 = bool(accept_low_r2)
        if self.fit_screening not in {'legacy', 'none'}:
            raise ValueError("fit_screening must be legacy or none")
        if fit_mode is not None:
            self.bpr_mode = fit_mode
        elif fixed_references:
            self.bpr_mode = 'capacity_fraction_strict'
        else:
            self.bpr_mode = 'legacy_compatible'
        self.validation_mode = validation_mode or (
            'parameter_complete'
            if self.bpr_mode == 'historical_artifact_compatible'
            else ('full' if require_full_fit else 'parameter_complete')
        )
        self.df = self.df.sort_values('link_id', kind='mergesort').reset_index(drop=True)
        records = self.df.to_dict('records')
        worker_count = int(workers) if workers is not None else (os.cpu_count() or 1)
        if workers == 1:
            self.fit_metadata = {
                'requested_workers': workers,
                'workers_used': 1,
                'execution': 'serial',
                'fixed_references': self.fixed_references,
                'bpr_mode': self.bpr_mode,
                'validation_mode': self.validation_mode,
                'fit_screening': self.fit_screening,
                'correlation_threshold': self.correlation_threshold,
                'variation_ratio_threshold': self.variation_ratio_threshold,
                'accept_low_r2': self.accept_low_r2,
            }
            results = [self.fit_and_evaluate(record) for record in records]
        else:
            self.fit_metadata = {
                'requested_workers': workers,
                'workers_used': worker_count,
                'execution': 'process',
                'fixed_references': self.fixed_references,
                'bpr_mode': self.bpr_mode,
                'validation_mode': self.validation_mode,
                'fit_screening': self.fit_screening,
                'correlation_threshold': self.correlation_threshold,
                'variation_ratio_threshold': self.variation_ratio_threshold,
                'accept_low_r2': self.accept_low_r2,
            }
            try:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    results = list(executor.map(self.fit_and_evaluate, records))
            except (PermissionError, OSError) as exc:
                # ProcessPoolExecutor may be blocked by semaphore limits even
                # when ordinary multiprocessing pools are allowed.  Try the
                # latter before falling back to serial fitting.
                try:
                    with Pool(processes=worker_count) as pool:
                        results = pool.map(self.fit_and_evaluate, records)
                    self.fit_metadata.update({
                        'execution': 'multiprocessing_pool',
                        'fallback_reason': str(exc),
                        'fixed_references': self.fixed_references,
                        'bpr_mode': self.bpr_mode,
                        'validation_mode': self.validation_mode,
                        'fit_screening': self.fit_screening,
                        'correlation_threshold': self.correlation_threshold,
                        'variation_ratio_threshold': self.variation_ratio_threshold,
                        'accept_low_r2': self.accept_low_r2,
                    })
                except (PermissionError, OSError) as pool_exc:
                    self.fit_metadata = {
                        'requested_workers': workers,
                        'workers_used': 1,
                        'execution': 'serial_fallback',
                        'fallback_reason': f'{exc}; pool: {pool_exc}',
                        'fixed_references': self.fixed_references,
                        'bpr_mode': self.bpr_mode,
                        'validation_mode': self.validation_mode,
                        'fit_screening': self.fit_screening,
                        'correlation_threshold': self.correlation_threshold,
                        'variation_ratio_threshold': self.variation_ratio_threshold,
                        'accept_low_r2': self.accept_low_r2,
                    }
                    print(f'Process-based BPR fitting unavailable; using serial fallback: {pool_exc}')
                    results = [self.fit_and_evaluate(record) for record in records]

        # Update DataFrame with fitted parameters and R^2 values
        if 'honest_R2' not in self.df.columns:
            self.df['honest_R2'] = np.nan
        if 'fallback_reason' not in self.df.columns:
            self.df['fallback_reason'] = ''
        else:
            self.df['fallback_reason'] = self.df['fallback_reason'].fillna('').astype(object)
        if 'observation_source' not in self.df.columns:
            source_series = self.df.get(
                'fit_status',
                pd.Series('simulated_contextual', index=self.df.index),
            )
            self.df['observation_source'] = source_series.map(
                lambda value: 'proxy' if value == 'proxy' else 'simulated_contextual'
            )
        for link_id, a_fit, b_fit, cap_fit, fft_fit, r2 in results:
            self.df.loc[self.df['link_id'] == link_id, ['a_fit', 'b_fit', 'cap_fit', 'fft_fit', 'R^2']] = a_fit, b_fit, cap_fit, fft_fit, r2
            row_mask = self.df['link_id'] == link_id
            source_status = self.df.loc[row_mask, 'fit_status'].iloc[0] if 'fit_status' in self.df.columns else None
            observation_source = self.df.loc[row_mask, 'observation_source'].iloc[0]
            is_constant = (np.isfinite(a_fit) and np.isfinite(b_fit)
                            and float(a_fit) == 0.0 and float(b_fit) == 0.0
                            and float(cap_fit) == 1.0)
            is_validated_flat = (
                self.fixed_references and np.isfinite(a_fit) and np.isfinite(b_fit)
                and float(a_fit) == 0.0 and float(b_fit) == 1.0
                and np.isfinite(cap_fit) and float(cap_fit) > 0
                and np.isfinite(fft_fit) and float(fft_fit) > 0
            )
            if self.bpr_mode == 'historical_artifact_compatible':
                if source_status == 'proxy' or observation_source == 'proxy':
                    status = 'proxy'
                    reason = self.df.loc[row_mask, 'fallback_reason'].iloc[0] if 'fallback_reason' in self.df.columns else ''
                elif is_constant:
                    status = 'constant_fallback'
                    reason = 'historical_screen_or_fit_quality'
                elif np.isfinite(a_fit) and np.isfinite(b_fit) and np.isfinite(cap_fit) and np.isfinite(fft_fit) and np.isfinite(r2):
                    if self.bpr_mode == 'historical_artifact_compatible' and r2 < self.r2_threshold:
                        status = 'full_relaxed'
                        reason = 'accepted_below_quality_threshold'
                    else:
                        status = 'full'
                        reason = ''
                else:
                    status = 'failed'
                    reason = 'non_finite_fit_parameters'
            else:
                status = 'validated_free_flow' if is_validated_flat else (
                    'full'
                    if np.isfinite(a_fit) and np.isfinite(b_fit) and np.isfinite(cap_fit)
                    and np.isfinite(fft_fit) and a_fit > 0 and b_fit > 0
                    and np.isfinite(r2) and r2 >= self.r2_threshold
                    else ('failed_quality' if np.isfinite(a_fit) and a_fit > 0 else 'constant')
                )
                reason = ''
            self.df.loc[self.df['link_id'] == link_id, 'fit_status'] = status
            self.df.loc[row_mask, 'fallback_reason'] = reason
            try:
                values = np.asarray(self.df.loc[row_mask, 'y_vector'].iloc[0], dtype=float)
                if is_constant:
                    prediction = np.full_like(values, float(fft_fit))
                    honest = float(r2_score(values, prediction)) if np.std(values) > 0 else 1.0
                elif np.isfinite(a_fit) and np.isfinite(b_fit) and np.isfinite(cap_fit) and np.isfinite(fft_fit):
                    honest = float(r2_score(values, model(
                        np.asarray(self.df.loc[row_mask, 'x_vector'].iloc[0], dtype=float),
                        a_fit, b_fit, cap_fit, fft_fit)))
                else:
                    honest = np.nan
                self.df.loc[row_mask, 'honest_R2'] = honest
            except (ValueError, TypeError, FloatingPointError):
                self.df.loc[row_mask, 'honest_R2'] = np.nan
        target_dir = output_dir or '.'
        os.makedirs(target_dir, exist_ok=True)
        self.save_results_to_csv(os.path.join(target_dir, 'fitter_results.csv'))
        if save_plots:
            self.plot_fitted_links(save_dir=os.path.join(target_dir, 'model_fit_plots'))
        return validate_bpr_fit_table(
            self.df,
            expected_link_ids=expected_link_ids,
            require_full_fit=require_full_fit,
            validation_mode=self.validation_mode,
        )

    def save_results_to_csv(self, filename="fitter_results.csv"):
        """Saves the DataFrame to a CSV file."""
        # Serialize arrays as valid JSON.  ``DataFrame.to_csv`` on a NumPy
        # array produces whitespace-separated values (without commas), which
        # the old loader cannot parse back reliably.
        df_copy = self.df.copy()
        for column in ('x_vector', 'y_vector'):
            if column in df_copy.columns:
                df_copy[column] = df_copy[column].apply(
                    lambda value: json.dumps(np.asarray(value, dtype=float).tolist())
                    if isinstance(value, (list, tuple, np.ndarray)) else value
                )
        df_copy.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def plot_fitted_links(self, save_dir="model_fit_plots"):
        """Plots and saves figures for all links with detailed fit information."""
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for _, row in self.df.iterrows():
            # Skip links with all zeros in x_vector
            if np.all(row['x_vector'] == 0):
                continue

            plt.figure(figsize=(10, 6), dpi=300)
            
            # Calculate metrics
            y_mean = np.mean(row['y_vector'])
            y_std = np.std(row['y_vector'])
            if getattr(self, 'bpr_mode', None) == 'historical_artifact_compatible':
                x_std = np.std(row['x_vector'])
                variation_ratio = y_std / x_std if x_std != 0 else float('inf')
            else:
                variation_ratio = y_std / max(abs(float(y_mean)), np.finfo(float).eps)
            correlation = (
                np.corrcoef(row['x_vector'], row['y_vector'])[0, 1]
                if np.std(row['x_vector']) > 0 and np.std(row['y_vector']) > 0
                else 0.0
            )
            if not np.isfinite(correlation):
                correlation = 0.0

            # Determine if mean model was used
            is_mean_model = (row['a_fit'] == 0 and row['b_fit'] == 0 and row['cap_fit'] == 1)
            fit_status = row.get('fit_status', 'unknown')
            observation_source = row.get('observation_source', 'unknown')

            # Scatter plot of original data
            plt.scatter(row['x_vector'], row['y_vector'],
                      label='Original Data', s=15, alpha=0.7)

            # Sort x values for smooth curve plotting
            x_sorted = np.sort(row['x_vector'])
            xs = np.linspace(min(x_sorted), max(x_sorted), 1000)
            
            # Plot fitted curve if parameters are available
            if not np.isnan(row['a_fit']):
                if is_mean_model:
                    plt.axhline(y=row['fft_fit'], color='red', linestyle='-', 
                              label='Mean Model (Selected)', linewidth=2)
                else:
                    ys = model(xs, row['a_fit'], row['b_fit'], row['cap_fit'], row['fft_fit'])
                    plt.plot(xs, ys, linewidth=2, color='red', 
                            label='Full Model (Selected)')
                    # Show mean as reference
                    plt.axhline(y=y_mean, color='green', linestyle='--', 
                              label='Mean Reference', alpha=0.5)

            # Add detailed information to the plot
            info_text = f"Link {row['link_id']}\n"
            info_text += f"Model: {'Mean' if is_mean_model else 'Full'}\n"
            info_text += f"Fit status: {fit_status}\n"
            info_text += f"Observation source: {observation_source}\n"
            info_text += f"Correlation: {correlation:.3f}\n"
            info_text += f"R² = {row['R^2']:.3f}\n"
            if 'honest_R2' in row and np.isfinite(row['honest_R2']):
                info_text += f"Honest R² = {row['honest_R2']:.3f}\n"
            info_text += f"y_mean = {y_mean:.2f}\n"
            info_text += f"y_std = {y_std:.2f}\n"
            info_text += f"Variation ratio = {variation_ratio:.3f}\n"
            if not is_mean_model:
                info_text += f"Fit params:\n"
                info_text += f"a = {row['a_fit']:.3f}\n"
                info_text += f"b = {row['b_fit']:.3f}\n"
                info_text += f"cap = {row['cap_fit']:.3f}\n"
            info_text += f"fft = {row['fft_fit']:.3f}"

            # Position the text box in the top left
            plt.text(0.02, 0.98, info_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.title(f'Link {row["link_id"]} - {"Mean" if is_mean_model else "Full"} Model Fit ({fit_status}; {observation_source})',
                     fontsize=16)
            plt.xlabel('Traffic Flow', fontsize=12)
            plt.ylabel(
                'Delay' if getattr(self, 'bpr_mode', None) == 'historical_artifact_compatible'
                else 'Travel time (s)',
                fontsize=12,
            )
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=10, loc='lower right')

            # Save the figure
            plt.savefig(os.path.join(save_dir, f'link_{row["link_id"]}_fit.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()  # Close the figure to free memory

        print(f"Plots saved in {save_dir}/")

    def fill_missing_link_ids(self):
        """Fill missing link_ids with default entries and report the number added."""
        existing_ids = set(self.df['link_id'])
        max_id = int(self.df['link_id'].max())
        all_ids = set(range(max_id + 1))
        missing_ids = sorted(list(all_ids - existing_ids))

        default_rows = []
        for link_id in missing_ids:
            default_rows.append({
                'link_id': link_id,
                'link_length': 1.0,
                'free_flow_speed': 1.0,
                'capacity': 1.0,
                'a_true': 0.0,
                'b_true': 0.0,
                'noise_scale': 0.0,
                'x_vector': np.zeros(self.n_samples),
                'y_vector': np.zeros(self.n_samples),
                'a_fit': 0.0,
                'b_fit': 0.0,
                'cap_fit': 1.0,
                'fft_fit': 1.0,
                'R^2': np.nan,
                'fit_status': 'degraded_missing'
            })

        if default_rows:
            default_df = pd.DataFrame(default_rows)
            self.df = pd.concat([self.df, default_df], ignore_index=True).sort_values('link_id').reset_index(drop=True)

        print(f"Filled {len(missing_ids)} missing link_id(s).")

def convert_string_to_array(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    def _convert(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.asarray(val, dtype=float)
        try:
            return np.array(ast.literal_eval(val))
        except (ValueError, SyntaxError):
            # Accept NumPy's legacy whitespace-separated representation too,
            # so historical fitter CSVs remain importable.
            if isinstance(val, str):
                stripped = val.strip().strip('[]')
                try:
                    parsed = np.fromstring(stripped, sep=',')
                    if parsed.size > 1:
                        return parsed
                    parsed = np.fromstring(stripped, sep=' ')
                    if parsed.size > 1:
                        return parsed
                except ValueError:
                    pass
            return val
    df[column_name] = df[column_name].apply(_convert)


def validate_bpr_fit_table(df, expected_link_ids=None, require_full_fit=False,
                           validation_mode=None):
    """Validate BPR coverage and fitted parameter completeness.

    ``full`` rejects historical sentinel constant/proxy rows.  The
    ``parameter_complete`` contract accepts those rows only when all four
    downstream parameters are finite and the canonical link coverage is
    exact.
    """
    validation_mode = validation_mode or ('full' if require_full_fit else 'parameter_complete')
    if validation_mode not in {'full', 'parameter_complete'}:
        raise ValueError("validation_mode must be full or parameter_complete")
    expected = set(int(value) for value in expected_link_ids) if expected_link_ids is not None else None
    actual = set(int(value) for value in df['link_id'])
    errors = []
    if df['link_id'].duplicated().any():
        errors.append('duplicate link_ids')
    if expected is not None:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f'missing link_ids={missing[:10]}')
        if extra:
            errors.append(f'unexpected link_ids={extra[:10]}')
    numeric = {
        name: pd.to_numeric(
            df[name] if name in df.columns
            else pd.Series(np.nan, index=df.index),
            errors='coerce',
        )
        for name in ('a_fit', 'b_fit', 'cap_fit', 'fft_fit')
    }
    invalid_mask = (
        ~np.isfinite(numeric['a_fit'])
        | ~np.isfinite(numeric['b_fit'])
        | ~np.isfinite(numeric['cap_fit'])
        | ~np.isfinite(numeric['fft_fit'])
        | (numeric['cap_fit'] <= 0)
        | (numeric['fft_fit'] < 0)
    )
    statuses = (
        df['fit_status']
        if 'fit_status' in df.columns
        else pd.Series('unknown', index=df.index, dtype=object)
    )
    validated_flat = statuses == 'validated_free_flow'
    if validation_mode == 'full':
        invalid_mask = invalid_mask | (
            ((numeric['a_fit'] <= 0) | (numeric['b_fit'] <= 0)) & ~validated_flat
        )
    invalid = df[invalid_mask]
    accepted_statuses = {'full', 'validated', 'validated_free_flow'}
    non_full = df[~statuses.isin(accepted_statuses)]
    if not invalid.empty:
        errors.append(
            f'invalid fitted parameters for link_ids={invalid.link_id.astype(int).tolist()[:10]}'
        )
    if require_full_fit or validation_mode == 'full':
        if not non_full.empty:
            errors.append(f'non-full fit statuses={non_full.link_id.astype(int).tolist()[:10]}')
    if errors and (require_full_fit or validation_mode in {'full', 'parameter_complete'}):
        label = 'Strict' if validation_mode == 'full' or require_full_fit else 'BPR'
        raise ValueError(f'{label} BPR fit validation failed: ' + '; '.join(errors))
    return {
        'link_count': int(len(df)),
        'full_fit_count': int(statuses.isin(accepted_statuses).sum()),
        'validated_fit_count': int((statuses == 'validated').sum()),
        'validated_free_flow_count': int((statuses == 'validated_free_flow').sum()),
        'full_relaxed_count': int((statuses == 'full_relaxed').sum()),
        'non_full_fit_count': int((~statuses.isin(accepted_statuses)).sum()),
        'constant_fallback_count': int((statuses == 'constant_fallback').sum()),
        'proxy_count': int((statuses == 'proxy').sum()),
        'validation_mode': validation_mode,
        'errors': errors,
    }
