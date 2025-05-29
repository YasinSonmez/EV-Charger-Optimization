import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor
import ast
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

        # Check if `x_vector` is all zeros
        if np.all(link_row['x_vector'] == 0):
            return link_row['link_id'], np.nan, np.nan, np.nan, np.nan, np.nan

        x, y = link_row['x_vector'], link_row['y_vector']
        
        # Calculate correlation coefficient to check for increasing trend
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Calculate variation metrics
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std == 0:  # Avoid division by zero
            variation_ratio = float('inf')
        else:
            variation_ratio = y_std / x_std

        # Use mean model if:
        # 1. Data shows weak positive correlation (not clearly increasing)
        # 2. Or if variation ratio is too low (y doesn't vary much with x)
        if correlation < correlation_threshold or variation_ratio < variation_ratio_threshold:
            fft_constant = np.mean(y)
            return link_row['link_id'], 0, 0, 1, fft_constant, 1.0  # Perfect R² for mean model

        # Try full model fit for data with clear increasing trend
        try:
            popt, _ = curve_fit(model, x, y, p0=[1, 1, 1, 1], bounds=([0, 0.8, 1, 0], [np.inf, 5, 1000, np.inf]), maxfev=100000)
            y_pred = model(x, *popt)
            r2 = r2_score(y, y_pred)
        except RuntimeError:
            # If fit fails, use mean model
            fft_constant = np.mean(y)
            return link_row['link_id'], 0, 0, 1, fft_constant, 1.0

        # If R² is too low, use mean model
        if r2 < r2_threshold:
            fft_constant = np.mean(y)
            return link_row['link_id'], 0, 0, 1, fft_constant, 1.0

        return link_row['link_id'], *popt, r2

    def parallel_fit_and_evaluate(self):
        """Performs parallel fitting and evaluation for all links."""
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.fit_and_evaluate, self.df.to_dict('records')))

        # Update DataFrame with fitted parameters and R^2 values
        for link_id, a_fit, b_fit, cap_fit, fft_fit, r2 in results:
            self.df.loc[self.df['link_id'] == link_id, ['a_fit', 'b_fit', 'cap_fit', 'fft_fit', 'R^2']] = a_fit, b_fit, cap_fit, fft_fit, r2
        self.save_results_to_csv()
        self.plot_fitted_links()

    def save_results_to_csv(self, filename="fitter_results.csv"):
        """Saves the DataFrame to a CSV file."""
        # Convert numpy arrays to string representation for CSV compatibility
        df_copy = self.df.copy()
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
            x_std = np.std(row['x_vector'])
            variation_ratio = y_std / x_std if x_std != 0 else float('inf')
            correlation = np.corrcoef(row['x_vector'], row['y_vector'])[0, 1]

            # Determine if mean model was used
            is_mean_model = (row['a_fit'] == 0 and row['b_fit'] == 0 and row['cap_fit'] == 1)

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
            info_text += f"Correlation: {correlation:.3f}\n"
            info_text += f"R² = {row['R^2']:.3f}\n"
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

            plt.title(f'Link {row["link_id"]} - {"Mean" if is_mean_model else "Full"} Model Fit', 
                     fontsize=16)
            plt.xlabel('Traffic Flow', fontsize=12)
            plt.ylabel('Delay', fontsize=12)
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
                'R^2': 1.0
            })

        if default_rows:
            default_df = pd.DataFrame(default_rows)
            self.df = pd.concat([self.df, default_df], ignore_index=True).sort_values('link_id').reset_index(drop=True)

        print(f"Filled {len(missing_ids)} missing link_id(s).")

def convert_string_to_array(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    for index, row in df.iterrows():
        try:
            array_list = ast.literal_eval(row[column_name])
            df.at[index, column_name] = np.array(array_list)
        except ValueError as e:
            print(f"Error converting row {index}: {e}")
            continue 