import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="MMM Budget Optimizer")

# --- START: MMMBudgetOptimizer CLASS ---
# (Pasted from your script, with modifications to plotting functions)

class MMMBudgetOptimizer:
    """
    Marketing Mix Modeling Budget Optimizer
    Optimizes budget allocation across brands and channels using regression-based MMM
    
    MODIFIED FOR STREAMLIT:
    - Plotting functions return matplotlib 'fig' objects instead of saving/showing.
    - export_results returns the dataframe.
    """
    
    def __init__(self, data_path: str):
        """Initialize with path to CSV data"""
        try:
            self.data = pd.read_csv(data_path)
        except FileNotFoundError:
            st.error(f"ERROR: Data file not found at {data_path}")
            st.stop()
        
        # Brand and channel configuration
        self.brand_channels = {
            'bb': ['m1_meta_bb_sp', 'm1_prog_bb_sp', 'm1_search_bb_sp', 
                   'm1_social_bb_sp', 'm3_ooh_bb_sp', 'm3_partnership_bb_sp', 
                   'm3_vod_bb_sp'],
            'mac': ['m1_meta_mac_sp', 'm1_prog_mac_sp', 'm1_search_mac_sp', 
                    'm1_social_mac_sp', 'm3_ooh_mac_sp', 'm3_tv_mac_sp', 
                    'm3_vod_mac_sp'],
            'tf': ['m1_meta_tf_sp', 'm1_prog_tf_sp', 'm1_search_tf_sp', 
                   'm1_social_tf_sp', 'm3_ooh_tf_sp']
        }
        
        self.brand_kpis = {
            'bb': 'm0_kpi_bb',
            'mac': 'm0_kpi_mac',
            'tf': 'm0_kpi_tf'
        }
        
        self.models = {}
        self.optimization_results = {}
        
    def apply_adstock(self, x: np.ndarray, rate: float = 0.5) -> np.ndarray:
        """Apply adstock transformation (geometric decay)"""
        adstocked = np.zeros_like(x, dtype=float)
        if len(x) > 0:
            adstocked[0] = x[0]
            for i in range(1, len(x)):
                adstocked[i] = x[i] + rate * adstocked[i-1]
        return adstocked
    
    def apply_saturation(self, x: np.ndarray, alpha: float = 1.0, 
                         gamma: float = 50000) -> np.ndarray:
        """Apply saturation transformation (Hill function)"""
        x_np = np.asarray(x)
        return np.power(x_np, alpha) / (np.power(gamma, alpha) + np.power(x_np, alpha))
    
    def build_mmm_models(self, adstock_rate: float = 0.4, 
                         saturation_alpha: float = 1.0,
                         saturation_gamma: float = 50000):
        """Build MMM regression models for each brand"""
        
        for brand in self.brand_channels.keys():
            kpi_col = self.brand_kpis[brand]
            y = self.data[kpi_col].values
            
            X_transformed = []
            feature_names = []
            
            for channel in self.brand_channels[brand]:
                raw_spend = self.data[channel].fillna(0).values
                adstocked = self.apply_adstock(raw_spend, rate=adstock_rate)
                saturated = self.apply_saturation(adstocked, 
                                                  alpha=saturation_alpha, 
                                                  gamma=saturation_gamma)
                X_transformed.append(saturated)
                feature_names.append(channel)
            
            X_transformed = np.column_stack(X_transformed)
            X_with_intercept = np.column_stack([np.ones(len(X_transformed)), X_transformed])
            
            try:
                XtX = X_with_intercept.T @ X_with_intercept
                XtY = X_with_intercept.T @ y
                ridge_lambda = 0.01
                XtX_reg = XtX + ridge_lambda * np.eye(XtX.shape[0])
                beta = np.linalg.solve(XtX_reg, XtY)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            y_pred = X_with_intercept @ beta
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
            
            self.models[brand] = {
                'intercept': beta[0],
                'coefficients': dict(zip(feature_names, beta[1:])),
                'X_transformed': X_transformed,
                'y_actual': y,
                'y_pred': y_pred,
                'r2': r2,
                'mape': mape,
                'adstock_rate': adstock_rate,
                'saturation_alpha': saturation_alpha,
                'saturation_gamma': saturation_gamma
            }
    
    def predict_kpi(self, brand: str, weekly_spend_dict: Dict[str, float]) -> float:
        """Predict KPI for given weekly spend allocation"""
        model = self.models[brand]
        prediction = model['intercept']
        
        for channel, spend in weekly_spend_dict.items():
            if channel in model['coefficients']:
                adstock_mult = 1 / (1 - model['adstock_rate'])
                adstocked = spend * adstock_mult
                saturated = self.apply_saturation(
                    np.array([adstocked]), 
                    alpha=model['saturation_alpha'],
                    gamma=model['saturation_gamma']
                )[0]
                prediction += model['coefficients'][channel] * saturated
        return max(0, prediction)
    
    def optimize_budget(self, total_budget: float = 5_000_000, 
                        weeks: int = 52,
                        min_brand_share: float = 0.15,
                        max_brand_share: float = 0.60,
                        channel_bound_pct: float = 0.30):
        """Optimize budget allocation across brands and channels"""
        
        historical = {}
        for brand in self.brand_channels.keys():
            channels = self.brand_channels[brand]
            total_spend = sum(self.data[ch].sum() for ch in channels)
            total_kpi = self.data[self.brand_kpis[brand]].sum()
            historical[brand] = {
                'spend': total_spend, 'kpi': total_kpi,
                'efficiency': total_kpi / (total_spend + 1) if total_spend > 0 else 0
            }
        
        brands = list(self.brand_channels.keys())
        n_brands = len(brands)
        total_weekly_budget = total_budget / weeks
        
        historical_proportions = {}
        for brand in brands:
            channels = self.brand_channels[brand]
            channel_spends = {ch: self.data[ch].sum() for ch in channels}
            total = sum(channel_spends.values())
            if total > 0:
                historical_proportions[brand] = {ch: spend/total for ch, spend in channel_spends.items()}
            else:
                historical_proportions[brand] = {ch: 1/len(channels) for ch in channels}
        
        all_channels = []
        brand_channel_indices = {}
        current_index = 0
        for brand, channels_list in self.brand_channels.items():
            indices = []
            for channel in channels_list:
                all_channels.append(channel)
                indices.append(current_index)
                current_index += 1
            brand_channel_indices[brand] = indices
        n_channels = len(all_channels)
        
        bounds = []
        for channel in all_channels:
            series = self.data[channel].fillna(0)
            hist_max = series.max()
            non_zero_min_series = series[series > 0]
            hist_min = non_zero_min_series.min() if not non_zero_min_series.empty else 0
            
            lower_bound = max(0, hist_min * (1.0 - channel_bound_pct))
            upper_bound = hist_max * (1.0 + channel_bound_pct)
            
            if hist_max == 0: upper_bound = 0
            if lower_bound > upper_bound: lower_bound = upper_bound
            
            bounds.append((lower_bound, upper_bound))

        def objective(weekly_channel_spends):
            total_kpi = 0
            for brand in brands:
                weekly_spend_dict = {}
                indices = brand_channel_indices[brand]
                channels_list = self.brand_channels[brand]
                for i, channel_name in enumerate(channels_list):
                    channel_index = indices[i]
                    weekly_spend_dict[channel_name] = weekly_channel_spends[channel_index]
                weekly_kpi = self.predict_kpi(brand, weekly_spend_dict)
                total_kpi += weekly_kpi * weeks
            return -total_kpi
        
        constraints = []
        constraints.append({'type': 'eq', 'fun': lambda x: total_weekly_budget - np.sum(x)})
        
        for brand in brands:
            indices = brand_channel_indices[brand]
            min_brand_weekly_budget = min_brand_share * total_weekly_budget
            max_brand_weekly_budget = max_brand_share * total_weekly_budget
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=indices: np.sum(x[idx]) - min_brand_weekly_budget})
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=indices: max_brand_weekly_budget - np.sum(x[idx])})

        x0 = np.zeros(n_channels)
        initial_brand_shares = np.ones(n_brands) / n_brands
        current_x0_index = 0
        for i, brand in enumerate(brands):
            brand_weekly_budget = total_weekly_budget * initial_brand_shares[i]
            props = historical_proportions[brand]
            total_prop = sum(props.values())
            if total_prop == 0: total_prop = 1
            for channel in self.brand_channels[brand]:
                prop = props.get(channel, 0) / total_prop
                x0[current_x0_index] = brand_weekly_budget * prop
                current_x0_index += 1
        
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        current_sum = x0.sum()
        if current_sum > 0:
            x0 = x0 * (total_weekly_budget / current_sum)
        else:
            x0 = np.full(n_channels, total_weekly_budget / n_channels)
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            current_sum = x0.sum()
            if current_sum > 0: x0 = x0 * (total_weekly_budget / current_sum)

        result = minimize(
            objective, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            st.warning(f"Optimization may not have converged: {result.message}")
        
        optimal_weekly_spends = result.x
        optimal_total_kpi = -result.fun
        optimal_allocation = {}
        optimal_shares_dict = {}
        current_index = 0
        
        for brand in brands:
            channel_allocation = {}
            brand_total_weekly_spend = 0
            for channel in self.brand_channels[brand]:
                weekly_spend = optimal_weekly_spends[current_index]
                annual_spend = weekly_spend * weeks
                channel_allocation[channel] = annual_spend
                brand_total_weekly_spend += weekly_spend
                current_index += 1
            brand_annual_budget = brand_total_weekly_spend * weeks
            brand_share = brand_annual_budget / total_budget if total_budget > 0 else 0
            optimal_allocation[brand] = {
                'total_budget': brand_annual_budget,
                'share': brand_share,
                'channels': channel_allocation
            }
            optimal_shares_dict[brand] = brand_share
        
        historical_total_kpi = sum(h['kpi'] for h in historical.values())
        improvement_pct = ((optimal_total_kpi - historical_total_kpi) / 
                           (historical_total_kpi + 1e-10) * 100)
        
        self.optimization_results = {
            'total_budget': total_budget,
            'optimal_allocation': optimal_allocation,
            'optimal_total_kpi': optimal_total_kpi,
            'historical': historical,
            'historical_total_kpi': historical_total_kpi,
            'improvement_pct': improvement_pct,
            'optimal_shares': optimal_shares_dict
        }
        return self.optimization_results
    
    def _print_optimization_results(self):
        # This function is now replaced by Streamlit UI elements
        pass
    
    def plot_model_performance(self, figsize=(15, 10)):
        """Plot actual vs predicted KPI for all brands"""
        brands = list(self.models.keys())
        n_brands = len(brands)
        
        if n_brands == 0:
            st.warning("No models found.")
            return None

        fig, axes = plt.subplots(n_brands, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, brand in enumerate(brands):
            model = self.models[brand]
            weeks = range(1, len(model['y_actual']) + 1)
            axes[i].plot(weeks, model['y_actual'], label='Actual', linewidth=2, marker='o', markersize=4, alpha=0.7)
            axes[i].plot(weeks, model['y_pred'], label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.7)
            axes[i].set_title(f'Brand {brand.upper()} - KPI Performance (R²={model["r2"]:.4f})', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Week')
            axes[i].set_ylabel('KPI')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig # <-- MODIFIED
    
    def plot_budget_allocation(self, figsize=(15, 8)):
        """Plot optimal vs historical budget allocation"""
        if not self.optimization_results:
            st.warning("Please run optimize_budget() first!")
            return None
        
        results = self.optimization_results
        brands = list(results['optimal_allocation'].keys())
        
        brand_labels = [b.upper() for b in brands]
        historical_budgets = [results['historical'][b]['spend']/1000 for b in brands]
        optimal_budgets = [results['optimal_allocation'][b]['total_budget']/1000 for b in brands]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        x = np.arange(len(brands))
        width = 0.35
        ax1.bar(x - width/2, historical_budgets, width, label='Historical', alpha=0.8)
        ax1.bar(x + width/2, optimal_budgets, width, label='Optimal', alpha=0.8)
        
        ax1.set_xlabel('Brand')
        ax1.set_ylabel('Budget (€K)')
        ax1.set_title('Budget Allocation Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(brand_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, (h, o) in enumerate(zip(historical_budgets, optimal_budgets)):
            ax1.text(i - width/2, h, f'€{h:.0f}K', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, o, f'€{o:.0f}K', ha='center', va='bottom', fontsize=9)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(brands)))
        if sum(optimal_budgets) > 0:
            wedges, texts, autotexts = ax2.pie(optimal_budgets, labels=brand_labels, autopct='%1.1f%%',
                                               colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, "No Budget Allocated", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        
        ax2.set_title('Optimal Budget Distribution', fontweight='bold')
        
        plt.tight_layout()
        return fig # <-- MODIFIED
    
    def plot_channel_allocation(self, brand: str, figsize=(12, 6)):
        """Plot channel-level budget allocation for a specific brand"""
        if not self.optimization_results:
            st.warning("Please run optimize_budget() first!")
            return None
        
        results = self.optimization_results
        if brand not in results['optimal_allocation']:
            st.warning(f"Brand '{brand}' not found in results.")
            return None
            
        allocation = results['optimal_allocation'][brand]
        channels = allocation['channels']
        sorted_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_channels:
            st.info(f"No channels to plot for brand {brand.upper()}")
            return None
        
        channel_names = [ch.replace(f'_{brand}_sp', '').replace('m1_', '').replace('m3_', '') for ch, _ in sorted_channels]
        budgets = [budget/1000 for _, budget in sorted_channels]
        
        fig, ax = plt.subplots(figsize=figsize)
        bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(channel_names)))
        bars = ax.barh(channel_names, budgets, color=bar_colors)
        
        ax.set_xlabel('Budget (€K)', fontweight='bold')
        ax.set_title(f'Brand {brand.upper()} - Channel Budget Allocation (Total: €{allocation["total_budget"]/1000:.0f}K)', 
                       fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        for i, (bar, budget) in enumerate(zip(bars, budgets)):
            width = bar.get_width()
            total_brand_budget = allocation['total_budget']
            pct = (sorted_channels[i][1] / total_brand_budget * 100) if total_brand_budget > 0 else 0
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f' €{budget:.0f}K ({pct:.1f}%)', 
                    ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        return fig # <-- MODIFIED
    
    def export_results(self, filename: str = 'mmm_optimization_results.csv'):
        """Export optimization results to CSV"""
        if not self.optimization_results:
            st.warning("Please run optimize_budget() first!")
            return pd.DataFrame()
        
        results = self.optimization_results
        export_data = []
        
        for brand, allocation in results['optimal_allocation'].items():
            hist = results['historical'][brand]
            export_data.append({
                'Brand': brand.upper(), 'Channel': 'TOTAL',
                'Historical_Spend': hist['spend'],
                'Optimal_Budget': allocation['total_budget'],
                'Change_Amount': allocation['total_budget'] - hist['spend'],
                'Change_Percent': ((allocation['total_budget'] - hist['spend']) / (hist['spend'] + 1) * 100),
                'Budget_Share': allocation['share'] * 100
            })
            for channel, budget in allocation['channels'].items():
                ch_name = channel.replace(f'_{brand}_sp', '').replace('m1_', '').replace('m3_', '')
                hist_channel = self.data[channel].sum()
                export_data.append({
                    'Brand': brand.upper(), 'Channel': ch_name,
                    'Historical_Spend': hist_channel,
                    'Optimal_Budget': budget,
                    'Change_Amount': budget - hist_channel,
                    'Change_Percent': ((budget - hist_channel) / (hist_channel + 1) * 100),
                    'Budget_Share': (budget / (allocation['total_budget'] + 1e-10)) * 100
                })
        
        df_export = pd.DataFrame(export_data)
        # We don't save to file, just return the df
        return df_export # <-- MODIFIED

# --- END: MMMBudgetOptimizer CLASS ---


# --- START: STREAMLIT APP LOGIC ---

# Title for the app
st.title("📈 Marketing Mix Model (MMM) Budget Optimizer")

# --- Sidebar for User Inputs ---
st.sidebar.header("1. Optimization Parameters")

total_budget_input = st.sidebar.number_input(
    "Total Annual Budget (€)", 
    min_value=1_000_000, 
    value=5_000_000, 
    step=250_000,
    format="%d"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Brand Constraints")

min_brand_share_input = st.sidebar.slider(
    "Minimum Brand Share (%)", 
    min_value=0.0, 
    max_value=40.0, 
    value=15.0, 
    step=1.0,
    format="%.0f%%"
) / 100.0  # Convert to decimal

max_brand_share_input = st.sidebar.slider(
    "Maximum Brand Share (%)", 
    min_value=40.0, 
    max_value=100.0, 
    value=60.0, 
    step=1.0,
    format="%.0f%%"
) / 100.0  # Convert to decimal

# Validate brand shares
if min_brand_share_input > max_brand_share_input:
    st.sidebar.error("Min Brand Share cannot be greater than Max Brand Share.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Channel Constraints")

channel_bound_pct_input = st.sidebar.slider(
    "Channel Spend Bounds (+/- % from Historical)", 
    min_value=0.0, 
    max_value=100.0, 
    value=30.0, 
    step=5.0,
    help="Sets the min/max weekly spend for each channel as a percentage of its historical min/max. E.g., 30% allows spend to be 30% higher than historical max and 30% lower than historical min."
) / 100.0 # Convert to decimal


# --- Caching the Model Building ---
# @st.cache_resource runs once and caches the returned object (the optimizer)
@st.cache_resource
def load_and_build_models(data_path):
    with st.spinner("Loading data and building base MMMs... (This happens only once)"):
        optimizer = MMMBudgetOptimizer(data_path)
        optimizer.build_mmm_models(
            adstock_rate=0.4,
            saturation_alpha=1.0,
            saturation_gamma=50000
        )
        return optimizer

# Load the data and build models
optimizer = load_and_build_models('final_kpi_weekly_reduced.csv')

st.header("Base Model Performance (R² Scores)")
st.write("These are the R² scores for the base models built on historical data.")

cols = st.columns(len(optimizer.models))
for i, (brand, model) in enumerate(optimizer.models.items()):
    with cols[i]:
        st.metric(label=f"Brand {brand.upper()} R²", value=f"{model['r2']:.3f}")


# --- Run Optimization ---
st.sidebar.markdown("---")
st.sidebar.header("2. Run Optimization")
run_button = st.sidebar.button("🚀 Run Budget Optimization", type="primary", use_container_width=True)

# Main content area
st.markdown("---")

if not run_button:
    st.info("Adjust the parameters in the sidebar and click 'Run Budget Optimization' to see the results.")
    st.stop()

# --- Display Results (if button is clicked) ---
with st.spinner(f"Optimizing a €{total_budget_input:,.0f} budget..."):
    results = optimizer.optimize_budget(
        total_budget=total_budget_input,
        weeks=52,
        min_brand_share=min_brand_share_input,
        max_brand_share=max_brand_share_input,
        channel_bound_pct=channel_bound_pct_input
    )

st.header("📊 Optimization Results")

# --- Key Metrics ---
st.subheader("High-Level Summary")
col1, col2, col3 = st.columns(3)
col1.metric(
    "Optimal Annual KPI", 
    f"{results['optimal_total_kpi']:,.0f}",
    f"{results['improvement_pct']:.2f}% vs Historical"
)
col2.metric(
    "Historical Annual KPI", 
    f"{results['historical_total_kpi']:,.0f}"
)
col3.metric(
    "Total Budget", 
    f"€{results['total_budget']:,.0f}"
)

# --- Budget Allocation Plot (User's Request) ---
st.subheader("Optimal vs. Historical Budget Allocation")
st.markdown("This plot shows the shift in budget allocation across brands.")
fig_budget = optimizer.plot_budget_allocation(figsize=(12, 6))
if fig_budget:
    st.pyplot(fig_budget)
    

# --- Channel & Model Plots in Tabs ---
st.subheader("Detailed Plots")
tab1, tab2, tab3, tab4 = st.tabs([
    "Brand BB Channels", 
    "Brand MAC Channels", 
    "Brand TF Channels", 
    "Model Performance"
])

with tab1:
    st.write("Optimal channel allocation for Brand BB")
    fig_ch_bb = optimizer.plot_channel_allocation('bb', figsize=(10, 5))
    if fig_ch_bb:
        st.pyplot(fig_ch_bb)

with tab2:
    st.write("Optimal channel allocation for Brand MAC")
    fig_ch_mac = optimizer.plot_channel_allocation('mac', figsize=(10, 5))
    if fig_ch_mac:
        st.pyplot(fig_ch_mac)
        
with tab3:
    st.write("Optimal channel allocation for Brand TF")
    fig_ch_tf = optimizer.plot_channel_allocation('tf', figsize=(10, 4))
    if fig_ch_tf:
        st.pyplot(fig_ch_tf)

with tab4:
    st.write("Actual vs. Predicted KPI for the base models")
    fig_model_perf = optimizer.plot_model_performance(figsize=(10, 8))
    if fig_model_perf:
        st.pyplot(fig_model_perf)

# --- Results Dataframe & Download ---
st.subheader("Detailed Allocation Data")
st.write("The complete optimization plan, including historical vs. optimal spend for every channel.")
df_results = optimizer.export_results()

if not df_results.empty:
    st.dataframe(df_results)
    
    # Provide download button
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="mmm_optimization_results.csv",
        mime="text/csv",
        use_container_width=True
    )

st.success("Optimization complete! ✨")
