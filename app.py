import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
from matplotlib.ticker import FuncFormatter # Added for plot formatting
from matplotlib.backends.backend_pdf import PdfPages # Added for PDF export

warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="MMM Budget Optimiser")

# --- Brand Name Mapping ---
BRAND_DISPLAY_MAP = {
    'bb': 'Bobbi Brown',
    'mac': 'MAC',
    'tf': 'Too Faced'
}

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# START: MMMBudgetOptimiser CLASS
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=

class MMMBudgetOptimiser:
    """
    Marketing Mix Modelling Budget Optimiser
    Optimises budget allocation across brands and channels using regression-based MMM
    """
    
    def __init__(self, data_path: str):
        """Initialize with path to CSV data"""
        try:
            self.data = pd.read_csv(data_path)
        except FileNotFoundError:
            st.error(f"ERROR: Data file not found at {data_path}")
            st.stop()
        
        # --- Define ATL/BTL Channel Identifiers ---
        self.atl_identifiers = ['ooh', 'partnership', 'vod', 'tv']
        
        # --- Define Saturation Priors (Approach 2) ---
        self.channel_saturation_priors = {
            'search':     {'alpha': 1.0, 'gamma': 10000}, # Saturates fast
            'meta':       {'alpha': 1.0, 'gamma': 40000}, # Mid
            'social':     {'alpha': 1.0, 'gamma': 35000}, # Mid
            'prog':       {'alpha': 1.0, 'gamma': 45000}, # Mid
            'partnership':{'alpha': 1.0, 'gamma': 70000}, # Slow
            'vod':        {'alpha': 1.0, 'gamma': 80000}, # Slow
            'ooh':        {'alpha': 1.0, 'gamma': 85000}, # Very Slow
            'tv':         {'alpha': 1.0, 'gamma': 90000}  # Very Slow
        }
        
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
        self.optimisation_results = {}

        # --- NEW: Add Seasonality Features (NO TREND) ---
        # Create 'week_of_year' dummies (assuming data is chronological weekly)
        self.data['week_of_year'] = self.data.index % 52
        # drop_first=True avoids multicollinearity (dummy variable trap)
        season_dummies = pd.get_dummies(self.data['week_of_year'], prefix='week', drop_first=True).astype(float)
        
        self.data = pd.concat([self.data, season_dummies], axis=1)
        
        # Store the names of these new control variables
        self.control_features = list(season_dummies.columns)
        # --- END NEW FEATURES ---

        
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
        # Add a small epsilon to prevent division by zero if gamma/x are 0
        epsilon = 1e-10
        return np.power(x_np, alpha) / (np.power(gamma, alpha) + np.power(x_np, alpha) + epsilon)
    
    def build_mmm_models(self, 
                         adstock_rate_atl: float = 0.6, 
                         adstock_rate_btl: float = 0.2):
        """
        Build MMM regression models for each brand.
        MODIFIED: Now includes Seasonality control variables.
        """
        
        for brand in self.brand_channels.keys():
            kpi_col = self.brand_kpis[brand]
            y = self.data[kpi_col].values
            
            # --- MODIFIED: Separate Media Features from Control Features ---
            
            # 1. Process Media Features (Adstock, Saturation)
            X_transformed_media = []
            feature_names_media = []
            channel_metadata = {}
            
            for channel in self.brand_channels[brand]:
                raw_spend = self.data[channel].fillna(0).values
                
                is_atl = any(identifier in channel for identifier in self.atl_identifiers)
                current_adstock_rate = adstock_rate_atl if is_atl else adstock_rate_btl
                adstocked = self.apply_adstock(raw_spend, rate=current_adstock_rate)
                
                current_alpha = 1.0   # Default
                current_gamma = 50000 # Default
                for identifier, params in self.channel_saturation_priors.items():
                    if identifier in channel:
                        current_alpha = params['alpha']
                        current_gamma = params['gamma']
                        break
                
                saturated = self.apply_saturation(adstocked, 
                                                  alpha=current_alpha, 
                                                  gamma=current_gamma)
                
                X_transformed_media.append(saturated)
                feature_names_media.append(channel)
                
                channel_metadata[channel] = {
                    'adstock_rate': current_adstock_rate,
                    'saturation_alpha': current_alpha,
                    'saturation_gamma': current_gamma
                }

            X_media_stack = np.column_stack(X_transformed_media)
            
            # 2. Get Control Features (Seasonality)
            # These do not get adstock or saturation
            X_controls = self.data[self.control_features].values
            
            # 3. Combine all features
            X_final_features = np.column_stack([X_media_stack, X_controls])
            all_feature_names = feature_names_media + self.control_features
            
            # 4. Add intercept and run regression
            X_with_intercept = np.column_stack([np.ones(len(X_final_features)), X_final_features])
            
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
            
            # Store the model, now including control feature names
            self.models[brand] = {
                'intercept': beta[0],
                'coefficients': dict(zip(all_feature_names, beta[1:])), # Now has all features
                'channel_metadata': channel_metadata, 
                'control_features': self.control_features, # Store for reference
                'X_transformed': X_final_features, # This is the full feature set
                'y_actual': y,
                'y_pred': y_pred,
                'r2': r2,
                'mape': mape
            }
            
            # --- MODIFIED: Removed R² from this message, now just confirms build ---
            # st.write(f"Model for {BRAND_DISPLAY_MAP.get(brand, brand.upper())} built.")
    
    def predict_kpi(self, brand: str, weekly_spend_dict: Dict[str, float]) -> float:
        """
        Predict KPI for given weekly spend allocation.
        MODIFIED: This function is for the *optimiser*. It *only*
        calculates the contribution from media spend, as the optimiser
        cannot change trend or seasonality. The 'intercept' from the
        model has absorbed the average effect of the control variables.
        """
        model = self.models[brand]
        
        # Start with intercept. This is CRITICAL.
        # The intercept from the model (Y = B0 + B_media*X_media + B_control*X_control)
        # represents the base KPI *plus* the average effect of all control variables.
        # The optimiser should only add the *media* contribution on top of this.
        prediction = model['intercept'] 
        
        for channel, spend in weekly_spend_dict.items():
            if channel in model['coefficients']:
                
                try:
                    metadata = model['channel_metadata'][channel]
                    adstock_rate = metadata['adstock_rate']
                    alpha = metadata.get('saturation_alpha', 1.0) 
                    gamma = metadata.get('saturation_gamma', 50000)
                except KeyError:
                    # This channel is a control variable (e.g., 'week_1')
                    # or an error occurred. Skip it.
                    continue 
                
                adstock_mult = 1 / (1 - adstock_rate) 
                adstocked = spend * adstock_mult
                
                saturated = self.apply_saturation(
                    np.array([adstocked]), 
                    alpha=alpha,
                    gamma=gamma
                )[0]
                
                prediction += model['coefficients'][channel] * saturated
        return max(0, prediction)
    
    def optimise_budget(self, total_budget: float = 5_000_000, 
                        weeks: int = 52,
                        min_brand_share: float = 0.15,
                        max_brand_share: float = 0.60,
                        channel_bound_pct: float = 0.30):
        """Optimise budget allocation across brands and channels"""
        
        # This function's internal logic does not need to change,
        # as the new model intelligence is in build_mmm_models() and predict_kpi()
        
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
        # --- CRITICAL FIX: Using Historical *Average* Spend ---
        for channel in all_channels:
            total_channel_spend = self.data[channel].fillna(0).sum()
            
            # Calculate average weekly spend
            # 'weeks' is an argument to this function (default 52)
            average_weekly_spend = total_channel_spend / weeks 
            
            # Apply the percentage bounds to the average
            lower_bound = max(0, average_weekly_spend * (1.0 - channel_bound_pct))
            upper_bound = average_weekly_spend * (1.0 + channel_bound_pct)

            # If average spend was 0, both bounds must be 0
            if average_weekly_spend == 0:
                lower_bound = 0
                upper_bound = 0
            
            # Failsafe in case min > max (e.g., if pct > 1.0)
            if lower_bound > upper_bound:
                lower_bound = upper_bound
            
            bounds.append((lower_bound, upper_bound))
        # --- END CRITICAL FIX ---

        def objective(weekly_channel_spends):
            total_kpi = 0
            for brand in brands:
                weekly_spend_dict = {}
                indices = brand_channel_indices[brand]
                channels_list = self.brand_channels[brand]
                for i, channel_name in enumerate(channels_list):
                    channel_index = indices[i]
                    weekly_spend_dict[channel_name] = weekly_channel_spends[channel_index]
                # This call now uses the "smarter" predict_kpi
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
                           (historical_total_kpi + 1e-10)) * 100
        
        self.optimisation_results = {
            'total_budget': total_budget,
            'optimal_allocation': optimal_allocation,
            'optimal_total_kpi': optimal_total_kpi,
            'historical': historical,
            'historical_total_kpi': historical_total_kpi,
            'improvement_pct': improvement_pct,
            'optimal_shares': optimal_shares_dict
        }
        return self.optimisation_results
    
    def _print_optimisation_results(self):
        # This function is not used by the script, but part of the class
        pass
    
    def plot_model_performance(self, figsize=(15, 10)):
        """
        Plot actual vs predicted KPI for all brands.
        REMOVED: R² from title.
        """
        brands = list(self.models.keys())
        n_brands = len(brands)
        
        if n_brands == 0:
            st.warning("No models found.")
            return None

        fig, axes = plt.subplots(n_brands, 1, figsize=figsize, squeeze=False)
        fig.patch.set_facecolor('#FFFFFF') # White background
        axes = axes.flatten()
        
        # Define colors
        COLOR_ACTUAL = '#184ec8' # ELC Blue
        COLOR_PREDICTED = '#ebd79a' # ELC Gold
        COLOR_TEXT = '#000000' # MAC Black
        COLOR_TITLE = '#040A2B' # ELC Navy
        COLOR_GRID = '#F0F2E9' # Light Neutral
        
        for i, brand in enumerate(brands):
            ax = axes[i] # Get current axis
            ax.set_facecolor('#FFFFFF') # White background for plot area

            model = self.models[brand]
            brand_display = BRAND_DISPLAY_MAP.get(brand, brand.upper())
            weeks = range(1, len(model['y_actual']) + 1)
            
            ax.plot(weeks, model['y_actual'], label='Actual', linewidth=2, marker='o', markersize=4, alpha=0.9, color=COLOR_ACTUAL)
            ax.plot(weeks, model['y_pred'], label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.9, color=COLOR_PREDICTED)
            
            # --- MODIFIED: Title no longer shows R² ---
            ax.set_title(f'Brand {brand_display} - KPI Performance', 
                         fontsize=12, fontweight='bold', color=COLOR_TITLE)
            
            ax.set_xlabel('Week', color=COLOR_TEXT)
            ax.set_ylabel('KPI', color=COLOR_TEXT)
            
            ax.grid(True, alpha=0.7, color=COLOR_GRID) # Light grid
            
            # Disable scientific notation on y-axis
            ax.ticklabel_format(style='plain', axis='y')
            
            # Set tick colors
            ax.tick_params(colors=COLOR_TEXT)
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_TEXT)
            
            # Set legend text color
            legend = ax.legend()
            for text in legend.get_texts():
                text.set_color(COLOR_TEXT)
        
        plt.tight_layout()
        return fig
    
    def plot_budget_allocation(self, figsize=(15, 8)):
        """Plot optimal vs historical budget allocation"""
        if not self.optimisation_results:
            st.warning("Please run optimise_budget() first!")
            return None
        
        results = self.optimisation_results
        brands = list(results['optimal_allocation'].keys()) # ['bb', 'mac', 'tf']
        
        # --- Define Colors ---
        COLOR_HISTORICAL = '#184ec8' # ELC Blue
        COLOR_OPTIMAL = '#040A2B' # ELC Navy
        COLOR_TEXT = '#000000' # MAC Black
        COLOR_TITLE = '#040A2B' # ELC Navy
        COLOR_GRID = '#F0F2E9' # Light Neutral
        
        PIE_COLORS = {
            'bb': '#DDCBBE', # Bobbi Brown Nude
            'mac': '#000000', # MAC Black
            'tf': '#FF5EA2'  # Too Faced Pink
        }
        pie_colors_list = [PIE_COLORS.get(b, '#999999') for b in brands]
        
        brand_labels = [BRAND_DISPLAY_MAP.get(b, b.upper()) for b in brands]
        historical_budgets = [results['historical'][b]['spend']/1_000_000 for b in brands]
        optimal_budgets = [results['optimal_allocation'][b]['total_budget']/1_000_000 for b in brands]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor('#FFFFFF')
        
        # --- Bar Chart (ax1) ---
        ax1.set_facecolor('#FFFFFF')
        x = np.arange(len(brands))
        width = 0.35
        ax1.bar(x - width/2, historical_budgets, width, label='Historical', alpha=0.8, color=COLOR_HISTORICAL)
        ax1.bar(x + width/2, optimal_budgets, width, label='Optimal', alpha=0.8, color=COLOR_OPTIMAL)
        
        ax1.set_xlabel('Brand', color=COLOR_TEXT)
        ax1.set_ylabel('Budget (£M)', color=COLOR_TEXT)
        ax1.set_title('Budget Allocation Comparison', fontweight='bold', color=COLOR_TITLE)
        ax1.set_xticks(x)
        ax1.set_xticklabels(brand_labels, color=COLOR_TEXT)
        
        legend = ax1.legend()
        for text in legend.get_texts():
            text.set_color(COLOR_TEXT)
            
        ax1.grid(True, alpha=0.7, axis='y', color=COLOR_GRID)
        
        for i, (h, o) in enumerate(zip(historical_budgets, optimal_budgets)):
            ax1.text(i - width/2, h, f'£{h:.2f}M', ha='center', va='bottom', fontsize=9, color=COLOR_TEXT)
            ax1.text(i + width/2, o, f'£{o:.2f}M', ha='center', va='bottom', fontsize=9, color=COLOR_TEXT)
        
        ax1.tick_params(colors=COLOR_TEXT)
        for spine in ax1.spines.values():
            spine.set_edgecolor(COLOR_TEXT)

        # --- Pie Chart (ax2) ---
        ax2.set_facecolor('#FFFFFF')
        if sum(optimal_budgets) > 0:
            wedges, texts, autotexts = ax2.pie(optimal_budgets, labels=brand_labels, autopct='%.0f%%',
                                               colors=pie_colors_list, startangle=90,
                                               textprops={'color': COLOR_TEXT})
            for autotext in autotexts:
                autotext.set_color('#FFFFFF')
                autotext.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, "No Budget Allocated", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color=COLOR_TEXT)
        
        ax2.set_title('Optimal Budget Distribution', fontweight='bold', color=COLOR_TITLE)
        
        plt.tight_layout()
        return fig
    
    def plot_channel_allocation(self, brand: str, figsize=(12, 6)):
        """Plot channel-level budget allocation for a specific brand"""
        if not self.optimisation_results:
            st.warning("Please run optimise_budget() first!")
            return None
        
        results = self.optimisation_results
        if brand not in results['optimal_allocation']:
            st.warning(f"Brand '{brand}' not found in results.")
            return None
            
        allocation = results['optimal_allocation'][brand]
        channels = allocation['channels']
        sorted_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_channels:
            st.info(f"No channels to plot for brand {brand.upper()}")
            return None

        # --- Define Colors ---
        BRAND_CHART_COLORS = {
            'bb': '#DDCBBE', # Bobbi Brown Nude
            'mac': '#000000', # MAC Black
            'tf': '#FF5EA2'  # Too Faced Pink
        }
        BAR_COLOR = BRAND_CHART_COLORS.get(brand, '#040A2B') # Default to Navy
        COLOR_TEXT = '#000000' # MAC Black
        COLOR_TITLE = '#040A2B' # ELC Navy
        COLOR_GRID = '#F0F2E9' # Light Neutral

        channel_names = [ch.replace(f'_{brand}_sp', '').replace('m1_', '').replace('m3_', '').capitalize() for ch, _ in sorted_channels]
        budgets = [budget/1_000_000 for _, budget in sorted_channels]
        brand_display = BRAND_DISPLAY_MAP.get(brand, brand.upper())
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        bars = ax.barh(channel_names, budgets, color=BAR_COLOR)
        
        ax.set_xlabel('Budget (£M)', fontweight='bold', color=COLOR_TEXT)
        ax.set_title(f'Brand {brand_display} - Channel Budget Allocation (Total: £{allocation["total_budget"]/1_000_000:.2f}M)', 
                       fontweight='bold', fontsize=14, color=COLOR_TITLE)
        ax.grid(True, alpha=0.7, axis='x', color=COLOR_GRID)
        ax.invert_yaxis()
        
        ax.tick_params(colors=COLOR_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_TEXT)

        for i, (bar, budget) in enumerate(zip(bars, budgets)):
            width = bar.get_width()
            total_brand_budget = allocation['total_budget']
            pct = (sorted_channels[i][1] / total_brand_budget * 100) if total_brand_budget > 0 else 0
            
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f' £{budget:.2f}M ({pct:.0f}%)', 
                    ha='left', va='center', fontweight='bold', fontsize=9,
                    color=COLOR_TEXT)
        
        plt.tight_layout()
        return fig
    
    def export_results(self, filename: str = 'mmm_optimisation_results.csv'):
        """Export optimisation results to CSV"""
        if not self.optimisation_results:
            st.warning("Please run optimise_budget() first!")
            return pd.DataFrame()
        
        results = self.optimisation_results
        export_data = []
        
        for brand, allocation in results['optimal_allocation'].items():
            hist = results['historical'][brand]
            brand_display = BRAND_DISPLAY_MAP.get(brand, brand.upper())
            
            export_data.append({
                'Brand': brand_display, 'Channel': 'TOTAL',
                'Historical_Spend': hist['spend'],
                'Optimal_Budget': allocation['total_budget'],
                'Change_Amount': allocation['total_budget'] - hist['spend'],
                'Change_Percent': ((allocation['total_budget'] - hist['spend']) / (hist['spend'] + 1) * 100),
                'Budget_Share': (allocation['share'] * 100)
            })
            for channel, budget in allocation['channels'].items():
                ch_name = channel.replace(f'_{brand}_sp', '').replace('m1_', '').replace('m3_', '').capitalize()
                hist_channel = self.data[channel].sum()
                export_data.append({
                    'Brand': brand_display, 'Channel': ch_name,
                    'Historical_Spend': hist_channel,
                    'Optimal_Budget': budget,
                    'Change_Amount': budget - hist_channel,
                    'Change_Percent': ((budget - hist_channel) / (hist_channel + 1) * 100),
                    'Budget_Share': (budget / (allocation['total_budget'] + 1e-10)) * 100
                })
        
        df_export = pd.DataFrame(export_data)
        return df_export

    # ========================================================================
    # --- RESPONSE CURVE PLOTTING METHODS (MODIFIED) ---
    # ========================================================================
    
    def _generate_response_curve(self, brand: str, channel: str):
        """
        Calculates the KPI response for a range of weekly spends
        for a single channel, holding all other channels at their
        *optimised* average.
        MODIFIED: Now uses per-channel saturation params.
        """
        if not self.optimisation_results:
            # Fallback: use historical average if not optimised
            all_media_channels = [c for brand_channels in self.brand_channels.values() for c in brand_channels]
            avg_spends = {ch: self.data[ch].mean() for ch in all_media_channels}
        else:
            # Use optimised average
            avg_spends = {}
            for b, alloc in self.optimisation_results['optimal_allocation'].items():
                for ch, annual_budget in alloc['channels'].items():
                    avg_spends[ch] = annual_budget / 52.0
        
        # --- MODIFIED: Read all params from channel_metadata ---
        model = self.models[brand]
        try:
            coeff = model['coefficients'][channel]
            metadata = model['channel_metadata'][channel]
            adstock_rate = metadata['adstock_rate']
            alpha = metadata.get('saturation_alpha', 1.0)
            gamma = metadata.get('saturation_gamma', 50000)
        except KeyError:
            # Channel not in model, or model built incorrectly
            return None, None, None
        # --- END MODIFIED LOGIC ---
        
        # Create a range of spend values
        max_spend = self.data[channel].max()
        plot_range_max = max(max_spend * 2, gamma * 2) 
        if plot_range_max == 0:
            plot_range_max = 50000 
        
        spend_range = np.linspace(0, plot_range_max, 100)
        kpi_response = []
        
        # --- MODIFIED: Get base prediction ---
        # Base KPI = Intercept + AVG(Seasonality Effect)
        # We get this by taking the model's intercept.
        # Then, we add the KPI from all *other* media channels.
        base_kpi = model['intercept'] 
        
        for ch, spend in avg_spends.items():
            if ch != channel and ch in model['coefficients']:
                # Call predict_kpi for *other* channels to get their contribution
                # We subtract intercept because predict_kpi *also* includes it
                base_kpi += self.predict_kpi(brand, {ch: spend}) - model['intercept']

        for spend in spend_range:
            # Calculate this channel's contribution
            adstock_mult = 1 / (1 - adstock_rate)
            adstocked = spend * adstock_mult
            saturated = self.apply_saturation(np.array([adstocked]), alpha, gamma)[0]
            
            channel_kpi = coeff * saturated
            
            # Add to base KPI
            kpi_response.append(base_kpi + channel_kpi)
            
        # Get current/optimised spend point
        current_spend = avg_spends.get(channel, 0)
        # Recalculate the KPI at this single point
        current_kpi_spend_dict = avg_spends.copy()
        current_kpi_spend_dict[channel] = current_spend
        # We need to *remove* control vars from this dict if they are in it
        for ctrl_var in self.control_features:
            current_kpi_spend_dict.pop(ctrl_var, None)
            
        current_kpi = self.predict_kpi(brand, current_kpi_spend_dict)

        return spend_range, kpi_response, (current_spend, current_kpi)

    def plot_all_response_curves(self, brand: str, weeks: int = 52):
        """
        Plots response curves for all channels of a given brand
        and saves them to a combined PDF.
        """
        
        channels = self.brand_channels[brand]
        brand_display = BRAND_DISPLAY_MAP.get(brand, brand.upper())
        
        n_channels = len(channels)
        n_cols = 3
        n_rows = int(np.ceil(n_channels / n_cols))
        
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(n_cols * 5, n_rows * 4), 
            squeeze=False
        )
        fig.patch.set_facecolor('#FFFFFF')
        
        # --- Define Colors ---
        BRAND_CHART_COLORS = {
            'bb': '#DDCBBE', # Bobbi Brown Nude
            'mac': '#000000', # MAC Black
            'tf': '#FF5EA2'  # Too Faced Pink
        }
        CURVE_COLOR = BRAND_CHART_COLORS.get(brand, '#040A2B') # Default to Navy
        POINT_COLOR = '#ebd79a' # ELC Gold
        COLOR_TEXT = '#000000' # MAC Black
        COLOR_TITLE = '#040A2B' # ELC Navy
        COLOR_GRID = '#F0F2E9' # Light Neutral

        fig.suptitle(f'{brand_display} - Channel Response Curves', 
                     fontsize=20, fontweight='bold', color=COLOR_TITLE)
        
        axes_flat = axes.flatten()
        plot_count = 0
        
        for i, channel in enumerate(channels):
            ax = axes_flat[i]
            ax.set_facecolor('#FFFFFF')
            
            spend_range, kpi_response, current_point = self._generate_response_curve(brand, channel)
            
            if spend_range is None:
                ax.text(0.5, 0.5, 'Channel not in model', 
                        ha='center', va='center', transform=ax.transAxes, 
                        color=COLOR_TEXT)
                ax.set_title(channel.replace(f'_{brand}_sp', '').replace('m1_', '').replace('m3_', '').capitalize(), 
                             fontweight='bold', color=COLOR_TITLE)
                continue

            # Plot the curve
            ax.plot(spend_range, kpi_response, color=CURVE_COLOR, linewidth=2)
            
            # Plot the current/optimised point
            if current_point:
                ax.plot(current_point[0], current_point[1], 
                        marker='o', markersize=8, color=POINT_COLOR,
                        label=f'Current Spend (£{current_point[0]:,.0f})')
                
                # Add text label for the point
                ax.text(current_point[0], current_point[1] * 1.01, 
                        f' £{current_point[0]:,.0f}\n {current_point[1]:,.0f} KPI',
                        ha='center', va='bottom', fontsize=9, color=COLOR_TEXT)
                
            # Formatting
            ax.set_title(channel.replace(f'_{brand}_sp', '').replace('m1_', '').replace('m3_', '').capitalize(), 
                         fontweight='bold', color=COLOR_TITLE)
            ax.set_xlabel('Weekly Spend (£)', color=COLOR_TEXT)
            ax.set_ylabel('Predicted Weekly KPI', color=COLOR_TEXT)
            ax.grid(True, alpha=0.7, color=COLOR_GRID)
            
            # Format x-axis to be non-scientific
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'£{x:,.0f}'))
            ax.tick_params(colors=COLOR_TEXT)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_TEXT)
            
            plot_count += 1

        # Turn off any unused subplots
        for j in range(plot_count, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        
        # Save to PDF
        pdf_filename = f'{brand}_response_curves.pdf'
        try:
            with PdfPages(pdf_filename) as pdf:
                pdf.savefig(fig)
            print(f"Saved response curves to {pdf_filename}")
        except Exception as e:
            print(f"Error saving PDF: {e}")
            st.error(f"Could not save PDF: {e}")

        plt.close(fig) # Close the figure to save memory
        return fig, pdf_filename


# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# END: MMMBudgetOptimiser CLASS
# =_=_=_-_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=


# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# START: STREAMLIT APP LOGIC
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=

# Title for the app
st.title("📈 MMM lite ELC Cluster Budget Optimiser")

# --- Sidebar for User Inputs ---
st.sidebar.header("1. Optimisation Parameters")

total_budget_input = st.sidebar.number_input(
    "Total Annual Budget (£)", 
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

# --- UPDATED: This slider now uses the "Historical Average" logic ---
channel_bound_pct_input = st.sidebar.slider(
    "Channel Spend Bounds (+/- % from Historical Avg.)",
    min_value=0.0, 
    max_value=100.0, 
    value=30.0, 
    step=5.0,
    help="Sets the min/max weekly spend for each channel as a percentage of its historical *average*. E.g., 30% allows spend to be within +/- 30% of the historical average weekly spend for that channel."
) / 100.0 # Convert to decimal


# --- Caching the Model *Building* ---
@st.cache_resource
def load_and_build_model(data_path='final_kpi_weekly_reduced.csv'):
    """
    Load data and build the MMM models.
    This is cached so it only runs once per session.
    """
    try:
        optimiser = MMMBudgetOptimiser(data_path)
    except Exception as e:
        st.error(f"Failed to load data file: {data_path}. Error: {e}")
        st.stop()
        
    spinner_msg = "Building base models with Seasonality... (This runs once per session)"
    with st.spinner(spinner_msg):
        # --- MODIFIED: Call no longer needs saturation params ---
        optimiser.build_mmm_models(
            adstock_rate_atl=0.6,
            adstock_rate_btl=0.2
            # Saturation params are now read from the class priors
        )
    st.sidebar.success("Models built successfully!")
    return optimiser

# Load and build the model
optimiser = load_and_build_model('final_kpi_weekly_reduced.csv')


# --- Glossary & Guide ---
st.sidebar.markdown("---")
st.sidebar.header("Usage Guide & Glossary")
with st.sidebar.expander("Click to see guide", expanded=False):
    st.markdown("""
    **How to Use:**
    1.  **Set Parameters:** Adjust the sliders and inputs on the left to define your total budget and spending rules (constraints).
    2.  **Set Channel Bounds:** Use the slider to set the allowed spend deviation for all channels. This is now based on the *historical average*, which gives tighter, more realistic control.
    3.  **Run Optimisation:** Click the `Run Budget Optimisation` button.
    4.  **Review Results:** Analyse the charts and data table to understand the recommended budget plan.
    
    **Glossary:**
    * **KPI (Key Performance Indicator):** The main metric you want to maximise (e.g., sales, revenue, conversions).
    * **Brand Constraints:** Rules that set the minimum and maximum percentage of the *total* budget a single brand can receive.
    * **Channel Constraints:** (You are here) A global rule that controls how much spend can change for *all* channels based on their historical *average*.
    * **Adstock:** A model of the carryover effect of advertising; spend from this week still has an impact in future weeks.
    * **Saturation (Diminishing Returns):** The point at which additional spend on a channel produces less and less of a return. Our model now uses *different* saturation curves for each channel type (e.g., Search saturates faster than TV).
    * **Seasonality:** Control variables added to the model to separate the effect of media from natural seasonal spikes (e.g., Christmas). This makes the R² (model fit) more accurate.
    """)


# --- Run Optimisation ---
st.sidebar.markdown("---")
st.sidebar.header("2. Run Optimisation")
run_button = st.sidebar.button("🚀 Run Budget Optimisation", type="primary", use_container_width=True)

# Main content area
st.markdown("---")

if not run_button:
    st.info("Adjust the parameters in the sidebar and click 'Run Budget Optimisation' to see the results.")
    st.stop()

# --- Display Results (if button is clicked) ---
with st.spinner(f"Optimising a £{total_budget_input:,.0f} budget..."):
    # Run the optimisation method on the loaded optimiser object
    results = optimiser.optimise_budget(
        total_budget=total_budget_input,
        weeks=52,
        min_brand_share=min_brand_share_input,
        max_brand_share=max_brand_share_input,
        channel_bound_pct=channel_bound_pct_input # <-- Pass the slider value
    )

st.header("📊 Optimisation Results")

# --- Key Metrics ---
st.subheader("High-Level Summary")
col1, col2, col3 = st.columns(3)
col1.metric(
    "Optimal Annual KPI", 
    f"{results['optimal_total_kpi']:,.0f}",
    f"{results['improvement_pct']:.0f}% vs Historical"
)
col2.metric(
    "Historical Annual KPI", 
    f"{results['historical_total_kpi']:,.0f}"
)
col3.metric(
    "Total Budget", 
    f"£{results['total_budget']:,.0f}"
)

# --- Budget Allocation Plot (User's Request) ---
st.subheader("Optimal vs. Historical Budget Allocation")
st.markdown("This plot shows the shift in budget allocation across brands.")
fig_budget = optimiser.plot_budget_allocation(figsize=(12, 6))
if fig_budget:
    st.pyplot(fig_budget)
    

# --- Channel & Model Plots in Tabs ---
st.subheader("Detailed Plots")
# --- UPDATED: Removed "Model Performance" tab ---
tab_list = [
    "Bobbi Brown Channels",
    "MAC Channels",
    "Too Faced Channels",
]
tabs = st.tabs(tab_list)

# --- NEW: Generate response curves but DO NOT display them ---
# They are still saved to PDF locally by the plot_all_response_curves method
with st.spinner("Generating response curve PDFs..."):
    _, pdf_bb = optimiser.plot_all_response_curves('bb')
    _, pdf_mac = optimiser.plot_all_response_curves('mac')
    _, pdf_tf = optimiser.plot_all_response_curves('tf')

# --- Populate Tabs (Simplified) ---
with tabs[0]: # Bobbi Brown Channels
    st.write("Optimal channel allocation for Brand Bobbi Brown")
    fig_ch_bb = optimiser.plot_channel_allocation('bb', figsize=(10, 5))
    if fig_ch_bb:
        st.pyplot(fig_ch_bb)

with tabs[1]: # MAC Channels
    st.write("Optimal channel allocation for Brand MAC")
    fig_ch_mac = optimiser.plot_channel_allocation('mac', figsize=(10, 5))
    if fig_ch_mac:
        st.pyplot(fig_ch_mac)
        
with tabs[2]: # Too Faced Channels
    st.write("Optimal channel allocation for Brand Too Faced")
    fig_ch_tf = optimiser.plot_channel_allocation('tf', figsize=(10, 4))
    if fig_ch_tf:
        st.pyplot(fig_ch_tf)

# --- REMOVED: Model Performance Tab ---
# with tabs[3]: # Model Performance
#     st.write("Actual vs. Predicted KPI for the base models (from one-time training)")
#     fig_model_perf = optimiser.plot_model_performance(figsize=(10, 8))
#     if fig_model_perf:
#         st.pyplot(fig_model_perf)

# --- Results Dataframe & Download ---
st.subheader("Detailed Allocation Data")
st.write("The complete optimisation plan, including historical vs. optimal spend for every channel.")
df_results = optimiser.export_results()

if not df_results.empty:
    st.dataframe(
        df_results,
        column_config={
            "Historical_Spend": st.column_config.NumberColumn(format="£%.2f"),
            "Optimal_Budget": st.column_config.NumberColumn(format="£%.2f"),
            "Change_Amount": st.column_config.NumberColumn(format="£%.2f"),
            "Change_Percent": st.column_config.NumberColumn(format="%.0f%%"),
            "Budget_Share": st.column_config.NumberColumn(format="%.0f%%")
        },
        use_container_width=True
    )
    
    # Provide download button
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="mmm_optimisation_results.csv",
        mime="text/csv",
        use_container_width=True
    )

st.success("Optimisation complete! ✨ Response curve PDFs have been saved locally.")

