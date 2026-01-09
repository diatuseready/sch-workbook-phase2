import pandas as pd

def _format_forecast_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format display values:
    1. Add thousand separators to all numeric columns
    2. Hide forecast values except for Rack/Lifting, Opening Inv, Close Inv
    """
    df_display = df.copy()
    
    # Columns that should show forecast values
    FORECAST_VISIBLE_COLS = ["Rack/Lifting", "Opening Inv", "Close Inv"]
    
    # Get all numeric columns
    numeric_cols = [
        "Opening Inv", "Close Inv", "Batch In", "Batch Out", 
        "Rack/Lifting", "Pipeline In", "Pipeline Out", 
        "Adjustments", "Gain/Loss", "Transfers", "Production"
    ]
    
    for col in numeric_cols:
        if col not in df_display.columns:
            continue
        
        # Convert entire column to list for processing
        formatted_values = []
        
        # For each row, check if it's a forecast row
        for idx in df_display.index:
            source = str(df_display.at[idx, "source"]).strip().lower() if "source" in df_display.columns else ""
            value = df_display.at[idx, col]
            
            # If it's a forecast row and column should be hidden
            if source == "forecast" and col not in FORECAST_VISIBLE_COLS:
                formatted_values.append("0.00")  # Hide with 0.00 as string
            elif pd.notna(value) and value != "":
                # Format with thousand separators
                try:
                    num_val = float(value)
                    formatted_values.append(f"{num_val:,.2f}")
                except (ValueError, TypeError):
                    formatted_values.append(str(value))  # Convert to string anyway
            else:
                formatted_values.append("0.00")  # Handle NaN/empty as "0.00"
        
        # Replace entire column with formatted strings
        df_display[col] = formatted_values
    
    return df_display