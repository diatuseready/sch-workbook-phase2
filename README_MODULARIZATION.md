# HF Sinclair Scheduler Dashboard - Modularized Version

## Overview
The Streamlit app has been successfully modularized into smaller, more manageable files for better readability and maintainability. This modular structure is fully supported by Streamlit, including when deployed on Snowflake.

## File Structure

### Main Application
- **`app.py`** - Main application entry point that orchestrates all modules

### Module Files
- **`config.py`** - Configuration settings, theme colors, defaults, and constants
- **`ui_components.py`** - UI styling, CSS, and reusable UI components
- **`data_loader.py`** - Snowflake connection and data loading functions
- **`sidebar_filters.py`** - Sidebar filter components and logic
- **`summary_tab.py`** - Regional Summary tab display and calculations
- **`details_tab.py`** - Details tab display for both Midcon and other regions

### Original Files (Preserved)
- **`scheduler_app_production.py`** - Original monolithic file (2000+ lines)
- **`old_files/`** - Directory containing previous versions

## Benefits of Modularization

1. **Better Readability**: Each module has a specific purpose and is easy to understand
2. **Easier Maintenance**: Changes can be made to specific modules without affecting others
3. **Code Reusability**: Functions can be easily imported and reused
4. **Team Collaboration**: Multiple developers can work on different modules simultaneously
5. **Easier Testing**: Individual modules can be tested independently

## How to Run

### For Local Development
```bash
streamlit run app.py
```

### For Snowflake Deployment
The modular structure is fully compatible with Snowflake's Streamlit support. Simply deploy the `app.py` as the main entry point, and ensure all module files are included in the deployment.

## Module Descriptions

### config.py (56 lines)
- Theme colors and styling constants
- Default values for calculations
- Table definitions
- Mock data sources

### ui_components.py (90 lines)
- Page setup configuration
- Custom CSS styling
- Header display
- Data freshness cards

### data_loader.py (128 lines)
- Snowflake session management
- Data loading from tables
- Data transformation and cleaning
- Session state management

### sidebar_filters.py (102 lines)
- Region selector
- Date range picker
- Location/System filter
- Product filter
- Filter application logic

### summary_tab.py (272 lines)
- Regional summary calculations
- 7-day average calculations
- Required maximums and intransit calculations
- Forecast table generation

### details_tab.py (100 lines)
- Midcon details view
- Location-based details view
- Editable data grids
- Save functionality placeholder

### app.py (72 lines)
- Main application orchestration
- Module integration
- Tab management
- Application flow control

## Migration from Original File

The original `scheduler_app_production.py` file (2000+ lines) has been preserved for reference. The new modular structure provides the same functionality but with better organization.

### Key Changes:
1. **No functional changes** - All features work exactly as before
2. **Import structure** - Functions are now imported from their respective modules
3. **Session state** - Managed centrally through data_loader.py
4. **Styling** - Centralized in ui_components.py

## Future Enhancements

With this modular structure, it's now easier to:
- Add new features to specific modules
- Update the UI without touching business logic
- Add unit tests for individual modules
- Implement proper error handling per module
- Add logging and monitoring
- Scale the application

## Compatibility

✅ **Fully compatible with:**
- Streamlit (all versions)
- Snowflake Streamlit apps
- Local development
- Cloud deployments

## Notes

- All modules follow Python best practices
- Docstrings are included for better documentation
- The structure allows for easy extension and modification
- No external dependencies were added beyond the original requirements
