"""
PredictX Flask API â€” Fixed Column Names
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

EXCEL_PATH = 'data/Pakistan_ML_Results.xlsx'
MODELS_DIR = 'models'

# â”€â”€ Cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_cache = {}

def load_sheet(sheet_name):
    if sheet_name not in _cache:
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
            df = df.where(pd.notnull(df), None)
            _cache[sheet_name] = df
        except Exception as e:
            print(f"âŒ Sheet load failed [{sheet_name}]: {e}")
            _cache[sheet_name] = pd.DataFrame()
    return _cache[sheet_name]

# â”€â”€ Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'ok',
        'excel'  : os.path.exists(EXCEL_PATH),
        'models' : os.path.exists(MODELS_DIR),
        'message': 'PredictX API is Working! âœ…'
    })

# â”€â”€ Dashboard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    try:
        profit_df = load_sheet('Profit_Analysis')
        inv_df    = load_sheet('Inventory_Status')

        total_revenue = int(profit_df['Total_Sales'].sum())  if not profit_df.empty else 0
        total_profit  = int(profit_df['Total_Profit'].sum()) if not profit_df.empty else 0
        total_orders  = int(profit_df['Total_Qty'].sum())    if not profit_df.empty else 0

        low_stock = 0
        if not inv_df.empty:
            low_stock = int(inv_df[
                inv_df['Status'].astype(str).str.contains(
                    'LOW|Order Now|ðŸ”´', na=False)
            ].shape[0])

        top_products = []
        if not profit_df.empty:
            top5 = profit_df.nlargest(5, 'Total_Sales')
            for _, row in top5.iterrows():
                top_products.append({
                    'name'   : str(row.get('Product_Name', '')),
                    'revenue': int(row.get('Total_Sales', 0) or 0),
                    'profit' : int(row.get('Total_Profit', 0) or 0),
                })

        return jsonify({
            'total_revenue': total_revenue,
            'total_profit' : total_profit,
            'total_orders' : total_orders,
            'top_products' : top_products,
            'summary'      : {'low_stock': low_stock},
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# â”€â”€ Model Accuracy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/model-accuracy', methods=['GET'])
def model_accuracy():
    try:
        df = load_sheet('Model_Accuracy')
        if df.empty:
            return jsonify([])

        result = []
        for _, row in df.iterrows():
            result.append({
                'City'      : str(row.get('City', 'Lahore')),
                'Model'     : str(row.get('Model', '')),
                'MAE'       : float(row['MAE'])  if row.get('MAE')  is not None else 0,
                'RMSE'      : float(row['RMSE']) if row.get('RMSE') is not None else 0,
                'R2'        : str(row.get('R2_Score', 'N/A')),
                'Accuracy_%': str(row.get('Accuracy_%', 'N/A')),
            })
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# â”€â”€ Ramadan Spike â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/ramadan-spike', methods=['GET'])
def ramadan_spike():
    try:
        df = load_sheet('Ramadan_Spike')
        if df.empty:
            return jsonify([])

        result = []
        for _, row in df.iterrows():
            result.append({
                'Product_Name': str(row.get('Product', '')),
                'Normal_Avg'  : float(row.get('Normal_Avg_Qty', 0) or 0),
                'Ramadan_Avg' : float(row.get('Ramadan_Avg_Qty', 0) or 0),
                'Spike_%'     : float(row.get('Spike_%', 0) or 0),
            })

        result.sort(key=lambda x: x['Spike_%'], reverse=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# â”€â”€ Profit Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/profit-analysis', methods=['GET'])
def profit_analysis():
    try:
        df = load_sheet('Profit_Analysis')
        if df.empty:
            return jsonify([])

        result = []
        for _, row in df.iterrows():
            result.append({
                'Product_Name': str(row.get('Product_Name', '')),
                'Total_Qty'   : int(row.get('Total_Qty_Sold', 0) or 0),
                'Total_Profit': float(row.get('Total_Profit_PKR', 0) or 0),
                'Avg_Profit'  : float(row.get('Avg_Daily_Profit', 0) or 0),
                'Total_Sales' : float(row.get('Total_Sales_PKR', 0) or 0),
                'Margin_%'    : float(row.get('Profit_Margin_%', 0) or 0),
            })

        result.sort(key=lambda x: x['Total_Profit'], reverse=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# â”€â”€ Inventory Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/inventory-status', methods=['GET'])
def inventory_status():
    try:
        df = load_sheet('Inventory_Status')
        if df.empty:
            return jsonify({'inventory': [], 'summary': {}})

        def clean_status(s):
            s = str(s)
            if any(x in s for x in ['ðŸ”´', 'LOW', 'Order Now']):
                return 'Low'
            elif any(x in s for x in ['ðŸ”µ', 'OVER', 'Stop']):
                return 'Overstock'
            return 'Normal'

        result = []
        for _, row in df.iterrows():
            status    = clean_status(row.get('Status', ''))
            avg_daily = float(row.get('Avg_Daily_Sales', 0) or 0)
            current   = int(row.get('Current_Stock', 0) or 0)
            days_rem  = round(current / avg_daily, 1) if avg_daily > 0 else 0

            result.append({
                'product'        : str(row.get('Product', '')),
                'status'         : status,
                'current_stock'  : current,
                'reorder_point'  : int(row.get('Reorder_Point', 0) or 0),
                'reorder_qty'    : int(row.get('Monthly_Order', 0) or 0),
                'avg_daily_sales': avg_daily,
                'days_remaining' : days_rem,
                'Product'        : str(row.get('Product', '')),
                'Status'         : status,
                'Current_Stock'  : current,
                'Reorder_Point'  : int(row.get('Reorder_Point', 0) or 0),
                'Monthly_Order'  : int(row.get('Monthly_Order', 0) or 0),
                'Avg_Daily'      : avg_daily,
            })

        order = {'Low': 0, 'Overstock': 1, 'Normal': 2}
        result.sort(key=lambda x: order.get(x['status'], 3))

        summary = {
            'low_stock': sum(1 for r in result if r['status'] == 'Low'),
            'normal'   : sum(1 for r in result if r['status'] == 'Normal'),
            'overstock': sum(1 for r in result if r['status'] == 'Overstock'),
        }

        return jsonify({'inventory': result, 'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# â”€â”€ Forecast â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/forecast', methods=['GET'])
def forecast():
    try:
        df = load_sheet('30Day_Forecast')
        if df.empty:
            return jsonify({'data': []})

        # Debug â€” terminal mein actual columns dikhao
        print(f"ðŸ“Š 30Day_Forecast columns : {list(df.columns)}")
        print(f"ðŸ“Š Total rows             : {len(df)}")
        if not df.empty:
            print(f"ðŸ“Š Sample row             : {df.iloc[0].to_dict()}")

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        result = []
        for _, row in df.iterrows():
            # âœ… ARIMAX_Forecast column pehle check â€” tab Forecasted_Qty
            qty = (row.get('ARIMAX_Forecast') or
                   row.get('Forecasted_Qty')  or
                   row.get('forecasted_qty')  or
                   row.get('Qty')             or 0)

            product = (row.get('Product') or
                       row.get('Product_Name') or '')

            result.append({
                'Date'           : str(row.get('Date', '')),
                'Product'        : str(product),
                'ARIMAX_Forecast': int(float(qty)),
                'Forecasted_Qty' : int(float(qty)),
                'Day_Type'       : str(row.get('Day_Type', 'ðŸ“… Normal')),
                'Is_Ramadan'     : int(row.get('Is_Ramadan', 0) or 0),
                'Is_Holiday'     : int(row.get('Is_Holiday', 0) or 0),
            })

        print(f"âœ… Forecast rows returned : {len(result)}")
        return jsonify({'data': result})

    except Exception as e:
        print(f"âŒ Forecast error: {e}")
        return jsonify({'error': str(e)}), 500

# â”€â”€ Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == '__main__':
    print("=" * 55)
    print("  PredictX Flask API Starting...")
    print("=" * 55)
    print(f"  Excel : {EXCEL_PATH} â€” {'âœ… Found' if os.path.exists(EXCEL_PATH) else 'âŒ NOT FOUND'}")
    print(f"  Models: {MODELS_DIR}/  â€” {'âœ… Found' if os.path.exists(MODELS_DIR) else 'âŒ NOT FOUND'}")
    print()
    print("  Endpoints:")
    print("  GET /health")
    print("  GET /api/dashboard")
    print("  GET /api/model-accuracy")
    print("  GET /api/ramadan-spike")
    print("  GET /api/profit-analysis")
    print("  GET /api/inventory-status")
    print("  GET /api/forecast")
    print()
    print("  Flutter real device: http://10.71.37.36:5000")
    print("=" * 55)

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
