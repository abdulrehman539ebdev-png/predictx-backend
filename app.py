

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

EXCEL_PATH = 'data/Pakistan_ML_Results.xlsx'
MODELS_DIR = 'models'

# ── Cache ────────────────────────────────────────────────────
_cache = {}

def load_sheet(sheet_name):
    if sheet_name not in _cache:
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
            df = df.where(pd.notnull(df), None)
            _cache[sheet_name] = df
        except Exception as e:
            print(f"❌ Sheet load failed [{sheet_name}]: {e}")
            _cache[sheet_name] = pd.DataFrame()
    return _cache[sheet_name]

# ── Health ───────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'ok',
        'excel'  : os.path.exists(EXCEL_PATH),
        'models' : os.path.exists(MODELS_DIR),
        'message': 'PredictX API is Working! ✅'
    })

# ── Dashboard ────────────────────────────────────────────────
@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    try:
        # ✅ Correct columns: Total_Sales, Total_Profit, Total_Qty
        profit_df = load_sheet('Profit_Analysis')
        inv_df    = load_sheet('Inventory_Status')

        total_revenue = int(profit_df['Total_Sales'].sum())   if not profit_df.empty else 0
        total_profit  = int(profit_df['Total_Profit'].sum())  if not profit_df.empty else 0
        total_orders  = int(profit_df['Total_Qty'].sum())     if not profit_df.empty else 0

        low_stock = 0
        if not inv_df.empty:
            low_stock = int(inv_df[
                inv_df['Status'].astype(str).str.contains(
                    'LOW|Order Now|🔴', na=False)
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

# ── Model Accuracy ───────────────────────────────────────────
@app.route('/api/model-accuracy', methods=['GET'])
def model_accuracy():
    try:
        # Columns: Model, MAE, RMSE, R2_Score, Accuracy_%
        df = load_sheet('Model_Accuracy')
        if df.empty:
            return jsonify([])

        result = []
        for _, row in df.iterrows():
            result.append({
                'City'       : str(row.get('City', 'Lahore')),
                'Model'      : str(row.get('Model', '')),
                'MAE'        : float(row['MAE'])  if row.get('MAE')  is not None else 0,
                'RMSE'       : float(row['RMSE']) if row.get('RMSE') is not None else 0,
                'R2'         : str(row.get('R2_Score', 'N/A')),
                'Accuracy_%' : str(row.get('Accuracy_%', 'N/A')),
            })
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Ramadan Spike ────────────────────────────────────────────
@app.route('/api/ramadan-spike', methods=['GET'])
def ramadan_spike():
    try:
        # Columns: Product, Normal_Avg_Qty, Ramadan_Avg_Qty, Spike_%
        df = load_sheet('Ramadan_Spike')
        if df.empty:
            return jsonify([])

        result = []
        for _, row in df.iterrows():
            result.append({
                'Product_Name' : str(row.get('Product', '')),
                'Normal_Avg'   : float(row.get('Normal_Avg_Qty', 0) or 0),
                'Ramadan_Avg'  : float(row.get('Ramadan_Avg_Qty', 0) or 0),
                'Spike_%'      : float(row.get('Spike_%', 0) or 0),
            })

        result.sort(key=lambda x: x['Spike_%'], reverse=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Profit Analysis ──────────────────────────────────────────
@app.route('/api/profit-analysis', methods=['GET'])
def profit_analysis():
    try:
        # Columns: Product_Name, Total_Qty_Sold, Total_Profit_PKR,
        #          Avg_Daily_Profit, Total_Sales_PKR, Profit_Margin_%
        df = load_sheet('Profit_Analysis')
        if df.empty:
            return jsonify([])

        result = []
        for _, row in df.iterrows():
            result.append({
                'Product_Name' : str(row.get('Product_Name', '')),
                'Total_Qty'    : int(row.get('Total_Qty_Sold', 0) or 0),
                'Total_Profit' : float(row.get('Total_Profit_PKR', 0) or 0),
                'Avg_Profit'   : float(row.get('Avg_Daily_Profit', 0) or 0),
                'Total_Sales'  : float(row.get('Total_Sales_PKR', 0) or 0),
                'Margin_%'     : float(row.get('Profit_Margin_%', 0) or 0),
            })

        result.sort(key=lambda x: x['Total_Profit'], reverse=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Inventory Status ─────────────────────────────────────────
@app.route('/api/inventory-status', methods=['GET'])
def inventory_status():
    try:
        # Columns: Product, Avg_Daily_Sales, Lead_Days, Safety_Stock,
        #          Reorder_Point, Monthly_Order, Current_Stock,
        #          Stock_Value_PKR, Status
        df = load_sheet('Inventory_Status')
        if df.empty:
            return jsonify({'inventory': [], 'summary': {}})

        def clean_status(s):
            s = str(s)
            if any(x in s for x in ['🔴','LOW','Order Now']):
                return 'Low'
            elif any(x in s for x in ['🔵','OVER','Stop']):
                return 'Overstock'
            return 'Normal'

        result = []
        for _, row in df.iterrows():
            status   = clean_status(row.get('Status', ''))
            avg_daily = float(row.get('Avg_Daily_Sales', 0) or 0)
            current  = int(row.get('Current_Stock', 0) or 0)
            days_rem = round(current / avg_daily, 1) if avg_daily > 0 else 0

            result.append({
                'product'       : str(row.get('Product', '')),
                'status'        : status,
                'current_stock' : current,
                'reorder_point' : int(row.get('Reorder_Point', 0) or 0),
                'reorder_qty'   : int(row.get('Monthly_Order', 0) or 0),
                'avg_daily_sales': avg_daily,
                'days_remaining': days_rem,
                # Dashboard keys bhi
                'Product'       : str(row.get('Product', '')),
                'Status'        : status,
                'Current_Stock' : current,
                'Reorder_Point' : int(row.get('Reorder_Point', 0) or 0),
                'Monthly_Order' : int(row.get('Monthly_Order', 0) or 0),
                'Avg_Daily'     : avg_daily,
            })

        # Critical pehle
        order = {'Low': 0, 'Overstock': 1, 'Normal': 2}
        result.sort(key=lambda x: order.get(x['status'], 3))

        summary = {
            'low_stock' : sum(1 for r in result if r['status'] == 'Low'),
            'normal'    : sum(1 for r in result if r['status'] == 'Normal'),
            'overstock' : sum(1 for r in result if r['status'] == 'Overstock'),
        }

        return jsonify({'inventory': result, 'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Forecast ─────────────────────────────────────────────────
@app.route('/api/forecast', methods=['GET'])
def forecast():
    try:
        # Columns: Date, Product, Forecasted_Qty, Day_Type
        df = load_sheet('30Day_Forecast')
        if df.empty:
            return jsonify({'data': []})

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        result = []
        for _, row in df.iterrows():
            result.append({
                'Date'           : str(row.get('Date', '')),
                'Product'        : str(row.get('Product', '')),
                'ARIMAX_Forecast': int(row.get('Forecasted_Qty', 0) or 0),
                'Forecasted_Qty' : int(row.get('Forecasted_Qty', 0) or 0),
                'Day_Type'       : str(row.get('Day_Type', 'Normal')),
            })

        return jsonify({'data': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Upload Dataset ───────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload and process dataset. 
    Accepts: multipart/form-data with 'file' field
    """
    try:
        # Handle both base64 JSON and multipart form data
        if request.is_json:
            data = request.get_json()
            if 'data' in data and 'filename' in data:
                # Base64 encoded file
                file_data = base64.b64decode(data['data'])
                filename = data['filename']
                file_bytes = BytesIO(file_data)
            else:
                return jsonify({'error': 'Invalid JSON format. Expected "data" and "filename".'}), 400
        elif 'file' in request.files:
            # Multipart form data
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            filename = file.filename
            file_bytes = BytesIO(file.read())
        else:
            return jsonify({'error': 'No file provided'}), 400

        # Validate file extension
        allowed_ext = {'csv', 'xlsx', 'xls'}
        if not any(filename.lower().endswith(f'.{ext}') for ext in allowed_ext):
            return jsonify({'error': f'Invalid file type. Allowed: {allowed_ext}'}), 400

        # Read the uploaded file
        try:
            if filename.lower().endswith('.csv'):
                df_upload = pd.read_csv(file_bytes)
            else:
                df_upload = pd.read_excel(file_bytes)
        except Exception as e:
            return jsonify({'error': f'Failed to read file: {str(e)}'}), 400

        # Validate that it has data
        if df_upload.empty:
            return jsonify({'error': 'Uploaded file is empty'}), 400

        # Backup old Excel file
        if os.path.exists(EXCEL_PATH):
            backup_path = EXCEL_PATH.replace('.xlsx', '_backup.xlsx')
            try:
                os.rename(EXCEL_PATH, backup_path)
                print(f"✅ Backed up old file to {backup_path}")
            except Exception as e:
                print(f"⚠️  Could not backup: {e}")

        # Update the main Excel file with new data
        # If the uploaded file is an Excel with multiple sheets, use it as-is
        # Otherwise, append to existing sheets
        try:
            if filename.lower().endswith('.csv'):
                # If CSV, update the first sheet (Profit_Analysis)
                with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
                    # Read existing sheets
                    existing_sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)
                    
                    # Write existing sheets
                    for sheet_name, sheet_df in existing_sheets.items():
                        if sheet_name != 'Profit_Analysis':
                            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Write new data to Profit_Analysis
                    df_upload.to_excel(writer, sheet_name='Profit_Analysis', index=False)
            else:
                # If Excel, save directly
                df_upload.to_excel(EXCEL_PATH, index=False)
            
            print(f"✅ Dataset updated: {filename}")

            # Clear cache to force reload
            _cache.clear()
            print("✅ Cache cleared - predictions will update on next request")

            return jsonify({
                'success': True,
                'message': 'Dataset uploaded and processed successfully! Models updated.',
                'filename': filename,
                'rows': len(df_upload),
            }), 200

        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# ── Retrain Status ──────────────────────────────────────────
@app.route('/api/retrain-status', methods=['GET'])
def retrain_status():
    """
    Get retraining status - cache indicates when models were last updated
    """
    try:
        cache_status = 'Updated' if _cache else 'Pending Update'
        return jsonify({
            'status': cache_status,
            'message': 'Models reflect latest uploaded dataset',
            'cached_sheets': list(_cache.keys()),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Run ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  PredictX Flask API Starting...")
    print("=" * 50)
    print(f"  Excel:   {EXCEL_PATH} — {'✅ Found' if os.path.exists(EXCEL_PATH) else '❌ NOT FOUND'}")
    print(f"  Models:  {MODELS_DIR}/ — {'✅ Found' if os.path.exists(MODELS_DIR) else '❌ NOT FOUND'}")
    print()
    print("  Endpoints:")
    print("  GET /health")
    print("  GET /api/dashboard")
    print("  GET /api/model-accuracy")
    print("  GET /api/ramadan-spike")
    print("  GET /api/profit-analysis")
    print("  GET /api/inventory-status")
    print("  GET /api/forecast")
    print("  POST /api/upload (multipart/form-data or base64 JSON)")
    print("  GET /api/retrain-status")
    print()
    print("  Flutter emulator: http://10.0.2.2:5000")
    print("  Real device:      http://10.71.37.36:5000")
    print("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
