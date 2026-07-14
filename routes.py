from flask import render_template, request, redirect, url_for, flash, send_file
from flask_login import login_user, logout_user, login_required, current_user
from app import app, users, predictions, model_performance, User
from tinydb import Query
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import ml_utils
import pandas as pd
import io
import os

UserQuery = Query()
PredQuery = Query()

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        mobile = request.form['mobile'].strip()
        address = request.form['address'].strip()
        password = request.form['password']
        confirm = request.form.get('confirm_password', '')

        if password != confirm:
            flash('Passwords do not match!', 'error')
            return render_template('signup.html')

        if users.search(UserQuery.email == email):
            flash('Email already registered!', 'error')
            return render_template('signup.html')

        new_id = len(users) + 1
        users.insert({
            'id': new_id,
            'name': name,
            'email': email,
            'mobile': mobile,
            'address': address,
            'password_hash': generate_password_hash(password),
            'is_admin': email.endswith('@admin.com'),  # Change or remove later
            'created_at': datetime.utcnow().isoformat()
        })
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        user_data = users.search(UserQuery.email == email)
        if user_data and check_password_hash(user_data[0]['password_hash'], password):
            user_obj = User(user_data[0])
            login_user(user_obj)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    df = ml_utils.load_and_preprocess_data()
    stats = {
        'total_records': len(df),
        'data_sources': df['Data_Source'].nunique(),
        'scheduler_types': df['Scheduler_Type'].nunique(),
        'avg_cpu': f"{df['CPU_Utilization (%)'].mean():.2f}%"
    }
    return render_template('home.html', stats=stats)

@app.route('/eda')
@login_required
def eda():
    df = ml_utils.load_and_preprocess_data()
    plots = ml_utils.generate_eda_plots(df)
    summary_stats = df.describe().to_html(classes='table table-sm table-striped')
    return render_template('eda.html', plots=plots, summary_stats=summary_stats)

@app.route('/classification/<target>')
@app.route('/classification/<target>/<classifier>')
@login_required
def classification(target, classifier='PAC'):
    target_map = {
        'job_priority': 'Job_Priority',
        'scheduler_type': 'Scheduler_Type',
        'resource_allocation': 'Resource_Allocation_Type'
    }
    target = target_map.get(target.lower().replace(' ', '_'), target)
    valid_targets = ['Job_Priority', 'Scheduler_Type', 'Resource_Allocation_Type']
    if target not in valid_targets:
        flash('Invalid target', 'error')
        return redirect(url_for('home'))

    results = ml_utils.evaluate_single_classifier(target, classifier)
    target_name = target.replace('_', ' ')
    return render_template('classification.html',
                         results=results, target=target, classifier=classifier, target_name=target_name)

@app.route('/performance')
@login_required
def performance():
    data = ml_utils.get_all_model_metrics()
    plot = ml_utils.plot_comparison(data)
    return render_template('performance.html', comparison_data=data, comparison_plot=plot)

@app.route('/model-performance')
@login_required
def model_performance():
    perfs = model_performance.all()
    perfs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    trend_plot = ml_utils.plot_accuracy_trends(perfs) if perfs else None
    return render_template('model_performance.html', performances=perfs, trend_plot=trend_plot)

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    single_result = None
    batch_results = None

    if request.method == 'POST':
        if 'error_rate' in request.form:
            # Single prediction
            try:
                input_data = [
                    float(request.form['error_rate']),
                    float(request.form['cpu_utilization']),
                    float(request.form['memory_consumption']),
                    float(request.form['task_execution_time']),
                    float(request.form['system_throughput']),
                    float(request.form['task_waiting_time']),
                    int(request.form['active_users']),
                    float(request.form['network_bandwidth']),
                    int(request.form['data_source'])
                ]
                results = {}
                for target in ['Job_Priority', 'Scheduler_Type', 'Resource_Allocation_Type']:
                    results[target] = ml_utils.predict_single(input_data, target)

                # Save prediction
                pred_id = len(predictions) + 1
                pred_record = {
                    'id': pred_id,
                    'user_id': int(current_user.id),
                    'prediction_type': 'single',
                    'input_data': json.dumps(input_data),
                    'created_at': datetime.utcnow().isoformat(),
                    'job_priority_pac': results['Job_Priority']['PAC']['prediction'],
                    'job_priority_nbc': results['Job_Priority']['NBC']['prediction'],
                    'job_priority_knn': results['Job_Priority']['KNN']['prediction'],
                    'job_priority_crngrim': results['Job_Priority']['CRN-GRIM']['prediction'],
                    'scheduler_type_pac': results['Scheduler_Type']['PAC']['prediction'],
                    'scheduler_type_nbc': results['Scheduler_Type']['NBC']['prediction'],
                    'scheduler_type_knn': results['Scheduler_Type']['KNN']['prediction'],
                    'scheduler_type_crngrim': results['Scheduler_Type']['CRN-GRIM']['prediction'],
                    'resource_allocation_pac': results['Resource_Allocation_Type']['PAC']['prediction'],
                    'resource_allocation_nbc': results['Resource_Allocation_Type']['NBC']['prediction'],
                    'resource_allocation_knn': results['Resource_Allocation_Type']['KNN']['prediction'],
                    'resource_allocation_crngrim': results['Resource_Allocation_Type']['CRN-GRIM']['prediction'],
                }
                predictions.insert(pred_record)
                single_result = results
            except Exception as e:
                flash(f'Prediction error: {str(e)}', 'error')

        elif 'batch_file' in request.files:
            file = request.files['batch_file']
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                batch_results = ml_utils.batch_predict(df)
                # Save batch entries (simplified)
                for i in range(min(len(batch_results), 50)):
                    predictions.insert({
                        'id': len(predictions) + 1 + i,
                        'user_id': int(current_user.id),
                        'prediction_type': 'batch',
                        'created_at': datetime.utcnow().isoformat()
                    })

    return render_template('prediction.html', single_result=single_result, batch_results=batch_results)

@app.route('/history')
@login_required
def history():
    page = int(request.args.get('page', 1))
    per_page = 20
    search = request.args.get('search', '').lower()
    ptype = request.args.get('type', '')

    all_preds = predictions.search(PredQuery.user_id == int(current_user.id))

    if ptype:
        all_preds = [p for p in all_preds if p.get('prediction_type') == ptype]
    if search:
        all_preds = [p for p in all_preds if any(search in str(v).lower() for v in p.values() if isinstance(v, str))]

    all_preds.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    total = len(all_preds)
    paginated = all_preds[(page-1)*per_page:page*per_page]

    # Convert string dates → datetime for .strftime()
    for p in paginated:
        dt_str = p.get('created_at', '')
        if isinstance(dt_str, str):
            try:
                p['created_at_dt'] = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            except:
                p['created_at_dt'] = datetime.utcnow()
        else:
            p['created_at_dt'] = datetime.utcnow()

    class Pagination:
        def __init__(self, items, page, per_page, total):
            self.items = items
            self.page = page
            self.per_page = per_page
            self.total = total
            self.pages = (total + per_page - 1) // per_page
            self.has_next = page < self.pages
            self.has_prev = page > 1
            self.prev_num = page - 1
            self.next_num = page + 1

    pagination = Pagination(paginated, page, per_page, total)

    return render_template('history.html',
                         predictions=pagination,
                         selected_type=ptype,
                         search_query=search)

@app.route('/export/predictions/csv')
@login_required
def export_predictions_csv():
    user_preds = predictions.search(PredQuery.user_id == int(current_user.id))
    if not user_preds:
        flash('No predictions to export', 'info')
        return redirect(url_for('history'))

    df = pd.DataFrame(user_preds)
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name=f"predictions_{current_user.name}_{datetime.now().strftime('%Y%m%d')}.csv",
        mimetype='text/csv'
    )

@app.route('/retrain', methods=['GET', 'POST'])
@login_required
def retrain():
    if request.method == 'POST':
        target = request.form['target']
        epochs = int(request.form.get('epochs', 20))
        batch_size = int(request.form.get('batch_size', 32))
        try:
            ml_utils.retrain_models(target, epochs=epochs, batch_size=batch_size)
            flash(f'Models retrained successfully for {target}!', 'success')
        except Exception as e:
            flash(f'Retraining failed: {str(e)}', 'error')
    return render_template('retrain.html')

@app.route('/admin')
@login_required
def admin():
    user_data = users.search(UserQuery.email == current_user.email)
    if not user_data or not user_data[0].get('is_admin', False):
        flash('Admin access denied', 'error')
        return redirect(url_for('home'))

    stats = {
        'total_users': len(users),
        'total_predictions': len(predictions),
        'total_trainings': len(model_performance),
        'recent_users': sorted(users.all(), key=lambda x: x.get('created_at', ''), reverse=True)[:5]
    }
    return render_template('admin.html', users=users.all(), stats=stats)