from flask import Flask, render_template, request, redirect, session, url_for, jsonify, flash
from werkzeug.utils import secure_filename
import csv
import os
import uuid
from datetime import datetime
from functools import wraps
import pandas as pd
from pydub import AudioSegment
import io

app = Flask(__name__)
app.secret_key = 'cle_de_session_finale_et_securisee_v4'

# --- CONFIGURATION CENTRALE ---
DATA_DIR = 'data'
REFERENCE_AUDIO_FOLDERS = {"tmz": 'static/audios-tmz', "darija": 'static/audios-darija'}
RECORDINGS_AUDIO_FOLDERS = {"tmz": 'static/audios-tmz-final', "darija": 'static/audios-darija-final'}
SOURCE_FILES = {"tmz": os.path.join(DATA_DIR, 'sentences-tmz.csv'), "darija": os.path.join(DATA_DIR, 'sentences-darija.csv')}
OUTPUT_FILES = {"tmz": os.path.join(DATA_DIR, 'final_dataset-tmz.csv'), "darija": os.path.join(DATA_DIR, 'final_darija_dataset.csv')}
OUTPUT_FIELDNAMES = {
    "tmz": ['user_id', 'nom', 'prenom', 'age', 'genre', 'langue', 'tifinagh', 'latin', 'arabe', 'audio_filename', 'duration_sec', 'timestamp'],
    "darija": ['user_id', 'nom', 'prenom', 'age', 'genre', 'langue', 'latin', 'arabe', 'audio_filename', 'duration_sec', 'timestamp']
}
ADMIN_USERNAME = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASS', 'password123')

# --- INITIALISATION AU DÉMARRAGE ---
for folder in list(REFERENCE_AUDIO_FOLDERS.values()) + list(RECORDINGS_AUDIO_FOLDERS.values()):
    os.makedirs(folder, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

for lang, path in OUTPUT_FILES.items():
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES[lang])
            writer.writeheader()

# --- FONCTIONS UTILITAIRES ---
def load_sentences(langue):
    filepath = SOURCE_FILES.get(langue)
    if not filepath or not os.path.exists(filepath): return []
    try:
        with open(filepath, newline='', encoding='utf-8') as f: return [row for row in csv.DictReader(f) if row.get('latin', '').strip()]
    except Exception as e:
        app.logger.error(f"Error loading sentences for '{langue}': {e}")
        return []

def get_user_recorded_data(user_id, langue):
    output_file = OUTPUT_FILES.get(langue)
    last_latin = None
    recorded_latins_set = set()

    if not os.path.exists(output_file) or os.path.getsize(output_file) < 5:
        return last_latin, recorded_latins_set

    try:
        df = pd.read_csv(output_file, on_bad_lines='skip', low_memory=False)
        
        if df.empty or 'user_id' not in df.columns:
            return last_latin, recorded_latins_set
        
        df['user_id'] = df['user_id'].astype(str)
        user_recordings = df[df['user_id'] == user_id]
        
        if user_recordings.empty:
            return last_latin, recorded_latins_set

        recorded_latins_set = set(user_recordings['latin'].astype(str).dropna())
        
        user_recordings_copy = user_recordings.copy()
        
        user_recordings_copy['timestamp'] = pd.to_datetime(user_recordings_copy['timestamp'], errors='coerce')
        user_recordings_copy.dropna(subset=['timestamp'], inplace=True)
        
        if not user_recordings_copy.empty:
            last_recording = user_recordings_copy.sort_values(by='timestamp', ascending=False).iloc[0]
            last_latin = last_recording.get('latin')

    except Exception as e:
        app.logger.error(f"Error processing user data from {output_file}: {e}", exc_info=True)
    
    return last_latin, recorded_latins_set

def add_entry_to_dataset(user_id, user_info, langue, phrase_data, audio_filename, duration):
    output_file = OUTPUT_FILES[langue]
    fieldnames = OUTPUT_FIELDNAMES[langue]
    new_entry = {'user_id': user_id, 'nom': user_info['nom'], 'prenom': user_info['prenom'], 'age': user_info.get('age', 'N/A'), 'genre': user_info['genre'], 'langue': langue, 'tifinagh': phrase_data.get('tifinagh', ''), 'latin': phrase_data.get('latin', ''), 'arabe': phrase_data.get('arabe', ''), 'audio_filename': audio_filename, 'duration_sec': str(duration), 'timestamp': datetime.now().isoformat()}
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writerow(new_entry)

# --- ROUTES DE L'APPLICATION ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prenom = request.form['prenom'].lower().strip(); nom = request.form['nom'].lower().strip()
        session['user'] = {'nom': nom, 'prenom': prenom, 'age': request.form.get('age'), 'genre': request.form['genre']}
        session['user_id'] = f"{prenom}_8_{nom}"
        session['langue'] = request.form['langue']
        session.pop('current_sentence_index', None)
        return redirect(url_for('recorder'))
    if 'user' in session: return redirect(url_for('recorder'))
    return render_template('index.html')

@app.route('/recorder')
def recorder():
    if 'user' not in session: return redirect(url_for('index'))
    langue, user_id = session['langue'], session['user_id']
    all_sentences = load_sentences(langue)
    if not all_sentences: return render_template('merci.html', message=f"Aucune phrase à enregistrer pour {langue}.")
    
    last_latin, recorded_latins_set = get_user_recorded_data(user_id, langue)
    
    current_idx_from_session = session.get('current_sentence_index')
    if current_idx_from_session is None:
        target_idx = 0
        if last_latin:
            try:
                last_idx = next(i for i, s in enumerate(all_sentences) if s.get('latin') == last_latin )
                if last_idx < len(all_sentences) - 1:
                    target_idx = last_idx + 1
                else:
                    return render_template('merci.html', message="Félicitations ! Vous avez terminé toutes les phrases.")
            except StopIteration: target_idx = 0
    else: target_idx = int(current_idx_from_session)
    
    session['current_sentence_index'] = target_idx
    phrase_to_display = all_sentences[target_idx]
    is_phrase_recorded_by_user = phrase_to_display.get('latin') in recorded_latins_set
    
    return render_template('recorder.html', phrase=phrase_to_display, current_phrase_number=target_idx + 1, total_phrases=len(all_sentences), is_phrase_recorded=is_phrase_recorded_by_user)

@app.route('/next', methods=['POST'])
def next_phrase():
    if 'user' not in session: return redirect(url_for('index'))
    num_sentences = len(load_sentences(session['langue']))
    current_index = session.get('current_sentence_index', 0)
    if current_index < num_sentences - 1: session['current_sentence_index'] = current_index + 1
    return redirect(url_for('recorder'))

@app.route('/previous', methods=['POST'])
def previous_phrase():
    if 'user' not in session: return redirect(url_for('index'))
    current_index = session.get('current_sentence_index', 0)
    if current_index > 0: session['current_sentence_index'] = current_index - 1
    return redirect(url_for('recorder'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session: return jsonify({'success': False, 'message': 'Non autorisé'}), 401
    
    user_info, user_id, langue = session['user'], session['user_id'], session['langue']
    file = request.files.get('audio_data')
    if not file: return jsonify({'success': False, 'message': 'Aucun fichier audio'}), 400
    
    duration_float = float(request.form.get('duration', '0'))
    phrase_data = {'tifinagh': request.form.get('tifinagh', ''), 'latin': request.form.get('latin', ''), 'arabe': request.form.get('arabe', '')}
    if not phrase_data['latin']: return jsonify({'success': False, 'message': 'Donnée "latin" manquante'}), 400
    
    target_audio_folder = RECORDINGS_AUDIO_FOLDERS[langue]
    final_filename = f"audio_{user_id}_{langue}_{uuid.uuid4().hex[:8]}.wav"
    final_filepath = os.path.join(target_audio_folder, secure_filename(final_filename))
    
    try:
        audio_stream = file.stream.read()
        sound = AudioSegment.from_file(io.BytesIO(audio_stream))
        sound.export(final_filepath, format="wav")
        app.logger.info(f"Successfully converted and saved audio to {final_filepath}")

        add_entry_to_dataset(user_id, user_info, langue, phrase_data, final_filename, duration_float)
            
        session.pop('current_sentence_index', None)
        flash("✅ Enregistrement réussi !", "success")
        return jsonify({'success': True, 'message': 'Enregistrement validé !'})
    except Exception as e:
        app.logger.error(f"Upload error during conversion or save: {e}", exc_info=True)
        if os.path.exists(final_filepath): os.remove(final_filepath)
        return jsonify({'success': False, 'message': f'Erreur serveur lors de la conversion audio: {e}'}), 500

# --- SECTION ADMIN ---
def format_duration(seconds_str):
    try:
        seconds = float(seconds_str)
    except (ValueError, TypeError):
        return "0s"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
    if minutes > 0:
        return f"{int(minutes)}m {int(secs)}s"
    return f"{int(secs)}s"

# ... (Le reste du code de l'admin reste identique car il fonctionne déjà sur les données combinées)
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'): return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(request.args.get('next') or url_for('admin_dashboard'))
        flash('Identifiants incorrects.', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    all_dfs = []
    for lang, filepath in OUTPUT_FILES.items():
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                df_lang = pd.read_csv(filepath)
                all_dfs.append(df_lang)
            except Exception as e:
                app.logger.error(f"Dashboard: Error reading {filepath}: {e}")
    
    if not all_dfs:
        flash("Aucune donnée d'enregistrement à afficher.", "warning")
        return render_template('admin_dashboard.html', stats={}, format_duration=format_duration)
        
    df = pd.concat(all_dfs, ignore_index=True)

    stats = {'total_recordings': 0, 'total_duration_sec': 0.0, 'gender_distribution': {}, 'user_stats': {}, 'recordings_per_day_labels': [], 'recordings_per_day_data': [], 'duration_per_user_labels': [], 'duration_per_user_data': [], 'duration_per_user_per_day_labels': [], 'duration_per_user_per_day_datasets': [], 'duration_per_day_data': []}
    try:
        if df.empty:
            flash("Les fichiers de données sont vides.", "warning")
            return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)

        df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['full_name'] = df['prenom'].astype(str).str.strip() + ' ' + df['nom'].astype(str).str.strip()
        df['full_name'] = df['full_name'].str.strip().replace('', 'Unknown User')

        stats['total_recordings'] = len(df)
        stats['total_duration_sec'] = df['duration_sec'].sum()
        
        gender_counts = df.drop_duplicates(subset=['full_name'])['genre'].fillna('Unknown').value_counts()
        stats['gender_distribution'] = (gender_counts / gender_counts.sum() * 100).round(1).to_dict() if not gender_counts.empty else {}

        user_grouped = df.groupby('full_name').agg(record_count=('audio_filename', 'count'), total_duration=('duration_sec', 'sum')).reset_index()
        for _, row in user_grouped.iterrows():
            user_name = row['full_name']
            if not user_name or user_name == 'Unknown User': continue
            stats['user_stats'][user_name] = {'record_count': row['record_count'], 'total_duration': row['total_duration'], 'daily_activity': {}}
            user_df = df[df['full_name'] == user_name].copy()
            if not user_df.empty:
                user_df.loc[:, 'date'] = user_df['timestamp'].dt.date
                daily_grouped = user_df.groupby('date').agg(daily_record_count=('audio_filename', 'count'), daily_total_duration=('duration_sec', 'sum')).reset_index()
                for _, daily_row in daily_grouped.iterrows():
                    stats['user_stats'][user_name]['daily_activity'][daily_row['date'].strftime('%Y-%m-%d')] = {'count': daily_row['daily_record_count'], 'duration': daily_row['daily_total_duration']}
        
        if not df.empty:
            df_for_daily_stats = df.copy()
            df_for_daily_stats.loc[:, 'date_only'] = df_for_daily_stats['timestamp'].dt.date
            recordings_per_day = df_for_daily_stats.groupby('date_only').size().sort_index()
            if not recordings_per_day.empty:
                stats['recordings_per_day_labels'] = [d.strftime('%Y-%m-%d') for d in recordings_per_day.index]
                stats['recordings_per_day_data'] = recordings_per_day.values.tolist()
            duration_per_day = df_for_daily_stats.groupby('date_only')['duration_sec'].sum().sort_index()
            if not duration_per_day.empty:
                stats['duration_per_day_data'] = [round(v / 60, 2) for v in duration_per_day.values]
        
        stats['duration_per_user_labels'] = [name for name in stats['user_stats'].keys() if name and name != 'Unknown User']
        stats['duration_per_user_data'] = [round(user_data['total_duration'] / 60, 2) for name, user_data in stats['user_stats'].items() if name and name != 'Unknown User']

        all_dates = sorted(list({date for user_data in stats['user_stats'].values() for date in user_data.get('daily_activity', {}).keys()}))
        stats['duration_per_user_per_day_labels'] = all_dates
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#F67019', '#00C49F']
        user_index = 0
        for full_name, user_data in stats['user_stats'].items():
            if not full_name or full_name == 'Unknown User': continue
            data_points = [round(user_data.get('daily_activity', {}).get(date, {}).get('duration', 0) / 60, 2) for date in all_dates]
            stats['duration_per_user_per_day_datasets'].append({'label': full_name, 'data': data_points, 'borderColor': colors[user_index % len(colors)], 'backgroundColor': colors[user_index % len(colors)], 'fill': False, 'tension': 0.1})
            user_index += 1
    except Exception as e:
        app.logger.error(f"Dashboard generation error: {e}", exc_info=True)
        flash(f"Erreur lors de la génération des statistiques: {e}", "danger")
    return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)