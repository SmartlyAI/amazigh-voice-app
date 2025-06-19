from flask import Flask, render_template, request, redirect, session, url_for, jsonify, flash
from werkzeug.utils import secure_filename
import csv
import os
import uuid
from datetime import datetime
from functools import wraps
import pandas as pd
from datasets import Dataset, Features, Value, Audio
from huggingface_hub import login

app = Flask(__name__)
# IMPORTANT: Changez cette clé pour votre application en production
app.secret_key = 'cle_secrete_a_changer_absolument'

# --- CONFIGURATION CENTRALE ---
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# !! MODIFIEZ AVEC VOS PROPRES NOMS DE DÉPÔTS HUGGING FACE !!
HF_REPO_IDS = {
    "tmz": "votre-compte/amazigh-voice-collection-tmz",
    "darija": "votre-compte/amazigh-voice-collection-darija"
}

DATA_DIR = 'data'
AUDIO_FOLDER = 'static/audios'

SOURCE_FILES = {
    "tmz": os.path.join(DATA_DIR, 'sentences-tmz.csv'),
    "darija": os.path.join(DATA_DIR, 'sentences-darija.csv')
}
OUTPUT_FILES = {
    "tmz": os.path.join(DATA_DIR, 'final_dataset-tmz.csv'),
    "darija": os.path.join(DATA_DIR, 'final_dataset-darija.csv')
}

# MODIFIÉ: En-têtes spécifiques pour chaque langue de sortie
OUTPUT_FIELDNAMES = {
    "tmz": ['user_id', 'nom', 'prenom', 'age', 'genre', 'langue', 'tifinagh', 'latin', 'arabe', 'audio_filename', 'duration_sec', 'timestamp'],
    "darija": ['user_id', 'nom', 'prenom', 'age', 'genre', 'langue', 'latin', 'arabe', 'audio_filename', 'duration_sec', 'timestamp']
}

ADMIN_USERNAME = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASS', 'password123')

# --- INITIALISATION AU DÉMARRAGE ---
try:
    login(token=HF_TOKEN, add_to_git_credential=False)
    app.logger.info("Successfully logged in to Hugging Face Hub.")
except Exception as e:
    app.logger.error(f"Failed to log in to Hugging Face Hub: {e}. Push will fail.")

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialise les fichiers de sortie avec les en-têtes corrects s'ils n'existent pas
for lang, path in OUTPUT_FILES.items():
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            # Utilise les en-têtes spécifiques à la langue
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES[lang])
            writer.writeheader()

# --- FONCTIONS UTILITAIRES ---
def load_sentences(langue):
    filepath = SOURCE_FILES.get(langue)
    if not filepath or not os.path.exists(filepath): return []
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            return [row for row in csv.DictReader(f) if row.get('latin', '').strip()]
    except Exception as e:
        app.logger.error(f"Error loading sentences for '{langue}': {e}")
        return []

def get_user_recorded_latins(user_id, langue):
    output_file = OUTPUT_FILES.get(langue)
    recorded_latins = set()
    if not output_file or not os.path.exists(output_file): return recorded_latins
    try:
        df = pd.read_csv(output_file, on_bad_lines='skip')
        if not df.empty and 'user_id' in df.columns and 'latin' in df.columns:
            user_recordings = df[df['user_id'] == user_id]['latin'].dropna().unique()
            recorded_latins.update(user_recordings)
    except pd.errors.EmptyDataError: pass
    except Exception as e:
        app.logger.error(f"Error reading user records from {output_file}: {e}")
    return recorded_latins

# MODIFIÉ: La fonction utilise maintenant les en-têtes spécifiques à la langue
def add_entry_to_dataset(user_id, user_info, langue, phrase_data, audio_filename, duration):
    output_file = OUTPUT_FILES[langue]
    fieldnames = OUTPUT_FIELDNAMES[langue] # Utilise les en-têtes corrects
    
    new_entry = {
        'user_id': user_id, 'nom': user_info['nom'], 'prenom': user_info['prenom'],
        'age': user_info.get('age', 'N/A'), 'genre': user_info['genre'], 'langue': langue,
        'tifinagh': phrase_data.get('tifinagh', ''), 'latin': phrase_data.get('latin', ''),
        'arabe': phrase_data.get('arabe', ''), 'audio_filename': audio_filename,
        'duration_sec': str(duration), 'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        # DictWriter ignorera les clés non présentes dans `fieldnames` (ex: 'tifinagh' pour darija)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writerow(new_entry)

def format_duration(seconds_str):
    try: seconds = float(seconds_str)
    except (ValueError, TypeError): return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0: return f"{hours}h {minutes}m {secs}s"
    if minutes > 0: return f"{minutes}m {secs}s"
    return f"{secs}s"

# --- ROUTES DE L'APPLICATION ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prenom = request.form['prenom'].lower().strip()
        nom = request.form['nom'].lower().strip()
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
    
    user_recorded_latins = get_user_recorded_latins(user_id, langue)
    current_idx_from_session = session.get('current_sentence_index')
    target_idx = -1

    if current_idx_from_session is None:
        for i, s in enumerate(all_sentences):
            if s.get('latin') not in user_recorded_latins:
                target_idx = i
                break
        if target_idx == -1: return render_template('merci.html', message="Félicitations ! Vous avez enregistré toutes les phrases.")
    else: target_idx = current_idx_from_session
    
    session['current_sentence_index'] = target_idx
    phrase_to_display = all_sentences[target_idx]
    is_current_phrase_recorded = phrase_to_display.get('latin') in user_recorded_latins
    
    return render_template('recorder.html', phrase=phrase_to_display, is_recorded_by_user=is_current_phrase_recorded, current_phrase_number=target_idx + 1, total_phrases=len(all_sentences))

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
    
    filename = f"audio_{user_id}_{langue}_{uuid.uuid4().hex[:8]}"
    filepath = os.path.join(AUDIO_FOLDER, secure_filename(filename))
    
    try:
        file.save(filepath)
        add_entry_to_dataset(user_id, user_info, langue, phrase_data, filename, duration_float)
        
        try:
            hf_repo_id = HF_REPO_IDS[langue]
            # Schéma UNIFIÉ pour Hugging Face pour la simplicité
            hf_data = {'user_id': [user_id], 'nom': [user_info['nom']], 'prenom': [user_info['prenom']], 'age': [str(user_info.get('age', 'N/A'))], 'genre': [user_info['genre']], 'langue': [langue], 'tifinagh': [phrase_data['tifinagh']], 'latin': [phrase_data['latin']], 'arabe': [phrase_data['arabe']], 'audio': [filepath], 'audio_filename': [filename], 'duration_sec': [duration_float], 'timestamp': [datetime.now().isoformat()]}
            hf_features = Features({'user_id': Value('string'), 'nom': Value('string'), 'prenom': Value('string'), 'age': Value('string'), 'genre': Value('string'), 'langue': Value('string'), 'tifinagh': Value('string'), 'latin': Value('string'), 'arabe': Value('string'), 'audio': Audio(sampling_rate=16000), 'audio_filename': Value('string'), 'duration_sec': Value('float'), 'timestamp': Value('string')})
            Dataset.from_dict(hf_data, features=hf_features).push_to_hub(repo_id=hf_repo_id, private=False)
            app.logger.info(f"Pushed to HF repo: {hf_repo_id}")
        except Exception as hf_e:
            app.logger.error(f"Error pushing to HF Hub: {hf_e}")
            flash("Enregistrement local réussi, mais l'envoi vers Hugging Face a échoué.", "warning")
            
        session.pop('current_sentence_index', None)
        return jsonify({'success': True, 'message': 'Enregistrement validé !'})
    except Exception as e:
        app.logger.error(f"Upload error: {e}", exc_info=True)
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'success': False, 'message': f'Erreur serveur: {e}'}), 500

# --- SECTION ADMIN ---
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
    # MODIFIÉ: Charger et combiner les données des deux fichiers
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

    # Le reste du code est identique à votre version, il opère sur le DataFrame `df` combiné
    stats = {'total_recordings': 0, 'total_duration_sec': 0.0, 'gender_distribution': {}, 'user_stats': {}, 'recordings_per_day_labels': [], 'recordings_per_day_data': [], 'duration_per_user_labels': [], 'duration_per_user_data': [], 'duration_per_user_per_day_labels': [], 'duration_per_user_per_day_datasets': [], 'duration_per_day_data': []}
    try:
        if df.empty:
            flash("Les fichiers de données sont vides.", "warning")
            return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)

        df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'].notna()]
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
            df_for_daily_stats['date_only'] = df_for_daily_stats['timestamp'].dt.date
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