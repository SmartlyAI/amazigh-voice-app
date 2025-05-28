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
app.secret_key = 'super_secret_key_pour_la_securite' 


HF_TOKEN = os.environ.get("HF_TOKEN", "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") 

HF_REPO_ID = "amazigh-voice-collection" 

# Attempt to login to Hugging Face Hub at startup
try:
    login(token=HF_TOKEN, add_to_git_credential=False) 
    app.logger.info("Successfully logged in to Hugging Face Hub.")
except Exception as e:
    app.logger.error(f"Failed to log in to Hugging Face Hub: {e}. Hugging Face push will likely fail.")


AUDIO_FOLDER = 'static/audios'
CSV_INPUT = 'data/sentences.csv'
CSV_OUTPUT = 'final_dataset.csv'

ADMIN_USERNAME = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASS', 'password123')


os.makedirs(AUDIO_FOLDER, exist_ok=True)
if not os.path.exists(CSV_OUTPUT):
    with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['nom', 'prenom', 'age', 'genre', 'tifinagh', 'latin', 'arabe' ,'audio_filename', 'duration_sec', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Ensure sentences.csv exists 
if not os.path.exists(CSV_INPUT):
    os.makedirs('data', exist_ok=True)
    with open(CSV_INPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['tifinagh', 'latin', 'arabe', 'recorded', 'default_audio'])
        writer.writeheader()
        writer.writerow({'tifinagh': 'ⴰⵣⵓⵍ', 'latin': 'azul', 'arabe': 'مرحبا', 'recorded': 'false', 'default_audio': ''})
        writer.writerow({'tifinagh': 'ⵜⴰⵎⴰⵣⵉⵖⵜ', 'latin': 'tamazight', 'arabe': 'الأمازيغية', 'recorded': 'false', 'default_audio': ''})
        writer.writerow({'tifinagh': 'ⴰⵎⴰⵡⴰⵍ', 'latin': 'amawal', 'arabe': 'معجم', 'recorded': 'false', 'default_audio': ''})

def load_sentences():
    try:
        with open(CSV_INPUT, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        os.makedirs('data', exist_ok=True)
        with open(CSV_INPUT, 'w', newline='', encoding='utf-8') as f_create:
            writer = csv.DictWriter(f_create, fieldnames=['tifinagh', 'latin','arabe', 'recorded','default_audio'])
            writer.writeheader()
        return []


def save_sentences(sentences):
    with open(CSV_INPUT, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['tifinagh', 'latin','arabe', 'recorded','default_audio']
        # Basic check for presence of keys
        if sentences and not all(key in sentences[0] for key in ['tifinagh', 'latin', 'recorded']):
             app.logger.warning("Attempting to save sentences with potentially missing keys. Ensure all expected keys are present.")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sentences)

def mark_sentence_recorded(tifinagh_to_mark, latin_to_mark):
    sentences = load_sentences()
    made_change = False
    for row in sentences:
        if row['tifinagh'] == tifinagh_to_mark and row['latin'] == latin_to_mark:
            if row.get('recorded', 'false').lower() != 'true':
                row['recorded'] = 'true'
                made_change = True
            break 
    if made_change:
        save_sentences(sentences)

def update_dataset(tifinagh, latin, arabe, filename, duration): 
    rows = []
    user = session['user']
    entry_updated = False
    
    
    fieldnames = ['nom', 'prenom', 'age', 'genre', 'tifinagh', 'latin','arabe', 'audio_filename', 'duration_sec', 'timestamp']

    if os.path.exists(CSV_OUTPUT) and os.path.getsize(CSV_OUTPUT) > 0:
        with open(CSV_OUTPUT, 'r', newline='', encoding='utf-8') as f_read:
            reader = csv.DictReader(f_read)
            current_fieldnames = reader.fieldnames
            if current_fieldnames and not all(fn in current_fieldnames for fn in ['tifinagh', 'latin', 'nom', 'prenom']):
                app.logger.warning(f"CSV_OUTPUT might have unexpected fieldnames. Expected core fields, Got: {current_fieldnames}")

            for row in reader:
                # Normalize row to contain all expected fieldnames, with None for missing ones
                normalized_row = {fn: row.get(fn) for fn in fieldnames}
                if (normalized_row['tifinagh'] == tifinagh and 
                    normalized_row['latin'] == latin and 
                    normalized_row['nom'] == user['nom'].lower() and 
                    normalized_row['prenom'] == user['prenom'].lower()):
                    # Update existing entry
                    normalized_row['arabe'] = arabe 
                    normalized_row['audio_filename'] = filename
                    normalized_row['duration_sec'] = str(duration) 
                    normalized_row['timestamp'] = datetime.now().isoformat()
                    entry_updated = True
                rows.append(normalized_row)

    if not entry_updated:
        rows.append({
            'nom': user['nom'].lower(),
            'prenom': user['prenom'].lower(),
            'age': str(user.get('age','N/A')), 
            'genre': user['genre'],
            'tifinagh': tifinagh,
            'latin': latin,
            'arabe': arabe, 
            'audio_filename': filename,
            'duration_sec': str(duration), 
            'timestamp': datetime.now().isoformat()
        })
    
    with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f_write:
        writer = csv.DictWriter(f_write, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['user'] = {
            'nom': request.form['nom'],
            'prenom': request.form['prenom'],
            'age': request.form.get('age'),
            'genre': request.form['genre']
        }
        session.pop('index', None) 
        session['fresh_login'] = True 
        return redirect(url_for('recorder'))

    if 'user' in session: 
        return redirect(url_for('recorder'))

    return render_template('index.html')

@app.route('/recorder')
def recorder():
    if 'user' not in session:
        return redirect(url_for('index'))

    sentences = load_sentences()
    if not sentences:
        return render_template('merci.html', message="Aucune phrase à enregistrer n'est disponible. Veuillez contacter l'administrateur.")

    current_idx_from_session = session.get('index')
    is_fresh_login_or_post_upload = session.pop('fresh_login', False) 

    target_idx = -1

    if is_fresh_login_or_post_upload or current_idx_from_session is None or \
       not (0 <= current_idx_from_session < len(sentences)):
        for i, s in enumerate(sentences):
            if s.get('recorded', 'false').lower() == 'false':
                target_idx = i
                break
        if target_idx == -1:
            return render_template('merci.html', message="✅ Toutes les phrases ont été marquées comme enregistrées globalement. Merci pour votre participation !")
    else:
        target_idx = current_idx_from_session
    
    session['index'] = target_idx 
    phrase_to_display = sentences[target_idx]

    return render_template(
        'recorder.html',
        phrase=phrase_to_display,
        recorded=phrase_to_display.get('recorded', 'false').lower(),
        current_phrase_number=target_idx + 1,
        total_phrases=len(sentences)
    )


@app.route('/next', methods=['POST'])
def next_phrase():
    if 'user' not in session:
        return redirect(url_for('index'))
        
    sentences = load_sentences()
    if not sentences:
        flash("Aucune phrase disponible.", "warning")
        return redirect(url_for('recorder'))

    current_index = session.get('index', 0)
    num_sentences = len(sentences)

    # Try to find the next unrecorded sentence starting from current_index + 1
    next_unrecorded_idx = -1
    for i in range(num_sentences):
        # Check current_index + 1, then +2, etc., wrapping around
        candidate_idx = (current_index + 1 + i) % num_sentences 
        if sentences[candidate_idx].get('recorded', 'false').lower() == 'false':
            next_unrecorded_idx = candidate_idx
            break
    
    if next_unrecorded_idx != -1:
        session['index'] = next_unrecorded_idx
    elif current_index < num_sentences -1 : # If all are recorded, but we are not at the end, advance
         session['index'] = current_index + 1
    else: # All sentences seem recorded, or we are at the end of already recorded ones
        # If all are truly recorded globally, fresh_login will trigger "merci"
        # Check again if ALL are recorded globally
        all_globally_recorded = True
        first_ever_unrecorded_idx = -1
        for i, s in enumerate(sentences):
            if s.get('recorded', 'false').lower() == 'false':
                all_globally_recorded = False
                if first_ever_unrecorded_idx == -1:
                    first_ever_unrecorded_idx = i
                

        if not all_globally_recorded and first_ever_unrecorded_idx != -1:
            session['index'] = first_ever_unrecorded_idx # Go to the first unrecorded one
        else: # All are recorded, show "merci"
            session.pop('index', None)
            session['fresh_login'] = True # 
            
    return redirect(url_for('recorder'))


@app.route('/previous', methods=['POST'])
def previous_phrase():
    if 'user' not in session:
        return redirect(url_for('index'))

    sentences = load_sentences() 
    if not sentences:
        flash("Aucune phrase disponible.", "warning")
        return redirect(url_for('recorder'))

    current_index = session.get('index', 0)

    # Try to find the previous unrecorded sentence
    prev_unrecorded_idx = -1
    for i in range(len(sentences)):
        candidate_idx = (current_index - 1 - i + len(sentences)) % len(sentences)
        if sentences[candidate_idx].get('recorded', 'false').lower() == 'false':
            prev_unrecorded_idx = candidate_idx
            break
            
    if prev_unrecorded_idx != -1:
        session['index'] = prev_unrecorded_idx
    elif current_index > 0: # If all previous are recorded, but we are not at the beginning
        session['index'] = current_index -1
    # else: index remains current_index (or 0 if it was 0), effectively stopping at first or first unrecorded
        
    return redirect(url_for('recorder'))


@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Non autorisé'}), 401

    file = request.files.get('audio_data')
    if not file:
        return jsonify({'success': False, 'message': 'Aucun fichier audio fourni'}), 400
        
    duration_str = request.form.get('duration', '0')
    try:
        duration_float = float(duration_str)
    except ValueError:
        app.logger.warning(f"Invalid duration format received: {duration_str}. Defaulting to 0.")
        duration_float = 0.0

    tifinagh = request.form.get('tifinagh') 
    latin = request.form.get('latin')     
    arabe = request.form.get('arabe') 

    if not tifinagh or not latin: 
        return jsonify({'success': False, 'message': 'Données de phrase manquantes (tifinagh/latin)'}), 400

    user_info = session['user']
    filename = f"audio_{user_info['nom'].lower()}_{user_info['prenom'].lower()}_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(AUDIO_FOLDER, secure_filename(filename))
    
    try:
        # 1. Save audio file locally
        file.save(filepath)
        
        # 2. Update local CSV dataset
        update_dataset(tifinagh, latin, arabe, filename, duration_float) # Pass float duration
        
        # 3. Mark sentence as recorded in the input CSV
        mark_sentence_recorded(tifinagh, latin) 
        
        # --- 4. Prepare data and push to Hugging Face Hub ---
        try:
            hf_data = {
                'nom': [user_info['nom'].lower()],
                'prenom': [user_info['prenom'].lower()],
                'age': [str(user_info.get('age', 'N/A'))],
                'genre': [user_info['genre']],
                'tifinagh': [tifinagh],
                'latin': [latin],
                'arabe': [arabe if arabe else ""], 
                'audio': [filepath],  # Path to the locally saved audio file
                'audio_filename': [filename], # The filename itself
                'duration_sec': [duration_float],
                'timestamp': [datetime.now().isoformat()]
            }

            # Define the features (schema) for the Hugging Face dataset
            # sampling_rate=16000 for Audio() will resample if original is different.
            hf_features = Features({
                'nom': Value('string'),
                'prenom': Value('string'),
                'age': Value('string'),
                'genre': Value('string'),
                'tifinagh': Value('string'),
                'latin': Value('string'),
                'arabe': Value('string'),
                'audio': Audio(sampling_rate=16000), 
                'audio_filename': Value('string'),
                'duration_sec': Value('float'),
                'timestamp': Value('string')
            })

            # Create a Dataset object for the single new entry
            dataset_entry = Dataset.from_dict(hf_data, features=hf_features)

            # Push to the Hugging Face Hub
            # This will create the repo if it doesn't exist, or append if it does (and schema matches)
            dataset_entry.push_to_hub(
                repo_id=HF_REPO_ID,
                # token=HF_TOKEN, # Not strictly needed if login() at startup was successful
                private=False 
            )
            app.logger.info(f"Successfully pushed data for {filename} to Hugging Face Hub: {HF_REPO_ID}")

        except Exception as hf_e:
            app.logger.error(f"Error pushing data to Hugging Face Hub: {hf_e}")
            # inform the user, but the local save was successful.
            flash("L'enregistrement local a réussi, mais l'envoi vers Hugging Face a échoué.", "warning")


        # --- End of Hugging Face Push ---

        # Find the next globally unrecorded sentence for the user to proceed to
        sentences = load_sentences()
        current_just_recorded_idx = session.get('index', 0) 

        next_unrecorded_idx_for_session = -1
        if sentences: # Check if sentences list is not empty
            # Start search from the sentence *after* the one just recorded, and wrap around
            for i in range(len(sentences)):
                candidate_idx = (current_just_recorded_idx + 1 + i) % len(sentences)
                if sentences[candidate_idx].get('recorded', 'false').lower() == 'false':
                    next_unrecorded_idx_for_session = candidate_idx
                    break
        
        if next_unrecorded_idx_for_session != -1:
            session['index'] = next_unrecorded_idx_for_session
        else: # All sentences are now recorded globally
            session.pop('index', None) 
            session['fresh_login'] = True # Tell recorder to show "all done" message

        return jsonify({'success': True, 'message': 'Enregistrement validé et envoyé avec succès !'})
    
    except Exception as e:
        app.logger.error(f"Erreur lors de l'upload (local save or general error): {e}")
        # Attempt to remove partially saved file if error occurred after save but before HF push
        if os.path.exists(filepath):
             try:
                 os.remove(filepath)
                 app.logger.info(f"Removed partially saved file due to error: {filepath}")
             except Exception as rm_e:
                 app.logger.error(f"Error removing file {filepath} after an upload error: {rm_e}")
        return jsonify({'success': False, 'message': f'Erreur serveur: {str(e)}'}), 500

# --- Admin Section  ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Connexion réussie!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('admin_dashboard'))
        else:
            flash('Identifiants incorrects.', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('admin_login'))

def format_duration(seconds_str):
    try:
        # Handle both float and string representations of seconds
        if isinstance(seconds_str, (int, float)):
            seconds = float(seconds_str)
        elif isinstance(seconds_str, str):
             seconds = float(seconds_str)
        else: # Fallback for unexpected types
            return "0s"
    except (ValueError, TypeError):
        return "0s" 

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60) # Use int for display
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    stats = {
        'total_recordings': 0,
        'total_duration_sec': 0.0,
        'gender_distribution': {},
        'user_stats': {},
        'recordings_per_day_labels': [],
        'recordings_per_day_data': [],
        'duration_per_user_labels': [],
        'duration_per_user_data': [],
        'duration_per_user_per_day_labels': [],
        'duration_per_user_per_day_datasets': [],
        'duration_per_day_data': [] 
    }

    if not os.path.exists(CSV_OUTPUT) or os.path.getsize(CSV_OUTPUT) == 0:
        flash("Le fichier de données est vide ou n'existe pas.", "warning")
        return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)

    try:
        df = pd.read_csv(CSV_OUTPUT)
        if df.empty:
            flash("Le fichier de données est vide.", "warning")
            return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)

        if 'arabe' not in df.columns:
            df['arabe'] = "" 

        df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'].notna()]


        df['full_name'] = df['prenom'].astype(str).fillna('').str.strip() + ' ' + df['nom'].astype(str).fillna('').str.strip()
        df['full_name'] = df['full_name'].str.strip().replace('', 'Unknown User')


        stats['total_recordings'] = len(df)
        stats['total_duration_sec'] = df['duration_sec'].sum()
        
        unique_users_gender = df.drop_duplicates(subset=['full_name'])['genre'].fillna('Unknown')
        gender_counts = unique_users_gender.value_counts()
        if not gender_counts.empty:
             stats['gender_distribution'] = (gender_counts / gender_counts.sum() * 100).round(1).to_dict()
        else:
            stats['gender_distribution'] = {}


        user_grouped = df.groupby('full_name').agg(
            record_count=('audio_filename', 'count'),
            total_duration=('duration_sec', 'sum')
        ).reset_index()

        for _, row in user_grouped.iterrows():
            user_name = row['full_name']
            if not user_name or user_name == 'Unknown User': continue 

            stats['user_stats'][user_name] = {
                'record_count': row['record_count'],
                'total_duration': row['total_duration'],
                'daily_activity': {}
            }

            user_df = df[df['full_name'] == user_name].copy() # Explicit copy
            if not user_df.empty and 'timestamp' in user_df.columns:
                user_df.loc[:, 'date'] = user_df['timestamp'].dt.date 
                daily_grouped = user_df.groupby('date').agg(
                    daily_record_count=('audio_filename', 'count'),
                    daily_total_duration=('duration_sec', 'sum')
                ).reset_index()
                for _, daily_row in daily_grouped.iterrows():
                    stats['user_stats'][user_name]['daily_activity'][daily_row['date'].strftime('%Y-%m-%d')] = {
                        'count': daily_row['daily_record_count'],
                        'duration': daily_row['daily_total_duration']
                    }
        
        if not df.empty and 'timestamp' in df.columns:
            # Make a copy before adding 'date_only' to avoid SettingWithCopyWarning on original df slice
            df_for_daily_stats = df.copy()
            df_for_daily_stats.loc[:, 'date_only'] = df_for_daily_stats['timestamp'].dt.date 
            
            recordings_per_day = df_for_daily_stats.groupby('date_only').size().sort_index()
            if not recordings_per_day.empty:
                stats['recordings_per_day_labels'] = [d.strftime('%Y-%m-%d') for d in recordings_per_day.index]
                stats['recordings_per_day_data'] = recordings_per_day.values.tolist()

            duration_per_day = df_for_daily_stats.groupby('date_only')['duration_sec'].sum().sort_index()
            if not duration_per_day.empty:
                # Convert duration to minutes for the chart
                stats['duration_per_day_data'] = [round(v / 60, 2) for v in duration_per_day.values] 
        
        stats['duration_per_user_labels'] = [name for name in stats['user_stats'].keys() if name and name != 'Unknown User'] 
        stats['duration_per_user_data'] = [
            round(user_data['total_duration'] / 60, 2) # in minutes
            for name, user_data in stats['user_stats'].items() if name and name != 'Unknown User'
        ]

        all_dates_set = set()
        for user_name, user_data in stats['user_stats'].items():
            if user_name and user_name != 'Unknown User': 
                all_dates_set.update(user_data.get('daily_activity', {}).keys())
        all_dates = sorted(list(all_dates_set)) 

        stats['duration_per_user_per_day_labels'] = all_dates

        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#F67019', '#00C49F', '#845EC2', '#D0C78F', '#2A6B99']
        stats['duration_per_user_per_day_datasets'] = []

        user_index = 0
        for full_name, user_data in stats['user_stats'].items():
            if not full_name or full_name == 'Unknown User': continue 

            daily_activity = user_data.get('daily_activity', {})
            data_points = [round(daily_activity.get(date, {}).get('duration', 0) / 60, 2) for date in all_dates] # in minutes
            stats['duration_per_user_per_day_datasets'].append({
                'label': full_name,
                'data': data_points,
                'borderColor': colors[user_index % len(colors)],
                'backgroundColor': colors[user_index % len(colors)], 
                'fill': False,
                'tension': 0.1
            })
            user_index += 1


    except pd.errors.EmptyDataError:
        flash("Le fichier de données CSV_OUTPUT est vide après tentative de lecture.", "warning")
    except Exception as e:
        app.logger.error(f"Erreur lors de la génération du dashboard: {e}", exc_info=True) # exc_info for traceback
        flash(f"Erreur lors de la génération des statistiques: {str(e)}", "danger")

    return render_template('admin_dashboard.html', stats=stats, format_duration=format_duration)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)