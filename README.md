# Amazigh Voice Collection App

Une application web Flask pour collecter des enregistrements audio de phrases en amazigh, destinée à la création de jeux de données pour l'entraînement de modèles TTS/STT.

##  Fonctionnalités

- Navigation entre les phrases à enregistrer
- Enregistrement et sauvegarde des fichiers audio
- Réenregistrement possible d'une phrase
- Indicateur visuel pour les phrases déjà enregistrées
- Page admin protégée (statistiques par utilisateur)


## ⚙️ Installation

#### 1. Clonez le dépôt :

```bash
git clone https://github.com/votre-utilisateur/amazigh-voice-app.git
cd amazigh-voice-app
```
#### 2. Installez les dépendances :

```bash
pip install datasets huggingface_hub pyarrow soundfile
```

#### 3.Lancez l'application :

```bash
python app.py
```
