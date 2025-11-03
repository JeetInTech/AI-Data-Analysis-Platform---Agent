import os
import json
import pickle
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import zipfile
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False

class ArtifactStorageManager:
    """Advanced storage system for AI analytics sessions with version control and cloud sync."""
    
    def __init__(self, base_dir="ai_analytics_storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Storage directories
        self.sessions_dir = self.base_dir / "sessions"
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "datasets"
        self.visualizations_dir = self.base_dir / "visualizations"
        self.exports_dir = self.base_dir / "exports"
        self.cache_dir = self.base_dir / "cache"
        
        # Create directories
        for directory in [self.sessions_dir, self.models_dir, self.data_dir, 
                         self.visualizations_dir, self.exports_dir, self.cache_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.base_dir / "analytics_storage.db"
        self._initialize_database()
        
        # Storage statistics and gamification
        self.storage_stats = {
            'sessions_created': 0,
            'models_saved': 0,
            'data_processed': 0,
            'visualizations_saved': 0,
            'total_storage_mb': 0
        }
        
        # Achievement system
        self.storage_achievements = {
            'data_hoarder': {'threshold': 10, 'unlocked': False, 'description': 'Saved 10 datasets'},
            'model_collector': {'threshold': 5, 'unlocked': False, 'description': 'Saved 5 trained models'},
            'session_master': {'threshold': 15, 'unlocked': False, 'description': 'Created 15 sessions'},
            'viz_artist': {'threshold': 20, 'unlocked': False, 'description': 'Saved 20 visualizations'},
            'storage_guru': {'threshold': 100, 'unlocked': False, 'description': 'Used 100MB+ storage'}
        }
        
        # Load existing stats
        self._load_storage_stats()
        
    def _initialize_database(self):
        """Initialize SQLite database for metadata storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP,
                        last_modified TIMESTAMP,
                        session_name TEXT,
                        data_shape TEXT,
                        model_count INTEGER,
                        status TEXT,
                        tags TEXT,
                        file_path TEXT,
                        size_mb REAL,
                        version INTEGER DEFAULT 1
                    )
                ''')
                
                # Models table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        model_name TEXT,
                        model_type TEXT,
                        created_at TIMESTAMP,
                        performance_metrics TEXT,
                        file_path TEXT,
                        size_mb REAL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                ''')
                
                # Datasets table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS datasets (
                        dataset_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        dataset_name TEXT,
                        original_filename TEXT,
                        rows INTEGER,
                        columns INTEGER,
                        created_at TIMESTAMP,
                        file_path TEXT,
                        size_mb REAL,
                        data_hash TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                ''')
                
                # Visualizations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS visualizations (
                        viz_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        viz_name TEXT,
                        viz_type TEXT,
                        created_at TIMESTAMP,
                        file_path TEXT,
                        size_mb REAL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                ''')
                
                # Storage stats table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS storage_stats (
                        stat_name TEXT PRIMARY KEY,
                        stat_value REAL,
                        last_updated TIMESTAMP
                    )
                ''')
                
                conn.commit()
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def create_new_session(self, session_name=None):
        """Create a new analytics session."""
        try:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if session_name is None:
                session_name = f"Analytics Session {session_id}"
            
            # Create session directory
            session_path = self.sessions_dir / session_id
            session_path.mkdir(exist_ok=True)
            
            # Create subdirectories
            (session_path / "data").mkdir(exist_ok=True)
            (session_path / "models").mkdir(exist_ok=True)
            (session_path / "visualizations").mkdir(exist_ok=True)
            (session_path / "exports").mkdir(exist_ok=True)
            
            # Save session metadata
            session_metadata = {
                'session_id': session_id,
                'session_name': session_name,
                'created_at': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'status': 'active',
                'data_shape': None,
                'model_count': 0,
                'tags': [],
                'version': 1
            }
            
            metadata_path = session_path / "session_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(session_metadata, f, indent=2)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions 
                    (session_id, created_at, last_modified, session_name, status, file_path, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, datetime.now(), datetime.now(), session_name, 
                     'active', str(session_path), 1))
                conn.commit()
            
            # Update statistics
            self.storage_stats['sessions_created'] += 1
            self._save_storage_stats()
            self._check_achievements()
            
            return session_id
            
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def save_session(self, session_id, session_data):
        """Save complete session data including models, data, and metadata."""
        try:
            session_path = self.sessions_dir / session_id
            if not session_path.exists():
                print(f"Session {session_id} not found")
                return False
            
            # Save datasets
            if 'current_data' in session_data and session_data['current_data'] is not None:
                data_path = session_path / "data" / "original_data.pkl"
                self._save_dataframe(session_data['current_data'], data_path)
                
                # Calculate data hash for versioning
                data_hash = self._calculate_data_hash(session_data['current_data'])
                self._save_dataset_metadata(session_id, "original_data", 
                                          session_data['current_data'], data_path, data_hash)
            
            if 'cleaned_data' in session_data and session_data['cleaned_data'] is not None:
                data_path = session_path / "data" / "cleaned_data.pkl"
                self._save_dataframe(session_data['cleaned_data'], data_path)
                
                data_hash = self._calculate_data_hash(session_data['cleaned_data'])
                self._save_dataset_metadata(session_id, "cleaned_data", 
                                          session_data['cleaned_data'], data_path, data_hash)
            
            # Save models
            if 'models' in session_data and session_data['models']:
                self._save_models(session_id, session_data['models'])
            
            # Save meta-features
            if 'meta_features' in session_data:
                meta_path = session_path / "meta_features.json"
                with open(meta_path, 'w') as f:
                    json.dump(session_data['meta_features'], f, indent=2, default=str)
            
            # Update session metadata
            self._update_session_metadata(session_id, session_data)
            
            # Update statistics
            self._update_storage_statistics()
            self._check_achievements()
            
            return True
            
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def load_session(self, session_id):
        """Load complete session data."""
        try:
            session_path = self.sessions_dir / session_id
            if not session_path.exists():
                print(f"Session {session_id} not found")
                return None
            
            session_data = {}
            
            # Load datasets
            original_data_path = session_path / "data" / "original_data.pkl"
            if original_data_path.exists():
                session_data['current_data'] = self._load_dataframe(original_data_path)
            
            cleaned_data_path = session_path / "data" / "cleaned_data.pkl"
            if cleaned_data_path.exists():
                session_data['cleaned_data'] = self._load_dataframe(cleaned_data_path)
            
            # Load meta-features
            meta_path = session_path / "meta_features.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    session_data['meta_features'] = json.load(f)
            
            # Load models (metadata only - models loaded on demand)
            session_data['models'] = self._load_models_metadata(session_id)
            
            # Load session metadata
            metadata_path = session_path / "session_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    session_data['metadata'] = json.load(f)
            
            return session_data
            
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def save_model(self, session_id, model_name, model, performance_metrics=None):
        """Save a trained model with metadata."""
        try:
            session_path = self.sessions_dir / session_id
            if not session_path.exists():
                print(f"Session {session_id} not found")
                return False
            
            model_id = f"{session_id}_{model_name}_{datetime.now().strftime('%H%M%S')}"
            
            # Choose serialization method
            if JOBLIB_AVAILABLE:
                model_filename = f"{model_id}.joblib"
                model_path = session_path / "models" / model_filename
                joblib.dump(model, model_path)
            elif CLOUDPICKLE_AVAILABLE:
                model_filename = f"{model_id}.cloudpickle"
                model_path = session_path / "models" / model_filename
                with open(model_path, 'wb') as f:
                    cloudpickle.dump(model, f)
            else:
                model_filename = f"{model_id}.pkl"
                model_path = session_path / "models" / model_filename
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Calculate file size
            size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Save model metadata to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO models 
                    (model_id, session_id, model_name, model_type, created_at, 
                     performance_metrics, file_path, size_mb)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (model_id, session_id, model_name, str(type(model).__name__),
                     datetime.now(), json.dumps(performance_metrics) if performance_metrics else None,
                     str(model_path), size_mb))
                conn.commit()
            
            # Update statistics
            self.storage_stats['models_saved'] += 1
            self._save_storage_stats()
            self._check_achievements()
            
            return model_id
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    def load_model(self, model_id):
        """Load a saved model."""
        try:
            # Get model metadata from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT file_path FROM models WHERE model_id = ?', (model_id,))
                result = cursor.fetchone()
                
                if not result:
                    print(f"Model {model_id} not found")
                    return None
                
                model_path = Path(result[0])
            
            # Load model based on file extension
            if model_path.suffix == '.joblib' and JOBLIB_AVAILABLE:
                return joblib.load(model_path)
            elif model_path.suffix == '.cloudpickle' and CLOUDPICKLE_AVAILABLE:
                with open(model_path, 'rb') as f:
                    return cloudpickle.load(f)
            else:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def save_visualization(self, session_id, viz_name, fig, viz_type="matplotlib"):
        """Save a visualization (matplotlib figure or plotly figure)."""
        try:
            session_path = self.sessions_dir / session_id
            if not session_path.exists():
                print(f"Session {session_id} not found")
                return False
            
            viz_id = f"{session_id}_{viz_name}_{datetime.now().strftime('%H%M%S')}"
            viz_dir = session_path / "visualizations"
            
            if viz_type == "matplotlib":
                # Save as PNG and PDF
                png_path = viz_dir / f"{viz_id}.png"
                pdf_path = viz_dir / f"{viz_id}.pdf"
                
                fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                fig.savefig(pdf_path, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                
                primary_path = png_path
                
            elif viz_type == "plotly":
                # Save as HTML and PNG
                html_path = viz_dir / f"{viz_id}.html"
                png_path = viz_dir / f"{viz_id}.png"
                
                fig.write_html(str(html_path))
                try:
                    fig.write_image(str(png_path))
                except:
                    # If kaleido not available, skip PNG export
                    pass
                
                primary_path = html_path
            
            # Calculate file size
            size_mb = primary_path.stat().st_size / (1024 * 1024)
            
            # Save visualization metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO visualizations 
                    (viz_id, session_id, viz_name, viz_type, created_at, file_path, size_mb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (viz_id, session_id, viz_name, viz_type, datetime.now(),
                     str(primary_path), size_mb))
                conn.commit()
            
            # Update statistics
            self.storage_stats['visualizations_saved'] += 1
            self._save_storage_stats()
            self._check_achievements()
            
            return viz_id
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return None
    
    def export_session(self, session_id, export_format="zip", include_models=True, 
                      include_data=True, include_visualizations=True):
        """Export session in various formats."""
        try:
            session_path = self.sessions_dir / session_id
            if not session_path.exists():
                print(f"Session {session_id} not found")
                return None
            
            export_filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if export_format == "zip":
                export_path = self.exports_dir / f"{export_filename}.zip"
                
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add session metadata
                    metadata_path = session_path / "session_metadata.json"
                    if metadata_path.exists():
                        zipf.write(metadata_path, "session_metadata.json")
                    
                    # Add data files
                    if include_data:
                        data_dir = session_path / "data"
                        if data_dir.exists():
                            for file_path in data_dir.rglob("*"):
                                if file_path.is_file():
                                    arcname = f"data/{file_path.relative_to(data_dir)}"
                                    zipf.write(file_path, arcname)
                    
                    # Add models
                    if include_models:
                        models_dir = session_path / "models"
                        if models_dir.exists():
                            for file_path in models_dir.rglob("*"):
                                if file_path.is_file():
                                    arcname = f"models/{file_path.relative_to(models_dir)}"
                                    zipf.write(file_path, arcname)
                    
                    # Add visualizations
                    if include_visualizations:
                        viz_dir = session_path / "visualizations"
                        if viz_dir.exists():
                            for file_path in viz_dir.rglob("*"):
                                if file_path.is_file():
                                    arcname = f"visualizations/{file_path.relative_to(viz_dir)}"
                                    zipf.write(file_path, arcname)
                    
                    # Add database export
                    self._export_session_database(session_id, zipf)
            
            elif export_format == "json":
                # Export as structured JSON
                export_path = self.exports_dir / f"{export_filename}.json"
                session_data = self.load_session(session_id)
                
                # Convert DataFrames to JSON-serializable format
                export_data = {}
                for key, value in session_data.items():
                    if isinstance(value, pd.DataFrame):
                        export_data[key] = {
                            'type': 'dataframe',
                            'data': value.to_json(orient='split'),
                            'shape': value.shape
                        }
                    else:
                        export_data[key] = value
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            return str(export_path)
            
        except Exception as e:
            print(f"Error exporting session: {e}")
            return None
    
    def import_session(self, import_path, new_session_name=None):
        """Import session from exported file."""
        try:
            import_path = Path(import_path)
            
            if import_path.suffix == '.zip':
                return self._import_zip_session(import_path, new_session_name)
            elif import_path.suffix == '.json':
                return self._import_json_session(import_path, new_session_name)
            else:
                print(f"Unsupported import format: {import_path.suffix}")
                return None
                
        except Exception as e:
            print(f"Error importing session: {e}")
            return None
    
    def get_session_list(self):
        """Get list of all sessions with metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT session_id, session_name, created_at, last_modified, 
                           status, data_shape, model_count, size_mb
                    FROM sessions 
                    ORDER BY last_modified DESC
                '''
                df = pd.read_sql_query(query, conn)
                return df.to_dict('records')
        except Exception as e:
            print(f"Error getting session list: {e}")
            return []
    
    def get_storage_statistics(self):
        """Get comprehensive storage statistics with achievements."""
        try:
            self._update_storage_statistics()
            
            # Get database statistics
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Session stats
                cursor.execute('SELECT COUNT(*) FROM sessions')
                total_sessions = cursor.fetchone()[0]
                
                # Model stats
                cursor.execute('SELECT COUNT(*), COALESCE(SUM(size_mb), 0) FROM models')
                model_count, models_size = cursor.fetchone()
                
                # Dataset stats
                cursor.execute('SELECT COUNT(*), COALESCE(SUM(size_mb), 0) FROM datasets')
                dataset_count, datasets_size = cursor.fetchone()
                
                # Visualization stats
                cursor.execute('SELECT COUNT(*), COALESCE(SUM(size_mb), 0) FROM visualizations')
                viz_count, viz_size = cursor.fetchone()
            
            # Calculate total storage
            total_storage_mb = (models_size or 0) + (datasets_size or 0) + (viz_size or 0)
            
            # Update achievements based on current stats
            self._update_achievement_progress(total_sessions, model_count, 
                                            dataset_count, viz_count, total_storage_mb)
            
            return {
                'sessions': {
                    'total': total_sessions,
                    'active': len([s for s in self.get_session_list() if s['status'] == 'active'])
                },
                'models': {
                    'total': model_count or 0,
                    'size_mb': models_size or 0
                },
                'datasets': {
                    'total': dataset_count or 0,
                    'size_mb': datasets_size or 0
                },
                'visualizations': {
                    'total': viz_count or 0,
                    'size_mb': viz_size or 0
                },
                'storage': {
                    'total_mb': total_storage_mb,
                    'total_gb': total_storage_mb / 1024,
                    'base_path': str(self.base_dir)
                },
                'achievements': self.storage_achievements,
                'recently_unlocked': self._get_recently_unlocked_achievements()
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def cleanup_storage(self, days_old=30, archive=True):
        """Clean up old sessions and optimize storage."""
        try:
            cleanup_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            # Find old sessions
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT session_id, file_path FROM sessions 
                    WHERE last_modified < datetime(?, 'unixepoch')
                ''', (cleanup_date,))
                old_sessions = cursor.fetchall()
            
            cleaned_count = 0
            archived_count = 0
            
            for session_id, file_path in old_sessions:
                session_path = Path(file_path)
                
                if archive:
                    # Archive before deletion
                    archive_path = self._archive_session(session_id)
                    if archive_path:
                        archived_count += 1
                
                # Remove session directory
                if session_path.exists():
                    shutil.rmtree(session_path)
                    cleaned_count += 1
                
                # Remove from database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
                    cursor.execute('DELETE FROM models WHERE session_id = ?', (session_id,))
                    cursor.execute('DELETE FROM datasets WHERE session_id = ?', (session_id,))
                    cursor.execute('DELETE FROM visualizations WHERE session_id = ?', (session_id,))
                    conn.commit()
            
            return {
                'sessions_cleaned': cleaned_count,
                'sessions_archived': archived_count,
                'storage_freed_mb': self._calculate_freed_storage()
            }
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return None
    
    # Helper methods
    def _save_dataframe(self, df, path):
        """Save DataFrame efficiently."""
        try:
            # Use pickle for speed, parquet for efficiency if available
            if hasattr(pd, 'to_parquet'):
                parquet_path = path.with_suffix('.parquet')
                df.to_parquet(parquet_path)
            else:
                df.to_pickle(path)
        except:
            # Fallback to pickle
            df.to_pickle(path)
    
    def _load_dataframe(self, path):
        """Load DataFrame efficiently."""
        try:
            if path.suffix == '.parquet':
                return pd.read_parquet(path)
            else:
                return pd.read_pickle(path)
        except:
            # Try alternative paths
            for ext in ['.parquet', '.pkl']:
                alt_path = path.with_suffix(ext)
                if alt_path.exists():
                    if ext == '.parquet':
                        return pd.read_parquet(alt_path)
                    else:
                        return pd.read_pickle(alt_path)
            return None
    
    def _calculate_data_hash(self, df):
        """Calculate hash of DataFrame for versioning."""
        try:
            # Create hash from DataFrame content
            df_string = df.to_string()
            return hashlib.md5(df_string.encode()).hexdigest()
        except:
            return None
    
    def _save_dataset_metadata(self, session_id, dataset_name, df, file_path, data_hash):
        """Save dataset metadata to database."""
        try:
            dataset_id = f"{session_id}_{dataset_name}"
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO datasets 
                    (dataset_id, session_id, dataset_name, rows, columns, 
                     created_at, file_path, size_mb, data_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (dataset_id, session_id, dataset_name, len(df), len(df.columns),
                     datetime.now(), str(file_path), size_mb, data_hash))
                conn.commit()
        except Exception as e:
            print(f"Error saving dataset metadata: {e}")
    
    def _save_models(self, session_id, models_dict):
        """Save multiple models."""
        for model_name, model_info in models_dict.items():
            if 'model' in model_info:
                performance = model_info.get('performance', {})
                self.save_model(session_id, model_name, model_info['model'], performance)
    
    def _load_models_metadata(self, session_id):
        """Load models metadata (not the actual models)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT model_id, model_name, model_type, performance_metrics, size_mb
                    FROM models WHERE session_id = ?
                ''', (session_id,))
                
                models_metadata = {}
                for row in cursor.fetchall():
                    model_id, model_name, model_type, performance_str, size_mb = row
                    performance = json.loads(performance_str) if performance_str else {}
                    
                    models_metadata[model_name] = {
                        'model_id': model_id,
                        'model_type': model_type,
                        'performance': performance,
                        'size_mb': size_mb,
                        'loaded': False  # Model not loaded yet
                    }
                
                return models_metadata
        except Exception as e:
            print(f"Error loading models metadata: {e}")
            return {}
    
    def _update_session_metadata(self, session_id, session_data):
        """Update session metadata."""
        try:
            # Calculate current session stats
            data_shape = None
            if 'cleaned_data' in session_data and session_data['cleaned_data'] is not None:
                data_shape = str(session_data['cleaned_data'].shape)
            elif 'current_data' in session_data and session_data['current_data'] is not None:
                data_shape = str(session_data['current_data'].shape)
            
            model_count = len(session_data.get('models', {}))
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions 
                    SET last_modified = ?, data_shape = ?, model_count = ?
                    WHERE session_id = ?
                ''', (datetime.now(), data_shape, model_count, session_id))
                conn.commit()
                
        except Exception as e:
            print(f"Error updating session metadata: {e}")
    
    def _load_storage_stats(self):
        """Load storage statistics from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT stat_name, stat_value FROM storage_stats')
                
                for stat_name, stat_value in cursor.fetchall():
                    if stat_name in self.storage_stats:
                        self.storage_stats[stat_name] = stat_value
        except:
            pass  # Use default values if table doesn't exist
    
    def _save_storage_stats(self):
        """Save storage statistics to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for stat_name, stat_value in self.storage_stats.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO storage_stats 
                        (stat_name, stat_value, last_updated)
                        VALUES (?, ?, ?)
                    ''', (stat_name, stat_value, datetime.now()))
                
                conn.commit()
        except Exception as e:
            print(f"Error saving storage stats: {e}")
    
    def _update_storage_statistics(self):
        """Update storage statistics from filesystem."""
        try:
            total_size = 0
            for directory in [self.sessions_dir, self.models_dir, self.data_dir, 
                             self.visualizations_dir]:
                if directory.exists():
                    for file_path in directory.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            
            self.storage_stats['total_storage_mb'] = total_size / (1024 * 1024)
            self._save_storage_stats()
        except Exception as e:
            print(f"Error updating storage statistics: {e}")
    
    def _check_achievements(self):
        """Check and unlock achievements."""
        stats = self.storage_stats
        achievements = self.storage_achievements
        
        # Data Hoarder
        if stats['data_processed'] >= achievements['data_hoarder']['threshold']:
            achievements['data_hoarder']['unlocked'] = True
        
        # Model Collector
        if stats['models_saved'] >= achievements['model_collector']['threshold']:
            achievements['model_collector']['unlocked'] = True
        
        # Session Master
        if stats['sessions_created'] >= achievements['session_master']['threshold']:
            achievements['session_master']['unlocked'] = True
        
        # Viz Artist
        if stats['visualizations_saved'] >= achievements['viz_artist']['threshold']:
            achievements['viz_artist']['unlocked'] = True
        
        # Storage Guru
        if stats['total_storage_mb'] >= achievements['storage_guru']['threshold']:
            achievements['storage_guru']['unlocked'] = True
    
    def _update_achievement_progress(self, sessions, models, datasets, vizs, storage_mb):
        """Update achievement progress based on current database stats."""
        self.storage_stats.update({
            'sessions_created': sessions,
            'models_saved': models,
            'data_processed': datasets,
            'visualizations_saved': vizs,
            'total_storage_mb': storage_mb
        })
        self._check_achievements()
    
    def _get_recently_unlocked_achievements(self):
        """Get recently unlocked achievements."""
        return [
            {
                'name': name.replace('_', ' ').title(),
                'description': achievement['description'],
                'icon': self._get_achievement_icon(name)
            }
            for name, achievement in self.storage_achievements.items()
            if achievement['unlocked']
        ]
    
    def _get_achievement_icon(self, achievement_name):
        """Get icon for achievement."""
        icons = {
            'data_hoarder': '💾',
            'model_collector': '🤖',
            'session_master': '👑',
            'viz_artist': '🎨',
            'storage_guru': '🏗️'
        }
        return icons.get(achievement_name, '🏆')
    
    def _export_session_database(self, session_id, zipf):
        """Export session-specific database entries to zip."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export session data as JSON
                session_db_data = {}
                
                for table in ['sessions', 'models', 'datasets', 'visualizations']:
                    cursor = conn.cursor()
                    cursor.execute(f'SELECT * FROM {table} WHERE session_id = ?', (session_id,))
                    rows = cursor.fetchall()
                    
                    # Get column names
                    cursor.execute(f'PRAGMA table_info({table})')
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    # Convert to list of dictionaries
                    session_db_data[table] = [dict(zip(columns, row)) for row in rows]
                
                # Add to zip as JSON
                import io
                json_str = json.dumps(session_db_data, indent=2, default=str)
                zipf.writestr("database_export.json", json_str)
                
        except Exception as e:
            print(f"Error exporting database: {e}")
    
    def _import_zip_session(self, zip_path, new_session_name):
        """Import session from zip file."""
        try:
            # Create new session
            new_session_id = self.create_new_session(new_session_name)
            new_session_path = self.sessions_dir / new_session_id
            
            # Extract zip contents
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(new_session_path)
            
            # Import database entries if present
            db_export_path = new_session_path / "database_export.json"
            if db_export_path.exists():
                self._import_database_entries(new_session_id, db_export_path)
            
            return new_session_id
            
        except Exception as e:
            print(f"Error importing zip session: {e}")
            return None
    
    def _import_json_session(self, json_path, new_session_name):
        """Import session from JSON file."""
        try:
            with open(json_path, 'r') as f:
                session_data = json.load(f)
            
            # Create new session
            new_session_id = self.create_new_session(new_session_name)
            
            # Convert and save data
            processed_data = {}
            for key, value in session_data.items():
                if isinstance(value, dict) and value.get('type') == 'dataframe':
                    # Reconstruct DataFrame
                    df_data = json.loads(value['data'])
                    processed_data[key] = pd.DataFrame(df_data['data'], 
                                                     columns=df_data['columns'],
                                                     index=df_data['index'])
                else:
                    processed_data[key] = value
            
            # Save the session
            self.save_session(new_session_id, processed_data)
            
            return new_session_id
            
        except Exception as e:
            print(f"Error importing JSON session: {e}")
            return None
    
    def _import_database_entries(self, new_session_id, db_export_path):
        """Import database entries from exported JSON."""
        try:
            with open(db_export_path, 'r') as f:
                db_data = json.load(f)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update session_id in all entries and insert
                for table_name, rows in db_data.items():
                    if rows and table_name != 'sessions':  # Skip sessions table
                        for row in rows:
                            row['session_id'] = new_session_id  # Update to new session ID
                            
                            # Create INSERT statement
                            columns = list(row.keys())
                            placeholders = ', '.join(['?' for _ in columns])
                            query = f'INSERT OR REPLACE INTO {table_name} ({", ".join(columns)}) VALUES ({placeholders})'
                            
                            cursor.execute(query, list(row.values()))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error importing database entries: {e}")
    
    def _archive_session(self, session_id):
        """Archive a session before deletion."""
        try:
            archive_dir = self.base_dir / "archives"
            archive_dir.mkdir(exist_ok=True)
            
            export_path = self.export_session(session_id, export_format="zip")
            if export_path:
                archive_path = archive_dir / Path(export_path).name
                shutil.move(export_path, archive_path)
                return archive_path
            
            return None
        except Exception as e:
            print(f"Error archiving session: {e}")
            return None
    
    def _calculate_freed_storage(self):
        """Calculate storage freed during cleanup (placeholder)."""
        # This would calculate actual freed storage
        return 0.0