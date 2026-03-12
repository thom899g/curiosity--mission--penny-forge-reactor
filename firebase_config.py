"""
Firebase configuration and initialization module.
Handles secure connection to Firebase services with error recovery.
"""
import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, auth
from firebase_admin.exceptions import FirebaseError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FirebaseConfig:
    """Configuration container for Firebase services."""
    project_id: str
    database_url: Optional[str] = None
    storage_bucket: Optional[str] = None

class FirebaseManager:
    """Manages Firebase connection lifecycle and provides service clients."""
    
    _initialized: bool = False
    _config: Optional[FirebaseConfig] = None
    
    def __init__(self, service_account_path: Optional[str] = None):
        """
        Initialize Firebase connection.
        
        Args:
            service_account_path: Path to service account JSON file.
                If None, uses GOOGLE_APPLICATION_CREDENTIALS environment variable.
        
        Raises:
            FirebaseError: If initialization fails
            ValueError: If no credentials are found
        """
        self.service_account_path = service_account_path
        self._app = None
        self._firestore = None
        
    def initialize(self) -> bool:
        """
        Initialize Firebase Admin SDK with error handling.
        
        Returns:
            bool: True if initialization successful
        """
        if self._initialized:
            logger.info("Firebase already initialized")
            return True
            
        try:
            # Check for credentials
            cred = None
            if self.service_account_path and os.path.exists(self.service_account_path):
                logger.info(f"Using service account from: {self.service_account_path}")
                cred = credentials.Certificate(self.service_account_path)
            elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                env_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                logger.info(f"Using service account from env: {env_path}")
                if os.path.exists(env_path):
                    cred = credentials.Certificate(env_path)
            
            if not cred:
                logger.error("No Firebase credentials found")
                return False
            
            # Initialize with configuration
            firebase_config = {}
            
            if not firebase_admin._apps:
                self._app = firebase_admin.initialize_app(cred, firebase_config)
                logger.info("Firebase Admin SDK initialized successfully")
            else:
                self._app = firebase_admin.get_app()
                logger.info("Using existing Firebase app")
            
            # Test connection
            self._firestore = firestore.client()
            test_doc = self._firestore.collection('_system_health').document('test')
            test_doc.set({'test_time': datetime.utcnow()})
            test_doc.delete()
            
            self._initialized = True
            logger.info("Firebase connection test passed")
            return True
            
        except FirebaseError as e:
            logger.error(f"Firebase initialization error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected initialization error: {str(e)}")
            return False
    
    @property
    def firestore(self):
        """Get Firestore client with lazy initialization."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Firebase not initialized")
        return self._firestore
    
    @property
    def auth_client(self):
        """Get Auth client."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Firebase not initialized")
        return auth
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check Firebase service health.
        
        Returns:
            Dict with health status and metrics
        """
        health = {
            'initialized': self._initialized,
            'timestamp': datetime.utcnow().isoformat(),
            'services': {}
        }
        
        if self._initialized:
            try:
                # Test Firestore
                start_time = datetime.utcnow()
                test_ref = self.firestore.collection('_health_check').document('ping')
                test_ref.set({'ping': start_time.isoformat()})
                test_ref.delete()
                latency = (datetime.utcnow() - start_time).total_seconds()
                
                health['services']['firestore'] = {
                    'status': 'healthy',
                    'latency_seconds': round(latency, 3)
                }
                
            except Exception as e:
                health['services']['firestore'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health

# Global instance
firebase_manager = FirebaseManager()

def get_firestore():
    """Helper function to get Firestore client."""
    return firebase_manager.firestore