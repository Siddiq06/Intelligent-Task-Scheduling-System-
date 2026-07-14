from flask import Flask
from flask_login import LoginManager, UserMixin
from tinydb import TinyDB, Query
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "super-secret-key-2025")

# TinyDB
db = TinyDB('tinydb_db.json')
users = db.table('users')
predictions = db.table('predictions')
model_performance = db.table('model_performance')

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = "info"

class User(UserMixin):
    def __init__(self, data):
        self.id = str(data['id'])
        self.name = data['name']
        self.email = data['email']
        self.mobile = data.get('mobile', '')
        self.address = data.get('address', '')
        self.is_admin = data.get('is_admin', False)

@login_manager.user_loader
def load_user(user_id):
    UserQuery = Query()
    result = users.search(UserQuery.id == int(user_id))
    return User(result[0]) if result else None

# Import routes after app is defined
import routes