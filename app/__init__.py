# app/__init__.py

from flask import Flask, render_template, request, session, redirect, url_for, flash, abort
from .agent import get_ai_response
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv

# --- DATABASE AND APP SETUP ---
db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()

# --- MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False, default="New Conversation")
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    user = db.relationship('User', backref=db.backref('conversations', lazy=True, cascade="all, delete-orphan"))

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    conversation = db.relationship('Conversation', backref=db.backref('messages', lazy=True, cascade="all, delete-orphan"))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- APP FACTORY FUNCTION ---
def create_app():
    # ✅ 1. Make Flask aware of instance/ folder
    app = Flask(__name__, instance_relative_config=True)

    # ✅ 2. Load environment variables from .env
    load_dotenv()

    # ✅ 3. App configuration
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///instance/database.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ✅ 4. Ensure instance folder exists (SQLite needs it)
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    # ✅ 5. Initialize extensions
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'welcome'

    # ✅ 6. Create tables automatically
    with app.app_context():
        db.create_all()

    # --- ROUTES ---
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            latest_convo = db.session.execute(
                db.select(Conversation).filter_by(user_id=current_user.id).order_by(desc(Conversation.timestamp))
            ).scalars().first()
            if latest_convo:
                return redirect(url_for('chat', conversation_id=latest_convo.id))
            else:
                return redirect(url_for('new_chat'))
        return redirect(url_for('welcome'))

    @app.route('/healthz')
    def health_check():
        return "OK", 200

    @app.route('/chat/<int:conversation_id>', methods=['GET', 'POST'])
    @login_required
    def chat(conversation_id):
        conversation = db.session.get(Conversation, conversation_id)
        if not conversation or conversation.user_id != current_user.id:
            abort(403)
        if request.method == 'POST':
            query = request.form['query']
            if query:
                answer = get_ai_response(query)
                if not conversation.messages:
                    conversation.title = ' '.join(query.split()[:5])
                new_message = ChatMessage(conversation_id=conversation.id, query=query, answer=answer)
                db.session.add(new_message)
                db.session.commit()
                return redirect(url_for('chat', conversation_id=conversation.id))
        all_conversations = db.session.execute(
            db.select(Conversation).filter_by(user_id=current_user.id).order_by(desc(Conversation.timestamp))
        ).scalars().all()
        return render_template('chat.html', messages=conversation.messages,
                               all_conversations=all_conversations, current_convo_id=conversation.id)

    @app.route('/new_chat')
    @login_required
    def new_chat():
        new_convo = Conversation(user_id=current_user.id)
        db.session.add(new_convo)
        db.session.commit()
        return redirect(url_for('chat', conversation_id=new_convo.id))

    @app.route('/guest', methods=['GET', 'POST'])
    def guest_chat():
        if 'guest_history' not in session:
            session['guest_history'] = []
        if request.method == 'POST':
            query = request.form['query']
            if query:
                answer = get_ai_response(query)
                session['guest_history'].append({'query': query, 'answer': answer})
                session.modified = True
            return redirect(url_for('guest_chat'))
        return render_template('guest_chat.html', history=session['guest_history'])

    @app.route('/welcome')
    def welcome():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        return render_template('welcome.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        if request.method == 'POST':
            username, password = request.form['username'], request.form['password']
            remember = True if request.form.get('remember') else False
            user = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user, remember=remember)
                if 'guest_history' in session:
                    session.pop('guest_history')
                return redirect(url_for('index'))
            else:
                flash('Login Unsuccessful. Please check username and password.', 'danger')
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        if request.method == 'POST':
            username, password = request.form['username'], request.form['password']
            user_exists = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
            if user_exists:
                flash('Username already exists.', 'danger')
                return redirect(url_for('register'))
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! You can now log in.', 'success')
            return redirect(url_for('login'))
        return render_template('register.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        session.clear()
        flash('You have been successfully logged out.', 'success')
        return redirect(url_for('welcome'))

    return app
