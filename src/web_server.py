from flask import Flask, jsonify
import numpy as np
import logging as flask_logging


def create_app(current_state, reward_history, episode_rewards):
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Configure Flask to suppress access logs for cleaner training logs
    flask_log = flask_logging.getLogger('werkzeug')
    flask_log.setLevel(flask_logging.ERROR)  # Only show errors, not access logs
    
    @app.route('/')
    def index():
        with open('index.html', 'r') as f:
            return f.read()

    @app.route('/styles.css')
    def styles():
        with open('styles.css', 'r') as f:
            return f.read(), 200, {'Content-Type': 'text/css'}

    @app.route('/visualization.js')
    def visualization():
        with open('visualization.js', 'r') as f:
            return f.read(), 200, {'Content-Type': 'application/javascript'}

    @app.route('/state')
    def get_state():
        return jsonify(current_state)

    @app.route('/history')
    def get_history():
        return jsonify({
            'rewards': list(reward_history),
            'avg_reward': float(np.mean(list(episode_rewards)) if episode_rewards else 0)
        })
    
    return app
