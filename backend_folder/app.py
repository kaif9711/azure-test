from flask import Flask, jsonify
from routes.auth import auth_bp


def create_app():
    app = Flask(__name__)

    # Basic configuration
    app.config['SECRET_KEY'] = 'your-secret-key-here'

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")

    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'fraud-detection-api',
            'version': '1.0.0'
        })

    # Root endpoint
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Fraudulent Claim Detection API',
            'version': '1.0.0',
            'status': 'running'
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
