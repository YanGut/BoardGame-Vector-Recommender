from flask import Flask
from src.controllers.recommendation_controller import reco_bp
import os
from src import create_app

if __name__ == "__main__":
    app: Flask = create_app()
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )