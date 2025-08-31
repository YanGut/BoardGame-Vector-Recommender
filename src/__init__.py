from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from flasgger import Flasgger

def create_app():
  app = Flask(__name__)
  
  CORS(app)
  Flasgger(app)
  
  app.config.from_object('src.config.config.Config')
  
  @app.route('/')
  def index():
    return jsonify({"message": "Welcome to the Recommendation API!"})
  
  from src.routes import api_bp as routes
  
  app.register_blueprint(routes, url_prefix='/api')
  
  return app