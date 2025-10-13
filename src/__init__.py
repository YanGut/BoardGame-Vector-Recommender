from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from flasgger import Flasgger
from src.dtos.group_dtos import (
    PreferenciasDTO,
    JogadorDTO,
    RestricoesDTO,
    CriarGruposRequest,
    JogoRecomendadoDTO,
    PerfilMesaDTO,
    MesaDTO,
    CriarGruposResponse
)
from src.dtos.recommendation_dtos import HybridRecommendationRequest

def create_app():
  app = Flask(__name__)
  
  CORS(app)

  # Generate OpenAPI schemas from Pydantic models
  pydantic_schemas = {
      "PreferenciasDTO": PreferenciasDTO.model_json_schema(),
      "JogadorDTO": JogadorDTO.model_json_schema(),
      "RestricoesDTO": RestricoesDTO.model_json_schema(),
      "CriarGruposRequest": CriarGruposRequest.model_json_schema(),
      "JogoRecomendadoDTO": JogoRecomendadoDTO.model_json_schema(),
      "PerfilMesaDTO": PerfilMesaDTO.model_json_schema(),
      "MesaDTO": MesaDTO.model_json_schema(),
      "CriarGruposResponse": CriarGruposResponse.model_json_schema(),
      "HybridRecommendationRequest": HybridRecommendationRequest.model_json_schema(),
  }

  # Flasgger configuration
  swagger_template = {
      "swagger": "2.0",
      "info": {
          "title": "Board Game Vector Recommender API",
          "description": "API for recommending board games and managing player groups.",
          "version": "1.0.0"
      },
      "host": "localhost:5000",  # Adjust as needed
      "basePath": "/api",
      "schemes": [
          "http"
      ],
      "definitions": pydantic_schemas # Add Pydantic schemas here
  }

  Flasgger(app, template=swagger_template)
  
  app.config.from_object('src.config.config.Config')
  
  @app.route('/')
  def index():
    return jsonify({"message": "Welcome to the Recommendation API!"})
  
  from src.routes import api_bp as routes
  
  app.register_blueprint(routes, url_prefix='/api')

  return app