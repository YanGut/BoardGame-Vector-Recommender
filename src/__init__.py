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
    CriarGruposResponse,
    MesaExistenteDTO,
    AssignPlayerRequest,
)
from src.dtos.recommendation_dtos import HybridRecommendationRequest

def create_app():
  app = Flask(__name__)
  
  CORS(app)

  # Generate OpenAPI schemas from Pydantic models with Swagger-friendly refs
  def _schema(model):
      return model.model_json_schema(ref_template="#/definitions/{model}")

  pydantic_schemas = {
      "PreferenciasDTO": _schema(PreferenciasDTO),
      "JogadorDTO": _schema(JogadorDTO),
      "RestricoesDTO": _schema(RestricoesDTO),
      "CriarGruposRequest": _schema(CriarGruposRequest),
      "JogoRecomendadoDTO": _schema(JogoRecomendadoDTO),
      "PerfilMesaDTO": _schema(PerfilMesaDTO),
      "MesaDTO": _schema(MesaDTO),
      "CriarGruposResponse": _schema(CriarGruposResponse),
      "MesaExistenteDTO": _schema(MesaExistenteDTO),
      "AssignPlayerRequest": _schema(AssignPlayerRequest),
      "HybridRecommendationRequest": _schema(HybridRecommendationRequest),
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
