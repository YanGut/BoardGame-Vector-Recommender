from flask import Blueprint
from src.controllers.recommendation_controller import reco_bp

api_bp = Blueprint('api', __name__)

api_bp.register_blueprint(reco_bp, url_prefix='/recommendations')