from flask import Blueprint, request, jsonify
from src.services.recommendation_service import recommendation_service_instance

reco_bp = Blueprint('recommendations', __name__)

@reco_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Recommendation service is healthy"})

@reco_bp.route('/recommend', methods=['POST'])
def recommend_games_route():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Parâmetro 'query' ausente"}), 400
    
    query = data['query']
    top_k = data.get('top_k', 5)
    
    recommendations = recommendation_service_instance.recommend_games(query, top_k)
    
    if recommendations is None: # Ou se o serviço retornar um erro específico
        return jsonify({"error": "Falha ao obter recomendações"}), 500
        
    return jsonify({
        "query": query,
        "recommendations": recommendations
    })

@reco_bp.route('/games', methods=['GET'])
def list_games_route():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    result = recommendation_service_instance.list_all_games(page, per_page)
    return jsonify(result)

@reco_bp.route('/game/mysql/<int:game_mysql_id>', methods=['GET'])
def get_game_by_mysql_id_route(game_mysql_id):
    game_details = recommendation_service_instance.get_game_by_mysql_id(game_mysql_id)
    if game_details:
        return jsonify(game_details)
    return jsonify({"error": f"Jogo com ID MySQL {game_mysql_id} não encontrado"}), 404