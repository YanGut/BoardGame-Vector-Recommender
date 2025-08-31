from flask import Blueprint, request, jsonify
from src.services.recommendation_service import recommendation_service_instance

reco_bp = Blueprint('recommendations', __name__)

@reco_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health Check
    ---
    tags:
      - Health
    responses:
      200:
        description: Service is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            message:
              type: string
              example: Recommendation service is healthy
    """
    return jsonify({"status": "ok", "message": "Recommendation service is healthy"})

@reco_bp.route('/recommend', methods=['POST'])
def recommend_games_route():
    """
    Get board game recommendations
    ---
    tags:
      - Recommendations
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
              description: The text query to search for recommendations.
              example: "A fun party game for 4-8 players"
            top_k:
              type: integer
              description: The number of recommendations to return.
              default: 5
    responses:
      200:
        description: A list of recommended games.
        schema:
          type: object
          properties:
            query:
              type: string
            recommendations:
              type: array
              items:
                type: object
      400:
        description: Missing 'query' parameter.
    """
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
    """
    List all games with pagination
    ---
    tags:
      - Games
    parameters:
      - in: query
        name: page
        type: integer
        description: The page number to retrieve.
        default: 1
      - in: query
        name: per_page
        type: integer
        description: The number of games per page.
        default: 20
    responses:
      200:
        description: A paginated list of games.
        schema:
          type: object
          properties:
            page:
              type: integer
            per_page:
              type: integer
            total:
              type: integer
            games:
              type: array
              items:
                type: object
    """
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    result = recommendation_service_instance.list_all_games(page, per_page)
    return jsonify(result)

@reco_bp.route('/game/mysql/<int:game_mysql_id>', methods=['GET'])
def get_game_by_mysql_id_route(game_mysql_id):
    """
    Get a specific game by its MySQL ID
    ---
    tags:
      - Games
    parameters:
      - in: path
        name: game_mysql_id
        type: integer
        required: true
        description: The original MySQL ID of the game.
    responses:
      200:
        description: The details of the game.
        schema:
          type: object
      404:
        description: Game not found.
    """
    game_details = recommendation_service_instance.get_game_by_mysql_id(game_mysql_id)
    if game_details:
        return jsonify(game_details)
    return jsonify({"error": f"Jogo com ID MySQL {game_mysql_id} não encontrado"}), 404
