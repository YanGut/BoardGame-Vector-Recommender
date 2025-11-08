from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from src.services.recommendation_service import recommendation_service_instance
from src.services.group_service import group_service_instance
from src.dtos.group_dtos import CriarGruposRequest, CriarGruposResponse
from src.dtos.recommendation_dtos import HybridRecommendationRequest

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

@reco_bp.route("/groups", methods=["POST"])
def create_groups():
    """
    Create player groups for an event.
    ---
    tags:
      - Groups
    parameters:
      - in: body
        name: body
        required: true
        schema:
          id: CriarGruposRequest
          properties:
            restricoes:
              type: object
              description: Optional constraints for group creation.
              properties:
                tamanhoMinimoMesa:
                  type: integer
                  example: 3
                tamanhoMaximoMesa:
                  type: integer
                  example: 5
                dbscanEps:
                  type: number
                  format: float
                  description: Cosine-distance threshold for DBSCAN clustering.
                  example: 0.35
                dbscanMinSamples:
                  type: integer
                  description: Minimum players per cluster for DBSCAN.
                  example: 2
            jogadores:
              type: array
              items:
                type: object
                properties:
                  idUsuario:
                    type: integer
                    example: 101
                  nome:
                    type: string
                    example: "Alice"
                  nivelExperiencia:
                    type: string
                    example: "iniciante"
                  preferencias:
                    type: object
                    properties:
                      mecanicasFavoritas:
                        type: array
                        items:
                          type: string
                        example: ["eurogame"]
                      temasFavoritos:
                        type: array
                        items:
                          type: string
                        example: ["fantasia"]
    responses:
      200:
        description: Successfully created groups.
        schema:
          id: CriarGruposResponse
          properties:
            mesas:
              type: array
              items:
                type: object
                properties:
                  mesaId:
                    type: integer
                    example: 1
                  jogadores:
                    type: array
                    items:
                      type: object
                      properties:
                        idUsuario:
                          type: integer
                          example: 101
                        nome:
                          type: string
                          example: "Alice"
                        nivelExperiencia:
                          type: string
                          example: "iniciante"
                        preferencias:
                          type: object
                          properties:
                            mecanicasFavoritas:
                              type: array
                              items:
                                type: string
                              example: ["eurogame"]
                            temasFavoritos:
                              type: array
                              items:
                                type: string
                              example: ["fantasia"]
                  perfilMesa:
                    type: object
                    properties:
                      nivelPredominante:
                        type: string
                        example: "intermediario"
                      mecanicasPredominantes:
                        type: array
                        items:
                          type: string
                        example: ["eurogame"]
                      temasPredominantes:
                        type: array
                        items:
                          type: string
                        example: ["fantasia"]
                  jogosRecomendados:
                    type: array
                    items:
                      type: object
                      properties:
                        idJogo:
                          type: integer
                          example: 101
                        nome:
                          type: string
                          example: "Catan"
                        similaridade:
                          type: number
                          format: float
                          example: 0.85
                        thumbnail:
                          type: string
                          example: "https://example.com/catan.jpg"
      400:
        description: Invalid request data.
    """
    try:
        data = CriarGruposRequest.parse_raw(request.data)
        response = group_service_instance.create_groups(data)
        return jsonify(response.dict())
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
            top_k:
              type: integer
              description: The number of recommendations to return.
          example:
            query: "A fun party game for 4-8 players"
            top_k: 5
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

@reco_bp.route('/recommend/hybrid', methods=['POST'])
def recommend_games_hybrid_route():
    """
    Get hybrid board game recommendations with re-ranking.
    ---
    tags:
      - Recommendations
    parameters:
      - in: body
        name: body
        required: true
        schema:
          id: HybridRecommendationRequest
        examples:
          default:
            value:
              query: "A fun party game for 4-8 players"
              top_k: 10
              candidate_pool_size: 100
              semantic_weight: 0.6
              popularity_weight: 0.4
    responses:
      200:
        description: A list of re-ranked recommended games.
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
        description: Invalid request data.
    """
    try:
        req_data = HybridRecommendationRequest.parse_raw(request.data)
        
        recommendations = recommendation_service_instance.recommend_games_hybrid(
            query_text=req_data.query,
            top_k=req_data.top_k,
            candidate_pool_size=req_data.candidate_pool_size,
            semantic_weight=req_data.semantic_weight,
            popularity_weight=req_data.popularity_weight
        )
        
        if recommendations is None:
            return jsonify({"error": "Falha ao obter recomendações híbridas"}), 500
            
        return jsonify({
            "query": req_data.query,
            "recommendations": recommendations
        })
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
