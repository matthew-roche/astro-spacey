from flask import Flask, jsonify, request
from flask_smorest import Api, Blueprint
from spacey_api.blueprint.model_blueprint import model_blueprint
import os

app = Flask(__name__)
app.config["API_TITLE"] = "SpaceY Question Answering - API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.2"
app.config["OPENAPI_URL_PREFIX"] = "/api"
app.config["OPENAPI_REDOC_PATH"] = "/docs/redoc"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/docs" 
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.18.2/"  # Swagger UI CDN

api = Api(app)

server_blueprint = Blueprint("Server", __name__, url_prefix="/health", description="api operations")

@server_blueprint.route("")
@server_blueprint.response(200, content_type="application/json")
def health_check():
    return jsonify({"status": "healthy"})

# cors
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*" # add known FE request origin or leave as is
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Demo-Secret"
    return response

@server_blueprint.after_request
def server_cors(response):
    return cors_headers(response)

@model_blueprint.after_request
def model_cors(response):
    return cors_headers(response)

# limit access
# comment for localhost dev, or use a prod config
@app.before_request
def gate():
    if request.method != "OPTIONS" and request.path != "/health":
        if request.headers.get("X-Demo-Secret") != os.getenv('API_SEC'): # refer step 7
            return jsonify({"error":"forbidden"}), 403

api.register_blueprint(server_blueprint)
api.register_blueprint(model_blueprint)

if __name__ == "__main__":
    app.run(debug=True)