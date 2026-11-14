from flask import Flask

from .config import Config
from .services.embedder import SemanticScorer
from .services.ranker import Ranker


def create_app(config_object: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)

    scorer = SemanticScorer(
        model_name=app.config["EMBEDDING_MODEL"],
        use_transformers=app.config["USE_TRANSFORMERS"],
    )
    app.extensions["ranker"] = Ranker(scorer, app.config["WEIGHTS"])

    from .routes import bp

    app.register_blueprint(bp)
    return app
