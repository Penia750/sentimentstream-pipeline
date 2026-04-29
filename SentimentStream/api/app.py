"""
SentimentStream · API REST Flask
Capas 3 y 4: MongoDB → Flask

Endpoints:
  GET  /sentiments          → todos los registros predichos
  GET  /stats               → conteo por clase + accuracy global
  POST /predict             → predicción de texto nuevo (reglas léxicas)
"""

from flask import Flask, jsonify, request
from pymongo import MongoClient
from datetime import datetime
import os
import re

app = Flask(__name__)

# ------------------------------------------------------------------
# Conexión a MongoDB (el host 'mongo' es el nombre del servicio
# en docker-compose, desde fuera usa localhost:27017)
# ------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/")
client     = MongoClient(MONGO_URI)
db         = client["sentimentstream"]
col        = db["predicciones"]


# ------------------------------------------------------------------
# Utilidad: clasificador léxico simple para /predict
# (el modelo Spark no corre dentro del contenedor Flask;
#  para producción se exportaría como microservicio separado)
# ------------------------------------------------------------------
POSITIVOS = [
    "amazing", "great", "good", "excellent", "recommend", "highly",
    "best", "love", "perfect", "awesome", "support", "helpful",
    "bueno", "excelente", "genial", "perfecto", "recomiendo", "increíble"
]
NEGATIVOS = [
    "poor", "terrible", "bad", "worst", "horrible", "not recommend",
    "awful", "disappointing", "slow", "broken", "malo", "terrible",
    "pésimo", "horrible", "no recomiendo", "deficiente"
]

def clasificar(texto: str) -> tuple[str, float]:
    t = texto.lower()
    p = sum(1 for w in POSITIVOS if w in t)
    n = sum(1 for w in NEGATIVOS if w in t)
    total = p + n or 1
    if p > n:
        return "positivo", round(p / total, 2)
    elif n > p:
        return "negativo", round(n / total, 2)
    else:
        return "neutral", 0.5


# ------------------------------------------------------------------
# GET /sentiments
# Devuelve todos los documentos de la colección (sin _id de Mongo)
# Params opcionales: ?clase=positivo|negativo|neutral
#                    ?limit=N  (default 100)
# ------------------------------------------------------------------
@app.route("/sentiments", methods=["GET"])
def get_sentiments():
    clase = request.args.get("clase")
    limit = int(request.args.get("limit", 100))

    filtro = {}
    if clase:
        filtro["prediccion"] = clase

    docs = list(col.find(filtro, {"_id": 0}).limit(limit))
    return jsonify({"total": len(docs), "data": docs}), 200


# ------------------------------------------------------------------
# GET /stats
# Devuelve estadísticas agregadas de las predicciones almacenadas
# ------------------------------------------------------------------
@app.route("/stats", methods=["GET"])
def get_stats():
    pipeline = [
        {"$group": {
            "_id": "$prediccion",
            "count": {"$sum": 1},
            "correctas": {
                "$sum": {
                    "$cond": [
                        {"$eq": ["$etiqueta_real", "$prediccion"]}, 1, 0
                    ]
                }
            }
        }},
        {"$project": {
            "clase":    "$_id",
            "total":    "$count",
            "correctas": 1,
            "precision": {
                "$round": [
                    {"$divide": ["$correctas", "$count"]}, 4
                ]
            },
            "_id": 0
        }},
        {"$sort": {"clase": 1}}
    ]

    por_clase = list(col.aggregate(pipeline))

    total_docs    = col.count_documents({})
    total_correct = col.count_documents({"$expr": {"$eq": ["$etiqueta_real", "$prediccion"]}})
    accuracy_global = round(total_correct / total_docs, 4) if total_docs else 0

    return jsonify({
        "total_registros": total_docs,
        "accuracy_global": accuracy_global,
        "por_clase": por_clase
    }), 200


# ------------------------------------------------------------------
# POST /predict
# Body JSON: {"texto": "Amazing service!"}
# Responde con la clase predicha y la confianza
# ------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or "texto" not in body:
        return jsonify({"error": "Se requiere el campo 'texto' en el body JSON"}), 400

    texto = str(body["texto"]).strip()
    if not texto:
        return jsonify({"error": "El campo 'texto' no puede estar vacío"}), 400

    prediccion, confianza = clasificar(texto)

    doc = {
        "texto_original": texto,
        "etiqueta_real":  body.get("etiqueta_real", "desconocida"),
        "prediccion":     prediccion,
        "confianza":      confianza,
        "timestamp":      datetime.utcnow().isoformat()
    }
    col.insert_one(doc)

    return jsonify({
        "texto":     texto,
        "prediccion": prediccion,
        "confianza":  confianza
    }), 200


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
