// infra/init_mongo.js
// Este script corre automáticamente la primera vez que el contenedor
// de MongoDB arranca (montado como /docker-entrypoint-initdb.d/)

db = db.getSiblingDB("sentimentstream");

db.createCollection("predicciones");

db.predicciones.createIndex({ prediccion: 1 });
db.predicciones.createIndex({ etiqueta_real: 1 });
db.predicciones.createIndex({ timestamp: -1 });

print("✅ Base de datos 'sentimentstream' e índices creados.");
