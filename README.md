# MLP Risk Classification API

This repository provides a FastAPI service that exposes a trained MLP model for
risk classification.

## Build and run with Docker

1. Place the trained TensorFlow model directory (`mlp_tf_model`) and
   `mlp_tf_pipeline.pkl` inside a folder on your host.
2. Build the container:
   ```bash
   docker build -t mlp-api .
   ```
3. Run the service, mounting the folder with the model artifacts:
   ```bash
   docker run -p 8000:8000 -v /path/to/artifacts:/app/model mlp-api
   ```
4. Send a prediction request:
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
        -d '{"sexo": "M", "glosa_red": "diagnostico", "edad_al_hospitalizarse": 40,
             "plan_para_asegurados": "plan1", "plan_catastrofico_asegurado": "no",
             "estado_vigencia_cobertura": "activa", "prestacion_basica": "si",
             "cto_dto": 123.0}'
   ```

The service will respond with the predicted risk class and probabilities for
each class.
