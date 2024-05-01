# pull pyspark image
FROM jupyter/pyspark-notebook:latest
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

USER $NB_UID
# Path: Dockerfile
# run the app
# CMD ["python", "src/kmeans.py"]

# docker build -t  ahmedsabry2024/jupyter-notebook-spark:latest .
# docker container run -d -p 8888:8888 --name jupyter-notebook-container ahmedsabry2024/jupyter-notebook-spark
# docker container stop jupyter-notebook-container
# docker container rm jupyter-notebook-container
# docker ps --all
# docker tag ahmedsabry2024/jupyter-notebook-spark analyzer.azurecr.io/jupyter-notebook-spark:latest
# docker push analyzer.azurecr.io/jupyter-notebook-spark:latest