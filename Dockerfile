FROM python:3.12

RUN pip install uv

WORKDIR /home

COPY ["pyproject.toml", "uv.lock", "main.py", "./"]

RUN uv sync

COPY scripts/ /home/scripts/   

COPY public/index.html /home/public/

COPY models/model_v2.bin /home/models/

COPY data/ /home/data/

#COPY car_results/ /home/car_results/

EXPOSE 80

ENTRYPOINT ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]