param(
    [int]$Port = 5000
)

mlflow ui --backend-store-uri "file:./mlruns" --host 127.0.0.1 --port $Port
