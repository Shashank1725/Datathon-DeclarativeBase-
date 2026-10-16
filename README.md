# Revenue Sentinel
    pip install -r requirements.txt
    python train.py                    # trains Isolation Forest + LightGBM, writes data/*.json
    uvicorn app:app --reload           # open http://localhost:8000
Deploy on Render: build `pip install -r requirements.txt && python train.py`, start `uvicorn app:app --host 0.0.0.0 --port $PORT`.
