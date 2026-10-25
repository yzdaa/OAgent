# 启动后端
uvicorn app.api.fastapi_app:app --host 0.0.0.0 --port 8000
<!-- uvicorn app.api.fastapi_app:app --host 0.0.0.0 --port 8000 -->

# 启动前端
python -m streamlit run app/web/streamlit_app.py --server.port 8501
