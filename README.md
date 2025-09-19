# AiWebserver

API Start:
```bash
git clone https://github.com/ReeceGibbo/AiWebserver.git
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

python basic_service.py
nohup python basic_service.py > /dev/null 2>&1 &

pkill -f "basic_service.py"
```