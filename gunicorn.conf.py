
# ---------- basic config ----------
bind = "0.0.0.0:8000"
workers = 6  
worker_class = "uvicorn.workers.UvicornWorker" 
timeout = 120 
keepalive = 5 
preload = False 

# ---------- logging  ----------
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

# ---------- optimization ----------
worker_tmp_dir = "/tmp"          
max_requests = 1000             
max_requests_jitter = 100        
graceful_timeout = 30