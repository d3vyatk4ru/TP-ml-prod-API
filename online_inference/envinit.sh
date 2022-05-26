apt-get update -y\
&& apt-get upgrade -y\
&& python -m pip install --upgrade pip\
&& pip install --no-cache-dir -r requirements.txt
