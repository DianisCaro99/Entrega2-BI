# Entrega2-BI

## API URL
Base URL: 
  - Predictores X de un registro (Recibe en el body un JSON): 

## Instalación
  Instalación de modulos adicionales:
```bash
pip install flask
pip install -U flask-cors
pip install pandas
pip install joblib
pip install scikit-learn
pip install pydantic
```
  Correr con el comando 
 ```bash
flask run
```
## Despliegue
1. Crear una máquina EC2 en AWS  con los siguientes requerimientos
  - **Instance type:** t2.medium - Ubuntu Server 18.04 LTS (HVM)
  - **Key pair:** ov42-europe
  - **Network settings:** Default - Allow SSH traffic from *Anywhere*
2. Habilitar el puerto 80 para comunicación HTTP
3. Correr los siguientes comandos 
```bash
sudo apt-get update
sudo apt-get install python3
sudo apt-get install python3-pip
pip3 install --upgrade pip

sudo pip3 install flask
pip install Flask
pip install -U flask-cors
pip install pandas
pip install joblib
pip install scikit-learn
pip install pydantic
sudo apt-get install nginx
sudo apt-get install gunicorn3

sudo python3 -m pip install -U setuptools
```
4. Clonar el repositorio de GitHUB
```bash
git clone https://github.com/DianisCaro99/Entrega2-BI.git
```
5. Ir a la carpeta del sistema de la máquina
```bash
cd /etc/systemd/system/
```
7. Crear el archivo .service
```bash
sudo vi bi.service
```
9. Escribir en el archivo bi.service
```bash
[Unit]
Description=Flaskapp Gunicorn to serve BI - Lab4
After=network.target
StartLimitIntervalSec=0

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/Entrega2-BI
ExecStart=/usr/bin/gunicorn3 --workers 3  --bind unix:bi.sock -m 007 app:app --timeout 0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
10. Correr los siguientes comandos para crear el .sock file
```bash
sudo systemctl daemon-reload
sudo systemctl enable bi.service

sudo service nginx restart
sudo service bi restart
```
11. Entrar a la configuracion del Nginx
```bash
cd /etc/nginx/sites-enabled/
```
12. Crear el archivo de configuracion
```bash
sudo vi flask_app
```
13. Escribir en el archivo flask_app
```bash
server {
    listen 80;
    server_name <ELASTIC IP>;

    location / {
        proxy_pass http://unix:/home/ubuntu/Entrega2-BI/bi.sock;
    }
}
```
14. Correr el siguiente comando para reiniciar el servidor
```bash
sudo service nginx restart
```
