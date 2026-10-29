FROM nginx:1.27-alpine
COPY frontend/index.html /usr/share/nginx/html/index.html
COPY frontend/style.css /usr/share/nginx/html/style.css
COPY frontend/app.js /usr/share/nginx/html/app.js
COPY frontend/network.js /usr/share/nginx/html/network.js
COPY frontend/charts.js /usr/share/nginx/html/charts.js
COPY frontend/config.js /usr/share/nginx/html/config.js
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
