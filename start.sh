docker run -it --rm  --name filippo_jupyter --user root -p 9000:8888 -p 9001:4040 -v /home/filippo/bigdata/:/home/jovyan/ -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes  filippo/big_data
