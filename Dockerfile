FROM jupyter/pyspark-notebook
LABEL maintainer="Filippo"
ENV JUPYTER_ENABLE_LAB=yes
USER root
RUN sudo wget http://central.maven.org/maven2/com/databricks/spark-xml_2.11/0.5.0/spark-xml_2.11-0.5.0.jar  -P $SPARK_HOME/jars
ADD ./spark-defaults.conf /usr/local/spark/conf/spark-defaults.conf
#ENV NB_UID $(id -u)
#ENV NB_GID $(id -g)
#ENV GRANT_SUDO yes
