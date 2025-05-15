#!/bin/bash

ENV_NAME="recommender"
YML_FILE="environment.yml"

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[INFO] '${ENV_NAME}' environment already exists. Updating..."
    conda env update --name $ENV_NAME --file $YML_FILE --prune
else
    echo "[INFO] '${ENV_NAME}' environment doesn\'t exists. Creating..."
    conda env create --file $YML_FILE
fi

echo "[INFO] '${ENV_NAME}' Ready!"
