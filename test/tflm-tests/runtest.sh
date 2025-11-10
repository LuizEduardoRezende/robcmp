#!/bin/bash
# Script para executar testes unitários TFLM
# Baseado no runtest.sh do test/general mas adaptado para TFLM

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

# Parâmetros
ROB_FILE=$1
TEST_NAME=$2

echo -n " ...           $ROB_FILE "

# Compilar usando o Makefile local
make -s out/$TEST_NAME &> /dev/null

if [ "$?" -eq 0 ]; then
    # Executar o programa com timeout
    timeout 10s ./out/$TEST_NAME &> /dev/null
    STATUS=$?
    
    if [ "${STATUS}" -eq 0 ]; then
        echo -e \\r${GREEN}[PASS]\\t\\t${NC}
    else
        echo -e \\r${RED}[FAILED] ${STATUS}\\t${NC}
    fi
else
    echo -e \\r${RED}[BUILD FAILED]\\t${NC}
fi