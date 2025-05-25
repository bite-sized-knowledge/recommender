#!/bin/bash

# conda 환경 이름
ENV_NAME="recommender"

# 프로젝트 루트 경로
PROJECT_DIR="$HOME/recommender"
LOG_DIR="$HOME/logs/rec_batch"

# 로그 파일명 (날짜별)
LOG_FILE="$LOG_DIR/rec_$(date +\%Y-\%m-\%d).log"

# Conda activate
source ~/.miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 스크립트 실행
cd "$PROJECT_DIR"
python main.py >> "$LOG_FILE" 2>&1 &
