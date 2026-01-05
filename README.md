# End-to-End MLOps Pipeline for Heart Disease Prediction

## Project Overview
This project demonstrates an end-to-end **MLOps pipeline** for building, deploying, and monitoring a machine learning model to predict the **risk of heart disease** based on patient health data.

The solution follows modern MLOps best practices including **experiment tracking, CI/CD automation, containerization, Kubernetes deployment, and monitoring**, closely mirroring real-world production ML systems.

---

## Problem Statement
Build a machine learning classifier that predicts the presence or absence of heart disease using patient clinical data, and deploy it as a **production-ready, monitored API**.

---

## Dataset
- **Name:** Heart Disease UCI Dataset  
- **Source:** UCI Machine Learning Repository  
- **Type:** CSV  
- **Target Variable:** Binary (presence / absence of heart disease)  
- **Features:** Age, sex, blood pressure, cholesterol, ECG results, etc.

---


---

## Data Processing & EDA
- Missing values handled appropriately
- Categorical features encoded
- Numerical features scaled
- EDA performed using:
  - Feature distributions
  - Correlation heatmaps
  - Class balance analysis

---

## Model Development
- Models trained:
  - Logistic Regression (baseline, interpretable)
  - Random Forest (non-linear, higher capacity)
- Model evaluation:
  - Accuracy
  - Precision
  - Recall
  - ROC-AUC
- Cross-validation used for robust performance estimation
- Final model selected based on balanced metric performance

---

## Experiment Tracking
- **MLflow** used to log:
  - Model parameters
  - Evaluation metrics
  - Trained models
  - Plots and artifacts
- Enables full reproducibility and comparison across experiments

---

## Reproducibility
- Preprocessing pipeline bundled with the model
- Model saved in a reusable serialized format
- `requirements.txt` ensures environment consistency
- Entire pipeline runs from a clean setup

---

## CI/CD Pipeline
Implemented using **GitHub Actions**, including:
- Code linting
- Unit testing (Pytest)
- Model training
- Artifact and log generation

Pipeline is designed to **fail fast** on errors, ensuring code quality and reliability.

---

## Model Serving API
- REST API built using **FastAPI / Flask**
- Endpoint:
  - `POST /predict`
- Input: JSON patient data
- Output:
  - Prediction (0/1)
  - Confidence score
- API containerized using Docker and tested locally

---

## Containerization
- Docker image includes:
  - Application code
  - Model artifact
  - Dependencies
- Ensures portability and environment isolation

---

## Deployment
- Dockerized API deployed on:
  - Local Kubernetes cluster (Minikube / Docker Desktop)
  - Virtual Machine running Rocky Linux
- Kubernetes manifests used for:
  - Deployment
  - Service exposure
- API endpoint verified using sample requests

---

## Monitoring & Logging
- API request logging enabled
- Basic monitoring implemented using:
  - Metrics endpoint / Prometheus + Grafana (or logs dashboard)
- Demonstrates production observability

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd <repository-name>

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3. Run Tests
pytest tests/

### 4. Build Docker Image and Run Container locally
docker build -t heart-disease-api -f docker/Dockerfile .
and docker run -p 8000:8000 heart-disease-api

### 5. Kubernetes containerization
Enable Kubernetes on Docker Desktop and then
kubectl apply –f k8s/deployment.apps/heart-disease-service created
kubectl get pods
kubectl get services

In case you need to restart deployment,
kubectl rollout restart deployment heart-disease-api

In case you need to set it up on any Linux Virtual Machine like OSHA lab.

Run the below commands on the VM

sudo dnf update -y
sudo dnf install -y dnf-utils 
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo 
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

Start Docker
sudo systemctl start docker
sudo systemctl enable docker

Allow non-root Docker
sudo usermod -aG docker $USER

Build Docker Image

docker build -t heart-disease-api -f docker/Dockerfile .
docker run -d -p 8000:8000 --name heart-api heart-disease-api