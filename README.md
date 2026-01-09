# 🤖 AI-Powered Kubernetes SRE Agent (Plan/Apply with Human-in-the-Loop)

This project demonstrates an **AI-assisted Site Reliability Engineering (SRE) agent**
that can **triage Kubernetes failures**, **generate safe remediation plans**, and
**apply fixes only after human approval**.

Built to mimic real-world **SRE and Platform Engineering workflows**.

---

## 🚀 Features

- 🔍 **Automatic Kubernetes Failure Triage**
- 🧠 **LLM-based Root Cause Analysis (Local Ollama)**
- 🛠️ **Deterministic Plan Generation**
- 🧍 **Human-in-the-loop Approval**
- 🔐 **Safe, Restricted Patch Application**
- 🧩 **No destructive actions (image-only fixes)**

---

## 🏗️ Architecture Overview
User (curl / API)
|
v
FastAPI SRE Agent
|
|--> Kubernetes API (pods, events, deployments)
|
|--> Ollama (Local LLM reasoning)
|
v
PLAN (safe patch)
|
v
Human Approval
|
v
Apply Patch → Rolling Update

---

## 🧠 Why This Matters

Traditional SRE workflows are manual and reactive.

This project shows how **AI can assist** — not replace — engineers by:
- Understanding real cluster state
- Suggesting safe remediations
- Enforcing approval gates

This mirrors how **production-grade internal tooling** works at scale.

---

## 🛠️ Tech Stack

- **Python + FastAPI**
- **Kubernetes Python Client**
- **Ollama (Local LLM runtime)**
- **Docker**
- **Kubernetes (Deployments, RBAC)**

---

## 🧪 Demo Scenario

1. Deploy a broken `nginx` image (`nginx:latesttt`)
2. Pod enters `ImagePullBackOff`
3. `/triage` identifies root cause
4. `/plan` proposes image fix
5. `/apply` requires approval
6. Deployment rolls out successfully

---

## 📦 How to Run

```bash
# Deploy Ollama
kubectl apply -f k8s/ollama.yaml

# Deploy SRE Agent
kubectl apply -f k8s/agents.yaml

# Port forward
kubectl port-forward svc/sre-agent-svc 8080:80 -n sre-ai

