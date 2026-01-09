from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import json
import uuid
import time
import requests

from kubernetes import client, config

# ---------------------------------------
# App
# ---------------------------------------
app = FastAPI(title="Private K8s SRE Agent (Deterministic Plan/Apply)")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "llama3.2:1b")

# Load Kubernetes config (in-cluster vs local)
if os.getenv("KUBERNETES_SERVICE_HOST"):
    config.load_incluster_config()
else:
    config.load_kube_config()

v1 = client.CoreV1Api()
apps = client.AppsV1Api()

# In-memory plan store (MVP)
PLANS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------
# Request Models
# ---------------------------------------
class TriageReq(BaseModel):
    namespace: str = "default"
    pod_name: str

class PlanReq(BaseModel):
    namespace: str = "default"
    deployment_name: str
    # Optional preference; LLM must still pick from evidence.images[].name
    container_name: Optional[str] = None

class ApplyReq(BaseModel):
    plan_id: str
    approve: bool = False


# ---------------------------------------
# K8s Evidence Helpers
# ---------------------------------------
def get_pod_summary(namespace: str, pod_name: str) -> Dict[str, Any]:
    try:
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Pod not found: {e}")

    cs = pod.status.container_statuses or []
    container_statuses = []
    for c in cs:
        st = c.state
        container_statuses.append({
            "name": c.name,
            "ready": c.ready,
            "restarts": c.restart_count,
            "waiting_reason": st.waiting.reason if st and st.waiting else None,
            "terminated_reason": st.terminated.reason if st and st.terminated else None,
            "running": bool(st and st.running),
        })

    return {
        "pod_phase": pod.status.phase,
        "node_name": pod.spec.node_name,
        "container_images": [c.image for c in pod.spec.containers],
        "container_statuses": container_statuses,
    }

def get_pod_events(namespace: str, pod_name: str, limit: int = 8) -> List[Dict[str, Any]]:
    ev = v1.list_namespaced_event(
        namespace=namespace,
        field_selector=f"involvedObject.name={pod_name},involvedObject.kind=Pod"
    )
    items = ev.items or []
    items = items[-limit:]
    return [{"type": e.type, "reason": e.reason, "message": e.message} for e in items]

def get_pod_logs(namespace: str, pod_name: str, tail: int = 50) -> str:
    try:
        return v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=tail)
    except Exception as e:
        return f"(could not fetch logs: {e})"

def selector_from_match_labels(match_labels: Dict[str, str]) -> str:
    return ",".join([f"{k}={v}" for k, v in (match_labels or {}).items()])

def get_deploy_evidence(namespace: str, deployment_name: str) -> Dict[str, Any]:
    try:
        dep = apps.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Deployment not found: {e}")

    match_labels = dep.spec.selector.match_labels or {}
    label_selector = selector_from_match_labels(match_labels)
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector).items

    pod_summaries = []
    for p in pods[:5]:
        cs = p.status.container_statuses or []
        pod_summaries.append({
            "pod": p.metadata.name,
            "phase": p.status.phase,
            "container_statuses": [{
                "name": c.name,
                "ready": c.ready,
                "restarts": c.restart_count,
                "waiting_reason": c.state.waiting.reason if c.state and c.state.waiting else None,
                "terminated_reason": c.state.terminated.reason if c.state and c.state.terminated else None,
            } for c in cs]
        })

    images = [{"name": c.name, "image": c.image} for c in dep.spec.template.spec.containers]

    return {
        "deployment": deployment_name,
        "namespace": namespace,
        "match_labels": match_labels,
        "images": images,
        "pod_summaries": pod_summaries,
    }


# ---------------------------------------
# Ollama + Parsing Helpers
# ---------------------------------------
def ollama_generate(prompt: str, timeout_s: int = 600) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=timeout_s
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

def try_fix_unclosed_json(text: str) -> str:
    # Pragmatic fix: append missing closing braces if LLM truncated the final braces
    if "{" not in text:
        return text
    opens = text.count("{")
    closes = text.count("}")
    if opens > closes:
        text = text + ("}" * (opens - closes))
    return text

def extract_first_json_object(text: str) -> dict:
    """
    Extract the first complete JSON object from a text blob.
    Handles extra text or multiple JSON objects.
    """
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise HTTPException(status_code=400, detail=f"LLM returned no JSON object. RAW:\n{text[:1200]}")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid JSON from LLM: {e}\nRAW_JSON:\n{candidate}\nFULL_RAW:\n{text[:1200]}"
                        )

    raise HTTPException(status_code=400, detail=f"Unclosed JSON object from LLM. RAW:\n{text[:1200]}")


# ---------------------------------------
# LLM: triage + recommendation (Option 1)
# ---------------------------------------
def ask_llm_triage(question: str, evidence: Dict[str, Any]) -> str:
    prompt = f"""
You are a senior Kubernetes SRE. Only use the EVIDENCE provided.
Do NOT invent resources.

Return exactly these sections:

SUMMARY:
- (2-4 bullets)

ROOT_CAUSE:
- #1
- #2
- #3

EVIDENCE_QUOTES:
- Quote 2-4 short lines from events/logs that support the root cause.

INVESTIGATION_COMMANDS (kubectl only):
- Provide 4-8 valid kubectl commands.

SAFE_REMEDIATION:
- 3-6 safe steps.
- Do NOT delete namespaces.
- Do NOT suggest "docker images".

QUESTION:
{question}

EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=300
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def ask_llm_for_recommendation(question: str, evidence: Dict[str, Any], container_name: Optional[str]) -> Dict[str, Any]:
    """
    LLM returns intent only:
      {
        "container": "nginx",
        "recommended_image": "nginx:latest",
        "reason": "short"
      }
    Patch is built deterministically in code.
    """
    available = [c.get("name") for c in evidence.get("images", []) if isinstance(c, dict)]
    container_hint = container_name or "(not specified; choose from available_containers)"

    prompt = f"""
You are a senior Kubernetes SRE. Only use the EVIDENCE provided.

Return ONLY a single JSON object (no markdown, no extra text) with keys:
- container (MUST be one of available_containers exactly)
- recommended_image (a valid image reference)
- reason (1-2 sentences)

Rules:
- Do NOT output Kubernetes patches.
- Do NOT repeat the evidence.
- container MUST be selected from available_containers exactly.

available_containers: {json.dumps(available)}

Container preference (optional): {container_hint}

QUESTION:
{question}

EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()

    text = ollama_generate(prompt, timeout_s=600)
    text = try_fix_unclosed_json(text)

    try:
        obj = json.loads(text)
    except Exception:
        obj = extract_first_json_object(text)

    if not isinstance(obj, dict) or "container" not in obj or "recommended_image" not in obj:
        raise HTTPException(status_code=400, detail=f"LLM must return container and recommended_image. RAW:\n{text[:1200]}")

    container = obj.get("container")
    rec_image = obj.get("recommended_image")

    if container not in available:
        raise HTTPException(status_code=400, detail=f"LLM chose invalid container '{container}'. Must be one of {available}. RAW:\n{text[:1200]}")

    if not isinstance(rec_image, str) or len(rec_image.strip()) < 3:
        raise HTTPException(status_code=400, detail=f"LLM recommended_image looks invalid: {rec_image}. RAW:\n{text[:1200]}")

    return obj


# ---------------------------------------
# Deterministic patch builder + validator
# ---------------------------------------
def build_image_patch(container: str, image: str) -> Dict[str, Any]:
    return {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"name": container, "image": image}
                    ]
                }
            }
        }
    }

def validate_patch_only_images(patch: dict) -> None:
    allowed_top = {"spec"}
    if set(patch.keys()) - allowed_top:
        raise HTTPException(400, "Patch contains unsupported top-level fields")

    spec = patch.get("spec", {})
    if set(spec.keys()) - {"template"}:
        raise HTTPException(400, "Patch contains unsupported spec fields")

    template = spec.get("template", {})
    if set(template.keys()) - {"spec"}:
        raise HTTPException(400, "Patch contains unsupported template fields")

    tspec = template.get("spec", {})
    if set(tspec.keys()) - {"containers"}:
        raise HTTPException(400, "Patch contains unsupported pod spec fields")

    containers = tspec.get("containers")
    if not isinstance(containers, list) or not containers:
        raise HTTPException(400, "Patch must include containers list")

    for c in containers:
        if set(c.keys()) - {"name", "image"}:
            raise HTTPException(400, "Only name and image allowed per container in patch")
        if "name" not in c or "image" not in c:
            raise HTTPException(400, "Each container patch must include name and image")
# ---------------------------------------
# Routes
# ---------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "ollama_url": OLLAMA_URL}


@app.post("/triage")
def triage(req: TriageReq):
    evidence = {
        "summary": get_pod_summary(req.namespace, req.pod_name),
        "events": get_pod_events(req.namespace, req.pod_name),
        "logs_tail": get_pod_logs(req.namespace, req.pod_name),
    }
    question = f"Why is pod {req.pod_name} failing in namespace {req.namespace}?"

    try:
        analysis = ask_llm_triage(question, evidence)
    except Exception as e:
        analysis = f"LLM not reachable: {e}"

    return {"question": question, "analysis": analysis, "evidence": evidence}


@app.post("/plan")
def plan(req: PlanReq):
    evidence = get_deploy_evidence(req.namespace, req.deployment_name)
    question = f"Fix the Kubernetes Deployment {req.deployment_name} in namespace {req.namespace}."

    # Default deterministic fallback values
    available = [c.get("name") for c in evidence.get("images", []) if isinstance(c, dict)]
    container = req.container_name or (available[0] if available else None)
    if not container:
        raise HTTPException(400, "No containers found in deployment evidence")

    current_image = None
    for c in evidence.get("images", []):
        if c.get("name") == container:
            current_image = c.get("image")
            break

    # Try LLM first
    try:
        rec = ask_llm_for_recommendation(question, evidence, req.container_name)
        container = rec["container"]
        recommended_image = rec["recommended_image"]
        reason = rec.get("reason", "LLM recommendation")
    except Exception as e:
        # Deterministic fallback (very safe for demo):
        # If image tag looks wrong (like latesttt), use nginx:latest
        reason = f"Fallback: LLM failed ({e}). Using safe image correction based on evidence."
        recommended_image = "nginx:latest"
        if isinstance(current_image, str) and ":" in current_image:
            base = current_image.split(":")[0]
            # Keep the same repo (nginx) but fix tag
            recommended_image = f"{base}:latest"

    patch = build_image_patch(container, recommended_image)
    validate_patch_only_images(patch)

    plan_id = str(uuid.uuid4())
    PLANS[plan_id] = {
        "namespace": req.namespace,
        "deployment": req.deployment_name,
        "container": container,
        "current_image": current_image,
        "recommended_image": recommended_image,
        "reason": reason,
        "patch": patch,
        "created": int(time.time()),
    }

    return {
        "plan_id": plan_id,
        "container": container,
        "current_image": current_image,
        "recommended_image": recommended_image,
        "reason": reason,
        "patch": patch,
    }
@app.post("/apply")
def apply(req: ApplyReq):
    if req.plan_id not in PLANS:
        raise HTTPException(status_code=404, detail="Unknown plan_id")

    plan_obj = PLANS[req.plan_id]

    if not req.approve:
        return {
            "status": "not_applied",
            "message": "Human-in-the-loop: set approve=true to apply.",
            "plan": plan_obj,
        }

    namespace = plan_obj["namespace"]
    deployment = plan_obj["deployment"]
    patch = plan_obj["patch"]

    # Apply strategic merge patch to Deployment
    apps.patch_namespaced_deployment(name=deployment, namespace=namespace, body=patch)

    return {
        "status": "applied",
        "namespace": namespace,
        "deployment": deployment,
        "patch": patch,
    }
