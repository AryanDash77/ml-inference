# ML Inference API with Kubernetes on AWS

Containerized, model-agnostic ML inference API, orchestrated with k3s
(lightweight Kubernetes) on a single free-tier AWS EC2 instance, with
a Zero Trust network boundary and GitHub Actions CI/CD.

## Project structure

```
app/main.py          FastAPI app - loads any joblib model, exposes /predict
models/               Put your model.joblib + feature_names.json here
Dockerfile             Container build
k8s/deployment.yaml     Pods running the API (2 replicas, resource-limited)
k8s/service.yaml        Stable network entrypoint (LoadBalancer via k3s)
k8s/hpa.yaml             Autoscaling rule (1-4 pods, scale on CPU > 70%)
k8s/networkpolicy.yaml   Zero Trust: default-deny + explicit allow
.github/workflows/deploy.yml   CI/CD: build image -> push to ECR -> deploy
```

## Status: what's already built and tested

- [x] Real fraud_xgb_model.pkl - already in models/, pulled from your repo
- [x] FastAPI app - rewritten specifically for this model (30 features:
      V1-V28 + scaled_amount + scaled_time, matching training exactly),
      tested end-to-end against your real model (health check, a real
      prediction, and validation all confirmed working)
- [x] Dockerfile - written, not yet build-tested (no Docker daemon in this
      environment - test this on your machine or the EC2 instance)
- [x] Kubernetes manifests - written, not yet applied to a real cluster
- [x] GitHub Actions workflow - written, not yet run (needs secrets set up)
- [ ] Real scaler_amount.pkl / scaler_time.pkl (see Step 1 - this fixes a
      real bug found in the original repo's scaler.pkl)
- [ ] Actual EC2 + k3s deployment
- [ ] Zero Trust CNI (Calico/Cilium) installed so NetworkPolicy is enforced
- [ ] Load testing to prove autoscaling

## Step 1 - fix the scaler bug (do this first)

Your original `scaler.pkl` was accidentally fit twice on the same object
(once on Amount, then again on Time - the second fit overwrote the
first), so it only holds Time's statistics. `fix_scalers.py` in this
folder regenerates two SEPARATE, correctly-fit scalers.

```bash
# Run this in the same folder as your creditcard.csv:
python fix_scalers.py
# Copy the two files it produces into this project's models/ folder:
cp scaler_amount.pkl scaler_time.pkl  /path/to/ml-inference-project/models/
```

Then test locally (no Docker needed for this step):

```bash
pip install -r requirements.txt
MODEL_DIR=./models uvicorn app.main:app --reload
# Open http://localhost:8000/docs and try /predict with:
# v: [0.0, 0.0, ... 28 values], amount: 50.0, time: 40000.0
```

## Step 2 - test the Docker build locally

```bash
docker build -t ml-inference-api .
docker run -p 8000:8000 ml-inference-api
curl http://localhost:8000/health
```

## Step 3 - AWS setup (do this closer to your review)

1. Launch a free-tier EC2 instance (t2.micro or t3.micro, Ubuntu 22.04).
2. Install k3s: `curl -sfL https://get.k3s.io | sh -`
3. **For the Zero Trust NetworkPolicy to actually work**, k3s's default
   CNI (flannel) does NOT enforce NetworkPolicy. Install k3s with
   `--flannel-backend=none --disable-network-policy` and then install
   Calico on top. Mention this explicitly if asked - it shows you
   understand the enforcement layer, not just the YAML.
4. Create an ECR repository: `aws ecr create-repository --repository-name ml-inference-api`
5. Apply the manifests: `kubectl apply -f k8s/`
6. Verify: `kubectl get pods`, `kubectl get hpa`, `kubectl get svc`

## Step 4 - CI/CD secrets (GitHub repo settings -> Secrets and variables -> Actions)

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - IAM user scoped to ECR push only
- `EC2_HOST` - your instance's public IP
- `EC2_SSH_USER` - usually `ubuntu`
- `EC2_SSH_KEY` - the private key content for SSH access

## Step 5 - load testing (for the "auto-scaling" demo)

```bash
pip install locust
# write a simple locustfile.py hitting /predict, then:
locust -f locustfile.py --host http://<EC2_PUBLIC_IP>
```

Watch `kubectl get hpa -w` while the load test runs - pod count should
increase as CPU usage crosses 70%, then decrease after load stops.

## Note on cost

Everything here fits in AWS's free-tier signup credit and the
always-free service tiers. Don't leave the EC2 instance running for
weeks unattended after you're done testing - stop it when not in use.
