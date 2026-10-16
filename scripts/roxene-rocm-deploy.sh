#!/usr/bin/env bash
# Rebuild the roxene images on enzo.local (native amd64) and redeploy them into
# the minikube node there.  usage: tools/roxene-rocm-deploy.sh
set -euo pipefail

REMOTE="${ROXENE_REMOTE:-joexner@enzo.local}"
REMOTE_DIR="${ROXENE_REMOTE_DIR:-roxene}"
LOCAL_SRC="${ROXENE_SRC:-$HOME/roxene}"
PROFILE="${ROXENE_PROFILE:-minikube}"
LOCAL_PORT="${ROXENE_LOCAL_PORT:-8443}"
KUBE_DIR="${ROXENE_KUBE_DIR:-$HOME/.kube/enzo}"

# minikube picks the apiserver's host port at random when the node is created.
remote_port=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" \
  "podman port ${PROFILE} 8443/tcp 2>/dev/null | head -1" \
  | sed 's/.*://' | tr -d '\r')

if [ -z "$remote_port" ]; then
  echo "error: could not discover the apiserver port on ${REMOTE} -- is the node running?"
  exit 1
fi

echo "apiserver: ${REMOTE}:${remote_port}  ->  localhost:${LOCAL_PORT}"

# The apiserver only listens on lo inside enzo.local, so reach it over ssh.
if lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "==> 1/4 ssh tunnel already listening"
else
  echo "==> 1/4 opening ssh tunnel"
  ssh -o BatchMode=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -fN -L "${LOCAL_PORT}:127.0.0.1:${remote_port}" "$REMOTE"
fi

echo "==> 2/4 syncing sources to enzo"
rsync -a --delete \
  --exclude '.venv/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.DS_Store' \
  --exclude '.idea/' \
  "$LOCAL_SRC"/ "$REMOTE":"${REMOTE_DIR}"/

echo "==> 3/4 building images on enzo and loading them into the node"
ssh -o BatchMode=yes "$REMOTE" "set -euo pipefail
cd ~/${REMOTE_DIR}
podman build --target training -t docker.io/library/roxene:latest .
podman build --target training-rocm -t docker.io/library/roxene-rocm:latest .
podman build --target notebook -t docker.io/library/roxene-notebook:latest .
# minikube buffers stdin into a temp archive; /tmp is a 16G tmpfs, so use /home.
mkdir -p ~/tmp
export TMPDIR=~/tmp
podman save docker.io/library/roxene:latest | minikube image load -
podman save docker.io/library/roxene-rocm:latest | minikube image load -
podman save docker.io/library/roxene-notebook:latest | minikube image load -"

echo "==> 4/4 refreshing certs and deploying"
mkdir -p "${KUBE_DIR}"
scp -q "${REMOTE}:.kube/config" "${KUBE_DIR}/config"
scp -q "${REMOTE}:.minikube/ca.crt" "${KUBE_DIR}/ca.crt"
scp -q "${REMOTE}:.minikube/profiles/${PROFILE}/client.crt" "${KUBE_DIR}/client.crt"
scp -q "${REMOTE}:.minikube/profiles/${PROFILE}/client.key" "${KUBE_DIR}/client.key"
chmod 600 "${KUBE_DIR}/client.key"

# Certs are rotated whenever the node is rebuilt, so refresh them from enzo.
sed -E -e "s#/home/[^/]*/\.minikube/ca\.crt#${KUBE_DIR}/ca.crt#g" \
       -e "s#/home/[^/]*/\.minikube/profiles/${PROFILE}/(client\.crt)#${KUBE_DIR}/\1#g" \
       -e "s#/home/[^/]*/\.minikube/profiles/${PROFILE}/(client\.key)#${KUBE_DIR}/\1#g" \
       -e "s#server:[[:space:]]*https://[^:]+:[0-9]+#server: https://127.0.0.1:${LOCAL_PORT}#g" \
       "${KUBE_DIR}/config" > "${KUBE_DIR}/config.new"
mv "${KUBE_DIR}/config.new" "${KUBE_DIR}/config"

export KUBECONFIG="${KUBE_DIR}/config"
kubectl config use-context "${PROFILE}" >/dev/null

kubectl apply -k "${LOCAL_SRC}/k8s/overlays/enzo.local"

# The tag is unchanged, so apply alone won't swap in the rebuilt image.
kubectl -n roxene rollout restart statefulset/worker deployment/notebook

kubectl -n roxene rollout status statefulset/worker --timeout=600s
kubectl -n roxene get pods -o wide
echo "==> done"
