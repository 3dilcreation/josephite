#!/usr/bin/env bash
# Generates a self-signed cert covering localhost plus every LAN IPv4 this host
# currently owns. getUserMedia and service workers both require a secure context,
# and a bare LAN IP over http:// is not one -- so HTTPS is mandatory even offline.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/server/certs"
mkdir -p "$DIR"

ips=$(node -e '
const os = require("os");
const out = new Set(["127.0.0.1"]);
for (const list of Object.values(os.networkInterfaces())) {
  for (const ni of list || []) if (ni.family === "IPv4") out.add(ni.address);
}
console.log([...out].join(" "));
')

alt="DNS:localhost"
i=1
for ip in $ips; do alt="$alt,IP:$ip"; i=$((i+1)); done

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$DIR/key.pem" -out "$DIR/cert.pem" \
  -days 825 -subj "/CN=nearby-ptt" \
  -addext "subjectAltName=$alt" 2>/dev/null

echo "certs written to $DIR"
echo "covering: $alt"
