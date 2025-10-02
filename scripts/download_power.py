import argparse, json, os, requests, yaml

def main(cfg):
    lat, lon = cfg["data"]["lat"], cfg["data"]["lon"]
    start, end = cfg["data"]["start"], cfg["data"]["end"]
    params = cfg["data"]["parameters"]
    raw_path = cfg["data"]["raw_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    q = {
        "parameters": ",".join(params),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON"
    }
    r = requests.get(url, params=q, timeout=120)
    r.raise_for_status()
    with open(raw_path, "w") as f:
        json.dump(r.json(), f)
    print(f"Saved {raw_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
