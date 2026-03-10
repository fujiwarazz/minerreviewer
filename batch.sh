OUT_DIR="data/processed/batch_2024"
mkdir -p "$OUT_DIR"

python - <<'PY' > /tmp/rows.txt
import pandas as pd, random
df = pd.read_parquet("/Users/peelsannaw/Downloads/crosseval/ICLR_2024.parquet")
rows = random.sample(range(len(df)), 20)
print("\n".join(str(r) for r in rows))
PY

while read -r ROW; do
  OUT="${OUT_DIR}/review_${ROW}.json"
  if [ -s "$OUT" ]; then
    echo "skip $ROW (exists)"
    continue
  fi
  peerreviewer review_paper --config configs/iclr.yaml \
    --parquet_path /Users/peelsannaw/Downloads/crosseval/ICLR_2024.parquet \
    --parquet_row "$ROW" --target_year 2024 \
    > "$OUT" 2> "${OUT_DIR}/review_${ROW}.err" || true
done < /tmp/rows.txt
