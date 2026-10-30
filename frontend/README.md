# AnimalVox React Frontend

This is the primary AnimalVox web interface. It records or uploads audio, sends it to
the FastAPI backend, and renders live translation, species-routing, health, emotion,
breed, and session-history data.

## Development

Start the backend from the repository root:

```sh
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```sh
pnpm install
pnpm run dev
```

Vite runs on `http://localhost:3000` and proxies `/api` to port 8000. To point at a
remote API, create `frontend/.env.local` with:

```env
VITE_API_BASE_URL=https://your-api.example.com
```

## Production

Run `pnpm run build`. The FastAPI server detects `frontend/dist` at startup and serves
the compiled React application, including client-side routes, at `http://localhost:8000`.

Original recordings are uploaded only for analysis. The browser keeps interpretation
history locally; it does not retain audio files.
