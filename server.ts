import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import Database from "better-sqlite3";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const db = new Database("wardrobe.db");

// Initialize database
db.exec(`
  CREATE TABLE IF NOT EXISTS wardrobe (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_data TEXT,
    analysis TEXT,
    recommendations TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    content TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '10mb' }));

  // API Routes
  app.get("/api/wardrobe", (req, res) => {
    const items = db.prepare("SELECT * FROM wardrobe ORDER BY created_at DESC").all();
    res.json(items.map(item => ({
      ...item,
      analysis: JSON.parse(item.analysis as string),
      recommendations: JSON.parse(item.recommendations as string)
    })));
  });

  app.post("/api/wardrobe", (req, res) => {
    const { image_data, analysis, recommendations } = req.body;
    const info = db.prepare(
      "INSERT INTO wardrobe (image_data, analysis, recommendations) VALUES (?, ?, ?)"
    ).run(image_data, JSON.stringify(analysis), JSON.stringify(recommendations));
    res.json({ id: info.lastInsertRowid });
  });

  app.get("/api/chat", (req, res) => {
    const history = db.prepare("SELECT * FROM chat_history ORDER BY created_at ASC").all();
    res.json(history);
  });

  app.post("/api/chat", (req, res) => {
    const { role, content } = req.body;
    db.prepare("INSERT INTO chat_history (role, content) VALUES (?, ?)").run(role, content);
    res.json({ status: "ok" });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, "dist")));
    app.get("*", (req, res) => {
      res.sendFile(path.join(__dirname, "dist", "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
