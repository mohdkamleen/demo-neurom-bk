import express from "express";
import fs from "fs";
import csv from "csv-parser";
import * as math from "mathjs";
import * as tf from "@tensorflow/tfjs";
import { ChartJSNodeCanvas } from "chartjs-node-canvas";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";

const app = express();
const PORT = 5000;
app.use(express.json());
app.use(cors());

// Fix __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------- Helper Functions ----------------

// Load CSV file
function loadCSV(filePath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on("data", (data) => rows.push(data))
      .on("end", () => resolve(rows))
      .on("error", reject);
  });
}

// Build features
function buildFeatures(data) {
  const lagKeys = [1, 2, 3, 4, 6, 8, 12];
  const CGM = data.map((r) => parseFloat(r["CGM (mg / dl)"] || 0));

  return data
    .map((r, i) => {
      const f = {};

      lagKeys.forEach((k) => (f[`lag${k}`] = CGM[i - k] ?? CGM[0]));

      const safeMean = (arr) => (arr.length ? math.mean(arr) : 0);
      const safeStd = (arr) => (arr.length > 1 ? math.std(arr) : 0);
      const slice = (n) => CGM.slice(Math.max(0, i - n), i);

      f["roll4_mean"] = safeMean(slice(4));
      f["roll8_mean"] = safeMean(slice(8));
      f["roll12_mean"] = safeMean(slice(12));
      f["roll4_std"] = safeStd(slice(4));
      f["roll8_std"] = safeStd(slice(8));

      const date = new Date(r["Date"]);
      const minutes = date.getHours() * 60 + date.getMinutes();
      f["tod_sin"] = Math.sin((2 * Math.PI * minutes) / 1440);
      f["tod_cos"] = Math.cos((2 * Math.PI * minutes) / 1440);

      const nutrCols = [
        "Calories",
        "Total Fat",
        "Saturated Fat",
        "Trans Fat",
        "Cholesterol",
        "Sodium",
        "Total Carbohydrates",
        "Dietary Fiber",
        "Sugars",
        "Protein",
      ];
      nutrCols.forEach((k) => (f[k] = parseFloat(r[k] || 0)));

      f["CGM_next"] = CGM[i + 1] ?? CGM[i];
      f["CGM"] = CGM[i];
      f["Date"] = date;
      return f;
    })
    .slice(12);
}

// R² score
function r2_score(yTrue, yPred) {
  const meanY = math.mean(yTrue);
  const ssTot = math.sum(yTrue.map((v) => (v - meanY) ** 2));
  const ssRes = math.sum(yTrue.map((v, i) => (v - yPred[i]) ** 2));
  return 1 - ssRes / ssTot;
}
 
// Train + predict
async function trainModel(csvPath) {
  const data = await loadCSV(csvPath);
  if (data.length < 20) throw new Error("CSV has too few rows to train the model.");

  const features = buildFeatures(data);
  const X = features.map((r) =>
    Object.values(r).filter((_, i, arr) => i < arr.length - 3)
  );
  const y = features.map((r) => r["CGM_next"]);

  const split = Math.floor(0.8 * X.length);
  const X_train = X.slice(0, split);
  const X_test = X.slice(split);
  const y_train = y.slice(0, split);
  const y_test = y.slice(split);
  const time_test = features.slice(split).map((f) => f["Date"]);

  const inputDim = X_train[0].length;

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, inputShape: [inputDim], activation: "relu" }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  const xs = tf.tensor2d(X_train);
  const ys = tf.tensor2d(y_train, [y_train.length, 1]);
  const xsTest = tf.tensor2d(X_test);

  await model.fit(xs, ys, { epochs: 50, batchSize: 32, verbose: 0 });

  const preds = model.predict(xsTest);
  const y_pred = Array.from(preds.dataSync());

  // Metrics
  const mse = math.mean(y_test.map((v, i) => (v - y_pred[i]) ** 2));
  const rmse = Math.sqrt(mse);
  const mae = math.mean(y_test.map((v, i) => Math.abs(v - y_pred[i])));
  const r2 = r2_score(y_test, y_pred);
  const mape =
    math.mean(y_test.map((v, i) => Math.abs((v - y_pred[i]) / v))) * 100;
  const accuracy = 100 - mape;

  // Window table
  const windowPoints = 16;
  const timeWindow = time_test.slice(0, windowPoints);
  const y_pred_window = y_pred.slice(0, windowPoints);
  const y_test_window = y_test.slice(0, windowPoints);
  const table = timeWindow.map((t, i) => ({
    Time: t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    Predicted: y_pred_window[i].toFixed(2),
    Actual: y_test_window[i]?.toFixed(2) || "",
  }));
 
  return {
    metrics: {
      MSE: mse.toFixed(2),
      RMSE: rmse.toFixed(2),
      MAE: mae.toFixed(2),
      R2: r2.toFixed(3),
      Accuracy: accuracy.toFixed(2) + "%",
    },
    table, 
  };
}

// Setup multer for file uploads
const upload = multer({ dest: path.join(__dirname, "uploads/") });

// ---------------- API ROUTE ----------------
app.post("/predict", upload.single("file"), async (req, res) => {
  try {
    const csvPath = req.file.path
    const result = await trainModel(csvPath);
    res.json({
      status: "success", 
      metrics: result.metrics,
      table: result.table,
      chart: result.chart,
    });
  } catch (err) {
    console.error("❌ Error in /predict:", err);
    res.status(500).json({ status: "error", message: err.message });
  }
});

app.get("/", (_, res) => {
  res.send("✅ CGM Prediction API is running. Use POST /predict.");
});

// Auto-run
app.listen(PORT, async () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
