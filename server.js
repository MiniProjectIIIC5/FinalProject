const express    = require("express");
const mongoose   = require("mongoose");
const bcrypt     = require("bcrypt");
const jwt        = require("jsonwebtoken");
const cors       = require("cors");
const path       = require("path");
const { spawn }  = require("child_process");
require("dotenv").config();

const app = express();

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(cors({
  origin: [
    "https://profileverifier.netlify.app",
    "http://localhost:5000"
  ],
  methods: ["GET", "POST", "DELETE"],
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

// ─── MongoDB Atlas ────────────────────────────────────────────────────────────
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("✅  MongoDB Atlas connected"))
  .catch(err => { console.error("❌  MongoDB error:", err.message); process.exit(1); });

// ─── Schemas ──────────────────────────────────────────────────────────────────
const userSchema = new mongoose.Schema({
  username:  { type: String, required: true, unique: true, trim: true },
  email:     { type: String, required: true, unique: true, trim: true, lowercase: true },
  password:  { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
});
const User = mongoose.model("User", userSchema);

const verificationSchema = new mongoose.Schema({
  userId:     { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  profileUrl: { type: String, required: true },
  platform:   { type: String, required: true },
  prediction: { type: String, enum: ["Real", "Fake"], required: true },
  confidence: { type: Number, required: true },
  username:   { type: String, default: "" },
  score:      { type: Number, default: 0 },
  method:     { type: String, default: "heuristic" },
  status:     { type: String, default: "verified" },
  timestamp:  { type: Date, default: Date.now },
});
const Verification = mongoose.model("Verification", verificationSchema);

// ─── JWT Middleware ───────────────────────────────────────────────────────────
const authenticate = (req, res, next) => {
  const header = req.headers["authorization"];
  const token  = header && header.split(" ")[1];
  if (!token) return res.status(401).json({ message: "No token provided." });
  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ message: "Invalid or expired token." });
    req.user = user;
    next();
  });
};

// ─── Python helper ────────────────────────────────────────────────────────────
// Passes URL + optional profile metadata as CLI args (works reliably on Windows).
// profile shape (all optional): { followers, following, posts, has_pic, verified, bio_len }
function runPython(url, profile = {}) {
  return new Promise((resolve, reject) => {
    const pyPath = process.env.PYTHON_PATH || "python";
    const script = path.join(__dirname, "ml", "predict.py");
    const proc   = spawn(pyPath, [script, url, JSON.stringify(profile)]);

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", d => { stdout += d.toString(); });
    proc.stderr.on("data", d => {
      const msg = d.toString();
      if (!msg.includes("Warning") && !msg.includes("DeprecationWarning")) {
        stderr += msg;
      }
    });

    proc.on("error", err => {
      reject(new Error(
        `Could not start Python (${pyPath}): ${err.message}. Check PYTHON_PATH in .env`
      ));
    });

    proc.on("close", code => {
      const trimmed = stdout.trim();
      if (!trimmed) {
        reject(new Error(
          `Python returned no output (exit ${code}).` +
          (stderr ? ` Stderr: ${stderr}` : "")
        ));
        return;
      }
      try {
        resolve(JSON.parse(trimmed));
      } catch {
        reject(new Error(`Could not parse Python output: ${trimmed}`));
      }
    });
  });
}

// ─── Auth Routes ──────────────────────────────────────────────────────────────
app.post("/api/signup", async (req, res) => {
  try {
    const { username, email, password } = req.body;
    if (!username || !email || !password)
      return res.status(400).json({ message: "All fields are required." });
    if (password.length < 6)
      return res.status(400).json({ message: "Password must be at least 6 characters." });

    const exists = await User.findOne({ $or: [{ email }, { username }] });
    if (exists) return res.status(409).json({ message: "Email or username already in use." });

    const hash = await bcrypt.hash(password, 12);
    await User.create({ username, email, password: hash });
    res.status(201).json({ message: "Account created! Please sign in." });
  } catch (err) {
    console.error("Signup error:", err);
    res.status(500).json({ message: "Server error. Please try again." });
  }
});

app.post("/api/signin", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ message: "Email and password are required." });

    const user = await User.findOne({ email });
    if (!user) return res.status(401).json({ message: "Invalid email or password." });

    const match = await bcrypt.compare(password, user.password);
    if (!match) return res.status(401).json({ message: "Invalid email or password." });

    const token = jwt.sign(
      { id: user._id, username: user.username, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: "7d" }
    );
    res.json({ token, user: { id: user._id, username: user.username, email: user.email } });
  } catch (err) {
    console.error("Signin error:", err);
    res.status(500).json({ message: "Server error. Please try again." });
  }
});

// ─── Verify Route ─────────────────────────────────────────────────────────────
app.post("/api/verify", authenticate, async (req, res) => {
  try {
    // ✅ FIX: profile is now properly destructured from req.body
    const { url, platform, profile } = req.body;

    if (!url || !platform)
      return res.status(400).json({ message: "URL and platform are required." });

    try { new URL(url); } catch {
      return res.status(400).json({ message: "Please enter a valid URL." });
    }

    // profile is optional — safe default to empty object if not sent by frontend
    const profileData = (profile && typeof profile === "object") ? profile : {};

    const result = await runPython(url, profileData);

    if (result.error && !result.prediction) {
      return res.status(500).json({ message: "Python error: " + result.error });
    }

    const doc = await Verification.create({
      userId:     req.user.id,
      profileUrl: url,
      platform,
      prediction: result.prediction,
      confidence: result.confidence,
      username:   result.username || "",
      score:      result.score    || 0,
      method:     result.method   || "heuristic",
    });

    res.json({
      id:         doc._id,
      prediction: result.prediction,
      confidence: result.confidence,
      username:   result.username,
      method:     result.method,
      features:   result.features || {},
    });
  } catch (err) {
    console.error("Verify error:", err.message);
    res.status(500).json({ message: "Verification failed: " + err.message });
  }
});

// ─── History Routes ───────────────────────────────────────────────────────────
app.get("/api/history", authenticate, async (req, res) => {
  try {
    const { platform, result } = req.query;
    const query = { userId: req.user.id };
    if (platform && platform !== "all") query.platform   = platform;
    if (result   && result   !== "all") query.prediction = result === "fake" ? "Fake" : "Real";
    const history = await Verification.find(query).sort({ timestamp: -1 }).limit(200);
    res.json(history);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

app.delete("/api/history/:id", authenticate, async (req, res) => {
  try {
    const doc = await Verification.findOneAndDelete({ _id: req.params.id, userId: req.user.id });
    if (!doc) return res.status(404).json({ message: "Record not found." });
    res.json({ message: "Deleted successfully." });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// ─── Stats Route ──────────────────────────────────────────────────────────────
app.get("/api/stats", authenticate, async (req, res) => {
  try {
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    const [total, fake] = await Promise.all([
      Verification.countDocuments({ userId: req.user.id, timestamp: { $gte: todayStart } }),
      Verification.countDocuments({ userId: req.user.id, timestamp: { $gte: todayStart }, prediction: "Fake" }),
    ]);
    res.json({ total, fake });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// ─── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀  Server running at http://localhost:${PORT}`));