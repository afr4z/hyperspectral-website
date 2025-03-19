
const express = require('express');
const multer = require('multer');
const path = require('path');
const cors = require('cors');
const { spawn } = require('child_process');
const crypto = require('crypto');

const app = express();
app.use(cors());

// Multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, 'uploads/'),
    filename: (req, file, cb) => {
        if (!req.randomFilename) {
            req.randomFilename = crypto.randomBytes(16).toString('hex');
        }
        const ext = path.extname(file.originalname);
        cb(null, req.randomFilename + ext);
    }
});

const upload = multer({ storage });

// API to predict moisture
app.post('/api/predict', upload.fields([{ name: 'img' }, { name: 'hdr' }]), (req, res) => {
    if (!req.files || !req.files.img || !req.files.hdr) {
        return res.status(400).json({ error: "Both .img and .hdr files are required" });
    }

    const imgPath = path.join(__dirname, 'uploads', req.files.img[0].filename);
    const hdrPath = path.join(__dirname, 'uploads', req.files.hdr[0].filename);
    console.log(imgPath, hdrPath);
    const pythonProcess = spawn('python3', ['predict.py', imgPath, hdrPath]);

    let result = '';
    pythonProcess.stdout.on('data', (data) => result += data.toString());
    pythonProcess.stderr.on('data', (data) => console.error(`Python error: ${data.toString()}`));

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        console.log(`Raw Python output: ${result}`);
        try {
            res.json(JSON.parse(result.trim())); // Trim whitespace and parse JSON
        } catch (err) {
            console.error("Error parsing Python response:", err);
            res.status(500).json({ error: "Error parsing Python response", details: result });
        }
    });
});

// Start server
app.listen(5000, () => console.log("Server running on http://127.0.0.1:5000"));

