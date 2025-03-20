
const express = require('express');
const multer = require('multer');
const path = require('path');
const cors = require('cors');
const { spawn } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const https = require('https');

const app = express();
app.use(cors());

const options = {
	key: fs.readFileSync('./ssl/key.pem'),
	cert: fs.readFileSync('./ssl/cert.pem')
};
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

app.get('/', (req, res) => {
	console.log(req.body);
    res.send('GET request received at /');
});
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

https.createServer(options, app).listen(5000, '0.0.0.0', () => {
    console.log("HTTPS Server running on https://0.0.0.0:5000");
});
