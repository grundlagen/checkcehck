const express = require('express');
const multer = require('multer');

// Configure multer to store files in memory.  In a real app you might
// stream the file to IPFS or local disk, but this prototype merely
// demonstrates how uploads could be handled.
const upload = multer({ storage: multer.memoryStorage() });

const app = express();
const PORT = process.env.PORT || 3000;

// Simple health check
app.get('/', (req, res) => {
  res.json({ message: 'QuackCoin backend running' });
});

// Endpoint to submit a WAV file.  The uploaded file will be available as
// `req.file`.  A real implementation would send this file to IPFS,
// compute the MFCC vector off‑chain and enqueue it for scoring.
app.post('/submit', upload.single('wav'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  // Log the filename and size
  console.log(`Received file: ${req.file.originalname}, ${req.file.size} bytes`);
  // Respond with a placeholder CID (this would normally be the IPFS hash)
  res.json({ status: 'queued', cid: 'QmFakeCidForDemo' });
});

// Start the server
app.listen(PORT, () => {
  console.log(`QuackCoin backend listening on port ${PORT}`);
});