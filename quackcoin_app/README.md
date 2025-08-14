# QuackCoin App

This repository contains a simple proof‑of‑concept implementation of the core
components behind **Quack Coin**, a playful experiment blending audio‑oracles,
meme‑driven tokenomics and on‑chain finance.  The goal of this project is to
demonstrate how quack‑like recordings can be used to mint tokens, reward
participants and drive a chaotic yet bounded price chart.

The code here is intentionally minimal.  It includes:

* **Solidity contracts** (under `contracts/`) implementing the ERC‑20 token,
  an oracle interface and a basic egg hatching mechanism.
* A **Node.js backend** (in `backend/`) that illustrates how uploaded WAV
  files might be handled, queued and ultimately scored off‑chain.
* A **React Native front‑end skeleton** (in `mobile/`) that provides a
  starting point for the mobile user interface.

This repository is intended for educational and prototyping purposes only.
Before deploying anything to a production network, please conduct a full
security review and adapt the code to your specific requirements.

## Structure

```text
quackcoin_app/
├── README.md           – project overview (this file)
├── contracts/          – Solidity smart contracts
│   ├── EggHatcher.sol  – egg hatching logic and reward tiers
│   ├── MFCCOracle.sol  – lightweight score storage
│   └── QuackToken.sol  – ERC‑20 with tail emission and burn
├── backend/            – simple Node.js backend
│   ├── index.js        – Express server placeholder
│   └── package.json    – backend dependencies
└── mobile/             – React Native starter
    └── App.js         – minimal mobile UI
```

## Getting started

### Contracts

This project does not include a build system like Hardhat or Foundry out of
the box.  To compile and deploy the contracts you can:

1. Install a Solidity toolchain (e.g. Hardhat or Foundry).
2. Copy the contents of `contracts/` into your project.
3. Adjust import paths if needed.
4. Compile and deploy via your favourite framework.

### Backend

The backend is a barebones Express server that listens for file uploads and
prints information to the console.  It does not implement any persistent
storage or blockchain interactions yet – those would be added by connecting
to the smart contracts and a database.

To run the backend:

```bash
cd backend
npm install
node index.js
```

### Mobile

The mobile app is a minimal React Native component.  To run it you can:

1. Install `expo-cli` or set up a React Native environment.
2. Copy `App.js` into a new React Native project.
3. Start the development server with `npm start` or `expo start`.

The mobile UI currently only shows a simple form; integrating live audio
recording, the staking flow and playback would come later.

## License

MIT – see individual files for more details where applicable.