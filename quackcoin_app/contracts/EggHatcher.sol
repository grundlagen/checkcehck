// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";

/// @title EggHatcher
/// @notice Implements the recording and reward logic for the QuackCoin
/// audio game.  Users submit an egg (NFT) along with a WAV hash and
/// potentially a stake; the contract consults an MFCC oracle for
/// similarity scores and mints or burns tokens accordingly.  Reward tiers
/// have a "soft landing" so that low scores receive a partial refund
/// rather than losing everything.

interface IQuackToken {
    function burn(address from, uint256 amount) external;
    function mint(address to, uint256 amount) external;
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IERC721Minimal {
    function ownerOf(uint256 tokenId) external view returns (address);
    function burn(uint256 tokenId) external;
    function mint(address to) external returns (uint256);
}

interface IMFCCOracle {
    /// @notice Returns similarity scores scaled 0–10000 (i.e. 0.9312 → 9312)
    function getScores(bytes32 wavHash) external view returns (uint16, uint16);
}

contract EggHatcher is Ownable {
    // ---------------------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------------------
    /// @notice Record fee (in QC) required to submit a WAV.  The fee is
    /// burned (70%) and partially refunded (30%) for a fail.  The DAO can
    /// update this value.
    uint256 public recordFee = 1e15; // 0.001 QC with 18 decimals

    /// @notice Basis points of the record fee burned on a failed submission.
    /// The remainder is refunded.  Default is 60% burn, 40% refund.
    uint256 public burnPct = 6000; // 60% in BPS

    /// @notice Daily reward pool allocated to hatching.  Refilled every
    /// 24 hours.  Rewards are paid from this balance.
    uint256 public poolDaily = 5e18; // 5 QC per day

    /// @dev Timestamp of the last time the pool was refilled.
    uint256 public lastPoolRefill;

    // ---------------------------------------------------------------------
    // External references
    // ---------------------------------------------------------------------
    IQuackToken public immutable QC;
    IERC721Minimal public immutable EggNFT;
    IERC721Minimal public immutable DuckNFT;
    IMFCCOracle public immutable Oracle;

    // ---------------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------------
    event Hatched(
        address indexed user,
        uint256 eggId,
        uint256 reward,
        uint16 deltaBps
    );

    // ---------------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------------
    constructor(
        address qc,
        address eggNFT,
        address duckNFT,
        address oracle
    ) {
        QC = IQuackToken(qc);
        EggNFT = IERC721Minimal(eggNFT);
        DuckNFT = IERC721Minimal(duckNFT);
        Oracle = IMFCCOracle(oracle);
        lastPoolRefill = block.timestamp;
    }

    // ---------------------------------------------------------------------
    // Public functions
    // ---------------------------------------------------------------------
    /// @notice Submit an egg along with a WAV hash for evaluation.  The
    /// caller must own the egg NFT.  A record fee in QC is burned from
    /// the caller.  Depending on the similarity delta (sRef - sAvg) a
    /// reward is paid from the pool and optionally a Duck NFT is minted.
    /// @param eggId Token ID of the egg NFT to hatch
    /// @param wavHash Hash of the uploaded WAV file
    function submitQuack(uint256 eggId, bytes32 wavHash) external {
        // Ensure caller owns the egg
        require(EggNFT.ownerOf(eggId) == msg.sender, "Not egg owner");

        // Burn the record fee from the user up front
        QC.burn(msg.sender, recordFee);

        // Refill the pool once per day
        if (block.timestamp - lastPoolRefill > 1 days) {
            QC.mint(address(this), poolDaily);
            lastPoolRefill = block.timestamp;
        }

        // Retrieve scores from the oracle.  sRef and sAvg are scaled by
        // 10000; convert to signed delta in BPS (basis points).
        (uint16 sRef, uint16 sAvg) = Oracle.getScores(wavHash);
        require(sRef > 0 && sAvg > 0, "Scores missing");
        int16 delta = int16(int(sRef) - int(sAvg));

        uint256 reward;
        uint16 absDelta = uint16(delta >= 0 ? delta : -delta);

        // Reward tiers (Delta in BPS):
        // ≤ 0       : fail – refund 40% of the record fee
        // < 500    : 0.1 QC
        // < 900    : 0.3 QC
        // < 1500   : 1 QC
        // ≥ 1500   : 2 QC + Duck NFT
        if (delta <= 0) {
            uint256 refund = (recordFee * (10_000 - burnPct)) / 10_000;
            reward = refund;
        } else if (absDelta < 500) {
            reward = 1e17; // 0.1 QC
        } else if (absDelta < 900) {
            reward = 3e17; // 0.3 QC
        } else if (absDelta < 1500) {
            reward = 1e18; // 1 QC
        } else {
            reward = 2e18; // 2 QC
            DuckNFT.mint(msg.sender);
        }

        // Pay reward from available pool balance (if any).  If the pool
        // balance is insufficient, pay only what is available.
        if (reward > 0) {
            uint256 bal = QC.balanceOf(address(this));
            if (reward > bal) reward = bal;
            QC.transfer(msg.sender, reward);
        }

        // Always burn the egg NFT
        EggNFT.burn(eggId);

        emit Hatched(msg.sender, eggId, reward, uint16(delta));
    }

    // ---------------------------------------------------------------------
    // Admin functions (access control omitted for brevity; add onlyOwner)
    // ---------------------------------------------------------------------
    function setRecordFee(uint256 fee) external onlyOwner {
        // In production, restrict this function to a DAO or owner
        recordFee = fee;
    }

    function setBurnPct(uint256 bps) external onlyOwner {
        require(bps <= 9000, "Too high");
        burnPct = bps;
    }

    function setDailyPool(uint256 amount) external onlyOwner {
        poolDaily = amount;
    }
}