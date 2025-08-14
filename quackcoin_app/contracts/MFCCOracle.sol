// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title MFCCOracle
/// @notice Lightweight on‑chain storage for MFCC similarity scores.  An
/// off‑chain worker is expected to compute the MFCC vectors and call
/// `setScores` and `setReference` to publish the results.  Eggs can
/// retrieve scores from this contract to determine rewards.  Only the
/// designated updater address can write to the mappings.
contract MFCCOracle {
    struct Scores {
        uint16 sRef;
        uint16 sAvg;
    }

    /// @notice Mapping of WAV hash → scores (scaled 0–10000).
    mapping(bytes32 => Scores) public scores;

    /// @notice Mapping of the Unix day (floor of timestamp/86400) to the
    /// reference WAV hash for that day.
    mapping(uint256 => bytes32) public dayRef;

    /// @notice Address allowed to publish scores and reference hashes.
    address public updater;

    event ScoresSet(bytes32 indexed wavHash, uint16 sRef, uint16 sAvg);
    event RefDuckSet(uint256 indexed day, bytes32 wavHash);

    modifier onlyUpdater() {
        require(msg.sender == updater, "Not updater");
        _;
    }

    constructor(address _updater) {
        updater = _updater;
    }

    /// @notice Sets similarity scores for a given WAV hash.  Only callable
    /// by the updater.
    function setScores(
        bytes32 wavHash,
        uint16 sRef,
        uint16 sAvg
    ) external onlyUpdater {
        scores[wavHash] = Scores(sRef, sAvg);
        emit ScoresSet(wavHash, sRef, sAvg);
    }

    /// @notice Sets the reference WAV hash for a given Unix day.  Only
    /// callable by the updater.  This function does not enforce a single
    /// reference per day; it is the updater's responsibility to avoid
    /// overwriting a reference.
    function setReference(uint256 unixDay, bytes32 wavHash) external onlyUpdater {
        dayRef[unixDay] = wavHash;
        emit RefDuckSet(unixDay, wavHash);
    }

    /// @notice Returns the scores for a WAV hash.  If no scores are
    /// available this function returns zeros.
    function getScores(bytes32 wavHash) external view returns (uint16, uint16) {
        Scores memory s = scores[wavHash];
        return (s.sRef, s.sAvg);
    }
}