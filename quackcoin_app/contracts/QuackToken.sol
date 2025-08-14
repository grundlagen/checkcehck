// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/extensions/draft-ERC20Permit.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title QuackCoin
/// @notice A humorous ERC‑20 token with tail‑emission mining, transfer burn
/// and protocol‑owned liquidity.  This contract is deliberately simple
/// and intended for experimentation.  It implements a halving schedule
/// similar to Bitcoin but allows tuning via an owner (e.g. a DAO proxy).
contract QuackToken is ERC20Permit, Ownable {
    // ---------------------------------------------------------------------
    // Emission parameters
    // ---------------------------------------------------------------------

    /// @notice Genesis supply minted to the deployer at construction.
    uint256 public constant GENESIS_SUPPLY = 1_000_000e18;

    /// @notice Current emission reward per block.  Starts at 42 QC and
    /// halves each `halvingInterval` blocks.  When halved many times it
    /// will eventually converge towards zero.
    uint256 public emissionPerBlock = 42e18;

    /// @notice Block number at which the next halving occurs.
    uint256 public nextHalvingBlock;

    /// @notice Interval in blocks between halvings.  Roughly two years
    /// on an optimistic L2 (~2.3 million blocks).  Adjustable by the owner.
    uint256 public halvingInterval = 2_300_000;

    /// @notice Fraction (in basis points) of mining rewards that go directly
    /// to the protocol‑owned liquidity vault.  The remainder is minted
    /// to the treasury address.  Default is 10% (1000 BPS).
    uint16 public constant LP_SHARE_BPS = 1000;

    // ---------------------------------------------------------------------
    // Burn configuration
    // ---------------------------------------------------------------------

    /// @notice Percentage (in basis points) of every transfer to burn.
    /// The owner can set this up to a maximum of 5%.  Default is 1%.
    uint16 public transferBurnBps = 100;

    /// @dev Burn address used by the `_transfer` override.
    address public constant BURN = 0x000000000000000000000000000000000000dEaD;

    // ---------------------------------------------------------------------
    // Treasury / liquidity addresses
    // ---------------------------------------------------------------------

    /// @notice Address receiving the non‑LP portion of minted rewards.
    address public treasury;

    /// @notice Address receiving the LP share of minted rewards.  This
    /// address could be a contract that automates the provision of
    /// liquidity into an AMM pair (QC/ETH) and holds the resulting LP
    /// tokens.
    address public lpVault;

    // ---------------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------------

    /// @notice Emitted when a new block reward is minted via `mine()`.
    /// @param miner Address that called the mine function
    /// @param reward Total amount minted in this call
    /// @param toLP Amount of the reward allocated to the LP vault
    event Mined(address indexed miner, uint256 reward, uint256 toLP);

    // ---------------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------------

    constructor(address _treasury, address _lpVault)
        ERC20("QuackCoin", "QC")
        ERC20Permit("QuackCoin")
    {
        treasury = _treasury;
        lpVault = _lpVault;
        nextHalvingBlock = block.number + halvingInterval;
        _mint(msg.sender, GENESIS_SUPPLY);
    }

    // ---------------------------------------------------------------------
    // Internal transfer with burn
    // ---------------------------------------------------------------------

    /// @dev Overrides the default ERC20 `_transfer` to implement a burn
    /// percentage on every transfer.  A portion of the sent amount is
    /// permanently destroyed by sending it to the `BURN` address.
    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal override {
        uint256 burnAmt = (amount * transferBurnBps) / 10_000;
        uint256 sendAmt = amount - burnAmt;
        super._transfer(from, to, sendAmt);
        if (burnAmt > 0) {
            super._transfer(from, BURN, burnAmt);
        }
    }

    // ---------------------------------------------------------------------
    // Mining function
    // ---------------------------------------------------------------------

    /// @notice Permissionless mining function.  Anyone may call this
    /// once per block to mint the emission reward.  A portion goes to the
    /// LP vault and the remainder to the treasury.
    function mine() external {
        // Halving: if the current block number is past the scheduled
        // halving block then halve the emission and set the next halving.
        if (block.number >= nextHalvingBlock && emissionPerBlock > 0) {
            emissionPerBlock = emissionPerBlock / 2;
            nextHalvingBlock = block.number + halvingInterval;
        }

        uint256 reward = emissionPerBlock;
        require(reward > 0, "Mining ended");

        uint256 toLP = (reward * LP_SHARE_BPS) / 10_000;
        uint256 toTreasury = reward - toLP;

        // Mint directly to LP vault and treasury.  These addresses
        // determine how the QC tokens are managed (e.g. providing
        // liquidity, swapping for ETH, funding grants).
        _mint(lpVault, toLP);
        _mint(treasury, toTreasury);

        emit Mined(msg.sender, reward, toLP);
    }

    // ---------------------------------------------------------------------
    // Admin functions
    // ---------------------------------------------------------------------

    /// @notice Updates the treasury address.  Only callable by the owner.
    function setTreasury(address _treasury) external onlyOwner {
        treasury = _treasury;
    }

    /// @notice Updates the LP vault address.  Only callable by the owner.
    function setLPVault(address _lpVault) external onlyOwner {
        lpVault = _lpVault;
    }

    /// @notice Sets the transfer burn rate (in basis points).  Must not
    /// exceed 500 (5%) to avoid excessive burns.  Only callable by the owner.
    function setBurnRate(uint16 bps) external onlyOwner {
        require(bps <= 500, "Burn rate too high");
        transferBurnBps = bps;
    }

    /// @notice Sets a new halving interval.  Must be at least 200k blocks.
    /// Only callable by the owner.  The next halving block is reset based
    /// on the current block number.
    function setHalvingInterval(uint256 blocks_) external onlyOwner {
        require(blocks_ >= 200_000, "Interval too low");
        halvingInterval = blocks_;
        nextHalvingBlock = block.number + blocks_;
    }
}