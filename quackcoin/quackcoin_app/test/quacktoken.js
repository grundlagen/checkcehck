const { expect } = require("chai");
const { ethers, network } = require("hardhat");

describe("QuackToken mining", function () {
  let token;
  let treasury, lpVault;

  beforeEach(async function () {
    [treasury, lpVault] = await ethers.getSigners();
    const QuackToken = await ethers.getContractFactory("QuackToken");
    token = await QuackToken.deploy(treasury.address, lpVault.address);
    await token.deployed();
  });

  it("reverts when mined twice in the same block", async function () {
    await network.provider.send("evm_setAutomine", [false]);

    const tx1 = await token.mine();
    const tx2 = token.mine();

    await network.provider.send("evm_mine");

    await tx1.wait();
    await expect(tx2.wait()).to.be.revertedWith("already mined");

    await network.provider.send("evm_setAutomine", [true]);
  });

  it("allows mining in different blocks", async function () {
    await token.mine();
    await network.provider.send("evm_mine");
    await expect(token.mine()).to.not.be.reverted;
  });
});
