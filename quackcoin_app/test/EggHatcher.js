const { expect } = require("chai");

describe("EggHatcher access control", function () {
  let eggHatcher, owner, other;

  beforeEach(async function () {
    [owner, other] = await ethers.getSigners();
    const EggHatcher = await ethers.getContractFactory("EggHatcher");
    eggHatcher = await EggHatcher.deploy(owner.address, owner.address, owner.address, owner.address);
    await eggHatcher.deployed();
  });

  it("non-owner cannot setRecordFee", async function () {
    await expect(eggHatcher.connect(other).setRecordFee(2)).to.be.revertedWith("Ownable: caller is not the owner");
  });

  it("non-owner cannot setBurnPct", async function () {
    await expect(eggHatcher.connect(other).setBurnPct(5000)).to.be.revertedWith("Ownable: caller is not the owner");
  });

  it("non-owner cannot setDailyPool", async function () {
    await expect(eggHatcher.connect(other).setDailyPool(10)).to.be.revertedWith("Ownable: caller is not the owner");
  });
});
