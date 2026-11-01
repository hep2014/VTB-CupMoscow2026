// Recalculate the scenario model without presentation/Figma build dependencies.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const model = require('./financial-model.js');
const value = model.viewModel();
assert.equal(model.customerFundsRub(2741000000, 0.15, 0.12), 49338000);
assert.equal(value.budget.totalRub, 42931250);
assert.equal(value.budget.headroomRub, 7068750);
assert.equal(value.bankQuarter.calculated, false);
assert.equal(value.bankReturn.calculated, false);
assert.throws(() => model.customerFundsRub(2741000000, 1.5, 0.12));
const result = {status: 'Scenario assumptions, not measured bank savings', ...value};
const target = path.join(__dirname, 'scenarios.json');
fs.writeFileSync(target, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({checks: 'passed', output: 'research/finance/scenarios.json', budgetRub: value.budget.totalRub}));
