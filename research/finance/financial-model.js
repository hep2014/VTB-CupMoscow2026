// Financial assumptions are editable here, independently of the slide layout.
// CBR is a whole-market observation. FDH is a synthetic experiment and is NOT
// used as the real-world loss-prevention coefficient in this model.
const financialModel = (()=>{
  const evidence = typeof financialEvidence!=='undefined' ? financialEvidence : require('./financial-evidence.js');
  const parameters = {
    observedCbrSbpRub:2741024490,
    marketRub:2741000000, // 2.741 billion RUB, rounded as requested in the brief.
    marketPeriod:'II квартал 2026',
    marketScope:'Россия, физические лица, СБП',
    cbrUrl:'https://www.cbr.ru/statistics/ib/review_2q_2026/',
    budget:{personnelRub:24831250,otherRub:12600000,reserveRub:5500000,capRub:50000000},
    scenarios:[
      {key:'conservative',name:'Консервативный',bankShare:0.10,additionalPrevention:0.06},
      {key:'base',name:'Базовый',bankShare:0.15,additionalPrevention:0.12},
      {key:'optimistic',name:'Оптимистичный',bankShare:0.20,additionalPrevention:0.18}
    ],
    sensitivity:{bankShares:[0.10,0.15,0.20],preventionRates:[0.06,0.12,0.18]},
    // No unknown bank coefficient is silently treated as zero or a known fact.
    bank:{compensationShare:null,quarterlyFewerCases:null,costPerCaseRub:null,
      additionalQuarterlyOpExRub:null,quarterlyMaintenanceRub:null,
      quarterlyRetentionBenefitRub:null,actualInvestmentRub:null,horizonQuarters:null}
  };
  const finite=(n,label)=>{if(typeof n!=='number'||!Number.isFinite(n))throw Error(label+' must be a finite number');return n;};
  const share=(n,label)=>{finite(n,label);if(n<0||n>1)throw Error(label+' must be between 0 and 1');return n;};
  function customerFundsRub(marketRub,bankShare,additionalPrevention){
    finite(marketRub,'marketRub');if(marketRub<0)throw Error('marketRub must be nonnegative');
    return marketRub*share(bankShare,'bankShare')*share(additionalPrevention,'additionalPrevention');
  }
  function budgetTotals(){const b=parameters.budget,plannedRub=b.personnelRub+b.otherRub,totalRub=plannedRub+b.reserveRub;return{...b,plannedRub,totalRub,headroomRub:b.capRub-totalRub};}
  function scenarioResults(){return parameters.scenarios.map(s=>({...s,customerFundsRub:customerFundsRub(parameters.marketRub,s.bankShare,s.additionalPrevention)}));}
  function sensitivity(){return parameters.sensitivity.bankShares.map(s=>({bankShare:s,values:parameters.sensitivity.preventionRates.map(e=>({additionalPrevention:e,customerFundsRub:customerFundsRub(parameters.marketRub,s,e)}))}));}
  function bankQuarterlyNetRub(customerFunds,inputs=parameters.bank){
    const required=['compensationShare','quarterlyFewerCases','costPerCaseRub','additionalQuarterlyOpExRub','quarterlyMaintenanceRub','quarterlyRetentionBenefitRub'];
    const missing=required.filter(k=>inputs[k]===null||inputs[k]===undefined);if(missing.length)return{calculated:false,missing,value:null};
    finite(customerFunds,'customerFunds');share(inputs.compensationShare,'compensationShare');
    for(const k of required.slice(1))finite(inputs[k],k);
    for(const k of ['costPerCaseRub','additionalQuarterlyOpExRub','quarterlyMaintenanceRub'])if(inputs[k]<0)throw Error(k+' must be nonnegative');
    return{calculated:true,missing:[],value:customerFunds*inputs.compensationShare+inputs.quarterlyFewerCases*inputs.costPerCaseRub+inputs.quarterlyRetentionBenefitRub-inputs.additionalQuarterlyOpExRub-inputs.quarterlyMaintenanceRub};
  }
  // ROI is available only for explicitly provided, net BANK cash flows per
  // quarter, an explicit horizon, and an actual initial cash investment.
  // Quarterly customer funds from the scenario are never passed automatically.
  function bankReturn({investmentRub=null,netBankCashFlowsRub=null,horizonQuarters=null}={}){
    if(investmentRub===null||netBankCashFlowsRub===null||horizonQuarters===null)return{calculated:false,roi:null,paybackQuarter:null,reason:'Нужны вложения, банковские денежные потоки и горизонт'};
    finite(investmentRub,'investmentRub');if(investmentRub<=0)throw Error('investmentRub must be positive');
    if(!Number.isInteger(horizonQuarters)||horizonQuarters<1||!Array.isArray(netBankCashFlowsRub)||netBankCashFlowsRub.length!==horizonQuarters)throw Error('One explicit bank cash flow per horizon quarter is required');
    let cumulative=0,paybackQuarter=null;
    netBankCashFlowsRub.forEach((v,i)=>{cumulative+=finite(v,'net bank cash flow');if(paybackQuarter===null&&cumulative>=investmentRub)paybackQuarter=i+1;});
    return{calculated:true,roi:(cumulative-investmentRub)/investmentRub,paybackQuarter,horizonQuarters,netBankCashFlowsRub:[...netBankCashFlowsRub]};
  }
  const b=evidence.policies.baseline,t=evidence.policies.adaptive;
  const experiment={recallBaseline:b.count_recall,recallTrustflow:t.count_recall,recallGainPp:(t.count_recall-b.count_recall)*100,tpGain:t.tp-b.tp,unflaggedBaseline:b.unflagged_fraud_amount,unflaggedTrustflow:t.unflagged_fraud_amount,unflaggedReduction:1-t.unflagged_fraud_amount/b.unflagged_fraud_amount};
  function viewModel(){const scenarios=scenarioResults(),base=scenarios.find(s=>s.key==='base');return{parameters,budget:budgetTotals(),scenarios,base,sensitivity:sensitivity(),experiment,evidence,bankQuarter:bankQuarterlyNetRub(base.customerFundsRub),bankReturn:bankReturn(),
    onePpPreventionRub:customerFundsRub(parameters.marketRub,base.bankShare,0.01),onePpBankShareRub:customerFundsRub(parameters.marketRub,0.01,base.additionalPrevention)};}
  return {parameters,evidence,customerFundsRub,budgetTotals,scenarioResults,sensitivity,bankQuarterlyNetRub,bankReturn,experiment,viewModel};
})();
if(typeof module!=='undefined')module.exports=financialModel;
