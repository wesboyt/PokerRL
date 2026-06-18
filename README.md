# PokerRL
CFR_RNAD_PSRO multiway gto

Example run:
========== RUNTIME PROFILING REPORT (Update 11500) ==========
Worker: Mixed-opponent CFR Rollouts      | 500.7983        | 76       | 6589.4515      
Trainer: Waiting for Queue               | 139.9143        | 50       | 2798.2851      
Worker: Generate Prefixes                | 30.1885         | 77       | 392.0578       
Worker: Batched Representative Rollout   | 16.5189         | 76       | 217.3546       
Trainer: Forward & Backward R-NaD v6.4   | 5.1932          | 50       | 103.8631       
Trainer: Optimizer Step                  | 1.9117          | 50       | 38.2345        
Worker: Compute Pivot Priors             | 1.7804          | 77       | 23.1216        
========================================================================

Update 11500 | Loss: +0.6328 (Pol: +0.5707, Ent: -0.0130, Size: +0.0036, Gram: +0.0051) | RefFwd: +0.0123 | Reg-KL: 0.0020(it0) | EMA-KL: 0.0020 | Ref-KL: 0.1627 | Unif-KL: 0.8065 | PF-KL: 0.00143 | |Adv|: 4.464 | MinP: 0.0426 | Clip: 26.2% | rEps: 0.193 | SG: IDLE(0) street=-
Per-street diagnostics (incl. KL):
  Street 0: n=560, |Adv|=3.617bb, PolLoss=+1.5020, EMA-KL=0.00145, Ref-KL=0.07427
  Street 1: n=116, |Adv|=4.164bb, PolLoss=+0.2358, EMA-KL=0.00374, Ref-KL=0.37694
  Street 2: n=69, |Adv|=5.496bb, PolLoss=-0.0598, EMA-KL=0.00313, Ref-KL=0.33796
  Street 3: n=55, |Adv|=13.861bb, PolLoss=+0.6802, EMA-KL=0.00266, Ref-KL=0.39163
v24 verification diagnostics (H=entropy, supp=#actions>1%, SNR=advGap/SE):
  preflop n=563  H=0.126 supp=1.45 pFold=0.764 pAggr=0.180 | advGapSE=3.476bb advGap=5.072bb SNR=1.46 | SCV(Qself-Qref)=-0.540bb(n=35) | domAct=67.3%(n=563)
  flop    n=117  H=0.368 supp=1.93 pFold=0.148 pAggr=0.279 | advGapSE=3.563bb advGap=6.259bb SNR=1.76 | SCV(Qself-Qref)=-4.308bb(n=3) | domAct=62.4%(n=117)
  turn    n=70   H=0.365 supp=1.93 pFold=0.170 pAggr=0.273 | advGapSE=2.601bb advGap=8.366bb SNR=3.22 | SCV(Qself-Qref)=-0.854bb(n=3) | domAct=48.6%(n=70)
  river   n=56   H=0.408 supp=2.04 pFold=0.200 pAggr=0.345 | advGapSE=0.923bb advGap=22.780bb SNR=24.68 | SCV(Qself-Qref)=-5.167bb(n=3) | domAct=62.5%(n=56)
  advGapSE by p_aggr (rarity): <2%:3.257(n451)  2-10%:3.078(n96)  10-30%:3.001(n64)  >30%:3.338(n195)
  advGapSE by pot size:        <10bb:2.197(n602)  10-30:4.824(n124)  30-100:7.420(n64)  >100:13.231(n16)
  reach: hands=616 flop=45.3% turn=28.4% river=21.4%
Saving R-NaD v6.4 checkpoint at 11500...

*** [EXPLOITER PHASE #23] entering at update 11500: freezing current policy as target, training a best response for 800 updates. ***

=== REAL TRAINING-PIVOT Q-VECTORS (one per street) ===

--- PREFLOP (pos 5) hole=Qs6d board= ---
Action     | pi (behavior)  | Q (bb)     | valid 
--------------------------------------------------
fold       | 0.9983         | +0.000     | Y     
check      | 0.0000         | +0.000     | .     
call       | 0.0013         | +2.336     | Y     
raise      | 0.0004         | +3.874     | Y     
allin      | 0.0000         | +0.000     | .     

--- FLOP (pos 2) hole=JcKc board=5c7h3d ---
Action     | pi (behavior)  | Q (bb)     | valid 
--------------------------------------------------
fold       | 0.0000         | +0.000     | .     
check      | 0.7879         | -2.012     | Y     
call       | 0.0000         | +0.000     | .     
raise      | 0.2121         | -5.818     | Y     
allin      | 0.0000         | +0.000     | .     

--- TURN (pos 1) hole=2h8s board=6hAh5s6c ---
Action     | pi (behavior)  | Q (bb)     | valid 
--------------------------------------------------
fold       | 0.0000         | +0.000     | .     
check      | 0.5312         | -2.456     | Y     
call       | 0.0000         | +0.000     | .     
raise      | 0.4688         | -6.221     | Y     
allin      | 0.0000         | +0.000     | .     

--- RIVER (pos 2) hole=JcKc board=5c7h3dJh9h ---
Action     | pi (behavior)  | Q (bb)     | valid 
--------------------------------------------------
fold       | 0.0000         | +0.000     | .     
check      | 0.2451         | +6.631     | Y     
call       | 0.0000         | +0.000     | .     
raise      | 0.7549         | +3.070     | Y     
allin      | 0.0000         | +0.000     | .     
============================================================
