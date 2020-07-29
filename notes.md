# Calibration
## HJA WS 1
Evaluated on mean daily Q from 2011-2015.
### Baseflow recession coefficient and initial storage
100 calibration runs altering initial storage (S) and recession coefficient (kb).
Parameter ranges were 0-500mm (s) and 0.001-0.1 (kb).

Best param combo was S=226.4 and kb=0.0014. NSE=0.431

Worst param combo was S=360 and kb=0.0997. NSE=0.339

Seems to give some evidence this model is robust for prediction in this watershed 
because even the worst parameter combination produced NSE > 0.30
for daily values (Brooks et al. 2016).

### Deep seepage coefficient
Added deep seepage coefficient (ks) to calibration. Values ranged from 0.0-0.1. 
Kept S and kb same as above.

Best param combo was S = 43.3, kb = 0.0406, ks=0.068. NSE=0.596

Worst param combo was S = 158.5, kb = 0.061, ks = 0.0005. NSE=0.363