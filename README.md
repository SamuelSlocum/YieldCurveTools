# YieldCurveTools
This repository houses code to estimate the Adrian, Crump, Moench (2012) term structure model and examine yield curves under different fed funds rate scenarios. The code allows the user to pull yield curve data from the Fed and estimate the ACM model based off of this data. The model breaks down yields at each maturity into a component that reflects expected short rates and a component that reflects a risk premium. The ACM model generally does not produce accurate rate path forecasts. We also allow the user to input their own assumptions about the future rate path. This allows the user to study how the shape of the curve is expected to change under their personal assumptions about fed policy.
# Sample Script
## Estimating Term Premiums
![BaselineSample](https://github.com/user-attachments/assets/90af3f39-ba99-438a-806b-4c1ca4eff5df)
## Scenario Analysis
![ScenarioSample](https://github.com/user-attachments/assets/535a3613-f608-4c95-abde-acae44c8a1cc)
# Description of Code
The main code for the repository is found in ACMTools.py. The code is based on an ACMModel object. This object contains methods to estimate the ACM model 
