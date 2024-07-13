# Yield Curve Tools
This repository houses code to pull yield curve data and estimate the Adrian, Crump, Moench (2012) term structure model and examine yield curves under different fed funds rate scenarios. \
\
The code allows the user to pull yield curve data from the Fed and estimate the ACM model based off of this data. The model breaks down yields at each maturity into a component that reflects expected short rates and a component that reflects a risk premium. The ACM model generally does not produce accurate rate path forecasts. We also allow the user to input their own assumptions about the future rate path. This allows the user to study how the shape of the curve is expected to change under their personal assumptions about fed policy.
# Sample.py Script
The Sample.py script shows the main features of the ACMTools.py module. This script first estimates the ACM model, plots the term premiums, and also shows the scenario analysis capabilities of the model. Below we discuss the output of each of these sections.
## Estimating Term Premiums
The plot on the left shows the fitted yield curve, the Fed-provided yields (to ensure we are getting a good fit), as well as the implied average short rate over each time horizon. The plot on the right shows us the history of the 10-year fitted yield, term premium, and expected average short rate.\
\
![BaselineSample](https://github.com/user-attachments/assets/90af3f39-ba99-438a-806b-4c1ca4eff5df)

## Scenario Analysis
The plots below show the yield curve under user-provided assumptions about the path of interest rates. In this case, the user input short rate assumptions at 6,12, and 24 months. These rate assumptions were 4.5%, 3.5%, and 2% respectively. We can then make inference about how the shape of the yield curve changes over time. Confidence intervals are also provided.\
\
![ScenarioSample](https://github.com/user-attachments/assets/535a3613-f608-4c95-abde-acae44c8a1cc)

# Description of Code
The main code for the repository is found in ACMTools.py. The code is based on an ACMModel object. This object contains methods to estimate the ACM model as well as do basic functions with the model after it is estimated. The methodology for estimating the model follows the ACM (2012) paper. The procedure for producing scenario estimates based on user input uses Kalman filtering to walk forward the yield curve factors conditional on the user's inputs. The ACM model depends on an underlying dynamic factor structure. The short rate is an observation of a combination of these underlying factors, and we can use Kalman filtering to make inference about the expected factors conditional on the short rate being at the user's input position. Confidence intervals are produced with a simulation approach using the estimator covariance matrices we get from the Kalman filter.
