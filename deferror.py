import numpy as np
import pandas as pd
from sklearn import metrics

def df_errors(y_test, y_pred_REGR, name):
    MAE = metrics.mean_absolute_error(y_test,y_pred_REGR) 
    MBE = np.mean(y_test- y_pred_REGR)
    MSE = metrics.mean_squared_error(y_test,y_pred_REGR)  
    RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred_REGR))
    cvRMSE = RMSE/np.mean(y_test)
    NMBE = MBE/np.mean(y_test)
    errors_REGR = {
        "Error Type": ["Mean Absolute Error (MAE)", "Mean Bias Error (MBE)", "Mean Squared Error (MSE)",
                       "Root Mean Squared Error (RMSE)", "Coefficient of Variation of RMSE (cvRMSE)",
                       "Normalized Mean Bias Error (NMBE)"],
        "Values": [MAE.round(3), MBE.round(3), MSE.round(3), RMSE.round(3), cvRMSE.round(3), NMBE.round(3)]
    }
    return pd.DataFrame(errors_REGR)

def df_errors_SVR(y_test, y_test_SVR, y_pred_REGR, name):
    MAE = metrics.mean_absolute_error(y_test_SVR,y_pred_REGR) 
    MBE = np.mean(y_test- y_pred_REGR)
    MSE = metrics.mean_squared_error(y_test_SVR,y_pred_REGR)  
    RMSE = np.sqrt(metrics.mean_squared_error(y_test_SVR,y_pred_REGR))
    cvRMSE = RMSE/np.mean(y_test)
    NMBE = MBE/np.mean(y_test)
    errors_REGR = {
        "Error Type": ["Mean Absolute Error (MAE)", "Mean Bias Error (MBE)", "Mean Squared Error (MSE)",
                       "Root Mean Squared Error (RMSE)", "Coefficient of Variation of RMSE (cvRMSE)",
                       "Normalized Mean Bias Error (NMBE)"],
        "Values": [MAE.round(3), MBE.round(3), MSE.round(3), RMSE.round(3), cvRMSE.round(3), NMBE.round(3)]
    }
    return pd.DataFrame(errors_REGR)