# import the classes for accessing DSS objects from the recipe
import dataiku

# Import the helpers for custom recipes
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
model_name = get_input_names_for_role('Model')[0]
model_name_split = model_name.split('.')

# For outputs, the process is the same:
output_dataset_name = get_output_names_for_role('Dataset')[0]

#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
from dataikuapi.dss.future import DSSFuture

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_pred_type(models, active_model_id):
    for model in models.list_models():
        if model['id'] == active_model_id:
            return model['predictionType']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def compute_shapley_feature_importance_rev(self):
    """
    Launches computation of Shapley feature importance for this trained model

    :returns: A future for the computation task
    :rtype: :class:`dataikuapi.dss.future.DSSFuture`
    """
    future_response = self.saved_model.client._perform_json(
        "POST", "/projects/%s/savedmodels/%s/versions/%s/shapley-feature-importance" %
                (self.saved_model.project_key, self.saved_model.sm_id, self.saved_model_version),
    )
    future = DSSFuture(self.saved_model.client, future_response.get("jobId", None), future_response)
    return future

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

client = dataiku.api_client()
p = client.get_project(model_name_split[0])
sm = p.get_saved_model(model_name_split[1])
smd = sm.get_version_details(sm.get_active_version()['id'])

models = dataiku.Model(model_name_split[1])
pred_type = get_pred_type(models, model_name_split[1])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if (pred_type == 'BINARY_CLASSIFICATION') | (pred_type == 'MULTICLASS'):
    result = compute_shapley_feature_importance_rev(smd)
    df = pd.DataFrame.from_dict(result.wait_for_result()['absoluteImportance'], orient = 'index').reset_index()
    df.columns = ['feature_name', 'importance']
    df.sort_values('importance', inplace = True, ascending = False)
    df = df.head(20).reset_index(drop=True)
    df['importance'] = df['importance'] / df['importance'].sum()
    # Write recipe outputs
    shapley_global_importance = dataiku.Dataset(output_dataset_name)
    shapley_global_importance.write_with_schema(df)
else:
    raise Exception("Error: Model does not have Prediction Type of Two-Class Classification or Multiclass Classification")
