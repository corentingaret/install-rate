# Install Rate

## Description
Predicting the probability that a user will install an app after seeing an ad for it.

## Resources
- This project that outputs a model in joblib format.
- A notebook where I show my thinking process when exploring the dataset.

## How to Run the Project

1. **Add the Data**
   - Place your data in the `data/origin` folder.

2. **Build the Docker Image**
   - Make sure that you have Docker installed in your machine.
   - Run the following command:
   ```bash
   docker compose build
   ```

3. **Run the Project**
   - Review the parameters in `docker-compose.yml`.
   - Use the following commands to run different stages of the project:

   ### Prepare the Data
   - List the features you want to use in the `training_features.yaml` file.
   - Run:
   ```bash
   docker compose up install-rate-prepare
   ```

   ### Train the Model
   - Run:
   ```bash
   docker compose up install-rate-train
   ```

   ### Evaluate the Model
   - Run:
   ```bash
   docker compose up install-rate-evaluate
   ```

# Current model

My current model is a Logisitic Regression with a log loss criterion: this loss function evaluates the probability estimates, penalizing confident incorrect predictions more heavily.

# TODO:
### Model performance
- Understand current overfitting: cross-check temporal leakage (see notebook), iterate on balancing the dataset
- Find a way ot merge Android and IOS data together
- Leverage session data
- OneHot encoder to avoid hierarchical encoding on categorical features
- Find a way for the model to understand similarity between apps
- Try Gradient Boosting models

### Pipeline
- Remove automatically features too correlated to each other
- Implement unit tests
